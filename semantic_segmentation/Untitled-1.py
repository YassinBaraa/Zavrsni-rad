#!/usr/bin/env python3
# This line is a shebang that tells the system to execute this script using the python3 interpreter.

import rclpy
from rclpy.node import Node
from PIL import Image as PILImage, ImageDraw, ImageFilter 
from sensor_msgs.msg import Image  # Import ROS 2 Image message
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, AutoImageProcessor
import numpy as np
from numpy import asarray

# Import necessary libraries/modules.

class SegmentationNode(Node):
    # Define a class for our segmentation node which inherits from the Node class provided by rclpy.

    def __init__(self):
        # Constructor method for the SegmentationNode class.

        super().__init__('segmentation_node')
        # Call the constructor of the parent class.

        self.publisher_ = self.create_publisher(Image, '/instance_segmentation', 10)
        # Create a publisher to publish segmented images to the topic '/instance_segmentation'.

        self.subscription = self.create_subscription(Image, '/image_raw', self.listener_callback, 10)
        # Create a subscription to listen for incoming raw images from the topic '/image_raw'.
        self.subscription  # prevent unused variable warning

        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model.eval()
        # Load pre-trained segmentation model and image processor.

        # Define preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((1024, 1024)),  # Resize image to model's input dimensions
            transforms.Lambda(lambda img: img.convert('RGB')),  # Convert image to RGB format
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])
        # Define a series of preprocessing transformations to be applied to input images.

        self.bridge = CvBridge()
        # Initialize CvBridge object for converting between ROS Image messages and OpenCV images.

    def listener_callback(self, msg):
        # Callback function to process incoming raw images.

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Convert ROS Image message to OpenCV image format.

        image = PILImage.fromarray(cv_image)
        # Convert OpenCV image to PIL image format.

        input_tensor = self.preprocess(image).unsqueeze(0)
        # Preprocess the input image for model inference.

        with torch.no_grad():
            outputs = self.model(input_tensor)
        # Perform inference using the pre-trained segmentation model.

        predicted_masks = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        # Obtain the segmentation masks from the model's output.

        color_map = {
            # Define a color map for different segmentation classes.
            # Each class is assigned a specific RGB color.
            # 0 is reserved for the background.
            0: [0, 0, 0],       # Background (Black)
            1: [255, 0, 0],     # Person (Red)
            2: [0, 255, 0],     # Object 1 (Green)
            3: [0, 0, 255],     # Object 2 (Blue)
            4: [255, 255, 255], # Object 3 (White)
            5: [255, 0, 255],   # Object 4 (Magenta)
            6: [255, 255, 0],   # Object 5 (Yellow)
            7: [0, 255, 255],   # Object 6 (Cyan)
            8: [128, 0, 0],     # Object 7 (Maroon)
            9: [0, 128, 0],     # Object 8 (Green)
            10: [0, 0, 128],    # Object 9 (Navy)
            11: [128, 128, 128],# Object 10 (Gray)
            12: [128, 0, 128],  # Object 11 (Purple)
            13: [0, 128, 128],  # Object 12 (Teal)
            14: [192, 192, 192] # Object 13 (Silver)
        }

        predicted_masks_resized = np.array(PILImage.fromarray(predicted_masks.astype(np.uint8)).resize((cv_image.shape[1], cv_image.shape[0]), PILImage.NEAREST))
        # Resize the predicted segmentation mask to match the dimensions of the original image.

        output_image = np.zeros((cv_image.shape[0], cv_image.shape[1], 3), dtype=np.uint8)
        # Create an empty array to store the colored segmentation output.

        for class_idx, color in color_map.items():
            output_image[predicted_masks_resized == class_idx] = color
        # Apply the color map to the segmentation mask.

        im1 = PILImage.fromarray(cv_image)
        im2 = PILImage.fromarray(output_image)
        im2 = im2.resize(im1.size)

        mask = PILImage.new("L", im1.size, 128)
        im = PILImage.composite(im1, im2, mask)
        # Blend the original image and the segmentation output.

        numpy_arr_out = asarray(im)
        # Convert the blended image to a numpy array.

        output_image_msg = self.bridge.cv2_to_imgmsg(numpy_arr_out, encoding="rgb8")
        # Convert the numpy array image to a ROS Image message.

        self.publisher_.publish(output_image_msg)
        # Publish the segmented image.

def main(args=None):
    rclpy.init(args=args)
    # Initialize the ROS 2 node.

    segmentation_node = SegmentationNode()
    # Create an instance of the SegmentationNode class.

    rclpy.spin(segmentation_node)
    # Spin the node to keep it running until shutdown.

    segmentation_node.destroy_node()
    rclpy.shutdown()
    # Clean up and shutdown the node when finished.

if __name__ == '__main__':
    main()
    # Entry point of the script. Calls the main function if the script is executed directly.
