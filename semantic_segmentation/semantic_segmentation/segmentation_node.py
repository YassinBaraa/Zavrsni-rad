#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from PIL import Image as PILImage
from sensor_msgs.msg import Image  # Import ROS 2 Image message
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, AutoImageProcessor
import numpy as np


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        self.publisher_ = self.create_publisher(Image, '/instance_segmentation', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model.eval()

        # Define preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((1024, 1024)),  # Resize image to model's input dimensions
            transforms.Lambda(lambda img: img.convert('RGB')),  # Convert image to RGB format
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])


        self.bridge = CvBridge()

    def listener_callback(self, msg):
        # Load and preprocess input data
        #image = Image.open('/root/ros2_ws/src/semantic_segmentation/pic2.jpeg')
        #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        #image = Image.open(requests.get(url, stream=True).raw)
        
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image = PILImage.fromarray(cv_image)

        input_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Get segmentation mask
        predicted_masks = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

        # Define color map for segmentation classes
        color_map = {
            0: [0, 0, 0],  # Background (black)
            1: [255, 0, 0],  # Person (red)
            2: [0, 255, 0],
            3: [0, 0, 255],
            4: [255, 255, 255],
            5: [255, 0, 255],
            6: [255, 255, 0],
            7: [0, 255, 255],
        }

    #########################black box###################################
    
        # Resize the segmentation output to match the dimensions of the original image
        predicted_masks_resized = np.array(PILImage.fromarray(predicted_masks.astype(np.uint8)).resize((cv_image.shape[1], cv_image.shape[0]), PILImage.NEAREST))

        # Generate colored segmentation output image
        output_image = np.zeros((cv_image.shape[0], cv_image.shape[1], 3), dtype=np.uint8)
        for class_idx, color in color_map.items():
            output_image[predicted_masks_resized == class_idx] = color
        
        
        # Blend the segmentation output with the input image
            #output_image = output_image * 0.3 + cv_image * 0.7
        # Convert numpy array image to ROS Image message
        output_image_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="rgb8")
        # Publish the segmented image
        self.publisher_.publish(output_image_msg)


def main(args=None):
    rclpy.init(args=args)

    segmentation_node = SegmentationNode()

    rclpy.spin(segmentation_node)

    segmentation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
    
    
#ros2 run usb_cam usb_cam_node_exe
#rviz2
#root@baraa-G3-3590:~/ros2_ws/src# ros2 run usb_cam usb_cam_node_exe
