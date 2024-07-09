import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from numpy import asarray
from utils import color_map
import time

class SegmentationNode(Node):
    
    def __init__(self):
        super().__init__('segmentation_node')
        self.publisher_ = self.create_publisher(Image, '/instance_segmentation', 1)
        self.subscription = self.create_subscription(Image, '/camera1/image_raw', self.listener_callback, 1)

        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.device = torch.device("cuda")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", device_map = 'cuda')
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((1024, 1024)),  
            transforms.Lambda(lambda img: img.convert('RGB')),  
            transforms.ToTensor(),  
            #transforms.cuda.ToTensor(),
        ])
       
        self.bridge = CvBridge()
        
        self.cv_image = None
        self.predicted_masks = None
        print("init")
      
      
      
    def listener_callback(self, msg):
        start_time = time.time()  # Start time for measuring prediction time
        
        
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image = PILImage.fromarray(self.cv_image)
        input_tensor = self.preprocess(image).unsqueeze(0)
        input_tensor =input_tensor.cuda()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        self.predicted_masks = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        
        
        end_time = time.time()  # End time for measuring prediction time
        print("Prediction time:", end_time - start_time, "seconds")
        
        
        
    def run_segmentation(self):
        while rclpy.ok():
            if self.cv_image is not None and self.predicted_masks is not None:
                start_time = time.time()  # Start time for measuring plotting time
                
                
                predicted_masks_resized = np.array(PILImage.fromarray(self.predicted_masks.astype(np.uint8)).resize((self.cv_image.shape[1], self.cv_image.shape[0]), PILImage.NEAREST))

                output_image = np.zeros((self.cv_image.shape[0], self.cv_image.shape[1], 3), dtype=np.uint8)
                for class_idx, color in color_map.items():
                    output_image[predicted_masks_resized == class_idx] = color

                im1 = PILImage.fromarray(self.cv_image)
                im2 = PILImage.fromarray(output_image)
                im2 = im2.resize(im1.size)

                mask = PILImage.new("L", im1.size, 128)
                im = PILImage.composite(im1, im2, mask)

                numpy_arr_out = asarray(im)
                output_image_msg = self.bridge.cv2_to_imgmsg(numpy_arr_out, encoding="rgb8")
                self.publisher_.publish(output_image_msg)
                
                
                end_time = time.time()  # End time for measuring plotting time
                print("Plotting time:", end_time - start_time, "seconds")
        
        

def main(args=None):
    print("Initializing segmentation node")
    rclpy.init(args=args)
    segmentation_node = SegmentationNode()
    
    # Create a task executor
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    
    # Add the segmentation node to the executor
    executor.add_node(segmentation_node)

    # Create a future to run the segmentation method asynchronously
    future = executor.create_task(segmentation_node.run_segmentation)

    # Spin until shutdown
    executor.spin()

    # Shutdown
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# ros2 launch usb_cam camera.launch.py
# rviz2
# root@baraa-G3-3590:~/ros2_ws/src/semantic_segmentation/semantic_segmentation# python3 segmentation_node.py 


#~/ros2_ws# ros2 launch semantic_segmentation segmentation_node_launch.py camera_topic:=/camera2/image_raw model_name:=custom_model
