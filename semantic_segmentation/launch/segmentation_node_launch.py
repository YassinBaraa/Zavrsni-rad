# segmentation_node_launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    # Define parameters
    camera_topic = '/camera1/image_raw'  # Default camera topic
    model_name = 'nvidia/segformer-b0-finetuned-ade-512-512'  # Default model name

    # Override parameters if provided in launch file
    camera_topic_param = DeclareLaunchArgument('camera_topic', default_value=camera_topic)
    model_name_param = DeclareLaunchArgument('model_name', default_value=model_name)

    # Launch the SegmentationNode
    segmentation_node = Node(
        package='semantic_segmentation',
        executable='segmentation_node',
        name='segmentation_node',
        namespace='',
        output='screen',
        parameters=[
            {'camera_topic': camera_topic},
            {'model_name': model_name}
        ]
    )
    
    #TODO:
    # Launch rviz2
    #Launch usb_cam

    return LaunchDescription([
        camera_topic_param,
        model_name_param,
        segmentation_node
    ])


#ros2 launch your_package_name segmentation_node_launch.py camera_topic:=/camera2/image_raw model_name:=custom_model