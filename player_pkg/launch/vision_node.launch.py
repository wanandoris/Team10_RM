from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='player_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[{
                'conf_threshold': 0.5,
                'nms_threshold': 0.4,
                'model_path': 'src/player_pkg/model/best.onnx'
            }]
        )
    ])