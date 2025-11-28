from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='player_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen'
        ),
        Node(
            package='player_pkg',
            executable='shooter_node', 
            name='shooter_node',
            output='screen'
        )
    ])