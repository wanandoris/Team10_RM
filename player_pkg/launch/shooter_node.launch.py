from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='player_pkg',
            executable='shooter_node',
            name='shooter_node',
            output='screen',
            parameters=[{
                'bullet_speed': 1.5,
                'gravity': 9.8
            }]
        )
    ])