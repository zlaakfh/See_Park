from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 네 카메라 4대 노드 (그대로)
        Node(
            package='valet_parking',
            executable='camera_publish',
            output='screen',
        ),

        Node(
            package='valet_parking',
            executable='joystick_node',
            output='screen',
        ),

        Node(
            package='valet_parking',
            executable='cmd_mux_node',
            parameters=[{
                'default_mode': 'policy',
            }],
            output='screen',
        ),

        Node(
            package='valet_parking',
            executable='cmd_bridge_mux_node',
            parameters=[{
                'default_mode': 'joy',
            }],
            output='screen',
        ),
    ])
