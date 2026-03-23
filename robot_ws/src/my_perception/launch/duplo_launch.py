from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. RealSense 카메라 런칭 포함
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ]),
        launch_arguments={
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true'
        }.items()
    )

    # 2. 방금 만든 듀플로 인지 노드 실행
    perception_node = Node(
        package='my_perception', # Ella님의 패키지 이름
        executable='duplo_perception_node',
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        perception_node
    ])
