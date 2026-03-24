# My_Duplo 사용법 

find /home/(ROS_ID) -name "best.pt" 를 터미널에 입력 후 맞는 주소 찾기 <Ex)ROS_ID=han >

찾은 주소를 duplo_perception_node.py 파일의 self.model = YOLO( ' 찾은 주소 ') 에 넣어주기.

### 기본 도구 설치
sudo apt update && sudo apt install -y python3-pip python3-colcon-common-extensions
### ROS2 의존성 관리 도구
sudo apt install -y python3-rosdep
sudo rosdep init
rosdep update
### RealSense SDK 설치 (바이너리 방식이 가장 빠름)
sudo apt install ros-humble-realsense2-camera ros-humble-realsense2-description
### 1. ultralytics 설치 (YOLOv8)
pip install ultralytics

### 2. ★중요★ NumPy 버전 다운그레이드 (ROS2 cv_bridge 호환성용)
pip install "numpy<2"

### 3. OpenCV 및 기타 도구 (보통 자동으로 깔리지만 확인차)
pip install opencv-python cvbridge-python

## 필요한 드라이버 및 환경 세팅

cd ~/My_Duplo/robot_ws

### 1. 의존성 자동 설치 (혹시 빠진 게 있을까봐)
rosdep install -i --from-path src --rosdistro humble -y

### 2. 빌드 실행
colcon build --packages-select my_perception

### 3. 환경 등록
source install/setup.bash


# 실행 코드
  ros2 launch my_perception duplo_launch.py
