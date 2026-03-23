import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import math

class DuploPerception(Node):
    def __init__(self):
        super().__init__('duplo_perception')
        # 1. 모델 로드 (경로 확인 필수!)
        self.model = YOLO('/home/han/Duplo-3.0-23/runs/segment/duplo_project/train_v1/weights/best.pt')
        self.bridge = CvBridge()
        
        # 2. 퍼블리셔 (로봇 제어용 좌표 전송)
        self.pose_pub = self.create_publisher(Pose, '/target_pose', 10)
        
        # 3. 구독 설정
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        
        self.depth_image = None
        
        # 카메라 파라미터 (D435 기본값 - 필요시 보정 가능)
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def get_yaw_angle(self, mask_points):
        # 마스크로부터 최소 면적 사각형을 구해 회전 각도 계산
        rect = cv2.minAreaRect(mask_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[2]
        
        # 각도 정규화 (로봇 집게 방향에 맞게 조정)
        width = rect[1][0]
        height = rect[1][1]
        if width < height:
            angle += 90
        return angle

    def image_callback(self, msg):
        if self.depth_image is None:
            return
            
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame, stream=True, conf=0.5)
        
        for r in results:
            if r.masks is not None:
                for i, mask in enumerate(r.masks.xy):
                    # 클래스 이름과 색상 가져오기
                    cls_id = int(r.boxes.cls[i])
                    cls_name = r.names[cls_id]

                    # 마스크 중심점(u, v) 계산
                    points = np.int32([mask])
                    M = cv2.moments(points)
                    if M['m00'] == 0: continue
                    u, v = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    
                    # 깊이 데이터로 거리(z) 확인
                    z = self.depth_image[v, u] / 1000.0 # Meter 단위
                    if z == 0: continue

                    # 2D 픽셀 -> 3D 실제 좌표 변환
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    
                    # 블록 회전 각도(Yaw) 계산
                    yaw = self.get_yaw_angle(points)

                    # ROS2 메시지 발행
                    target_pose = Pose()
                    target_pose.position.x = x
                    target_pose.position.y = y
                    target_pose.position.z = z
                    target_pose.orientation.z = math.radians(yaw)
                    self.pose_pub.publish(target_pose)

                    # --- 시각화 부분 ---
                    # 1. 외곽선 및 중심점 표시
                    cv2.drawContours(frame, [points], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)

                    # 2. 텍스트 정보 표시 (클래스명, 각도, 좌표)
                    info_text = f"{cls_name} | Yaw: {yaw:.1f}deg"
                    pos_text = f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
                    
                    cv2.putText(frame, info_text, (u - 60, v - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, pos_text, (u - 60, v - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Duplo 3D Perception", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DuploPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()