import time
import logging
import warnings
import serial

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import rbpodo as rb


# ==============================
# 설정값
# ==============================
ROBOT_IP = "10.0.2.7"
MODEL_PATH = "runs/segment/duplo_project/train_v1/weights/best.pt"

GRIPPER_PORT = "/dev/ttyACM0"
GRIPPER_BAUD = 115200

HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0], dtype=float)

CAM_TO_TCP_OFFSET_X_MM = -51.0
CAM_TO_TCP_OFFSET_Y_MM = 32.0

# PICK / INSERT 공통 Z 기준
Z_OFFSET_MM = -85.0
Z_APPROACH_MARGIN_MM = 20.0

J_VEL = 255
J_ACC = 255
L_VEL = 500
L_ACC = 800

MOVE_START_TIMEOUT_SEC = 1.0

NUM_SAMPLES = 3
SAMPLE_DELAY_SEC = 0.10


# ==============================
# 로그 최소화
# ==============================
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# ==============================
# 안전 복귀 함수
# ==============================
def safe_return_home(robot, reason="UNKNOWN"):
    print(f"⚠️ FAIL: {reason} → GO HOME")
    try:
        robot.go_home()
    except Exception as e:
        print(f"❌ HOME FAILED: {e}")


# ==============================
# Gripper
# ==============================
class Gripper:
    def __init__(self, port=GRIPPER_PORT, baud=GRIPPER_BAUD):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)
        print("✅ Gripper Connected")

    def open(self):
        self.ser.write(b"open\n")
        print("📌 Sent: open")
        time.sleep(1.0)
        return True

    def grip(self, timeout=5.0):
        self.ser.write(b"grip\n")
        print("📌 Sent: grip")

        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()

                if "[RESULT] GRASP_OK" in line or "GRASP_OK" in line:
                    print("✅ GRASP_OK")
                    return True

                if "[RESULT] GRASP_FAIL" in line or "GRASP_FAIL" in line:
                    print("❌ GRASP_FAIL")
                    return False

            time.sleep(0.01)

        print("❌ GRIP TIMEOUT")
        return False

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("✅ Gripper Closed")


# ==============================
# Vision
# ==============================
class Vision:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(self.config)

        time.sleep(2.0)

        self.align = rs.align(rs.stream.color)
        self.intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        print("✅ Vision Ready")

    def stop(self):
        self.pipeline.stop()
        print("✅ RealSense Stopped")

    def _extract_rightmost_pose_from_result(self, frame, depth_frame, results):
        if results.masks is None or len(results.masks.xy) == 0:
            return None

        best = None
        max_u = -1

        for mask in results.masks.xy:
            pts = np.int32([mask])

            M = cv2.moments(pts)
            if M["m00"] == 0:
                continue

            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            z = depth_frame.get_distance(u, v)
            if z == 0:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], z)

            rect = cv2.minAreaRect(pts)
            yaw = rect[2] % 90.0
            if yaw > 45:
                yaw -= 90

            if u > max_u:
                max_u = u
                best = {
                    "x": float(X),
                    "y": float(Y),
                    "z": float(Z),
                    "yaw": float(yaw),
                    "u": int(u),
                    "v": int(v),
                }

        return best

    def detect_rightmost_once(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        frame = np.asanyarray(color_frame.get_data())
        results = self.model(frame, verbose=False)[0]

        return self._extract_rightmost_pose_from_result(frame, depth_frame, results)

    def get_stable_rightmost_pose(self):
        samples = []

        for _ in range(NUM_SAMPLES):
            pose = self.detect_rightmost_once()
            if pose is not None:
                samples.append(pose)
            time.sleep(SAMPLE_DELAY_SEC)

        if len(samples) < 2:
            return None

        return {
            "x": float(np.median([p["x"] for p in samples])),
            "y": float(np.median([p["y"] for p in samples])),
            "z": float(np.median([p["z"] for p in samples])),
            "yaw": float(np.median([p["yaw"] for p in samples])),
        }

    def count_objects(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return 0

        frame = np.asanyarray(color_frame.get_data())
        results = self.model(frame, verbose=False)[0]

        if results.masks is None:
            return 0

        return len(results.masks.xy)


# ==============================
# Robot
# ==============================
class RobotController:
    def __init__(self, ip):
        self.robot = rb.Cobot(ip)
        self.rc = rb.ResponseCollector()
        self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
        print("✅ Robot Ready")

    def wait_move(self, name):
        started = self.robot.wait_for_move_started(
            self.rc, MOVE_START_TIMEOUT_SEC
        ).is_success()

        if not started:
            print(f"⚠️ {name} START SKIPPED (already in position?)")
            return True

        self.robot.wait_for_move_finished(self.rc)
        print(f"✅ {name} DONE")
        return True

    def go_home(self):
        print("➡️ GO HOME")
        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
        return self.wait_move("HOME")

    def align_xy(self, x, y):
        dx = -(x * 1000.0) + CAM_TO_TCP_OFFSET_Y_MM
        dy =  (y * 1000.0) + CAM_TO_TCP_OFFSET_X_MM

        print(f"➡️ ALIGN XY | dx={dx:.1f}, dy={dy:.1f}")

        self.robot.move_l_rel(
            self.rc,
            np.array([dy, dx, 0, 0, 0, 0], dtype=float),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        return self.wait_move("ALIGN_XY")

    def align_yaw(self, yaw):
        print(f"➡️ ALIGN YAW | {yaw:.2f}")

        self.robot.move_l_rel(
            self.rc,
            np.array([0, 0, 0, 0, 0, yaw], dtype=float),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        return self.wait_move("ALIGN_YAW")

    def move_z(self, dz, name):
        print(f"↕️ {name} | {dz:.1f}")

        self.robot.move_l_rel(
            self.rc,
            np.array([0, 0, dz, 0, 0, 0], dtype=float),
            L_VEL,
            L_ACC,
            rb.ReferenceFrame.Tool
        )
        return self.wait_move(name)


# ==============================
# 동작 함수
# ==============================
def pick_rightmost(vision, robot, gripper):
    print("\n--- PICK RIGHTMOST ---")

    pose = vision.get_stable_rightmost_pose()
    if pose is None:
        print("❌ PICK XY detect fail")
        return False

    if not robot.align_xy(pose["x"], pose["y"]):
        print("❌ PICK XY move fail")
        return False

    pose = vision.get_stable_rightmost_pose()
    if pose is None:
        print("❌ PICK YAW detect fail")
        return False

    if not robot.align_yaw(pose["yaw"]):
        print("❌ PICK YAW move fail")
        return False

    pose = vision.get_stable_rightmost_pose()
    if pose is None:
        print("❌ PICK Z detect fail")
        return False

    z_move = pose["z"] * 1000.0 + Z_OFFSET_MM

    if abs(z_move) < 1.0:
        print("❌ PICK Z invalid")
        return False

    if not robot.move_z(z_move - Z_APPROACH_MARGIN_MM, "PICK_Z1"):
        print("❌ PICK Z1 fail")
        return False

    if not robot.move_z(Z_APPROACH_MARGIN_MM, "PICK_Z2"):
        print("❌ PICK Z2 fail")
        return False

    if not gripper.grip():
        print("❌ PICK Grip fail")
        return False

    if not robot.move_z(-50.0, "PICK_UP"):
        print("❌ PICK UP fail")
        return False

    return True


def insert_to_next_rightmost(vision, robot, gripper):
    print("\n--- INSERT TO NEXT RIGHTMOST ---")

    pose = vision.get_stable_rightmost_pose()
    if pose is None:
        print("❌ PLACE XY detect fail")
        return False

    if not robot.align_xy(pose["x"], pose["y"]):
        print("❌ PLACE XY move fail")
        return False

    pose = vision.get_stable_rightmost_pose()
    if pose is None:
        print("❌ PLACE YAW detect fail")
        return False

    if not robot.align_yaw(pose["yaw"]):
        print("❌ PLACE YAW move fail")
        return False

    pose = vision.get_stable_rightmost_pose()
    if pose is None:
        print("❌ PLACE Z detect fail")
        return False

    z_move = pose["z"] * 1000.0 + Z_OFFSET_MM

    if abs(z_move) < 1.0:
        print("❌ PLACE Z invalid")
        return False

    # 들고 있는 블록(또는 결합체)을 아래 블록과 결합시키기 위해
    # PICK과 동일한 방식으로 2단계 접근
    if not robot.move_z(z_move - Z_APPROACH_MARGIN_MM, "PLACE_Z1"):
        print("❌ PLACE Z1 fail")
        return False

    if not robot.move_z(Z_APPROACH_MARGIN_MM, "PLACE_Z2"):
        print("❌ PLACE Z2 fail")
        return False

    # 결합 완료 후 release
    if not gripper.open():
        print("❌ PLACE release fail")
        return False

    if not robot.move_z(-50.0, "PLACE_UP"):
        print("❌ PLACE UP fail")
        return False

    return True


# ==============================
# Main
# ==============================
def main():
    vision = None
    gripper = None
    robot = None

    try:
        vision = Vision()
        robot = RobotController(ROBOT_IP)
        gripper = Gripper()

        def fail(reason):
            print(f"❌ {reason}")
            safe_return_home(robot, reason)
            return True

        print("\n=== START ===\n")

        if not robot.go_home():
            return

        if not gripper.open():
            if fail("Initial gripper open failed"):
                return

        num_blocks = vision.count_objects()
        print(f"📦 Detected blocks: {num_blocks}")

        if num_blocks < 2:
            print("❌ Need at least 2 blocks")
            return

        num_loops = num_blocks - 1
        print(f"🔁 Total loops: {num_loops}")

        for i in range(num_loops):
            print(f"\n==============================")
            print(f"=== LOOP {i + 1} / {num_loops} ===")
            print(f"==============================")

            # 1) 가장 오른쪽 블록(또는 결합체) PICK
            if not pick_rightmost(vision, robot, gripper):
                if fail(f"Loop {i+1}: pick_rightmost failed"):
                    return

            # 2) HOME 복귀
            if not robot.go_home():
                if fail(f"Loop {i+1}: home after pick failed"):
                    return

            # 3) 남아있는 가장 오른쪽 블록 위로 가서 아래에 결합
            if not insert_to_next_rightmost(vision, robot, gripper):
                if fail(f"Loop {i+1}: insert_to_next_rightmost failed"):
                    return

            # 4) HOME 복귀
            if not robot.go_home():
                if fail(f"Loop {i+1}: home after place failed"):
                    return

        print("\n🎉 ALL DONE")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        try:
            safe_return_home(robot, "Exception")
        except Exception:
            pass

    finally:
        if vision:
            vision.stop()
        if gripper:
            gripper.close()


if __name__ == "__main__":
    main()