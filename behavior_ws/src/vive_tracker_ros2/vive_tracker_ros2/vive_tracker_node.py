#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import math
from typing import Optional, Dict, List

import numpy as np
import openvr

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

from vive_tracker_interfaces.srv import ViveCalibration

from vive_tracker_ros2.utils import (
    calculate_calibration_matrix,
    matrix_to_pose,
    matrix_to_twist,
    pose_to_matrix,
)

import ament_index_python.packages


# -------------------------
# Basic rotations
# -------------------------
def rot_x(th_rad: float) -> np.ndarray:
    c = np.cos(th_rad)
    s = np.sin(th_rad)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]], dtype=np.float64)


def rot_y(th_rad: float) -> np.ndarray:
    c = np.cos(th_rad)
    s = np.sin(th_rad)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def rot_z(th_rad: float) -> np.ndarray:
    c = np.cos(th_rad)
    s = np.sin(th_rad)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def openvr_pose_to_np44(pose) -> np.ndarray:
    m = pose.mDeviceToAbsoluteTracking
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = m[0][0]; M[0, 1] = m[0][1]; M[0, 2] = m[0][2]
    M[1, 0] = m[1][0]; M[1, 1] = m[1][1]; M[1, 2] = m[1][2]
    M[2, 0] = m[2][0]; M[2, 1] = m[2][1]; M[2, 2] = m[2][2]
    M[0, 3] = m[0][3]
    M[1, 3] = m[1][3]
    M[2, 3] = m[2][3]
    return M


# -------------------------
# Rotvec (continuous) utilities
# -------------------------
def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Return quaternion as [w, x, y, z] from 3x3 rotation matrix.
    """
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_to_rotvec_cont(q_wxyz: np.ndarray, prev_w: Optional[np.ndarray]) -> np.ndarray:
    """
    Quaternion [w,x,y,z] -> rotation vector w (rad), with continuity.
    Uses two equivalent candidates: axis*angle and axis*(angle-2π) (same rotation).
    Chooses closer one to prev_w if prev provided.
    """
    w, x, y, z = [float(v) for v in q_wxyz]

    # angle in [0, π]
    vnorm = math.sqrt(x*x + y*y + z*z)
    if vnorm < 1e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * math.atan2(vnorm, w)
    # map to [0, π]
    if angle > math.pi:
        angle = 2.0 * math.pi - angle
        x, y, z = -x, -y, -z
        vnorm = math.sqrt(x*x + y*y + z*z)
        if vnorm < 1e-12:
            return np.zeros(3, dtype=np.float64)

    axis = np.array([x, y, z], dtype=np.float64) / vnorm

    cand1 = axis * angle
    cand2 = axis * (angle - 2.0 * math.pi)  # same rotation (2π 주기)

    if prev_w is None:
        return cand1

    if float(np.linalg.norm(cand2 - prev_w)) < float(np.linalg.norm(cand1 - prev_w)):
        return cand2
    return cand1


# -------------------------
# DEFAULT_T_SA fallback
# -------------------------
DEFAULT_T_SA = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class ViveTracker(Node):
    def __init__(self):
        super().__init__("vive_tracker")

        self.vr_system = None
        self.trackers: Dict[str, dict] = {}

        self.prev_time = self.get_clock().now()

        self._init_ros()
        self._init_vr()
        self._init_json_calib()
        self._init_yaml_calib()

    # ------------------------------------------------------------------
    # ROS init
    # ------------------------------------------------------------------
    def _init_ros(self):
        self.raw_pose_pub = self.create_publisher(Odometry, "vive_tracker_ros/raw_pose", 10)
        self.calibrated_pose_pub_odom = self.create_publisher(Odometry, "vive_tracker_ros/calibrated_pose", 10)

        # /raw_pose PoseStamped
        self.raw_pose_pub_pose = self.create_publisher(PoseStamped, "/raw_pose", 10)

        # /calibrated_pose: [x y z wx wy wz] (m, rad)
        self.calibrated_pose_pub = self.create_publisher(Float64MultiArray, "/calibrated_pose", 10)

        self.calibrate_srv = self.create_service(ViveCalibration, "vive_tracker_ros/calibrate", self.cb_calibrate)

        self.declare_parameter("publish_tf", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("child_frame", "vive_tracker")

        # --- T_SA 적용: ver5(vr_calibration) 전제는 "right-multiply"
        self.declare_parameter("apply_T_SA", True)
        self.declare_parameter("T_SA_side", "right")  # ✅ default RIGHT (ver5와 맞춤)
        self.declare_parameter("debug_print_T_SA", False)

        # --- rotvec 연속화 (π 근처 튐 완화)
        self.declare_parameter("rotvec_continuous", True)

        # --- 출력 좌표계/부호 보정 (물리적으로 가능한 보정만 제공)
        # none
        # rot_y_pi_left : world에서 y축으로 180도 회전(= x,z 부호가 뒤집힌 것처럼 보이는 케이스에 대응)
        # rot_x_pi_left, rot_z_pi_left 도 필요하면 사용
        # ✅ 네가 쓰던 flip_x_wx_wz 는 alias로 rot_y_pi_left 처리
        self.declare_parameter("out_fix_mode", "none")

        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)

        self.apply_T_SA = bool(self.get_parameter("apply_T_SA").value)
        self.T_SA_side = str(self.get_parameter("T_SA_side").value).lower()
        self.debug_print_T_SA = bool(self.get_parameter("debug_print_T_SA").value)

        self.rotvec_continuous = bool(self.get_parameter("rotvec_continuous").value)

        self.out_fix_mode = str(self.get_parameter("out_fix_mode").value).lower()
        if self.out_fix_mode == "flip_x_wx_wz":
            self.out_fix_mode = "rot_y_pi_left"

        self.tf_broadcaster = TransformBroadcaster(self)
        # self.timer = self.create_timer(0.008, self.cb_vive_timer) # 125 Hz
        self.timer = self.create_timer(0.002, self.cb_vive_timer) # 500 Hz

    # ------------------------------------------------------------------
    # VR init
    # ------------------------------------------------------------------
    def _init_vr(self):
        try:
            self.get_logger().info("Initializing VR system...")
            self.vr_system = openvr.init(openvr.VRApplication_Other)
            self.get_logger().info("VR system initialized successfully!")

            poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            _ = poses  # just to trigger
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                device_class = self.vr_system.getTrackedDeviceClass(i)
                if device_class != openvr.TrackedDeviceClass_Invalid:
                    self.get_logger().info(f"[vive_tracker_node] Found VR device: (index {i})")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize VR system: {e}")
            self.vr_system = None

    # ------------------------------------------------------------------
    # json calib (service 결과)
    # ------------------------------------------------------------------
    def _init_json_calib(self):
        share_dir = ament_index_python.packages.get_package_share_directory("vive_tracker_ros2")
        self.config_dir_install = os.path.join(share_dir, "config")

        json_path = os.path.join(self.config_dir_install, "calibration_matrix.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            self.T_tool_opt = np.array(data.get("T_tool_opt", np.eye(4)), dtype=np.float64)
            self.T_trans_opt = np.array(data.get("T_trans_opt", np.eye(4)), dtype=np.float64)
        else:
            self.T_tool_opt = np.eye(4, dtype=np.float64)
            self.T_trans_opt = np.eye(4, dtype=np.float64)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_T44(mat_like):
        if mat_like is None:
            return None
        M = np.array(mat_like, dtype=np.float64)
        if M.shape == (4, 4):
            return M
        if M.shape == (3, 3):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = M
            return T
        return None

    @staticmethod
    def _is_valid_T(T: Optional[np.ndarray]) -> bool:
        if T is None or T.shape != (4, 4):
            return False
        if not np.all(np.isfinite(T)):
            return False
        if np.linalg.norm(T[3, :] - np.array([0, 0, 0, 1], dtype=np.float64)) > 1e-3:
            return False
        return True

    def _fix_left_matrix(self) -> np.ndarray:
        """
        out_fix_mode에 따라 M_cal에 '좌곱'으로 들어가는 보정행렬 반환.
        (물리적으로 가능한 회전만 제공)
        """
        if self.out_fix_mode in ["none", "", "off"]:
            return np.eye(4, dtype=np.float64)

        R = np.eye(3, dtype=np.float64)
        if self.out_fix_mode == "rot_y_pi_left":
            R = rot_y(math.pi)
        elif self.out_fix_mode == "rot_x_pi_left":
            R = rot_x(math.pi)
        elif self.out_fix_mode == "rot_z_pi_left":
            R = rot_z(math.pi)
        else:
            self.get_logger().warn(f"[out_fix_mode] unknown '{self.out_fix_mode}', using none")
            R = np.eye(3, dtype=np.float64)

        F = np.eye(4, dtype=np.float64)
        F[:3, :3] = R
        return F

    # ------------------------------------------------------------------
    # yaml calib
    # ------------------------------------------------------------------
    def _init_yaml_calib(self):
        src_yaml = os.path.join(os.path.expanduser("~/nrs_imitation/behavior_ws/src/vive_tracker_ros2/yaml"),
                                "calibration_matrix.yaml")
        share_dir = ament_index_python.packages.get_package_share_directory("vive_tracker_ros2")
        install_yaml = os.path.join(share_dir, "yaml", "calibration_matrix.yaml")

        yaml_path = src_yaml if os.path.exists(src_yaml) else install_yaml
        self.get_logger().info(f"[vive_tracker_node] load yaml: {yaml_path}")

        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        self.T_AD = np.array(data.get("T_AD", np.eye(4)), dtype=np.float64)
        self.T_BC = np.array(data.get("T_BC", np.eye(4)), dtype=np.float64)
        self.T_CE = np.array(data.get("T_CE", np.eye(4)), dtype=np.float64)
        self.R_Adj = np.array(data.get("R_Adj", np.eye(3)), dtype=np.float64)
        T_FIX_loaded = self._to_T44(data.get("T_FIX", None))

        # ROS1 순서 호환: T_Adj = R_Adj.T
        self.T_Adj = np.eye(4, dtype=np.float64)
        self.T_Adj[:3, :3] = self.R_Adj.T

        # T_SA: YAML 우선, 없으면 DEFAULT
        T_SA_loaded = self._to_T44(data.get("T_SA", None))
        if self._is_valid_T(T_SA_loaded):
            self.T_SA = T_SA_loaded
            self.get_logger().info("[vive_tracker_node] T_SA loaded from yaml.")
        else:
            self.T_SA = DEFAULT_T_SA.copy()
            self.get_logger().warn("[vive_tracker_node] T_SA not found/invalid in yaml. Using Identity.")

        # out_fix + YAML z-plane correction, both left-multiplied.
        self.T_FIX_LEFT = self._fix_left_matrix()
        if self._is_valid_T(T_FIX_loaded):
            self.T_FIX = T_FIX_loaded @ self.T_FIX_LEFT
            self.get_logger().info("[vive_tracker_node] T_FIX loaded from yaml and combined with out_fix.")
        else:
            self.T_FIX = self.T_FIX_LEFT.copy()
            self.get_logger().warn("[vive_tracker_node] T_FIX not found/invalid in yaml. Using out_fix only.")

        self.get_logger().info(
            f"[vive_tracker_node] apply_T_SA={self.apply_T_SA}, T_SA_side={self.T_SA_side}, "
            f"rotvec_continuous={self.rotvec_continuous}, out_fix_mode={self.out_fix_mode}"
        )
        if self.debug_print_T_SA:
            self.get_logger().info("T_SA=\n" + np.array2string(self.T_SA, precision=6, suppress_small=True))
            self.get_logger().info("T_FIX=\n" + np.array2string(self.T_FIX, precision=6, suppress_small=True))

    # ------------------------------------------------------------------
    # service
    # ------------------------------------------------------------------
    def cb_calibrate(self, request, response):
        robot_matrices = [pose_to_matrix(p) for p in request.robot_poses]
        tracker_matrices = [pose_to_matrix(p) for p in request.tracker_poses]

        if len(robot_matrices) != len(tracker_matrices):
            self.get_logger().error("robot pose count != tracker pose count")
            response.success = False
            return response

        try:
            self.T_tool_opt, self.T_trans_opt = calculate_calibration_matrix(robot_matrices, tracker_matrices)
        except Exception as e:
            self.get_logger().error(f"calibration failed: {e}")
            response.success = False
            return response

        os.makedirs(self.config_dir_install, exist_ok=True)
        with open(os.path.join(self.config_dir_install, "calibration_matrix.json"), "w") as f:
            json.dump({"T_tool_opt": self.T_tool_opt.tolist(),
                       "T_trans_opt": self.T_trans_opt.tolist()}, f, indent=4)

        response.success = True
        return response

    # ------------------------------------------------------------------
    # trackers update
    # ------------------------------------------------------------------
    def _update_trackers_from_vr(self) -> List[str]:
        if self.vr_system is None:
            return []

        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )

        current_ids: List[str] = []
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vr_system.getTrackedDeviceClass(i)
            if device_class != openvr.TrackedDeviceClass_GenericTracker:
                continue

            pose = poses[i]
            if not (pose.bDeviceIsConnected and pose.bPoseIsValid):
                continue

            try:
                serial = self.vr_system.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
            except Exception:
                serial = f"tracker_{i}"

            safe_serial = re.sub(r"[^a-zA-Z0-9_]", "_", serial)

            if serial not in self.trackers:
                raw_topic = f"vive_tracker_ros/{safe_serial}/raw_pose"
                cali_topic = f"vive_tracker_ros/{safe_serial}/calibrated_pose"
                child_frame = f"{self.child_frame}_{safe_serial}"

                self.trackers[serial] = {
                    "device_index": i,
                    "child_frame": child_frame,
                    "publisher_raw": self.create_publisher(Odometry, raw_topic, 10),
                    "publisher_calibrated": self.create_publisher(Odometry, cali_topic, 10),
                    "prev_raw_matrix": np.eye(4, dtype=np.float64),
                    "prev_calibrated_matrix": np.eye(4, dtype=np.float64),
                    "prev_rotvec": None,     # 연속 rotvec용
                    "prev_quat_wxyz": None,  # 연속 quat sign용
                }
                self.get_logger().info(f"새 트래커 발견: {serial}")

            raw_pose_matrix = openvr_pose_to_np44(pose)
            self.trackers[serial]["raw_pose_matrix"] = raw_pose_matrix
            current_ids.append(serial)

        return current_ids

    # ------------------------------------------------------------------
    # message helpers
    # ------------------------------------------------------------------
    def create_vive_msg(self, pose: Pose, twist: Twist, frame_id="world") -> Odometry:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.pose = pose
        msg.twist.twist = twist
        return msg

    def create_pose_stamped(self, pose: Pose, frame_id="world") -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose = pose
        return msg

    def _apply_T_SA_to_M_cal(self, M_cal: np.ndarray) -> np.ndarray:
        if not self.apply_T_SA:
            return M_cal

        # ✅ ver5(vr_calibration) 전제 = right-multiply
        if self.T_SA_side == "left":
            return self.T_SA @ M_cal
        return M_cal @ self.T_SA

    # ------------------------------------------------------------------
    # main timer
    # ------------------------------------------------------------------
    def cb_vive_timer(self):
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 1e-6:
            dt = 1e-3

        current_ids = self._update_trackers_from_vr()
        if not current_ids:
            self.prev_time = now
            return

        for serial in current_ids:
            tdata = self.trackers[serial]
            raw_M = tdata["raw_pose_matrix"]

            # base chain (ROS1 순서 유지)
            M_adj = self.T_Adj @ raw_M
            M_cal = self.T_AD @ M_adj @ self.T_CE

            # out_fix + z-plane correction: left-multiplied in base/world frame.
            M_cal = self.T_FIX @ M_cal

            # ✅ spatial-angle alignment
            M_cal = self._apply_T_SA_to_M_cal(M_cal)

            # pose로
            raw_pose = matrix_to_pose(raw_M)
            cal_pose = matrix_to_pose(M_cal)

            raw_twist = matrix_to_twist(raw_M, tdata["prev_raw_matrix"], dt)
            cal_twist = matrix_to_twist(M_cal, tdata["prev_calibrated_matrix"], dt)

            # tracker별 odom publish
            tdata["publisher_raw"].publish(self.create_vive_msg(raw_pose, raw_twist))
            tdata["publisher_calibrated"].publish(self.create_vive_msg(cal_pose, cal_twist))

            # /calibrated_pose publish: [x y z wx wy wz]
            px = float(M_cal[0, 3])
            py = float(M_cal[1, 3])
            pz = float(M_cal[2, 3])

            Rm = M_cal[:3, :3]

            # --- rotvec (연속 옵션) ---
            if self.rotvec_continuous:
                q = rotmat_to_quat(Rm)  # [w,x,y,z]
                q_prev = tdata["prev_quat_wxyz"]
                if q_prev is not None and float(np.dot(q_prev, q)) < 0.0:
                    q = -q  # quaternion sign continuity
                w_prev = tdata["prev_rotvec"]
                wvec = quat_to_rotvec_cont(q, w_prev)
                tdata["prev_quat_wxyz"] = q
                tdata["prev_rotvec"] = wvec
            else:
                # fallback: 비연속(권장X)
                q = rotmat_to_quat(Rm)
                wvec = quat_to_rotvec_cont(q, None)

            arr = Float64MultiArray()
            arr.data = [px, py, pz, float(wvec[0]), float(wvec[1]), float(wvec[2])]
            self.calibrated_pose_pub.publish(arr)

            # TF
            if self.publish_tf:
                t = TransformStamped()
                t.header.stamp = now.to_msg()
                t.header.frame_id = self.base_frame
                t.child_frame_id = tdata["child_frame"]
                t.transform.translation.x = cal_pose.position.x
                t.transform.translation.y = cal_pose.position.y
                t.transform.translation.z = cal_pose.position.z
                t.transform.rotation = cal_pose.orientation
                self.tf_broadcaster.sendTransform(t)

            tdata["prev_raw_matrix"] = raw_M.copy()
            tdata["prev_calibrated_matrix"] = M_cal.copy()

        # 호환용 퍼블리시 (첫 트래커)
        first_id = current_ids[0]
        first = self.trackers[first_id]

        raw_pose = matrix_to_pose(first["raw_pose_matrix"])
        raw_twist = matrix_to_twist(first["raw_pose_matrix"], first["prev_raw_matrix"], dt)

        self.raw_pose_pub_pose.publish(self.create_pose_stamped(raw_pose, frame_id="world"))
        self.raw_pose_pub.publish(self.create_vive_msg(raw_pose, raw_twist))

        cal_pose = matrix_to_pose(first["prev_calibrated_matrix"])
        cal_twist = matrix_to_twist(first["prev_calibrated_matrix"], first["prev_calibrated_matrix"], dt)
        self.calibrated_pose_pub_odom.publish(self.create_vive_msg(cal_pose, cal_twist))

        self.prev_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ViveTracker()

    if node.vr_system is None:
        node.get_logger().error("VR system is not initialized. Exiting.")
        return 1

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            openvr.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    main()
