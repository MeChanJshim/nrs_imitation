#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vr_demo_joy_controller.py

Joystick command node for VR demo HDF5 recorder.

Requires:
  ros2 run joy joy_node

This node subscribes:
  /joy                      sensor_msgs/Joy

This node publishes:
  /vr_demo_recorder/command std_msgs/String

Default Logitech F710 / Xbox-like mapping:
  A        buttons[0]  -> start_recording
  B        buttons[1]  -> end_recording
  X        buttons[2]  -> erase_current_episode
  Y        buttons[3]  -> terminate_node
  D-pad L  axes[6]     -> prev_episode
  D-pad R  axes[6]     -> next_episode

Published command strings:
  start_recording
  end_recording
  erase_current_episode
  terminate_node
  prev_episode
  next_episode
"""

import time
from typing import List, Optional

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import String


class VRDemoJoyController(Node):
    def __init__(self):
        super().__init__("vr_demo_joy_controller")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("command_topic", "/vr_demo_recorder/command")

        # Logitech F710 / Xbox-like default button mapping
        self.declare_parameter("button_a", 0)
        self.declare_parameter("button_b", 1)
        self.declare_parameter("button_x", 2)
        self.declare_parameter("button_y", 3)

        # D-pad left/right axis
        # For many Logitech F710 / Xbox layouts:
        #   axes[6] = +1 or -1 depending on left/right
        self.declare_parameter("dpad_lr_axis", 6)
        self.declare_parameter("dpad_threshold", 0.5)

        # If your left/right direction is reversed, flip these two values by ros2 param.
        self.declare_parameter("dpad_left_positive", True)

        # Safety / debounce
        self.declare_parameter("button_debounce_sec", 0.20)
        self.declare_parameter("dpad_debounce_sec", 0.20)

        # Load params
        self.joy_topic = str(self.get_parameter("joy_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)

        self.button_a = int(self.get_parameter("button_a").value)
        self.button_b = int(self.get_parameter("button_b").value)
        self.button_x = int(self.get_parameter("button_x").value)
        self.button_y = int(self.get_parameter("button_y").value)

        self.dpad_lr_axis = int(self.get_parameter("dpad_lr_axis").value)
        self.dpad_threshold = float(self.get_parameter("dpad_threshold").value)
        self.dpad_left_positive = bool(self.get_parameter("dpad_left_positive").value)

        self.button_debounce_sec = float(self.get_parameter("button_debounce_sec").value)
        self.dpad_debounce_sec = float(self.get_parameter("dpad_debounce_sec").value)

        # Runtime state
        self.prev_buttons: Optional[List[int]] = None
        self.prev_dpad_lr_state = 0  # -1, 0, +1

        self.last_button_cmd_time = 0.0
        self.last_dpad_cmd_time = 0.0

        # ROS I/O
        self.cmd_pub = self.create_publisher(String, self.command_topic, 10)
        self.joy_sub = self.create_subscription(Joy, self.joy_topic, self.joy_callback, 20)

        self.get_logger().info("============================================================")
        self.get_logger().info("VRDemoJoyController initialized")
        self.get_logger().info(f"  joy_topic     : {self.joy_topic}")
        self.get_logger().info(f"  command_topic : {self.command_topic}")
        self.get_logger().info("")
        self.get_logger().info("  Mapping:")
        self.get_logger().info(f"    A button [{self.button_a}] -> start_recording")
        self.get_logger().info(f"    B button [{self.button_b}] -> end_recording")
        self.get_logger().info(f"    X button [{self.button_x}] -> erase_current_episode")
        self.get_logger().info(f"    Y button [{self.button_y}] -> terminate_node")
        self.get_logger().info(f"    D-pad LR axis [{self.dpad_lr_axis}] -> prev/next episode")
        self.get_logger().info(f"    dpad_left_positive={self.dpad_left_positive}")
        self.get_logger().info("============================================================")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def publish_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"[JOY CMD] {cmd}")

    @staticmethod
    def _button_pressed(curr_buttons: List[int], prev_buttons: List[int], idx: int) -> bool:
        if idx < 0:
            return False
        if idx >= len(curr_buttons) or idx >= len(prev_buttons):
            return False
        return curr_buttons[idx] == 1 and prev_buttons[idx] == 0

    def _get_dpad_lr_state(self, axes: List[float]) -> int:
        """
        Return:
          -1 : left
           0 : neutral
          +1 : right
        """
        if self.dpad_lr_axis < 0 or self.dpad_lr_axis >= len(axes):
            return 0

        v = float(axes[self.dpad_lr_axis])

        if abs(v) < self.dpad_threshold:
            return 0

        # Many controllers use +1 for left, -1 for right.
        # dpad_left_positive=True:
        #   v > threshold  -> left
        #   v < -threshold -> right
        if self.dpad_left_positive:
            if v > self.dpad_threshold:
                return -1
            if v < -self.dpad_threshold:
                return +1
        else:
            if v > self.dpad_threshold:
                return +1
            if v < -self.dpad_threshold:
                return -1

        return 0

    # ------------------------------------------------------------
    # Joy callback
    # ------------------------------------------------------------
    def joy_callback(self, msg: Joy):
        curr_buttons = list(msg.buttons)
        curr_axes = list(msg.axes)

        if self.prev_buttons is None:
            self.prev_buttons = curr_buttons
            self.prev_dpad_lr_state = self._get_dpad_lr_state(curr_axes)
            return

        now = time.time()

        # -------------------------
        # Button rising-edge commands
        # -------------------------
        if now - self.last_button_cmd_time >= self.button_debounce_sec:
            if self._button_pressed(curr_buttons, self.prev_buttons, self.button_a):
                self.publish_command("start_recording")
                self.last_button_cmd_time = now

            elif self._button_pressed(curr_buttons, self.prev_buttons, self.button_b):
                self.publish_command("end_recording")
                self.last_button_cmd_time = now

            elif self._button_pressed(curr_buttons, self.prev_buttons, self.button_x):
                self.publish_command("erase_current_episode")
                self.last_button_cmd_time = now

            elif self._button_pressed(curr_buttons, self.prev_buttons, self.button_y):
                self.publish_command("terminate_node")
                self.last_button_cmd_time = now

        # -------------------------
        # D-pad left/right edge commands
        # -------------------------
        curr_dpad_lr_state = self._get_dpad_lr_state(curr_axes)

        # Only trigger when neutral -> left/right
        if now - self.last_dpad_cmd_time >= self.dpad_debounce_sec:
            if self.prev_dpad_lr_state == 0 and curr_dpad_lr_state == -1:
                self.publish_command("prev_episode")
                self.last_dpad_cmd_time = now

            elif self.prev_dpad_lr_state == 0 and curr_dpad_lr_state == +1:
                self.publish_command("next_episode")
                self.last_dpad_cmd_time = now

        self.prev_buttons = curr_buttons
        self.prev_dpad_lr_state = curr_dpad_lr_state


def main(args=None):
    rclpy.init(args=args)
    node = VRDemoJoyController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()