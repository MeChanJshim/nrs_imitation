#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    joy_dev = LaunchConfiguration("joy_dev")
    deadzone = LaunchConfiguration("deadzone")
    autorepeat_rate = LaunchConfiguration("autorepeat_rate")
    dpad_left_positive = LaunchConfiguration("dpad_left_positive")

    return LaunchDescription([
        DeclareLaunchArgument(
            "joy_dev",
            default_value="/dev/input/js0",
            description="Joystick device path"
        ),
        DeclareLaunchArgument(
            "deadzone",
            default_value="0.05",
            description="Joystick deadzone"
        ),
        DeclareLaunchArgument(
            "autorepeat_rate",
            default_value="20.0",
            description="joy_node autorepeat rate"
        ),
        DeclareLaunchArgument(
            "dpad_left_positive",
            default_value="true",
            description="If true, axes[6] positive means D-pad left"
        ),

        # ------------------------------------------------------------
        # joy_node
        # ------------------------------------------------------------
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            output="screen",
            parameters=[{
                "dev": joy_dev,
                "deadzone": deadzone,
                "autorepeat_rate": autorepeat_rate,
            }],
        ),

        # ------------------------------------------------------------
        # command mapper node
        # ------------------------------------------------------------
        Node(
            package="nrs_imitation",
            executable="vr_demo_joy_controller",
            name="vr_demo_joy_controller",
            output="screen",
            parameters=[{
                "joy_topic": "/joy",
                "command_topic": "/vr_demo_recorder/command",

                # Logitech F710 / Xbox-like mapping
                "button_a": 0,
                "button_b": 1,
                "button_x": 2,
                "button_y": 3,

                # D-pad left/right
                "dpad_lr_axis": 6,
                "dpad_threshold": 0.5,
                "dpad_left_positive": dpad_left_positive,

                # debounce
                "button_debounce_sec": 0.20,
                "dpad_debounce_sec": 0.20,
            }],
        ),
    ])