import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'nrs_imitation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='eunseop',
    maintainer_email='lexondms1@g.skku.edu',
    description='ROS2 node for ACT policy inference with Isaac Sim UR10e',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'act_policy_infer = nrs_imitation.act_policy_infer:main',
            'vr_demo_recorder = nrs_imitation.vr_demo_recorder:main',
            'vr_demo_txt_recorder = nrs_imitation.vr_demo_txt_recorder:main',
            'vr_demo_hdf5_recorder = nrs_imitation.vr_demo_hdf5_recorder:main',
            'vr_demo_hdf5_episode_pusher = nrs_imitation.vr_demo_hdf5_episode_pusher:main',
            'robot_playback_act_hdf5_recorder = nrs_imitation.robot_playback_act_hdf5_recorder:main',
            'act_keyboard_infer_node = nrs_imitation.act_keyboard_infer_node:main',
            'act_txt_postprocessor = nrs_imitation.act_txt_postprocessor:main',
            'vr_demo_txt_wo_filter = nrs_imitation.vr_demo_txt_wo_filter:main',
            'node_act_cmdmotion_infer = nrs_imitation.node_act_cmdmotion_infer:main',
            'node_check_inference = nrs_imitation.node_check_inference:main',
            'ur10_keyboard_recorder = nrs_imitation.ur10_keyboard_recorder:main',

            # Joystick command node for VR demo recorder
            'vr_demo_joy_controller = nrs_imitation.vr_demo_joy_controller:main',
        ],
    },
)