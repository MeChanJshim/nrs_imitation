### Task parameters
# Author: Chemin Ahn (chemx3937@gmail.com)
# Modified for UR10e Isaac Sim ACT dataset (by Eunseop)

import os
import math

# -------------------------------------------------------------------------
# 기본 경로 설정
#   C++ 레코더가 저장하는 최상위 디렉토리:
#   /home/eunseop/nrs_lab2/datasets/ACT/<MMDD_HHMM>/act_data.hdf5
# -------------------------------------------------------------------------
# DATA_DIR = "/home/eunseop/nrs_lab2/datasets/ACT"
DATA_DIR = "/home/eunseop/nrs_imitation/datasets/ACT"


def _find_latest_run_dir(base_dir: str) -> str:
    """ACT/ 아래에서 가장 최근에 만든 폴더 하나 골라서 반환"""
    base_dir = os.path.expanduser(base_dir)
    if not os.path.isdir(base_dir):
        return base_dir

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        return base_dir

    # 수정 시간 기준으로 최신 폴더 반환
    return max(subdirs, key=os.path.getmtime)


# -------------------------------------------------------------------------
# 가장 최근 기록 폴더 자동 인식
# 변환 스크립트(demo_data_act_form.py)가 episodes/ 하위에 HDF5 생성한다고 가정
# -------------------------------------------------------------------------
_latest = _find_latest_run_dir(DATA_DIR)
_default_episode_dir = os.path.join(_latest, "episodes_ft")

# -------------------------------------------------------------------------
# Task 설정 (현재 UR10e 환경용)
# -------------------------------------------------------------------------
TASK_CONFIGS = {
    "ur10e_swing": {
        "dataset_dir": _default_episode_dir,  # 자동으로 최신 경로 선택
        "camera_names": ["cam_front", "cam_head"],  # front / top camera
        "num_episodes": 51,  # episode_0.hdf5 ~ episode_50.hdf5
        "num_actions": 6,    # UR10e: 6 DOF
        "obs_horizon": 1,
        "action_horizon": 8,
        "pred_horizon": 8,
        "episode_len": 420,  # 한 에피소드 프레임 수 (기본 100~200 사이 추천)
    }
}


# -------------------------------------------------------------------------
# ALOHA / UR10e 제어 고정 상수
# -------------------------------------------------------------------------
DT = 0.05  # 20Hz (기록 주기와 동일)

# UR10e 조인트 이름
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# -------------------------------------------------------------------------
# 초기 포즈 (deg → rad)
# 0, -90, -90, -90, 90, 0
# -------------------------------------------------------------------------
base     = 0.0   * math.pi / 180.0
shoulder = -90.0 * math.pi / 180.0
elbow    = -90.0 * math.pi / 180.0
wrist1   = -90.0 * math.pi / 180.0
wrist2   = 90.0  * math.pi / 180.0
wrist3   = 0.0   * math.pi / 180.0

START_ARM_POSE = [base, shoulder, elbow, wrist1, wrist2, wrist3]
