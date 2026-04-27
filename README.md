# nrs_act

Refactored imitation-learning codebase for robotic polishing / manipulation experiments.

This repository now supports **two parallel policy branches** on top of the same dataset pipeline:

- **ACT**
- **Diffusion Policy-style branch**

Both branches share the same demonstration format:

- observation = `position(6) + force(3) + image(cam0)`
- action = `position(6) + force(3)`

The current workflow is designed so that you can go from **VR teaching / demonstration recording** all the way to **training / evaluation / inference** by following this README only.

---

# 0. End-to-End Quick Start

This is the shortest path from recording to training.

## 0-1. Build the ROS 2 behavior workspace

```bash
cd ~/nrs_act/behavior_ws
colcon build
source install/setup.bash
```

If you modify `nrs_imitation` only:

```bash
cd ~/nrs_act/behavior_ws
colcon build --packages-select nrs_imitation
source install/setup.bash
```

---

## 0-2. Run the joystick controller

This launch file starts both:

- `joy_node`
- `vr_demo_joy_controller`

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
```

If D-pad left/right is reversed:

```bash
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py dpad_left_positive:=false
```

You can check the command topic:

```bash
ros2 topic echo /vr_demo_recorder/command
```

Joystick mapping:

```text
A             -> start_recording
B             -> end_recording
X             -> erase_current_episode
Y             -> terminate_node
D-pad left    -> prev_episode
D-pad right   -> next_episode
```

---

## 0-3. Run the required sensor / input nodes before the recorder

Before running `vr_demo_hdf5_recorder`, make sure the following producer nodes are already running:

- Vive tracker node for position
- force/torque sensor node
- camera node
- optional image visualization node

### A. Vive tracker node

Alias:

```bash
alias vive='ros2 launch vive_tracker_ros2 vive_bringup.launch.py'
```

Run:

```bash
vive
```

This provides the calibrated pose stream used by the recorder in tracker mode.

---

### B. Force/torque sensor node

You currently use either one of the following:

```bash
alias ft='ros2 launch nrs_ft_aq2 nrsvr_ft_aq.launch.py'
```

Run:

```bash
ft
```

or:

```bash
alias ftget='ros2 run Y2FT_AQ FTGetMain'
```

Run:

```bash
ftget
```

Use whichever force node matches your current setup.

---

### C. Camera node

`rsv` and `rsr` are aliases of the following RealSense launch helper functions.

#### VR camera launch helper (`rs_vr`)

Use this when bringing up the VR-side camera (`camera_name:=vr`) in RGB-only mode.

Default:
- `640x480@30`

Function:

```bash
rs_vr() {
  local profile="${1:-$RS_VR_PROFILE_DEFAULT}"
  ros2 launch realsense2_camera rs_launch.py     camera_namespace:="$RS_NS" camera_name:=vr     serial_no:="'243622073271'"     $(_rs_common_args)     rgb_camera.profile:="${profile}"
}
```

Usage:

```bash
rsv
rsv 640x480x15
```

Expected image topic:

```text
/realsense/vr/color/image_raw
```

#### Robot camera launch helper (`rs_robot`)

Use this when bringing up the robot-side / end-effector camera (`camera_name:=robot`) in RGB-only mode.

Default:
- `640x480@15`  (more stable on USB2.1)

Function:

```bash
rs_robot() {
  local profile="${1:-$RS_ROBOT_PROFILE_DEFAULT}"
  ros2 launch realsense2_camera rs_launch.py     camera_namespace:="$RS_NS" camera_name:=robot     serial_no:="'244222070489'"     $(_rs_common_args)     rgb_camera.profile:="${profile}"
}
```

Usage:

```bash
rsr
rsr 424x240x30
```

Expected image topic:

```text
/realsense/robot/color/image_raw
```

---

### D. Optional camera visualization

VR camera viewer:

```bash
rs_view_vr(){ ros2 run image_tools showimage --ros-args -r image:=/${RS_NS}/vr/color/image_raw; }
```

Run:

```bash
rs_view_vr
```

Robot camera viewer:

```bash
rs_view_robot() { ros2 run image_tools showimage --ros-args -r image:=/${RS_NS}/robot/color/image_raw; }
```

Run:

```bash
rs_view_robot
```

---

## 0-4. Run the recorder

The recorder now supports **two recording modes** selected by ROS parameter.

### A. Tracker recording mode

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args   -p recording_mode:=tracker
```

This mode records from:

```text
position : /calibrated_pose
force    : /ftsensor/measured_Cvalue
image    : /realsense/vr/color/image_raw
command  : /vr_demo_recorder/command
```

### B. Robot recording mode

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args   -p recording_mode:=robot
```

This mode records from:

```text
position : /ur10skku/currentP
force    : /ur10skku/currentF
image    : /realsense/robot/color/image_raw
command  : /vr_demo_recorder/command
```

### C. Optional manual topic override

If needed, topic names can still be overridden manually:

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args   -p recording_mode:=robot   -p image_topic:=/realsense/vr/color/image_raw
```

Recorder output path:

```text
/home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/
└── vr_demo_merged_YYYYMMDD_HHMM.hdf5
```

Merged HDF5 layout:

```text
episodes/
├── ep_0000/
│   ├── position
│   ├── ft
│   └── images/
│       └── cam0
├── ep_0001/
│   ├── position
│   ├── ft
│   └── images/
│       └── cam0
...
```

All joystick behavior and episode-management logic are shared across both modes.

---

## 0-5. Convert merged HDF5 into ACT / Diffusion training episodes

Converter is a normal Python script.

### A. Raw image version

```bash
cd ~/nrs_act/source/custom

python3 demo_data_act_form_single_cam.py \
  --cam_preprocess off
```

This creates:

```text
/home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/episodes_ft/
```

### B. Camera-preprocessed version (recommended if hand jitter is visible)

```bash
cd ~/nrs_act/source/custom

python3 demo_data_act_form_single_cam.py \
  --cam_preprocess stabilize_crop \
  --cam_crop_h 384 \
  --cam_crop_w 384 \
  --cam_resize_hw 256
```

This creates:

```text
/home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/episodes_ft_camproc/
```

By default, after successful conversion the merged HDF5 is deleted to save disk space.

If you want to keep the merged HDF5 for debugging:

```bash
python3 demo_data_act_form_single_cam.py \
  --cam_preprocess stabilize_crop \
  --cam_crop_h 384 \
  --cam_crop_w 384 \
  --cam_resize_hw 256 \
  --keep-merged
```

---

## 0-6. Train ACT

### A. Raw image dataset

```bash
cd ~/nrs_act/scripts/act
python3 train_act.py --cam_preprocess off
```

### B. Camera-preprocessed dataset

```bash
cd ~/nrs_act/scripts/act
python3 train_act.py --cam_preprocess stabilize_crop
```

Default behavior:
- latest dataset directory is found automatically
- `camera_names = ["cam0"]`
- `use_force_history = True`
- `force_history_len = 10`
- periodic checkpoint every 100 epochs

---

## 0-7. Train Diffusion

### A. Raw image dataset

```bash
cd ~/nrs_act
python3 scripts/diffusion/train_diffusion.py --cam_preprocess off
```

### B. Camera-preprocessed dataset

```bash
cd ~/nrs_act
python3 scripts/diffusion/train_diffusion.py --cam_preprocess stabilize_crop
```

---

## 0-8. Evaluate checkpoint loading

### ACT

```bash
cd ~/nrs_act/scripts/act
python3 train_act.py --eval
```

### Diffusion

```bash
cd ~/nrs_act
python3 scripts/diffusion/train_diffusion.py --eval
```

---

## 0-9. Run inference node

The inference node now supports both **ACT** and **DIFFUSION** through a parameter.

### ACT

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 run nrs_imitation node_act_cmdmotion_infer --ros-args \
  -p policy_class:=ACT \
  -p act_root:=/home/eunseop/nrs_act \
  -p ckpt_dir:=/home/eunseop/nrs_act/checkpoints/act/ur10e_swing/20260423_1549 \
  -p image_topic:=/realsense/robot/color/image_raw \
  -p chunk_size:=200 \
  -p use_force_history:=true \
  -p force_history_len:=10
```

### Diffusion

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 run nrs_imitation node_act_cmdmotion_infer --ros-args \
  -p policy_class:=DIFFUSION \
  -p act_root:=/home/eunseop/nrs_act \
  -p ckpt_dir:=/home/eunseop/nrs_act/checkpoints/diffusion/ur10e_swing/20260424_1301 \
  -p image_topic:=/realsense/robot/color/image_raw \
  -p chunk_size:=200 \
  -p use_force_history:=true \
  -p force_history_len:=10 \
  -p diffusion_infer_steps:=10
```

---

# 1. Current End-to-End Pipeline

## Previous pipeline

The older workflow used robot playback to generate image data:

```text
1. vr_demo_hdf5_recorder.py
   -> save position / force demonstration

2. vr_demo_hdf5_episode_pusher.py
   -> send one episode to robot

3. robot_playback_act_hdf5_recorder.py
   -> record robot playback with position + force + image

4. demo_data_act_form.py
   -> split merged HDF5 into episode_*.hdf5
```

## Current pipeline

The current workflow records **position + force + image directly during teaching**:

```text
1. joystick + vr_demo_hdf5_recorder.py
   -> record position + force + cam0 into merged_hdf5
   -> supports tracker mode and robot mode

2. demo_data_act_form_single_cam.py
   -> convert merged_hdf5 into episode_*.hdf5
   -> raw or camera-preprocessed variant

3. train_act.py
   -> ACT training

4. scripts/diffusion/train_diffusion.py
   -> Diffusion Policy-style training

5. node_act_cmdmotion_infer.py
   -> ACT / DIFFUSION inference
```

---

# 2. Repository Structure

```text
nrs_act/
├── LICENSE
├── README.md
├── behavior_ws/
│   ├── build/
│   ├── install/
│   ├── log/
│   └── src/
│       ├── nrs_ft_aq2/
│       ├── nrs_imitation/
│       │   ├── launch/
│       │   │   └── vr_demo_joy_controller.launch.py
│       │   ├── nrs_imitation/
│       │   │   ├── vr_demo_hdf5_recorder.py
│       │   │   ├── vr_demo_joy_controller.py
│       │   │   ├── node_act_cmdmotion_infer.py
│       │   │   └── ...
│       │   ├── package.xml
│       │   └── setup.py
│       ├── vive_tracker_interfaces/
│       ├── vive_tracker_ros2/
│       └── vr_calibration/
├── checkpoints/
│   ├── act/
│   │   └── ur10e_swing/
│   │       └── YYYYMMDD_HHMM/
│   └── diffusion/
│       └── ur10e_swing/
│           └── YYYYMMDD_HHMM/
├── datasets/
│   └── ACT/
│       └── YYYYMMDD_HHMM/
│           ├── merged_hdf5/
│           ├── episodes_ft/
│           └── episodes_ft_camproc/
├── scripts/
│   ├── act/
│   │   └── train_act.py
│   └── diffusion/
│       └── train_diffusion.py
└── source/
    ├── common/
    │   ├── fs.py
    │   └── utils.py
    ├── custom/
    │   ├── custom_constants.py
    │   ├── custom_real_env.py
    │   ├── custom_robot_utils.py
    │   ├── demo_data_act_form.py
    │   └── demo_data_act_form_single_cam.py
    ├── data/
    │   ├── dataset.py
    │   ├── loader.py
    │   └── normalization.py
    ├── models/
    │   ├── act_core.py
    │   ├── backbone.py
    │   ├── diffusion_core.py
    │   ├── encoder.py
    │   ├── policy.py
    │   └── transformer.py
    └── training/
        ├── debug.py
        ├── engine.py
        └── plotting.py
```

---

# 3. What Each Folder Does

## `behavior_ws/`
ROS 2 workspace for the real experiment pipeline.

Main role:
- Vive tracker communication
- calibration
- force/torque acquisition
- image acquisition
- joystick command mapping
- demonstration recording
- online inference

This is the **real-world behavior side** of the repository.

---

## `datasets/`
Stores:
- merged HDF5 files from direct teaching
- final episode files used by dataloaders

Typical layout:

```text
datasets/ACT/
└── YYYYMMDD_HHMM/
    ├── merged_hdf5/
    │   └── vr_demo_merged_YYYYMMDD_HHMM.hdf5
    ├── episodes_ft/
    │   ├── episode_0.hdf5
    │   └── ...
    └── episodes_ft_camproc/
        ├── episode_0.hdf5
        └── ...
```

---

## `checkpoints/`
Now split by policy family:

```text
checkpoints/
├── act/
│   └── ur10e_swing/
│       └── YYYYMMDD_HHMM/
└── diffusion/
    └── ur10e_swing/
        └── YYYYMMDD_HHMM/
```

Each timestamp directory typically contains:

```text
policy_best.ckpt
policy_last.ckpt
dataset_stats.pkl
policy_epoch_0_seed_0.ckpt
policy_epoch_100_seed_0.ckpt
policy_epoch_200_seed_0.ckpt
...
train_val_loss_seed_0.png
train_val_l1_seed_0.png
train_val_kl_seed_0.png      # ACT
train_val_diffusion_seed_0.png   # Diffusion if applicable
```

---

## `scripts/`
Thin training / evaluation entrypoints.

- `scripts/act/train_act.py`
- `scripts/diffusion/train_diffusion.py`

These scripts:
- parse CLI args
- resolve latest dataset automatically
- construct policy config
- call shared dataloader / engine code

---

## `source/`
Core reusable modules.

### `source/data/`
- episode HDF5 reading
- normalization
- force history generation
- dataloaders

### `source/models/`
- ACT model
- Diffusion Policy-style model
- observation encoders
- image backbones

### `source/training/`
- common training loop
- checkpoint save
- debug print
- plot save

### `source/custom/`
- task-specific utilities
- merged-HDF5 to episode conversion

---

# 4. Recorder and Joystick Details

## 4-1. Recorder node

Run:

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_demo_hdf5_recorder
```

This node records:
- position
- force
- image

into merged HDF5.

Merged structure:

```text
episodes/
├── ep_0000/
│   ├── position
│   ├── ft
│   └── images/
│       └── cam0
├── ep_0001/
│   ├── position
│   ├── ft
│   └── images/
│       └── cam0
...
```

Semantic meaning:

```text
position = [x, y, z, wx, wy, wz]
ft       = [fx, fy, fz]
cam0     = RGB image
```

---

## 4-2. Recording modes

The recorder supports two source modes.

### Tracker recording mode

```text
position : /calibrated_pose
force    : /ftsensor/measured_Cvalue
image    : /realsense/vr/color/image_raw
```

This is the default direct-teaching mode.

### Robot recording mode

```text
position : /ur10skku/currentP
force    : /ur10skku/currentF
image    : /realsense/robot/color/image_raw
```

This is useful when you want to record robot-side playback / execution data while keeping the same HDF5 format.

Run examples:

```bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=tracker
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=robot
```

---
## 4-2. Joystick commands

The recorder is controlled through `/vr_demo_recorder/command`.

Commands:

```text
start_recording
end_recording
erase_current_episode
terminate_node
prev_episode
next_episode
```

Joystick mapping:

```text
A  -> start_recording
B  -> end_recording
X  -> erase_current_episode
Y  -> terminate_node
←  -> prev_episode
→  -> next_episode
```

---

# 5. Converter: Raw vs Camera-Preprocessed Dataset

## 5-1. Raw conversion

```bash
cd ~/nrs_act/source/custom

python3 demo_data_act_form_single_cam.py \
  --cam_preprocess off
```

Output:

```text
.../episodes_ft/
```

---

## 5-2. Camera-preprocessed conversion

Recommended when:
- hand jitter is visible
- camera shake is large
- raw teaching image is too noisy

```bash
cd ~/nrs_act/source/custom

python3 demo_data_act_form_single_cam.py \
  --cam_preprocess stabilize_crop \
  --cam_crop_h 384 \
  --cam_crop_w 384 \
  --cam_resize_hw 256
```

Output:

```text
.../episodes_ft_camproc/
```

---

## 5-3. What `stabilize_crop` means

The camera-preprocessed converter performs:

```text
raw cam0
→ episode-wise global stabilization
→ crop
→ resize
→ final episode_*.hdf5
```

This is meant to reduce camera shake from direct tracker teaching.

---

## 5-4. Keep merged HDF5 for debugging

```bash
python3 demo_data_act_form_single_cam.py \
  --cam_preprocess stabilize_crop \
  --cam_crop_h 384 \
  --cam_crop_w 384 \
  --cam_resize_hw 256 \
  --keep-merged
```

---

# 6. Final Episode Format

Each final episode file contains:

```text
episode_0.hdf5
├── action/
│   ├── position
│   └── force
├── observations/
│   ├── position
│   ├── force
│   ├── images/
│   │   └── cam0
│   └── is_pad
└── meta/
    ├── orig_len
    ├── T_pad
    ├── pad_starts_at
    ├── truncated
    └── camera_name
```

Current dimensionality:

```text
observation qpos = position(6) + force(3) = 9D
action          = position(6) + force(3) = 9D
image           = cam0
```

So the current action definition is still:

\[
a_t = [x, y, z, w_x, w_y, w_z, f_x, f_y, f_z]
\]

---

# 7. Shared Observation Encoding

Both ACT and Diffusion reuse the same observation-side modular design as much as possible.

## Position / force state

\[
q_t = [x, y, z, w_x, w_y, w_z, f_x, f_y, f_z]
\]

## Optional force history

\[
H_t = [f_{t-L+1}, \dots, f_t]
\]

The codebase uses:
- position encoder
- force-history GRU encoder
- fusion encoder
- image encoder for `cam0`

This allows ACT and Diffusion to share:
- dataset format
- force-history generation
- observation encoders
- image backbone modules

---

# 8. ACT Training

## 8-1. Default training

```bash
cd ~/nrs_act/scripts/act
python3 train_act.py
```

This will automatically use the latest `episodes_ft` dataset.

---

## 8-2. Camera-preprocessed training

```bash
cd ~/nrs_act/scripts/act
python3 train_act.py --cam_preprocess stabilize_crop
```

This will automatically use the latest `episodes_ft_camproc` dataset.

---

## 8-3. Important ACT defaults

Current practical defaults:

```text
ckpt_dir        = /home/eunseop/nrs_act/checkpoints/act/ur10e_swing
policy_class    = ACT
task_name       = ur10e_swing
camera_names    = ["cam0"]
batch_size      = 6
num_epochs      = 500
lr              = 1e-4
chunk_size      = 200
train_seq_len   = 200
val_seq_len     = 200
hidden_dim      = 512
dim_feedforward = 3200
use_force_history = True
force_history_len = 10
save_every      = 100
```

---

## 8-4. ACT specific dataset selection

Raw:

```bash
python3 train_act.py --cam_preprocess off
```

Camera-preprocessed:

```bash
python3 train_act.py --cam_preprocess stabilize_crop
```

Specific dataset:

```bash
python3 train_act.py \
  --dataset_dir /home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/episodes_ft_camproc
```

---

## 8-5. ACT eval

```bash
cd ~/nrs_act/scripts/act
python3 train_act.py --eval
```

Specific checkpoint root:

```bash
python3 train_act.py \
  --eval \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/act/ur10e_swing
```

---

# 9. Diffusion Training

## 9-1. What this branch is

This repository now includes a **Diffusion Policy-style** branch.

The added diffusion class is inspired by the core idea of **Diffusion Policy**:
- condition on current observation
- model a future action chunk through diffusion
- train by predicting noise
- infer by iterative denoising

In this repository, the diffusion branch:
- keeps the same HDF5 dataset format
- keeps the same observation encoders when possible
- adds a diffusion denoiser model and a separate training script

Important note:

```text
This is a Diffusion Policy-style implementation integrated into this repository.
It is not an official upstream port of the original authors' code.
```

---

## 9-2. Why diffusion was added

ACT remains a strong baseline for action chunk prediction.

Diffusion was added because diffusion-based visuomotor policies are widely used as very strong modern baselines for manipulation / imitation learning, especially for multi-modal action distributions and long-horizon chunk generation.

---

## 9-3. Default Diffusion training

```bash
cd ~/nrs_act
python3 scripts/diffusion/train_diffusion.py
```

This automatically uses the latest `episodes_ft` dataset.

---

## 9-4. Camera-preprocessed Diffusion training

```bash
cd ~/nrs_act
python3 scripts/diffusion/train_diffusion.py --cam_preprocess stabilize_crop
```

This automatically uses the latest `episodes_ft_camproc` dataset.

---

## 9-5. Important Diffusion defaults

```text
ckpt_dir              = /home/eunseop/nrs_act/checkpoints/diffusion/ur10e_swing
camera_names          = ["cam0"]
chunk_size            = 200
train_seq_len         = 200
val_seq_len           = 200
use_force_history     = True
force_history_len     = 10
diffusion_train_steps = 100
diffusion_infer_steps = 10
diffusion_beta_start  = 1e-4
diffusion_beta_end    = 2e-2
save_every            = 100
```

---

## 9-6. Diffusion eval

```bash
cd ~/nrs_act
python3 scripts/diffusion/train_diffusion.py --eval
```

Specific checkpoint root:

```bash
python3 scripts/diffusion/train_diffusion.py \
  --eval \
  --ckpt_dir /home/eunseop/nrs_act/checkpoints/diffusion/ur10e_swing
```

---

# 10. Inference Node

The current inference node supports both ACT and DIFFUSION:

```text
policy_class = ACT | DIFFUSION
```

Common required topics:

```text
pose_topic   = /ur10skku/currentP
force_topic  = /ur10skku/currentF
image_topic  = /realsense/robot/color/image_raw
cmd_topic    = /ur10skku/cmdMotion
```

---

## 10-1. ACT inference example

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 run nrs_imitation node_act_cmdmotion_infer --ros-args \
  -p policy_class:=ACT \
  -p act_root:=/home/eunseop/nrs_act \
  -p ckpt_dir:=/home/eunseop/nrs_act/checkpoints/act/ur10e_swing/20260423_1549 \
  -p image_topic:=/realsense/robot/color/image_raw \
  -p chunk_size:=200 \
  -p use_force_history:=true \
  -p force_history_len:=10
```

---

## 10-2. Diffusion inference example

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash

ros2 run nrs_imitation node_act_cmdmotion_infer --ros-args \
  -p policy_class:=DIFFUSION \
  -p act_root:=/home/eunseop/nrs_act \
  -p ckpt_dir:=/home/eunseop/nrs_act/checkpoints/diffusion/ur10e_swing/20260424_1301 \
  -p image_topic:=/realsense/robot/color/image_raw \
  -p chunk_size:=200 \
  -p use_force_history:=true \
  -p force_history_len:=10 \
  -p diffusion_infer_steps:=10
```

---

## 10-3. Important inference notes

- current node is **single-camera only**
- `cam0` / `image_topic` are used
- force history is built online from the current force stream
- current no-recover style control is kept for simpler debugging

If checkpoint load prints a large mismatch such as:

```text
missing=364, unexpected=364
```

then the training config and inference config do not match and policy output should not be trusted.

---

# 11. Checkpoint Layout

## ACT

```text
/home/eunseop/nrs_act/checkpoints/act/ur10e_swing/YYYYMMDD_HHMM/
├── dataset_stats.pkl
├── policy_best.ckpt
├── policy_last.ckpt
├── policy_epoch_0_seed_0.ckpt
├── policy_epoch_100_seed_0.ckpt
├── policy_epoch_200_seed_0.ckpt
└── train_val_*.png
```

## Diffusion

```text
/home/eunseop/nrs_act/checkpoints/diffusion/ur10e_swing/YYYYMMDD_HHMM/
├── dataset_stats.pkl
├── policy_best.ckpt
├── policy_last.ckpt
├── policy_epoch_0_seed_0.ckpt
├── policy_epoch_100_seed_0.ckpt
├── policy_epoch_200_seed_0.ckpt
└── train_val_*.png
```

---

# 12. Useful Commands

## Latest recorder command topic

```bash
ros2 topic echo /vr_demo_recorder/command
```

## Verify image topic

```bash
ros2 topic hz /realsense/robot/color/image_raw
ros2 topic echo --once /realsense/robot/color/camera_info
```

## Verify pose / force topics

```bash
ros2 topic echo --once /ur10skku/currentP
ros2 topic echo --once /ur10skku/currentF
```


## Run recorder in tracker mode

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=tracker
```

## Run recorder in robot mode

```bash
cd ~/nrs_act/behavior_ws
source install/setup.bash
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=robot
```

## Check final episode structure

```bash
python3 - <<'PY'
import glob, h5py
files = sorted(glob.glob('/home/eunseop/nrs_act/datasets/ACT/*/episodes_ft/episode_0.hdf5'))
if not files:
    raise RuntimeError("No episode file found")
path = files[-1]
print("file:", path)
with h5py.File(path, 'r') as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    f.visititems(visit)
PY
```

## Check camera-preprocessed dataset

```bash
python3 - <<'PY'
import glob, h5py
files = sorted(glob.glob('/home/eunseop/nrs_act/datasets/ACT/*/episodes_ft_camproc/episode_0.hdf5'))
if not files:
    raise RuntimeError("No camproc episode file found")
path = files[-1]
print("file:", path)
with h5py.File(path, 'r') as f:
    print(f['observations/images/cam0'].shape)
PY
```

---

# 13. Troubleshooting

## A. `ros2 launch ... vr_demo_joy_controller.launch.py` works but recorder does not react
Check:

```bash
ros2 topic echo /vr_demo_recorder/command
```

If no commands appear, verify:
- joystick is recognized
- `joy_node` is running
- button mapping is correct

---

## B. Converter ran but no episode files were created
Check:
- merged HDF5 exists
- input path is correct
- selected `--cam_preprocess` is correct
- converter did not delete merged file before debugging (`--keep-merged`)

---

## C. Training starts but dataset count is wrong
Both training scripts auto-count `episode_*.hdf5`.
If you want to force a subset:

```bash
python3 train_act.py --num_episodes 20
```

or

```bash
python3 scripts/diffusion/train_diffusion.py --num_episodes 20
```

---

## D. Inference loads checkpoint but output is poor
First check checkpoint match.

If the node prints many missing / unexpected keys, fix config mismatch first.

Typical mismatch sources:
- wrong `policy_class`
- wrong `chunk_size`
- wrong checkpoint root
- wrong camera config
- wrong force-history setting

---

## E. Contact logic is unstable
Check:
- force sign
- force axis order
- force offset / zeroing
- whether the force signal distribution matches the training dataset

---

## F. Camera shake is harming direct-teaching performance
Use:

```bash
python3 demo_data_act_form_single_cam.py \
  --cam_preprocess stabilize_crop \
  --cam_crop_h 384 \
  --cam_crop_w 384 \
  --cam_resize_hw 256
```

and train on:

```bash
python3 train_act.py --cam_preprocess stabilize_crop
```

or

```bash
python3 scripts/diffusion/train_diffusion.py --cam_preprocess stabilize_crop
```

---

# 14. Practical Recommended Workflow

If you want the current recommended practical workflow:

## Raw baseline
```bash
vive
ft              # or ftget
rsv
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=tracker

cd ~/nrs_act/source/custom
python3 demo_data_act_form_single_cam.py --cam_preprocess off

cd ~/nrs_act/scripts/act
python3 train_act.py --cam_preprocess off
```

## Recommended camera-preprocessed workflow
```bash
vive
ft              # or ftget
rsv
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=tracker

cd ~/nrs_act/source/custom
python3 demo_data_act_form_single_cam.py \
  --cam_preprocess stabilize_crop \
  --cam_crop_h 384 \
  --cam_crop_w 384 \
  --cam_resize_hw 256

cd ~/nrs_act/scripts/act
python3 train_act.py --cam_preprocess stabilize_crop
```

## Diffusion camera-preprocessed workflow
```bash
vive
ft              # or ftget
rsv
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=tracker

cd ~/nrs_act/source/custom
python3 demo_data_act_form_single_cam.py \
  --cam_preprocess stabilize_crop \
  --cam_crop_h 384 \
  --cam_crop_w 384 \
  --cam_resize_hw 256

cd ~/nrs_act
python3 scripts/diffusion/train_diffusion.py --cam_preprocess stabilize_crop
```

---


## Robot-side recording workflow

If you want to record robot-side position / force / image into the same merged HDF5 format:

```bash
rsr
ros2 launch nrs_imitation vr_demo_joy_controller.launch.py
ros2 run nrs_imitation vr_demo_hdf5_recorder --ros-args -p recording_mode:=robot

cd ~/nrs_act/source/custom
python3 demo_data_act_form_single_cam.py --cam_preprocess off
```

This creates the same final episode format as tracker mode, but with robot-side topics as the source.

---

# 15. Summary

Current repository status:

- single-camera direct-teaching dataset pipeline
- joystick-controlled recording
- merged HDF5 to episode conversion
- raw and stabilized camera dataset variants
- ACT training branch
- Diffusion Policy-style training branch
- separate checkpoint trees for ACT and Diffusion
- one inference node with `policy_class` switch
- shared dataset / encoder / backbone pipeline

So the current full path is:

```text
teaching (position + force + cam0)
→ merged_hdf5
→ episode_*.hdf5
→ ACT or Diffusion training
→ policy_class-selected inference
```
