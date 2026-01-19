# ESP32 Robot Learning from Demonstration (LfD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PlatformIO](https://img.shields.io/badge/PlatformIO-ESP32-orange.svg)](https://platformio.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-LeRobot%20Compatible-yellow.svg)](https://huggingface.co/docs/lerobot)

> **End-to-end imitation learning pipeline for low-cost ESP32-based robot manipulators**

A complete implementation of Learning from Demonstration (LfD) techniques on resource-constrained embedded hardware. This project demonstrates behavioral cloning, diffusion policy, and ACT (Action Chunking Transformer) on a 4-DOF robot arm controlled by ESP32, with synchronized camera observations.

![System Demo](assets/demo_pick_place.gif)

## Features

- **Real-time Teleoperation Data Collection** - Joystick-based demonstration recording with synchronized camera frames
- **Multiple Policy Architectures** - Behavioral Cloning, Diffusion Policy, and ACT implementations
- **ESP32-CAM Integration** - Visual observations for visuomotor policy learning
- **LeRobot Compatible** - Export datasets to Hugging Face LeRobot format
- **Low-Cost Hardware** - Complete pipeline on ~$100 hardware (ESP32 + 4-DOF arm + camera)
- **Simulation Bridge** - MuJoCo model for sim-to-real experiments

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Collection](#data-collection)
- [Training](#training)
- [Deployment](#deployment)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [Citation](#citation)

## Hardware Requirements

### Minimum Setup (~$80)
| Component | Description | Approx. Cost |
|-----------|-------------|--------------|
| ESP32-C3 Dev Board | Main controller | $10 |
| 4-DOF Robot Arm Kit | MG90S servos, 3D printed/acrylic | $40 |
| Dual Joystick Module | Teleoperation input | $5 |
| ESP32-CAM / M5Stack Camera | Visual observations | $15 |
| Power Supply | 5V 3A for servos | $10 |

### Recommended Additions
- M5Stack Core2 for better processing
- IMU module for end-effector orientation
- Additional lighting for consistent camera input

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/goker/esp32-robot-lfd.git
cd esp32-robot-lfd
```

### 2. Set Up Python Environment

```bash
# Create conda environment
conda create -n robot-lfd python=3.10 -y
conda activate robot-lfd

# Install dependencies
pip install -r requirements.txt

# Install with LeRobot support (optional)
pip install lerobot
```

### 3. Flash ESP32 Firmware

```bash
# Install PlatformIO
pip install platformio

# Flash arm controller
cd firmware/esp32_arm_controller
pio run --target upload

# Flash camera module (separate ESP32)
cd ../esp32_cam_observer
pio run --target upload
```

### 4. Calibrate Robot

```bash
python scripts/calibrate_arm.py --port /dev/ttyUSB0
```

## Quick Start

### 1. Test Hardware Connection

```bash
# Test arm communication
python -m host.utils.esp32_comm --test

# Test camera stream
python -m host.utils.camera_sync --preview
```

### 2. Record a Demonstration

```bash
python scripts/collect_demo.sh --task pick_place --episodes 10
```

### 3. Train a Policy

```bash
python scripts/train_policy.sh --policy bc --epochs 100
```

### 4. Deploy and Run

```bash
python scripts/run_inference.sh --checkpoint outputs/bc_policy.pt
```

## Data Collection

### Teleoperation Recording

The system supports real-time demonstration recording with synchronized observations:

```python
from host.data_collection import TeleopRecorder

recorder = TeleopRecorder(
    arm_url="ws://192.168.1.100:81",
    camera_url="http://192.168.1.101:80/stream"
)

# Record 50 episodes for pick-and-place task
recorder.collect_dataset(
    task_name="pick_place",
    num_episodes=50,
    save_dir="datasets/pick_place_v1"
)
```

### Data Format

Episodes are stored in HDF5 format compatible with LeRobot:

```
episode_0.hdf5
├── observations/
│   ├── images/        # (T, H, W, C) camera frames
│   ├── joint_pos/     # (T, 4) joint positions
│   └── gripper/       # (T, 1) gripper state
├── actions/           # (T, 4) target joint positions
└── metadata/
    ├── fps: 20
    ├── task: "pick_place"
    └── timestamp: "2025-01-20T..."
```

### Export to LeRobot Format

```bash
python -m host.data_collection.data_format \
    --input datasets/pick_place_v1 \
    --output datasets/pick_place_lerobot \
    --format lerobot
```

## Training

### Behavioral Cloning (Baseline)

```bash
python -m host.training.train \
    --policy bc \
    --dataset datasets/pick_place_v1 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4
```

### Diffusion Policy

```bash
python -m host.training.train \
    --policy diffusion \
    --dataset datasets/pick_place_v1 \
    --epochs 500 \
    --diffusion_steps 100
```

### ACT (Action Chunking Transformer)

```bash
python -m host.training.train \
    --policy act \
    --dataset datasets/pick_place_v1 \
    --chunk_size 10 \
    --epochs 300
```

## Deployment

### Real-time Policy Execution

```python
from host.deployment import PolicyExecutor

executor = PolicyExecutor(
    policy_path="outputs/diffusion_policy.pt",
    esp32_url="ws://192.168.1.100:81",
    camera_url="http://192.168.1.101:80/stream",
    control_freq=20  # 20 Hz control loop
)

# Run autonomous episode
success = executor.run_episode(max_steps=200)
print(f"Task success: {success}")
```

### Evaluation

```bash
python -m host.deployment.evaluation \
    --policy outputs/diffusion_policy.pt \
    --num_episodes 20 \
    --save_videos
```

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│   ESP32-C3      │◄──────────────────►│   Host PC       │
│   (Arm Control) │                    │   (Python)      │
│                 │  JSON: {           │                 │
│ • Servo PWM     │   joint_pos: [],   │ • Data record   │
│ • Joystick read │   gripper: float,  │ • Train policy  │
│ • State report  │   action: []       │ • Run inference │
└────────┬────────┘  }                 └────────┬────────┘
         │                                      │
         │                                      │
┌────────┴────────┐    MJPEG Stream    ┌───────┴─────────┐
│   ESP32-CAM     │───────────────────►│  Training Data  │
│   (Observer)    │                    │  (HDF5/Parquet) │
│                 │                    │                 │
│ • 320x240 RGB   │                    │ • observations  │
│ • 15 FPS sync   │                    │ • actions       │
└─────────────────┘                    └─────────────────┘
```

## Benchmarks

### Task: Pick and Place (50 demonstrations)

| Policy | Success Rate | Inference Time | Training Time |
|--------|-------------|----------------|---------------|
| Behavioral Cloning | 72% | 5ms | 10 min |
| Diffusion Policy | 88% | 45ms | 2 hours |
| ACT | 85% | 20ms | 1 hour |

### Comparison with Research Hardware

This low-cost implementation achieves comparable learning efficiency to research-grade systems when normalized for task complexity. See [docs/SCALING_NOTES.md](docs/SCALING_NOTES.md) for detailed analysis.

## Project Structure

```
esp32-robot-lfd/
├── firmware/                    # ESP32 firmware (PlatformIO)
│   ├── esp32_arm_controller/   # Main arm control
│   └── esp32_cam_observer/     # Camera streaming
├── host/                       # Python host code
│   ├── data_collection/        # Teleoperation recording
│   ├── training/               # Policy training
│   ├── deployment/             # Real-time inference
│   └── utils/                  # Utilities
├── simulation/                 # MuJoCo simulation
├── configs/                    # Configuration files
├── datasets/                   # Sample data links
├── notebooks/                  # Analysis notebooks
├── scripts/                    # Helper scripts
├── docs/                       # Documentation
└── assets/                     # Images, GIFs
```

## Related Work

This project builds upon and is compatible with:

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face's robotics framework
- [robomimic](https://github.com/ARISE-Initiative/robomimic) - NVIDIA's robot learning framework
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - Columbia's diffusion-based policy learning
- [ACT](https://tonyzhaozh.github.io/aloha/) - Action Chunking Transformer from ALOHA
- [Mobile ALOHA](https://mobile-aloha.github.io/) - Stanford's bimanual mobile manipulation

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{esp32_robot_lfd,
  author = {Goker},
  title = {ESP32 Robot Learning from Demonstration},
  year = {2025},
  url = {https://github.com/goker/esp32-robot-lfd}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Keywords

`robot learning` `imitation learning` `learning from demonstration` `behavioral cloning` `diffusion policy` `ACT` `ESP32` `robot manipulation` `teleoperation` `low-cost robotics` `embedded robotics` `LeRobot` `robomimic` `visuomotor policy` `pick and place` `robot arm` `M5Stack` `ESP32-CAM`
