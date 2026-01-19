# ESP32 Robot Learning from Demonstration - Project Tracker

## Overview

**Repository:** [github.com/goker/esp32-robot-lfd](https://github.com/goker/esp32-robot-lfd)

**Objective:** Build a complete Learning from Demonstration (LfD) pipeline on low-cost ESP32 hardware to demonstrate proficiency in:
1. Modern learning-from-demonstration tools (LeRobot, Diffusion Policy, ACT)
2. Robot data collection, training, and hardware deployment for manipulation tasks

**Target Skills for Resume:**
- Experience with modern learning-from-demonstration tools
- Experience with robot data collection, training, and testing on hardware for manipulation tasks

---

## Goals

| Goal | Status | Target Date |
|------|--------|-------------|
| Complete ESP32 firmware for arm control | 🟡 In Progress | |
| Implement camera streaming module | 🟡 In Progress | |
| Build teleoperation data collection system | ⬜ Not Started | |
| Collect 50+ demonstration episodes | ⬜ Not Started | |
| Train Behavioral Cloning policy | ⬜ Not Started | |
| Implement Diffusion Policy | ⬜ Not Started | |
| Achieve 80%+ success rate on pick-and-place | ⬜ Not Started | |
| Create demo videos for portfolio | ⬜ Not Started | |
| Update resume with new skills | ⬜ Not Started | |

---

## Learning Roadmap

### Phase 1: Hardware Setup (Week 1-2)
- [ ] Assemble 4-DOF robot arm
- [ ] Flash ESP32-C3 with arm controller firmware
- [ ] Configure WiFi and WebSocket communication
- [ ] Set up M5Stack camera module
- [ ] Calibrate servo limits and home positions
- [ ] Test joystick teleoperation

### Phase 2: Data Collection Pipeline (Week 2-3)
- [ ] Install Python environment and dependencies
- [ ] Test WebSocket communication from host PC
- [ ] Test camera streaming and frame capture
- [ ] Record first test episodes
- [ ] Verify HDF5 data format
- [ ] Collect 10 practice demonstrations
- [ ] Debug synchronization issues

### Phase 3: Policy Training (Week 3-4)
- [ ] Train Behavioral Cloning baseline
- [ ] Analyze training curves and loss
- [ ] Implement data augmentation
- [ ] Train Diffusion Policy variant
- [ ] Compare policy performance
- [ ] Tune hyperparameters

### Phase 4: Deployment & Evaluation (Week 4-5)
- [ ] Deploy BC policy on real robot
- [ ] Measure success rate (target: 70%+)
- [ ] Deploy Diffusion Policy
- [ ] Measure improved success rate (target: 85%+)
- [ ] Record evaluation videos
- [ ] Document results

### Phase 5: Portfolio & Resume (Week 5-6)
- [ ] Create demo GIF for README
- [ ] Write detailed documentation
- [ ] Record YouTube demo video (optional)
- [ ] Update resume with achievements
- [ ] Prepare talking points for interviews

---

## Tasks

### Current Sprint

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Configure WiFi credentials in firmware | High | ⬜ | Edit config.h |
| Flash ESP32-C3 with PlatformIO | High | ⬜ | |
| Flash ESP32-CAM module | High | ⬜ | |
| Test arm servo movement | High | ⬜ | |
| Test joystick input | Medium | ⬜ | |

### Backlog

- [ ] Add IMU module for end-effector tracking
- [ ] Implement ACT (Action Chunking Transformer)
- [ ] Add simulation environment in MuJoCo
- [ ] Upload dataset to Hugging Face Hub
- [ ] Write blog post about the project

---

## Resources

### Documentation
- [LeRobot Docs](https://huggingface.co/docs/lerobot)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [Mobile ALOHA](https://mobile-aloha.github.io/)
- [robomimic](https://github.com/ARISE-Initiative/robomimic)

### Hardware
- ESP32-C3 DevKit
- 4-DOF Robot Arm (MG90S servos)
- M5Stack Camera Module
- Dual Joystick Module

### Code References
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [trzy/robot-arm](https://github.com/trzy/robot-arm) - iPhone teleoperation reference
- [AhaRobot](https://arxiv.org/html/2503.10070v1) - Low-cost ESP32 teleoperation

---

## Progress Log

### January 2025

**Week 1**
- Created GitHub repository: `esp32-robot-lfd`
- Implemented ESP32 arm controller firmware
- Implemented ESP32-CAM streaming module
- Created Python data collection pipeline
- Created Behavioral Cloning training code
- Created real-time deployment code

---

## Metrics & Results

| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| Demonstrations collected | 50 | 0 | |
| BC Success Rate | 70% | - | |
| Diffusion Policy Success Rate | 85% | - | |
| Inference latency | <50ms | - | |
| Control frequency | 20Hz | - | |

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Servo precision issues | Medium | High | Add position feedback, tune PID |
| WiFi latency spikes | High | Medium | Use wired USB fallback |
| Limited DOF (4 vs 6) | Medium | Certain | Focus on simpler tasks |
| Camera frame drops | Medium | Medium | Add frame buffering |

---

## Resume Bullet Points (Draft)

```
Robotics & Imitation Learning Engineer | Personal Research

• Designed and implemented end-to-end learning-from-demonstration pipeline
  on ESP32-based robot manipulator, demonstrating core LfD concepts on
  resource-constrained embedded hardware

• Built real-time teleoperation data collection system with synchronized
  camera observations using ESP32-CAM, WebSocket communication, and
  custom HDF5 recording

• Trained behavioral cloning and diffusion policy models achieving 85%+
  success rate on pick-and-place tasks with only 50 demonstrations

• Open-sourced complete codebase with documentation, enabling reproducible
  robot learning research on low-cost hardware

Skills: LeRobot, Diffusion Policy, ACT, Behavioral Cloning, ESP32,
        PyTorch, Robot Manipulation, Teleoperation, Embedded Systems
```

---

## Links

- **GitHub:** https://github.com/goker/esp32-robot-lfd
- **LeRobot:** https://github.com/huggingface/lerobot
- **Diffusion Policy:** https://diffusion-policy.cs.columbia.edu/
