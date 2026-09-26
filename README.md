# Embodied Reach

A from-scratch robot learning project built around a planar 2-link robotic arm.

The goal is to progressively move from classical robotics and simulation to reinforcement learning and learned robot control.

## Current Status

**Day 1: MuJoCo environment + Forward Kinematics validation**

* Built a 2-link planar arm in MuJoCo
* Implemented revolute joints and motor actuators
* Added end-effector (`ee`) and target sites
* Verified MuJoCo forward kinematics against my Product of Exponentials (PoE) implementation

## Planned Work

* Classical Differential IK / Damped Least Squares controller
* REINFORCE from scratch
* PPO from scratch
* Compare learned control against classical control
* Reproduce selected experiments from robot learning research
* Explore imitation learning and ACT
* Integrate the learned policy with a robotics stack

## Project Structure

```text
embodied-reach/
├── envs/
│   └── arm2.xml
├── test_arm.py
└── README.md
```

## Setup

The project uses Python with MuJoCo and runs in a WSL2 Ubuntu environment.

More detailed setup and experiment documentation will be added as the project develops.
