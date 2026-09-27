# Embodied Reach

A from-scratch robot learning project built around a planar 2-link robotic arm.

The goal is to progress from classical robotics and simulation to reinforcement learning and learned robot control, implementing the core algorithms from scratch and comparing their performance.

## Current Status

### Day 1: MuJoCo Environment + Forward Kinematics

* Built a 2-link planar robotic arm in MuJoCo.
* Configured revolute joints, motor actuators, and end-effector (`ee`) and target sites.
* Implemented Product of Exponentials (PoE) forward kinematics.
* Validated the analytical forward kinematics against MuJoCo.

### Day 2: Jacobian + Damped Least Squares Control

* Derived the analytical 2R position Jacobian and its determinant.
* Validated the analytical Jacobian against MuJoCo's `mj_jacSite`.
* Implemented a resolved-rate reaching controller using Damped Least Squares (DLS).
* Compared the pseudo-inverse and DLS controllers near a singular configuration.
* Plotted Cartesian reaching error and joint update magnitude across iterations.

**Result:** Both controllers reached the target. The pseudo-inverse produced a much larger initial joint update, while DLS limited the update through damping.

#### Key Learnings

1. The Jacobian connects joint movement to end-effector movement.
2. A singularity occurs when the robot loses the ability to move independently in one or more directions.
3. Near a singularity, the pseudo-inverse can produce very large joint updates.
4. DLS uses a damping factor to limit these large updates and make movement more stable.
5. The tradeoff is that DLS may not reach the target as precisely near singularities, but it provides more stable joint movements.

#### Experiment

The experiment compares the pseudo-inverse (\(\lambda = 0\)) against DLS (\(\lambda = 0.05\)) using the same initial configuration and target.

![DLS vs Pseudo-Inverse](results/dls_vs_pinv.png)

## Planned Work

* Implement REINFORCE from scratch.
* Implement PPO from scratch.
* Compare learned control against the classical DLS baseline.
* Reproduce selected experiments from robot learning research.
* Explore imitation learning and Action Chunking with Transformers (ACT).
* Integrate learned policies with a robotics stack.

## Project Structure

```text
embodied-reach/
├── envs/
│   └── arm2.xml
├── kinematics/
├── controllers/
├── results/
│   └── dls_vs_pinv.png
├── test_arm.py
└── README.md
```

## Setup

The project uses Python, NumPy, and MuJoCo and runs in a WSL2 Ubuntu environment.

More detailed setup instructions and experiment documentation will be added as the project develops.
