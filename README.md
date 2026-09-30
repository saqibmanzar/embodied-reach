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

### Day 3: Gymnasium Reaching Environment + MDP Formulation

* Built a custom Gymnasium reaching environment around the MuJoCo 2-link arm.
* Defined a 6-dimensional physical task state consisting of joint positions, joint velocities, and target position.
* Defined a 10-dimensional observation containing joint angles represented using sine/cosine, joint velocities, target position, and end-effector-to-target displacement.
* Defined continuous 2-dimensional motor actions in the range \([-1,1]\).
* Implemented a distance-based reward with a small control penalty.
* Added random target sampling and a 200-step episode horizon.
* Validated the environment using Gymnasium's environment checker.
* Ran a 100-episode random-policy baseline.
* Implemented discounted return-to-go analysis for \(\gamma=0.9\) and \(\gamma=0.99\).

#### MDP Formulation

| Component       | Definition                                                                        |
| --------------- | --------------------------------------------------------------------------------- |
| **State**       | \(s=(q_1,q_2,\dot q_1,\dot q_2,x_t,y_t)\)                                         |
| **Observation** | `[cos q1, cos q2, sin q1, sin q2, qdot1, qdot2, target_x, target_y, dx, dy]`      |
| **Action**      | Continuous \(a_t\in[-1,1]^2\) applied to the two motors                           |
| **Transition**  | MuJoCo physics, with 5 simulation steps per environment step                      |
| **Reward**      | \(-\|x_{ee}-x_{target}\|_2 - 0.01\|a_t\|_2^2\)                                    |
| **Termination** | No task termination; reaching the target is recorded through `info["is_success"]` |
| **Truncation**  | Episode ends after 200 environment steps                                          |
| **Discount**    | \(\gamma\in\{0.9,0.99\}\)                                                         |

Under the simplified task definition, the physical state is Markov because the next state depends on the current state and action rather than the previous history.

#### Random Policy Baseline

100 episodes with randomly sampled actions:

* **Success rate:** 1.0%
* **Mean return:** -173.66 ± 79.49

The random policy provides a baseline for evaluating the learned policies developed in later experiments.

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
│   ├── arm2.xml
│   └── reach_env.py
├── kinematics/
├── controllers/
├── tests/
│   └── test_env.py
├── notebooks/
│   └── returns.ipynb
├── results/
│   └── dls_vs_pinv.png
├── test_arm.py
└── README.md
```

## Setup

The project uses Python, NumPy, Gymnasium, and MuJoCo and runs in a WSL2 Ubuntu environment.

More detailed setup instructions and experiment documentation will be added as the project develops.
