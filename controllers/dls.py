import mujoco
import numpy as np
import os
import matplotlib.pyplot as plt

def run_reach_experiment(lam, steps=150, k=0.1):
    model = mujoco.MjModel.from_xml_path("envs/arm2.xml")
    data = mujoco.MjData(model)

    data.qpos[0] = np.deg2rad(10)
    data.qpos[1] = np.deg2rad(1)

    data.site("target").xpos[:2] = np.array([0.99, 0.0])

    error_history = []
    delta_q_history = []


    site_id = data.site('ee').id

    for step in range(steps):
        mujoco.mj_forward(model, data)

        jacp = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, None, site_id)

        J = jacp[:2, :2]

        x_current = data.site('ee').xpos[:2]
        x_target = data.site('target').xpos[:2]

        e = x_target - x_current
        norm_e = np.linalg.norm(e)
        error_history.append(norm_e)

        M = J @ J.T + lam**2 * np.eye(2)
        inv_term = np.linalg.solve(M, e)

        delta_q = k * (J.T @ inv_term) 
        norm_delta_q = np.linalg.norm(delta_q)
        delta_q_history.append(norm_delta_q)

        data.qpos[:2] += delta_q

    return error_history, delta_q_history

def main():
    err_pinv, dq_pinv = run_reach_experiment(lam=0.0)
    err_dls, dq_dls = run_reach_experiment(lam=0.05)

    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(err_pinv, label="Pseudo-Inverse (λ=0.0)", color="crimson", lw=2)
    ax1.plot(
        err_dls, label="DLS (λ=0.05)", color="navy", linestyle="--", lw=2
    )
    ax1.set_ylabel("||e|| [m]")
    ax1.set_title("Cartesian Reaching Error Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(dq_pinv, label="Pseudo-Inverse (λ=0.0)", color="crimson", lw=2)
    ax2.plot(
        dq_dls, label="DLS (λ=0.05)", color="navy", linestyle="--", lw=2
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("||Δq|| [rad]")
    ax2.set_title("Joint Update Magnitude Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/dls_vs_pinv.png", dpi=300)
    print("Saved plot to results/dls_vs_pinv.png")

if __name__ == "__main__":
    main()