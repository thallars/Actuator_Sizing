import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from motors import get_motors_from_xml

# ------------------ Функции ------------------

def add_payload(model, link_name="ee_link", mass=3.0, payload_pos=np.array([0.0, 0.0, 0.05])):
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    model.body_mass[link_id] += mass
    current_com = model.body_ipos[link_id].copy()
    new_com = (current_com * (model.body_mass[link_id]-mass) + payload_pos*mass)/model.body_mass[link_id]
    model.body_ipos[link_id] = new_com
    r2 = np.sum(payload_pos**2)
    model.body_inertia[link_id] += np.array([r2, r2, r2]) * mass
    return link_id

def scalar_quintic(t, T):
    tau = t / T
    s   = 10*tau**3 - 15*tau**4 + 6*tau**5
    sd  = (30*tau**2 - 60*tau**3 + 30*tau**4)/T
    sdd = (60*tau - 180*tau**2 + 120*tau**3)/(T**2)
    return s, sd, sdd

def cartesian_traj(t, T, p_start, p_goal):
    dp = p_goal - p_start
    s, sd, sdd = scalar_quintic(t, T)
    p = p_start + s*dp
    v = sd*dp
    a = sdd*dp
    return p, v, a

def ik_position(model, data, target_pos, q_init, link_id, tol=1e-4, max_iter=50, step_lim=0.2):
    nq = model.nq
    q = q_init.copy()
    for _ in range(max_iter):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        p_cur = data.xpos[link_id].copy()
        err = target_pos - p_cur
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, link_id)
        J = jacp[:, :nq]
        dq = np.clip(np.linalg.pinv(J) @ err, -step_lim, step_lim)
        q += dq
    return q

def joint_velocity_from_cartesian(model, data, q, v_des, link_id):
    nq = model.nq
    data.qpos[:] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, link_id)
    J = jacp[:, :nq]
    return np.linalg.pinv(J) @ v_des

def compute_ee_trajectory(model, data, q_start, q_goal, dt=0.01, v_max=0.30):
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    def get_ee_pos(q):
        data.qpos[:] = q; data.qvel[:] = 0; data.qacc[:] = 0
        mujoco.mj_forward(model, data)
        return data.xpos[link_id].copy()
    p_start = get_ee_pos(q_start)
    p_goal  = get_ee_pos(q_goal)
    dist = np.linalg.norm(p_goal - p_start)
    T = 15.0 / 8.0 * dist / v_max
    time = np.arange(0.0, T+dt, dt)
    N = len(time)
    nq = model.nq
    q_traj   = np.zeros((N, nq))
    qd_traj  = np.zeros((N, nq))
    qdd_traj = np.zeros((N, nq))
    q_curr = q_start.copy()
    for i, t in enumerate(time):
        p_des, v_des, _ = cartesian_traj(t, T, p_start, p_goal)
        q_curr = ik_position(model, data, p_des, q_curr, link_id)
        q_traj[i] = q_curr
    for i, t in enumerate(time):
        _, v_des, _ = cartesian_traj(t, T, p_start, p_goal)
        qd_traj[i] = joint_velocity_from_cartesian(model, data, q_traj[i], v_des, link_id)
    for j in range(nq):
        qdd_traj[:, j] = np.gradient(qd_traj[:, j], dt, edge_order=2)
    return time, q_traj, qd_traj, qdd_traj, p_start, p_goal, get_ee_pos

def compute_dynamics(model, data, q_traj, qd_traj, qdd_traj):
    N, nq = q_traj.shape
    torques = np.zeros((N, nq))
    for i in range(N):
        data.qpos[:] = q_traj[i]
        data.qvel[:] = qd_traj[i]
        data.qacc[:] = qdd_traj[i]
        mujoco.mj_inverse(model, data)
        torques[i] = data.qfrc_inverse[:nq].copy()
    return torques

def plot_joint_torques(time, torques):
    plt.figure(figsize=(10,6))
    for j in range(torques.shape[1]):
        plt.plot(time, torques[:, j], label=f'Joint {j}')
    plt.xlabel('Time [s]'); plt.ylabel('Torque [Nm]')
    plt.title('Dynamic joint torques for Cartesian trajectory')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_ee_trajectory(get_ee_pos, q_traj, p_start, p_goal):
    N = len(q_traj)
    ee_positions = np.zeros((N, 3))
    for i in range(N):
        ee_positions[i] = get_ee_pos(q_traj[i])
    fig = plt.figure(figsize=(6,6))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.plot(ee_positions[:,0], ee_positions[:,1], ee_positions[:,2], label='EE path')
    ax3d.scatter(p_start[0], p_start[1], p_start[2], marker='o', label='start')
    ax3d.scatter(p_goal[0], p_goal[1], p_goal[2], marker='^', label='goal')
    ax3d.set_xlabel('X [m]'); ax3d.set_ylabel('Y [m]'); ax3d.set_zlabel('Z [m]')
    ax3d.set_title('EE Cartesian trajectory'); ax3d.legend(); plt.tight_layout(); plt.show()

def plot_motor_performance(qd_traj, torques, motor_list):
    torques_abs = np.abs(torques)
    qd_rpm = np.abs(qd_traj)*60.0/(2.0*np.pi)
    fig, axes = plt.subplots(3,2, figsize=(14,12))
    axes = axes.flatten()
    for j in range(torques.shape[1]):
        ax = axes[j]
        ax.scatter(qd_rpm[:, j], torques_abs[:, j], s=10, alpha=0.6, label='Trajectory points')
        w_axis = np.linspace(0.0, motor_list[j].peak_speed, 200)
        ax.fill_between(w_axis, 0.0, motor_list[j].rated_torque, alpha=0.3, hatch='///', facecolor='none', edgecolor='green', label='Nominal region')
        ax.fill_between(w_axis, motor_list[j].rated_torque, motor_list[j].peak_torque, alpha=0.3, hatch='\\\\\\', facecolor='none', edgecolor='red', label='Overload region')
        ax.set_xlabel('Speed [rpm]'); ax.set_ylabel('Torque [Nm]')
        ax.set_title(f"Joint {j} (Motor: {motor_list[j].name})"); ax.grid(True); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

# ------------------ Main ------------------

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = 'universalUR3_variants/variant_016.xml'
    model_path = os.path.join(current_dir, model_name)

    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    add_payload(model, "ee_link", mass=3.0)

    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_goal  = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0])

    time, q_traj, qd_traj, qdd_traj, p_start, p_goal, get_ee_pos = compute_ee_trajectory(model, data, q_start, q_goal)
    torques = compute_dynamics(model, data, q_traj, qd_traj, qdd_traj)

    plot_joint_torques(time, torques)
    plot_ee_trajectory(get_ee_pos, q_traj, p_start, p_goal)

    motor_list = get_motors_from_xml(model_path)
    plot_motor_performance(qd_traj, torques, motor_list)
