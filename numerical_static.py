import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from motors import get_motors_from_xml

def static_experiments(model, data):

    # Joint configurations in generalized coordinates
    configs = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.57, 0.0, 0.0],
        [3.14, -1.57, 0.0, -1.57, 0.0, 0.0],
        [1.57, -0.78, 0.0, -1.57, -3.14, 0.0],
        [1.57, 1.57, 1.57, 1.57, 1.57, 1.57],
        [-1.57, 0.9, -2.17, -3.9, 3.14, 5.4],
    ]

    all_torques = []

    """ 
    # Test without payload
    for qpos in configs:
        # Set config
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        
        # Get inverse dynamics
        mujoco.mj_inverse(model, data)
        
        all_torques.append(data.qfrc_inverse.copy())
    """

    # Test with payload (point mass)
    link_name = "ee_link"
    link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)

    # Add mass
    payload_mass = 3.0  # kg
    model.body_mass[link_id] += payload_mass

    # Define COM
    current_com = model.body_ipos[link_id].copy()
    payload_pos = np.array([0.0, 0.0, 0.05])
    new_com = (current_com * (model.body_mass[link_id] - payload_mass) + payload_pos * payload_mass) / model.body_mass[link_id]
    model.body_ipos[link_id] = new_com

    # Define inertia
    r2 = np.sum(payload_pos**2)
    model.body_inertia[link_id] += np.array([r2, r2, r2]) * payload_mass    # I = mr**2

    for qpos in configs:
        data.qpos[:] = qpos
        data.qvel[:] = 0
        data.qacc[:] = 0
        mujoco.mj_inverse(model, data)
        all_torques.append(data.qfrc_inverse.copy())

    return np.array(all_torques)

def evaluate_variants(variants_dir):
    xml_files = sorted(glob(os.path.join(variants_dir, "*.xml")))
    results = []

    for xml_file in xml_files:
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)

        all_torques = static_experiments(model, data)
        max_torques = np.max(np.abs(all_torques), axis=0)

        motor_list = get_motors_from_xml(xml_file)
        
        peak_limits = np.array([m.peak_torque for m in motor_list])
        rated_limits = np.array([m.rated_torque for m in motor_list])

        exceeded_peak = max_torques > peak_limits
        exceeded_rated = max_torques > rated_limits

        is_safe_peak = not exceeded_peak.any()
        is_safe_rated = not exceeded_rated.any()

        results.append({
            "file": os.path.basename(xml_file),
            "max_torques": max_torques,
            "peak_limits": peak_limits,
            "rated_limits": rated_limits,
            "exceeded_peak": exceeded_peak,
            "exceeded_rated": exceeded_rated,
            "safe_peak": is_safe_peak,
            "safe_rated": is_safe_rated
        })

        status_list = []
        if not is_safe_peak:
            status_list.append("PEAK_EXCEEDED")
        if not is_safe_rated:
            status_list.append("RATED_EXCEEDED")
        if not status_list:
            status_list.append("SAFE")

        status_str = ", ".join(status_list)
        print(f"{os.path.basename(xml_file)} -> {status_str}")

        for i in range(len(max_torques)):
            if exceeded_peak[i]:
                print(f"  Joint {i}: torque {max_torques[i]:.2f} Nm exceeds PEAK limit {peak_limits[i]:.2f} Nm")
            elif exceeded_rated[i]:
                print(f"  Joint {i}: torque {max_torques[i]:.2f} Nm exceeds RATED limit {rated_limits[i]:.2f} Nm")

    return results

def plot_torques(all_torques):

    max_torques = np.max(np.abs(all_torques), axis=0)
    for i, torque in enumerate(max_torques):
        print(f"Joint {i}: Max torque = {torque:.5f} Nm")

    plt.figure(figsize=(10,6))
    sns.violinplot(data=all_torques)
    plt.ylabel('Torque [Nm]')
    plt.xlabel('Joint index')
    plt.title('Static torques for multiple configurations')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Test single model and draw plot
    """ 
    model_name = 'universalUR3_variants/variant_000.xml'
    model_path = os.path.join(current_dir, model_name)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    all_torques = static_experiments(model, data)
    plot_torques(all_torques)
    """

    # Test all models and print conclusion
    variants_dir_name = "universalUR3_variants"
    variants_dir_path = os.path.join(current_dir, variants_dir_name)
    results = evaluate_variants(variants_dir_path)
