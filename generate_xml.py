import os
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from itertools import product

import numpy as np
from lxml import etree

from motors import Motor, joint_motors, JOINT_GROUPS


# XML helpers
def load_xml(path: str) -> etree._ElementTree:
    return etree.parse(path)

def save_xml(tree: etree._ElementTree, path: str) -> None:
    tree.write(path, pretty_print=True, xml_declaration=True, encoding="utf-8")

def get_joints(tree: etree._ElementTree):
    return tree.getroot().findall(".//joint")

def map_joint_to_body(tree: etree._ElementTree):
    res = {}
    for joint in get_joints(tree):
        name = joint.get("name")
        if not name:
            continue
        res[name] = (joint, joint.getparent())
    return res

def get_or_create_actuator_root(root: etree._Element) -> etree._Element:
    act = root.find("actuator")
    if act is None:
        act = etree.SubElement(root, "actuator")
    return act

def fix_meshdir(tree: etree._ElementTree):
    compiler_el = tree.getroot().find("compiler")
    if compiler_el is not None:
        meshdir = compiler_el.get("meshdir", "")
        if meshdir.startswith("meshes/visual"):
            compiler_el.set("meshdir", "../meshes/visual/")


# Inertia helpers
def parse_inertial(el: etree._Element):
    m = float(el.get("mass"))
    pos = np.fromstring(el.get("pos", "0 0 0"), sep=" ")
    I_diag = np.fromstring(el.get("diaginertia"), sep=" ")
    return m, pos, I_diag

def cylinder_inertia_diag(m: float, R: float, h: float) -> np.ndarray:
    i_xy = (1/12) * m * (3 * R**2 + h**2)
    i_z = 0.5 * m * R**2
    return np.array([i_xy, i_xy, i_z])

# Old motors (assumption)
OLD_MOTORS = {          # mass  radius  lenght
    "shoulder_pan_joint":  (1.0, 0.04, 0.08),
    "shoulder_lift_joint": (1.0, 0.04, 0.08),
    "elbow_joint":         (0.6, 0.035, 0.07),
    "wrist_1_joint":       (0.3, 0.025, 0.05),
    "wrist_2_joint":       (0.3, 0.025, 0.05),
    "wrist_3_joint":       (0.15, 0.020, 0.04),
}

# Model update
def update_model_with_motor(tree: etree._ElementTree, joint_name: str, motor: Motor):
    joint_map = map_joint_to_body(tree)
    if joint_name not in joint_map:
        raise ValueError(f"Joint '{joint_name}' not found in XML")
    joint_el, body_el = joint_map[joint_name]

    inertial_el = body_el.find("inertial")
    if inertial_el is None:
        raise RuntimeError(f"No <inertial> in body '{body_el.get('name')}'")

    # remove old motor contribution
    M_link, pos_link, I_link = parse_inertial(inertial_el)
    if joint_name in OLD_MOTORS:
        old_mass, old_R, old_h = OLD_MOTORS[joint_name]
        I_old = cylinder_inertia_diag(old_mass, old_R, old_h)
        M_link -= old_mass
        I_link -= I_old
        assert M_link > 0.0, f"Cleaned mass for {joint_name} must be positive"

    # add new motor contribution
    joint_el.set("armature", f"{motor.armature_inertia:.6e}")
    I_motor = cylinder_inertia_diag(motor.mass, motor.radius, motor.length)
    M_link += motor.mass
    I_link += I_motor

    # update inertial element
    inertial_el.set("mass", f"{M_link:.6e}")
    inertial_el.set("pos", " ".join(f"{v:.6e}" for v in pos_link))
    inertial_el.set("diaginertia", " ".join(f"{v:.6e}" for v in I_link))

    # actuator
    actuator_root = get_or_create_actuator_root(tree.getroot())
    motor_el = etree.SubElement(actuator_root, "general")
    motor_el.set("name", f"{joint_name}_{motor.name}")
    motor_el.set("joint", joint_name)
    motor_el.set("gear", f"{motor.gear_ratio:.6f}")
    motor_el.set("ctrlrange", f"{-motor.peak_torque:.3f} {motor.peak_torque:.3f}")


# Build variants
JOINT_GROUPS = {
    "shoulder_joint": ["shoulder_pan_joint", "shoulder_lift_joint"],
    "elbow_joint": ["elbow_joint"],
    "wrist_joint": ["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
}

def build_single_variant(base_xml_path: str, output_path: str, motor_indices: dict = None):
    tree = load_xml(base_xml_path)

    for group_name, motors in joint_motors.items():
        idx = 0
        if motor_indices and group_name in motor_indices:
            idx = motor_indices[group_name]
        motor = motors[idx]

        if group_name not in JOINT_GROUPS:
            raise ValueError(f"Unknown joint group: {group_name}")
        
        joints = JOINT_GROUPS[group_name]
        for joint_name in joints:
            update_model_with_motor(tree, joint_name, motor)

    fix_meshdir(tree)
    save_xml(tree, output_path)
    print(f"Saved: {output_path}")


def generate_all_variants(base_xml_path: str, output_dir: str):
    base_tree = load_xml(base_xml_path)
    joint_groups = list(joint_motors.keys())
    variants_lists = [joint_motors[g] for g in joint_groups]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, combo in enumerate(product(*variants_lists)):
        tree = deepcopy(base_tree)
        name_parts = []

        for group_name, motor in zip(joint_groups, combo):
            if group_name not in JOINT_GROUPS:
                raise ValueError(f"Unknown joint group: {group_name}")
            
            joints = JOINT_GROUPS[group_name]
            for joint_name in joints:
                update_model_with_motor(tree, joint_name, motor)
            name_parts.append(f"{group_name}-{motor.name}")

        out_name = f"variant_{idx:03d}.xml"
        out_path = out_dir / out_name

        fix_meshdir(tree)
        save_xml(tree, str(out_path))
        print(f"Saved: {out_path}  ({'__'.join(name_parts)})")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_name = 'universalUR3.xml'
    out_xml_name = "universalUR3_variant.xml"
    output_dir_name = "universalUR3_variants"

    model_path = os.path.join(current_dir, model_name)
    out_xml_path = os.path.join(current_dir, out_xml_name)
    output_dir_path = os.path.join(current_dir, output_dir_name)

    #build_single_variant(model_path, out_xml_path)
    generate_all_variants(model_path, output_dir_path)
