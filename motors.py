from dataclasses import dataclass
import xml.etree.ElementTree as ET

@dataclass
class Motor:
    name: str
    gear_ratio: float           # N
    armature_inertia: float     # kg·m^2
    mass: float                 # kg
    radius: float               # m
    length: float               # m
    peak_torque: float          # N·m
    rated_torque: float         # N·m
    peak_speed: float           # RPM
    rated_speed: float          # RPM

gear_ratio = 101.0

# Base and shoulder
CRA_RI60_80_PRO_101 = Motor(
    name="CRA-RI60-80-PRO-101",
    gear_ratio=gear_ratio,
    armature_inertia=441.0 * 1e-7 * gear_ratio**2,
    mass=1.01,
    radius=80.0 / 2.0 * 1e-3,
    length=70.4 * 1e-3,
    peak_torque=66.0,
    rated_torque=30.0,
    peak_speed=41.0,
    rated_speed=34.0
)

CRA_RI60_70_PRO_S_101 = Motor(
    name="CRA-RI60-70-PRO-S-101",
    gear_ratio=gear_ratio,
    armature_inertia=382.0 * 1e-7 * gear_ratio**2,
    mass=0.69,
    radius=70.0 / 2.0 * 1e-3,
    length=70.7 * 1e-3,
    peak_torque=66.0,
    rated_torque=30.0,
    peak_speed=41.0,
    rated_speed=34.0
)

CRA_RI70_90_NH_101 = Motor(
    name="CRA-RI70-90-NH-101",
    gear_ratio=gear_ratio,
    armature_inertia=594.0 * 1e-7 * gear_ratio**2,
    mass=1.3,
    radius=90.0 / 2.0 * 1e-3,
    length=71.9 * 1e-3,
    peak_torque=80.0,
    rated_torque=40.0,
    peak_speed=37.0,
    rated_speed=30.0
)

# Elbow
CRA_RI50_70_PRO_101 = Motor(
    name="CRA-RI50-70-PRO-101",
    gear_ratio=gear_ratio,
    armature_inertia=124.0 * 1e-7 * gear_ratio**2,
    mass=0.63,
    radius=70.0 / 2.0 * 1e-3,
    length=60.0 * 1e-3,
    peak_torque=34.0,
    rated_torque=9.6,
    peak_speed=49.0,
    rated_speed=37.0
)

CRA_RI50_60_PRO_S_101 = Motor(
    name="CRA-RI50-60-PRO-S-101",
    gear_ratio=gear_ratio,
    armature_inertia=112.0 * 1e-7 * gear_ratio**2,
    mass=0.42,
    radius=60.0 / 2.0 * 1e-3,
    length=60.0 * 1e-3,
    peak_torque=34.0,
    rated_torque=9.6,
    peak_speed=49.0,
    rated_speed=37.0
)

CRA_RI60_80_NH_101 = Motor(
    name="CRA-RI60-80-NH-101",
    gear_ratio=gear_ratio,
    armature_inertia=213.0 * 1e-7 * gear_ratio**2,
    mass=0.95,
    radius=80.0 / 2.0 * 1e-3,
    length=70.4 * 1e-3,
    peak_torque=48.0,
    rated_torque=35.0,
    peak_speed=34.0,
    rated_speed=20.0
)

# Wrist
CRA_RI40_52_PRO_101 = Motor(
    name="CRA-RI40-52-PRO-101",
    gear_ratio=gear_ratio,
    armature_inertia=60.3 * 1e-7 * gear_ratio**2,
    mass=0.3,
    radius=52.0 / 2.0 * 1e-3,
    length=70.4 * 1e-3,
    peak_torque=11.0,
    rated_torque=6.5,
    peak_speed=59.0,
    rated_speed=40.0
)

CRA_RI40_52_NH_101 = Motor(
    name="CRA-RI40-52-NH-101",
    gear_ratio=gear_ratio,
    armature_inertia=60.3 * 1e-7 * gear_ratio**2,
    mass=0.31,
    radius=52.0 / 2.0 * 1e-3,
    length=70.4 * 1e-3,
    peak_torque=9.0,
    rated_torque=6.0,
    peak_speed=59.0,
    rated_speed=40.0
)

joint_motors = {
    "shoulder_joint":   [CRA_RI60_80_PRO_101, CRA_RI60_70_PRO_S_101, CRA_RI70_90_NH_101],
    "elbow_joint":      [CRA_RI50_70_PRO_101, CRA_RI50_60_PRO_S_101, CRA_RI60_80_NH_101],
    "wrist_joint":      [CRA_RI40_52_PRO_101, CRA_RI40_52_NH_101],
}

JOINT_GROUPS = {
    "shoulder_joint":   ["shoulder_pan_joint", "shoulder_lift_joint"],
    "elbow_joint":      ["elbow_joint"],
    "wrist_joint":      ["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
}

MOTOR_BY_NAME = {}
for group, motors in joint_motors.items():
    for motor in motors:
        MOTOR_BY_NAME[motor.name] = motor

def get_motors_from_xml(xml_path):
    tree = ET.parse(xml_path)
    actuators = tree.findall(".//general")

    motor_list = []
    for act in actuators:
        name = act.get("name")  # пример: "shoulder_pan_joint_CRA-RI60-80-PRO-101"
        # извлекаем имя мотора из name
        motor_name = name.split("_")[-1]  # берём последнюю часть после "_"
        motor = MOTOR_BY_NAME[motor_name]
        motor_list.append(motor)

    return motor_list
