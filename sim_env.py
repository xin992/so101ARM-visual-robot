# -*- coding: utf-8 -*-
"""PyBullet IK自动到达目标点位，适配robot_lib.py，SO101"""
import time
import numpy as np
import pybullet as p
import pybullet_data
from robot_lib import PyBulletKin

kin = PyBulletKin(mode="GUI")
p = kin.p
rid = kin.rid

p.setAdditionalSearchPath(pybullet_data.getDataPath())

arm_joints = kin.arm_joints
arm_names = kin.arm_names

q_deg = np.array([0.0, 45.0, -60.0, 0.0, 0.0])
step_deg = 2.0

# 夹爪 0=闭合，100=张开
gripper_val = 0.0
gripper_step = 5.0

# 打印全部关节，识别夹爪双关节
print("\n====全部URDF关节列表====")
gripper_joint_list = []
for j in range(p.getNumJoints(rid)):
    jname = p.getJointInfo(rid, j)[1].decode()
    print(f"joint {j}: {jname}")
    if "gripper" in jname.lower():
        gripper_joint_list.append(j)
print(f"\n找到夹爪关节索引: {gripper_joint_list}")

print("=====键盘控制说明=====")
print("1/2 ： shoulder_pan 关节0 左右旋转")
print("3/4 ： shoulder_lift 关节1 肩抬升")
print("5/6 ： elbow_flex    关节2 手肘")
print("7/8 ： wrist_flex    关节3 手腕俯仰")
print("9/0 ： wrist_roll    关节4 手腕旋转")
print("z/x ： gripper 夹爪开合(z闭合，x张开)")
print("q：打印当前关节角+末端FK位置")
print("s：输入目标XYZ，IK自动移动机械臂")
print("r：重置回到初始姿态")
print("关闭窗口右上角 × 退出程序")
print("======================")

# 用于标记：是否需要执行IK
run_ik_flag = False
target_xyz = None

while True:
    keys = p.getKeyboardEvents()

    # ==========手动手臂控制==========
    if ord('1') in keys and keys[ord('1')] & p.KEY_WAS_TRIGGERED:
        q_deg[0] += step_deg
    if ord('2') in keys and keys[ord('2')] & p.KEY_WAS_TRIGGERED:
        q_deg[0] -= step_deg

    if ord('3') in keys and keys[ord('3')] & p.KEY_WAS_TRIGGERED:
        q_deg[1] += step_deg
    if ord('4') in keys and keys[ord('4')] & p.KEY_WAS_TRIGGERED:
        q_deg[1] -= step_deg

    if ord('5') in keys and keys[ord('5')] & p.KEY_WAS_TRIGGERED:
        q_deg[2] += step_deg
    if ord('6') in keys and keys[ord('6')] & p.KEY_WAS_TRIGGERED:
        q_deg[2] -= step_deg

    if ord('7') in keys and keys[ord('7')] & p.KEY_WAS_TRIGGERED:
        q_deg[3] += step_deg
    if ord('8') in keys and keys[ord('8')] & p.KEY_WAS_TRIGGERED:
        q_deg[3] -= step_deg

    if ord('9') in keys and keys[ord('9')] & p.KEY_WAS_TRIGGERED:
        q_deg[4] += step_deg
    if ord('0') in keys and keys[ord('0')] & p.KEY_WAS_TRIGGERED:
        q_deg[4] -= step_deg

    # ==========夹爪控制==========
    if ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
        gripper_val = max(0.0, gripper_val - gripper_step)
    if ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
        gripper_val = min(100.0, gripper_val + gripper_step)

    # ==========触发IK：按s键==========
    if ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
        print("\n请输入目标XYZ，空格分隔，例如：0.4 0.0 0.2")
        try:
            in_str = input("目标x y z: ")
            x, y, z = map(float, in_str.strip().split())
            target_xyz = np.array([x, y, z])
            run_ik_flag = True
        except Exception as e:
            print(f"输入错误：{e}")

    # ==========重置==========
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        q_deg = np.array([0.0, 45.0, -60.0, 0.0, 0.0])
        gripper_val = 0.0
        run_ik_flag = False

    # ==========打印状态==========
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        T = kin.fk(q_deg)
        print(f"\n当前手臂关节(°): {np.round(q_deg,1)}")
        print(f"夹爪值: {gripper_val:.1f}  (0闭合，100张开)")
        print(f"末端位置XYZ: {np.round(T[:3,3],3)}")

    # ==========执行IK逻辑==========
    if run_ik_flag and target_xyz is not None:
        run_ik_flag = False
        # 构造目标变换矩阵：位置为输入XYZ，姿态保持当前末端姿态
        T_target = kin.fk(q_deg).copy()
        T_target[:3, 3] = target_xyz

        print(f"\n===IK求解目标：{target_xyz} ===")
        q_sol = kin.ik(q_deg, T_target)

        if q_sol is None:
            print("❌ IK求解失败：点位不可达！")
        else:
            print(f"✅ 求解成功，得到关节角度：{np.round(q_sol,1)}")
            q_deg = q_sol

    # 更新手臂姿态
    kin.fk(q_deg)

    # 更新夹爪双关节
    if len(gripper_joint_list) == 2:
        j1, j2 = gripper_joint_list
        grip_rad = np.deg2rad(gripper_val * 0.75)
        p.resetJointState(rid, j1, grip_rad)
        p.resetJointState(rid, j2, -grip_rad)

    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()

