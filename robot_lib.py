# -*- coding: utf-8 -*-
"""公共函数库（Plan B 用）——适配 LeRobot main 分支（lerobot.robots 新 API）"""
import os
import glob
import time
import numpy as np

URDF_PATH = "/home/x/robot_learning/pick_place/urdf_so101/so101_new_calib.urdf"

# 6 个电机顺序（和机械臂一致，夹爪是第 6 个，范围 0~100）
ALL_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# 5 个臂关节（运动学只用这 5 个）
ARM_MOTOR_NAMES = ALL_MOTOR_NAMES[:5]


def find_urdf():
    roots = ["/home/x/robot_learning/lerobot",
             "/home/x/robot_learning/conda/lerobot/lib/python3.12/site-packages"]
    for root in roots:
        if not os.path.exists(root):
            continue
        for f in sorted(glob.glob(os.path.join(root, "**", "*.urdf"), recursive=True)):
            print(f)


def make_robot():
    """用命令行参数创建机械臂并连接（LeRobot main 分支新 API）。
    用法：python xxx.py --robot.type=so101_follower --robot.port=/dev/ttyACM0
    """
    from dataclasses import dataclass
    from lerobot.configs import parser
    from lerobot.robots import RobotConfig, make_robot_from_config
    import lerobot.robots.so_follower  # 关键：先导入 so_follower 注册 so101/so100 类型，否则 --robot.type 没得选

    @dataclass
    class _RobotOnlyConfig:
        robot: RobotConfig

    @parser.wrap()
    def _build(cfg: _RobotOnlyConfig):
        return make_robot_from_config(cfg.robot)

    robot = _build()
    robot.connect()          # main 分支要自己连接
    # main 分支的 Robot 没有 motor_names 属性，补一个，旧脚本就能直接用
    if not hasattr(robot, "motor_names"):
        robot.motor_names = list(ALL_MOTOR_NAMES)
    return robot


def get_q(robot):
    """读取 6 个电机当前值（5臂=度，夹爪=0~100），返回 numpy 数组"""
    obs = robot.get_observation()
    return np.array([float(obs[f"{m}.pos"]) for m in ALL_MOTOR_NAMES])


def move_robot(robot, target, duration=0.2):
    """把机械臂移动到目标值。target 是 6 个数（5臂+夹爪）。"""
    target = np.asarray(target, dtype=float)
    action = {f"{m}.pos": float(v) for m, v in zip(ALL_MOTOR_NAMES, target)}
    if hasattr(robot, "send_action"):
        robot.send_action(action)
    else:
        raise RuntimeError("robot 没有 send_action，请把报错发给我")
    time.sleep(duration)


def wait_settled(robot, target, tol_deg=3.0, timeout=8.0):
    """等到机械臂真的到位再返回（电机卡顿时的保险）"""
    target = np.asarray(target, dtype=float)
    t0 = time.time()
    while time.time() - t0 < timeout:
        q = get_q(robot)
        if np.max(np.abs(q[:5] - target[:5])) < tol_deg:
            return q
        time.sleep(0.05)
    print("警告：机械臂没完全到位（可能卡顿），继续下一步")
    return get_q(robot)


def detect_board(BOARD, gray):
    import cv2
    ad = cv2.aruco.ArucoDetector(BOARD.getDictionary())
    marker_corners, marker_ids, _ = ad.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) < 4:
        return None, None, None, None
    return marker_corners, marker_ids, None, None


def board_pose_from_markers(BOARD, marker_corners, marker_ids, K, dist):
    import cv2
    X = int(BOARD.getChessboardSize()[0])
    S = float(BOARD.getSquareLength())
    M = float(BOARD.getMarkerLength())
    h = M / 2.0
    base = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], dtype=float)
    order = [2, 3, 0, 1]
    obj, imgp = [], []
    for corners, mid in zip(marker_corners, marker_ids):
        mid = int(mid[0])
        k = 2 * mid + 1
        j, i = k // X, k % X
        cx, cy = (i + 0.5) * S, (j + 0.5) * S
        for t, idx in enumerate(order):
            obj.append(base[idx] + [cx, cy, 0.0])
            imgp.append(corners.reshape(4, 2)[t])
    ok, rvec, tvec = cv2.solvePnP(np.array(obj, float), np.array(imgp, float), K, dist)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)
    return T


def match_image_points(BOARD, corners, ids):
    import cv2
    res = BOARD.matchImagePoints(corners, ids)
    if isinstance(res, (tuple, list)) and len(res) == 3:
        return res[1], res[2]
    return res


def _rot2quat(R):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_quat()


def _quat2rot(q):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(q).as_matrix()


class PyBulletKin:
    def __init__(self, motor_names=None, mode="DIRECT"):
        import pybullet as p
        if not os.path.exists(URDF_PATH):
            raise FileNotFoundError("找不到 URDF 文件：" + URDF_PATH +
                                    "\n请把 urdf_so101 文件夹放到 ~/robot_learning/pick_place/ 下")
        self.p = p
        info = p.getConnectionInfo()
        connected = info.get("isConnected", 0) if isinstance(info, dict) else 0
        if connected < 1:
            p.connect(p.DIRECT if mode == "DIRECT" else p.GUI)
        self.rid = p.loadURDF(URDF_PATH, useFixedBase=True)
        if motor_names is None:
            motor_names = ARM_MOTOR_NAMES
        arm_names = [m for m in motor_names if "gripper" not in m.lower()]
        joint_name2idx = {}
        for j in range(p.getNumJoints(self.rid)):
            joint_name2idx[p.getJointInfo(self.rid, j)[1].decode()] = j
        self.arm_joints = []
        used = set()
        for mn in arm_names:
            found = None
            for jn, ji in joint_name2idx.items():
                if ji in used:
                    continue
                if mn.lower() == jn.lower() or mn.lower() in jn.lower():
                    found = ji
                    break
            if found is None:
                for ji in joint_name2idx.values():
                    if ji not in used:
                        found = ji
                        break
            self.arm_joints.append(found)
            used.add(found)
        self.arm_names = list(arm_names)
        self.ee_link_name = p.getJointInfo(self.rid, self.arm_joints[-1])[12]
        if isinstance(self.ee_link_name, bytes):
            self.ee_link_name = self.ee_link_name.decode()
        self.ee_link = self.arm_joints[-1]   # pybullet 约定：关节 j 的子 link 索引 = j
        print("手臂关节映射（电机 -> URDF关节索引）:", dict(zip(self.arm_names, self.arm_joints)))
        print("末端 link 名字:", self.ee_link_name)

    def fk(self, q_arm_deg):
        for ji, val in zip(self.arm_joints, q_arm_deg):
            self.p.resetJointState(self.rid, ji, np.radians(float(val)))
        st = self.p.getLinkState(self.rid, self.ee_link)
        T = np.eye(4)
        T[:3, 3] = st[4]          # worldLinkFramePosition（link 坐标系原点，不是质心 st[0]）
        T[:3, :3] = _quat2rot(st[5])  # worldLinkFrameOrientation
        return T

    def ik(self, q_current_arm_deg, T_target, max_err=0.02):
        # 只要求位置（不管朝向）——5 自由度臂这样最稳，夹爪自然朝下
        q_cur = np.asarray(q_current_arm_deg, dtype=float)
        rng = np.random.default_rng(0)
        candidates = [q_cur] + [q_cur + rng.uniform(-20, 20, len(q_cur)) for _ in range(8)]
        target_pos = T_target[:3, 3]
        best_err_sol, best_err = None, 1e9
        best_near, best_near_d = None, 1e9
        for q0 in candidates:
            for ji, val in zip(self.arm_joints, q0):
                self.p.resetJointState(self.rid, ji, np.radians(float(val)))
            q_all = self.p.calculateInverseKinematics(
                self.rid, self.ee_link, target_pos.tolist(),
                maxNumIterations=300, residualThreshold=1e-5)
            if len(q_all) > max(self.arm_joints):
                res = np.degrees(np.array([q_all[j] for j in self.arm_joints]))
            else:
                res = np.degrees(np.array(q_all[:5]))
            err = np.linalg.norm(self.fk(res)[:3, 3] - target_pos)
            if err < best_err:
                best_err, best_err_sol = err, res
            if err <= max_err:
                d = np.linalg.norm(res - q_cur)
                if d < best_near_d:
                    best_near_d, best_near = d, res
        if best_near is not None:
            return best_near
        return best_err_sol if best_err <= max_err else None
