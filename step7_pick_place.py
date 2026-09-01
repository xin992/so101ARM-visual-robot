# -*- coding: utf-8 -*-
"""第 6 步：主程序——抓取方块放进框（Plan B）
用法：
  1) 先跑"只检测不动臂"模式（不需要 robot 参数）：
       python step7_pick_place.py --dry
  2) 检测没问题了，再跑真机抓取（带 robot 参数）：
       python step7_pick_place.py --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
         --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
"""
import os
import sys
import json
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from ultralytics import YOLO
from robot_lib import make_robot, get_q, move_robot, wait_settled, PyBulletKin
# ================== 你要改的配置 ==================
CUBE_H = 0.030     # 方块高度（米），按实际量
PRE_LIFT = 0.05    # 预抓/预放高度（米），离桌面 10cm
STEP_DUR = 0.35     # 每小步移动时长（秒），电机卡顿就调大（0.3~0.5）
TOL_DEG = 3.0      # 到位允许误差（度）
CONF = 0.5         # YOLO 置信度阈值
GRIPPER_CLOSE = 18.0
GRIPPER_OPEN = 43.0
LIFT_H = 0.11        # 抓住后抬升高度(米)，能越过框边；撞框就调大
PLACE_LIFT_H = 0.07  # 放完方块之后抬升高度

# ★ 抓取偏移随距离变化（近=偏后，远=偏前）——两点校准：
DIST_NEAR = 0.20        # 近点：物块离底座约 20cm
OFFSET_NEAR = -0.02      # 近点夹爪偏后多少米（偏后填正，偏前填负）
DIST_FAR = 0.30         # 远点：物块离底座约 30cm
OFFSET_FAR = -0.037      # 远点夹爪偏前多少米（偏后填正，偏前填负）
CUBE_OFFSET_Y = 0.007   # 中间位置的左右偏移
Y_OFFSET_SLOPE = -0.30   # 左右偏移随物块位置变化：物块偏右(+y)夹爪也偏右就填负，偏左就填正
PLACE_SAFE = 0.035       # 放物块时停在框上方多高松手(米)，防撞框；撞就调大
BOX_OFFSET = np.array([-0.05, 0.0, 0.0])      # 放框补偿：太前调更负，偏后调正
TOOL_OFFSET = 0.11
GRASP_LOW = 0.02    # 抓取再低一点(米)，夹到方块中部；太高就加大，太低就减小   # 腕部到夹爪中心距离(米)，按实际量；防夹爪插桌面
DEBUG_DIR = "debug"
# ==================================================
DRY = "--dry" in sys.argv
# ★ 你的 YOLO 类别名（按你训练时的改，以后加类别就加进对应列表）★
CUBE_CLASSES = ["red_fang", "brown_fang"]   # 可以抓的方块
BOX_CLASSES = ["bule_box"]                  # 要放进去的框
os.makedirs(DEBUG_DIR, exist_ok=True)
K = np.load("calib/calib_intrinsics.npz")["K"]
dist = np.load("calib/calib_intrinsics.npz")["dist"]
T_cam2ee = np.load("calib/calib_handeye.npz")["T_cam2ee"]
pl = np.load("calib/table_plane.npz")
N = pl["n"]
D = float(pl["d"])
model = YOLO("weights/best.pt")
if DRY:
    robot = None
    kin = PyBulletKin(None)
    print("【DRY 模式】只检测，不动机械臂")
else:
    robot = make_robot()
    kin = PyBulletKin(list(getattr(robot, "motor_names", [])))
    print("机械臂电机：", getattr(robot, "motor_names", []))
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FLIP = 0   # 摄像头方向修正：0不翻转 1左右 -1旋转180（test_camera.py 测）
step = 0

def save_debug(img, det, note=""):
    global step
    for cls, items in det.items():
        for it in items:
            x1, y1, x2, y2 = it["bbox"]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, cls, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(f"{DEBUG_DIR}/{step:03d}_{note}.jpg", img)
    json.dump({k: [i["point"].tolist() for i in v] for k, v in det.items()},
              open(f"{DEBUG_DIR}/{step:03d}_{note}.json", "w"), indent=2)
    step += 1

def T_cam2base(q):
    return kin.fk(q[:5]) @ T_cam2ee

def pixel_to_3d(u, v, Tcb):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dc = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    dc /= np.linalg.norm(dc)
    R, o = Tcb[:3, :3], Tcb[:3, 3]
    db = R @ dc
    t = (D - N @ o) / (N @ db)
    return o + t * db

def detect(frame, Tcb):
    res = model.predict(frame, conf=CONF, verbose=False)[0]
    out = {}
    for i, box in enumerate(res.boxes):
        cls = model.names[int(box.cls)]
        if cls in BOX_CLASSES:
            key = "box"
        elif cls in CUBE_CLASSES:
            key = "cube"
        else:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cu, cv_ = (x1 + x2) / 2, (y1 + y2) / 2
        p3 = pixel_to_3d(cu, cv_, Tcb)
        ang = 0.0
        if res.masks is not None:
            cnt = res.masks[i].xy[0].astype(np.int32)
            (_, _), (w, h), a = cv2.minAreaRect(cnt)
            ang = a if w >= h else a + 90
        out.setdefault(key, []).append({"point": p3, "angle": ang, "bbox": (x1, y1, x2, y2)})
    return out

def grasp_pose(pt, yaw):
    Rz = Rot.from_euler("z", yaw, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rz
    T[:3, 3] = pt + N * (CUBE_H / 2 + TOOL_OFFSET - GRASP_LOW)
    return T

def _quat_slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    q = (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1
    return q / np.linalg.norm(q)

def interpolate(T0, T1, steps=15):
    out = []
    for s in np.linspace(0, 1, steps):
        T = np.eye(4)
        T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
        T[:3, :3] = Rot.from_quat(_quat_slerp(Rot.from_matrix(T0[:3, :3]).as_quat(), Rot.from_matrix(T1[:3, :3]).as_quat(), s)).as_matrix()
        out.append(T)
    return out

def pose_at(pos):
    T = np.eye(4)
    T[:3, 3] = pos
    return T


def cartesian_move(q, T_target, steps=20):
    p0 = kin.fk(q[:5])[:3, 3]
    p1 = T_target[:3, 3]
    for s in np.linspace(0, 1, steps):
        pos = (1 - s) * p0 + s * p1
        qi = kin.ik(q[:5], pose_at(pos))
        if qi is None:
            continue
        act = np.concatenate([qi, [q[5]]])
        move_robot(robot, act, duration=STEP_DUR)
        q = wait_settled(robot, act, tol_deg=6.0, timeout=3.0)
    return q


def joint_move(q, q_target, steps=28):
    q_target = np.asarray(q_target, dtype=float)
    for s in np.linspace(0, 1, steps):
        qi = (1 - s) * q[:5] + s * q_target[:5]
        act = np.concatenate([qi, [q[5]]])
        move_robot(robot, act, duration=STEP_DUR)
        q = wait_settled(robot, act, tol_deg=6.0, timeout=3.0)
    return q


def move_to(q, T_target, steps=28, hold_gripper_close=False):
    # 关节空间插值：先求目标关节角，再平滑移动（不会中途 IK 无解，也更慢更稳）
    q_goal = kin.ik(q[:5], T_target)
    grip = q[5]   # 固定夹爪值，防止抖动
    if q_goal is None:
        raise RuntimeError("目标位姿不可达（IK 无解）：\n" + str(np.round(T_target, 3)))
    for s in np.linspace(0, 1, steps):
        qi = (1 - s) * q[:5] + s * q_goal
        # ★修改：hold_gripper_close=True时强制使用GRIPPER_CLOSE，不读取回弹后的夹爪位置
        if hold_gripper_close:
            act = np.concatenate([qi, [GRIPPER_CLOSE]])
        else:
            act = np.concatenate([qi, [grip]])
        move_robot(robot, act, duration=STEP_DUR)
        q = wait_settled(robot, act, tol_deg=6.0, timeout=3.0)
        okf, frm = cap.read()
        if okf:
            if FLIP:
                frm = cv2.flip(frm, FLIP)
            cv2.imshow("run", frm)
            cv2.waitKey(1)
    return q

ok, frame = cap.read()
if FLIP:
    frame = cv2.flip(frame, FLIP)
if not ok:
    raise SystemExit("摄像头读不到画面")
q = get_q(robot) if robot else np.zeros(6)
q_start = q.copy()   # 记住起始姿态，结束回去
det = detect(frame, T_cam2base(q))
save_debug(frame, det, "detect")
cv2.imshow("debug", frame)
cv2.waitKey(1)
cubes = sorted(det.get("cube", []), key=lambda c: (c["bbox"][2]-c["bbox"][0])*(c["bbox"][3]-c["bbox"][1]), reverse=True)
boxes = det.get("box", [])
if not cubes or not boxes:
    print("没有同时检测到方块和框！先用 step6_detect_preview.py 看检测效果，")
    print("调整位置/光照/角度后重试。")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit("检测失败")
cube = cubes[0]
box_ = boxes[0]

# 检查框是否在机械臂够得着的范围（SO-101 约 0.35m 内）
_box_dist = np.hypot(box_["point"][0], box_["point"][1])
if _box_dist > 0.35:
    print("框太远（%.2f m），机械臂够不到，请把框放近一点（0.35m 以内）再运行。" % _box_dist)
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit("框太远")

# 框的精确位置在抬升后会重新检测（见第4步之后）
kx = (OFFSET_FAR - OFFSET_NEAR) / (DIST_FAR - DIST_NEAR)
bx = OFFSET_NEAR - kx * DIST_NEAR
cube["point"][0] += kx * np.hypot(cube["point"][0], cube["point"][1]) + bx
cube["point"][1] += CUBE_OFFSET_Y + Y_OFFSET_SLOPE * cube["point"][1]
box_["point"] = box_["point"] + BOX_OFFSET
print("方块 3D 位置:", np.round(cube["point"], 3), "  朝向:", round(cube["angle"], 1))
print("框   3D 位置:", np.round(box_["point"], 3))
if DRY:
    print()
    print("【DRY 模式】只检测，不动机械臂。")
    print("请拿尺子量一下方块的真实位置，和上面打印的 3D 位置比一比，")
    print("如果差很多，说明标定有问题，回到第 2~4 步检查。")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit("DRY 模式结束（这是正常结束，不是报错）")
grasp = grasp_pose(cube["point"], cube["angle"])
pre = grasp.copy()
pre[:3, 3] = cube["point"] + N * (CUBE_H / 2 + TOOL_OFFSET + PRE_LIFT)
place = grasp_pose(box_["point"], box_["angle"])
pre_place = place.copy()
pre_place[:3, 3] = box_["point"] + N * (CUBE_H / 2 + TOOL_OFFSET + PRE_LIFT)
print("1) 移动到方块上方...")
q = move_to(q, pre)
print("2) 使用初始检测的抓取点（跳过重检测，避免误检偏移）")
print("3) 下抓...")
# 先张开夹爪（确保能包住方块）
act = q.copy()
act[5] = GRIPPER_OPEN
move_robot(robot, act, duration=0.5)
q = wait_settled(robot, act, tol_deg=5.0, timeout=5.0)
for _ in range(20):
    if abs(get_q(robot)[5] - GRIPPER_OPEN) < 5:
        break
    time.sleep(0.2)
print("   缓慢下降...  (当前z=%.3f, 目标z=%.3f)" % (kin.fk(q[:5])[:3, 3][2], grasp[:3, 3][2]))
target_z = grasp[:3, 3][2]
prev_z = None
for i in range(40):
    ee = kin.fk(q[:5])[:3, 3]
    if ee[2] <= target_z + 0.005:
        print("   已到下抓高度 (z=%.3f)" % ee[2])
        break
    if prev_z is not None and ee[2] > prev_z - 0.003:
        print("   被桌面/物块顶住，停止下降 (z=%.3f)" % ee[2])
        break
    prev_z = ee[2]
    step_target = grasp.copy()
    step_target[:3, 3] = ee - N * 0.01
    try:
        q = move_to(q, step_target, steps=4)
    except RuntimeError:
        print("   IK 无解，停止下降")
        break

act = q.copy()
act[5] = GRIPPER_CLOSE
move_robot(robot, act, duration=0.5)
wait_settled(robot, act, tol_deg=5.0, timeout=5.0)
time.sleep(0.3)
print("4) 抬升...")
lift = grasp.copy()
lift[:3, 3] = grasp[:3, 3] + N * (LIFT_H)
# ★握住物体，强制保持夹紧
q = move_to(q, lift, hold_gripper_close=True)

# 抬起来后，重新检测框（能拍到完整框，位置更准）
ok, frame = cap.read()
if FLIP:
    frame = cv2.flip(frame, FLIP)
q_now = get_q(robot)
det3 = detect(frame, T_cam2base(q_now))
save_debug(frame, det3, "regrasp_box")
if det3.get("box"):
    box2 = det3["box"][0]
    new_pt = box2["point"] + BOX_OFFSET
    _d2 = np.hypot(new_pt[0], new_pt[1])
    if _d2 < 0.34 and np.linalg.norm(new_pt - box_["point"]) < 0.15:
        box_["point"] = new_pt
        box_["angle"] = box2["angle"]
        print("   重新检测到框位置:", np.round(new_pt, 3))
        place = grasp_pose(box_["point"], box_["angle"])
    else:
        print("   重检测框不可达或偏差大，用最初位置")
else:
    print("   抬升后没检测到框，用最初位置")

print("5) 平移到框上方（保持抬升高度）...")
transit = place.copy()
transit[:3, 3] = np.array([box_["point"][0], box_["point"][1], lift[:3, 3][2]])
q = move_to(q, transit, hold_gripper_close=True)

print("6) 下降到框内并松开...")
place_high = place.copy()
place_high[:3, 3] = place[:3, 3] + N * (PLACE_SAFE)
q = cartesian_move(q, place_high, steps=10)   # 垂直下降，不往前甩
act = q.copy()
act[5] = GRIPPER_OPEN
move_robot(robot, act, duration=0.5)
q = wait_settled(robot, act, tol_deg=5.0, timeout=5.0)
for _ in range(20):
    if abs(get_q(robot)[5] - GRIPPER_OPEN) < 5:
        break
    time.sleep(0.2)
time.sleep(0.3)
print("7) 抬升收尾...")
lift2 = place.copy()
lift2[:3, 3] = place[:3, 3] + N * (PLACE_LIFT_H)
q = move_to(q, lift2)
print("8) 回到起始姿态...")
act = q.copy()
act[5] = GRIPPER_CLOSE
move_robot(robot, act, duration=0.5)
q = wait_settled(robot, act, tol_deg=5.0, timeout=5.0)
try:
    # 先抬到高处（当前位置上方 12cm）
    cur = kin.fk(q[:5])[:3, 3]
    q = cartesian_move(q, pose_at(cur + N * 0.12), steps=10)
    # 平移到起点上方 12cm
    start_pos = kin.fk(q_start[:5])[:3, 3]
    q = cartesian_move(q, pose_at(start_pos + N * 0.12), steps=20)
    # 落回起点（关节空间，和开始完全一样）
    q = joint_move(q, q_start[:5], steps=30)
except RuntimeError:
    print("   回位失败（跳过）")
print("完成！检查方块是否放进了框。")
robot.disconnect()
cap.release()
cv2.destroyAllWindows()

