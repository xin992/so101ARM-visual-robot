# -*- coding: utf-8 -*-
"""实时位置显示：连上机械臂，键盘控制摆姿态，画面实时显示每个物体在基座坐标系下的 3D 位置（米）
用法：python step7_live.py --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower
按键：q/w e/r t/y u/i o/p = 5 个臂关节  g = 夹爪  x = 退出
"""
import cv2
import numpy as np
from ultralytics import YOLO
from robot_lib import make_robot, get_q, move_robot, PyBulletKin

# ★ 你的类别名 ★
CUBE_CLASSES = ["red_fang", "brown_fang"]
BOX_CLASSES = ["bule_box"]
CONF = 0.3
FLIP = 0
STEP = 3.0
DUR = 0.2

K = np.load("calib/calib_intrinsics.npz")["K"]
dist = np.load("calib/calib_intrinsics.npz")["dist"]
T_cam2ee = np.load("calib/calib_handeye.npz")["T_cam2ee"]
pl = np.load("calib/table_plane.npz")
N = pl["n"]; D = float(pl["d"])

model = YOLO("weights/best.pt")
robot = make_robot()
kin = PyBulletKin(list(getattr(robot, "motor_names", [])))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def pixel_to_3d(u, v, Tcb):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    dc = np.array([(u-cx)/fx, (v-cy)/fy, 1.0]); dc /= np.linalg.norm(dc)
    R, o = Tcb[:3,:3], Tcb[:3,3]
    db = R @ dc
    t = (D - N @ o) / (N @ db)
    return o + t * db

print("实时显示中：坐标 = 机械臂基座坐标系（米）。q/w...=关节 g=夹爪 x=退出")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if FLIP:
        frame = cv2.flip(frame, FLIP)
    q = get_q(robot)                        # 真实关节角
    Tcb = kin.fk(q[:5]) @ T_cam2ee          # 相机在基座系下的位姿
    res = model.predict(frame, conf=CONF, verbose=False)[0]
    for i, box in enumerate(res.boxes):
        cls = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cu, cv_ = (x1+x2)/2, (y1+y2)/2
        p3 = pixel_to_3d(cu, cv_, Tcb)
        label = "box" if cls in BOX_CLASSES else ("cube" if cls in CUBE_CLASSES else cls)
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({p3[0]:.3f},{p3[1]:.3f},{p3[2]:.3f})",
                    (int(x1),int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(frame, "q/w e/r t/y u/i o/p=jnt g=gripper x=quit", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    cv2.imshow("live", frame)
    k = cv2.waitKey(1) & 0xFF
    key = chr(k) if k < 128 else ""
    target = q.copy()
    if key == "q": target[0] += STEP
    elif key == "w": target[0] -= STEP
    elif key == "e": target[1] += STEP
    elif key == "r": target[1] -= STEP
    elif key == "t": target[2] += STEP
    elif key == "y": target[2] -= STEP
    elif key == "u": target[3] += STEP
    elif key == "i": target[3] -= STEP
    elif key == "o": target[4] += STEP
    elif key == "p": target[4] -= STEP
    elif key == "g": target[5] = 90.0 if target[5] < 45 else 0.0
    if key in "qwertyuiopg":
        move_robot(robot, target, duration=DUR)
    elif k == ord("x"):
        break

robot.disconnect()
cap.release()
cv2.destroyAllWindows()
