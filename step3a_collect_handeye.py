# -*- coding: utf-8 -*-
"""第 3 步 A（增强版）：手眼标定数据采集
标定板平放在桌面上固定，用键盘控制机械臂摆 15~20 个姿态，空格保存。
相比原版新增：
1) 画面上实时显示当前检测到的 marker 数（太少=板没拍全/太糊/角度太刁）。
2) 保存时检查：marker < 6 会警告；与上一个保存姿态的【末端旋转变化】< 15° 会警告
   （手眼标定需要姿态之间旋转变化大，否则解出来是偏的——你现在 ② 偏高就是这个）。
用法：python step3a_collect_handeye.py --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower
按键：q/w e/r t/y u/i o/p = 5 个臂关节正/反转；g = 夹爪开/关；空格 = 保存；x = 退出
"""
import os
import json
import cv2
import numpy as np
from robot_lib import make_robot, get_q, move_robot, wait_settled, detect_board, PyBulletKin

BOARD = cv2.aruco.CharucoBoard((7, 5), squareLength=0.0375, markerLength=0.028125,
                               dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
OUT = os.environ.get("PP_OUT", "calib/handeye")
os.makedirs(OUT, exist_ok=True)

STEP = 5.0
DUR = 0.2
MIN_MARKERS = 6        # 保存时低于这个数会警告
MIN_ROT_DIFF = 15.0    # 与上一个保存姿态的末端旋转变化小于这个角度会警告

# ========== SO101关节角度限位【放在顶部常量区】 ==========
JOINT_LIMITS = [
    (-160, 160),   # joint0
    (-120, 120),   # joint1 e/r
    (-120, 120),   # joint2 t/y
    (-120, 120),   # joint3 u/i
    (-160, 160),   # joint4 o/p
    (0, 90)        # 夹爪
]

robot = make_robot()
motor_names = list(getattr(robot, "motor_names", []))
print("机械臂电机：", motor_names)
kin = PyBulletKin(motor_names)   # 用于算末端旋转变化（和 step3b 同一套 FK）

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FLIP = 0   # 摄像头方向修正：0不翻转 1左右 -1旋转180（test_camera.py 测）

print("""
操作说明：
1. 把标定板【平放在桌面上】，用胶带固定，全程不要动它。
2. 用按键摆到能看到标定板的姿态（板要完整、清晰、不反光，markers 数量 ≥ 6）。
3. 每个姿态按【空格】保存，共 15~20 个；姿态之间变化要大（远近/高低/倾斜，
   尤其是【旋转方向】要明显不同：俯仰、偏航、横滚都要有）。
4. 完成后按 x 退出。
""")

last_q = None
count = 0

while True:
    ok, frame = cap.read()
    if FLIP:
        frame = cv2.flip(frame, FLIP)
    if not ok:
        print("摄像头读不到画面")
        break

    q = get_q(robot)
    disp = frame.copy()

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mc, mi, _, _ = detect_board(BOARD, gray)
        n_markers = 0 if mi is None else len(mi)
    except Exception:
        n_markers = -1

    status = f"saved={count}  markers={n_markers}  q={np.round(q, 0)}"
    cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("handeye", disp)

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
    elif key == "g": target[5] = 90 if target[5] < 45 else 0.0

    # ========== 钳位，把target限制在JOINT_LIMITS范围内 ==========
    for i in range(5):
        lo, hi = JOINT_LIMITS[i]
        target[i] = np.clip(target[i], lo, hi)

    if key in "qwertyuiopg":
        print(f"下发目标关节: {np.round(target,1)}")
        move_robot(robot, target, duration=DUR)
        wait_settled(robot, target, tol_deg=2.0, timeout=5.0)

    elif k == ord(" "):
        if n_markers >= 0 and n_markers < MIN_MARKERS:
            print(f"  ⚠ 当前只检测到 {n_markers} 个 marker（<{MIN_MARKERS}），"
                  f"板可能没拍全/太糊/角度太刁，建议重摆！")
        rot_diff = 0.0
        if last_q is not None:
            R_now = kin.fk(np.asarray(q, dtype=float)[:5])[:3, :3]
            R_last = kin.fk(np.asarray(last_q, dtype=float)[:5])[:3, :3]
            R = R_last.T @ R_now
            rot_diff = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        if last_q is not None and rot_diff < MIN_ROT_DIFF:
            print(f"  ⚠ 这个姿态相对上一个的末端旋转只差 {rot_diff:.0f}°（<{MIN_ROT_DIFF:.0f}°），"
                  f"太像了；手眼标定需要姿态间旋转变化大，建议换个旋转方向再摆！")
        fn_img = f"{OUT}/pose_{count:03d}.png"
        cv2.imwrite(fn_img, frame)
        json.dump({"image": fn_img, "q": q.tolist(), "motor_names": motor_names},
                  open(f"{OUT}/pose_{count:03d}.json", "w"), indent=2)
        last_q = q.copy()
        count += 1
        print(f"已保存第 {count} 个姿态（markers={n_markers}，末端旋转差 {rot_diff:.0f}°）")

    elif k == ord("x"):
        break

robot.disconnect()
cap.release()
cv2.destroyAllWindows()
print(f"完成，共保存 {count} 个姿态到 {OUT}/")

