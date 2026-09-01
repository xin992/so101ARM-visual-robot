# -*- coding: utf-8 -*-
"""第 4 步：桌面平面标定（marker 版检测 + 法向量强制朝上）"""
import os, glob, json
import cv2
import numpy as np
from robot_lib import PyBulletKin, detect_board, board_pose_from_markers

BOARD = cv2.aruco.CharucoBoard((7, 5), squareLength=0.0375, markerLength=0.028125,
                               dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
intr = np.load("calib/calib_intrinsics.npz")
K, dist = intr["K"], intr["dist"]
T_cam2ee = np.load("calib/calib_handeye.npz")["T_cam2ee"]

files = sorted(glob.glob("calib/handeye/pose_*.json"))
if not files:
    raise SystemExit("没有数据，先运行 step3a 采集手眼数据")
first = json.load(open(files[0]))
kin = PyBulletKin(first.get("motor_names"))

ns, pts = [], []
for f in files:
    d = json.load(open(f))
    img = cv2.imread(d["image"])
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mc, mi, _, _ = detect_board(BOARD, gray)
    if mi is None or len(mi) < 4:
        continue
    T_board2cam = board_pose_from_markers(BOARD, mc, mi, K, dist)
    if T_board2cam is None:
        continue
    T_ee2base = kin.fk(np.asarray(d["q"], dtype=float)[:5])
    T_cam2base = T_ee2base @ T_cam2ee
    T_board2base = T_cam2base @ T_board2cam
    n_ = T_board2base[:3, 2]
    n_ = n_ / np.linalg.norm(n_)
    p_ = T_board2base[:3, 3]
    ns.append(n_)
    pts.append(p_)

n = np.mean(ns, axis=0)
n = n / np.linalg.norm(n)
if n[2] < 0:
    n = -n
    print("法向量原为朝下，已翻转为朝上（+z）")
p_avg = np.mean(pts, axis=0)
d = float(n @ p_avg)

np.savez("calib/table_plane.npz", n=n, d=d)
print("桌面平面：法向量 n =", np.round(n, 4))
print("常数 d =", round(d, 4))
print("桌面高度 z ≈", round(p_avg[2], 4), "米（应在 0 附近，负一点点正常）")
print("已保存 calib/table_plane.npz")
