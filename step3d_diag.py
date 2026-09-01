# -*- coding: utf-8 -*-
"""第 3 步 D：检测诊断（看检测到底卡在哪）
运行：python step3d_diag.py
"""
import glob, json, os
import cv2
import numpy as np
from robot_lib import detect_board, match_image_points

BOARD = cv2.aruco.CharucoBoard((11,8), 0.030, 0.0225,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
files = sorted(glob.glob("calib/handeye/pose_*.json"))
print("姿态文件数:", len(files))

for idx in [0, 15, 30, 59]:
    if idx >= len(files):
        continue
    d = json.load(open(files[idx]))
    print("\n===== pose_%03d =====" % idx)
    print("json image 字段:", d["image"])
    img = cv2.imread(d["image"])
    if img is None:
        print("  imread 失败（None）")
        continue
    print("  分辨率:", img.shape[1], "x", img.shape[0], " 大小:", os.path.getsize(d["image"]), "字节")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("  平均亮度:", round(gray.mean(), 1))

    # 方法1：robot_lib 的 detect_board
    try:
        corners, ids, mc, mi = detect_board(BOARD, gray)
        n_c = 0 if corners is None else len(corners)
        n_i = 0 if ids is None else len(ids)
        print("  robot_lib.detect_board -> 角点:", n_c, " ids:", n_i)
    except Exception as e:
        print("  robot_lib.detect_board 报错:", repr(e))

    # 方法2：直接用 ArucoDetector 找 marker
    try:
        ad = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
        mcorners, mids, _ = ad.detectMarkers(gray)
        n_m = 0 if mids is None else len(mids)
        print("  ArucoDetector 找到 marker 数:", n_m)
        if mids is not None and len(mids) > 0:
            print("  前5个 marker id:", [int(x) for x in mids.flatten()[:5]])
    except Exception as e:
        print("  ArucoDetector 报错:", repr(e))


