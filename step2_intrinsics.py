# -*- coding: utf-8 -*-
"""第 2 步：相机内参标定（采集 + 计算一步完成） 运行：python step2_intrinsics.py
操作：标定板在相机前摆不同姿态，空格=拍照，q=退出并自动计算。"""
import os
import glob
import cv2
import numpy as np

FLIP = 0
os.makedirs("calib/calib_imgs", exist_ok=True)
BOARD = cv2.aruco.CharucoBoard(
    (7, 5), squareLength=0.0375, markerLength=0.028125,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
)
CAM_INDEX = 0
W, H = 640, 480

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

n = 0
print("采集开始：标定板摆不同姿态（远近/倾斜/左右），空格拍照，q 结束；至少 30 张")
while True:
    ok, frame = cap.read()
    if FLIP:
        frame = cv2.flip(frame, FLIP)
    if not ok:
        print("摄像头读不到画面，检查 CAM_INDEX 和摄像头连接")
        break
    disp = frame.copy()
    cv2.putText(disp, f"saved={n}  SPACE:save  q:finish", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("intrinsics", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        fn = f"calib/calib_imgs/{n:03d}.png"
        cv2.imwrite(fn, frame)
        n += 1
        print("已保存", fn)
    elif k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"共采集 {n} 张")

files = sorted(glob.glob("calib/calib_imgs/*.png")) + sorted(glob.glob("calib/calib_imgs/*.jpg"))
print(f"找到 {len(files)} 张图片，开始计算内参...")

det = cv2.aruco.CharucoDetector(BOARD)
obj_pts, img_pts = [], []
used = 0
for f in files:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < 4:
        print("  跳过（检测不到标定板）：", os.path.basename(f))
        continue
    res = BOARD.matchImagePoints(corners, ids)
    if isinstance(res, (tuple, list)) and len(res) == 3:
        obj, imgp = res[1], res[2]
    else:
        obj, imgp = res
    obj_pts.append(obj)
    img_pts.append(imgp)
    used += 1

print(f"有效图片 {used} 张")
if used < 10:
    raise SystemExit("有效图片太少（<10 张）。常见原因：照片模糊、反光、板不完整、格子边长写错。重新采集。")

h, w = gray.shape
ret, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
print("重投影误差（像素，越小越好：<0.5 很好，<1 能用）:", round(ret, 3))
np.savez("calib/calib_intrinsics.npz", K=K, dist=dist)
print("内参矩阵 K:")
print(K)
print("已保存 calib/calib_intrinsics.npz")
