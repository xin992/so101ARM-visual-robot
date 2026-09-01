# -*- coding: utf-8 -*-
"""测摄像头方向：0=不翻转 1=左右镜像 r=旋转180 q=退出  运行：python test_camera.py"""
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FLIP = -1
print("按键：0=不翻转  1=左右镜像  r=旋转180°  q=退出")
while True:
    ok, frame = cap.read()
    if not ok:
        print("摄像头读不到画面"); break
    if FLIP:
        frame = cv2.flip(frame, FLIP)
    cv2.putText(frame, f"FLIP={FLIP}   0:no 1:mirror r:180 q:exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("camera", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("0"): FLIP = 0
    elif k == ord("1"): FLIP = 1
    elif k == ord("r"): FLIP = -1
    elif k == ord("q"): break
cap.release(); cv2.destroyAllWindows()
print(f"请记住：FLIP = {FLIP}（填到所有脚本顶部同一个位置）")
