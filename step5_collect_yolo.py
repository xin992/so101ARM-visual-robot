# -*- coding: utf-8 -*-
"""第 5 步 A：采集 YOLO 训练数据（在 Ubuntu 上，用你的真实相机）
运行：python step5_collect_yolo.py   操作：空格=保存，q=退出
"""
import os
import cv2

OUT = "datasets/raw"
os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FLIP = 0   # 摄像头方向修正：0不翻转 1左右 -1旋转180（test_camera.py 测）

print("""
采集建议（决定 YOLO 泛化能力，很重要）：
  1. 方块换多种颜色（红/蓝/绿/黄...）、多种大小
  2. 方块和框放在不同位置、不同距离
  3. 换光照（开灯/关灯/拉窗帘），桌上放点杂物
  4. 相机（机械臂）换不同高度、角度
  目标 300~600 张，宁可多拍。
""")

n = 0
while True:
    ok, frame = cap.read()
    if FLIP:
        frame = cv2.flip(frame, FLIP)
    if not ok:
        print("摄像头读不到画面")
        break
    disp = frame.copy()
    cv2.putText(disp, f"saved={n}  SPACE:save  q:quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("collect", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        fn = f"{OUT}/{n:04d}.jpg"
        cv2.imwrite(fn, frame)
        n += 1
        print("已保存", fn)
    elif k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"共保存 {n} 张到 {OUT}/")
