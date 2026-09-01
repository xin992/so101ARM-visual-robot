# -*- coding: utf-8 -*-
"""第 5 步 C：检测预览（把训练好的 best.pt 在真实相机上跑起来看效果）
运行：python step6_detect_preview.py   需要：weights/best.pt 已存在
"""
import cv2
from ultralytics import YOLO

model = YOLO("weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FLIP = 0   # 摄像头方向修正：0不翻转 1左右 -1旋转180（test_camera.py 测）

print("窗口里应该看到 cube 和 box 的绿色框。q 退出。")
while True:
    ok, frame = cap.read()
    if FLIP:
        frame = cv2.flip(frame, FLIP)
    if not ok:
        break
    res = model.predict(frame, conf=0.3, verbose=False)[0]
    for box in res.boxes:
        cls = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, cls, (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("yolo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
