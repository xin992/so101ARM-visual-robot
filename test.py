import os, re

TEST_CAMERA = '''# -*- coding: utf-8 -*-
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
'''

# 1) 修 robot_lib.py：pybullet 新版没有 getConnectionId
if os.path.exists("robot_lib.py"):
    s = open("robot_lib.py", encoding="utf-8").read()
    if "getConnectionId" in s:
        s = s.replace(
            "if p.getConnectionId() < 0:",
            'info = p.getConnectionInfo()\n        connected = info.get("isConnected", 0) if isinstance(info, dict) else 0\n        if connected < 1:'
        )
        open("robot_lib.py", "w", encoding="utf-8").write(s)
        print("[OK] robot_lib.py 已修复")
    else:
        print("[--] robot_lib.py 已是新版本，跳过")
else:
    print("[!!] 没找到 robot_lib.py，请确认在 ~/robot_learning/pick_place 下运行")

# 2) 创建 test_camera.py
if not os.path.exists("test_camera.py"):
    open("test_camera.py", "w", encoding="utf-8").write(TEST_CAMERA)
    print("[OK] 已创建 test_camera.py")
else:
    print("[--] test_camera.py 已存在，跳过")

# 3) 给采集/检测脚本加摄像头翻转 FLIP（幂等）
for f in ["step2_intrinsics.py", "step3a_collect_handeye.py", "step5_collect_yolo.py", "step6_detect_preview.py", "step7_pick_place.py"]:
    if not os.path.exists(f):
        print(f"[!!] {f} 不存在，跳过")
        continue
    s = open(f, encoding="utf-8").read()
    changed = False
    if "FLIP" not in s:
        s = s.replace(
            "cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)",
            "cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)\nFLIP = -1   # 摄像头方向修正：0不翻转 1左右 -1旋转180（test_camera.py 测）",
            1,
        )
        changed = True
    if "cv2.flip(frame, FLIP)" not in s:
        s = re.sub(
            r"^(\s*)(ok, frame = cap\.read\(\))",
            lambda m: m.group(1) + m.group(2) + "\n" + m.group(1) + "if FLIP:\n" + m.group(1) + "    frame = cv2.flip(frame, FLIP)",
            s,
            flags=re.MULTILINE,
        )
        changed = True
    open(f, "w", encoding="utf-8").write(s)
    print(f"[{'OK' if changed else '--'}] {f}")

print("\n完成！下一步：python test_camera.py 测翻转方向")
