# -*- coding: utf-8 -*-
"""第 1 步：生成标定板图片（A4 打印用） 运行：python step1_charuco.py"""
import os
import cv2

os.makedirs("calib", exist_ok=True)
BOARD = cv2.aruco.CharucoBoard(
    (7, 5), squareLength=0.0375, markerLength=0.028125,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
)
# marginSize改成80，减少四周留白，让标定板占满图片
img = BOARD.generateImage((2100, 1500), marginSize=80)
out = "calib/charuco_board.png"
cv2.imwrite(out, img)
print("已生成：" + out)
print("请用 A4 纸、100% 比例打印，关闭适应页面，然后量一下黑色格子实际边长（米），")
print("如果和 0.028 不一样，修改 step2、step3b 里面的 squareLength、markerLength。")
