
'''python gripper_test.py --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101_follower'''

import numpy as np, time
from robot_lib import make_robot, move_robot, get_q
robot = make_robot()
act = get_q(robot).copy()
act[5] = 60.0
move_robot(robot, act, duration=0.5)
time.sleep(0.5)
input("请把 3cm 方块放进夹爪中间，放好后按 回车 开始慢慢闭合")
v = 60.0
while True:
    act[5] = float(v)
    move_robot(robot, act, duration=0.1)
    print("当前夹爪值:", v, "  ← 看到刚好夹住方块时，输 q 回车记住")
    k = input("按 回车 = 再闭合2   |   输 q 回车 = 退出: ")
    if k.strip().lower() == "q":
        break
    v = max(0, v - 2)
robot.disconnect()
print("记住：GRIPPER_CLOSE =", v)
