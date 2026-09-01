# -*- coding: utf-8 -*-
"""第 3 步 B（修正版 v2）：手眼标定 + 验证（marker 版检测）

相对 v1 又改了什么：
- v1 修复了角点顺序 bug（不再手写 order=[2,3,0,1]），② 从 56.6px 降到 ~17px。
- v2：重投影真正用 cv2.projectPoints（带畸变）做公平对比（v1 文档写了但代码漏了）。
- v2：新增“逐姿态诊断”，打印每个姿态的重投影误差 / 板到相机距离 / 视角，
  帮你找出那 1~2 个把平均误差拉高的坏姿态（模糊、板不完整、角度太刁）。

用法：python step3b_calib_handeye.py
"""
import glob, json
import cv2
import numpy as np
from robot_lib import PyBulletKin, detect_board

BOARD = cv2.aruco.CharucoBoard((7, 5), squareLength=0.0375, markerLength=0.028125,
                               dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
intr = np.load("calib/calib_intrinsics.npz")
K, dist = intr["K"], intr["dist"]

def _board_obj_points(B):
    if hasattr(B, "getObjPoints"):
        return B.getObjPoints()
    return B.getObjectPoints()

_ids = [int(x) for x in BOARD.getIds()]
OBJ_BY_ID = {mid: np.asarray(pt, dtype=float) for mid, pt in zip(_ids, _board_obj_points(BOARD))}

def board_pose_from_markers(mc, mi):
    """所有 marker 角点一起 solvePnP -> T_board2cam（不依赖 robot_lib 版本）"""
    obj_pts, img_pts = [], []
    for corners, mid in zip(mc, mi):
        mid = int(mid[0])
        if mid not in OBJ_BY_ID:
            continue
        obj_pts.append(OBJ_BY_ID[mid])
        img_pts.append(np.asarray(corners, dtype=float).reshape(4, 2))
    if len(obj_pts) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(np.vstack(obj_pts), np.vstack(img_pts),
                                  K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T

def project3d(pts3, T_base2cam):
    """把 3D 点用【带畸变】的内参投到像素（和 detectBoard 出来的原始像素公平对比）"""
    rvec, tvec = cv2.Rodrigues(T_base2cam[:3, :3])[0], T_base2cam[:3, 3]
    p, _ = cv2.projectPoints(pts3, rvec, tvec, K, dist)
    return p.reshape(-1, 2)

# ================= 读姿态数据 =================
files = sorted(glob.glob("calib/handeye/pose_*.json"))
print(f"找到 {len(files)} 个姿态")
if not files:
    raise SystemExit("没有姿态数据，请先运行 step3a_collect_handeye.py")
kin = PyBulletKin(json.load(open(files[0])).get("motor_names"))

used_files, T_ee2base_list, T_b2c_list = [], [], []
for f in files:
    d = json.load(open(f))
    img = cv2.imread(d["image"])
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mc, mi, _, _ = detect_board(BOARD, gray)
    if mi is None or len(mi) < 4:
        print(f"  跳过（检测不到板）：{f}")
        continue
    T_b2c = board_pose_from_markers(mc, mi)
    if T_b2c is None:
        continue
    T_ee2base = kin.fk(np.asarray(d["q"], dtype=float)[:5])
    used_files.append(f)
    T_ee2base_list.append(T_ee2base)
    T_b2c_list.append(T_b2c)

used = len(used_files)
print(f"有效姿态 {used} 个")
if used < 10:
    raise SystemExit("有效姿态太少（<10）。重新采集。")

# ================= 姿态多样性诊断 =================
rel_angles, rel_trans = [], []
T0 = T_ee2base_list[0]
for T in T_ee2base_list[1:]:
    R = T0[:3, :3].T @ T[:3, :3]
    rel_angles.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    rel_trans.append(np.linalg.norm(T[:3, 3] - T0[:3, 3]))
print(f"姿态多样性：相对首姿态 旋转 平均 {np.mean(rel_angles):.1f}° 最大 {np.max(rel_angles):.1f}°；"
      f"平移 平均 {np.mean(rel_trans)*100:.1f} cm 最大 {np.max(rel_trans)*100:.1f} cm")
if np.max(rel_angles) < 40:
    print("警告：姿态间旋转变化偏小（最大 <40°）。手眼标定对旋转最敏感，建议重采：")
    print("  每个姿态让手腕绕【不同方向】明显转动 30~60°，板保持完整清晰，共 15~20 个。")

# ================= 手眼求解（两种方法都算，选离散度小的） =================
results = []
for method, name in [(cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS"),
                     (cv2.CALIB_HAND_EYE_TSAI, "TSAI")]:
    R_c2g, t_c2g = cv2.calibrateHandEye(
        [T[:3, :3] for T in T_ee2base_list], [T[:3, 3] for T in T_ee2base_list],
        [T[:3, :3] for T in T_b2c_list], [T[:3, 3] for T in T_b2c_list], method=method)
    T_cam2ee = np.eye(4)
    T_cam2ee[:3, :3] = R_c2g
    T_cam2ee[:3, 3] = t_c2g.reshape(3)
    T_boards = np.array([A @ T_cam2ee @ B for A, B in zip(T_ee2base_list, T_b2c_list)])
    spread = np.linalg.norm(T_boards[:, :3, 3] - T_boards[:, :3, 3].mean(0), axis=1)
    print(f"{name}: 板位姿离散度 平均 {spread.mean()*100:.1f} cm，最大 {spread.max()*100:.1f} cm")
    results.append((spread.mean(), T_cam2ee, T_boards, name))

results.sort(key=lambda x: x[0])
_, T_cam2ee, T_boards, name = results[0]
np.savez("calib/calib_handeye.npz", T_cam2ee=T_cam2ee)
print(f"\n采用 {name} 的结果：")
print("T_cam2ee（相机相对末端的变换）:")
print(np.round(T_cam2ee, 4))
print("已保存 calib/calib_handeye.npz")

# ================= 验证 ① 板位姿离散度 =================
center = T_boards[:, :3, 3].mean(axis=0)
spread = np.linalg.norm(T_boards[:, :3, 3] - center, axis=1)
print(f"① 板位姿离散度：平均 {spread.mean()*100:.1f} cm，最大 {spread.max()*100:.1f} cm  <- 应 < 2~3cm")

# ================= 验证 ② 重投影（带畸变 + 逐姿态诊断） =================
T_avg = T_boards.mean(axis=0)
U, _, Vt = np.linalg.svd(T_avg[:3, :3])
T_avg[:3, :3] = U @ Vt

center_errs, corner_errs = [], []
per_pose = []            # (文件名, 平均角点误差, 最大角点误差, 板距cm, 视角°)
for f, A, B in zip(used_files, T_ee2base_list, T_b2c_list):
    img = cv2.imread(json.load(open(f))["image"])
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mc, mi, _, _ = detect_board(BOARD, gray)
    if mi is None or len(mi) < 4:
        continue
    T_cam2base = A @ T_cam2ee
    T_base2cam = np.linalg.inv(T_cam2base)
    c_errs, co_errs, cam_dists = [], [], []
    for corners, mid in zip(mc, mi):
        mid = int(mid[0])
        if mid not in OBJ_BY_ID:
            continue
        obj = OBJ_BY_ID[mid]
        det = np.asarray(corners, dtype=float).reshape(4, 2)
        p_base = T_avg[:3, :3] @ obj.T + T_avg[:3, 3:4]
        p_cam = T_base2cam[:3, :3] @ p_base + T_base2cam[:3, 3:4]
        p_img = project3d(p_cam.T, np.eye(4))          # p_cam 已是相机系
        co_errs.extend(np.linalg.norm(p_img - det, axis=1).tolist())
        c3 = obj.mean(axis=0)
        p_base = T_avg[:3, :3] @ c3 + T_avg[:3, 3]
        p_cam = T_base2cam[:3, :3] @ p_base + T_base2cam[:3, 3]
        p_img = project3d(p_cam[None, :], np.eye(4))
        c_errs.append(np.linalg.norm(p_img[0] - det.mean(axis=0)))
        cam_dists.append(np.linalg.norm(p_cam))
    co_errs = np.array(co_errs); c_errs = np.array(c_errs)
    corner_errs.extend(co_errs.tolist())
    center_errs.extend(c_errs.tolist())
    n_b = B[:3, :3] @ np.array([0, 0, 1.0])            # 板法线（相机系）
    view_ang = np.degrees(np.arccos(np.clip(n_b[2], -1, 1)))
    per_pose.append((f, co_errs.mean(), co_errs.max(),
                     float(np.mean(cam_dists)), view_ang))

center_errs = np.array(center_errs)
corner_errs = np.array(corner_errs)
print(f"marker 中心重投影误差：平均 {center_errs.mean():.1f} 像素，最大 {center_errs.max():.1f} 像素")
print(f"② 重投影误差（角点）：平均 {corner_errs.mean():.1f} 像素，最大 {corner_errs.max():.1f} 像素  <- 应 < 10")

per_pose.sort(key=lambda x: -x[1])
print("\n--- 逐姿态诊断（最差 3 个）---")
for f, mean_e, max_e, bd, va in per_pose[:3]:
    print(f"  {f}: 平均 {mean_e:.1f}px 最大 {max_e:.1f}px | 板距 {bd*100:.0f}cm 视角 {va:.0f}°")
med = np.median([x[1] for x in per_pose])
for f, mean_e, max_e, bd, va in per_pose:
    if mean_e > 3 * med:
        print(f"  建议删掉坏姿态：{f}（误差 {mean_e:.1f}px，是中位数的 3 倍+）。"
              f"删掉 pose_xxx.json 和 pose_xxx.png 后重跑本脚本。")

if spread.mean()*100 < 3 and corner_errs.mean() < 10:
    print("合格：手眼标定没问题，可以继续第 4 步。")
else:
    print("下一步建议：先删掉上面标出的坏姿态重跑；若仍 >10px，就按提示重采姿态（旋转变化加大）。")

