import glob, json, cv2, numpy as np
from robot_lib import PyBulletKin, detect_board

BOARD = cv2.aruco.CharucoBoard((7,5), 0.0375, 0.028125,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
K = np.load("calib/calib_intrinsics.npz")["K"]
dist = np.load("calib/calib_intrinsics.npz")["dist"]
files = sorted(glob.glob("calib/handeye/pose_*.json"))
kin = PyBulletKin(json.load(open(files[0])).get("motor_names"))
X = int(BOARD.getChessboardSize()[0]); S = float(BOARD.getSquareLength())
M = float(BOARD.getMarkerLength()); h = M/2.0
base = np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]], float)

def board_pose(mc, mi, order):
    obj, imgp = [], []
    for corners, mid in zip(mc, mi):
        mid = int(mid[0]); k = 2*mid+1; j, i = k//X, k%X
        cx, cy = (i+0.5)*S, (j+0.5)*S
        for t, idx in enumerate(order):
            obj.append(base[idx]+[cx,cy,0.0]); imgp.append(corners.reshape(4,2)[t])
    ok, rvec, tvec = cv2.solvePnP(np.array(obj,float), np.array(imgp,float), K, dist)
    if not ok: return None
    T = np.eye(4); T[:3,:3] = cv2.Rodrigues(rvec)[0]; T[:3,3] = tvec.reshape(3)
    return T

poses = []
for f in files:
    d = json.load(open(f))
    img = cv2.imread(d["image"])
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mc, mi, _, _ = detect_board(BOARD, gray)
    if mi is None or len(mi) < 4: continue
    poses.append((kin.fk(np.asarray(d["q"], dtype=float)[:5]), mc, mi))
print("有效姿态:", len(poses))

for name, order in [('A[0,1,2,3]',[0,1,2,3]), ('B[1,2,3,0]',[1,2,3,0]),
                    ('C[2,3,0,1]',[2,3,0,1]), ('D[3,0,1,2]',[3,0,1,2])]:
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for T_ee, mc, mi in poses:
        T_b2c = board_pose(mc, mi, order)
        if T_b2c is None: continue
        R_g2b.append(T_ee[:3,:3]); t_g2b.append(T_ee[:3,3])
        R_t2c.append(T_b2c[:3,:3]); t_t2c.append(T_b2c[:3,3])
    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                        method=cv2.CALIB_HAND_EYE_DANIILIDIS)
    T_c2e = np.eye(4); T_c2e[:3,:3] = R_c2g; T_c2e[:3,3] = t_c2g.reshape(3)
    Tbs = []
    for T_ee, mc, mi in poses:
        T_b2c = board_pose(mc, mi, order)
        if T_b2c is None: continue
        Tbs.append(T_ee @ T_c2e @ T_b2c)
    Tbs = np.array(Tbs)
    c = Tbs[:, :3, 3].mean(axis=0)
    sp = np.linalg.norm(Tbs[:, :3, 3] - c, axis=1)
    print('顺序 %s: ① 离散度 平均 %.1f cm, 最大 %.1f cm' % (name, sp.mean()*100, sp.max()*100))
