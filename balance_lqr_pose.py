# -*- coding: utf-8 -*-
"""webm 姿态(+90°绕x、轮着地、legs=0)的缩聚 LQR 自平衡 + 腿高闭环.

状态: θ(CoM-轮心连线相对竖直的倾角), θ̇, v(基座+y 速度)
控制: u = 单轮力矩(两轮同号); 腿: 高度误差 → 后曲柄=+off/前曲柄=−off.
运行: python balance_lqr_pose.py            # 离线 12s
      python balance_lqr_pose.py --viewer   # 弹窗
"""
import sys
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are

XML = "real_wheelleg_ground.xml"
WHEEL_R = 0.068
Q_DIAG = np.array([900.0, 600.0, 60.0])
R_VAL  = np.array([1.8])
LEG_DAMP = 1.0
KIV = 2.0
U_SMOOTH = 0.12
CTRL_EVERY = 5
W_DAMP = 0.15
KIT = 3.0
KH_L = 6.0
KHI_L = 0.5
LEG_LF = 0.08

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
ACT = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
JNT = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
BID = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
RQP = m.jnt_qposadr[JNT["root"]]
RDF = m.jnt_dofadr[JNT["root"]]
M_TOTAL = float(np.sum(m.body_mass))
DT = m.opt.timestep
LEG_JOINTS = ["L_J_base_L1","L_J_L1_L2","L_J_base_L5","L_J_L3_L5","L_J_L3_L4","L_J_L5_L6",
              "R_J_base_L1","R_J_L1_L2","R_J_base_L5","R_J_L3_L5","R_J_L3_L4","R_J_L5_L6"]
LEG_DOF = [m.jnt_dofadr[JNT[j]] for j in LEG_JOINTS]


def rx_quat(a):
    return np.array([np.cos(a/2), np.sin(a/2), 0.0, 0.0])


def setup_pose(perturb_deg=0.0):
    """webm 姿态: legs=0, base 绕轮心整体 +90°(x轴), 轮子贴地."""
    mujoco.mj_resetData(m, d)
    d.qpos[RQP:RQP+7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    W0 = 0.5*(d.xpos[BID("L_WHEEL")] + d.xpos[BID("R_WHEEL")])
    wz = W0[2]
    B0 = np.array([0.0, 0.0, WHEEL_R - wz])
    W = np.array([W0[0], W0[1], WHEEL_R])
    a = np.pi/2
    c, s = np.cos(a), np.sin(a)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    B1 = W + Rx @ (B0 - W)
    B1[2] -= 0.003
    q = np.array(d.qpos)
    q[RQP:RQP+3] = B1
    q[RQP+3:RQP+7] = rx_quat(a + np.radians(perturb_deg))
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)



_phip = None
def phi_body():
    """机体-水平俯仰角 φ(绕轮轴x, 右手螺旋): body z 的世界z反推."""
    q = d.qpos[RQP+3:RQP+7]
    _, qx, qy, _z = q
    zz = 1.0 - 2.0*(qx*qx + qy*qy)
    return float(np.arcsin(np.clip(-zz, -1.0, 1.0)))

def phi_rate():
    global _phip
    p = phi_body()
    r = 0.0 if _phip is None else (p - _phip)/DT
    _phip = p
    return r

def wheel_mid():
    return 0.5*(d.xpos[BID("L_WHEEL")] + d.xpos[BID("R_WHEEL")])


def lean():
    masses = np.asarray(m.body_mass)
    com = (masses[:, None]*d.xipos).sum(0) / masses.sum()
    w = wheel_mid()
    return float(np.arctan2(com[1]-w[1], com[2]-w[2]))


def measure_L():
    masses = np.asarray(m.body_mass)
    com_z = float((masses*d.xipos[:, 2]).sum() / masses.sum())
    return max(com_z - wheel_mid()[2], 0.02)


def probe():
    setup_pose(0.0)
    v0 = d.qvel[RDF+1]; w0 = d.qvel[RDF+3]
    U, N = 0.5, 80
    for _ in range(N):
        ctrl_legs()
        d.ctrl[ACT["L_wheel_motor"]] = U
        d.ctrl[ACT["R_wheel_motor"]] = U
        apply_damp()
        mujoco.mj_step(m, d)
    T = N * DT
    ay = (d.qvel[RDF+1]-v0)/(U*T)
    aw = (d.qvel[RDF+3]-w0)/(U*T)
    setup_pose(0.0)
    return ay, aw


def build_model():
    L = measure_L()
    ay, aw = probe()
    A = np.array([[1.0, DT, 0.0], [(9.81/L)*DT, 1.0, 0.0], [0.0, 0.0, 1.0]])
    B = np.array([[0.0], [aw*DT], [ay*DT]])
    return A, B, L, ay, aw


def solve_lqr(A, B):
    P = solve_discrete_are(A, B, np.diag(Q_DIAG), np.diag(R_VAL))
    return (np.linalg.solve(np.diag(R_VAL)+B.T@P@B, B.T@P@A)).reshape(-1)


def _measured_params():
    M_cart = float(np.sum([m.body_mass[BID('L_WHEEL')], m.body_mass[BID('R_WHEEL')]]))
    M_body = float(M_TOTAL - M_cart)
    l = measure_L()
    masses = np.asarray(m.body_mass); com = (masses[:, None]*d.xipos).sum(0)/masses.sum()
    Ibody = float(np.sum(masses*((d.xipos[:,1]-com[1])**2 + (d.xipos[:,2]-com[2])**2)))
    return M_cart, M_body, l, Ibody


def test_m_report():
    """test.m 同款 设计/验算(实测参数)"""
    from scipy.linalg import solve_continuous_are
    M, mb, l, I = _measured_params()
    g, b = 9.81, 0.5
    C = 1.0/((M+mb)*(I+mb*l*l) - (mb*l)**2)
    A = C*np.array([[0.,1,0,0],[0,-(I+mb*l*l)*b,-(mb*l)**2*g,0],[0,0,0,1],[0,mb*l*b,(M+mb)*mb*g*l,0]])
    B = C*np.array([[0.],[I+mb*l*l],[0.],[-mb*l]])
    C_out = np.eye(4); D = np.zeros((4,1))
    print("="*62)
    print("test.m 风格设计/验算 (数值为实测/换算)")
    seg = []
    for i in range(m.nbody):
        seg.append("%s=%.3g" % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i), m.body_mass[i]))
    print("  质量来源(XML body_mass): " + ", ".join(seg))
    print(f"  M(两轮)= {M:.3f} kg | m(车体+腿=base+左右连杆)= {mb:.2f} kg | l(CoM-轮轴)= {l:.3f} m | I≈{I:.3f} kg·m² | b={b}")
    print("  状态顺序 [x, x_dot, theta, theta_dot]; 输入=轮接触力 u")
    print("  开环极点:", np.round(np.linalg.eigvals(A), 3))
    print("  能控性矩阵秩:", np.linalg.matrix_rank(np.hstack([B, A@B, A@A@B, A@A@A@B])))
    print("  能观性矩阵秩:", np.linalg.matrix_rank(np.vstack([C_out, C_out@A, C_out@A@A, C_out@A@A@A])))
    Q = np.diag([10.0, 50.0, 5.0e4, 5.0e4]); Rm = np.array([[0.05]])
    P = solve_continuous_are(A, B, Q, Rm)
    Kk = (np.linalg.inv(Rm)@B.T@P).flatten()
    Acl = A - B@Kk.reshape(1, -1)
    print("  Q 对角: [x 10, x_dot 50, theta 5e4, theta_dot 5e4] | R:", 0.05)
    print("  K (顺序 x, x_dot, theta, theta_dot):")
    print("   ", np.round(Kk, 3))
    print("  闭环极点(应全在左半平面):", np.round(np.linalg.eigvals(Acl), 3))
    print("="*62)


def _hip_mid():
    # 髋 = 两曲柄安装体(L_L1/L_L5)原点中点在 base 系 yz 下的方向
    L = 0.5*(d.xpos[BID("L_L1")] + d.xpos[BID("L_L5")])
    Rr = 0.5*(d.xpos[BID("R_L1")] + d.xpos[BID("R_L5")])
    return 0.5*(L + Rr)

def _wheel_mid_world():
    return 0.5*(d.xpos[BID("L_WHEEL")] + d.xpos[BID("R_WHEEL")])

def measure_test_state():
    """按 test.m 定义把仿真测量转成状态 [θ,θ̇,x_b,ẋ_b,φ,φ̇].

    θ : 等效腿杆(髋→轮) 与竖直方向的夹角(取向下竖直为0, 前倾为正)
    φ : 机体与水平面的俯仰角(绕轮轴x, 右手螺旋)
    x_b/ẋ_b : 机体前向位移/速度 (≈车身 +y)
    平衡站姿应约等于 [0,0,0,0,0,0].
    """
    global _phi_prev
    hip = _hip_mid()
    w = _wheel_mid_world()
    v = w - hip                                     # 等效腿杆方向(髋→轮)
    # θ: 杆与"向下竖直"夹角 (向下=0, 前(+y)倾为正)
    th = float(np.arctan2(v[1], -v[2]))
    # φ: 机体绕 x 的俯仰角, 右手螺旋; 由 body z 轴世界 z 分量反推
    q = d.qpos[RQP+3:RQP+7]
    _, qx, qy, qz = q
    zz = 1.0 - 2.0*(qx*qx + qy*qy)                 # body z 轴的世界z
    phi = float(np.arcsin(np.clip(-zz, -1.0, 1.0)))
    xb = float(d.qpos[RQP+1])
    xbd = float(d.qvel[RDF+1])
    # 数值微分求 角速度
    now = np.array([th, phi])
    if _phi_prev is None:
        _phi_prev = np.array([th, phi])
    dth = (th - _phi_prev[0])/DT
    dph = (phi - _phi_prev[1])/DT
    _phi_prev = np.array([th, phi])
    return np.array([th, dth, xb, xbd, phi, dph])

_phi_prev = None

def test_angle_report():
    """打印 按 test.m 定义转换后的当前状态(平衡点应接近0)."""
    s = measure_test_state()
    print("="*62)
    print("角度/坐标转换(test.m 定义) 当前值:")
    print(f"  θ(腿杆-竖直) = {np.degrees(s[0]):+.2f}°   θ̇ = {np.degrees(s[1]):+.2f}°/s")
    print(f"  x_b(前向)   = {s[2]:+.3f} m     ẋ_b = {s[3]:+.3f} m/s")
    print(f"  φ(机体-水平) = {np.degrees(s[4]):+.2f}°   φ̇ = {np.degrees(s[5]):+.2f}°/s")
    print("="*62)

def ctrl_legs():
    d.ctrl[ACT["L_back_drive_pos"]] = 0.0
    d.ctrl[ACT["R_back_drive_pos"]] = 0.0
    d.ctrl[ACT["L_front_drive_pos"]] = 0.0
    d.ctrl[ACT["R_front_drive_pos"]] = 0.0


def apply_damp():
    d.qfrc_applied[:] = 0.0
    if LEG_DAMP > 0.0:
        for df in LEG_DOF:
            d.qfrc_applied[df] = -LEG_DAMP*d.qvel[df]
    if W_DAMP > 0.0:
        for wj in ("L_J_W", "R_J_W"):
            df = m.jnt_dofadr[JNT[wj]]
            d.qfrc_applied[df] = -W_DAMP*d.qvel[df]


setup_pose(0.0)
TH0 = lean()
HREF = float(d.subtree_com[0][2])
A, B, L, ay, aw = build_model()
K = solve_lqr(A, B)
print("="*60)
print("webm姿态 | 平衡倾角(初)=%.2f° | 摆长L=%.3f m | M=%.1fkg" % (np.degrees(TH0), L, M_TOTAL))
print("探针: 每轮+1Nm → 平动acc %+.4f m/s², x角acc %+.4f rad/s²" % (ay, aw))
print("K = [θ %+.2f, θ̇ %+.2f, v %+.2f]" % tuple(K))
print("="*60)
test_m_report()
test_angle_report()

viewer = "--viewer" in sys.argv
if not viewer:
    dur = 12.0; n = int(dur/DT)
    dtc = CTRL_EVERY*DT
    vint = 0.0; it = 0.0; iH = 0.0; legOff = 0.0; uf = 0.0; u = 0.0; hist = []
    for i in range(n):
        t = i*DT
        th = phi_body()          # φ: 机体-水平
        thd = phi_rate()
        v = d.qvel[RDF+1]
        if i % CTRL_EVERY == 0:
            vint += float(np.clip(-v, -3, 3))*dtc
            vint = float(np.clip(vint, -6, 6))
            it += float(np.clip(th, -0.2, 0.2))*dtc
            it = float(np.clip(it, -0.25, 0.25))
            cmd = -K @ np.array([th, thd, v]) + KIV*vint + KIT*it
            uf += U_SMOOTH*(float(cmd)-uf)
            u = float(np.clip(uf, -4.92, 4.92))
            h = float(d.subtree_com[0][2])
            eh = HREF - h
            iH += float(np.clip(eh, -0.05, 0.05))*dtc
            iH = float(np.clip(iH, -0.3, 0.3))
            sq = KH_L*eh + KHI_L*iH
            sq = float(np.clip(sq, -0.7, 0.7))
            legOff += LEG_LF*(sq - legOff)
            for nm, val in (("L_back_drive_pos", legOff), ("R_back_drive_pos", legOff),
                            ("L_front_drive_pos", -legOff), ("R_front_drive_pos", -legOff)):
                d.ctrl[ACT[nm]] = val
        d.ctrl[ACT["L_wheel_motor"]] = u
        d.ctrl[ACT["R_wheel_motor"]] = u
        apply_damp()
        qv = d.qvel; qv[RDF+0]*=.93; qv[RDF+4]*=.95; qv[RDF+5]*=.95
        mujoco.mj_step(m, d)
        hist.append(abs(th))
        if abs(th) > 0.5:
            print(f"t={t:5.2f}s |θ|={np.degrees(abs(th)):.1f}° 发散停止")
            break
        if int(t) != int(t-DT):
            print(f"t={t:5.1f}s φ={np.degrees(th):+7.2f}° φ̇={np.degrees(thd):+7.1f}°/s v={v:+5.2f} u={u:+5.2f}")
    print("-"*60)
    if hist:
        print("俯仰|max|=%.2f°  平均=%.2f°" % (np.degrees(max(hist)), np.degrees(np.mean(hist))))
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as view:
        print("查看器: webm姿态 LQR。ESC退出")
        vint = 0.0; it = 0.0; iH = 0.0; legOff = 0.0; uf = 0.0; u = 0.0; ii = 0
        while view.is_running():
            t = d.time
            th = phi_body(); thd = phi_rate(); v = d.qvel[RDF+1]
            if ii % CTRL_EVERY == 0:
                vint += float(np.clip(-v, -3, 3))*CTRL_EVERY*DT
                vint = float(np.clip(vint, -6, 6))
                it += float(np.clip(th, -0.2, 0.2))*CTRL_EVERY*DT
                it = float(np.clip(it, -0.25, 0.25))
                cmd = -K @ np.array([th, thd, v]) + KIV*vint + KIT*it
                uf += U_SMOOTH*(float(cmd)-uf)
                u = float(np.clip(uf, -4.92, 4.92))
                h = float(d.subtree_com[0][2])
                eh = HREF - h
                iH += float(np.clip(eh, -0.05, 0.05))*CTRL_EVERY*DT
                iH = float(np.clip(iH, -0.3, 0.3))
                sq = KH_L*eh + KHI_L*iH
                sq = float(np.clip(sq, -0.7, 0.7))
                legOff += LEG_LF*(sq - legOff)
                for nm, val in (("L_back_drive_pos", legOff), ("R_back_drive_pos", legOff),
                                ("L_front_drive_pos", -legOff), ("R_front_drive_pos", -legOff)):
                    d.ctrl[ACT[nm]] = val
            d.ctrl[ACT["L_wheel_motor"]] = u
            d.ctrl[ACT["R_wheel_motor"]] = u
            apply_damp()
            qv = d.qvel; qv[RDF+0]*=.93; qv[RDF+4]*=.95; qv[RDF+5]*=.95
            mujoco.mj_step(m, d)
            view.sync()
            ii += 1
