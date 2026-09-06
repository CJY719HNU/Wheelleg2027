# -*- coding: utf-8 -*-
"""左右腿 phi/theta 对齐 台架接口.

机身固定(base 焊死,轮子悬空,重力 0),手动调左右四条曲柄,
实时看每侧换算出的 phi/theta/LegLength/Height, 用于你自己对齐 L/R.

用法(在终端运行):
    python leg_align.py --viewer     # 带查看器看几何
    python leg_align.py              # 纯命令行

按键(大小写同效, 每按一下 step 弧度):
  a / A   左腿 后曲柄 -1
  s / S   左腿 后曲柄 +1
  z / Z   左腿 前曲柄 -1
  x / X   左腿 前曲柄 +1
  d / D   右腿 后曲柄 -1
  c / C   右腿 后曲柄 +1
  f / F   右腿 前曲柄 -1
  v / V   右腿 前曲柄 +1
  p / P   打印一行当前左右对比
  q / Q   退出
"""
import sys
import numpy as np
import mujoco

XML = "real_wheelleg_fixed_base.xml"     # 固定基座 = 机身架起
STEP = 0.05
LA, LS, L5 = 0.21, 0.25, 0.0              # active/slave/hip距(等效, 可按需改)
# phi 约定: True -> pi/2 - q ; False -> pi/2 + q
USE_PHIMINUS = True

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
ACT = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
JNT = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(m.njnt)}
BID = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
joints = {s: {"back": JNT[f"{s}_J_base_L1"], "front": JNT[f"{s}_J_base_L5"]} for s in ("L", "R")}
actu = {s: {"back": ACT[f"{s}_back_drive_pos"], "front": ACT[f"{s}_front_drive_pos"]} for s in ("L", "R")}


def set_joints_and_settle(q: dict):
    for s in ("L", "R"):
        d.ctrl[actu[s]["back"]] = q[s]["back"]
        d.ctrl[actu[s]["front"]] = q[s]["front"]
    for _ in range(300):                 # 让闭链/伺服 settle
        mujoco.mj_step(m, d)


def phi_of(q, s=+1 if not USE_PHIMINUS else -1):
    return np.pi / 2 + s * q


def analytic_leg(qr, qf):
    """解析 pantograph(hip 距 0): 返回 dict(phi1,phi2,phi3,phi4, Cx,Cy, Leg, alpha, theta_abs)."""
    p1 = phi_of(qr)
    p4 = phi_of(qf)
    xD = L5 + LA*np.cos(p4); yD = LA*np.sin(p4)
    xB = LA*np.cos(p1); yB = LA*np.sin(p1)
    BD = np.hypot(xD-xB, yD-yB)
    A0 = 2*LS*(xD-xB); B0 = 2*LS*(yD-yB); C0 = LS**2 + BD**2 - LS**2
    disc = max(A0*A0 + B0*B0 - C0*C0, 0.0)
    p2 = 2*np.arctan2(B0 + np.sqrt(disc), A0 + C0)
    Cx = LA*np.cos(p1) + LS*np.cos(p2)
    Cy = LA*np.sin(p1) + LS*np.sin(p2)
    p3 = np.arctan2(Cy - yD, Cx - xD)
    p5 = np.arctan2(Cy, Cx - L5/2)
    alpha = p5 - np.pi/2
    Leg = np.hypot(Cx - L5/2, Cy)
    return dict(phi1=p1, phi2=p2, phi3=p3, phi4=p4, Cx=Cx, Cy=Cy,
                alpha=alpha, Leg=Leg)


def measure_wheel_hip(side):
    """MuJoCo 实测: 髋(两曲柄体原点平均)->轮, 在 base 系 (x,y,z)."""
    hip = 0.5*(d.xpos[BID(f"{side}_L1")] + d.xpos[BID(f"{side}_L5")])
    W = d.xpos[BID(f"{side}_WHEEL")]
    Rb = d.xmat[BID("base")].reshape(3, 3)
    return Rb.T @ (W - hip)


def report(q, tag=""):
    print("\n" + tag)
    print(f"{'侧':<3}{'q后':>7}{'q前':>7}"
          f"{'phi1':>7}{'phi4':>7}"
          f"{'LegF':>6}{'HeightF':>8}"
          f"{'Leg测':>6}{' 髋轮Y':>8}{'髋轮Z':>8}")
    for s in ("L", "R"):
        qr, qf = q[s]["back"], q[s]["front"]
        a = analytic_leg(qr, qf)
        v = measure_wheel_hip(s)
        print(f"{s:<3}{qr:+7.3f}{qf:+7.3f}"
              f"{np.degrees(a['phi1']):+7.2f}{np.degrees(a['phi4']):+7.2f}"
              f"{a['Leg']:6.3f}{a['Leg']*np.cos(a['alpha']):8.3f}"
              f"{np.linalg.norm(v):6.3f}{v[1]:+8.3f}{v[2]:+8.3f}")
    print(" 注: 表头 = 解析F(0.21/0.25) vs MuJoCo实测髋轮矢量; 对齐目标: LegF≈Leg测 且 L/R 两行一致")


def main():
    if "--viewer" in sys.argv:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(m, d) as view:
            q = {"L": {"back": 0.0, "front": 0.0}, "R": {"back": 0.0, "front": 0.0}}
            report(q, "初始(左/右应一致)")
            print("按键... 在终端按(窗口也开着); 输入 q 退出")
            while view.is_running():
                cmd = input("cmd> ").strip().lower()
                if cmd in ("q", "quit"):
                    break
                mp = {"a": ("L", "back", -1), "s": ("L", "back", +1),
                      "z": ("L", "front", -1), "x": ("L", "front", +1),
                      "d": ("R", "back", -1), "c": ("R", "back", +1),
                      "f": ("R", "front", -1), "v": ("R", "front", +1)}
                if cmd in mp:
                    s, j, dr = mp[cmd]
                    q[s][j] = min(1.2, max(-1.2, q[s][j] + dr*STEP))
                    set_joints_and_settle(q)
                    report(q, f"after {cmd}")
                elif cmd == "p":
                    report(q, "report")
        return
    # 纯命令行
    q = {"L": {"back": 0.0, "front": 0.0}, "R": {"back": 0.0, "front": 0.0}}
    set_joints_and_settle(q)
    report(q, "初始")
    print("按键帮助见文件头 (q 退出)")
    while True:
        cmd = input("cmd> ").strip().lower()
        if cmd in ("q", "quit"):
            break
        mp = {"a": ("L", "back", -1), "s": ("L", "back", +1),
              "z": ("L", "front", -1), "x": ("L", "front", +1),
              "d": ("R", "back", -1), "c": ("R", "back", +1),
              "f": ("R", "front", -1), "v": ("R", "front", +1)}
        if cmd in mp:
            s, j, dr = mp[cmd]
            q[s][j] = min(1.2, max(-1.2, q[s][j] + dr*STEP))
            set_joints_and_settle(q)
            report(q, f"after {cmd}")
        elif cmd == "p":
            report(q, "report")


if __name__ == "__main__":
    main()
