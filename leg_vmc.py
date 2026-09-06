# -*- coding: utf-8 -*-
"""轮腿 VMC —— 移植 Left_Leg (每侧腿) 到 MuJoCo 力矩腿版.

按你 function 的口径:
  phi1 = pi/2 - Joint1 (后曲柄),  phi4 = pi/2 - Joint2 (前曲柄)
  五连杆 FK: l1=l4=active(0.21), l2=l3=slave(0.25), l5=joint_distance(~0)
  解 phi2/phi3/C → phi5 → alpha=phi5-pi/2 → theta=alpha-pitch
  LegLength, Height=LegLength*cos(theta)
  F_Leg=0.5*M*g*cos(theta)+PD(Height);  T1,T2=-Trans_Jacobian*[F_Leg;T_Leg]
轮矩: 控制机体-水平 φ(=pitch) 的 LQR(探针)
Joint↔qpos 映射: 先用 Joint=+q(后), Joint=+q(前) 试(标定可调 PHI_REAR/FRONT 偏置符号)
运行: python leg_vmc.py [--viewer]
"""
import sys
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are

XML="real_wheelleg_ground_torque.xml"
WHEEL_R=0.068
L_ACT=0.21; L_SLV=0.25; L5=0.0      # 五连杆长度(与 test.m 一致)
Q_DIAG=np.array([900.,600.,60.]); R_VAL=np.array([1.8])
KP_H=1200.; KD_H=300.
KP_TH=6.; KD_TH=0.2
TAU_LIM=6.0
# Joint↔qpos 初步映射: Joint = +q  (偏置如需 -pi/2 在这里调)
OFF_R=0.0; OFF_F=0.0

m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
M=float(np.sum(m.body_mass)); DT=m.opt.timestep
ACT={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i):i for i in range(m.nu)}
JNT={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i):i for i in range(m.njnt)}
BID=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
RQP=m.jnt_qposadr[JNT["root"]]; RDF=m.jnt_dofadr[JNT["root"]]
LDOF=[m.jnt_dofadr[JNT[j]] for j in ["L_J_base_L1","L_J_L1_L2","L_J_base_L5","L_J_L3_L5","L_J_L3_L4","L_J_L5_L6",
                                      "R_J_base_L1","R_J_L1_L2","R_J_base_L5","R_J_L3_L5","R_J_L3_L4","R_J_L5_L6"]]

def rx(a): return np.array([np.cos(a/2),np.sin(a/2),0,0])
def setup():
    mujoco.mj_resetData(m,d); d.qpos[RQP:RQP+7]=[0,0,0,1,0,0,0]; mujoco.mj_forward(m,d)
    W0=0.5*(d.xpos[BID("L_WHEEL")]+d.xpos[BID("R_WHEEL")]); B0=np.array([0.,0.,WHEEL_R-W0[2]])
    W=np.array([W0[0],W0[1],WHEEL_R]); a=np.pi/2;c,s=np.cos(a),np.sin(a)
    Rx=np.array([[1,0,0],[0,c,-s],[0,s,c]]); B1=W+Rx@(B0-W); B1[2]-=0.003
    q=np.array(d.qpos); q[RQP:RQP+3]=B1; q[RQP+3:RQP+7]=rx(a); d.qpos[:]=q; d.qvel[:]=0.
    mujoco.mj_forward(m,d)
def phi_body():
    q=d.qpos[RQP+3:RQP+7]; _,x,y,_z=q; zz=1-2*(x*x+y*y)
    return float(np.arcsin(np.clip(-zz,-1,1)))

def leg_fk(side, qr, qf):
    """复刻 Left_Leg 解析五连杆 FK; 返回 (theta,LegLength,Height,T1m,T2m系数已含? ) 与 Tj."""
    phi1 = np.pi/2 + (qr+OFF_R)
    phi4 = np.pi/2 + (qf+OFF_F)
    l1=L_ACT; l4=L_ACT; l2=L_SLV; l3=L_SLV; l5=L5
    xD=l5+l4*np.cos(phi4); yD=l4*np.sin(phi4)
    xB=0+l1*np.cos(phi1); yB=l1*np.sin(phi1)
    BD=np.sqrt((xD-xB)**2+(yD-yB)**2)
    A0=2*l2*(xD-xB); B0=2*l2*(yD-yB); C0=l2**2+BD**2-l3**2
    phi2=2*np.arctan2(B0+np.sqrt(A0**2+B0**2-C0**2), A0+C0)
    xC=l1*np.cos(phi1)+l2*np.cos(phi2); yC=l1*np.sin(phi1)+l2*np.sin(phi2)
    phi3=np.arctan2(yC-yD, xC-xD)
    phi5=np.arctan2(yC, xC-l5/2)
    alpha=phi5-np.pi/2
    pitch=phi_body()
    theta=alpha-pitch
    Leg=np.sqrt((xC-l5/2)**2+yC**2)
    H=Leg*np.cos(theta)
    # 抗劈叉 与 高度PD 在此处由上层调用完成; 这里返回几何量+雅可比所需角
    return dict(phi1=phi1,phi2=phi2,phi3=phi3,phi4=phi4,phi5=phi5,
                theta=theta, Leg=Leg, H=H)

def trans_jacobian(g):
    """Left_Leg 的 Trans_Jacobian(2x2)."""
    f1=f2=np.pi/2-0.0  # phi1 phi4 由 g 提供
    l1=L_ACT; l4=L_ACT
    p1,p2,p3,p4=g["phi1"],g["phi2"],g["phi3"],g["phi4"]
    p5=g["phi5"]; Leg=g["Leg"]
    J=np.array([
      [l1*np.sin(p5-p3)*np.sin(p1-p2)/np.sin(p3-p2), l1*np.cos(p5-p3)*np.sin(p1-p2)/(Leg*np.sin(p3-p2))],
      [l4*np.sin(p5-p2)*np.sin(p3-p4)/np.sin(p3-p2), l4*np.cos(p5-p2)*np.sin(p3-p4)/(Leg*np.sin(p3-p2))]])
    return J

_prevH=None; _prevT=None; _prevL=None
def leg_ctrl(side, H_target, theta_err=0.0):
    """返回该腿 [T1(后),T2(前)] 与 当前 Height/theta."""
    global _prevH,_prevT,_prevL
    qr=d.qpos[JNT[f"{side}_J_base_L1"]]; qf=d.qpos[JNT[f"{side}_J_base_L5"]]
    g=leg_fk(side, qr, qf)
    H=g["H"]; theta=g["theta"]
    Hd=0.0 if _prevH is None else (H-_prevH)/DT
    _prevH=H
    F_Leg=0.5*M*9.81*np.cos(theta) + KP_H*(H_target-H) - KD_H*Hd
    T_Leg = KP_TH*(0-theta_err) - KD_TH*theta_err*0  # 抗劈叉通道暂置0
    T_Leg = 0.0
    J=trans_jacobian(g)
    Tj=J@np.array([F_Leg, T_Leg])
    T1=-Tj[0]; T2=-Tj[1]
    return float(np.clip(T1,-TAU_LIM,TAU_LIM)), float(np.clip(T2,-TAU_LIM,TAU_LIM)), H, theta

def damp():
    d.qfrc_applied[:]=0.
    for df in LDOF: d.qfrc_applied[df]=-1.0*d.qvel[df]
    for wj in ("L_J_W","R_J_W"): d.qfrc_applied[m.jnt_dofadr[JNT[wj]]]=-0.15*d.qvel[m.jnt_dofadr[JNT[wj]]]

def wheel_probe():
    setup(); v0=d.qvel[RDF+1]; w0=d.qvel[RDF+3]; U,N=0.5,60
    for _ in range(N):
        d.ctrl[:]=0.; d.ctrl[ACT["L_wheel_motor"]]=U; d.ctrl[ACT["R_wheel_motor"]]=U; damp(); mujoco.mj_step(m,d)
    ay=(d.qvel[RDF+1]-v0)/(U*N*DT); aw=(d.qvel[RDF+3]-w0)/(U*N*DT)
    setup(); return ay,aw

setup(); L0=leg_fk("L",0.,0.)["Leg"]; H_target=leg_fk("L",0.,0.)["H"]
# 轮 LQR(φ 模型)
ay,aw=wheel_probe(); setup()
A=np.array([[1.,DT,0.],[(9.81/max(L0,0.1))*DT,1.,0.],[0.,0.,1.]])
B=np.array([[0.],[aw*DT],[ay*DT]])
P=solve_discrete_are(A,B,np.diag(Q_DIAG),np.diag(R_VAL))
K=(np.linalg.solve(np.diag(R_VAL)+B.T@P@B,B.T@P@A)).reshape(-1)
print("="*60)
print("leg_vmc | M=%.1f  L0=%.3f  H0=%.3f | wheel K=[φ %.1f φ̇ %.1f v %.1f]"
      %(M,L0,H_target,K[0],K[1],K[2]))
print("="*60)

viewer="--viewer" in sys.argv
if not viewer:
    n=int(5.0/DT); vint=0.;uf=0.;u=0.; hist=[]; ii=0; phip=None
    _prevH=None; _prevT=None; _prevL=None
    for i in range(n):
        t=i*DT
        ph=phi_body()
        phd = 0.0 if phip is None else (ph-phip)/DT
        phip = ph
        v=d.qvel[RDF+1]
        if i%5==0:
            vint+=float(np.clip(-v,-3,3))*5*DT; vint=float(np.clip(vint,-6,6))
            cmd=-K@np.array([ph, phd, v])+1.0*vint
            uf+=0.12*(float(cmd)-uf); u=float(np.clip(uf,-4.92,4.92))
            d.ctrl[:]=0.
            for side in ("L","R"):
                T1,T2,Hh,th=leg_ctrl(side, H_target)
                d.ctrl[ACT[f"{side}_back_drive_pos"]]=T1
                d.ctrl[ACT[f"{side}_front_drive_pos"]]=T2
            d.ctrl[ACT["L_wheel_motor"]]=u; d.ctrl[ACT["R_wheel_motor"]]=u
        damp()
        qv=d.qvel; qv[RDF+0]*=.93; qv[RDF+4]*=.95; qv[RDF+5]*=.95
        mujoco.mj_step(m,d)
        hist.append(abs(ph))
        if abs(ph)>0.5:
            print(f"t={t:5.2f} φ发散 {np.degrees(abs(ph)):.0f}°"); break
        if abs(t-round(t,1))<0.005:
            print(f"t={t:4.1f}s φ={np.degrees(ph):+6.1f}° u={u:+.2f} T1={d.ctrl[ACT['L_back_drive_pos']]:+.2f} T2={d.ctrl[ACT['L_front_drive_pos']]:+.2f}")
    if hist: print("φ max=%.2f° avg=%.2f°"%(np.degrees(max(hist)),np.degrees(np.mean(hist))))
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m,d) as view:
        print("leg_vmc viewer ESC退出")
        vint=0.;uf=0.;u=0.;ii=0;phip=None;_prevH=None
        while view.is_running():
            t=d.time; ph=phi_body()
            phd=0.0 if phip is None else (ph-phip)/DT
            phip=ph
            v=d.qvel[RDF+1]
            if ii%5==0:
                vint+=float(np.clip(-v,-3,3))*5*DT; vint=float(np.clip(vint,-6,6))
                cmd=-K@np.array([ph,phd,v])+1.0*vint
                uf+=0.12*(float(cmd)-uf); u=float(np.clip(uf,-4.92,4.92))
                d.ctrl[:]=0.
                for side in ("L","R"):
                    T1,T2,Hh,th=leg_ctrl(side,H_target)
                    d.ctrl[ACT[f"{side}_back_drive_pos"]]=T1
                    d.ctrl[ACT[f"{side}_front_drive_pos"]]=T2
                d.ctrl[ACT["L_wheel_motor"]]=u; d.ctrl[ACT["R_wheel_motor"]]=u
            damp()
            qv=d.qvel; qv[RDF+0]*=.93; qv[RDF+4]*=.95; qv[RDF+5]*=.95
            mujoco.mj_step(m,d); view.sync(); ii+=1
