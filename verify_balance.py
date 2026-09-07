"""Independent kinematics/power checks and offscreen rendering."""
import json
import numpy as np
import mujoco
from balance_control import Robot, Controller, ROOT
r=Robot();checks=[]
for h in [.27,.30,.35]:
    z=np.array([.08,0,0,h]);d=r.pose(z);mujoco.mj_forward(r.m,d)
    g=r.leg(d,'L');eps=1e-5
    for side in 'LR':
        active=d.qpos[r.qa[side][[0,2]]];enc=r.forward_leg(*active,side)
        assert abs(enc['h']-h)<1e-7 and abs(enc['alpha']-.08)<1e-7
    # Differentiate the inverse map, independent of the loop velocity elimination.
    inv=np.zeros((2,2))
    for k in range(2):
        dh=eps if k==0 else 0;da=eps if k==1 else 0
        a=r.inverse(h-dh,.08-da);b=r.inverse(h+dh,.08+da)
        inv[:,k]=(b[[0,2]]-a[[0,2]])/(2*eps)
    err=np.max(np.abs(g['J']@inv-np.eye(2)))
    assert err<1e-4,(h,err)
    dq=np.array([.23,-.13]);force=np.array([100.,-3.])
    power=abs((g['J'].T@force)@dq-force@(g['J']@dq))
    assert power<1e-10
    assert r.gap(d)<1e-7
    checks.append(dict(length=h,jacobian_identity_error=float(err),virtual_work_error=float(power),site_gap=r.gap(d)))
d=mujoco.MjData(r.m);r.kin(d);phi=r.phi_angles(d,'L')
assert 0<phi[1]<np.pi/2<phi[0]<np.pi
for s in 'LR':
    for i,k in [(2,0),(0,1)]:
        p=r.phi_angles(d,s);d.qpos[r.qa[s][i]]+=.01
        assert abs(r.phi_angles(d,s)[k]-p[k]-.01)<1e-10
        d.qpos[r.qa[s][i]]-=.01
rows=json.loads((ROOT/'balance_gains.json').read_text())['rows'];c=Controller(r,rows);c.initialize(d)
for _ in range(1000):c.control(d);mujoco.mj_step(r.m,d)
r.m.vis.global_.offwidth=1200;r.m.vis.global_.offheight=900
with mujoco.Renderer(r.m,height=900,width=1200) as renderer:
    cam=mujoco.MjvCamera();cam.lookat[:]=d.xpos[r.base];cam.distance=1.5;cam.azimuth=35;cam.elevation=-18
    renderer.update_scene(d,camera=cam)
    from PIL import Image
    Image.fromarray(renderer.render()).save(ROOT/'balance_preview.png')
report=dict(kinematics=checks,reference_phi1_phi4_deg=np.degrees(phi).tolist(),mass=float(r.m.body_mass.sum()))
(ROOT/'balance_kinematics_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
