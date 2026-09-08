"""Initial-condition get-up tests. No state or external-force writes after t=0 setup."""
import sys,json
import numpy as np
import mujoco
from advanced_control import *

def run(case):
    r=SensorRobot();c=HybridController(r,load_gains());c.initialize(r.d);d=r.d
    if case!='collapsed':
        angle={'front':np.pi/2,'back':-np.pi/2,'inverted':np.pi,'side':np.pi/2}[case]
        axis=np.array([0,1,0]) if case=='side' else np.array([-1,0,0])
        d.qpos[3:7]=np.r_[np.cos(angle/2),axis*np.sin(angle/2)];d.qpos[2]=1
        mujoco.mj_forward(r.m,d);minimum=10
        for name in ['chassis_col','L_wheel_col','R_wheel_col']:
            g=r.m.geom(name).id;R=d.geom_xmat[g].reshape(3,3)
            extent=np.abs(R[2])@r.m.geom_size[g] if name=='chassis_col' else r.radius*np.sqrt(1-R[2,2]**2)+r.m.geom_size[g,1]*abs(R[2,2])
            minimum=min(minimum,d.geom_xpos[g,2]-extent)
        d.qpos[2]-=minimum-.001;mujoco.mj_forward(r.m,d)
        c.disable(d,'initial fallen pose')
    logs=[];maxturn=0.;ctrlmax=0.;zero_wrench=True;mutation=False
    for i in range(30000):
        c.submit(Command(stop=case=='collapsed' and i==1000,recover=case=='collapsed' and i==2500))
        before=d.qpos.copy();bv=d.qvel.copy();t=d.time;x=c.control(d)
        mutation |= not(np.array_equal(before,d.qpos) and np.array_equal(bv,d.qvel) and d.time==t)
        zero_wrench &= not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
        ctrlmax=max(ctrlmax,float(max(abs(d.ctrl))))
        mujoco.mj_step(r.m,d)
        driver=getattr(c,'recovery_driver',None)
        if driver:maxturn=max(maxturn,float(max(abs(driver.turns))))
        if driver and d.time-c.mode_time>3 and (c.mode=='GROUND' or (c.mode=='DISABLED' and c.reason=='motor recovery failed; manual retry required')):break
        if i%20==0:logs.append(dict(t=float(d.time),phase=driver.phase if driver else '',**x))
    success=c.mode=='GROUND' and all(x['contact']) and not x['chassis_contact'] and abs(x['state'][4])<.1
    result=dict(case=case,seconds=float(d.time),recovery_history=getattr(getattr(c,'recovery_driver',None),'history',[]),success=success,max_hip_rotation_deg=float(np.degrees(maxturn)),max_torque=ctrlmax,controller_mutated_state=mutation,zero_external_forces=zero_wrench,events=c.events,final=x)
    (ROOT/f'recovery_{case}.json').write_text(json.dumps(dict(summary=result,samples=logs),indent=2));print(json.dumps(result),flush=True)
    assert not mutation and zero_wrench and ctrlmax<=40.0001
    return result
if __name__=='__main__':
    results=[run(n) for n in sys.argv[1:] or ['collapsed','front','back','inverted','side']]
    assert all(x['success'] if x['case']!='side' else x['final']['mode']=='DISABLED' for x in results)
