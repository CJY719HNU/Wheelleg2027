"""Measure speed, deep crouch/jump and the 20 cm launch ramp without recovery aids."""
import json,sys
import numpy as np
from advanced_control import *

results=[]
for name in sys.argv[1:] or ['speed','low','jump','launch']:
    r=SensorRobot(terrain='launch' if name.startswith('launch') else 'flat');c=HybridController(r,load_gains(),height=.35 if name=='launch35' else .30,assist=False);c.initialize(r.d)
    if name.startswith('jump') and name[4:]:c.jump_force=float(name[4:])
    # A runway is an initial condition, never a live recovery/reset.
    if name.startswith('launch'):r.d.qpos[1]-=7.;mujoco.mj_forward(r.m,r.d)
    samples=[];vmax=0.;clearmax=0.;minh=1.;peak=0.
    for i in range(14000):
        cmd=Command(speed=2. if (name=='speed' or name.startswith('launch')) and i<9500 else 0.,jump=name.startswith('jump') and i==1000)
        if name=='low':c.target_height=.20 if i<7000 else .30
        c.submit(cmd);info=step(r,c)
        vmax=max(vmax,float(r.d.qvel[1]));clear=min(r.d.xpos[r.m.body(s+'_WHEEL').id,2]-r.radius for s in 'LR')
        clearmax=max(clearmax,float(clear));minh=min(minh,*info['height']);peak=max(peak,float(max(abs(r.d.ctrl))))
        if i%20==0:samples.append(dict(t=float(r.d.time),true_v=float(r.d.qvel[1]),clearance=float(clear),y=float(r.d.qpos[1]),**info))
        if c.mode=='DISABLED':break
    stable=c.mode=='GROUND' and all(info['contact']) and not info['chassis_contact'] and abs(info['velocity_est'])<.05 and abs(info['state'][4])<.08 and abs(info['state'][0])<.25
    if name=='speed':stable=stable and abs(np.mean([s['true_v'] for s in samples if 7<s['t']<9])-2.)<.05
    if name=='low':stable=stable and .18<minh<.21
    if name.startswith('jump'):stable=stable and clearmax>.10
    result=dict(case=name,passed=bool(stable),final_mode=c.mode,reason=c.reason,max_speed=vmax,max_clearance=clearmax,min_length=minh,peak_torque=peak,events=c.events,final=info)
    (ROOT/f'motion_{name}.json').write_text(json.dumps(dict(summary=result,samples=samples),indent=2));print(json.dumps(result),flush=True)
    results.append(bool(stable))
raise SystemExit(0 if all(results) else 1)
