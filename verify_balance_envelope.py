import json
import numpy as np
import mujoco
from balance_control import Robot,Controller,ROOT,DT
rows=json.loads((ROOT/'balance_gains.json').read_text())['rows'];results=[]
for label,height,seconds in [('low',.27,10),('high',.35,10),('velocity',.30,10)]:
    r=Robot();c=Controller(r,rows,height);d=r.d;c.initialize(d);stats=[]
    for i in range(int(seconds/DT)):
        if label=='velocity':c.velocity=.10 if 2<d.time<6 else 0
        x=c.control(d);mujoco.mj_step(r.m,d)
        if i%20==0:stats.append(x)
    v=dict(case=label,max_pitch=float(max(abs(x['state'][4]) for x in stats)),max_gap=max(x['gap'] for x in stats),height_error=max(max(abs(np.array(x['height'])-height)) for x in stats),chassis_contact=max(x['chassis_contacts'] for x in stats),min_wheel_contact=min(min(x['wheel_contacts']) for x in stats[20:]),final=stats[-1])
    assert v['max_pitch']<.1 and v['height_error']<.015 and not v['chassis_contact'] and v['min_wheel_contact']>0,v
    results.append(v);print(label,'passed',v['height_error'],flush=True)
(ROOT/'balance_envelope_checks.json').write_text(json.dumps(results,indent=2))
