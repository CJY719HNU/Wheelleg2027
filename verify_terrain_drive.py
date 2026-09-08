import json,sys
import numpy as np
import mujoco
from advanced_control import *
for name in sys.argv[1:] or ['slope','launch']:
    r=SensorRobot(terrain=name);c=HybridController(r,load_gains(),assist=False);c.initialize(r.d);samples=[]
    for i in range(26000):
        c.submit(Command(speed=.2 if i<23000 else 0.));x=step(r,c)
        if i%50==0:samples.append(dict(t=r.d.time,y=float(r.d.qpos[1]),**x))
    result=dict(terrain=name,final_mode=c.mode,y=float(r.d.qpos[1]),events=c.events,final=x)
    (ROOT/f'terrain_drive_{name}.json').write_text(json.dumps(dict(summary=result,samples=samples),indent=2));print(json.dumps(result),flush=True)
