import json
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from advanced_control import SensorRobot,HybridController,load_gains,step,ROOT
results={};images=[]
for name in ['slope','launch','course']:
    r=SensorRobot(terrain=name);c=HybridController(r,load_gains(),assist=False);c.initialize(r.d)
    for _ in range(200):step(r,c)
    probes=[]
    for x,y in ([(0,1.6),(0,2.5),(0,3.4)] if name=='slope' else [(0 if name=='launch' else 1.5,1.2)]):
        gid=np.array([-1],dtype=np.int32);dist=mujoco.mj_ray(r.m,r.d,np.array([x,y,2.]),np.array([0.,0.,-1.]),None,True,r.base,gid)
        probes.append(dict(x=x,y=y,z=2-float(dist),geom=r.m.geom(int(gid[0])).name))
    expected=[.05,.10,.05] if name=='slope' else [.10]
    assert np.allclose([p['z'] for p in probes],expected,atol=1e-6)
    results[name]=dict(objects=r.terrain_names,probes=probes)
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,2,.15];cam.distance=6;cam.azimuth=130;cam.elevation=-35
    with mujoco.Renderer(r.m,height=480,width=640) as render:
        render.update_scene(r.d,camera=cam);im=Image.fromarray(render.render());ImageDraw.Draw(im).text((15,15),name,fill='white');images.append(im)
canvas=Image.new('RGB',(640*3,480));[canvas.paste(im,(640*i,0)) for i,im in enumerate(images)]
canvas.save(ROOT/'terrain_preview.png');(ROOT/'terrain_checks.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
