"""Independent, static terrain assets. Robot XML and inertial parameters stay unchanged."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent

def add_terrain(root,preset='flat',path=None):
    if path is None and preset=='flat':return []
    source=Path(path) if path else ROOT/'terrains'/f'{preset}.json'
    root.find("./worldbody/geom[@name='ground']").set('size','8 8 .05')
    data=json.loads(source.read_text(encoding='utf-8-sig'));world=root.find('worldbody');assets=root.find('asset');names=[]
    for i,obj in enumerate(data['objects']):
        name=f'terrain_{i}_{obj.get("name","obstacle")}'
        xyz=obj['pos'];width,run,height=obj['size']
        if not all(math.isfinite(float(x)) for x in [*xyz,width,run,height]) or min(width,run,height)<=0:raise ValueError('Invalid terrain dimensions')
        attrs=dict(name=name,contype='1',conaffinity='2',friction='1 .005 .0001',condim='3',rgba=obj.get('rgba','.45 .55 .65 1'))
        if obj['type']=='box':
            ET.SubElement(world,'geom',type='box',pos=' '.join(map(str,xyz)),size=f'{width/2} {run/2} {height/2}',**attrs)
        elif obj['type']=='ramp':
            # Triangular prism: local y=0 starts flush, y=run ends at height (reverse for descent).
            low,high=(height,0) if obj.get('descending',False) else (0,height)
            pts=[]
            for x in [-width/2,width/2]:pts.extend([(x,0,-.03),(x,run,-.03),(x,run,high),(x,0,low)])
            mesh=name+'_mesh';ET.SubElement(assets,'mesh',name=mesh,vertex=' '.join(str(v) for p in pts for v in p))
            ET.SubElement(world,'geom',type='mesh',mesh=mesh,pos=' '.join(map(str,xyz)),**attrs)
        else:raise ValueError('Terrain type must be box or ramp')
        names.append(name)
    return names
