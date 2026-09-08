"""Headless scenario runner; failures remain in reports, never silently relabelled."""
import argparse,json
import numpy as np
import mujoco
from advanced_control import SensorRobot,HybridController,load_gains,step,DT,ROOT,Command,attitude

def run(name,seconds=10,source='fused'):
    r=SensorRobot(terrain_angle=.05 if name=='terrain' else 0);c=HybridController(r,load_gains(),assist=name in ['recovery','fall'],contact_source=source)
    c.initialize(r.d);d=r.d
    stats=[];peak={'torque':0.,'pitch':0.,'roll':0.,'wheel_speed':0.,'gap':0.};air_steps=0;zero_air=True;zero_disabled=True;finite=True
    velocity_errors=[];wheel_clearance=[]
    for i in range(int(seconds/DT)):
        t=d.time;cmd=Command();external=np.zeros(6)
        if name in ['drive','turn','noise'] and 1<t<6:cmd.speed=.10
        if name=='turn' and 2<t<5:cmd.yaw_rate=.4
        if name=='jump' and i==1500:cmd.jump=True
        if name=='lift' and 2<t<2.15:external[2]=280
        if name=='lift_high' and 2<t<2.30:external[2]=360
        if name=='push' and 2<t<2.2:external[1]=30
        if name=='fall' and 2<t<2.3:external[3]=-90
        if name=='recovery' and i==1500:cmd.stop=True
        if name=='recovery' and i==3500:cmd.recover=True
        if name=='estop' and i>=1500:cmd.stop=True
        if name=='height':c.target_height=.34 if 1<t<5 else .28
        c.submit(cmd)
        if name=='noise':
            d.sensor('imu_acc').data[:]+=np.array([.08,.12,0.])+np.random.default_rng(i).normal(0,.08,3)
        info=step(r,c,external)
        finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        for key,val in [('torque',max(abs(d.ctrl))),('pitch',abs(info['state'][4])),('roll',abs(info['roll'])),('wheel_speed',info['wheel_speed']),('gap',info['gap'])]:peak[key]=max(peak[key],val)
        if info['mode']=='AIR':
            air_steps+=1;zero_air &= bool(np.all(d.ctrl[[r.acts['L'][2],r.acts['R'][2]]]==0))
        if info['mode']=='DISABLED':zero_disabled &= bool(np.all(d.ctrl==0))
        hpos=np.mean([r.leg(d,s,False)['hip'] for s in 'LR'],axis=0)
        jp,_=r.jac(d,hpos,r.base);true_v=float(attitude(d,r)[1]@(jp@d.qvel))
        if t>.5 and info['mode']=='GROUND':velocity_errors.append(info['velocity_est']-true_v)
        clear=min(d.xpos[r.m.body(s+'_WHEEL').id,2]-r.radius for s in 'LR');wheel_clearance.append(float(clear))
        if i%10==0:stats.append(dict(t=float(t),true_v=true_v,clearance=float(clear),**info))
    summary=dict(case=name,source=source,seconds=seconds,peak=peak,finite=finite,air_seconds=air_steps*DT,
                 zero_air=zero_air,zero_disabled=zero_disabled,max_clearance=max(wheel_clearance),
                 velocity_rmse=float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else None,
                 final=info,events=c.events)
    tests={'finite':finite,'air_wheels_zero':zero_air,'disabled_zero':zero_disabled,'torque_limited':peak['torque']<=40.0001}
    if name not in ['estop','lift_high']:tests['returns_ground']=info['mode']=='GROUND';tests['final_upright']=abs(info['state'][4])<.08 and abs(info['roll'])<.08
    if name=='jump':tests['real_takeoff']=max(wheel_clearance)>.003 and air_steps>20
    if name=='lift':tests['lift_detected']=air_steps>50
    if name in ['recovery','fall']:tests['motor_recovery_attempted']=any(x['to']=='RECOVERY' for x in c.events)
    if name=='terrain':tests['terrain_detected']=abs(info['terrain'])>.01;tests['unequal_lengths']=abs(info['height'][0]-info['height'][1])>.01
    if name=='lift_high':tests['safe_large_drop']=info['mode']=='DISABLED' and peak['wheel_speed']<65
    if name=='estop':tests['latched']=info['mode']=='DISABLED'
    if name in ['stand','drive','turn','height','noise','push']:tests['no_unexpected_disable']=not any(x['to']=='DISABLED' for x in c.events)
    if name in ['drive','turn','noise']:tests['velocity_error']=summary['velocity_rmse']<.05
    tests={k:bool(v) for k,v in tests.items()};summary['tests']=tests;summary['passed']=all(tests.values())
    (ROOT/('advanced_'+name+'_'+source+'.json')).write_text(json.dumps(dict(summary=summary,samples=stats),indent=2))
    print(json.dumps(summary),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='*',default=['stand','drive','turn','jump','lift','height','terrain','recovery','estop','noise']);p.add_argument('--seconds',type=float,default=10);p.add_argument('--source',default='fused');args=p.parse_args()
    reports=[run(name,args.seconds,args.source) for name in args.cases]
    raise SystemExit(0 if all(x['passed'] for x in reports) else 1)
