"""Interface and protection tests; no GUI or real controller required."""
import json
import numpy as np
import mujoco
from advanced_control import SensorRobot,HybridController,load_gains,step,DT,ROOT,VelocityKF,Command
from gamepad_control import XInput,Gamepad,deadzone
checks={}
pad=XInput.__new__(XInput);pad.previous=0
p=Gamepad();p.lx=-32768;p.ly=32767;p.buttons=0x0100
cmd=pad.decode(p);assert cmd.speed==2. and cmd.yaw_rate==.5
p.buttons=0;assert pad.decode(p).speed==0
p.buttons=0x1100;assert pad.decode(p).jump;assert not pad.decode(p).jump
p.buttons=0x2110;cmd=pad.decode(p);assert cmd.stop and cmd.recover
assert deadzone(7849)==0 and deadzone(-7849)==0 and deadzone(32767)==1
checks['gamepad_deadzone_deadman_axes_edges']=True
r=SensorRobot();c=HybridController(r,load_gains(),assist=False);c.initialize(r.d)
for i in range(1000):c.submit(Command(speed=.1));step(r,c)
for i in range(1600):step(r,c)
assert abs(c.speed)<1e-8 and abs(c.turn)<1e-8
checks['stale_command_ramps_to_zero']=True
c.submit(Command(speed=.2));step(r,c);c.submit(Command(connected=False))
for i in range(1000):step(r,c)
assert abs(c.speed)<1e-8;checks['disconnect_stops_command']=True
c.submit(Command(stop=True,recover=True));info=step(r,c)
assert c.mode=='DISABLED' and np.all(r.d.ctrl==0) and not c.recover_requested
checks['emergency_overrides_recovery']=True
c.initialize(r.d);r.d.qvel[r.va['L'][6]]=70;mujoco.mj_forward(r.m,r.d);c.control(r.d)
assert c.mode=='DISABLED' and c.reason=='overspeed' and np.all(r.d.ctrl==0)
checks['overspeed_zero_torque']=True
c.initialize(r.d);c.saturation_time=.4;c.control(r.d)
assert c.mode=='GROUND'
c.saturation_time=c.torque_saturation_timeout+.001;c.control(r.d)
assert c.mode=='DISABLED' and c.reason=='persistent torque saturation' and np.all(r.d.ctrl==0)
checks['torque_saturation_grace_and_timeout']=True
c.initialize(r.d);c.submit(Command(speed=float('nan')));assert c.mode=='DISABLED'
checks['invalid_command_disables']=True
c.initialize(r.d);r.d.sensor('imu_acc').data[0]=float('nan');c.control(r.d)
assert c.mode=='DISABLED' and np.all(r.d.ctrl==0);checks['invalid_sensor_disables']=True
c.initialize(r.d);r.d.qvel[0]=float('nan');info=step(r,c)
assert c.mode=='DISABLED' and np.all(r.d.ctrl==0);checks['invalid_physics_not_stepped']=True
k=VelocityKF();heading=np.array([0.,1.,0.]);lateral=np.array([1.,0.,0.])
for _ in range(100):k.update(np.zeros(3),heading,lateral,10.,False)
assert np.linalg.norm(k.x)==0 and not k.accepted;checks['airborne_encoder_ignored']=True
for _ in range(1000):k.update(np.zeros(3),heading,lateral,10.,True)
assert abs(k.x[1])<.1;checks['large_slip_innovation_rejected']=True
(ROOT/'advanced_interface_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
