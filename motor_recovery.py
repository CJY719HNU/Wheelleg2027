"""Actuator-only recovery phases; unwrapped hip targets support full revolutions."""
import numpy as np
from balance_control import DT,wrap

class MotorRecovery:
    def __init__(self,r,d,controller):
        self.controller=controller;self.r=r;self.start=float(d.time);self.phase='FOLD';self.phase_start=d.time;self.good=0.
        self.reference=np.array([r.leg(d,s)['alpha'] for s in 'LR'])
        self.qstart=[d.qpos[r.qa[s][:6]].copy() for s in 'LR'];self.seeds=[q.copy() for q in self.qstart]
        self.targets=np.array([q[[0,2]] for q in self.qstart]);self.last_plan=-1.
        self.failure='';self.repacks=0;self.total_start=self.start;self.travel=0.;self.turns=np.zeros(2);self.history=[]
        self.alpha_command=self.reference.copy();self.length_command=float(np.mean([r.leg(d,s)['h'] for s in 'LR']))
    def switch(self,phase,d):
        self.history.append(dict(t=float(d.time),phase=phase));self.phase=phase;self.phase_start=d.time
    def repack(self,d):
        count=self.repacks+1;origin=self.total_start;history=self.history.copy()
        self.__init__(self.r,d,self.controller)
        self.repacks=count;self.total_start=origin;self.history=history+[dict(t=float(d.time),phase='REPACK')]
    def fail(self,reason):
        self.failure=reason;return 'failed'
    def update(self,d,legs,pitch,roll,grounded):
        r=self.r;elapsed=d.time-self.start;contact_force,contact,_=r.contact_forces(d);d.ctrl[:]=0
        if abs(roll)>.75:return self.fail('sideways posture outside recovery envelope')
        if d.time-self.total_start>24 or self.repacks>2:return self.fail('recovery time or repack limit')
        if r.gap(d)>.006:return self.fail('closed-chain gap too large')
        hip_speed=max(abs(d.qvel[r.va[s][[0,2]]]).max() for s in 'LR')
        wheel_speed=max(abs(d.qvel[r.va[s][6]]) for s in 'LR')
        if (hip_speed>22 or wheel_speed>65) and self.phase!='SETTLE':self.switch('SETTLE',d)
        if self.phase=='SETTLE':
            if d.time-self.phase_start>1 and abs(pitch)<.2 and np.linalg.norm(d.sensor('imu_gyro').data)<1 and hip_speed<2:
                self.repack(d)
            elif d.time-self.phase_start>4:return self.fail('overspeed settling failed')
            return 'running'
        if self.phase=='RIGHT' and d.time-self.phase_start>1 and abs(pitch)<.12 and r.contact_forces(d)[2] and max(abs(g['rate'][0]) for g in legs)<.2:
            self.repack(d);return 'running'
        if self.phase=='FOLD':
            h=.10
            if max(g['h'] for g in legs)<.125:
                self.sweep_start=self.alpha_command.copy()
                self.sweep_end=self.alpha_command+2*np.pi+np.mod(-pitch-self.alpha_command,2*np.pi)
                self.switch('SWEEP',d)
        if self.phase=='SWEEP':
            h=.10;self.alpha_command+=np.clip(self.sweep_end-self.alpha_command,-5*DT,5*DT)
            if max(abs(self.sweep_end-self.alpha_command))<.02:self.switch('EXTEND',d)
        if self.phase=='EXTEND':
            h=.38
            if all(contact):self.switch('PUSH',d)
        if self.phase=='PUSH':
            h=.36
            self.alpha_command+=wrap(-pitch-self.alpha_command)*min(1,8*DT)
            if min(g['h'] for g in legs)>.31 and all(contact):self.switch('RIGHT',d)
        if self.phase=='RIGHT':
            h=.34
            goal=self.alpha_command+wrap(.082-self.alpha_command)
            self.alpha_command+=np.clip(goal-self.alpha_command,-1.5*DT,1.5*DT)
            if all(contact) and abs(pitch)<.10 and abs(roll)<.10 and all(.275<g['h']<.345 and abs(g['rate'][0])<.10 and abs(g['theta']-.082)<.12 for g in legs) and abs(d.sensor('imu_gyro').data[0])<.4:
                self.good+=DT
                if self.good>.25:return 'done'
            else:self.good=0.
        speed=.35 if self.phase in ['FOLD','SWEEP'] else .3
        self.length_command+=np.clip(h-self.length_command,-speed*DT,speed*DT)
        if d.time-self.last_plan>.01:
            for i,s in enumerate('LR'):
                r.seed=self.seeds[i]
                try:
                    q=r.inverse(self.length_command,float(self.alpha_command[i]));self.seeds[i]=q
                    desired=q[[0,2]];desired+=2*np.pi*np.round((self.targets[i]-desired)/(2*np.pi));self.targets[i]=desired
                except ValueError:return self.fail('recovery inverse kinematics unreachable')
            self.last_plan=d.time
        for i,s in enumerate('LR'):
            ids=r.qa[s][[0,2]];vs=r.va[s][[0,2]];g=legs[i]
            tau=180*(self.targets[i]-d.qpos[ids])-4*d.qvel[vs]
            wheel=0.
            if self.phase in ['PUSH','RIGHT']:
                pr=-float(d.sensor('imu_gyro').data[0]);theta=float(np.mean([v['theta'] for v in legs]))
                td=float(np.mean([v['rate'][1] for v in legs])+pr)
                K,tr=self.controller.schedule(np.mean([v['h'] for v in legs]))
                v=float(np.clip(np.mean(self.controller.last_info.get('velocity_encoder',[0.,0.])),-.4,.4)) if all(contact) else 0.
                state=np.array([theta,td,0,v,pitch,pr]);ref=np.array([tr[0],0,0,0,0,0])
                u=tr[1:3]+K@(ref-state)
                force=np.clip(tr[3]/2+2200*((.34 if abs(pitch)>.25 else .30)-g['h'])-90*g['rate'][0],-120,300)
                sync=np.clip(80*wrap(legs[0]['alpha']-legs[1]['alpha'])+8*(legs[0]['rate'][1]-legs[1]['rate'][1]),-12,12)
                roll_force=np.clip(500*roll-35*d.sensor('imu_gyro').data[1],-80,80)
                force=np.clip(force+(1 if i==0 else -1)*roll_force,-120,300)
                common=np.clip(u[1]/2,-25,25)
                tp=common+(-1 if i==0 else 1)*sync
                tau=g['J'].T@np.array([force,tp])
                wheel=u[0]/2-.12*d.qvel[r.va[s][6]]
            d.ctrl[r.acts[s][:2]]=np.clip(tau,-40,40)
            wheel_limit=min(4.92,.65*contact_force[i]*r.radius)
            d.ctrl[r.acts[s][2]]=np.clip(wheel,-wheel_limit,wheel_limit) if contact[i] else 0.
            self.turns[i]=float(np.mean(d.qpos[ids]-self.qstart[i][[0,2]]))
        return 'running'
