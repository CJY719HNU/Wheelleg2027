"""Closed-chain VMC and six-state, height-scheduled discrete LQR.
Axes: forward +Y, up +Z, pitch about -X. Units SI.
"""
from pathlib import Path
import argparse, json, time, re, hashlib, csv
import numpy as np
import mujoco
from scipy.optimize import least_squares, brentq
from scipy.linalg import expm, solve_discrete_are
ROOT=Path(__file__).resolve().parent
DT=.001

def wrap(x): return np.arctan2(np.sin(x),np.cos(x))

def costs():
    text=(ROOT/'test.m').read_text(encoding='utf-8')
    out=[]
    for name,n in [('Q_cost',6),('R_cost',2)]:
        a=re.search(name+r'\s*=\s*diag\s*\(\s*\[([^]]+)\]',text)
        if not a: raise ValueError('Missing '+name+' in test.m')
        v=np.fromstring(a[1].replace(',',' '),sep=' ')
        if len(v)!=n or np.any(v<=0): raise ValueError('Invalid '+name)
        out.append(np.diag(v))
    return out

class Robot:
    def __init__(self):
        self.m=mujoco.MjModel.from_xml_path(str(ROOT/'real_wheelleg_balance.xml'))
        self.d=mujoco.MjData(self.m);m=self.m
        self.base=m.body('base').id; self.radius=.068
        self.jnames=['base_L1','L1_L2','base_L5','L3_L5','L3_L4','L5_L6','W']
        self.joints={s:np.array([m.joint(s+'_J_'+n).id for n in self.jnames]) for s in 'LR'}
        self.qa={s:m.jnt_qposadr[j] for s,j in self.joints.items()}
        self.va={s:m.jnt_dofadr[j] for s,j in self.joints.items()}
        self.acts={s:np.array([m.actuator(s+'_'+n+'_torque').id for n in ['L1','L5','W']]) for s in 'LR'}
        self.hip=np.mean([m.body_pos[m.body('L_L1').id],m.body_pos[m.body('L_L5').id]],axis=0)
        self.hip[0]=0
        self.seed=np.zeros(6)
    def kin(self,d):
        mujoco.mj_kinematics(self.m,d);mujoco.mj_comPos(self.m,d)
    def jac(self,d,p,b):
        jp=np.zeros((3,self.m.nv));jr=jp.copy();mujoco.mj_jac(self.m,d,jp,jr,p,b);return jp,jr
    def leg(self,d,s,need_j=True):
        m=self.m; b=m.body(s+'_WHEEL').id
        rot=d.xmat[self.base].reshape(3,3)
        hiplocal=self.hip.copy();hiplocal[0]=(-1 if s=='L' else 1)*.1682055
        hip=d.xpos[self.base]+rot@hiplocal
        v=d.xpos[b]-hip; local=rot.T@v
        h=np.hypot(local[1],local[2]);alpha=np.arctan2(-local[1],-local[2])
        theta=np.arctan2(-v[1],-v[2]);height=-v[2]
        result=dict(h=h,alpha=alpha,theta=theta,height=height,hip=hip,vec=v)
        if not need_j:return result
        C=[]
        for a,c in [('L2','L3'),('L4','L6')]:
            u=m.site(s+'_'+a+'_pin').id;w=m.site(s+'_'+c+'_pin').id
            ju,_=self.jac(d,d.site_xpos[u],m.site_bodyid[u]);jw,_=self.jac(d,d.site_xpos[w],m.site_bodyid[w])
            C.extend((rot.T@(ju-jw))[1:3,self.va[s][:6]])
        C=np.array(C);active=[0,2];passive=[1,3,4,5]
        S=np.zeros((6,2));S[active]=np.eye(2)
        S[passive]=np.linalg.lstsq(C[:,passive],-C[:,active],rcond=1e-9)[0]
        jw,_=self.jac(d,d.xpos[b],b);jh,jr=self.jac(d,hip,self.base)
        Jpos=(rot.T@(jw-jh))[:,self.va[s][:6]]
        Jcart=Jpos[1:3]@S
        Jtask=np.array([[local[1]/h,local[2]/h],[local[2]/h**2,-local[1]/h**2]])@Jcart
        result.update(J=Jtask,condition=float(np.linalg.cond(C[:,passive])),rate=Jtask@d.qvel[self.va[s][active]])
        return result
    def forward_leg(self,q_L1,q_L5,side='L'):
        """Encoder-only forward kinematics on the reference assembly branch.
        Input is hinge displacement, not absolute phi. Returns h/alpha/J and passive q.
        """
        d=mujoco.MjData(self.m);adr=self.qa[side];d.qpos[adr[[0,2]]]=[q_L1,q_L5]
        def residual(passive):
            d.qpos[adr[[1,3,4,5]]]=passive;self.kin(d);out=[]
            for a,b in [('L2','L3'),('L4','L6')]:
                out.extend((d.site_xpos[self.m.site(side+'_'+a+'_pin').id]-d.site_xpos[self.m.site(side+'_'+b+'_pin').id])[1:3])
            return out
        sol=least_squares(residual,np.zeros(4),xtol=1e-11,gtol=1e-11,ftol=1e-11,max_nfev=60)
        if np.linalg.norm(sol.fun)>1e-7:raise ValueError('Encoder configuration cannot close the reference branch')
        residual(sol.x);out=self.leg(d,side);out['passive_q']=sol.x;return out
    def inverse(self,h,alpha):
        d=mujoco.MjData(self.m)
        def res(q):
            d.qpos[self.qa['L'][:6]]=q;self.kin(d)
            g=self.leg(d,'L',False);r=[]
            for a,b in [('L2','L3'),('L4','L6')]:
                r.extend((d.site_xpos[self.m.site('L_'+a+'_pin').id]-d.site_xpos[self.m.site('L_'+b+'_pin').id])[1:3])
            r.extend([g['h']-h,wrap(g['alpha']-alpha)*h]);return r
        sol=least_squares(res,self.seed,xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=50)
        if np.linalg.norm(sol.fun)>1e-7:raise ValueError(f'Unreachable leg pose {h}, {alpha}: {sol.fun}')
        self.seed=sol.x.copy();return sol.x
    def pose(self,z):
        # Reduced coordinates [world leg tilt theta, hip Y, body pitch, length].
        theta,y,pitch,h=z;q=self.inverse(h,theta-pitch)
        d=mujoco.MjData(self.m)
        d.qpos[3:7]=[np.cos(pitch/2),-np.sin(pitch/2),0,0]
        cp,sp=np.cos(pitch),np.sin(pitch)
        rot=np.array([[1,0,0],[0,cp,sp],[0,-sp,cp]])
        d.qpos[:3]=np.array([0,y,self.radius+h*np.cos(theta)])-rot@self.hip
        for s in 'LR':
            d.qpos[self.qa[s][:6]]=q
            d.qpos[self.qa[s][6]]=(y-h*np.sin(theta))/self.radius-pitch-q[2]-q[5]
        self.kin(d);return d
    def tangent(self,z):
        eps=2e-5;E=np.zeros((self.m.nv,4))
        for i in range(4):
            dz=np.eye(4)[i]*eps;a=self.pose(z-dz);b=self.pose(z+dz)
            v=np.zeros(self.m.nv);mujoco.mj_differentiatePos(self.m,v,2*eps,a.qpos,b.qpos);E[:,i]=v
        return E
    def allocation(self,d):
        # Inputs total wheel torque T, total relative-leg torque Tp, total extension force F.
        U=np.zeros((self.m.nu,3))
        for s in 'LR':
            g=self.leg(d,s);a=self.acts[s]
            U[a[2],0]=.5;U[a[:2],1]=g['J'][1]*.5;U[a[:2],2]=g['J'][0]*.5
        return U
    def reduced(self,z):
        d=self.pose(z);mujoco.mj_forward(self.m,d);E=self.tangent(z)
        M=np.zeros((self.m.nv,self.m.nv));mujoco.mj_fullM(self.m,M,d.qM)
        B=np.zeros((self.m.nv,self.m.nu))
        for s in 'LR':B[self.va[s][[0,2,6]],self.acts[s]]=1
        return E.T@M@E,E.T@d.qfrc_bias,E.T@B@self.allocation(d),E.T@np.diag(self.m.dof_damping)@E
    def phi_angles(self,d,s):
        # phi1 uses L5 -> L6, phi4 uses L1 -> L2, both measured from +Y toward -Z.
        v5=self.m.body_pos[self.m.body(s+'_L6').id];v1=self.m.body_pos[self.m.body(s+'_L2').id]
        return np.array([np.arctan2(-v5[2],v5[1])+d.qpos[self.qa[s][2]],np.arctan2(-v1[2],v1[1])+d.qpos[self.qa[s][0]]])
    def gap(self,d):
        return max(np.linalg.norm(d.site_xpos[self.m.site(s+'_'+a+'_pin').id]-d.site_xpos[self.m.site(s+'_'+b+'_pin').id]) for s in 'LR' for a,b in [('L2','L3'),('L4','L6')])

def dlqr(A,B,Q,R,dt=DT):
    # Exact ZOH dynamics and integrated continuous quadratic cost (MATLAB lqrd).
    n,k=B.shape;F=np.zeros((n+k,n+k));F[:n,:n]=A;F[:n,n:]=B
    W=np.zeros_like(F);W[:n,:n]=Q;W[n:,n:]=R
    V=expm(np.block([[-F.T,W],[np.zeros_like(F),F]])*dt)
    Phi=V[n+k:,n+k:];Wd=Phi.T@V[:n+k,n+k:];Wd=(Wd+Wd.T)*.5
    Ad=Phi[:n,:n];Bd=Phi[:n,n:];Qd=Wd[:n,:n];Rd=Wd[n:,n:];Nd=Wd[:n,n:]
    P=solve_discrete_are(Ad,Bd,Qd,Rd,s=Nd)
    K=np.linalg.solve(Rd+Bd.T@P@Bd,Bd.T@P@Ad+Nd.T)
    return K,float(max(abs(np.linalg.eigvals(Ad-Bd@K))))

def design(robot,heights):
    Q,R=costs();rows=[]
    for h in heights:
        def equilibrium(v):
            z=[v[0],0,0,h];_,g,B,_=robot.reduced(z)
            return g-B@v[1:]
        e=least_squares(equilibrium,[.06,0,0,220],diff_step=1e-3,xtol=1e-8,gtol=1e-7,ftol=1e-8,max_nfev=25)
        if np.linalg.norm(e.fun)>1e-3:raise ValueError('Static trim failed '+str(e.fun))
        theta,T,Tp,F=e.x;z=np.array([theta,0,0,h]);M,g,B,D=robot.reduced(z)
        A=np.zeros((6,6));BB=np.zeros((6,2));A[[0,2,4],[1,3,5]]=1
        for i in range(3):
            eps=1e-4;dz=np.eye(4)[i]*eps
            _,ga,Ba,_=robot.reduced(z-dz);_,gb,Bb,_=robot.reduced(z+dz)
            stiffness=((Bb@e.x[1:]-gb)-(Ba@e.x[1:]-ga))/(2*eps)
            A[[1,3,5],2*i]=np.linalg.solve(M[:3,:3],stiffness[:3])
        A[np.ix_([1,3,5],[1,3,5])]=-np.linalg.solve(M[:3,:3],D[:3,:3])
        BB[[1,3,5]]=np.linalg.solve(M[:3,:3],B[:3,:2])
        K,rho=dlqr(A,BB,Q,R)
        if rho>=1:raise ValueError('Unstable design')
        row=dict(h=float(h),trim=e.x.tolist(),K=K.tolist(),A=A.tolist(),B=BB.tolist(),rho=rho)
        rows.append(row);print('design',round(h,3),'trim',np.round(e.x,4),'rho',rho,flush=True)
    return rows

class Controller:
    def __init__(self,r,rows,height=.30):
        self.r=r;self.rows=rows;
        if not rows[0]['h']<=height<=rows[-1]['h']:raise ValueError('Height outside designed range')
        self.height=height;self.target_height=height;self.velocity=0.;self.yaw_target=0.;self.yref=0.
        self.previous=None;self.filtered=np.zeros(3);self.height_i=np.zeros(2)
    def schedule(self,h):
        hs=[x['h'] for x in self.rows];h=np.clip(h,hs[0],hs[-1]);i=min(max(np.searchsorted(hs,h)-1,0),len(hs)-2)
        w=(h-hs[i])/(hs[i+1]-hs[i]);a,b=self.rows[i:i+2]
        return (1-w)*np.array(a['K'])+w*np.array(b['K']),(1-w)*np.array(a['trim'])+w*np.array(b['trim'])
    def set_vertical_height(self,meters):
        """Set hip-to-wheel vertical height at the scheduled equilibrium (meters)."""
        def error(length):
            _,trim=self.schedule(length)
            return length*np.cos(trim[0])-meters
        self.target_height=brentq(error,self.rows[0]['h'],self.rows[-1]['h'])
    def initialize(self,d):
        _,tr=self.schedule(self.height);p=self.r.pose([tr[0],0,0,self.height]);mujoco.mj_resetData(self.r.m,d);d.qpos[:]=p.qpos
        mujoco.mj_forward(self.r.m,d);self.previous=None;self.filtered[:]=0;self.height_i[:]=0;self.yref=0;self.velocity=0;self.yaw_target=0
    def control(self,d):
        r=self.r;m=r.m
        self.target_height=float(np.clip(self.target_height,.27,.35))
        self.height+=np.clip(self.target_height-self.height,-.015*DT,.015*DT)
        self.yref+=self.velocity*DT
        legs=[r.leg(d,s) for s in 'LR'];rot=d.xmat[r.base].reshape(3,3)
        pitch=np.arctan2(rot[1,2],rot[2,2]);roll=np.arctan2(-rot[0,2],np.hypot(rot[1,2],rot[2,2]));yaw=np.arctan2(-rot[0,1],rot[1,1])
        if not np.isfinite(d.qpos).all() or max(abs(pitch),abs(roll))>.65:
            d.ctrl[:]=0;raise RuntimeError('Outside balance recovery envelope; reset required')
        pos=np.array([np.mean([g['theta'] for g in legs]),np.mean([g['hip'][1] for g in legs]),pitch])
        if self.previous is None:rates=np.zeros(3);self.old_roll=roll;self.old_yaw=yaw
        else:rates=(pos-self.previous)/DT
        self.filtered+=.2*(rates-self.filtered);self.previous=pos.copy()
        rollrate=wrap(roll-self.old_roll)/DT;yawrate=wrap(yaw-self.old_yaw)/DT;self.old_roll=roll;self.old_yaw=yaw
        if any(g['condition']>1e5 for g in legs):
            d.ctrl[:]=0;raise RuntimeError('Closed-chain Jacobian near singularity')
        K,tr=self.schedule(np.mean([g['h'] for g in legs]));state=np.empty(6);state[::2]=pos;state[1::2]=self.filtered
        ref=np.array([tr[0],0,self.yref,self.velocity,0,0]);u=tr[1:3]+K@(ref-state)
        rollF=np.clip(500*roll+35*rollrate,-80,80);yawT=np.clip(2*wrap(self.yaw_target-yaw)-.5*yawrate,-1.5,1.5)
        sync=np.clip(80*wrap(legs[0]['alpha']-legs[1]['alpha'])+8*(legs[0]['rate'][1]-legs[1]['rate'][1]),-8,8)
        d.ctrl[:]=0
        for i,s in enumerate('LR'):
            g=legs[i];err=self.height-g['h'];self.height_i[i]=np.clip(self.height_i[i]+err*DT,-.08,.08)
            force=tr[3]/2+1800*err-65*g['rate'][0]+150*self.height_i[i]+(1 if i==0 else -1)*rollF
            force=np.clip(force,0,400)
            tp=u[1]/2+(-1 if i==0 else 1)*sync
            a=r.acts[s];d.ctrl[a[:2]]=g['J'].T@np.array([force,tp]);d.ctrl[a[2]]=u[0]/2+(-1 if i==0 else 1)*yawT
        d.ctrl[:]=np.clip(d.ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        contacts=[set([int(d.contact[k].geom1),int(d.contact[k].geom2)]) for k in range(d.ncon)]
        ground=m.geom('ground').id;chassis=m.geom('chassis_col').id
        wc=[sum(ground in pair and m.geom(s+'_wheel_col').id in pair for pair in contacts) for s in 'LR']
        return dict(phi={s:r.phi_angles(d,s).tolist() for s in 'LR'},wheel_contacts=wc,chassis_contacts=sum(ground in pair and chassis in pair for pair in contacts),base_z=float(d.xpos[r.base,2]),vertical_height=[g['height'] for g in legs],state=state.tolist(),height=[g['h'] for g in legs],target=self.height,roll=float(roll),yaw=float(yaw),gap=r.gap(d),ctrl=d.ctrl.tolist())

def run(args):
    r=Robot();cache=ROOT/'balance_gains.json'
    signature=hashlib.sha256((ROOT/'real_wheelleg_balance.xml').read_bytes()).hexdigest()
    if args.design or not cache.exists():
        rows=design(r,np.linspace(.27,.35,9));cache.write_text(json.dumps(dict(Q=costs()[0].tolist(),R=costs()[1].tolist(),model_sha256=signature,dt=DT,rows=rows),indent=2))
    else:
        saved=json.loads(cache.read_text());Q,R=costs()
        if saved.get('model_sha256')!=signature or saved.get('dt')!=DT:raise ValueError('Model changed; run --design')
        if saved['Q']!=Q.tolist() or saved['R']!=R.tolist():raise ValueError('test.m changed; run --design')
        rows=saved['rows']
    c=Controller(r,rows,args.height);d=r.d;c.initialize(d)
    if args.headless:
        logs=[]
        for i in range(int(args.seconds/DT)):
            if args.scenario=='height': c.target_height=.33 if d.time>3 and d.time<8 else .29
            if args.scenario=='push' and 3<d.time<3.15:d.xfrc_applied[r.base,1]=12
            else:d.xfrc_applied[:]=0
            info=c.control(d);mujoco.mj_step(r.m,d)
            if i%20==0: logs.append(dict(t=float(d.time),**info))
            if not np.isfinite(d.qpos).all() or abs(info['state'][4])>.65:raise RuntimeError(f'Balance failed at {d.time:.3f}s: {info}')
        result=dict(scenario=args.scenario,seconds=args.seconds,max_pitch=max(abs(x['state'][4]) for x in logs),max_gap=max(x['gap'] for x in logs),max_roll=max(abs(x['roll']) for x in logs),max_height_error=max(max(abs(np.array(x['height'])-x['target'])) for x in logs),max_chassis_contacts=max(x['chassis_contacts'] for x in logs),min_wheel_contacts=min(min(x['wheel_contacts']) for x in logs if x['t']>.2),final=logs[-1])
        (ROOT/('balance_'+args.scenario+'_log.json')).write_text(json.dumps(dict(summary=result,samples=logs),indent=2));print(json.dumps(result,indent=2))
        with (ROOT/('balance_'+args.scenario+'_log.csv')).open('w',newline='') as f:
            out=csv.writer(f);out.writerow(['time','theta','theta_dot','hip_y','hip_v','pitch','pitch_rate','length_L','length_R','target_length','roll','yaw','connect_gap']+[f'torque_{i}' for i in range(6)])
            for x in logs:out.writerow([x['t']]+x['state']+x['height']+[x['target'],x['roll'],x['yaw'],x['gap']]+x['ctrl'])
        if result['max_chassis_contacts'] or result['max_height_error']>.015 or result['min_wheel_contacts']==0:raise RuntimeError('Acceptance failed: height or ground contact')
        return
    from mujoco import viewer as mjviewer
    reset_requested=False
    def key(k):
        nonlocal reset_requested
        if k in [82,114]:reset_requested=True
        elif k in [87,119]:c.target_height=min(.35,c.target_height+.01)
        elif k in [83,115]:c.target_height=max(.27,c.target_height-.01)
        elif k==265:c.velocity=min(.2,c.velocity+.025)
        elif k==264:c.velocity=max(-.2,c.velocity-.025)
        elif k==32:c.velocity=0

    print('W/S height +/- 1cm; arrows velocity; SPACE stop velocity; R reset; close window to exit.')
    with mjviewer.launch_passive(r.m,d,key_callback=key) as viewer:
        viewer.cam.distance=1.5;viewer.cam.azimuth=40;viewer.cam.elevation=-20
        while viewer.is_running():
            start=time.perf_counter()
            with viewer.lock():
                if reset_requested:c.initialize(d);reset_requested=False
                for _ in range(10):info=c.control(d);mujoco.mj_step(r.m,d)
                viewer.cam.lookat[:]=d.xpos[r.base]
            if hasattr(viewer,'set_texts'):
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,'LQR + VMC | W/S height | R reset',f"Length L/R: {info['height'][0]:.3f} / {info['height'][1]:.3f} m\nTarget: {c.target_height:.3f} m | pitch: {np.degrees(info['state'][4]):.2f} deg")])
            viewer.sync();time.sleep(max(0,.01-(time.perf_counter()-start)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--design',action='store_true');p.add_argument('--headless',action='store_true');p.add_argument('--seconds',type=float,default=12);p.add_argument('--height',type=float,default=.30);p.add_argument('--scenario',choices=['stand','height','push'],default='stand');run(p.parse_args())


