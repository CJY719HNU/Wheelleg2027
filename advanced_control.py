"""Hybrid wheel-leg simulation controller. See advanced_control_math.md.
Recovery is motor-only: no root pose reset, lifting fixture, or external recovery wrench.
"""
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from balance_control import Robot, Controller, ROOT, DT, wrap, costs #DT=.001
from gamepad_control import Command, XInput

def airborne_lqr(K,state):
    """Paper p.658: retain K21/K22 (virtual-leg tilt/rate), not body pitch.

    State order: [theta, theta_dot, path, velocity, pitch, pitch_rate].
    Outputs: total wheel torque T, total virtual hip torque Tp.
    No standing trim or ground references belong in this airborne feedback.
    """
    K_air=np.zeros_like(K)
    K_air[1,:2]=K[1,:2]
    return -(K_air@state)

class SensorRobot(Robot): #添加传感器的机器人类，继承自Robot类
    def __init__(self,terrain_angle=0.,terrain="flat",terrain_file=None): #self是指当前对象，terrain_angle是地形角度，terrain是地形类型，terrain_file是地形文件
        super().__init__()
        root=ET.parse(ROOT/'real_wheelleg_balance.xml').getroot() #读取xml文件，ET是指ElementTree，ROOT是指当前文件所在目录，real_wheelleg_balance.xml是一个xml文件，getroot()方法返回xml文件的根节点
        root.find('compiler').set('meshdir',str(ROOT/'stl')) #设置mesh目录，mesh是3D模型文件
        root.find("./worldbody/geom[@name='ground']").set('quat',f'{np.cos(terrain_angle/2)} 0 {np.sin(terrain_angle/2)} 0')
        
        from terrain_scene import add_terrain #从terrain_scene.py中导入add_terrain函数，add_terrain函数用于添加地形
        self.terrain_names=add_terrain(root,terrain,terrain_file) #这里添加了一个地形
        base=root.find("./worldbody/body[@name='base']") #找到xml文件中worldbody/body[@name='base']节点，base就是机体

        ET.SubElement(base,'site',name='imu',pos=' '.join(map(str,self.hip)),size='.003',rgba='0 0 0 0') #在base节点下添加一个site节点，name为imu，pos为hip位置，size为0.003，rgba为0 0 0 0
        
        sensor=ET.SubElement(root,'sensor') #传感器为根节点下的sensor节点
        ET.SubElement(sensor,'accelerometer',name='imu_acc',site='imu') #在sensor节点下添加一个accelerometer节点，name为imu_acc，site为imu，即加速度计
        ET.SubElement(sensor,'gyro',name='imu_gyro',site='imu') #在sensor节点下添加一个gyro节点，name为imu_gyro，site为imu，即陀螺仪
        self.m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode')) #self.m是mujoco模型，mujoco.MjModel.from_xml_string()方法从xml字符串创建一个mujoco模型，ET.tostring()方法将xml节点转换为字符串
        self.d=mujoco.MjData(self.m) #self.d是mujoco数据，mujoco.MjData()方法创建一个mujoco数据对象
        self.ground=self.m.geom('ground').id #self.ground是地面几何体的id，self.m.geom('ground')方法返回ground几何体，.id属性返回几何体的id
        self.wheels=[self.m.geom(s+'_wheel_col').id for s in 'LR'] #self.wheels是轮子几何体的id，self.m.geom(s+'_wheel_col')方法返回轮子几何体，.id属性返回几何体的id，for s in 'LR'表示左轮和右轮
        self.chassis=self.m.geom('chassis_col').id #self.chassis是底盘几何体的id，self.m.geom('chassis_col')方法返回底盘几何体，.id属性返回几何体的id
        self.mass=float(self.m.body_mass.sum()) #self.mass是机器人总质量，self.m.body_mass.sum()方法返回所有刚体的质量之和
    def contact_forces(self,d): #接触力
        forces=np.zeros(2);chassis=False;contact=np.zeros(2,dtype=bool) #初始化一个forces数组，长度为2，表示左右轮的接触力；chassis表示底盘是否接触地面；contact表示左右轮是否接触地面（bool值控制）
        for k in range(d.ncon): #遍历所有接触点，d.ncon是接触点的数量
            c=d.contact[k];pair=[int(c.geom1),int(c.geom2)] #c是接触点，pair是接触点的两个几何体的id
            # 任何外部接触，不仅仅是命名的地面：地形可以被添加。
            for i,w in enumerate(self.wheels):
                if w in pair and c.efc_address>=0:
                    f=np.zeros(6);mujoco.mj_contactForce(self.m,d,k,f)
                    forces[i]+=max(0.,float(f[0]));contact[i]=True
            if self.chassis in pair and c.efc_address>=0:chassis=True
        return forces,contact,chassis

def attitude(d,r): #姿态，对比于上面的imu，这个是计算姿态的函数，d是mujoco数据，r是机器人对象
    R=d.xmat[r.base].reshape(3,3)  #R是旋转矩阵，d.xmat[r.base]是机体的旋转矩阵，reshape(3,3)将其转换为3x3矩阵
    yaw=np.arctan2(-R[0,1],R[1,1]);heading=np.array([-np.sin(yaw),np.cos(yaw),0.]) #yaw是偏航角，heading是航向向量
    pitch=np.arctan2(-R[2,1],R[2,2])
    roll=np.arctan2(R[2,0],np.hypot(R[2,1],R[2,2]))
    gyro=d.sensor('imu_gyro').data.copy()
    return R,heading,float(pitch),float(roll),float(yaw),gyro

class VelocityKF: #卡尔曼滤波速度估计
    """2-D world horizontal velocity and accelerometer bias; no root velocity input.""" # 2D世界水平速度和加速度计偏差；没有根速度输入
    def __init__(self): #初始化self
        self.x=np.zeros(4);self.P=np.diag([.01,.01,.0025,.0025]);self.path=0. #self.x是状态向量，长度为4，分别是vx，vy，加速度计偏差bx，加速度计偏差by；self.P是协方差矩阵，4x4对角矩阵；self.path是路径长度
        self.innovation=0.;self.accepted=False #self.innovation是创新值，self.accepted是是否接受观测值
    def update(self,acc,heading,lateral,measurement,grounded): #更新
        A=np.eye(4);A[:2,2:]=-DT*np.eye(2) #A是状态转移矩阵，4x4单位矩阵，eye是得到一个单位阵I
        #这个矩阵最后会变成
        #  [ 1.   0.   -DT.   0.  ]
        #  [ 0.   1.   0.   -DT.  ]
        #  [ 0.   0.   1.   0.    ]
        #  [ 0.   0.   0.   1.    ]
        self.x=A@self.x;self.x[:2]+=DT*acc[:2] #@是矩阵乘法，self.x=A@self.x是状态预测相当于x(k+1)=Ax(k)；self.x[:2]+=DT*acc[:2]是将加速度乘以时间步长后，累加到状态向量的前两个元素上
        Q=np.diag([.4**2*DT**2]*2+[.003**2*DT]*2) #Q是过程噪声协方差矩阵，4x4对角矩阵，前两个元素是速度噪声，后两个元素是加速度计偏差噪声
        self.P=A@self.P@A.T+Q;self.accepted=False #数学语言是P(k+1)=AP(k)A^T+Q，self.accepted=False表示没有接受观测值
        if grounded: #如果落地
            H=np.zeros((2,4));H[0,:2]=heading[:2];H[1,:2]=lateral[:2] #H是观测矩阵，2x4矩阵，第一行是heading向量的前两个元素，第二行是lateral向量的前两个元素，KF可以把世界坐标系转换为机器人坐标系
            z=np.array([measurement,0.]);innovation=z-H@self.x #z是观测值，这个矩阵意思是正常运行时，正常接地且没有侧滑的时候，机器人横向速度应该接近 0。
            #创新值，例如 KF 当前预测的速度是 0.1 m/s，而测量值是 0.05 m/s，那么创新值就是 -0.05 m/s。这个创新值会被用来更新 KF 的状态估计。
            self.innovation=float(innovation[0]);scale=1+min(100.,(innovation[0]/.15)**2) #scale的意思是如果创新值过大，说明测量值和预测值差距过大，可能是测量噪声过大或者模型不准确，这时候就需要增加噪声协方差矩阵的值，让 KF 更加保守地更新状态估计。
            noise=np.diag([.025**2*scale,.06**2])
            S=H@self.P@H.T+noise #S是观测噪声协方差矩阵，2x2矩阵，S=HPH^T+R，其中R是观测噪声协方差矩阵
            if innovation@np.linalg.solve(S,innovation)<100: #异常值检测，马氏距离<100，则认为是正常值，接受观测值更新状态估计。
                #前一段写法等价于K=self.P@H^T(S^{-1})，求转置能成通常是因为P和S都是对称矩阵，其中S=HPH^T+R，R是观测噪声协方差矩阵，P是状态协方差矩阵，H是观测矩阵。
                K=np.linalg.solve(S,H@self.P).T;self.x+=K@innovation 
                E=np.eye(4)-K@H;self.P=E@self.P@E.T+K@noise@K.T;self.accepted=True #通过检查后，更新状态估计和协方差矩阵，self.accepted=True表示接受观测值
        v=float(heading[:2]@self.x[:2]);self.path+=v*DT
        return v

class SupportObserver: #这一段是为了估计机器人是否接地，使用了一个简单的低通滤波器来平滑接触力的变化，同时还考虑了接触力的阈值和时间窗口来判断是否接地。
    """Paper force estimate with filtered derivatives + optional simulator contact veto."""
    def __init__(self,source='fused'):
        self.source=source;self.force=np.zeros(2);self.grounded=np.ones(2,dtype=bool)
        self.on=np.zeros(2);self.off=np.zeros(2);self.previous=None;self.deriv=np.zeros((2,2))
    def update(self,r,d,legs,pitch_rate,acc_z,contact_force,contact):
        rate=np.array([[g['rate'][0],g['rate'][1]+pitch_rate] for g in legs])
        if self.previous is not None:self.deriv+=.04*((rate-self.previous)/DT-self.deriv)
        self.previous=rate.copy();raw=[]
        for i,s in enumerate('LR'):
            g=legs[i];h=g['h'];theta=g['theta'];hd,td=rate[i];hdd,tdd=self.deriv[i]
            tau=d.actuator_force[r.acts[s][:2]]
            F,Tp=np.linalg.lstsq(g['J'].T,tau,rcond=1e-6)[0]
            azw=acc_z-hdd*np.cos(theta)+2*hd*td*np.sin(theta)+h*tdd*np.sin(theta)+h*td*td*np.cos(theta)
            mw=r.m.body_mass[r.m.body(s+'_WHEEL').id]
            # Our positive Tp acts along increasing alpha: downward load is F*cos - Tp*sin/h.
            raw.append(F*np.cos(theta)-Tp*np.sin(theta)/h+mw*(9.81+azw))
        self.force+=.08*(np.array(raw)-self.force)
        for i in range(2):
            enough=self.force[i]>25
            low=self.force[i]<12
            if self.source=='fused':
                enough=enough and contact[i] and contact_force[i]>15
                low=low or not contact[i] or contact_force[i]<5
            self.on[i]=self.on[i]+DT if enough else 0.
            self.off[i]=self.off[i]+DT if low else 0.
            if self.off[i]>.008:self.grounded[i]=False
            if self.on[i]>.02:self.grounded[i]=True
        return self.grounded.copy()

class HybridController(Controller):
    def __init__(self,r,rows,height=.30,assist=True,contact_source='fused'):
        super().__init__(r,rows,height)
        self.assist=assist;self.contact_source=contact_source
        self.mode='GROUND';self.reason='';self.mode_time=0.;self.events=[]
        self.command=Command();self.command_time=-1.;self.auto_height=True
        self.jump_force=260.
        self.drive_accel=.6 # m/s^2; both acceleration and controlled braking reference slope
        self.torque_saturation_timeout=.8 # seconds of continuous saturation before disabling
    def initialize(self,d): #初始化
        super().initialize(d)
        self.estimator=VelocityKF();self.support=SupportObserver(self.contact_source) #从上面两个类中创建一个速度估计器和一个支撑观察器的实例
        self.mode='GROUND';self.reason='';self.mode_time=d.time;self.events=[]
        # 赋值，初始化一些变量包括速度、转向、参考路径、地形高度、腿部高度偏移、饱和时间、命令、上次控制输入、跳跃请求、恢复请求、辅助力矩、辅助计数、稳定时间、飞行状态、上次跳跃时间、上次信息和空气接触状态。
        self.speed=0.;self.turn=0.;self.yref=0.;self.terrain=0.;self.height_offsets=np.zeros(2)
        self.saturation_time=0.;self.command=Command();self.command_time=-1.
        self.last_ctrl=np.zeros(self.r.m.nu);self.jump_requested=False;self.recover_requested=False
        self.assist_wrench=np.zeros(6); # 兼容性遥测，始终为零
        self.assist_count=0;self.settle=0.
        self.flight_seen=False;self.last_jump=-10.;self.last_info={};self.air_no_contact=0.
    def submit(self,command,now=None): #
        if not np.isfinite([command.speed,command.yaw_rate,command.height_rate]).all():
            command=Command(stop=True)
        command.speed=float(np.clip(command.speed,-2.,2.));command.yaw_rate=float(np.clip(command.yaw_rate,-.5,.5));command.height_rate=float(np.clip(command.height_rate,-.02,.02))
        self.command=command;self.command_time=self.r.d.time if now is None else now
        self.jump_requested |= bool(command.jump);self.recover_requested |= bool(command.recover) and self.mode=='DISABLED' and not command.stop
        if command.stop:
            self.transition('DISABLED',self.r.d,'operator stop');self.reason='operator stop';self.recover_requested=False
    def transition(self,mode,d,reason=''): #状态机的转换
        if self.mode==mode:return
        self.events.append(dict(t=float(d.time),from_mode=self.mode,to=mode,reason=reason))
        self.mode=mode;self.mode_time=d.time;self.reason=reason; #在 self.events 列表中追加一条记录，包含时间、旧模式、新模式和切换原因，用于日志或调试。
        self.height_i[:]=0;self.yref=self.estimator.path;self.saturation_time=0.;self.settle=0.
        if mode in ['DISABLED','RECOVERY','CROUCH']:self.speed=0.;self.turn=0.# 重置控制器状态
        if mode=='AIR':
            #将目标腿长 self.height 设置为当前左右腿实际长度的平均值，以便落地后能平缓恢复到该长度。
            self.height=float(np.mean([self.r.leg(d,s,False)['h'] for s in 'LR']));self.flight_seen=False
        if mode=='DISABLED':
            d.ctrl[:]=0;self.last_ctrl[:]=0 #若失能，则将控制输入清零
        if mode=='RECOVERY':
            #动态导入 MotorRecovery 类（避免循环依赖）。
            #增加恢复次数计数 assist_count。
            #创建恢复驱动器实例 recovery_driver，该驱动器负责执行电机恢复流程（可能是从堵转或异常状态恢复）
            from motor_recovery import MotorRecovery
            self.assist_count+=1;self.recovery_driver=MotorRecovery(self.r,d,self)

    def disable(self,d,reason):self.transition('DISABLED',d,reason) #定义一个disable方法，用于将控制器状态切换为DISABLED，并记录原因。
    
    def recovery(self,d,legs,R,pitch,roll,yaw,grounded): #定义recovery方法，用于在RECOVERY模式下执行电机恢复操作。该方法调用recovery_driver的update方法，并根据返回结果进行相应处理。
        result=self.recovery_driver.update(d,legs,pitch,roll,grounded)
        if result=='done':
            self.height=self.target_height=float(np.clip(np.mean([g['h'] for g in legs]),.27,.35))
            v=float(np.mean(self.last_info.get('velocity_encoder',[0.,0.])))
            self.estimator=VelocityKF();self.estimator.x[:2]=v*np.array([-np.sin(yaw),np.cos(yaw)])
            self.support=SupportObserver(self.contact_source)
            self.yaw_target=yaw;self.transition('LANDING',d,'motor recovery handed to balance')
        elif result=='failed':self.disable(d,'motor recovery failed; manual retry required')

    def control(self,d): #这一段是控制器的核心逻辑，负责根据当前状态和传感器数据计算控制输入，并更新状态机。
        r=self.r;m=r.m;self.assist_wrench[:]=0
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            d.ctrl[:]=0;self.disable(d,'nonfinite state');return dict(mode=self.mode,reason=self.reason)
        R,heading,pitch,roll,yaw,gyro=attitude(d,r)
        try:legs=[r.leg(d,s) for s in 'LR']
        except (np.linalg.LinAlgError,ValueError,FloatingPointError):
            self.disable(d,'invalid kinematics');d.ctrl[:]=0
            return dict(self.last_info,mode=self.mode,reason=self.reason)
        for g in legs:g['theta']=float(wrap(g['alpha']+pitch))
        pr=-float(gyro[0]);rr=-float(gyro[1]);yr=float((R@gyro)[2])
        acc=R@d.sensor('imu_acc').data+m.opt.gravity
        if not np.isfinite(np.r_[gyro,acc]).all():
            self.disable(d,'invalid sensor');d.ctrl[:]=0
            return dict(self.last_info,mode=self.mode,reason=self.reason)
        cf,contact,chassis=r.contact_forces(d)
        grounded=self.support.update(r,d,legs,pr,float(acc[2]),cf,contact)
        measures=[]
        for i,s in enumerate('LR'):
            g=legs[i];active=d.qvel[r.va[s][[0,2]]];qdot=g['S']@active
            omega=pr+qdot[2]+qdot[5]+d.qvel[r.va[s][6]]
            h,theta=g['h'],g['theta'];hd,ad=g['rate']
            measures.append(r.radius*omega+h*(ad+pr)*np.cos(theta)+hd*np.sin(theta))
        use=[measures[i]-float(heading@np.cross(R@gyro,R@np.array([(-1 if i==0 else 1)*.1682055,0.,0.]))) for i in range(2) if grounded[i]]
        lateral=np.array([heading[1],-heading[0],0.])
        v=self.estimator.update(acc,heading,lateral,float(np.mean(use)) if use else 0.,bool(use))
        theta=float(np.mean([g['theta'] for g in legs]));td=float(np.mean([g['rate'][1] for g in legs])+pr)
        state=np.array([theta,td,self.estimator.path,v,pitch,pr])
        wheel_speed=max(abs(d.qvel[r.va[s][6]]) for s in 'LR')
        if self.mode not in ['DISABLED','RECOVERY']:
            if max(abs(pitch),abs(roll))>.60:self.disable(d,'tilt envelope')
            elif abs(v)>3.0 or wheel_speed>65:self.disable(d,'overspeed')
            elif any(g['condition']>1e5 or not .17<g['h']<.43 for g in legs):self.disable(d,'leg envelope')
            elif r.gap(d)>.008:self.disable(d,'closure error')
            elif self.saturation_time>self.torque_saturation_timeout:self.disable(d,'persistent torque saturation')
        fresh=d.time-self.command_time<.25
        cmd=self.command if fresh and self.command.connected else Command()
        if self.mode=='DISABLED':
            d.ctrl[:]=0;self.height_i[:]=0;self.yref=self.estimator.path
            auto=self.assist and self.reason not in ['operator stop','nonfinite state','recovery timeout','motor recovery failed; manual retry required'] and self.assist_count<1
            settled=(chassis or any(contact)) and np.linalg.norm(d.qvel[:3])<.4 and np.linalg.norm(gyro)<1.
            if self.recover_requested or (auto and d.time-self.mode_time>1 and settled):
                self.transition('RECOVERY',d,'motor-only recovery');self.recover_requested=False
        if self.mode=='RECOVERY':self.recovery(d,legs,R,pitch,roll,yaw,grounded)
        elif self.mode!='DISABLED':
            touching=(any(contact) and max(cf)>5) if self.contact_source=='fused' else any(grounded)
            if not any(grounded) and self.mode not in ['AIR'] and not (self.mode=='LANDING' and touching):
                self.transition('AIR',d,'both wheels unloaded')
            elif not all(grounded) and self.mode in ['GROUND','CROUCH']:
                self.transition('SINGLE',d,'single wheel unloaded')
            elif (self.mode=='AIR' and self.flight_seen and touching) or (self.mode=='SINGLE' and all(grounded)):
                self.height=float(np.clip(np.mean([g['h'] for g in legs]),.27,.35))
                if self.contact_source=='fused':
                    self.support.grounded[:]=contact;self.support.force[contact]=np.maximum(40,self.support.force[contact]);self.support.off[:]=0
                self.transition('LANDING',d,'contact confirmed')
            elif self.mode=='LANDING' and d.time-self.mode_time>.4 and all(grounded):
                self.transition('GROUND',d,'landing settled')
            if self.mode=='AIR' and not any(contact if self.contact_source=='fused' else grounded):self.flight_seen=True
            if self.mode=='AIR' and not self.flight_seen and d.time-self.mode_time>.18 and all(grounded):
                self.transition('LANDING',d,'no takeoff; return to stance')
            if self.jump_requested:
                if self.mode=='GROUND' and all(grounded) and max(abs(pitch),abs(roll))<.08 and abs(v)<.08 and d.time-self.last_jump>3:
                    self.transition('CROUCH',d,'jump command');self.last_jump=d.time
                self.jump_requested=False
            if self.mode=='CROUCH' and abs(np.mean([g['h'] for g in legs])-.20)<.004 and d.time-self.mode_time>.6:
                self.transition('THRUST',d,'crouch reached')
            if self.mode=='CROUCH' and d.time-self.mode_time>3:self.transition('GROUND',d,'crouch timeout')
            if self.mode=='THRUST' and (np.mean([g['h'] for g in legs])>.37 or d.time-self.mode_time>.20):
                self.transition('AIR',d,'thrust completed')
            self.speed+=np.clip((cmd.speed if self.mode in ['GROUND','LANDING','SINGLE'] else self.speed if self.mode=='AIR' else 0.)-self.speed,-self.drive_accel*DT,self.drive_accel*DT)
            self.turn+=np.clip((cmd.yaw_rate if self.mode=='GROUND' else 0.)-self.turn,-.8*DT,.8*DT)
            self.target_height=float(np.clip(self.target_height+cmd.height_rate*DT,.20,.35))
            target=.20 if self.mode=='CROUCH' else (.30 if self.mode=='AIR' else self.target_height)
            self.height+=np.clip(target-self.height,-.06*DT,.06*DT)
            self.yref+=self.speed*DT;self.yref=float(np.clip(self.yref,self.estimator.path-.12,self.estimator.path+.12))
            self.yaw_target=wrap(self.yaw_target+self.turn*DT)
            if self.mode!='GROUND':self.yref=self.estimator.path;self.yaw_target=yaw
            K,tr=self.schedule(np.mean([g['h'] for g in legs]));ref=np.array([tr[0],0,self.yref,self.speed,0,0])
            u=tr[1:3]+K@(ref-state)
            # Terrain height difference is wheel-center difference, reconstructed from attitude and legs.
            if self.auto_height and self.mode=='GROUND' and all(grounded):
                local_delta=np.array([.336411,0.,0.])+R.T@(legs[1]['vec']-legs[0]['vec'])
                dz=(R@local_delta)[2]
                self.terrain+=.002*(float(dz)-self.terrain)
            desired_offsets=np.array([self.terrain/2,-self.terrain/2])
            self.height_offsets+=np.clip(desired_offsets-self.height_offsets,-.025*DT,.025*DT)
            if not self.auto_height:self.height_offsets[:]=0
            targets=np.clip(self.height+self.height_offsets,.20,.35)
            rollF=np.clip(500*roll+35*rr,-80,80)
            yawT=np.clip(2*wrap(self.yaw_target-yaw)+.7*(self.turn-yr),-1.5,1.5)
            sync=np.clip(80*wrap(legs[0]['alpha']-legs[1]['alpha'])+8*(legs[0]['rate'][1]-legs[1]['rate'][1]),-8,8)
            d.ctrl[:]=0
            for i,s in enumerate('LR'):
                g=legs[i];a=r.acts[s];err=targets[i]-g['h']
                force=tr[3]/2+1800*err-65*g['rate'][0]+150*self.height_i[i]+(1 if i==0 else -1)*rollF
                tp=u[1]/2+(-1 if i==0 else 1)*sync
                tw=u[0]/2+(-1 if i==0 else 1)*yawT
                if self.mode=='THRUST':force=self.jump_force-12*g['rate'][0]+(1 if i==0 else -1)*rollF
                if self.mode=='SINGLE' and grounded[i]:force+=tr[3]/2
                if self.mode=='LANDING':force=tr[3]/2+2200*(max(.30,targets[i])-g['h'])-90*g['rate'][0]+(1 if i==0 else -1)*rollF
                if self.mode=='AIR' or (self.mode=='SINGLE' and not grounded[i]):
                    # Paper gain mask; VMC length and left/right synchronization are separate loops.
                    air_u=airborne_lqr(K,state)
                    tp=float(np.clip(.5*air_u[1],-6,6))+(-1 if i==0 else 1)*sync
                    air_length=.34 if (self.mode=='SINGLE' or d.time-self.mode_time>.08) else .30
                    force=np.clip(1000*(air_length-g['h'])-30*g['rate'][0],-70,70)
                    tw=0.;self.height_i[i]=0.
                else:force=np.clip(force,0,400)
                # Preserve extension force first during touchdown, then fit angular torque.
                if self.mode=='THRUST':
                    # Reserve posture torque before spending the remaining hip budget on extension.
                    angular=g['J'][1];radial=g['J'][0]
                    tp=float(np.clip(tp,-15,15))
                    angular_tau=angular*tp
                    cap=400.
                    for j in range(2):
                        if abs(radial[j])>1e-8:
                            bound=(39*np.sign(radial[j])-angular_tau[j])/radial[j]
                            cap=min(cap,bound)
                    force=float(np.clip(force,0,max(0,cap)))
                if self.mode=='LANDING':
                    force=min(force,float(np.min(39/np.maximum(abs(g['J'][0]),1e-6))))
                    radial=g['J'][0]*force;angular=g['J'][1];lo,hi=-100.,100.
                    for j in range(2):
                        if abs(angular[j])>1e-8:
                            bounds=sorted([(-39-radial[j])/angular[j],(39-radial[j])/angular[j]])
                            lo=max(lo,bounds[0]);hi=min(hi,bounds[1])
                    tp=float(np.clip(tp,lo,hi))
                # Contact load bounds traction; do not spin an unloaded wheel during detection delay.
                if self.contact_source=='fused':
                    traction=min(4.92,.65*cf[i]*r.radius) if contact[i] else 0.
                    tw=float(np.clip(tw,-traction,traction))
                raw=g['J'].T@np.array([force,tp]);d.ctrl[a[:2]]=raw;d.ctrl[a[2]]=tw
                if self.mode=='GROUND' and all(abs(raw)<39) and 0<force<390:
                    self.height_i[i]=np.clip(self.height_i[i]+err*DT,-.08,.08)
            saturated=np.any(abs(d.ctrl)>m.actuator_ctrlrange[:,1]*.999)
            self.saturation_time=self.saturation_time+DT if saturated and self.mode=='GROUND' else 0.
        d.ctrl[:]=np.clip(np.nan_to_num(d.ctrl),m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        if self.mode=='DISABLED':d.ctrl[:]=0
        self.last_ctrl=d.ctrl.copy()
        self.last_info=dict(mode=self.mode,reason=self.reason,state=state.tolist(),height=[g['h'] for g in legs],
            target=float(self.height),roll=roll,yaw=yaw,grounded=grounded.tolist(),contact_force=cf.tolist(),
            support_estimate=self.support.force.tolist(),contact=contact.tolist(),chassis_contact=bool(chassis),
            wheel_speed=float(wheel_speed),ctrl=d.ctrl.tolist(),gap=r.gap(d),velocity_est=float(v),
            velocity_encoder=measures,terrain=float(self.terrain),assist_wrench=self.assist_wrench.tolist(),
            estimator_update=bool(self.estimator.accepted))
        driver=getattr(self,'recovery_driver',None)
        self.last_info.update(recovery_phase=driver.phase if driver else '',recovery_failure=driver.failure if driver else '',recovery_repacks=driver.repacks if driver else 0)
        return self.last_info

def load_gains():
    saved=json.loads((ROOT/'balance_gains.json').read_text());Q,R=costs()
    if saved['model_sha256']!=hashlib.sha256((ROOT/'real_wheelleg_balance.xml').read_bytes()).hexdigest() or saved['Q']!=Q.tolist() or saved['R']!=R.tolist() or saved['dt']!=DT:
        raise ValueError('Run python balance_control.py --design first; physical model or LQR weights changed')
    return saved['rows']

def step(r,c,external=None):
    info=c.control(r.d)
    r.d.xfrc_applied[:]=0
    if not np.isfinite(r.d.qpos).all() or not np.isfinite(r.d.qvel).all():return info
    if external is not None:r.d.xfrc_applied[:]+=external if np.shape(external)==r.d.xfrc_applied.shape else 0
    if external is not None and np.shape(external)==(6,):r.d.xfrc_applied[r.base]+=external
    mujoco.mj_step(r.m,r.d)
    return info

def main():
    p=argparse.ArgumentParser();p.add_argument('--gamepad',action='store_true');p.add_argument('--pad-index',type=int,default=0)
    p.add_argument('--no-auto-recovery','--physical-only',dest='no_auto_recovery',action='store_true',help='Disable automatic get-up; START/R still requests motor-only recovery');p.add_argument('--contact-source',choices=['fused','estimated'],default='fused')
    p.add_argument('--terrain',choices=['flat','slope','launch','course'],default='flat');p.add_argument('--terrain-file')
    p.add_argument('--slope-deg',type=float,default=0.);p.add_argument('--no-adaptive-height',action='store_true')
    p.add_argument('--start-y',type=float,default=0.,help='Initial forward position in meters; -7 provides a launch runway')
    p.add_argument('--accel',type=float,default=.6,help='Drive reference acceleration/braking limit in m/s^2 (default: 0.6)')
    p.add_argument('--headless',action='store_true');p.add_argument('--seconds',type=float,default=10)
    args=p.parse_args();r=SensorRobot(np.deg2rad(args.slope_deg),args.terrain,args.terrain_file);c=HybridController(r,load_gains(),assist=not args.no_auto_recovery,contact_source=args.contact_source);c.initialize(r.d);c.auto_height=not args.no_adaptive_height
    if not np.isfinite(args.accel) or args.accel<=0:p.error('--accel must be positive and finite')
    c.drive_accel=args.accel
    if not np.isfinite(args.start_y):p.error('--start-y must be finite')
    r.d.qpos[1]+=args.start_y;mujoco.mj_forward(r.m,r.d) # Startup placement only; never used by recovery.
    if args.headless:
        for _ in range(int(args.seconds/DT)):info=step(r,c)
        print(json.dumps(dict(final=info,events=c.events),indent=2));return
    from mujoco import viewer as mv
    pad=XInput(args.pad_index) if args.gamepad else None;keyboard=Command()
    def key(k):
        if k==265:keyboard.speed=min(2.,keyboard.speed+.2)
        elif k==264:keyboard.speed=max(-2.,keyboard.speed-.2)
        elif k==263:keyboard.yaw_rate=min(.5,keyboard.yaw_rate+.1)
        elif k==262:keyboard.yaw_rate=max(-.5,keyboard.yaw_rate-.1)
        elif k==32:keyboard.speed=0;keyboard.yaw_rate=0
        elif k in [87,119]:c.target_height=min(.35,c.target_height+.01)
        elif k in [83,115]:c.target_height=max(.20,c.target_height-.01)
        elif k in [74,106]:keyboard.jump=True
        elif k in [69,101]:keyboard.stop=True
        elif k in [82,114]:keyboard.recover=True
    print('Arrows drive/turn; SPACE brake; W/S height; J jump; E disable; R motor recovery/re-arm.')
    with mv.launch_passive(r.m,r.d,key_callback=key) as viewer:
        viewer.cam.distance=1.7;viewer.cam.azimuth=40;viewer.cam.elevation=-20
        while viewer.is_running():
            start=time.perf_counter()
            with viewer.lock():
                command=pad.poll() if pad else keyboard
                if pad:
                    command.stop |= keyboard.stop;command.recover |= keyboard.recover
                c.submit(command)
                keyboard.jump=keyboard.stop=keyboard.recover=False
                mouse_wrench=r.d.xfrc_applied.copy()
                for _ in range(10):info=step(r,c,mouse_wrench)
                viewer.cam.lookat[:]=r.d.xpos[r.base]
            if hasattr(viewer,'set_texts'):
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'HYBRID VMC | '+c.mode+' '+(info.get('recovery_phase','') if c.mode=='RECOVERY' else ''),
                    f"v={info.get('velocity_est',0):.3f} m/s | h={c.height:.3f} m\n{c.reason}\nLB+stick drive | A jump | B disable | START recovery\nMotor recovery: {'AUTO' if c.assist else 'MANUAL'}")])
            r.d.xfrc_applied[:]=0 # viewer.sync writes the next mouse perturbation
            viewer.sync();time.sleep(max(0,.01-(time.perf_counter()-start)))

if __name__=='__main__':main()
