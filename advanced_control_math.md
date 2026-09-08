# 轮腿控制代码、数学原理与强化学习接入

本文对应 `advanced_control.py`、`gamepad_control.py` 与复用的 `balance_control.py`。这是当前 MuJoCo 模型的仿真实现，原来的 `python balance_control.py` 仍运行旧版基础演示；新功能请用下面的入口。

## 1. 先运行，再理解控制链

```powershell
python advanced_control.py --gamepad
```

不接手柄时：

```powershell
python advanced_control.py
```

模拟横向斜坡（角度单位为度）：

```powershell
python advanced_control.py --gamepad --slope-deg 2.86
```

只关闭自动起身尝试（START/R 仍可请求电机起身；所有模式均无外力扶正或状态复位）：

```powershell
python advanced_control.py --gamepad --physical-only
```

使用支持力估计决定离地、触地，关闭普通运动状态切换中的仿真接触辅助：

```powershell
python advanced_control.py --physical-only --contact-source estimated
```

`estimated` 仍使用模拟 IMU、模型姿态和关节状态，并不代表已经接入实机。电机起身仍使用仿真接触反馈；研究纯估计时关闭自动恢复且不发送手动恢复请求。

### Xbox / XInput 操作

| 操作 | 效果 |
|---|---|
| 按住 LB，左摇杆前后 | 前进/后退，最大 2 m/s |
| 按住 LB，左摇杆左右 | 左转/右转，最大 0.5 rad/s；不是横向平移 |
| 按住 LB，方向十字键上下 | 改变腿长期望，最大 0.02 m/s，范围 0.20–0.35 m |
| 按住 LB，按一次 A | 请求小跳；稳定、低速、双轮接地时接受 |
| 松开 LB、摇杆归中或手柄断开 | 速度和转向指令归零，经斜率限制停车；保持平衡控制 |
| B | 六电机力矩立即归零，急停锁存；机体可能自然倒下 |
| START | 仅在失能后请求恢复；B 与 START 同按时 B 优先 |

`--pad-index 0` 是第一只手柄，可改为 1、2、3。无需安装 pygame。未检测到手柄时保持零运动指令，不会沿用断连前的摇杆输入。当前工作机的 XInput 返回未连接，实体按钮实测尚未完成；协议解析和断连逻辑已通过自动测试。

键盘：上下方向键调整速度，左右方向键调整转向角速度，空格将两者归零，W/S 改腿长，J 小跳，E 急停，R 恢复。键盘速度是逐次累加的，松键不会自动回中；停车请按空格。仿真连续运行，不自动暂停。

### 与论文的关系

《轮腿式平衡机器人控制》PDF 第 8–9 页给出速度解算，第 9 页给出转向、双腿协调、腿长和横滚补偿，第 10–11 页给出支持力估计与离地控制。第 11 页明确指出尚未研究跳跃，也未实现侧翻和倒置自恢复。因此本实现的跳跃状态机、保护逻辑和电机起身状态机是工程扩展，不能称为论文原算法的直接复现。

## 2. 每个周期按什么顺序运行

控制与物理步长均为 0.001 s：

1. `SensorRobot` 装入浮体模型，添加位于髋轴中心的加速度计和陀螺仪；惯量、碰撞、闭环约束及电机保持原模型。
2. `Robot.leg` 求每条腿的长度、角度、速度、闭环雅可比。
3. `attitude` 将状态转换到当前航向，读取模拟 IMU。
4. `SupportObserver.update` 从电机反馈力矩与加速度估计支持力，判断接地。
5. 轮速、定子连杆角速度、陀螺仪组成编码器速度观测。
6. `VelocityKF.update` 融合加速度预测和接地轮速，输出速度、积分路径。
7. `HybridController.control` 检查故障、切换状态、计算 LQR、VMC、转向与地形补偿。
8. 根据状态分配力矩、限幅，写入 `d.ctrl`。
9. `step` 仅添加用户指定的扰动力，调用 `mj_step`；恢复动作只通过电机 ctrl 输出。

GUI 每 10 个物理步刷新一次、轮询一次手柄。这个 Python 实现便于验证，不等于 MCU 已能在 1 ms 内执行全部运算。

## 3. 坐标、角度和闭环运动学

### 3.1 用户指定的角度没有改变

车体前方为局部 +Y，上方为 +Z；两侧铰链轴均为局部 -X。正转遵守右手定则。

- L1 的绝对杆角是 `phi4`，参考姿态约 33.178268°。
- L5 的绝对杆角是 `phi1`，参考姿态约 143.320161°。
- `phi4 = phi4_ref + q_L1`，`phi1 = phi1_ref + q_L5`。
- 三角运算一律使用 rad；锐角/钝角描述参考装配姿态，运动时不能强制把角度折回锐角/钝角区间。

本代码雅可比的列顺序是 `[q_L1,q_L5]`，论文按 `[phi1,phi4]` 排列时必须交换两列，不能只改变量名字。

### 3.2 闭环约束如何变成两个输入

每侧两个 site-connect 各提供平面 YZ 两个位置约束。记主动关节为 q_a=[q_L1,q_L5]，四个被动角为 q_p，闭合方程为 c(q_a,q_p)=0。微分后：

$$C_a\dot q_a+C_p\dot q_p=0,\qquad \dot q_p=-C_p^\dagger C_a\dot q_a.$$

组成六关节速度映射 S，使 $\dot q=S\dot q_a$。新增返回项 `g['S']` 正是此矩阵，被动角速度来自闭环运动学，而不是假定 L1/L5 必须同步转动。

设车体坐标中髋轴到轮心向量为 $(y,z)$：

$$h=\sqrt{y^2+z^2},\quad \alpha=\operatorname{atan2}(-y,-z),$$

$$J=\begin{bmatrix}y/h&z/h\\z/h^2&-y/h^2\end{bmatrix}J_{yz}S,\qquad
\begin{bmatrix}\dot h\\\dot\alpha\end{bmatrix}=J\dot q_a.$$

`g['h']` 是腿长，垂直腿高近似为 $H=h\cos\theta$；二者不能混用。闭环条件数过大时，代码失能，不继续放大奇异处的控制输入。

### 3.3 转向以后仍使用正确的平衡平面

航向为 $\psi$，水平前向单位向量为 $e_f=[-\sin\psi,\cos\psi,0]^T$。纵向速度投影到这个方向。平衡状态里的世界腿角用 $\theta=\alpha+\varphi$ 构造，避免继续把固定世界 Y 当成车头。

这是近似解耦的平面模型，适用于本次验证的小横滚运动。大姿态时由保护接管，不宣称上述欧拉角表达能全姿态无奇异工作。

## 4. 速度为什么不能只乘轮半径

### 4.1 对应论文 1.5、1.6 节

编码器量到的是轮转子相对定子的转速，而定子装在 L6 上，L6 也在相对车体转动。实际轮轴转速为：

$$\omega_w=\dot\varphi+\dot q_{L5}+\dot q_{L6}+\dot q_W.$$

代码用 `S @ active_velocity` 算出包含被动角在内的速度，取第 2、5 项作为 L5 和 L6 的相对转动贡献，再加轮编码器和陀螺仪。

纯滚动时轮心速度 $v_w=R_w\omega_w$。由于髋轴相对轮心还有腿部运动：

$$v_{hip}=R_w\omega_w+h\dot\theta\cos\theta+\dot h\sin\theta.$$

这三个项分别对应轮的滚动、腿的摆动、腿长的变化。漏掉后两项会在变腿高时把腿部运动误认为整车速度。

双轮接地时取两侧观测平均；只有一轮接地时，先减去 $e_f^T(\omega_{body}\times r_{hip})$，消除该侧髋轴因转向产生的附加速度。腾空轮的观测不进入滤波更新。

### 4.2 加速度计去重力

加速度计位于髋轴中心，减少从 IMU 安装点换算到髋轴时的杠杆臂项。MuJoCo 输出传感器局部比力 $f_b$，因此：

$$a_w=R_{wb}f_b+g_w,\qquad g_w=[0,0,-9.81]^T.$$

平地静止时 $f_b$ 约为向上 9.81 m/s²，换算后应接近零。不能把这个读数再次当成已经去重力的加速度。

姿态矩阵目前取自 MuJoCo `xmat`，陀螺仪/加速度计取自 `sensordata`。速度滤波没有读取 root 世界平动速度，但姿态仍是理想仿真姿态；真实 IMU 的姿态融合、安装角、偏置标定尚未实现。

### 4.3 `VelocityKF` 的四维卡尔曼滤波

滤波状态 $x=[v_X,v_Y,b_X,b_Y]^T$ 为世界水平速度与水平加速度偏置：

$$A=\begin{bmatrix}I_2&-\Delta t I_2\\0&I_2\end{bmatrix},\quad
x^-=Ax+\begin{bmatrix}a_{XY}\Delta t\\0\end{bmatrix},\quad P^-=APA^T+Q.$$

Q 对速度取 $0.4^2\Delta t^2$，偏置随机游走取 $0.003^2\Delta t$。这是当前仿真使用的噪声配置，需要按真实传感器数据辨识。

接地时，观测为前向编码器速度与横向零滑移约束：

$$z=\begin{bmatrix}v_{encoder}\\0\end{bmatrix},\quad
H=\begin{bmatrix}e_{f,XY}^T&0&0\\e_{lat,XY}^T&0&0\end{bmatrix}.$$

$$S_k=HP^-H^T+R,\quad K=P^-H^TS_k^{-1},\quad x=x^-+K(z-Hx^-).$$

代码用 Joseph 形式更新协方差：

$$P=(I-KH)P^-(I-KH)^T+KRK^T.$$

前向观测标准差为 0.025 m/s、横向为 0.06 m/s。前向创新过大时增大 R；马氏距离超过 100 时拒绝更新。打滑期间不会盲目认定高速空转等于车辆高速移动。

离地时只有加速度预测，不使用轮速或横向零滑移约束。位置状态是 $s\leftarrow s+\hat v\Delta t$ 的有符号路径积分，不是绝对世界定位；长时间腾空、持续打滑仍会漂移。当前偏置近似为世界水平随机游走，没有替代完整的惯性导航 ESKF。

## 5. LQR：保留你的权重，改变测量来源

状态顺序保持：

$$x=[\theta,\dot\theta,s,\hat v,\varphi,\dot\varphi]^T.$$

`test.m` 仍是权重的唯一来源：

```matlab
Q_cost = diag([1,1,500,100,5000,1]);
R_cost = diag([1,0.25]);
```

`balance_control.design` 在 0.20–0.38 m 上按 0.01 m 间隔，利用实际模型的质量矩阵、重力和闭环运动学求静态平衡，再线性化。`dlqr` 用零阶保持精确离散化动力学和二次型代价，解离散 Riccati 方程。

`load_gains` 校验物理 XML 的 SHA256、Q、R、步长；变更这些项后应重新运行：

```powershell
python balance_control.py --design --headless --seconds 2
```

运行时在相邻腿长之间线性插值 K 和静态前馈，计算：

$$[T,T_p]^T=[T_{eq},T_{p,eq}]^T+K(h)(x_{ref}-x).$$

T、Tp 都是整车总量，分给两侧时除以 2。不同腿长的平衡腿角不严格为零；实际几何与质心决定 `trim`，不能机械地把它删掉。

速度指令限制为 ±2 m/s，变化率 0.6 m/s²；转向指令限制为 ±0.5 rad/s，变化率 0.8 rad/s²。参考路径误差限制在 ±0.12 m，离地和保护切换时重新锚定路径，避免继续累计追赶旧目标的力矩。

## 6. VMC：从期望支撑力变成 L1、L5 电机力矩

虚功为：

$$\tau_a^T\delta q_a=F\delta h+T_p\delta\alpha,$$

结合 $[\delta h,\delta\alpha]^T=J\delta q_a$，得到：

$$\tau_a=J^T\begin{bmatrix}F\\T_p\end{bmatrix}.$$

这里 F 的单位为 N、Tp 与关节力矩为 N·m。J 的第一行是 m/rad，第二行是 rad/rad，不能直接把笛卡尔雅可比替换进此式。

普通接地腿长控制：

$$F_i=F_{eq}/2+1800(h_{d,i}-h_i)-65\dot h_i+150I_i\pm F_{roll}.$$

积分仅在普通接地、未饱和时累加，积分量限制 ±0.08 m·s；离地、失能和切换会清零。`d.ctrl` 对 gear=1 的 motor 就是关节输出轴力矩。髋电机限制 ±40 N·m、轮电机 ±4.92 N·m；前者仍是仿真假设，不是已经确认的实机性能。

## 7. 前后左右、双腿协调与腿高自适应

### 7.1 转向和防劈叉

航向目标由角速度积分得到：

$$\psi_d\leftarrow\operatorname{wrap}(\psi_d+\dot\psi_d\Delta t).$$

$$T_{yaw}=\operatorname{clip}(2\operatorname{wrap}(\psi_d-\psi)+0.7(\dot\psi_d-\dot\psi),\pm1.5).$$

左右轮分别为 $T_L=T/2-T_{yaw}$、$T_R=T/2+T_{yaw}$。

双腿协调项：

$$T_{sync}=\operatorname{clip}(80(\alpha_L-\alpha_R)+8(\dot\alpha_L-\dot\alpha_R),\pm8).$$

左右腿角力矩分别减、加此项，抑制转向时双腿向相反方向摆动。

### 7.2 自动适应左右地面高差

同时接地时，用姿态与两腿的相对位置重建左右轮心的垂直差 $\Delta z=z_{w,R}-z_{w,L}$。实现没有读取地面高度图来设定腿长。

相同轮半径、近水平姿态下：

$$h_{d,L}=h_d+\Delta z/2,\qquad h_{d,R}=h_d-\Delta z/2.$$

这使高地一侧缩腿、低地一侧伸腿，给机体横滚调平留出余量。高差采用一阶低通，左右偏置变化率限制为 0.025 m/s，并将两腿目标各自限制到 0.20–0.35 m。离地时冻结地面估计。

横滚力补偿为 $F_{roll}=\operatorname{clip}(500\gamma+35\dot\gamma,\pm80)$，左右腿反号叠加。它修正外扰和简化模型残差。当前验证是小角度横坡；高差超过腿行程、复杂台阶或大幅横滚不能由上述近似保证通过。

关闭自适应可用 `--no-adaptive-height`。共同腿长仍可手动调整。

## 8. 支持力估计与离地检测

### 8.1 从电机反馈反求虚拟力

对实际 `actuator_force`，用最小二乘求：

$$[F,T_p]^T=(J^T)^\dagger\tau_{feedback}.$$

本代码 Tp 正方向与论文简图的竖直力分解符号约定不同。从轮心向量 $[-h\sin\theta,-h\cos\theta]^T$ 的虚功展开，腿对轮竖直向下的载荷为：

$$P\approx F\cos\theta-\frac{T_p}{h}\sin\theta.$$

因此不要把论文中对应符号的加号不加转换地照抄进代码。

### 8.2 轮心加速度与支持力

$$\ddot z_w=a_{hip,z}-\ddot h\cos\theta+2\dot h\dot\theta\sin\theta+h\ddot\theta\sin\theta+h\dot\theta^2\cos\theta,$$

$$\hat F_N=P+m_w(g+\ddot z_w).$$

长度、角速度的差分先低通，再进入此式；最终支持力也低通。该近似忽略腿的分布质量与部分动态项，不能把它当成真实六维力传感器。本模型静止时它与直接仿真接触力有数牛顿差异。

离地阈值为 12 N、接地阈值为 25 N；连续低支持力超过 8 ms 判离地，连续高支持力超过 20 ms 判接地。滞回与时间确认降低阈值附近频繁切换。论文示例用 20 N，当前阈值增加了工程滞回。

默认 `fused` 另外用 MuJoCo 接触力和接触存在性作辅助：无接触或法向力低于 5 N 会触发卸载条件；接地确认要求存在接触且法向力大于 15 N。论文估计仍计算并记录。`estimated` 去掉普通运动切换里的这个接触辅助；两种模式的结果分开保存。

### 8.3 为什么离地后不会继续追轮速

代码通过 `airborne_lqr` 显式创建零矩阵，仅复制 K₂₁/K₂₂。θ 是虚拟腿倾角，φ/pitch 是机体倾角；论文保留的是前者。矩阵索引、控制边界及参数修改见 [修改教程](坡道加速度与离地增益教程.md)。

双轮离地后 T 直接置零，左右轮电机均为零力矩。Tp 只保留 K 的第二行前两项：

$$T_{p,air,i}=-\frac12(K_{21}\theta+K_{22}\dot\theta),$$

并限幅到 ±6 N·m，再加入双腿协调项。位置、速度、机体俯仰对应的 LQR 列不再参与空中输出。腿长使用无重力前馈的软弹簧阻尼，限制 ±70 N，准备落地；并非继续把站立支撑力施加给悬空腿。

轮力矩为零不等于轮速必然为零：已有角动量、被动运动仍可使它转动。防跑飞的含义是不给悬空轮持续的追速驱动力，超速时进一步失能；不能用这个策略抵消任意外力。

## 9. 跳跃和落地状态机

| 状态 | 控制含义 | 主要出口 |
|---|---|---|
| GROUND | 普通 LQR、VMC、自适应 | 跳跃请求、卸载或故障 |
| CROUCH | 腿长缓慢收到 0.20 m | 到达目标且至少 0.6 s 后起跳；3 s 超时取消 |
| THRUST | 每腿 F=260−12·腿长速度，优先分配姿态力矩 | 200 ms 或腿长超过 0.37 m |
| AIR | 零轮力矩、空中腿角和软腿长控制 | 确认发生离地后检测到触地 |
| SINGLE | 接地腿提供支撑，悬空腿零轮力矩并伸腿寻找接触 | 双轮接地或双轮离地 |
| LANDING | 提高长度阻尼、优先分配支撑力 | 双轮接地且持续 0.4 s 后回 GROUND |
| DISABLED | 六电机零力矩、冻结参考和积分 | 明确恢复请求，或满足条件的一次自动电机起身 |
| RECOVERY | 收腿、连续转髋、触地撑起和姿态稳定 | 满足稳定条件后进入 LANDING；失败保持失能 |

小跳请求要求双轮接地、俯仰与横滚小于 0.08 rad、估计速度小于 0.08 m/s、与上次跳跃间隔超过 3 s。手柄 A 仅上升沿触发。

起跳依据冲量关系 $M\Delta v_z\approx\int(F_L\cos\theta_L+F_R\cos\theta_R-Mg)dt$。以固定受限推力和短持续时间实现小跳，没有承诺指定厘米数的精确跳高控制。

进入 AIR 并不自动证明已经腾空，代码额外记录确实离地，避免刚结束蹬地仍接触地面时误切为落地；未实际腾空的情况有超时返回站立的处理。

空中前 0.08 s 腿长目标 0.30 m，随后伸到 0.34 m 准备缓冲。落地控制为：

$$F_i=F_{eq}/2+2200(\max(0.30,h_{d,i})-h_i)-90\dot h_i\pm F_{roll}.$$

落地时先按 39 N·m 的关节预算限制径向支撑力，再根据剩余关节力矩范围约束 Tp，最后统一执行电机限幅。这样不会让过大的角度修正挤掉承重力矩，导致腿迅速折叠。

单轮支撑模式不是空中模式：接地腿增加支撑前馈，悬空腿用受限软伸腿寻找地面。它用于过渡，不能视为已经实现持续单轮平衡行走。

### 起跳力矩预算更新

起跳先限制虚拟角力矩 Tp 到 ±15 N·m，再由每个髋电机约束 `|J_h F + J_alpha Tp| ≤ 39` 求允许的最大 F；不是先用径向力耗尽全部预算。接地轮的驱动力矩进一步限制为 `min(4.92, 0.65*N*R)`，无实际接触时立即为零。完整跳高测量见 [当前验收](速度跳跃与飞坡验收.md)。

## 10. 防跑飞与特殊起身究竟做了什么

### 10.1 保护触发条件

- 非有限状态或传感器数据；无效输入按急停处理。
- 俯仰或横滚超过 0.60 rad。
- 估计前向速度超过 3.0 m/s，或轮相对关节速度超过 65 rad/s。
- 运动模式下腿长超出 0.17–0.43 m，或闭环条件数超过 1e5。
- site 闭合误差超过 0.008 m。
- 普通 GROUND 连续力矩饱和超过 0.8 s（`torque_saturation_timeout`，由原 0.3 s 放宽）。电机输出限幅仍保持原值。

触发后直接六电机归零，不能仅抛异常让上一帧力矩继续存在。非有限 qpos/qvel 时停止继续调用物理步，避免依赖 MuJoCo 自动重置。0.17–0.43 m 是异常保护边界，不是扩大的 LQR 设计范围。

指令超过 0.25 s 未刷新、手柄断连或松开 LB：运动参考归零，并按斜率限制减速；不是急停。B/E 才是立即失能。

### 10.2 仅电机起身

旧版外力扶正和 `simulation_reset` 已删除。`MotorRecovery` 按 FOLD → SWEEP → EXTEND → PUSH → RIGHT 控制四个髋电机与两个轮电机；参考角连续展开以实现整圈转髋，关节误差不能折回最短角。力矩仍限制为髋 ±40 N·m、轮 ±4.92 N·m。

SWEEP 后通过闭链雅可比 `tau=J.T@[F,Tp]` 将径向支撑与姿态力矩映射到髋电机；接触载荷限制轮端牵引。超速先进入 SETTLE 零输出等待，机体翻正但腹部着地时 REPACK 重新收腿布置支点。总计最多 24 s、两次重新布置，失败保持失能，不修改物理状态。

完整阶段判据、公式、参数及实测边界见 [电机起身与功能地形](电机起身与功能地形.md)。四种前后倒地/塌腿测试成功；纯侧倒暂未实现，保护失能。

### 10.3 独立场地

`terrain_scene.py` 从 `terrains/*.json` 加载真实静态碰撞体，不修改机器人基础 XML。使用 `--terrain slope`、`--terrain launch` 或 `--terrain course`；自定义场地用 `--terrain-file`。旧坡道已低速通行验证；当前 20 cm 飞坡尚未通过，组合场地的台阶尚未完成通行验收。

## 11. 测试、数据和如何判断成功

当前验收见 [当前速度、跳跃与飞坡验收](速度跳跃与飞坡验收.md)。`verify_motor_recovery.py` 设置一次初始倒地姿态，随后逐步验证控制器没有改写 qpos/qvel/time、没有外加恢复力、髋输出不超过 40 N·m。成功必须返回 GROUND 并继续稳定 3 s。侧倒的预期结果是失能，不能计为起身成功。

`verify_terrain.py` 验证场景碰撞高度并生成预览；`verify_terrain_drive.py` 保存斜坡和飞坡通行轨迹。`verify_advanced.py` 的 stand/turn/jump 和十项接口测试已在本次重新运行。其他旧报告仅代表当时版本，不作为当前全场景通过的证据。

## 12. 后续强化学习怎么接

### 12.1 先选择清晰的学习目标

建议先做有界残差控制：保持当前 LQR/VMC 的站立能力，让策略学习小范围修正，例如腿长偏置、虚拟力修正和转向修正。跳跃更高、非平地落地、物理翻身应分别建立任务与验收；不要让一个奖励函数同时含糊地追求所有动作。

具体插入位置是在 `HybridController.control` 生成 F、Tp、Tw 之后、VMC 映射和最终限幅之前。示例未来动作：

$$a\in[-1,1]^6\mapsto[\Delta F_L,\Delta F_R,\Delta T_{p,L},\Delta T_{p,R},\Delta T_{w,L},\Delta T_{w,R}],$$

初始可将范围设为 ±20 N、±1 N·m、±0.3 N·m，再通过验证决定是否扩大。这些是建议起点，当前尚未实现残差策略或训练结果。

离地时仍屏蔽轮力矩，失能时屏蔽全部动作，最终限幅在残差叠加之后执行，保护不交给策略自行学习。

### 12.2 定义观察量

策略只接收未来真实机器人能测到或估到的量：

- 机体重力方向/姿态、角速度、估计水平速度。
- 主动髋角和角速度、轮角速度、腿长/腿角及其速度。
- 支持力估计、左右接地标志、控制状态。
- 速度/转向/高度目标、上一帧动作。

不要把 MuJoCo 精确世界速度、接触真值、外部施加的扰动力或隐藏地形高度图不加区分地放进实际策略输入。训练 critic 可以考虑特权信息，但 actor 应保持可部署观测；部署前必须移除仿真辅助依赖。

### 12.3 环境接口

创建 `gymnasium.Env`：`reset(seed=...)` 初始化物理状态、滤波器、状态机、积分器和随机数；`step(action)` 以例如 50 Hz 的策略动作驱动 20 个 1 ms 控制步，然后返回：

```python
observation, reward, terminated, truncated, info
```

跌倒、保护触发或不可恢复状态可设 `terminated=True`；到达固定训练时长是 `truncated=True`。不要把训练过程中的 reset 当作自主起身成功。

平衡策略训练时关闭自动恢复并在跌倒时终止 episode，避免恢复策略掩盖失败；起身策略训练可使用当前电机状态机作为基线或示范。当前已无恢复外力和状态复位路径。正常 episode reset 只能出现在环境边界。

### 12.4 奖励和约束

可从下式开始设计并分别记录每项：

$$r=w_v e^{-(v-v_d)^2/\sigma_v^2}+w_\psi e^{-(\dot\psi-\dot\psi_d)^2/\sigma_\psi^2}
-w_p\varphi^2-w_r\gamma^2-w_h\|h-h_d\|^2-w_u\|a\|^2-w_{\Delta u}\|a-a_{prev}\|^2.$$

另加失稳惩罚、关节极限/闭环残差约束。跳跃任务奖励应同时要求真实离地、有限腾空高度、正确落地与落地后持续平衡，不能只奖励机体 z 坐标增加。物理翻身任务必须禁用辅助外力和 qpos 写入。

PPO 是可尝试的连续动作基线，并非保证优于现有控制器。先用零动作验证环境能复现本次测试，再训练残差；同时保留零残差基线作对照。

### 12.5 课程学习和从仿真到实机

按定高站立 → 前后速度 → 转向 → 变腿长 → 小坡面 → 小扰动 → 小跳/落地逐步增加难度。随机化质量、惯量、摩擦、关节阻尼、传感器噪声、时间延迟、扭矩限制、模型几何误差和地面条件；特别需要标定当前按几何估计的惯量。

不要在修改质量/惯量后悄悄沿用“已精确匹配新模型”的 LQR 说法：固定名义 K 做鲁棒性训练，或重新设计 K，是两种不同实验设定，应记录清楚。

部署前先做未见随机种子的独立仿真验收，记录跌倒率、速度误差、能耗、离地轮速和最大力矩，再做硬件在环和受控实机实验。当前只提供接入路线，没有安装训练框架、训练策略或宣称已完成 sim-to-real。

## 13. 来源与校验依据

- 本地《轮腿式平衡机器人控制_陈阳.pdf》PDF 第 8–11 页（期刊页码 655–658）：速度解算、综合运动与离地检测；第 11 页结论说明跳跃和翻倒自恢复未实现。
- 本地 `轮腿机器人控制.md` 第 4.4 节：加速度与轮速融合思想；本实现扩展为二维速度加偏置估计。
- 本地 `test.m`：用户指定 LQR Q/R；本地 `力控接口说明.md`：力矩控制接口；旧说明中的版本和参数以当前 XML/Python 为准。
- [Microsoft XInput 入门](https://learn.microsoft.com/en-us/windows/win32/xinput/getting-started-with-xinput)与 [XINPUT_GAMEPAD](https://learn.microsoft.com/en-us/windows/win32/api/xinput/ns-xinput-xinput_gamepad)：状态读取、摇杆、按钮和死区。
- [MuJoCo 传感器实现](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_sensor.c)与[计算过程](https://mujoco.readthedocs.io/en/latest/computation/)：传感器和接触动力学；本机实际 API 已用 MuJoCo 3.9.0 验证。
- [Gymnasium Env 接口](https://gymnasium.farama.org/api/env/)与[终止/截断的区别](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)：未来训练环境的生命周期。

## 14. 代码定位

| 文件 | 函数/类 | 行 |
|---|---|---:|
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:14) | `airborne_lqr` | 14 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:25) | `SensorRobot` | 25 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:26) | `__init__` | 26 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:47) | `contact_forces` | 47 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:59) | `attitude` | 59 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:67) | `VelocityKF` | 67 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:69) | `__init__` | 69 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:72) | `update` | 72 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:96) | `SupportObserver` | 96 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:98) | `__init__` | 98 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:101) | `update` | 101 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:126) | `HybridController` | 126 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:127) | `__init__` | 127 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:134) | `initialize` | 134 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:145) | `submit` | 145 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:153) | `transition` | 153 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:171) | `disable` | 171 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:173) | `recovery` | 173 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:183) | `control` | 183 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:333) | `load_gains` | 333 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:339) | `step` | 339 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:348) | `main` | 348 |
| [advanced_control.py](E:/Document/RoboMaster/Wheelleg/advanced_control.py:366) | `key` | 366 |
| [motor_recovery.py](E:/Document/RoboMaster/Wheelleg/motor_recovery.py:5) | `MotorRecovery` | 5 |
| [motor_recovery.py](E:/Document/RoboMaster/Wheelleg/motor_recovery.py:6) | `__init__` | 6 |
| [motor_recovery.py](E:/Document/RoboMaster/Wheelleg/motor_recovery.py:13) | `switch` | 13 |
| [motor_recovery.py](E:/Document/RoboMaster/Wheelleg/motor_recovery.py:15) | `repack` | 15 |
| [motor_recovery.py](E:/Document/RoboMaster/Wheelleg/motor_recovery.py:19) | `fail` | 19 |
| [motor_recovery.py](E:/Document/RoboMaster/Wheelleg/motor_recovery.py:21) | `update` | 21 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:13) | `wrap` | 13 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:15) | `costs` | 15 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:26) | `Robot` | 26 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:27) | `__init__` | 27 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:39) | `kin` | 39 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:41) | `jac` | 41 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:43) | `leg` | 43 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:67) | `forward_leg` | 67 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:72) | `residual` | 72 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:80) | `inverse` | 80 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:82) | `res` | 82 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:91) | `pose` | 91 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:103) | `tangent` | 103 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:109) | `allocation` | 109 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:116) | `reduced` | 116 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:122) | `phi_angles` | 122 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:126) | `gap` | 126 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:129) | `dlqr` | 129 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:140) | `design` | 140 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:143) | `equilibrium` | 143 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:163) | `Controller` | 163 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:164) | `__init__` | 164 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:169) | `schedule` | 169 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:173) | `set_vertical_height` | 173 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:175) | `error` | 175 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:179) | `initialize` | 179 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:182) | `control` | 182 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:215) | `run` | 215 |
| [balance_control.py](E:/Document/RoboMaster/Wheelleg/balance_control.py:244) | `key` | 244 |
| [terrain_scene.py](E:/Document/RoboMaster/Wheelleg/terrain_scene.py:9) | `add_terrain` | 9 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:7) | `Command` | 7 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:16) | `deadzone` | 16 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:20) | `Gamepad` | 20 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:23) | `State` | 23 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:26) | `XInput` | 26 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:28) | `__init__` | 28 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:37) | `decode` | 37 |
| [gamepad_control.py](E:/Document/RoboMaster/Wheelleg/gamepad_control.py:45) | `poll` | 45 |
