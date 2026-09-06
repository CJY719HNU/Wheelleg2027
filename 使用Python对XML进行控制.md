# 使用 Python 对该 XML 进行控制

> 基于 MuJoCo 物理引擎 + LQR 最优控制的倒立摆仿真

---

## 目录

1. [系统概述](#1-系统概述)
2. [XML 模型结构](#2-xml-模型结构)
3. [LQR 控制器原理](#3-lqr-控制器原理)
4. [代码逐步讲解](#4-代码逐步讲解)
   - [4.1 加载 MuJoCo 模型](#41-加载-mujoco-模型)
   - [4.2 线性化状态空间推导](#42-线性化状态空间推导)
   - [4.3 求解 Riccati 方程](#43-求解-riccati-方程)
   - [4.4 传感器映射](#44-传感器映射)
   - [4.5 仿真主循环](#45-仿真主循环)
5. [Q/R 参数调优指南](#5-qr-参数调优指南)
6. [运行方式](#6-运行方式)
7. [与 MATLAB 的协同工作流](#7-与-matlab-的协同工作流)

---

## 1. 系统概述

### 1.1 物理系统

| 部件 | 参数 | 说明 |
|------|------|------|
| 小车 (Cart) | 0.5 kg | 沿 x 轴水平滑动，行程 ±5 m |
| 摆杆 (Pole) | 0.1 kg，长 0.5 m | 绕小车顶部铰接转动 |
| 电机 (Motor) | gear=10，限幅 ±10 | 驱动小车，gear 放大驱动力 |
| 传感器 ×4 | 位置/速度 | 测量小车位置/速度、摆杆角度/角速度 |

```
        θ →  (杆偏角)
         │
         ○  ← 铰链 (hinge joint)
        /│\
       / │ \   0.5m 杆
        │
    ┌───┴───┐
    │ 小车  │  ← 沿 x 轴滑动 (slider joint)
    └───────┘
  ═══════════════  地面
        ← x →
```

### 1.2 文件结构

```
self_bia/
├── 1.xml                          ← MuJoCo 模型定义 (物理参数、几何、传感器)
├── 1.py                           ← Python 控制脚本 (本文档讲解对象)
├── test.m                         ← MATLAB LQR 辅助设计脚本
└── 使用Python对XML进行控制.md       ← 本文档
```

---

## 2. XML 模型结构

XML 文件定义了仿真所需的全部物理元素，Python 代码通过 `mjModel` 加载它。

### 2.1 编译器与仿真选项

```xml
<compiler coordinate="local" inertiafromgeom="true"/>
<option timestep="0.002" gravity="0 0 -9.81"/>
```

| 属性 | 含义 |
|------|------|
| `coordinate="local"` | 子刚体的位姿相对父刚体定义 |
| `inertiafromgeom="true"` | MuJoCo 根据几何体形状自动计算转动惯量 |
| `timestep="0.002"` | 仿真步长 2 ms（500 Hz 控制频率） |
| `gravity="0 0 -9.81"` | z 轴向下，重力 9.81 m/s² |

### 2.2 刚体与关节

```xml
<body name="cart" pos="0 0 0.1">
  <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-5 5" damping="0.1"/>
  <geom name="cart" type="box" size="0.3 0.2 0.1" mass="0.5"/>

  <body name="pole" pos="0 0 0">
    <joint name="hinge" type="hinge" axis="0 1 0"/>
    <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
  </body>
</body>
```

关键点：
- **slider 关节** — 平动自由度，`damping="0.1"` 提供粘性阻尼，`range="-5 5"` 限制行程
- **hinge 关节** — 转动自由度，绕 y 轴旋转
- 杆是 `pole` 刚体嵌套在 `cart` 刚体内（父子关系），所以杆会随小车一起移动
- `inertiafromgeom="true"` 让 MuJoCo 自动从胶囊几何体计算杆的转动惯量

### 2.3 驱动器与传感器

```xml
<actuator>
  <motor name="motor" joint="slider" gear="10" ctrllimited="true" ctrlrange="-10 10"/>
</actuator>
<sensor>
  <jointpos name="cart_pos" joint="slider"/>
  <jointpos name="pole_angle" joint="hinge"/>
  <jointvel name="cart_vel" joint="slider"/>
  <jointvel name="pole_angvel" joint="hinge"/>
</sensor>
```

| 元素 | 作用 |
|------|------|
| `motor` | 直线电机，`gear=10` 将 `ctrl` 放大 10 倍作为驱动力 (N) |
| `ctrllimited="true"` | 启用控制限幅，`ctrlrange="-10 10"` 限制控制信号 |
| `jointpos` 传感器 | 读取关节位置（rad 或 m） |
| `jointvel` 传感器 | 读取关节速度（rad/s 或 m/s） |

---

## 3. LQR 控制器原理

### 3.1 什么是 LQR

**LQR**（Linear Quadratic Regulator）解决如下最优控制问题：

$$
\min_u \int_0^\infty \left( x^T Q x + u^T R u \right) dt
$$

约束条件：
$$
\dot{x} = A x + B u
$$

其中：
- $x \in \mathbb{R}^4$ — 状态向量
- $u \in \mathbb{R}^1$ — 控制输入
- $Q \succeq 0$ — **状态惩罚矩阵**（对角元素越大，对应状态越不能容忍偏差）
- $R > 0$ — **控制惩罚**（越大控制越保守，越小控制越激进）

**最优解**是状态反馈形式：
$$
u = -K x, \quad K = R^{-1} B^T P
$$

其中 $P$ 是**代数 Riccati 方程**的解：
$$
A^T P + P A - P B R^{-1} B^T P + Q = 0
$$

### 3.2 倒立摆的状态空间

状态向量定义（Python 中的顺序）：
$$
x = \begin{bmatrix} x \\ \theta \\ \dot{x} \\ \dot{\theta} \end{bmatrix}
\quad
\begin{aligned}
x &: \text{小车位置 (m)} \\
\theta &: \text{摆杆偏角 (rad)，竖直向上为 0} \\
\dot{x} &: \text{小车速度 (m/s)} \\
\dot{\theta} &: \text{摆杆角速度 (rad/s)}
\end{aligned}
$$

---

## 4. 代码逐步讲解

### 4.1 加载 MuJoCo 模型

```python
model = mujoco.MjModel.from_xml_path("1.xml")
data = mujoco.MjData(model)
```

**MjModel vs MjData：**

| 对象 | 性质 | 内容 |
|------|------|------|
| `MjModel` | **只读**，所有仿真实例共享 | 刚体树结构、几何参数、关节定义、驱动器/传感器配置、仿真选项 |
| `MjData` | **可读写**，每个仿真实例独有 | `qpos`（广义位置）、`qvel`（广义速度）、`ctrl`（控制输入）、`sensordata`（传感器读数） |

`model` 和 `data` 始终成对使用——`model` 提供"蓝图"，`data` 存储"运行时状态"。

### 4.2 线性化状态空间推导

#### 非线性动力学方程

由拉格朗日力学，倒立摆系统的运动方程为：

$$
\begin{aligned}
(M+m)\ddot{x} + ml\ddot{\theta}\cos\theta - ml\dot{\theta}^2\sin\theta + b\dot{x} &= F \\
(I+ml^2)\ddot{\theta} + ml\ddot{x}\cos\theta - mgl\sin\theta &= 0
\end{aligned}
$$

#### 线性化（θ ≈ 0 处）

在直立平衡点附近，$\cos\theta \approx 1$，$\sin\theta \approx \theta$，$\dot{\theta}^2 \approx 0$：

$$
\begin{aligned}
(M+m)\ddot{x} + ml\ddot{\theta} + b\dot{x} &= F \\
(I+ml^2)\ddot{\theta} + ml\ddot{x} - mgl\theta &= 0
\end{aligned}
$$

#### 解出加速度

写成矩阵形式：

$$
\begin{bmatrix} M+m & ml \\ ml & I+ml^2 \end{bmatrix}
\begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} =
\begin{bmatrix} F - b\dot{x} \\ mgl\theta \end{bmatrix}
$$

令 $\Delta = (M+m)(I+ml^2) - (ml)^2$，求逆：

$$
\begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} =
\frac{1}{\Delta}
\begin{bmatrix} I+ml^2 & -ml \\ -ml & M+m \end{bmatrix}
\begin{bmatrix} F - b\dot{x} \\ mgl\theta \end{bmatrix}
$$

展开得到状态空间矩阵：

$$
A = \begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & -\frac{(ml)^2 g}{\Delta} & -\frac{(I+ml^2)b}{\Delta} & 0 \\
0 & \frac{(M+m)mgl}{\Delta} & \frac{mlb}{\Delta} & 0
\end{bmatrix},
\quad
B = \begin{bmatrix}
0 \\ 0 \\ \frac{I+ml^2}{\Delta} \\ -\frac{ml}{\Delta}
\end{bmatrix}
$$

对应的 Python 代码：

```python
Delta = (M_cart + m_pole) * (I_pole + m_pole * l_half**2) - (m_pole * l_half)**2

A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, -(m_pole * l_half)**2 * g / Delta,
        -(I_pole + m_pole * l_half**2) * b_damp / Delta, 0],
    [0, (M_cart + m_pole) * m_pole * g * l_half / Delta,
        m_pole * l_half * b_damp / Delta, 0],
])

B = np.array([
    [0],
    [0],
    [(I_pole + m_pole * l_half**2) / Delta],
    [-m_pole * l_half / Delta],
])
```

**注意**：Python 的状态顺序是 `[x, θ, ẋ, θ̇]`，与 MATLAB (`test.m`) 的 `[x, ẋ, θ, θ̇]` 不同。K 矩阵在两者间需要**重新排列**：

```
K_python = [K_matlab(1), K_matlab(3), K_matlab(2), K_matlab(4)]
```

### 4.3 求解 Riccati 方程

```python
P = solve_continuous_are(A, B, Q, R)     # 求解 Riccati 方程
K = np.linalg.inv(R) @ B.T @ P           # 计算最优增益
K = K.flatten()                          # 展平为 1D 向量
```

- `scipy.linalg.solve_continuous_are` 实现了连续时间代数 Riccati 方程的数值求解
- 之后计算 $K = R^{-1} B^T P$ 得到反馈增益
- 最后验证闭环极点全部位于左半平面（实部 < 0），确保系统稳定

```python
A_cl = A - B @ K.reshape(1, -1)
eig_cl = np.linalg.eigvals(A_cl)
```

**Q/R 的选取逻辑**：

```python
Q = np.diag([10.0, 50000.0, 50.0, 50000.0])
#           x      θ       ẋ      θ̇
R = np.array([[0.05]])
```

| 权重 | 值 | 设计意图 |
|------|-----|---------|
| `Q(θ)` | 50000 | **最高优先级**— 杆角度是核心控制目标，偏差不可容忍 |
| `Q(θ̇)` | 50000 | **高优先级**— 大幅抑制角速度，消除平衡点附近的摇荡 |
| `Q(ẋ)` | 50 | 适度阻尼小车速度，但不过大以免阻碍小车移动去平衡杆 |
| `Q(x)` | 10 | 低优先级— 小车可以在 ±5 m 内自由移动，不影响控制目标 |
| `R` | 0.05 | 较小的控制惩罚，允许控制器"激进"地抑制震荡 |

### 4.4 传感器映射

```python
sensor_names = ["cart_pos", "pole_angle", "cart_vel", "pole_angvel"]
sensor_ids = [
    mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, name)
    for name in sensor_names
]
```

`mj_name2id` 将 XML 中定义的传感器**名称**映射为 `sensordata` 数组中的**索引**。

```
XML 中定义顺序: sensordata 数组:
  cart_pos    →  sensordata[0]  (由 mj_name2id 确定)
  pole_angle  →  sensordata[1]
  cart_vel    →  sensordata[2]
  pole_angvel →  sensordata[3]
```

之后通过 `data.sensordata[sensor_ids[i]]` 就能按名称逻辑读取传感器值。

### 4.5 仿真主循环

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # ① 读取状态
        x = np.array([data.sensordata[i] for i in sensor_ids])

        # ② LQR 控制律: u = -K·x
        u = -K @ x

        # ③ 施加到电机
        data.ctrl[0] = u

        # ④ 推进仿真一步 (0.002s)
        mujoco.mj_step(model, data)

        # ⑤ 刷新渲染窗口
        viewer.sync()
        time.sleep(0.001)
```

#### 数据流图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  sensordata  │ ──→ │  -K @ x      │ ──→ │  ctrl[0]     │
│  (4 个读数)   │     │  (LQR 控制律) │     │  (电机指令)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
       ↑                                         │
       │         ┌──────────────┐                 │
       └──────── │  mj_step()   │ ←───────────────┘
                 │  (物理仿真)   │
                 └──────────────┘
```

#### mj_step() 内部流程

```
mj_step(model, data):
  1. mj_fwdActuation()    — 根据 ctrl 计算驱动力（含 gear 变换、限幅）
  2. mj_fwdDynamics()     — 求解运动方程，得到加速度 qacc
  3. mj_Euler()           — 半隐式欧拉积分，更新 qpos, qvel
  4. mj_fwdPosition()     — 更新传感器数据 sensordata
```

#### 关键细节

- **`time.sleep(0.001)`**：不加这行仿真会以几千 FPS 跑完，画面太快无法观察。1ms 延迟 ≈ 1000 FPS 上限（实际受限于 0.002s 步长的 500Hz 物理频率）
- **初始偏角**：`data.qpos[1] = math.radians(2)` 给杆 2° 初始扰动，用于验证控制器能否镇定
- **500 步打印一次**（1 秒一次）：监控状态和控制的收敛情况

---

## 5. Q/R 参数调优指南

### 5.1 调参原则

| 现象 | 诊断 | 解决方案 |
|------|------|---------|
| 杆在平衡点附近**震荡** | 欠阻尼 | ↑ Q(θ̇) 或 ↓ R |
| 杆校正**很慢**，几乎倒下才反应 | Q(θ) 太小 或 R 太大 | ↑ Q(θ) 或 ↓ R |
| 小车**剧烈来回移动** | Q(x) 太大 或 Q(ẋ) 太小 | ↓ Q(x)，↑ Q(ẋ) |
| 控制信号**频繁达到 ±10 限幅** | R 太小，控制器过于激进 | ↑ R |
| 小车**漂移到轨道尽头** | Q(x) 太小，没有归零趋势 | ↑ Q(x) |
| 杆反应**僵硬、高频抖动** | Q(θ̇) 太大 | ↓ Q(θ̇) |

### 5.2 Bryson 法则（经验公式）

一种系统化的初始 Q/R 选择方法：

$$
Q_{ii} = \frac{1}{(\text{该状态最大允许偏差})^2}, \quad
R = \frac{1}{(\text{最大允许控制量})^2}
$$

例如：
- 允许角度偏差 3° ≈ 0.05 rad → $Q_{θθ} = 1/0.05^2 = 400$
- 允许力 5 N → $R = 1/5^2 = 0.04$

Bryson 值作为**起点**，然后根据实际效果缩放和微调。

### 5.3 调参流程

```
Bryson 初值 → 运行仿真 → 观察效果 → 调整 Q/R 比例 → 更新 K → 重复
                                  │
                          ┌───────┼───────┐
                    震荡？  响应太慢？  抖动？
                     ↑θ̇      ↑θ 或 ↓R    ↓θ̇ 或 ↑R
```

---

## 6. 运行方式

### 环境要求

```bash
pip install mujoco numpy scipy
```

### 启动仿真

```bash
cd d:\Document\Mujoco_Doc\self_bia
python 1.py
```

### 预期输出

```
============================================================
倒立摆 LQR 控制器
============================================================
物理参数: M=0.5kg  m=0.1kg  l=0.25m  b=0.1
开环不稳定极点: [...]

Q 对角值: [   10. 50000.    50. 50000.]
R 值:     0.05

LQR 增益矩阵 K = [...]
  K_x   =   ...  (小车位置反馈)
  K_θ   =   ...  (摆杆角度反馈)
  K_ẋ   =   ...  (小车速度反馈)
  K_θ̇  =   ...  (摆杆角速度反馈)

闭环极点 (应全部位于左半平面):
  λ1 = ...
  ...

传感器映射: [('cart_pos', 0), ('pole_angle', 1), ...]

仿真启动 — 按任意键或关闭窗口退出

t=1.0s | x=+...m | θ=+...° | ẋ=+...m/s | θ̇=+...rad/s | u=+...
t=2.0s | x=+...m | θ=+...° | ẋ=+...m/s | θ̇=+...rad/s | u=+...
...
```

---

## 7. 与 MATLAB 的协同工作流

`test.m` 可以作为 LQR 的辅助设计工具：

```
┌──────────────────────────────────────────────────┐
│  test.m (MATLAB)                                 │
│  ① 设定 Q, R                                     │
│  ② K = lqr(A, B, Q, R)                          │
│  ③ 检查闭环极点 eig(A-B*K)                        │
│  ④ 手动将 K 复制到 1.py                           │
└──────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│  1.py (Python)                                   │
│  K = np.array([从 MATLAB 粘贴的数值])              │
│  → 运行 MuJoCo 仿真                              │
│  → 在真实的非线性物理环境中验证                    │
└──────────────────────────────────────────────────┘
```

**状态顺序转换**（MATLAB → Python）：

| MATLAB: `[x, ẋ, θ, θ̇]` | Python: `[x, θ, ẋ, θ̇]` |
|--------------------------|--------------------------|
| `K_matlab(1)` — 小车位置 | `K_python[0]` — 小车位置 |
| `K_matlab(2)` — 小车速度 | `K_python[2]` — 小车速度 |
| `K_matlab(3)` — 摆杆角度 | `K_python[1]` — 摆杆角度 |
| `K_matlab(4)` — 摆杆角速度 | `K_python[3]` — 摆杆角速度 |

或者直接用 Python 的 `solve_continuous_are` 代替 MATLAB，在 Python 中完成全部 LQR 设计，避免手动转换。

---

> **参考**
> - [MuJoCo Documentation](https://mujoco.readthedocs.io/)
> - [1.py](1.py) — Python 控制脚本
> - [test.m](test.m) — MATLAB LQR 辅助设计
> - [1.xml](1.xml) — MuJoCo 模型定义
