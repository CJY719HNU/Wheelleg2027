# 轮腿机器人 MuJoCo 仿真与 LQR/VMC 自平衡 —— 完整使用文档

> 目录结构基于 `real_wheelleg_fixed_base.xml`(左右五连杆轮腿)自上而下建立:
> 对称镜像模型 → 地面版(重力+轮地接触) → 缩聚 LQR 自平衡控制器。

---

## 0. 环境要求

```bash
pip install mujoco numpy scipy
python --version          # 建议 3.10+
```

Windows 终端中文乱码时先执行:

```bash
set PYTHONIOENCODING=utf-8        # CMD
export PYTHONIOENCODING=utf-8     # Git Bash / PowerShell
```

---

## 1. 文件清单与分工

| 文件 | 作用 | 何时用 |
|---|---|---|
| `real_wheelleg_fixed_base.xml` | **原始模型**(固定基座、零重力、无碰撞),不要改动它 | 参照 / 备份 |
| `real_wheelleg_sym.xml` | 对称副本:左腿 = 右腿关于 x=0 的镜像(以右腿为准) | 生成地面版的前置 |
| `real_wheelleg_ground.xml` | **地面版**:自由基座 `root`、重力 −9.81、地面、轮地接触;底座显式 8 kg | **主仿真模型** |
| `stl-wheellegyuelu26/` | STL 零件(含镜像件 `sym_L*.STL`) | 网格资源 |
| `assets/` | 指向 `stl-wheellegyuelu26` 的目录联接(XML 里 `meshdir="assets"`) | 加载依赖 |
| `joint_points.csv` | 机构关键铰点坐标(运动学参考) | IK/轨迹 |
| `_build_sym.py` | 由原模型生成对称副本 + 镜像 STL | 重建对称版 |
| `_make_ground.py` | 由对称版生成地面版 | 重建地面版 |
| **`balance_lqr_pose.py`** | **主控制器**:webm 姿态自平衡 LQR + 腿高闭环 | **日常运行** |
| `wheelleg_balance.py` | 参考版:完全对齐 `使用Python对XML进行控制.md` / `test.m`(解析模型,含 `K(l)`/速度矩阵占位) | 学习/对照 |
| `balance_lqr_full.py` | 旧“直立姿态”数值探针版本(被主控制器取代,保留参考) | 参考 |
| `使用Python对XML进行控制.md` | md/test.m 教程(控制器思路出处) | 阅读 |
| `Screencast ... .webm` | 你录的期望姿态参考 | 姿态对照 |

> 提示:删除 `assets/` 联接或换机器后若报 `Error opening file 'assets/L4.stl'`,重建:
> `cmd /c mklink /J assets stl-wheellegyuelu26`(或在 XML 把 `meshdir` 改成实际文件夹)。

---

## 2. 快速开始

### 2.1 看自平衡(推荐)

```bash
python balance_lqr_pose.py --viewer     # 弹窗,ESC 退出
python balance_lqr_pose.py              # 无窗口,12s 逐秒打印,看是否收敛
```

预期:机器人保持 **webm 姿态(绕轮轴 x 整体 +90°、两轮着地)**,轮矩 LQR 保持姿态,腿根据 CoM 高度自动微调。离线 12s 俯仰通常 ≤ ±3°。

### 2.2 看纯姿态预览

```bash
python balance_lqr_full.py --viewer     # 旧版直立姿态(参考)
python balance_lqr_pose.py --viewer     # 当前 webm 姿态
```

---

## 3. 模型是怎么一步步来的(想重建/改模型时)

```mermaid
real_wheelleg_fixed_base.xml
        │  python _build_sym.py
        ▼
real_wheelleg_sym.xml   (左腿=右腿镜像,左右严格对称,0.000mm 级)
        │  python _make_ground.py
        ▼
real_wheelleg_ground.xml
   (base 加 free joint [root];重力 0 0 -9.81;
    地面 plane z=0;左右轮加圆柱碰撞体 r=0.068,cont=1,只与地面接触;
    base 显式惯性 8 kg → 整车 ≈10.2 kg;底盘/连杆仍不碰撞,允许穿地)
```

- 生成脚本只在顶层文件改动时重跑:`python _build_sym.py` → `python _make_ground.py`。
- 任何控制器都从 `real_wheelleg_ground.xml` 加载,别直接改固定基座那份。

### 3.1 关键建模决定(易踩坑)

| 项 | 值 | 说明 |
|---|---|---|
| 底座质量 | 显式 8 kg | STL 会自动算 ~109 kg,轮子带不动,必须给 `<inertial>` |
| 轮半径 | 0.068 m | 由 `RW.STL` 实测 |
| 车身几何 | 车底到基座系 z≈−0.316 m | 直接“直立”会让车底插地,故姿态按 webm(绕 x +90°) |
| 站立姿态 | base 绕轮心整体转 +90°,legs=0,轮着地(轮微下陷 3 mm 预压) | 与 webm 一致,CoM 真正在轮上方 |
| 轮矩上限 | ±4.92 N·m/轮(XML `ctrlrange`) | 抗扰上限由它决定 |

---

## 4. 主控制器 `balance_lqr_pose.py` 详解

### 4.1 控制结构

```
外环(腿高): HREF 对比实测 CoM 高度 → 输出腿指令 off = KLP*(HREF-h)+KLI*∫
             后曲柄 setpoint=+off, 前曲柄 setpoint=−off   (变等效腿长保持高度)
内环(轮矩 LQR): 状态 [θ, θ̇, v]  + 速度/俯仰积分配平
             状态 θ = CoM-轮心连线相对竖直的倾角(前倾为正); θ̇ = −ω_x
             模型: 缩聚倒立摆 A(摆长 L 由实测 CoM-轮轴高) + 动力学探针测 B
             控制 u = −K·x + KIV·∫v + KIT·∫θ   →  左右轮同力矩 u
```

### 4.2 参数(文件顶部)

| 参数 | 默认 | 作用 |
|---|---|---|
| `Q_DIAG=[θ,θ̇,v]` | `[900, 600, 60]` | 状态权重(越大越“硬”) |
| `R_VAL` | `1.8` | 轮矩惩罚(越小越激进,越小越抖) |
| `LEG_DAMP` | `1.0` | 腿关节虚拟阻尼(真实齿轮摩擦的等效) |
| `W_DAMP` | `0.15` | 轮轴粘性阻尼(压高频细震) |
| `U_SMOOTH` | `0.12` | 轮矩低通(0~1),抑制 bang-bang |
| `CTRL_EVERY` | `5` | 控制降频(200 Hz) |
| `KIV` | `2.0` | 速度积分 |
| `KIT` | `3.0` | 俯仰积分配平(消静差) |
| `KLP` / `KLI` | `5.0 / 0.6` | 腿高 P / I |

### 4.3 调参口诀

| 现象 | 改法 |
|---|---|
| 俯仰在平衡点小幅摇、不抖 | ↑`Q_DIAG[θ̇]` 或 ↓`R_VAL` |
| 高频细抖 | ↑`W_DAMP` / ↑`U_SMOOTH`(或 ↑`CTRL_EVERY`) |
| 被推一下就倒(抗扰弱) | ↑`Q_DIAG[θ]`、↓`R_VAL`,或**提高轮矩上限**(需改 XML `ctrlrange`) |
| 高度回不去/腿不动 | ↑`KLP`;先微增 `KLI` |
| 响应慢 | ↓`R_VAL`、↑`U_SMOOTH` 会反直觉地更钝,取 0.1~0.3 |

### 4.4 物理边界(重要)

- 抗扰上限 ≈ 由 **轮矩 ±4.92 N·m** 与摆长 L≈0.29 m 决定;实测能扛 ~1 N·m 级别俯仰扰动。
- 想扛更大推力:①放宽 XML 轮电机 `ctrlrange`;②或给腿加“辅助推地”通道。

---

## 5. 参考版 `wheelleg_balance.py`(与 md/test.m 对齐)

用途:对照 `使用Python对XML进行控制.md` 理解 LQR 写法的“解析模型版”,含两个【FILL】占位:

```bash
python wheelleg_balance.py            # 解析模型 + K(l) 调度(需在真实模型上调参)
python wheelleg_balance.py --viewer
```

| 占位 | 含义 |
|---|---|
| `【FILL 1】VELOCITY_MATRIX` | 期望速度命令矩阵,每行 `[t(s), 前进速度m/s]` |
| `【FILL 2】LEG_LUT_X / LEG_LUT_K` | 你的 `K(l)` 增益表(节点摆长 l + 每档 1×4),填了就插值;默认每 10 mm 解 LQR→多项式拟合 |

> 该版状态顺序为 `[x, ẋ, θ, θ̇]`(test.m 顺序),连续 LQR(照 `test.m` 的 `lqr`)。**它只是设计对照,直接跑需要按真实 plant 调参。**

---

## 6. 常见问题 FAQ

| 现象 | 处理 |
|---|---|
| `Error opening file 'assets/L4.stl'` | 重建 `assets` 联接,或改 XML `meshdir`(见 §1 提示) |
| 中文乱码 | `set PYTHONIOENCODING=utf-8` |
| 加载报 XML 错误 | 用生成脚本重跑 `_build_sym.py` / `_make_ground.py` 保证 xml 一致 |
| 一跑就倒(发散) | 先跑 `python balance_lqr_pose.py` 无窗口看日志;调 §4.3;确认不是轮子上限/扰动过大 |
| 轮子高频细震 | ↑`W_DAMP`、↑`U_SMOOTH` 或 ↑`CTRL_EVERY` |
| 想换站立腿高工作点 | 改 `balance_lqr_pose.py` 的初始 pose(legs setpoint)+ `HREF`,或扩展“目标高度轨迹”接口 |

---

## 7. 一键命令速查

```bash
# 预览/自平衡(webm 姿态)
python balance_lqr_pose.py --viewer
python balance_lqr_pose.py

# 参考版(md/test.m 对齐)
python wheelleg_balance.py --viewer

# 重建模型(改了顶层 XML 后)
python _build_sym.py
python _make_ground.py
```

---

## 8. 下一步(可能的扩展)

- 高度目标随命令变化(高姿态站立 / 低姿态行驶);
- 提高轮矩上限或加腿推力通道增强抗扰;
- 前进/转向命令(`VELOCITY_MATRIX` 接入主控制器);
- 与 `wheel_legged_robot_sim/src`(数值线性化/腿高 LUT)进一步整合。

---

## 9. 探索存档:力矩 VMC(王洪玺/跃鹿文档架构)的实验结论

> 这一节只存档“为什么当前主交付选用了腿 setpoint 架构”,不参与日常运行。

**目标**:按跃鹿文档做“6 态 2 输入 LQR + VMC(把虚拟髋力分到前后曲柄力矩)”。
**结论(实验得出)**:

1. `mj_forward` 直接扰动某根曲柄**不算闭链约束** → 数值雅可比无效(后曲柄对轮位无影响),力矩≈0。
2. 对整段工作空间用**单个 2D 三次多项式**拟合 `L0/φ0(qr,qf)` **不泛化**:网格外随机 60 点验证,腿长误差最大 106 mm、摆角最大 150°(强非线性 + arctan 相位跳变 + 奇异区)。→ 力矩 VMC 发散的主因是**运动学映射不对**,不只是增益。
3. 因此要做到位必须先做**解析 pantograph 闭链正运动学**(可按真机杆长:上腿 0.335、下腿 0.251,四连杆 `AE=0`),并做 mm 级验证后再接双输入 LQR。

**遗留实验文件**(当前不稳定/未收敛,仅存档,勿当正式交付):
- `real_wheelleg_ground_torque.xml` — 4 曲柄改成力矩电机的实验模型;
- `balance_vmc.py`(v1→v4)— 闭链多项式 FK、双输入探针 LQR、VMC 分矩的实验链。

**正式主交付(已验证收敛)**:`balance_lqr_pose.py` —— webm 姿态、轮矩 LQR 稳俯仰 + 腿高闭环(高度误差 → 后曲柄+/前曲柄−),12 s 稳定、连杆主动伸缩保持高度。
