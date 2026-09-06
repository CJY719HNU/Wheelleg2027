# -*- coding: utf-8 -*-
"""最简: 打开模型并仿真, 不加控制, 关节可自由拖动.

4 个曲柄位置伺服从"锁0"改成"跟随当前角"(跟随模式) → 拖动后不会弹回.
双击选中杆件并按住拖动 = 拉它看各关节联动.
ESC 退出.
"""
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("real_wheelleg_fixed_base.xml")
d = mujoco.MjData(m)

# 把 position 型伺服设成"跟随"：目标 = 当前角度
pos_acts = []
for i in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
    if name.endswith("_drive_pos"):
        jnt = m.actuator_trnid[i, 0]
        pos_acts.append((i, m.jnt_qposadr[jnt]))

with mujoco.viewer.launch_passive(m, d) as view:
    print("双击选中杆件并拖动; ESC 退出")
    while view.is_running():
        for ai, qa in pos_acts:
            d.ctrl[ai] = d.qpos[qa]      # 跟随当前角 → 拖了不弹回
        mujoco.mj_step(m, d)
        view.sync()
