# -*- coding: utf-8 -*-
"""
浮体落地演示 + 明亮环境

运行:  python land_demo.py

行为:
    - 打开 MuJoCo 原生 Simulate 窗口(已调亮: 4 盏灯 + 提亮材质)
    - 机器人从 base z=0.12 处开始受重力下落
    - 触地后不停机，持续仿真；随时可旋转视角查看
    - 按键: R 重放下落 / Esc 退出

说明: 该腿只有 2 轮、无前后脚, 落地后没有控制器, 会自然倾倒,
      这是结构特性; 本脚本目的是演示“下落→触地”过程与亮环境。
"""
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
DROP_Z = 0.12       # 起始 base 高度(m)
REALTIME = 1.0      # 速度倍率


def main():
    xml = os.path.join(HERE, "real_wheelleg_sim_float_bright.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    free_adr = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_adr = model.jnt_qposadr[j]  # x y z qw qx qy qz
            break

    def reset():
        mujoco.mj_resetData(model, data)
        if free_adr is not None:
            data.qpos[free_adr + 2] = DROP_Z   # 抬高起点
        mujoco.mj_forward(model, data)

    reset()
    reported = False

    def on_key(key):
        ch = key & 0x7f
        if ch in (ord('r'), ord('R')):
            reset()

    print("正在打开 MuJoCo Simulate(明亮环境)，机器人将从 %.2f m 处下落…" % DROP_Z)
    print("触地后不停机、持续仿真; 按键: R=重放下落  Esc=退出。")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        try:  # 初始视角对准机器人
            viewer.cam.lookat = [0.0, -0.03, -0.25]
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 55.0
            viewer.cam.elevation = -25.0
            viewer.sync()
        except Exception:
            pass

        last = time.time()
        while viewer.is_running():
            now = time.time()
            elapsed = (now - last) * REALTIME
            last = now
            n = max(int(elapsed / model.opt.timestep), 1)  # 每帧至少一步
            for _ in range(n):
                mujoco.mj_step(model, data)
            if not reported and data.ncon > 0:
                reported = True
                print("已触地(接触点 %d)，持续仿真中。" % data.ncon)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()
