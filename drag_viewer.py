# -*- coding: utf-8 -*-
"""拖拽测试: 力矩腿(无位置伺服)+持续步进.
用法: python drag_viewer.py
在窗口里: 双击选中某根杆/轮 -> 按住拖动 即可徒手拉. ESC 退出.
"""
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("real_wheelleg_drag.xml")
d = mujoco.MjData(m)
# 腿给 0 力矩 + 一点点阻尼(防乱飘, 又不锁死)
print("双击杆件并拖动 = 徒手拉腿; ESC 退出")
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        d.ctrl[:] = 0.0
        # 微小阻尼让杆不至于乱甩: 给全部关节加小阻尼(可选)
        for i in range(m.nu):
            pass
        mujoco.mj_step(m, d)
        v.sync()
