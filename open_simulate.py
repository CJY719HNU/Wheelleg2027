# -*- coding: utf-8 -*-
"""
在原生 MuJoCo Simulate 界面中打开模型(带可拖动基座 + 地面背景)

用法:
    python open_simulate.py                      # 打开 real_wheelleg_sim.xml
    python open_simulate.py 其他.xml             # 打开任意模型

界面内怎么拖:
    双击某个 body    -> 选中并高亮
    Ctrl + 左键拖    -> 转动选中的 body (暂停时直接摆姿态 / 运行时施加力矩)
    Ctrl + 右键拖    -> 在 x-z 竖直平面内平移(加力)
    Ctrl + Shift+右键 -> 在 x-y 水平面内平移
    F1              -> 查看该版本完整键位帮助

    基座是 6 自由度(freejoint): 双击基座后 Ctrl 拖动即可整机移动。
    想要暂停/继续物理: 按 空格。
"""
import os
import sys

import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    xml = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "real_wheelleg_sim.xml")
    if not os.path.exists(xml):
        print("找不到模型文件:", xml)
        sys.exit(1)
    print("正在用原生 MuJoCo Simulate 打开:", xml)
    print("提示: 双击选中 body, 然后 Ctrl+左键/右键 拖动; 按 F1 看完整帮助。")
    mujoco.viewer.launch_from_path(xml)   # 会阻塞, 直到关闭窗口


if __name__ == "__main__":
    main()
