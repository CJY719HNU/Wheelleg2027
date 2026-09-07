# -*- coding: utf-8 -*-
"""
显示机器人当前(建模/零位)状态

用法:
    python show_model.py                      # 打开交互窗口显示 real_wheelleg_fixed_base_rotated.xml
    python show_model.py 文件.xml             # 显示指定模型
    python show_model.py --save 预览.png      # 不弹窗，离屏渲染一张图后退出(用于无窗口/CI)
"""
import os
import sys
import time
import argparse

import numpy as np
import mujoco
import mujoco.viewer

# Windows 控制台默认是 GBK，强制用 UTF-8 输出，避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OBJ = mujoco.mjtObj


def body_name(model, i):
    return mujoco.mj_id2name(model, OBJ.mjOBJ_BODY, i) or f"body{i}"


def joint_name(model, j):
    return mujoco.mj_id2name(model, OBJ.mjOBJ_JOINT, j) or f"joint{j}"


def print_state(model, data):
    """打印每个铰链所在连杆的世界位置与转轴，方便与孔位坐标核对。"""
    line = "=" * 62
    print(line)
    print(f"bodies/geoms : {model.nbody} / {model.ngeom}")
    print("joints/qpos0 : " + ", ".join(
        f"{joint_name(model, j)}={data.qpos[model.jnt_qposadr[j]]:.4f}"
        for j in range(model.njnt)))
    print("-" * 62)
    print("带铰链的连杆(关节轴在 body 原点上, 世界坐标):")
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        b = model.jnt_bodyid[j]
        xp = data.xpos[b]
        ax = model.jnt_axis[j]
        print(f"  {body_name(model, b):<10s} x={xp[0]: .4f} y={xp[1]: .4f} z={xp[2]: .4f}"
              f"    axis=({ax[0]:+.1f} {ax[1]:+.1f} {ax[2]:+.1f})")
    print(line)


def frame_camera(model, data):
    """按各 body 位置粗略框住模型, 返回 (相机注视点, 距离)。"""
    pts = np.array([data.xpos[i] for i in range(1, model.nbody)])  # 跳过 worldbody
    if pts.size == 0:
        return np.zeros(3), 1.5
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2.0
    radius = float(np.max(hi - lo))
    return center, 1.5 * radius + 0.6


def render_offscreen(model, data, out_png):
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth = 1280   # 提高离屏 framebuffer, 否则默认只有 640
    model.vis.global_.offheight = 960
    renderer = mujoco.Renderer(model, height=960, width=1280)
    try:
        renderer.update_scene(data)
        img = renderer.render()
    finally:
        renderer.close()
    try:
        from PIL import Image
    except Exception:
        import png  # 极少情况: 用 pypng
        with open(out_png, "wb") as f:
            png.Writer(width=img.shape[1], height=img.shape[0]).write(f, img.reshape(-1, img.shape[2] * img.shape[1]))
    else:
        Image.fromarray(img).save(out_png)
    print(f"已保存渲染图: {out_png}")


def launch_viewer(model, data, center, distance):
    """打开交互式查看器, 仅显示当前位形, 不做任何控制/步进。"""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            viewer.cam.lookat = center.tolist()
            viewer.cam.distance = distance
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -45.0
            mujoco.mj_forward(model, data)
            viewer.sync()
        except Exception:
            pass
        print("查看器已打开: 拖动旋转 / 滚轮缩放 / 右键平移, 关闭窗口即退出。")
        while viewer.is_running():
            time.sleep(0.01)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xml", nargs="?", default=None,
                    help="要显示的 xml(默认取本目录 real_wheelleg_fixed_base_rotated.xml)")
    ap.add_argument("--save", metavar="PNG", default=None,
                    help="离屏渲染到 PNG 后退出(不弹窗)")
    args = ap.parse_args()

    xml_path = args.xml or os.path.join(HERE, "real_wheelleg_fixed_base_rotated.xml")
    if not os.path.exists(xml_path):
        print(f"找不到模型文件: {xml_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0      # 零位 = XML 定义的建模姿态
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print_state(model, data)
    center, dist = frame_camera(model, data)

    if args.save:
        render_offscreen(model, data, args.save)
        return

    try:
        launch_viewer(model, data, center, dist)
    except Exception as e:
        print(f"无法打开交互窗口({e}), 改为保存离屏渲染图 preview.png")
        render_offscreen(model, data, os.path.join(HERE, "preview.png"))


if __name__ == "__main__":
    main()
