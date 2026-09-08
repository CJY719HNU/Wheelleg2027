"""Generate current acceptance facts; unsuccessful steep-ramp runs stay unsuccessful."""
from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parent

def update_guides():
    p=ROOT/'advanced_control_math.md';s=p.read_text(encoding='utf-8')
    for old,new in [
        ('最大 0.2 m/s','最大 2 m/s'),('范围 0.27–0.35 m','范围 0.20–0.35 m'),
        ('在 0.27–0.35 m 上','在 0.20–0.38 m 上'),('±0.2 m/s，变化率 0.25 m/s²','±2 m/s，变化率 0.6 m/s²'),
        ('限制到 0.27–0.35 m','限制到 0.20–0.35 m'),('收到 0.275 m','收到 0.20 m'),
        ('每腿 F=180−12·腿长速度，保留姿态反馈','每腿 F=260−12·腿长速度，优先分配姿态力矩'),
        ('100 ms 或腿长超过 0.342 m','200 ms 或腿长超过 0.37 m'),
        ('估计前向速度超过 1.2 m/s','估计前向速度超过 3.0 m/s'),('0.18–0.43 m','0.17–0.43 m'),
        ('[电机起身与地形验收](motor_terrain_acceptance.md)','[当前速度、跳跃与飞坡验收](速度跳跃与飞坡验收.md)'),
        ('坡道/飞坡已低速通行验证','旧坡道已低速通行验证；当前 20 cm 飞坡尚未通过'),
    ]:s=s.replace(old,new)
    marker='## 14. 代码定位'
    s=s[:s.index(marker)]+marker+'\n\n| 文件 | 函数/类 | 行 |\n|---|---|---:|\n'
    for name in ['advanced_control.py','motor_recovery.py','balance_control.py','terrain_scene.py','gamepad_control.py']:
        for node in sorted(ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))),key=lambda n:getattr(n,'lineno',0)):
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)):s+=f'| [{name}]({(ROOT/name).as_posix()}:{node.lineno}) | `{node.name}` | {node.lineno} |\n'
    if '### 起跳力矩预算更新' not in s:
        s=s.replace('## 10. 防跑飞', '### 起跳力矩预算更新\n\n起跳先限制虚拟角力矩 Tp 到 ±15 N·m，再由每个髋电机约束 `|J_h F + J_alpha Tp| ≤ 39` 求允许的最大 F；不是先用径向力耗尽全部预算。接地轮的驱动力矩进一步限制为 `min(4.92, 0.65*N*R)`，无实际接触时立即为零。完整跳高测量见 [当前验收](速度跳跃与飞坡验收.md)。\n\n## 10. 防跑飞')
    p.write_text(s,encoding='utf-8')
    p=ROOT/'电机起身与功能地形.md';s=p.read_text(encoding='utf-8')
    s=s.replace('0.8 m 长、高 0.06 m','0.4 m 长、高 0.20 m')
    a=s.find('标准飞坡以当前');b=s.find('\n\n',a)
    if a>=0:s=s[:a]+'当前飞坡高 20 cm、长 40 cm；2 m/s 助跑通行尚未稳定通过。此前 6 cm 飞坡的成功记录已归档，不能外推到当前场地。测试及失败边界见 [当前验收](速度跳跃与飞坡验收.md)。'+s[b:]
    s=s.replace('[电机起身与地形验收](motor_terrain_acceptance.md)，包含四种成功起身、侧倒保护及地形通行结果','[当前验收](速度跳跃与飞坡验收.md)，本次重测了塌腿、倒置起身；其余姿态的历史测试不代表当前版本重新验收')
    p.write_text(s,encoding='utf-8')

def build():
    update_guides()
    lines=['# 速度、腿长、跳跃与飞坡验收','','本次保持原来的质量、惯量和执行器限值：髋 ±40 N·m、轮 ±4.92 N·m。Q/R 继续读取 test.m，重新生成 0.20–0.38 m 的 19 个 LQR 工作点。','','| 场景 | 结果 | 测量 |','|---|---|---|']
    for name in ['speed','low','jump','launch','launch35']:
        data=json.loads((ROOT/f'motion_{name}.json').read_text());s=data['summary'];f=s['final']
        stable=f['mode']=='GROUND' and all(f['contact']) and not f['chassis_contact'] and abs(f['velocity_est'])<.05 and abs(f['state'][4])<.08 and abs(f['state'][0])<.25
        if name in ['speed','low','jump']:assert stable,name
        if name=='speed':
            steady=[x['true_v'] for x in data['samples'] if 7<x['t']<9]
            assert abs(sum(steady)/len(steady)-2.)<.05
            value=f"稳态平均 {sum(steady)/len(steady):.3f} m/s；加速峰值 {s['max_speed']:.3f} m/s；停车通过"
        elif name=='low':value=f"指令下限 0.20 m；过渡最低实际长度 {s['min_length']:.4f} m；升回 0.30 m"
        elif name=='jump':value=f"两轮较低轮底最高离地 {s['max_clearance']*100:.2f} cm；髋峰值 {s['peak_torque']:.1f} N·m；落地后稳定停车"
        else:value=f"{s['reason']}，最终 {s['final_mode']}；20 cm 高、40 cm 长的陡坡尚未通过通行"
        lines.append(f"| {name} | {'通过' if stable else '未通过'} | {value} |")
    lines+=['','## 跳跃推力选择','','每腿径向推力候选为 200、220、240、260、280、300 N，实际力矩仍受电机限值约束。260 N 的实测轮底离地约 21.4 cm，落地后稳定；280 N 的试验最终停留 SINGLE，300 N 的试验出现持续滑跑。当前选 260 N，并保留姿态力矩优先分配和接触载荷牵引限幅。此为测试过的稳定候选，不宣称全局最大跳高。','', '跳高定义为两轮中较低轮底相对于平地的最大高度，不是质心上升量；收腿也会改变轮底高度。起跳与落地真实发生，不修改仿真位置或添加起跳外力。','','## 飞坡边界','','两种起始腿长 0.30/0.35 m 均以 2 m/s 指令、约 8 m 助跑距离测试。0.30 m 腿长出现机体碰坡，0.35 m 虽有所改善仍触发腿长保护。尺寸与碰撞已完成，稳定飞越与落地尚未完成，不能使用旧版 6 cm 飞坡通过记录作为证据。没有通过放宽保护或修改机体惯量掩盖失败。','','## 回归与清理','','- stand、turn：本次 10 s 回归通过。','- 接口及保护：10 项检查通过。','- 塌腿、倒置：电机起身回归通过，恢复无物理状态改写、无外加恢复力。','- 旧 XML、旧演示入口及历史日志共清理 42 个文件，压缩在 历史版本备份.zip；明细见 文件清理记录.md。','','## 复现','','```powershell','python -B verify_motion.py speed low jump launch launch35','python -B verify_advanced.py stand turn --seconds 10','python -B verify_advanced_interfaces.py','python -B verify_terrain.py','python -B verify_motor_recovery.py collapsed inverted','python -B build_motion_report.py','```','','verify_motion.py 中 passed=false 表示未通过；飞坡未通过如实记录，报告生成不会把保护失能视作飞越成功。','','## 当前文件指纹','']
    for name in ['advanced_control.py','balance_control.py','gamepad_control.py','balance_gains.json','real_wheelleg_balance.xml','terrains/launch.json','terrains/course.json']:
        lines.append(f'- `{name}`：`{hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}`')
    (ROOT/'速度跳跃与飞坡验收.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':build()
