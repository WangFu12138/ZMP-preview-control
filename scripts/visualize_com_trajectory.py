import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# 配置中文字体支持 - 使用字体文件路径方式
try:
    # 方法1：尝试使用系统中文字体
    chinese_fonts = [f for f in fm.fontManager.ttflist 
                     if any(name in f.name for name in ['Noto Sans CJK', 'Noto Serif CJK', 'AR PL', 'Droid'])]
    if chinese_fonts:
        font_prop = fm.FontProperties(fname=chinese_fonts[0].fname)
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f"使用字体: {chinese_fonts[0].name} ({chinese_fonts[0].fname})")
    else:
        # 方法2：直接指定常见中文字体路径
        font_paths = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
        ]
        for font_path in font_paths:
            try:
                fm.fontManager.addfont(font_path)
                prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = prop.get_name()
                print(f"使用字体文件: {font_path}")
                break
            except:
                continue
except Exception as e:
    print(f"字体配置警告: {e}，将使用默认字体")
    
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 辅助函数：确保轨迹数据为 (N, 3) 格式
def ensure_xyz_format(arr, name):
    """将轨迹数据转换为 (N, 3) 格式，如果是 (N, 2) 则补齐 Z=0"""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} 数据维度错误: {arr.shape}，期望 (N, 2) 或 (N, 3)")
    if arr.shape[1] == 2:
        # 补齐Z轴为0
        arr = np.hstack([arr, np.zeros((arr.shape[0], 1))])
    elif arr.shape[1] != 3:
        raise ValueError(f"{name} 数据列数错误: {arr.shape[1]}，期望2或3列")
    return arr

# 加载三条轨迹数据
CoM_trajectory = ensure_xyz_format(
    np.load('/home/wzn/双足/Biped-Locomotion/CoM_trajectory.npy'), "质心轨迹")
foot_left = ensure_xyz_format(
    np.load('/home/wzn/双足/Biped-Locomotion/footTrj_left.npy'), "左脚轨迹")
foot_right = ensure_xyz_format(
    np.load('/home/wzn/双足/Biped-Locomotion/footTrj_right.npy'), "右脚轨迹")

# 分析CoM轨迹的Y值极值
y_min_idx = np.argmin(CoM_trajectory[:, 1])  # Y最小值的索引
y_max_idx = np.argmax(CoM_trajectory[:, 1])  # Y最大值的索引

y_min_value = CoM_trajectory[y_min_idx, 1]  # Y最小值
y_max_value = CoM_trajectory[y_max_idx, 1]  # Y最大值

x_at_y_min = CoM_trajectory[y_min_idx, 0]   # Y最小时的X坐标
x_at_y_max = CoM_trajectory[y_max_idx, 0]   # Y最大时的X坐标

z_at_y_min = CoM_trajectory[y_min_idx, 2]   # Y最小时的Z坐标
z_at_y_max = CoM_trajectory[y_max_idx, 2]   # Y最大时的Z坐标

# ========== CoM Y轴正弦波周期和幅度分析 ==========
from scipy import signal

# 提取Y轴数据
com_y = CoM_trajectory[:, 1]
com_x = CoM_trajectory[:, 0]

# 寻找所有波峰和波谷
peaks, _ = signal.find_peaks(com_y, distance=50)  # 波峰
troughs, _ = signal.find_peaks(-com_y, distance=50)  # 波谷

# 计算周期（连续两个波峰或波谷之间的距离）
if len(peaks) > 1:
    periods_peaks = np.diff(peaks)  # 波峰间隔
    avg_period_steps = np.mean(periods_peaks)
    # 转换为X方向的距离（步长）
    x_periods = [com_x[peaks[i+1]] - com_x[peaks[i]] for i in range(len(peaks)-1)]
    avg_period_distance = np.mean(x_periods) if x_periods else 0
else:
    avg_period_steps = 0
    avg_period_distance = 0

if len(troughs) > 1:
    periods_troughs = np.diff(troughs)
    avg_period_steps_troughs = np.mean(periods_troughs)
else:
    avg_period_steps_troughs = 0

# 计算每个周期的幅度（峰值到谷值的一半）
amplitudes = []
for i in range(min(len(peaks), len(troughs))):
    # 找到每对相邻的峰谷
    if i < len(troughs):
        amp = (com_y[peaks[i]] - com_y[troughs[i]]) / 2
        amplitudes.append(amp)

# 计算整体幅度
overall_amplitude = (y_max_value - y_min_value) / 2

# 分析幅度衰减趋势（如果有多个周期）
if len(amplitudes) > 2:
    amplitude_decay_rate = (amplitudes[-1] - amplitudes[0]) / len(amplitudes)
else:
    amplitude_decay_rate = 0

# ========== Y值符号变换点分析 ==========
# 检测Y值从正到负或从负到正的变换点
zero_crossings = []  # 存储符号变换点的索引
crossing_types = []  # 存储变换类型：'pos_to_neg' 或 'neg_to_pos'

for i in range(1, len(com_y)):
    # 检测符号变化
    if com_y[i-1] > 0 and com_y[i] <= 0:
        # 从正变负
        zero_crossings.append(i)
        crossing_types.append('正→负')
    elif com_y[i-1] < 0 and com_y[i] >= 0:
        # 从负变正
        zero_crossings.append(i)
        crossing_types.append('负→正')
    elif com_y[i-1] == 0 and i > 1:
        # 处理恰好为0的情况
        if com_y[i-2] > 0 and com_y[i] < 0:
            zero_crossings.append(i)
            crossing_types.append('正→负')
        elif com_y[i-2] < 0 and com_y[i] > 0:
            zero_crossings.append(i)
            crossing_types.append('负→正')

# 获取变换点的详细信息
zero_crossing_details = []
for idx, crossing_idx in enumerate(zero_crossings):
    detail = {
        '序号': crossing_idx,
        'X坐标': com_x[crossing_idx],
        'Y值': com_y[crossing_idx],
        '变换类型': crossing_types[idx]
    }
    zero_crossing_details.append(detail)

# 打印数据概览
print("=" * 70)
print("CoM轨迹数据分析")
print("=" * 70)
print(f"质心轨迹形状: {CoM_trajectory.shape}")
print(f"数据范围: X[{CoM_trajectory[:, 0].min():.3f}, {CoM_trajectory[:, 0].max():.3f}] "
      f"Y[{CoM_trajectory[:, 1].min():.3f}, {CoM_trajectory[:, 1].max():.3f}] "
      f"Z[{CoM_trajectory[:, 2].min():.3f}, {CoM_trajectory[:, 2].max():.3f}]")
print()
print("CoM Y方向极值分析:")
print("-" * 70)
print(f"Y最小值: {y_min_value:.6f} m (时间步: {y_min_idx})")
print(f"  对应X坐标: {x_at_y_min:.6f} m")
print(f"  对应Z坐标: {z_at_y_min:.6f} m")
print(f"  完整坐标: [{x_at_y_min:.6f}, {y_min_value:.6f}, {z_at_y_min:.6f}]")
print()
print(f"Y最大值: {y_max_value:.6f} m (时间步: {y_max_idx})")
print(f"  对应X坐标: {x_at_y_max:.6f} m")
print(f"  对应Z坐标: {z_at_y_max:.6f} m")
print(f"  完整坐标: [{x_at_y_max:.6f}, {y_max_value:.6f}, {z_at_y_max:.6f}]")
print()
print(f"Y方向摆动幅度: {y_max_value - y_min_value:.6f} m")
print(f"X方向跨度(在Y极值点间): {x_at_y_max - x_at_y_min:.6f} m")
print()
print("CoM Y轴正弦波分析:")
print("-" * 70)
print(f"检测到的波峰数量: {len(peaks)}")
print(f"检测到的波谷数量: {len(troughs)}")
if len(peaks) > 0:
    print(f"波峰位置(时间步): {peaks.tolist()}")
    print(f"波峰位置(X坐标): {[f'{com_x[p]:.3f}' for p in peaks]}")
if len(troughs) > 0:
    print(f"波谷位置(时间步): {troughs.tolist()}")
    print(f"波谷位置(X坐标): {[f'{com_x[t]:.3f}' for t in troughs]}")
print()
if avg_period_steps > 0:
    print(f"平均周期 (时间步): {avg_period_steps:.1f} 步")
    print(f"平均周期 (X方向距离): {avg_period_distance:.6f} m")
else:
    print("无法计算周期（数据点不足）")
print()
print(f"整体幅度: {overall_amplitude:.6f} m")
if len(amplitudes) > 0:
    print(f"各周期幅度:")
    for i, amp in enumerate(amplitudes):
        print(f"  周期 {i+1}: {amp:.6f} m")
    print(f"平均幅度: {np.mean(amplitudes):.6f} m")
    print(f"幅度标准差: {np.std(amplitudes):.6f} m")
    if len(amplitudes) > 2:
        print(f"幅度衰减率: {amplitude_decay_rate:.6f} m/周期")
        print(f"初始幅度: {amplitudes[0]:.6f} m")
        print(f"最终幅度: {amplitudes[-1]:.6f} m")
        print(f"幅度衰减: {((amplitudes[-1] - amplitudes[0]) / amplitudes[0] * 100):.2f}%")
print()
print("CoM Y值符号变换点分析:")
print("-" * 70)
print(f"检测到的符号变换点数量: {len(zero_crossings)}")
if len(zero_crossings) > 0:
    print(f"\n变换点详细信息:")
    print(f"{'#':<5} {'时间步':<10} {'X坐标(m)':<15} {'Y值(m)':<15} {'变换类型':<10}")
    print("-" * 70)
    for i, detail in enumerate(zero_crossing_details, 1):
        print(f"{i:<5} {detail['序号']:<10} {detail['X坐标']:<15.6f} {detail['Y值']:<15.6f} {detail['变换类型']:<10}")
    
    # 统计变换类型
    pos_to_neg_count = sum(1 for t in crossing_types if t == '正→负')
    neg_to_pos_count = sum(1 for t in crossing_types if t == '负→正')
    print()
    print(f"统计信息:")
    print(f"  正→负 变换: {pos_to_neg_count} 次")
    print(f"  负→正 变换: {neg_to_pos_count} 次")
    
    # 计算相邻变换点间的距离
    if len(zero_crossings) > 1:
        intervals_steps = np.diff(zero_crossings)
        intervals_x = [com_x[zero_crossings[i+1]] - com_x[zero_crossings[i]] 
                       for i in range(len(zero_crossings)-1)]
        print(f"\n相邻变换点间隔:")
        print(f"  平均时间步间隔: {np.mean(intervals_steps):.1f} 步")
        print(f"  平均X方向距离: {np.mean(intervals_x):.6f} m")
        print(f"  最小间隔: {np.min(intervals_steps)} 步 ({np.min(intervals_x):.6f} m)")
        print(f"  最大间隔: {np.max(intervals_steps)} 步 ({np.max(intervals_x):.6f} m)")
else:
    print("未检测到Y值符号变换点")
print("=" * 70)
print()
print(f"左脚轨迹形状: {foot_left.shape} | "
      f"X[{foot_left[:, 0].min():.3f}, {foot_left[:, 0].max():.3f}] "
      f"Y[{foot_left[:, 1].min():.3f}, {foot_left[:, 1].max():.3f}] "
      f"Z[{foot_left[:, 2].min():.3f}, {foot_left[:, 2].max():.3f}]")
print(f"右脚轨迹形状: {foot_right.shape} | "
      f"X[{foot_right[:, 0].min():.3f}, {foot_right[:, 0].max():.3f}] "
      f"Y[{foot_right[:, 1].min():.3f}, {foot_right[:, 1].max():.3f}] "
      f"Z[{foot_right[:, 2].min():.3f}, {foot_right[:, 2].max():.3f}]")
print("=" * 70)

# 创建图形
fig = plt.figure(figsize=(15, 10))

# 1. 3D轨迹图
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot(CoM_trajectory[:, 0], CoM_trajectory[:, 1], CoM_trajectory[:, 2], 
         color='tab:blue', linestyle='-', linewidth=2.0, alpha=0.8, label='质心 CoM')
ax1.plot(foot_left[:, 0], foot_left[:, 1], foot_left[:, 2], 
         color='tab:orange', linestyle='--', linewidth=1.8, alpha=0.8, label='左脚')
ax1.plot(foot_right[:, 0], foot_right[:, 1], foot_right[:, 2], 
         color='tab:green', linestyle='-.', linewidth=1.8, alpha=0.8, label='右脚')
# 标注质心起点和终点
ax1.scatter(CoM_trajectory[0, 0], CoM_trajectory[0, 1], CoM_trajectory[0, 2], 
            c='darkblue', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
ax1.scatter(CoM_trajectory[-1, 0], CoM_trajectory[-1, 1], CoM_trajectory[-1, 2], 
            c='darkblue', s=120, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
ax1.set_xlabel('X (米)')
ax1.set_ylabel('Y (米)')
ax1.set_zlabel('Z (米)')
ax1.set_title('三维轨迹对比：质心与左右脚')
ax1.legend(loc='best')
ax1.grid(True)

# 2. XY平面投影（俯视图）
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(CoM_trajectory[:, 0], CoM_trajectory[:, 1], 
         color='tab:blue', linestyle='-', linewidth=2.0, alpha=0.8, label='质心 CoM')
ax2.plot(foot_left[:, 0], foot_left[:, 1], 
         color='tab:orange', linestyle='--', linewidth=1.8, alpha=0.8, label='左脚')
ax2.plot(foot_right[:, 0], foot_right[:, 1], 
         color='tab:green', linestyle='-.', linewidth=1.8, alpha=0.8, label='右脚')
# 标注Y最小值和Y最大值点
ax2.scatter(x_at_y_min, y_min_value, color='red', s=150, marker='v', 
            edgecolors='black', linewidths=2, zorder=10, label=f'Y最小 ({x_at_y_min:.3f}, {y_min_value:.3f})')
ax2.scatter(x_at_y_max, y_max_value, color='purple', s=150, marker='^', 
            edgecolors='black', linewidths=2, zorder=10, label=f'Y最大 ({x_at_y_max:.3f}, {y_max_value:.3f})')
# 标注Y值符号变换点
if len(zero_crossings) > 0:
    zero_x = com_x[zero_crossings]
    zero_y = com_y[zero_crossings]
    ax2.scatter(zero_x, zero_y, color='orange', s=80, marker='x', 
                linewidths=2, zorder=11, label=f'Y过零点 (n={len(zero_crossings)})')
    # 添加Y=0参考线
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
ax2.set_xlabel('X (米)')
ax2.set_ylabel('Y (米)')
ax2.set_title('XY平面投影（俯视图）- 标注Y极值点')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True)
ax2.axis('equal')

# 3. XZ平面投影（侧视图）
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(CoM_trajectory[:, 0], CoM_trajectory[:, 2], 
         color='tab:blue', linestyle='-', linewidth=2.0, alpha=0.8, label='质心 CoM')
ax3.plot(foot_left[:, 0], foot_left[:, 2], 
         color='tab:orange', linestyle='--', linewidth=1.8, alpha=0.8, label='左脚')
ax3.plot(foot_right[:, 0], foot_right[:, 2], 
         color='tab:green', linestyle='-.', linewidth=1.8, alpha=0.8, label='右脚')
ax3.set_xlabel('X (米)')
ax3.set_ylabel('Z (米)')
ax3.set_title('XZ平面投影（侧视图）')
ax3.legend(loc='best')
ax3.grid(True)

# 4. 各轴随时间变化（支持不同长度轨迹）
ax4 = fig.add_subplot(2, 2, 4)
t_com = np.arange(len(CoM_trajectory))
t_left = np.arange(len(foot_left))
t_right = np.arange(len(foot_right))

# 质心XYZ
ax4.plot(t_com, CoM_trajectory[:, 0], color='tab:blue', linestyle='-', alpha=0.9, label='质心 X')
ax4.plot(t_com, CoM_trajectory[:, 1], color='tab:blue', linestyle='--', alpha=0.9, linewidth=2.5, label='质心 Y')
ax4.plot(t_com, CoM_trajectory[:, 2], color='tab:blue', linestyle=':', alpha=0.9, label='质心 Z')

# 标注Y轴的波峰和波谷
if len(peaks) > 0:
    ax4.scatter(peaks, com_y[peaks], color='red', s=100, marker='^', 
                edgecolors='black', linewidths=1.5, zorder=10, label=f'Y波峰 (n={len(peaks)})')
if len(troughs) > 0:
    ax4.scatter(troughs, com_y[troughs], color='purple', s=100, marker='v', 
                edgecolors='black', linewidths=1.5, zorder=10, label=f'Y波谷 (n={len(troughs)})')

# 标注Y值符号变换点（过零点）
if len(zero_crossings) > 0:
    ax4.scatter(zero_crossings, com_y[zero_crossings], color='orange', s=80, marker='x', 
                linewidths=2.5, zorder=11, label=f'Y过零点 (n={len(zero_crossings)})')
    # 添加水平参考线Y=0
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

# 左脚XYZ
ax4.plot(t_left, foot_left[:, 0], color='tab:orange', linestyle='-', alpha=0.5, linewidth=1, label='左脚 X')
ax4.plot(t_left, foot_left[:, 1], color='tab:orange', linestyle='--', alpha=0.5, linewidth=1, label='左脚 Y')
ax4.plot(t_left, foot_left[:, 2], color='tab:orange', linestyle=':', alpha=0.5, linewidth=1, label='左脚 Z')

# 右脚XYZ
ax4.plot(t_right, foot_right[:, 0], color='tab:green', linestyle='-', alpha=0.5, linewidth=1, label='右脚 X')
ax4.plot(t_right, foot_right[:, 1], color='tab:green', linestyle='--', alpha=0.5, linewidth=1, label='右脚 Y')
ax4.plot(t_right, foot_right[:, 2], color='tab:green', linestyle=':', alpha=0.5, linewidth=1, label='右脚 Z')

ax4.set_xlabel('时间步')
ax4.set_ylabel('位置 (米)')
ax4.set_title('XYZ分量随时间变化 - 标注Y轴正弦波峰谷')
ax4.legend(loc='best', ncol=2, fontsize=7)
ax4.grid(True)

plt.tight_layout()
plt.show()
