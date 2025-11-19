import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

def load_polygon_data(filepath):
    """
    从pickle文件中加载多边形数据
    
    参数:
        filepath: 保存多边形数据的pickle文件路径
    
    返回:
        多边形数据列表，每个元素是一个numpy数组，包含多边形的顶点坐标
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件 {filepath} 不存在")
    
    with open(filepath, 'rb') as f:
        polygon_data = pickle.load(f)
    
    return polygon_data

def visualize_single_polygon(polygon_points, ax=None, color='blue', alpha=0.7, 
                           edge_color='black', point_color='red', point_size=20):
    """
    可视化单个多边形
    
    参数:
        polygon_points: 多边形顶点坐标，形状为(N, 2)或(N, 3)
        ax: matplotlib轴对象，如果为None则创建新的
        color: 多边形填充颜色
        alpha: 多边形透明度
        edge_color: 多边形边缘颜色
        point_color: 顶点颜色
        point_size: 顶点大小
    
    返回:
        ax: matplotlib轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # 如果是3D坐标，只取前两维(x, y)
    if polygon_points.shape[1] >= 3:
        points_2d = polygon_points[:, :2]
    else:
        points_2d = polygon_points
    
    # 绘制多边形
    if len(points_2d) >= 3:
        # 至少3个点才能构成多边形
        polygon = Polygon(points_2d, closed=True, facecolor=color, alpha=alpha, 
                         edgecolor=edge_color, linewidth=2)
        ax.add_patch(polygon)
    elif len(points_2d) == 2:
        # 只有两个点，绘制线段
        ax.plot(points_2d[:, 0], points_2d[:, 1], color=edge_color, linewidth=2)
    
    # 绘制顶点
    ax.scatter(points_2d[:, 0], points_2d[:, 1], color=point_color, s=point_size, zorder=5)
    
    # 标注顶点序号
    for i, (x, y) in enumerate(points_2d):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, color='black')
    
    # 设置坐标轴
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    return ax

def visualize_polygon_sequence(polygon_data, output_dir="./polygon_visualizations", 
                              save_frames=True, show_plot=True):
    """
    可视化多边形序列
    
    参数:
        polygon_data: 多边形数据列表
        output_dir: 输出图像保存目录
        save_frames: 是否保存每一帧图像
        show_plot: 是否显示图像
    """
    if save_frames and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_frames = len(polygon_data)
    print(f"总共 {num_frames} 帧多边形数据")
    
    # 为每个帧创建图像
    for i, polygon_points in enumerate(polygon_data):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 可视化当前帧的多边形
        visualize_single_polygon(polygon_points, ax=ax)
        
        ax.set_title(f'Polygon Frame {i+1}/{num_frames}')
        
        # 保存图像
        if save_frames:
            filename = os.path.join(output_dir, f"polygon_frame_{i:04d}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"已保存帧 {i+1} 到 {filename}")
        
        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

def plot_polygon_data_statistics(polygon_data):
    """
    绘制多边形数据统计信息
    
    参数:
        polygon_data: 多边形数据列表
    """
    # 计算每帧的顶点数
    vertex_counts = [len(polygon) for polygon in polygon_data]
    
    # 绘制顶点数变化图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(vertex_counts, marker='o', linestyle='-', markersize=4)
    plt.xlabel('Frame')
    plt.ylabel('Number of Vertices')
    plt.title('Vertex Count per Frame')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.hist(vertex_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Frequency')
    plt.title('Distribution of Vertex Counts')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"顶点数统计:")
    print(f"  最小值: {min(vertex_counts)}")
    print(f"  最大值: {max(vertex_counts)}")
    print(f"  平均值: {np.mean(vertex_counts):.2f}")
    print(f"  中位数: {np.median(vertex_counts):.2f}")

def main():
    # 数据文件路径
    data_file = '/home/wzn/双足/Biped-Locomotion/convex_points_data.pkl'
    
    # 加载多边形数据
    try:
        polygon_data = load_polygon_data(data_file)
        print(f"成功加载 {len(polygon_data)} 帧多边形数据")
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return
    
    # 显示数据统计信息
    plot_polygon_data_statistics(polygon_data)
    
    # 可视化前几帧（避免生成太多图像）
    num_frames_to_show = min(10, len(polygon_data))
    print(f"\n可视化前 {num_frames_to_show} 帧:")
    
    for i in range(num_frames_to_show):
        polygon_points = polygon_data[i]
        print(f"  帧 {i+1}: {len(polygon_points)} 个顶点")
        
        # 可视化前3帧
        if i < 3:
            fig, ax = plt.subplots(figsize=(8, 6))
            visualize_single_polygon(polygon_points, ax=ax)
            ax.set_title(f'Frame {i+1} - {len(polygon_points)} vertices')
            plt.show()

if __name__ == "__main__":
    main()