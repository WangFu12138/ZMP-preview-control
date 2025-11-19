import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os

class PolygonAnimationGenerator:
    """
    多边形动图生成器，用于绘制多边形并生成动态图像
    """
    
    def __init__(self, save_dir="./output"):
        """
        初始化多边形动图生成器
        
        参数:
            save_dir: 保存输出文件的目录
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
    def load_polygon_data(self, polygon_data):
        """
        加载多边形点集数据
        
        参数:
            polygon_data: 多边形点集数据，可以是以下格式之一：
                         - 3D numpy数组: (时间步, 点数, 2)，每个时间步包含一个多边形的点集
                         - 列表的列表: [[polygon1_points], [polygon2_points], ...]
        """
        # 验证数据格式
        if isinstance(polygon_data, np.ndarray):
            if polygon_data.ndim != 3 or polygon_data.shape[2] != 2:
                raise ValueError("numpy数组格式应为 (时间步, 点数, 2)")
            self.polygon_data = polygon_data
        elif isinstance(polygon_data, list):
            # 转换列表为numpy数组
            max_points = max(len(poly) for poly in polygon_data)
            num_frames = len(polygon_data)
            self.polygon_data = np.zeros((num_frames, max_points, 2))
            for i, poly in enumerate(polygon_data):
                num_points = len(poly)
                self.polygon_data[i, :num_points, :] = np.array(poly)
                # 对于点数不足的情况，填充最后一个点
                if num_points < max_points:
                    self.polygon_data[i, num_points:, :] = self.polygon_data[i, num_points-1, :]
        else:
            raise TypeError("polygon_data应为numpy数组或列表")
        
        return self
    
    def generate_sample_data(self, num_frames=100, num_points_range=(3, 6), motion_range=1.0):
        """
        生成示例多边形数据，用于测试
        
        参数:
            num_frames: 帧数
            num_points_range: 多边形点数范围
            motion_range: 运动范围
            
        返回:
            self: 返回实例本身，支持链式调用
        """
        # 为每帧随机生成不同点数的多边形
        polygons = []
        for i in range(num_frames):
            # 随机生成多边形点数
            num_points = np.random.randint(num_points_range[0], num_points_range[1] + 1)
            
            # 生成圆形排列的点，并添加随机偏移实现运动效果
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            radius = 0.5 + 0.3 * np.sin(i * 0.1)
            
            # 添加随时间变化的位置偏移
            x_offset = motion_range * 0.5 * np.sin(i * 0.05)
            y_offset = motion_range * 0.5 * np.cos(i * 0.05)
            
            # 生成点集
            x = radius * np.cos(angles) + x_offset
            y = radius * np.sin(angles) + y_offset
            
            # 组合成点集
            points = np.column_stack((x, y))
            polygons.append(points)
        
        # 加载生成的数据
        self.load_polygon_data(polygons)
        return self
    
    def draw_single_polygon(self, points, ax=None, color='blue', alpha=0.5, edge_color='black', linewidth=1):
        """
        绘制单个多边形
        
        参数:
            points: 多边形点集，形状为 (点数, 2)
            ax: matplotlib轴对象，如果为None则创建新的
            color: 多边形填充颜色
            alpha: 透明度
            edge_color: 边框颜色
            linewidth: 边框线宽
            
        返回:
            ax: matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # 过滤掉无效点（如果有的话）
        valid_points = points[~np.isnan(points).any(axis=1)]
        
        if len(valid_points) >= 3:
            # 创建多边形并添加到轴上
            polygon = Polygon(valid_points, closed=True, 
                             facecolor=color, alpha=alpha,
                             edgecolor=edge_color, linewidth=linewidth)
            ax.add_patch(polygon)
            
            # 绘制顶点
            ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                      color=edge_color, s=30, zorder=5)
            
            # 标记顶点顺序
            for i, (x, y) in enumerate(valid_points):
                ax.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points')
        
        # 设置坐标轴范围以确保多边形完整显示
        if len(valid_points) > 0:
            x_min, y_min = valid_points.min(axis=0) - 0.5
            x_max, y_max = valid_points.max(axis=0) + 0.5
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        ax.set_aspect('equal')
        ax.set_title('Polygon Visualization')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return ax
    
    def create_animation(self, fps=10, duration=None, 
                        color='blue', alpha=0.5, 
                        edge_color='black', linewidth=1,
                        output_format='gif', filename='polygon_animation'):
        """
        创建多边形动画
        
        参数:
            fps: 帧率（每秒帧数）
            duration: 动画持续时间（秒），如果为None则使用所有帧
            color: 多边形填充颜色
            alpha: 透明度
            edge_color: 边框颜色
            linewidth: 边框线宽
            output_format: 输出格式，'gif'或'video'
            filename: 输出文件名（不包含扩展名）
            
        返回:
            animation: matplotlib动画对象
        """
        if not hasattr(self, 'polygon_data'):
            raise ValueError("请先加载多边形数据")
        
        # 确定要使用的帧数
        total_frames = len(self.polygon_data)
        if duration is not None:
            num_frames = int(duration * fps)
            if num_frames > total_frames:
                print(f"警告：请求的帧数({num_frames})超过数据总帧数({total_frames})，将使用所有帧")
                num_frames = total_frames
        else:
            num_frames = total_frames
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 确定全局坐标轴范围
        all_points = self.polygon_data.reshape(-1, 2)
        valid_points = all_points[~np.isnan(all_points).any(axis=1)]
        
        x_min, y_min = valid_points.min(axis=0) - 0.5
        x_max, y_max = valid_points.max(axis=0) + 0.5
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title('Polygon Animation')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 初始化多边形对象
        polygon = Polygon([[0, 0]], closed=True, 
                         facecolor=color, alpha=alpha,
                         edgecolor=edge_color, linewidth=linewidth)
        ax.add_patch(polygon)
        
        # 初始化顶点散点和标签
        scatter = ax.scatter([], [], color=edge_color, s=30, zorder=5)
        annotations = []
        
        def init():
            """初始化动画"""
            polygon.set_xy([[0, 0]])
            scatter.set_offsets(np.empty((0, 2)))
            # 清除旧的注释
            for ann in annotations:
                ann.remove()
            annotations.clear()
            return polygon, scatter, 
        
        def update(frame):
            """更新每一帧"""
            # 获取当前帧的多边形点集
            points = self.polygon_data[frame]
            
            # 过滤掉无效点
            valid_points = points[~np.isnan(points).any(axis=1)]
            
            if len(valid_points) >= 3:
                # 更新多边形
                polygon.set_xy(valid_points)
                
                # 更新顶点散点
                scatter.set_offsets(valid_points)
                
                # 清除旧的注释
                for ann in annotations:
                    ann.remove()
                annotations.clear()
                
                # 添加新的顶点标签
                for i, (x, y) in enumerate(valid_points):
                    ann = ax.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points')
                    annotations.append(ann)
            
            # 将所有返回对象放入一个列表中
            return_objects = [polygon, scatter]
            return_objects.extend(annotations)
            return tuple(return_objects)
        
        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init,
                                     interval=1000/fps, blit=True)
        
        # 保存动画
        output_path = os.path.join(self.save_dir, f"{filename}.{output_format}")
        
        if output_format == 'gif':
            ani.save(output_path, writer='pillow', fps=fps, dpi=100)
        elif output_format == 'video':
            # 需要安装ffmpeg或其他视频编码器
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
        else:
            raise ValueError("输出格式应为'gif'或'video'")
        
        print(f"动画已保存到: {output_path}")
        plt.close(fig)
        
        return ani
    
    def create_smooth_transition_animation(self, fps=10, duration=None, 
                                          color='blue', alpha=0.5, 
                                          edge_color='black', linewidth=1,
                                          transition_frames=5,
                                          output_format='gif', filename='smooth_polygon_animation'):
        """
        创建带有平滑过渡效果的多边形动画
        
        参数:
            fps: 帧率（每秒帧数）
            duration: 动画持续时间（秒），如果为None则使用所有帧
            color: 多边形填充颜色
            alpha: 透明度
            edge_color: 边框颜色
            linewidth: 边框线宽
            transition_frames: 两帧之间的过渡帧数
            output_format: 输出格式，'gif'或'video'
            filename: 输出文件名（不包含扩展名）
            
        返回:
            animation: matplotlib动画对象
        """
        if not hasattr(self, 'polygon_data'):
            raise ValueError("请先加载多边形数据")
        
        # 确定要使用的原始帧数
        total_frames = len(self.polygon_data)
        if duration is not None:
            orig_frames = int(duration * fps / (1 + transition_frames))
            if orig_frames > total_frames:
                print(f"警告：请求的帧数超过数据总帧数，将使用所有帧")
                orig_frames = total_frames
        else:
            orig_frames = total_frames
        
        # 生成平滑过渡的帧
        smoothed_frames = []
        
        for i in range(orig_frames - 1):
            # 添加原始帧
            smoothed_frames.append(self.polygon_data[i])
            
            # 添加过渡帧
            for t in range(1, transition_frames + 1):
                # 线性插值计算过渡帧
                alpha = t / (transition_frames + 1)
                interp_frame = (1 - alpha) * self.polygon_data[i] + alpha * self.polygon_data[i + 1]
                smoothed_frames.append(interp_frame)
        
        # 添加最后一帧
        smoothed_frames.append(self.polygon_data[orig_frames - 1])
        
        # 转换为numpy数组
        smoothed_data = np.array(smoothed_frames)
        
        # 保存原始数据
        original_data = self.polygon_data
        
        try:
            # 使用平滑后的数据创建动画
            self.polygon_data = smoothed_data
            ani = self.create_animation(
                fps=fps,
                color=color,
                alpha=alpha,
                edge_color=edge_color,
                linewidth=linewidth,
                output_format=output_format,
                filename=filename
            )
        finally:
            # 恢复原始数据
            self.polygon_data = original_data
        
        return ani

def main():
    # 创建多边形动图生成器实例
    generator = PolygonAnimationGenerator(save_dir="./polygon_animations")
    
    # 生成示例数据
    print("生成示例多边形数据...")
    generator.generate_sample_data(num_frames=50, num_points_range=(4, 6))
    
    # 绘制单个多边形（第一帧）
    print("绘制示例多边形...")
    fig, ax = plt.subplots(figsize=(10, 8))
    generator.draw_single_polygon(generator.polygon_data[0], ax=ax, color='red')
    plt.savefig("polygon_animations/sample_polygon.png")
    plt.close(fig)
    print("示例多边形已保存到: polygon_animations/sample_polygon.png")
    
    # 创建基本动画
    print("创建基本动画...")
    generator.create_animation(
        fps=10,
        color='blue',
        alpha=0.6,
        edge_color='navy',
        output_format='gif',
        filename='basic_animation'
    )
    
    # 创建平滑过渡动画
    print("创建平滑过渡动画...")
    generator.create_smooth_transition_animation(
        fps=20,
        color='green',
        alpha=0.6,
        edge_color='darkgreen',
        transition_frames=3,
        output_format='gif',
        filename='smooth_transition_animation'
    )
    
    print("所有任务完成！")

if __name__ == "__main__":
    main()