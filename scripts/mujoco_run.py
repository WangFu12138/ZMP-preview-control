import numpy as np
import time
import mujoco
from draw_contact_polygon import *

def walk():
    """
    演示如何使用PD控制器控制Unitree H1机器人
    """
    # 加载Unitree H1模型
    model = mujoco.MjModel.from_xml_path(
        "/home/wzn/双足/Biped-Locomotion/urdf/empty_scene.xml"
    )

    data = mujoco.MjData(model)

    com_pos_data = np.load('/home/wzn/双足/Biped-Locomotion/com_pos_data.npy')
    com_rpy_data = np.load('/home/wzn/双足/Biped-Locomotion/com_rpy_data.npy')
    joint_pos_data = np.load('/home/wzn/双足/Biped-Locomotion/joint_pos_data.npy')

    # 创建用于存储convex_points的列表
    all_convex_points = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置可视化选项
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # 显示接触点
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True  # 显示接触力
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True   # 启用透明渲染
        
        # 调整可视化比例
        model.vis.scale.contactwidth = 0.05
        model.vis.scale.contactheight = 0.02
        model.vis.scale.forcewidth = 0.03
        model.vis.map.force = 0.005  # 力的缩放因子，根据实际力的大小调整

        min_length = min(len(com_pos_data), len(com_rpy_data), len(joint_pos_data))

        while viewer.is_running():

            for i in range(min_length):
                # 获取当前时间步的数据
                com_pos = com_pos_data[i]
                com_rpy = com_rpy_data[i]
                joint_pos = joint_pos_data[i]
                
                # 将RPY转换为四元数
                # 注意：mujoco使用[w, x, y, z]格式的四元数
                quat = np.zeros(4)
                mujoco.mju_euler2Quat(quat, com_rpy, 'xyz')
                
                # 构造qpos向量
                # [x, y, z, qx, qy, qz, qw, joint1, joint2, ..., joint12]
                data.qpos[:3] = com_pos  # 位置
                data.qpos[2] -= 0.01
                data.qpos[3:7] = [quat[0], quat[1], quat[2], quat[3]]  # 四元数 (w, x, y, z)
                
                # 设置关节角度（12个关节）
                # 注意：需要确保joint_pos_data中的关节顺序与模型中的关节顺序一致
                if len(joint_pos) >= 12:
                    data.qpos[7:19] = joint_pos[:12]  # 12个腿部关节
                
                # 设置速度为0（静态展示）
                data.qvel[:] = 0
                
                # 更新模型
                mujoco.mj_step(model, data)

                # 2. 获取支撑点（双脚与地面的接触点）
                support_points = get_support_points(
                    model, data, 
                    foot_names=["right_foot_tip_link", "left_foot_tip_link"],  # 替换为你的脚部件名称
                    ground_name="world"  # 替换为你的地面部件名称
                )
                
                # 3. 绘制支撑多边形（仅在有足够支撑点时）
                if len(support_points) >= 3:
                    # 对支撑点进行凸包处理（确保多边形是凸的）
                    try:
                        hull = ConvexHull(support_points[:, :2])  # 取x-y平面
                        convex_points = support_points[hull.vertices]
                        # 保存convex_points
                        all_convex_points.append(convex_points.copy())
                        draw_polygon(viewer, convex_points, color=(0, 1, 0, 0.8))  # 绿色多边形
                    except:
                        # 若无法计算凸包，直接使用原始点
                        draw_polygon(viewer, support_points, color=(0, 1, 0, 0.8))
                        # 保存support_points
                        all_convex_points.append(support_points.copy())
                elif len(support_points) >= 2:
                    # 如果只有两个点，则绘制线段
                    draw_polygon(viewer, support_points, color=(0, 1, 0, 0.8))
                    # 保存support_points
                    all_convex_points.append(support_points.copy())
                elif len(support_points) == 1:
                    # 如果只有一个点，则绘制点
                    draw_polygon(viewer, support_points, color=(0, 1, 0, 0.8))
                    # 保存support_points
                    all_convex_points.append(support_points.copy())

                viewer.sync()

                time.sleep(0.01)  # 控制播放速度
            
            # 循环结束后保存所有convex_points到文件
            # 保存为列表格式
            import pickle
            with open('/home/wzn/双足/Biped-Locomotion/convex_points_data.pkl', 'wb') as f:
                pickle.dump(all_convex_points, f)
            
            print(f"已保存 {len(all_convex_points)} 帧的convex_points数据到文件")
            print(f"不规则数据形状: {[frame.shape for frame in all_convex_points]}")

            break

if __name__ == "__main__":
    walk()