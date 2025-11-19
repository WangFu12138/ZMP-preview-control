import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.biped import Biped
from motion_planning.walking import PreviewControl

import matplotlib.pyplot as plt

CoM_x_list = []
CoM_y_list = []
ZMP_x_list = []
ZMP_y_list = []

# 新增列表用于记录额外的数据
com_pos_list = []  # 质心位置 xyz
com_rpy_list = []  # 质心朝向 rpy
joint_pos_list = []  # 关节位置 qpos
support_polygon_list = []  # 支撑多边形顶点

class BipedMotionController:
    def __init__(self):
        # 创建双足机器人实例，被控制对象
        self.biped = Biped()
        self.CoM_height = 0.45
        self.targetRPY = [0.0, 0.0, 0.0]
        self.targetPosL = [0.0, 0.114, -self.CoM_height]
        self.targetPosR = [0.0, -0.114, -self.CoM_height]

    def _update_incline(self):
        # 从环境中获取当前坡度信息
        incline = self.biped.env.get_incline()
        self.biped.env.reset_incline()
        # 根据坡度来调整目标的姿态角度
        self.targetRPY[1] = incline
        print(self.targetRPY)
    
    def stand(self):
        self.biped.initialize_position(init_time=0.2)
        while True:
            self._update_incline()
            self.biped.set_leg_positions(self.targetPosL, self.targetPosR, self.targetRPY)
            # 记录数据
            self._log_data()
            self.biped.env.step()
    
    def squat(self):
        self.biped.initialize_position(init_time=0.1)
        dp = 0.002 # Step size

        while True:
            self._update_incline()
            for _ in range(100):
                self.biped.set_leg_positions(self.targetPosL, self.targetPosR, self.targetRPY)
                # 记录数据
                self._log_data()
                self.biped.env.step()
                self.targetPosL[2] += dp
                self.targetPosR[2] += dp

            for _ in range(100):
                self.biped.set_leg_positions(self.targetPosL, self.targetPosR, self.targetRPY)
                # 记录数据
                self._log_data()
                self.biped.env.step()
                self.targetPosL[2] -= dp
                self.targetPosR[2] -= dp
        
    def walk(self):
        """
        Executes a dynamic walking cycle for the biped robot using preview control.
        """
        # 初始化机器人位置到标准站立姿态
        self.biped.initialize_position(init_time=0.2)
        # ZMP-based preview controller responsible for generating smooth CoM and foot trajectories
        pre = PreviewControl(dt=1. / 240., Tsup_time=0.3, Tdl_time=0.1, previewStepNum=190)

        # Logs all CoM positions over time
        # 记录质心轨迹
        CoM_trajectory = np.empty((0, 3), float)
        # 记录右脚轨迹
        trjR_log = np.empty((0, 3), float)
        # 记录左脚轨迹
        trjL_log = np.empty((0, 3), float)

        # Starting support point (initial ZMP target) - assumed under the left leg (y = +0.065)
        # 设置初始支撑点
        supPoint = np.array([0., 0.065])

        count = 0
        while count < 25:
            # 1. Inclination Handling
            self._update_incline()

            # 2. Preview Control Trajectory Generation
            # Both foot placement and ZMP are set to supPoint
            # 获取步高
            stepHeight = self.biped.get_step_height()
            # 生成质心和足部轨迹，足部放置点和ZMP目标点都设为当前支撑点
            CoM_trj, footTrjL, footTrjR = pre.footPrintAndCoM_trajectoryGenerator(
                inputTargetZMP=supPoint,
                inputFootPrint=supPoint,
                stepHeight=stepHeight
            )

            # Log all trajectories
            CoM_trajectory = np.vstack((CoM_trajectory, CoM_trj))
            trjR_log = np.vstack((trjR_log, footTrjR))
            trjL_log = np.vstack((trjL_log, footTrjL))

            # 3. Apply Control at Each Time Step
            for j in range(len(CoM_trj)):
                # 存储质心xy坐标
                CoM_x_list.append(CoM_trj[j][0])
                CoM_y_list.append(CoM_trj[j][1])

                # 计算相对与质心的腿部目标位置（右腿和左腿）
                self.targetPosR = footTrjR[j] - CoM_trj[j]
                self.targetPosL = footTrjL[j] - CoM_trj[j]
                # 设置腿部位置并执行仿真
                self.biped.set_leg_positions(self.targetPosL, self.targetPosR, self.targetRPY)
                # 记录数据
                self._log_data()
                # To simulate the next timestep
                self.biped.env.step()

            # 4. Update Support Leg & Stride
            # Moves the ZMP target forward by the stride length (x-direction)
            # Switches leg (y value flips sign: +0.065 -> -0.065 and vice versa)
            # 更新支撑点 stride 一个步长的距离
            supPoint[0] += self.biped.get_stride()
            # y坐标实现左右腿变换
            supPoint[1] = -supPoint[1]

            count += 1

        # Get desired ZMP from preview controller logs
        ZMP_x_list.extend(pre.px_ref_log)
        ZMP_y_list.extend(pre.py_ref_log)

    def _log_data(self):
        """记录机器人的质心位置、朝向和关节位置"""
        # 记录质心位置 (xyz)
        com_pos = self.biped.get_position()
        com_pos_list.append(com_pos)
        
        # 记录质心朝向 (rpy)
        com_rpy = self.biped.get_euler()
        com_rpy_list.append(com_rpy)
        
        # 记录所有关节位置 (qpos)
        # 获取左右腿关节位置并合并
        left_joints = self.biped.get_joint_positions(self.biped.left_foot_offset)
        right_joints = self.biped.get_joint_positions(self.biped.right_foot_offset)
        all_joints = left_joints + right_joints
        joint_pos_list.append(all_joints)
        
        # 记录支撑多边形顶点（双脚的x,y坐标）
        left_foot_pos = self.biped.forward_kinematics(left_joints, self.biped.left_foot_offset)[1]
        right_foot_pos = self.biped.forward_kinematics(right_joints, self.biped.right_foot_offset)[1]
        support_polygon = np.array([[left_foot_pos[0], left_foot_pos[1]], 
                                   [right_foot_pos[0], right_foot_pos[1]]])
        support_polygon_list.append(support_polygon)

def main():

    parser = argparse.ArgumentParser(
        description="Execute a motion behavior for the robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--motion',
        help='Type of motion behavior to execute.',
        type=str,
        choices=['stand', 'squat', 'walk'],
        default='walk'
    )
    args = parser.parse_args()

    controller = BipedMotionController()

    motion_dispatch = {
        'stand': controller.stand,
        'squat': controller.squat,
        'walk': controller.walk,
    }
    # get()方法从字典中获取与args.motion对应的函数，如果没有找到则默认调用controller.stand
    motion_fn = motion_dispatch.get(args.motion, controller.stand)
    # 实际执行对应的动作
    motion_fn()
    
    left_leg_joint_values = np.array(controller.biped.left_leg_joints_value_logs)
    right_leg_joint_values = np.array(controller.biped.right_leg_joints_value_logs)
    
    # 将新记录的数据转换为numpy数组以便绘图
    com_pos_array = np.array(com_pos_list)
    com_rpy_array = np.array(com_rpy_list)
    joint_pos_array = np.array(joint_pos_list)
    support_polygon_array = np.array(support_polygon_list)

    # 保存数据到numpy文件
    np.save('com_pos_data.npy', com_pos_array)
    np.save('com_rpy_data.npy', com_rpy_array)
    np.save('joint_pos_data.npy', joint_pos_array)
    print("数据已保存到以下文件:")
    print("- com_pos_data.npy: 质心位置数据")
    print("- com_rpy_data.npy: 质心朝向数据")
    print("- joint_pos_data.npy: 关节位置数据")

    plt.figure(figsize=(12, 8))  # Adjust size as needed

    # Plot 1
    plt.subplot(2, 2, 1)
    plt.plot(CoM_x_list, label='CoM_x')
    plt.plot(ZMP_x_list, label='ZMP_x')
    plt.xlabel('Time Step')
    plt.ylabel('X Position')
    plt.title('CoM_x and ZMP_x over Time')
    plt.legend()

    # Plot 2
    plt.subplot(2, 2, 2)
    plt.plot(CoM_y_list, label='CoM_y')
    plt.plot(ZMP_y_list, label='ZMP_y')
    plt.xlabel('Time Step')
    plt.ylabel('Y Position')
    plt.title('CoM_y and ZMP_y over Time')
    plt.legend()

    # Plot 3
    plt.subplot(2, 2, 3)
    for i in range(6):
        plt.plot(left_leg_joint_values[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Value')
    plt.title('Left Leg Joint Positions over Time')
    plt.legend()

    # Plot 4
    plt.subplot(2, 2, 4)
    for i in range(6):
        plt.plot(right_leg_joint_values[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Value')
    plt.title('Right Leg Joint Positions over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # 绘制新增的图表
    plt.figure(figsize=(15, 10))
    
    # 绘制质心位置XYZ
    plt.subplot(3, 2, 1)
    plt.plot(com_pos_array[:, 0], label='CoM X')
    plt.plot(com_pos_array[:, 1], label='CoM Y')
    plt.plot(com_pos_array[:, 2], label='CoM Z')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('CoM Position XYZ over Time')
    plt.legend()
    
    # 绘制质心朝向RPY
    plt.subplot(3, 2, 2)
    plt.plot(com_rpy_array[:, 0], label='Roll')
    plt.plot(com_rpy_array[:, 1], label='Pitch')
    plt.plot(com_rpy_array[:, 2], label='Yaw')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (rad)')
    plt.title('CoM Orientation RPY over Time')
    plt.legend()
    
    # 绘制关节位置
    plt.subplot(3, 2, 3)
    for i in range(6):
        plt.plot(joint_pos_array[:, i], label=f'Left Joint {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Position (rad)')
    plt.title('Left Leg Joint Positions over Time')
    plt.legend()
    
    # 绘制右腿关节位置
    plt.subplot(3, 2, 4)
    for i in range(6):
        plt.plot(joint_pos_array[:, i+6], label=f'Right Joint {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Position (rad)')
    plt.title('Right Leg Joint Positions over Time')
    plt.legend()
    
    # 绘制支撑多边形
    plt.subplot(3, 2, 5)
    # 绘制所有支撑点
    left_foot_x = support_polygon_array[:, 0, 0]
    left_foot_y = support_polygon_array[:, 0, 1]
    right_foot_x = support_polygon_array[:, 1, 0]
    right_foot_y = support_polygon_array[:, 1, 1]
    
    plt.plot(left_foot_x, left_foot_y, 'r-', label='Left foot', alpha=0.7)
    plt.plot(right_foot_x, right_foot_y, 'b-', label='Right foot', alpha=0.7)
    
    # 绘制当前时刻的支撑多边形（最后一个时刻）
    foot_points = np.array([[left_foot_x[-1], left_foot_y[-1]], 
                           [right_foot_x[-1], right_foot_y[-1]]])
    # 对脚部点按x坐标排序，以便正确绘制多边形
    sorted_indices = np.argsort(foot_points[:, 0])
    sorted_foot_points = foot_points[sorted_indices]
    
    # 添加第一个点以闭合多边形
    polygon_points = np.vstack([sorted_foot_points, sorted_foot_points[0]])
    plt.plot(polygon_points[:, 0], polygon_points[:, 1], 'g-', linewidth=2, label='Support Polygon')
    plt.scatter(sorted_foot_points[:, 0], sorted_foot_points[:, 1], color='black', s=30, zorder=5)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Support Polygon')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()