import numpy as np
import time
import os
import sys
import mujoco
import mujoco.viewer as viewer
from pynput import keyboard
from collections import deque  # 新增：导入deque用于FIFO队列
from com import read_robot_state
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from motion_planning.walking import PreviewControl
import math
from ik_walk import *

CoM_x_list = []
CoM_y_list = []
ZMP_x_list = []
ZMP_y_list = []

body_id_dict = {
    "freejoint": 0,
    "base_link": 1,
    "left_leg_pitch": 2,
    "left_leg_rol": 3,
    "left_leg_yaw": 4,
    "left_knee": 5,
    "left_ankle_pitch": 6,
    "left_ankle_raw": 7,
    "right_leg_pitch": 8,
    "right_leg_rol": 9,
    "right_leg_yaw": 10,
    "right_knee": 11,
    "right_ankle_pitch": 12,
    "right_ankle_raw": 13
}

def read_joint_state(data, joint_name):
    joint_pos = data.xpos[body_id_dict[joint_name]]
    # WXYZ
    joint_orent = data.xquat[body_id_dict[joint_name]]
    return joint_pos, joint_orent

# 全局变量用于存储键盘状态
key_pressed = None

def on_press(key):
    global key_pressed
    try:
        if key.char == 'd':
            key_pressed = 'd'
        elif key.char == 'w':
            key_pressed = 'w'
    except AttributeError:
        # 特殊键（如ctrl, alt等）会触发这个异常
        pass


def execute_d_command(model, data,q_data):

    data.qpos[7:] = q_data

def walk():
    global key_pressed
    
    # 获取项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scene_path = os.path.join(project_root, "R2/urdf/scene.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)

    data = mujoco.MjData(model)
    
    # 加载初始姿态 "home" 或 "stand"
    # 使用 "home" 姿态（零位直立）更稳定
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "dunxia2")
    target_qpos = None
    if key_id >= 0:
        # 将关键帧数据复制到当前状态
        data.qpos[:] = model.key_qpos[key_id]
        data.qvel[:] = 0  # 确保初始速度为零
        target_qpos = model.key_qpos[key_id][7:].copy()  # 保存目标关节角度
        data.ctrl[:] = target_qpos  # 设置控制目标
        
        # 让机器人静态平衡稳定一下
        for _ in range(100):
            data.ctrl[:] = target_qpos
            mujoco.mj_step(model, data)
        
    else:
        # 如果没有找到关键帧，使用零位置
        target_qpos = np.zeros(model.nu)
        print("警告: 未找到关键帧，使用零位置")

    q_data_path = os.path.join(project_root, "ik-walk_0.70.npy")
    q_data = np.load(q_data_path)
    print(f"加载轨迹数据: q_data.shape = {q_data.shape}")
    max_trajectory_index = len(q_data) - 1  # 最大有效索引

    # ========== 外力配置 ==========
    # 设置要施加外力的body（例如：base_link）
    force_body_name = "base_link"  # 可以修改为其他关节名称
    force_body_id = body_id_dict.get(force_body_name, 1)  # 默认为base_link (id=1)
    
    # 设置垂直向上的外力大小（牛顿）
    upward_force_magnitude = 50.0  # 可以调整力的大小
    
    print(f"\n外力配置:")
    print(f"  作用body: {force_body_name} (ID: {force_body_id})")
    print(f"  力的大小: {upward_force_magnitude} N")
    print(f"  力的方向: 世界坐标系Z轴正方向 (垂直向上)")
    print(f"  力的向量: [0, 0, {upward_force_magnitude}]\n")

    supPoint = np.array([0., 0.065])
    count = 0
    
    # 启动键盘监听器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    angle_l = np.radians(0)  # -10度转弧度
    angle_r = np.radians(0)  # +10度转弧度
    R_l = np.array([
        [1, 0, 0],
        [0, np.cos(angle_l), -np.sin(angle_l)],
        [0, np.sin(angle_l), np.cos(angle_l)]
    ])
    R_r = np.array([
        [1, 0, 0],
        [0, np.cos(angle_r), -np.sin(angle_r)],
        [0, np.sin(angle_r), np.cos(angle_r)]
    ])
    
    
    with viewer.launch_passive(model, data) as mjviewer:
        
        mjviewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # 显示接触点
        mjviewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True  # 显示接触力
        
        # 调整可视化比例
        model.vis.scale.contactwidth = 0.05
        model.vis.scale.contactheight = 0.02
        model.vis.scale.forcewidth = 0.03
        model.vis.map.force = 0.005  # 力的缩放因子，根据实际力的大小调整
        
        count = 0
        last_warning_time = 0  # 记录上次警告时间，避免频繁输出
        last_trajectory_index = -1  # 记录上次使用的轨迹索引
        trajectory_index = 0  # 初始化轨迹索引
        
        # 使用 deque 实现先进先出队列
        combined_vectors = deque()
        print("\n轨迹队列已初始化（FIFO模式）")
        
        stepHeight = 0.02
        supPoint = np.array([0., 0.065])
        pre = PreviewControl(dt=1. / 240., Tsup_time=0.5, Tdl_time=0.1, previewStepNum=190)
        
        while mjviewer.is_running():
            
            global key_pressed

            
            if key_pressed == 'w':
                print("\n执行 'w' 键操作: 生成新轨迹并添加到队列")
                
                CoM_trj, footTrjL, footTrjR = pre.footPrintAndCoM_trajectoryGenerator(
                    inputTargetZMP=supPoint,
                    inputFootPrint=supPoint,
                    stepHeight=stepHeight
                )
                
                current_qpos = data.qpos[7:].copy()

                for i in range(len(CoM_trj)):

                    ik_qpos_left = inverse_kinematics(6,current_qpos,R_l,
                                                    [
                                                        footTrjL[i][0]-CoM_trj[i][0],
                                                        CoM_trj[i][1]-footTrjL[i][1],
                                                        -(0.70-footTrjL[i][2])
                                                    ]
                                                    )
                    ik_qpos_right = inverse_kinematics(12,current_qpos,R_r,
                                                    [
                                                        footTrjR[i][0]-CoM_trj[i][0],
                                                        CoM_trj[i][1]-footTrjR[i][1],
                                                        -(0.70-footTrjR[i][2])
                                                        ]
                                                    )
                    
                    new_vector = np.concatenate([ik_qpos_left[:6], ik_qpos_right[6:]])
                    current_qpos = new_vector
                    # 添加到队列尾部
                    combined_vectors.append(new_vector)
                
                supPoint[0] += 0.075
                # y坐标实现左右腿变换
                supPoint[1] = -supPoint[1]
                # 重置按键状态
                key_pressed = None
            
            if count % 4 == 0:
                # 从队列头部取出并执行动作（FIFO）
                if len(combined_vectors) > 0:
                    # 从队列左侧（头部）取出一个动作
                    data.ctrl[:] = combined_vectors.popleft()
                    print(f"[步数: {count}] 执行队列动作 | 剩余队列长度: {len(combined_vectors)}")
                else:
                    # 队列为空，保持当前姿态
                    if count == 0:
                        print("[提示] 队列为空，等待按 'w' 键生成轨迹...")
            
            count += 1
            
            # data.xfrc_applied[:] = 0
            # data.xfrc_applied[force_body_id] = [0, 0, upward_force_magnitude, 0, 0, 0]
            
            # 更新模型
            mujoco.mj_step(model, data)

            mjviewer.sync()



if __name__ == "__main__":
    walk()