import numpy as np
import time
import mujoco
import mujoco.viewer as viewer
from com import read_robot_state

import math
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
    except AttributeError:
        # 特殊键（如ctrl, alt等）会触发这个异常
        pass



def execute_d_command(model, data,q_data):

    data.qpos[7:] = q_data

def walk():
    global key_pressed
    
    model = mujoco.MjModel.from_xml_path(
        "/home/wzn/双足/Biped-Locomotion/23-TR-R2人形机器人0815/urdf/scene.xml")

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
    # pin_model = pin.buildModelFromMJCF("/home/wzn/双足/Biped-Locomotion/23-TR-R2人形机器人0815/urdf/R2_mujoco.xml")
    # pin_data  = pin_model.createData()
    # for i in range(model.ngeom):
    #     for j in range(i+1, model.ngeom):
    #         model.geom_contype[i] = 0
    #         model.geom_conaffinity[j] = 0


    q_data = np.load('/home/wzn/双足/Biped-Locomotion/ik-walk_0.70.npy')
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
        
        while mjviewer.is_running():
            # 每1000步更新一次控制指令
            trajectory_index = count // 5
            if trajectory_index != last_trajectory_index and trajectory_index <= max_trajectory_index:
                data.ctrl[:] = q_data[trajectory_index]
                last_trajectory_index = trajectory_index
                print(f"[步数: {count}] 加载轨迹索引 {trajectory_index}/{max_trajectory_index}")
            elif trajectory_index > max_trajectory_index:
                # 超出轨迹数据范围，保持最后一个姿态
                if last_trajectory_index != max_trajectory_index:
                    data.ctrl[:] = q_data[max_trajectory_index]
                    last_trajectory_index = max_trajectory_index
                    print(f"[步数: {count}] 已到达轨迹末端，保持最后姿态")
            
            # 检测姿态异常（但不自动重置）
            current_time = time.time()
            if data.qpos[2] < 0.2 and (current_time - last_warning_time) > 2.0:
                print(f"\n[警告] 机器人高度过低: {data.qpos[2]:.3f}m")
                print(f"  关节角度: {data.qpos[7:].round(3)}")
                print(f"  控制信号: {data.ctrl[:].round(3)}")
                last_warning_time = current_time
            
            #data.qpos[2:7] = [0.50,1,0,0,0]
            
            # ========== 施加外力：世界坐标系Z轴正方向 ==========
            # data.xfrc_applied 格式: [nbody x 6]
            # 每个body有6个分量: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            # 力和力矩都是在世界坐标系中定义的
            
            # 清零所有外力（每帧重置）
            data.xfrc_applied[:] = 0
            
            # 对指定body施加垂直向上的力
            # 格式: [Fx, Fy, Fz, Tx, Ty, Tz]
            # Z轴正方向 = 垂直向上
            data.xfrc_applied[force_body_id] = [0, 0, upward_force_magnitude, 0, 0, 0]
            
            # 更新模型
            mujoco.mj_step(model, data)

            mjviewer.sync()

            base_pos, base_orent = read_joint_state(data,"base_link")

            # 获取渲染上下文
            con = mjviewer.user_scn
            
            # 仿真步计数器递增
            count += 1


if __name__ == "__main__":
    walk()