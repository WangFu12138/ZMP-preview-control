import numpy as np
import time
import mujoco
import mujoco.viewer
import pinocchio as pin


class PDController:
    """
    PD控制器类，用于控制机器人关节角度
    """

    def __init__(self, kp, kd):
        """
        初始化PD控制器
        
        Args:
            kp: 比例增益
            kd: 微分增益
        """
        self.kp = kp
        self.kd = kd

    def calculate(self, target_position, current_position, target_velocity=None, current_velocity=None):
        """
        计算PD控制输出
        
        Args:
            target_position: 目标位置
            current_position: 当前位置
            target_velocity: 目标速度（默认为0）
            current_velocity: 当前速度（默认为0）
            
        Returns:
            控制输出
        """
        if target_velocity is None:
            target_velocity = 0
            
        if current_velocity is None:
            current_velocity = 0
            
        # 计算误差
        position_error = target_position - current_position
        velocity_error = target_velocity - current_velocity
        
        # PD控制律: output = Kp * error + Kd * d(error)/dt
        control_output = self.kp * position_error + self.kd * velocity_error
        
        return control_output

    def set_gains(self, kp=None, kd=None):
        """
        设置控制器增益
        
        Args:
            kp: 比例增益
            kd: 微分增益
        """
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd
def read_robot_state(model, data):


    mujoco.mj_comPos(model, data)  
    com = data.subtree_com[0]
    
    # 获取左右脚的接触力
    left_foot_force = np.zeros(6)
    right_foot_force = np.zeros(6)
    
    # 遍历所有接触点
    for i in range(data.ncon):
        contact = data.contact[i]
        # 获取接触点的力和力矩
        mujoco.mj_contactForce(model, data, i, left_foot_force)
        
        # 判断是左脚还是右脚
        body1_name = mujoco.mj_id2name(model, 4, contact.geom1)
        body2_name = mujoco.mj_id2name(model, 4, contact.geom2)
        
        if body1_name is None or body2_name is None:
            continue  # 跳过无法识别的接触
            
        if 'left_ankle_link' in body1_name or 'left_ankle_link' in body2_name:
            left_foot_force += left_foot_force
        elif 'right_ankle_link' in body1_name or 'right_ankle_link' in body2_name:
            right_foot_force += right_foot_force
    
    # 计算ZMP (零力矩点)
    # ZMP_x = (M_y - F_z * COM_x) / F_y
    # ZMP_y = (M_x - F_z * COM_y) / F_x
    total_vertical_force = left_foot_force[2] + right_foot_force[2]
    total_moment_x = left_foot_force[3] + right_foot_force[3]
    total_moment_y = left_foot_force[4] + right_foot_force[4]
    total_force_x = left_foot_force[0] + right_foot_force[0]
    total_force_y = left_foot_force[1] + right_foot_force[1]
    
    zmp = np.zeros(2)
    if abs(total_force_x) > 1e-6:
        zmp[0] = (total_moment_y - total_vertical_force * com[0]) / total_force_x
    else:
        zmp[0] = com[0]
        
    if abs(total_force_y) > 1e-6:
        zmp[1] = (total_moment_x - total_vertical_force * com[1]) / total_force_y
    else:
        zmp[1] = com[1]
    
    return {
        'com': com,
        'zmp': zmp,
        'left_foot_force': left_foot_force[:3],  # 只返回力分量
        'right_foot_force': right_foot_force[:3],  # 只返回力分量
    }

class FootSteps:
    def __init__(self, left0, right0):
        self.phases = []    # list of (duration, support, target)
        self.left0 = np.array(left0)
        self.right0 = np.array(right0)
        self.times = [0.]

    # 添加一个新的行走阶段到序列中
    def add_phase(self, duration, support, target=None):
        self.phases.append((duration, support, np.array(target) if target is not None else None))
        self.times.append(self.times[-1] + duration)

    # 获取阶段索引
    def get_phase_index(self, t):
        # find i s.t. times[i] <= t < times[i+1]
        for i in range(len(self.times)-1):
            if self.times[i] <= t < self.times[i+1]: return i
        return len(self.times)-2

    # 获取阶段信息
    def get_phase(self, t):
        i = self.get_phase_index(t)
        return self.phases[i], t - self.times[i]

class ZmpRef:
    def __init__(self, footsteps: FootSteps):
        self.footsteps = footsteps

    # 按照时间t返回参考ZMP位置
    def __call__(self, t):
        # Determine phase and compute ZMP
        (dur, sup, target), tau = self.footsteps.get_phase(t)
        if sup == 'none':
            # double support: interpolate between last foot and next
            # TODO: implement interpolation
            return np.zeros(2)
        elif sup == 'left':
            # ZMP under left foot
            return np.array(self.footsteps.left0)
        else:
            return np.array(self.footsteps.right0)

class ComRef:
    def __init__(self, zmp_ref, h=0.3, g=9.81):
        self.zmp_ref = zmp_ref
        self.h = h
        self.omega = np.sqrt(g/h)
        # build LQR gain
        A = np.array([[0, 1], [self.omega**2, 0]])
        B = np.array([[0], [-self.omega**2]])
        Q = np.diag([1e3, 1.])
        R = np.array([[1e-6]])
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P
        self.state = np.zeros((2,))

    def update(self, dt, t):
        # simple Euler integration of x' = A x + B r_zmp + K*(x - x_ref)
        x = self.state
        ref_z = self.zmp_ref(t)
        A = np.array([[0, 1], [self.omega**2, 0]])
        B = np.array([[0], [-self.omega**2]])
        # state reference if CoM should track ZMP exactly
        x_ref = np.array([ref_z[0], 0.])  # forward direction only
        u_fb = -self.K @ (x - x_ref)
        xdot = A @ x + B.flatten()*ref_z[0] + B.flatten()*u_fb
        self.state = x + xdot*dt
        return self.state.copy()
def com_pin(qpos, model_pin, data_pin):
    """添加打印验证的质心计算函数"""

    # 2. 执行正运动学
    pin.forwardKinematics(model_pin, data_pin, qpos)
    pin.updateFramePlacements(model_pin, data_pin)
    
    # 3. 计算总质心
    com = pin.centerOfMass(model_pin, data_pin, False)

    
    return com

def demo_h1_pd_control():
    """
    演示如何使用PD控制器控制Unitree H1机器人
    """
    # 加载Unitree H1模型
    model = mujoco.MjModel.from_xml_path(
        "/home/wzn/双足/mujoco_menagerie/unitree_h1/scene.xml"
    )
    model_pin = pin.buildModelFromMJCF("/home/wzn/双足/mujoco_menagerie/unitree_h1/h1.xml")
    data_pin  = model_pin.createData()
    data = mujoco.MjData(model)
        
    # 获取关键帧中的初始位置
    home_key = model.key("home")
    # 如果有home关键帧，则初始化到该位置
    if home_key and home_key.qpos is not None:
        data.qpos[:] = home_key.qpos[:]

    # 创建PD控制器实例
    kp = 1000   # 位置增益
    kd = 0.1 # 速度增益
    pd_controller = PDController(kp, kd)
    
    # 创建pin实例
    pin_model = pin.buildModelFromMJCF("/home/wzn/双足/mujoco_menagerie/unitree_h1/h1.xml")
    pin_data  = pin_model.createData()

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
        
        # 初始化用于可视化COM和ZMP的几何体
        # 创建一个球体表示COM
        mjr_com = mujoco.MjvGeom()
        mjr_com.type = mujoco.mjtGeom.mjGEOM_SPHERE
        mjr_com.size = np.array([0.02, 0.02, 0.02])  # 半径
        mjr_com.pos = np.zeros(3)
        mjr_com.mat = np.eye(3)
        mjr_com.rgba = np.array([0, 0, 1, 1])  # 蓝色

        # 创建一个球体表示COM_pin
        mjr_com_pin = mujoco.MjvGeom()
        mjr_com_pin.type = mujoco.mjtGeom.mjGEOM_SPHERE
        mjr_com_pin.size = np.array([0.02, 0.02, 0.02])  # 半径
        mjr_com_pin.pos = np.zeros(3)
        mjr_com_pin.mat = np.eye(3)
        mjr_com_pin.rgba = np.array([1, 0, 0, 1])  # 红色
        # 创建一个箭头表示ZMP
        mjr_zmp = mujoco.MjvGeom()
        mjr_zmp.type = mujoco.mjtGeom.mjGEOM_ARROW
        mjr_zmp.size = np.array([0.01, 0.01, 0.05])
        mjr_zmp.pos = np.zeros(3)
        mjr_zmp.mat = np.eye(3)
        mjr_zmp.rgba = np.array([1, 0, 0, 1])  # 红色
        
        # Unitree H1的执行器与关节qpos索引映射
        joint_qpos_indices = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        
        # 使用PD控制器保持关节在初始位置，但允许外力扰动
        count = 0
        while viewer.is_running():
            # 对每个执行器关节应用PD控制
            # if home_key and home_key.qpos is not None:
            #     # 应用PD控制到每个执行器对应的关节

            #     for i in range(10):
            #         joint_qpos_idx = joint_qpos_indices[i]
                    
            #         # 使用PD控制器计算控制输出
            #         control_output = pd_controller.calculate(
            #             0,  # 目标位置
            #             data.qpos[joint_qpos_idx],      # 当前位置
            #             0,                              # 目标速度
            #             data.qvel[joint_qpos_idx]       # 当前速度
            #         )
            #         # 将控制输出（力矩）应用到执行器
            #         data.ctrl[i] = control_output



            mujoco.mj_step(model, data)
            viewer.sync()
            count += 1
            
            # 获取渲染上下文
            con = viewer.user_scn
            
            # 更新并可视化COM和ZMP
            state = read_robot_state(model, data)
            
            # 更新COM位置
            mjr_com.pos[:] = state['com']
            # 更新ZMP位置（在地面上）
            mjr_zmp.pos[:] = [state['zmp'][0], state['zmp'][1], 0.01]

            mjr_com_pin.pos[:] = com_pin(data.qpos, pin_model, data_pin)

            # 添加：将几何体添加到用户场景
            if con.ngeom < con.maxgeom:
                mujoco.mjv_initGeom(
                    con.geoms[con.ngeom],
                    mjr_com.type,
                    mjr_com.size,
                    mjr_com.pos,
                    mjr_com.mat.flatten(),
                    mjr_com.rgba
                )
                con.ngeom += 1
            
            if con.ngeom < con.maxgeom:
                mujoco.mjv_initGeom(
                    con.geoms[con.ngeom],
                    mjr_com_pin.type,
                    mjr_com_pin.size,
                    mjr_com_pin.pos,
                    mjr_com_pin.mat.flatten(),
                    mjr_com_pin.rgba
                )
                con.ngeom += 1

            # if con.ngeom < con.maxgeom:
            #     mujoco.mjv_initGeom(
            #         con.geoms[con.ngeom],
            #         mjr_zmp.type,
            #         mjr_zmp.size,
            #         mjr_zmp.pos,
            #         mjr_zmp.mat.flatten(),
            #         mjr_zmp.rgba
            #     )
            #     con.ngeom += 1
            
            # 每100步打印一次状态信息
            if count % 100 == 0:
                #print(mjr_com_pin.pos-mjr_com.pos)
                print(mjr_com.pos-data.qpos[0:3])
            time.sleep(0.01)

if __name__ == "__main__":
    demo_h1_pd_control()