import mujoco
import numpy as np
import glfw
from scipy.optimize import minimize

import numpy as np
import pinocchio
from numpy.linalg import norm, solve
import mujoco.viewer as viewer
import time

def inverse_kinematics(target_joint,current_q, target_dir, target_pos):

    urdf_filename = '/home/wzn/双足/Biped-Locomotion/23-TR-R2人形机器人0815/urdf/23-TR-R2人形机器人0815.urdf'
    # 从 URDF 文件构建机器人模型
    model = pinocchio.buildModelFromUrdf(urdf_filename)
    # 为模型创建数据对象，用于存储计算过程中的中间结果
    data = model.createData()

    # 指定要控制的关节 ID
    JOINT_ID = target_joint
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = 1e-4
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = 1000
    # 定义积分步长，用于更新关节角度
    DT = 1e-2
    # 定义阻尼因子，用于避免矩阵奇异
    damp = 1e-12

    # 初始化迭代次数为 0
    i = 0
    while True:
        # 进行正运动学计算，得到当前关节角度下机器人各关节的位置和姿态
        pinocchio.forwardKinematics(model, data, q)
        # 计算目标位姿到当前位姿之间的变换
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        # 通过李群对数映射将变换矩阵转换为 6 维误差向量（包含位置误差和方向误差），用于量化当前位姿与目标位姿的差异
        err = pinocchio.log(iMd).vector

        # 判断误差是否小于收敛阈值，如果是则认为算法收敛
        if norm(err) < eps:
            success = True
            break
        # 判断迭代次数是否超过最大迭代次数，如果是则认为算法未收敛
        if i >= IT_MAX:
            success = False
            break

        # 计算当前关节角度下的雅可比矩阵，关节速度与末端速度的映射关系
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
        # 对雅可比矩阵进行变换，转换到李代数空间，以匹配误差向量的坐标系，同时取反以调整误差方向
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        # 使用阻尼最小二乘法求解关节速度
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        # 根据关节速度更新关节角度
        q = pinocchio.integrate(model, q, v * DT)

        # 每迭代 10 次打印一次当前的误差信息
        if not i % 10:
            print(f"{i}: error = {err.T}")
        # 迭代次数加 1
        i += 1

    # 根据算法是否收敛输出相应的信息
    if success:
        print("Convergence achieved!")
    else:
        print(
            "\n"
            "Warning: the iterative algorithm has not reached convergence "
            "to the desired precision"
        )

    # 打印最终的关节角度和误差向量
    print(f"\nresult: {q.flatten().tolist()}")
    print(f"\nfinal error: {err.T}")
    # 返回最终的关节角度向量（以列表形式）
    return q.flatten().tolist()

def walk():

    model = mujoco.MjModel.from_xml_path(
        "/home/wzn/双足/Biped-Locomotion/23-TR-R2人形机器人0815/urdf/scene.xml")
    
    data = mujoco.MjData(model)
    global init_qpos
    init_qpos = [0,0,0.87,1,0,0,0]
    with viewer.launch_passive(model, data) as mjviewer:

        #mjviewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # 显示接触点
        #mjviewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True  # 显示接触力


        #current_position = data.qpos[:7].copy()

        #count = 0

        while mjviewer.is_running():

            x = 0.0
            theta = np.pi

            R_x = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            current_qpos = data.qpos[7:].copy()
            current_qpos[0] += 0.1
            current_qpos[3] += 0.1
            current_qpos[6] += 0.1
            current_qpos[9] += 0.1
            ik_qpos_right = inverse_kinematics(12,current_qpos,R_x,[data.xpos[13][0]-data.xpos[1][0], data.xpos[13][1]-data.xpos[1][1], data.xpos[13][2]-data.xpos[1][2]])            
            ik_qpos_left = inverse_kinematics(6,current_qpos,R_x,[data.xpos[7][0]-data.xpos[1][0], data.xpos[7][1]-data.xpos[1][1], 0.45])

            # # data.qpos[7:13] = ik_qpos_left[:6]
            # # data.qpos[13:] = ik_qpos_right[6:]
            # # data.qpos[:7]= init_qpos
            # data.ctrl[:6] = ik_qpos_left[:6]
            # data.ctrl[6:] = ik_qpos_right[6:]


            mujoco.mj_step(model, data)

            mjviewer.sync()

            # 获取渲染上下文
            con = mjviewer.user_scn

            #count += 1
            time.sleep(0.01)


if __name__ == "__main__":
    walk()