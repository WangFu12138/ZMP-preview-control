import pinocchio
import numpy as np

def test_model_loading():
    urdf_filename = '/home/wzn/双足/Biped-Locomotion/23-TR-R2人形机器人0815/urdf/23-TR-R2人形机器人0815.urdf'
    
    try:
        # 从 URDF 文件构建机器人模型
        model = pinocchio.buildModelFromUrdf(urdf_filename)
        print(f"Model loaded successfully. Number of joints: {model.njoints}")
        print(f"Joint names: {model.names}")
        
        # 为模型创建数据对象
        data = model.createData()
        
        # 检查 oMi 的初始状态
        print(f"oMi before forwardKinematics: {data.oMi}")
        print(f"Type of oMi: {type(data.oMi)}")
        
        # 进行正运动学计算
        q = np.zeros(model.nq) if model.nq is not None else np.zeros(1)
        pinocchio.forwardKinematics(model, data, q)
        
        # 检查 oMi 的状态
        print(f"oMi after forwardKinematics: {data.oMi}")
        print(f"Type of oMi after FK: {type(data.oMi)}")
        
        # 如果 oMi 不是 None，尝试访问第一个关节
        if data.oMi is not None:
            try:
                print(f"Length of oMi: {len(data.oMi)}")
                if len(data.oMi) > 0:
                    print(f"First joint placement: {data.oMi[0]}")
            except Exception as e:
                print(f"Error accessing oMi elements: {e}")
        else:
            print("oMi is still None after forwardKinematics")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()