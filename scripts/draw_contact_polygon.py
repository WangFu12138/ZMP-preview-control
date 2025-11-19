import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial import ConvexHull


def get_support_points(model, data, foot_names=["left_foot", "right_foot"], ground_name="world"):
    """获取支撑脚与地面的接触点（世界坐标系）"""
    support_points = []
    ground_id = model.body(ground_name).id  # 地面body ID
    
    # 遍历所有接触对
    for i in range(data.ncon):
        contact = data.contact[i]
        # 获取接触双方的body ID
        body1_id = model.geom_bodyid[contact.geom1]
        body2_id = model.geom_bodyid[contact.geom2]
        
        # 获取脚部body ID
        foot1_id = model.body(foot_names[0]).id if foot_names[0] in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)] else -1
        foot2_id = model.body(foot_names[1]).id if foot_names[1] in [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)] else -1
        
        is_foot_ground_contact = (
            (body1_id == ground_id and (body2_id == foot1_id or body2_id == foot2_id)) or
            (body2_id == ground_id and (body1_id == foot1_id or body1_id == foot2_id))
        )
        
        if is_foot_ground_contact:
            # 接触点在世界坐标系中的位置
            contact_pos = contact.pos.copy()
            support_points.append(contact_pos)
    
    return np.array(support_points)


def draw_polygon(viewer, points, color=(1, 0, 0, 0.8), line_width=2000):
    """在Mujoco Viewer中绘制多边形（线段连接点）"""
    if len(points) < 2:
        return  # 至少2个点才能绘制成线或多个点构成多边形
    
    # 清除之前的自定义几何
    viewer.user_scn.ngeom = 0
    
    # 添加多边形顶点（绘制点）
    for i, p in enumerate(points):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],  # 点的大小
            pos=p,
            mat=np.eye(3).flatten(),
            rgba=np.array(color, dtype=np.float32)
        )
        viewer.user_scn.ngeom += 1
    
    # 如果点数大于等于3，连接顶点为多边形（按顺序连线，最后闭合）
    if len(points) >= 3:
        n = len(points)
        for i in range(n):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            p1 = points[i]
            p2 = points[(i+1) % n]  # 最后一个点连回第一个点
            
            # 使用mjv_connector绘制线段
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_NONE,  # 先初始化几何体
                size=[0, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array(color, dtype=np.float32)
            )
            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_LINE,
                line_width,
                p1,
                p2
            )
            viewer.user_scn.ngeom += 1


def main():
    # 加载模型
    model_path = "/home/wzn/双足/Biped-Locomotion/urdf/empty_scene.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 启动Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置可视化选项以显示接触点
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # 显示接触点
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False  # 不显示接触力
        
        # 调整可视化比例
        model.vis.scale.contactwidth = 0.05
        model.vis.scale.contactheight = 0.02
        
        # 初始化机器人姿态
        mujoco.mj_forward(model, data)
        
        while viewer.is_running():
            # 1. 运行仿真步
            mujoco.mj_step(model, data)
            
            # 2. 获取支撑点（双脚与地面的接触点）
            support_points = get_support_points(
                model, data, 
                foot_names=["left_ankle_link", "right_ankle_link"],  # 替换为你的脚部件名称
                ground_name="world"  # 替换为你的地面部件名称
            )
            
            # 3. 绘制支撑多边形（仅在有足够支撑点时）
            if len(support_points) >= 3:
                # 对支撑点进行凸包处理（确保多边形是凸的）
                try:
                    hull = ConvexHull(support_points[:, :2])  # 取x-y平面
                    convex_points = support_points[hull.vertices]
                    draw_polygon(viewer, convex_points, color=(0, 1, 0, 0.8))  # 绿色多边形
                except:
                    # 若无法计算凸包，直接使用原始点
                    draw_polygon(viewer, support_points, color=(0, 1, 0, 0.8))
            elif len(support_points) >= 2:
                # 如果只有两个点，则绘制线段
                draw_polygon(viewer, support_points, color=(0, 1, 0, 0.8))
            elif len(support_points) == 1:
                # 如果只有一个点，则绘制点
                draw_polygon(viewer, support_points, color=(0, 1, 0, 0.8))
            
            # 4. 同步Viewer
            viewer.sync()


if __name__ == "__main__":
    main()