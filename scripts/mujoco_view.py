import time
import mujoco
import mujoco.viewer
import numpy as np

class MjRobotAdapter:
    def __init__(self,model, data, left_act_ids, right_act_ids):
        self.model = model
        self.data = data
        self.left_ids = left_act_ids
        self.right_ids = right_act_ids
        self.dt = model.opt.timestep

    def set_leg_targets(self, q_left, q_right):
        # 将左右腿关节目标写入 data.ctrl 对应索引
        for i, idx in enumerate(self.left_ids):
            self.data.ctrl[idx] = q_left[i]
        for i, idx in enumerate(self.right_ids):
            self.data.ctrl[idx] = q_right[i]

class ZmpPreviewWalker:
    def __init__(self, preview_cls, dt, Tsup_time, Tdl_time, previewStepNum):
        self.pre = preview_cls(dt=dt, Tsup_time=Tsup_time, Tdl_time=Tdl_time, previewStepNum=previewStepNum)
        self.supPoint = [0.0, 0.065]     # 初始支撑点
        self.stepHeight = 0.05           # 初始步高
        self.stride = 0.10               # 步幅
        # 当前段缓存
        self.seg_CoM = []
        self.seg_L = []
        self.seg_R = []
        self.j = 0                       # 段内采样点索引

    def initialize(self, supPoint=None, stepHeight=None):
        if supPoint: self.supPoint = supPoint
        if stepHeight: self.stepHeight = stepHeight
        self.plan_next_segment()

    def plan_next_segment(self):
        # 一次性生成一个步态段的轨迹（含多个采样点）
        CoM_trj, footTrjL, footTrjR = self.pre.footPrintAndCoM_trajectoryGenerator(
            inputTargetZMP=self.supPoint,
            inputFootPrint=self.supPoint,
            stepHeight=self.stepHeight
        )
        self.seg_CoM = CoM_trj
        self.seg_L = footTrjL
        self.seg_R = footTrjR
        self.j = 0

    def ik_solve(self, foot_pos_L, foot_pos_R, foot_rpy):
        # 占位：把足端位姿解算为左右腿关节角
        qL = np.zeros(6)
        qR = np.zeros(6)
        return qL, qR

    def step(self, adapter):
        # 段结束则更新支撑点并重新规划
        if self.j >= len(self.seg_CoM):
            self.supPoint[0] += self.stride
            self.supPoint[1] = -self.supPoint[1]
            self.plan_next_segment()

        # 取当前采样点
        CoM = self.seg_CoM[self.j]
        footL = self.seg_L[self.j]
        footR = self.seg_R[self.j]

        # 足端目标（相对 CoM）
        targetPosL = footL - CoM
        targetPosR = footR - CoM
        targetRPY = [0.0, 0.0, 0.0]

        # IK → 关节角 → 写入 ctrl
        qL, qR = self.ik_solve(targetPosL, targetPosR, targetRPY)
        adapter.set_leg_targets(qL, qR)

        # 前进一步（仅索引递增，物理前进由外层循环执行）
        self.j += 1

class CustomViewer:
    def __init__(self, model_path, distance=3, azimuth=0, elevation=-30):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation
    
        self.adapter = MjRobotAdapter(self.model, self.data, left_act_ids, right_act_ids)
        self.walker = ZmpPreviewWalker(preview_cls,
                                       dt=self.adapter.dt,
                                       Tsup_time=0.3, Tdl_time=0.1,
                                       previewStepNum=190)

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def run_loop(self):

        self.runBefore()

        while self.is_running():

            self.runFunc()
            mujoco.mj_step(self.model, self.data)
            self.sync()
            time.sleep(self.model.opt.timestep)
    
    def runBefore(self):
        pass

    def runFunc(self):
        pass