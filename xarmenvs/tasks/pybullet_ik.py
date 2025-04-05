import pybullet as p
import torch
import numpy as np


class PyBulletIKWrapper:
    def __init__(self, urdf_path, num_envs=8, device="cuda"):
        # 启动 PyBullet 物理服务器（DIRECT 或 GUI 模式）
        self.physics_client = p.connect(p.DIRECT)  # 用 p.GUI 可调试可视化
        p.loadURDF(urdf_path, physicsClientId=self.physics_client)
        self.movable_joint_indices = [1, 2, 3, 4, 5, 6]  # xArm 的关节索引
        self.movable_joint_indices = [1, 2, 3, 4, 5, 6]
        self.device = torch.device(device)
        self.num_envs = num_envs
        p.connect(p.DIRECT)  # 使用独立物理客户端

        # 加载URDF并验证关节结构
        self.robot_ids = [p.loadURDF(urdf_path, useFixedBase=True) for _ in range(num_envs)]
        self._validate_urdf_structure()

    def _validate_urdf_structure(self):
        """ 严格验证URDF关节配置 """
        num_joints = p.getNumJoints(self.robot_ids[0])
        self.movable_joints = []

        print("\n[URDF结构验证]")
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_ids[0], i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            link_name = info[12].decode('utf-8')

            print(f"Joint {i}: {joint_name} ({'Fixed' if joint_type == 4 else 'Revolute'}) -> {link_name}")

            if joint_type == p.JOINT_REVOLUTE:
                self.movable_joints.append({
                    'index': i,
                    'name': joint_name,
                    'limits': info[8:10],
                    'axis': info[13]
                })

        if len(self.movable_joints) != 6:
            raise ValueError(f"需要6个旋转关节，但找到{len(self.movable_joints)}个")

        self.ee_link_idx = self.movable_joints[-1]['index']
        print(f"\n活动关节索引: {[j['index'] for j in self.movable_joints]}")
        print(f"末端执行器: Joint {self.ee_link_idx} ({self.movable_joints[-1]['name']})")

    def solve_batch(self, target_positions):
        """ 工业级精度IK求解 """
        # 输入处理
        target_pos = target_positions.float().cpu().numpy()

        # 预分配结果数组
        results = np.zeros((self.num_envs, 6), dtype=np.float32)

        for env_id in range(self.num_envs):
            # 获取当前关节状态作为初始猜测
            current_pos = np.array([p.getJointState(self.robot_ids[env_id], j['index'])[0]
                                    for j in self.movable_joints])

            # 工业级IK参数配置
            joint_limits = np.array([j['limits'] for j in self.movable_joints])
            joint_ranges = joint_limits[:, 1] - joint_limits[:, 0]

            angles = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_ids[env_id],
                endEffectorLinkIndex=self.ee_link_idx,
                targetPosition=target_pos[env_id],
                lowerLimits=joint_limits[:, 0],
                upperLimits=joint_limits[:, 1],
                jointRanges=joint_ranges,
                restPoses=current_pos,
                jointDamping=[0.05] * 6,  # 精细调节的阻尼系数
                solver=p.IK_DLS,  # 使用阻尼最小二乘法
                maxNumIterations=500,  # 增加迭代次数
                residualThreshold=1e-6,  # 更高精度
                physicsClientId=p.DIRECT
            )

            # 转换为numpy数组并截断到6个关节
            angles = np.array(angles[:6], dtype=np.float32)

            # 关节限位保护
            angles = np.clip(angles, joint_limits[:, 0], joint_limits[:, 1])
            results[env_id] = angles

            # 验证输出
            self._debug_output(env_id, target_pos[env_id], angles)

        return torch.tensor(results, dtype=torch.float32, device=self.device)

    def _debug_output(self, env_id, target_pos, solution):
        """ 专业级调试输出 """
        ee_pos, _ = p.getLinkState(self.robot_ids[env_id], self.ee_link_idx)[:2]
        error = np.linalg.norm(np.array(ee_pos) - target_pos)

        print(f"\n[Env {env_id} IK验证]")
        print(f"目标位置: {np.round(target_pos, 4)}")
        print(f"实际到达: {np.round(ee_pos, 4)}")
        print(f"位置误差: {error:.6f} m")
        print("关节解算结果(deg):")
        for i, j in enumerate(self.movable_joints):
            print(f"{j['name']}({j['index']}): {np.rad2deg(solution[i]):.2f}°")

    def __del__(self):
        p.disconnect()