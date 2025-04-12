import numpy as np
import os
import torch
from isaacgym import gymapi, gymtorch
from xarmenvs.tasks.base.vec_task import VecTask
from isaacgym.torch_utils import torch_rand_float
from isaacgym import gymutil
from isaacgym.torch_utils import quat_conjugate, quat_mul



class Xarm(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.control_type = self.cfg["env"].get("controlType", "ee")  # "ee" or "joints"
        self.block_gripper = self.cfg["env"].get("blockGripper", True)

        # 状态和动作的大小定义
        self.cfg["env"]["numObservations"] = 6+3 if self.block_gripper else 7+3  # ee_pos(3) + ee_vel(3) + [gripper_width]
        self.cfg["env"]["numActions"] = 3 if self.control_type == "ee" else 6  # 根据控制类型ee或joint调整

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]


        # 初始pose
        default_pose = torch.tensor([0.0, -0.3, 0.0, 1.0, 0.0, 0.0], device=self.device)
        self.initial_dof_pos = default_pose.unsqueeze(0).expand(self.num_envs, -1)
        self.initial_dof_vel = torch.zeros_like(self.initial_dof_pos)
        # 关节限制。这个要调，不清楚参数是否正确
        self.dof_limits_lower = torch.tensor([-6.28318530718, -2.059, -3.927, -6.28318530718, -1.69297, -6.28318530718], device=self.device)
        self.dof_limits_upper = torch.tensor([6.28318530718, 2.0944, 0.19198, 6.28318530718, 3.14159265359, 6.28318530718], device=self.device)
        # 中性位置
        self.neutral_gripper_jnt = 0.43
        self.neutral_joint_values = torch.tensor(
            [0.0, -1.103, -0.524, 0.0, 1.627, 0.0] +
            [self.neutral_gripper_jnt] * 6 if not self.block_gripper else [0.0, -1.103210210442149, -0.5245520784266224, 0.0, 1.62760333106346, 0.0],
            device=self.device
        )
        # 末端执行器链接索引
        self.ee_link_index = 6 if self.block_gripper else 14

        # 任务相关参数
        self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        # self.target_radius = cfg["env"].get("targetRadius", 0.05)  # 目标区域半径
        self.reach_threshold = cfg["env"].get("reachThreshold", 0.03)  # 判定阈值
        # 可视化目标（小球）
        self.target_handles = []
        # 目标位置缓冲区 (x,y,z)
        self.visual_handles = []  # 用于存储视觉对象的句柄
        # 创建可视化目标
        self._create_target_visualization()

        # 添加 Jacobian 张量初始化
        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "xarm6")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)

        # 获取末端执行器刚体索引
        self.ee_index = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.robot_handles[0], "link6", gymapi.DOMAIN_SIM
        )

        # 提取末端 Jacobian（控制前6个关节）
        self.j_eef = self.jacobian[:, self.ee_index - 1, :, :6]  # shape: [num_envs, 6, 6]

        # 阻尼系数
        self.damping = 0.05  # 不清楚应该为多少

        # 重置环境
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def _create_target_visualization(self):
        # 使用Debug Line绘制目标标记，不怎么吃性能，和_update_target_visualization搭配使用
        #self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self._update_target_visualization()

    def _update_target_visualization(self):
        # 绘制球体线框，非实体
        radius = 0.03  # 球体半径
        num_segments = 64  # 球体的线段精度（越高越圆滑）

        self.gym.clear_lines(self.viewer)  # 保持场上每个环境只有一个目标小球

        for i in range(self.num_envs):
            pos = self.target_positions[i].cpu().numpy()
            # 生成线框点
            lines = []
            for theta in np.linspace(0, 2 * np.pi, num_segments):
                for phi in np.linspace(0, np.pi, num_segments // 2):
                    x = radius * np.sin(phi) * np.cos(theta)
                    y = radius * np.sin(phi) * np.sin(theta)
                    z = radius * np.cos(phi)
                    lines.append([x + pos[0], y + pos[1], z + pos[2]])

            # 绘制
            for j in range(len(lines) - 1):
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,  # 每条线1个线段
                    [lines[j], lines[j + 1]],  # 线段的起点和终点
                    [[0.0, 1.0, 0.0]]  # 绿色
                )

    def create_sim(self):
        self.up_axis_idx = 2  # Z轴向上
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["plane"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["plane"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # 和场景的大小有关
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # 加载URDF资产
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../urdf/")
        asset_file = "xarm/xarm6_with_gripper.urdf" if not self.block_gripper else "xarm/xarm6_robot_white.urdf"

        asset_options = gymapi.AssetOptions()

        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = 0  # =0保持URDF中定义的原始视觉附件方向，=1翻转视觉附件的局部坐标系方向
        asset_options.disable_gravity = 0  # 重力相关参数

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)

        # 设置关节驱动模式
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i in range(self.num_dof):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = 400.0  # 不知道正确值，暂且dafault 400，越大机械臂抖动越激烈
            dof_props["damping"][i] = 40.0  # 不知道正确值，暂且dafault 40，减震

        self.envs = []
        self.robot_handles = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # 设置初始位置
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self.cfg["env"].get("basePosition", [-0.0, 0.0, 0.0]))

            # 创建actor
            robot_handle = self.gym.create_actor(
                env_ptr, robot_asset, pose, "xarm6", i, 1, 0
            )

            # 设置关节属性
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)
            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)


    def compute_observations(self):
        # 获取末端执行器位置和速度
        ee_pos = self._get_ee_position()
        ee_vel = self._get_ee_velocity()


        # 检查维度
        if ee_pos.dim() == 1:
            ee_pos = ee_pos.unsqueeze(0).expand(self.num_envs, -1)  # [3] -> [32, 3]
        if ee_vel.dim() == 1:
            ee_vel = ee_vel.unsqueeze(0).expand(self.num_envs, -1)  # [3] -> [32, 3]

        target_rel_pos = self.target_positions - ee_pos
        if not self.block_gripper:
            fingers_width = self._get_fingers_width()
            self.obs_buf = torch.cat([
                ee_pos, ee_vel, target_rel_pos,
                fingers_width.unsqueeze(-1)
            ], dim=-1)
        else:
            self.obs_buf = torch.cat([
                ee_pos, ee_vel, target_rel_pos
            ], dim=-1)

        return self.obs_buf


    def compute_reward(self, actions):
        # 获取当前末端位置
        ee_pos = self._get_ee_position()
        self.reward_type = "sparse"
        # 计算到目标的距离
        dist_to_target = torch.norm(ee_pos - self.target_positions, dim=-1)

        # 确保 tensor 在同一设备上
        device = dist_to_target.device

        # 稀疏奖励：如果机器人成功到达目标（即距离小于阈值），奖励为-1，否则为0
        if self.reward_type == "sparse":
            reach_reward = -torch.where(dist_to_target < self.reach_threshold, torch.tensor(1.0, device=device),
                                        torch.tensor(0.0, device=device))

        # 密集奖励：根据到目标的距离计算奖励，距离越小，奖励越大
        else:
            reach_reward = -dist_to_target.float()

        # 成功奖励（与稀疏奖励结合）
        success = dist_to_target < self.reach_threshold
        success_reward = success.float() * 2.0

        # 动作惩罚（防止抖动）
        action_penalty = torch.sum(actions ** 2, dim=-1) * 0.001

        # 组合奖励：稀疏奖励/密集奖励 + 成功奖励 - 动作惩罚
        self.rew_buf[:] = reach_reward + success_reward - action_penalty

        # 更新reset_buf（成功或超时）
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            success.float()
        )

        return self.rew_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # 重置关节状态
        self.dof_pos[env_ids] = self.neutral_joint_values[:self.num_dof]
        self.dof_vel[env_ids] = 0.0

        # 生成新目标位置（新的小球位置）
        rand_range = torch.tensor([0.4, 0.6, 0.4], device=self.device)  # x,y,z范围
        rand_offset = torch.tensor([0.15, -0.3, 0.15], device=self.device)
        new_targets = torch.rand((len(env_ids), 3), device=self.device) * rand_range + rand_offset

        self.target_positions[env_ids] = new_targets

        # 更新可视化小球
        self._update_target_visualization()

        # 创建完整状态张量
        target_states = torch.zeros((len(env_ids), 13), device=self.device)

        target_states[:, :3] = new_targets  # 位置

        target_states[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # 默认四元数(无旋转)

        target_states[:, 7:13] = 0


        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )
        #print(self.initial_root_states)
        root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(root_states)
        self.initial_root_states.copy_(root_states)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _solve_ik_batch(self, target_positions, target_orientations=None):
        # 使用 Jacobian 伪逆法计算逆运动学
        # 刷新张量
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 获取当前末端状态
        hand_pos = self._get_ee_position()

        hand_rot = self._get_ee_orientation()

        pos_err = target_positions - hand_pos  # 计算位置误差，目标pos减去目前pos得到差值

        # 计算旋转误差（默认保持当前朝向）
        if target_orientations is None:
            orn_err = torch.zeros_like(pos_err)
        else:
            orn_err = self.orientation_error(target_orientations, hand_rot)  # 没实现，没考虑这种情况


        dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)  # 合并误差 [num_envs, 6, 1]

        # Damped Least Squares 求解，j_eef_T[num_envs, 6, 6]
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose


        target_angle = self.dof_pos[:, :6] + u.squeeze(-1) * 0.05
        self.dof_pos[:, :6] = target_angle
        return target_angle

    def _get_ee_orientation(self):
        # 获取末端执行器朝向（四元数）
        body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        body_states = gymtorch.wrap_tensor(body_states)
        return body_states.view(self.num_envs, -1, 13)[:, self.ee_index, 3:7]

    def orientation_error(desired, current):
        # 计算四元数旋转误差（轴角形式）
        cc = quat_conjugate(current)

        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


    def pre_physics_step(self, actions):
        if self.control_type == "ee":
            target_pos = self.target_positions
            target_pos[:, 2] = torch.max(torch.tensor(0, device=target_pos.device), target_pos[:, 2])  # 保证末端在水平面之上
            target_angles = self._solve_ik_batch(target_pos)  # target_pos为末端的目标位置

            # 应用关节限制
            target_angles = torch.clamp(
                target_angles,
                self.dof_limits_lower[:6],
                self.dof_limits_upper[:6]
            )

            # 设置目标
            self._set_joint_position_targets(target_angles)


        else:  # 这边还没定义
            joint_ctrl = actions[:, :6] * 0.05  # 限制最大关节变化

            target_angles = self.dof_pos[:, :6] + joint_ctrl


        # 处理夹爪控制，还没定义
        if not self.block_gripper:
            gripper_ctrl = actions[:, -1] * 0.2  # 限制夹爪变化
            current_width = self._get_fingers_width()
            target_width = current_width + gripper_ctrl
            target_gripper_angle = self._width_to_angle(target_width)

            # 合并所有关节目标
            target_angles = torch.cat([
                target_angles,
                target_gripper_angle.unsqueeze(-1).repeat(1, 6)
            ], dim=-1)

        # 设置关节目标
        self._set_joint_position_targets(target_angles)

    def post_physics_step(self, actions):
        self.progress_buf += 1

        # 检查是否需要重置
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()

        self.compute_reward(actions)



    def _get_ee_position(self):
        # 获取末端执行器位置，直接读取环境获取
        body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        body_states = gymtorch.wrap_tensor(body_states)

        # 计算每个环境的刚体偏移量
        env_ptr = self.envs[0]  # 假设所有环境结构相同
        rigid_body_count = self.gym.get_env_rigid_body_count(env_ptr)
        ee_index = self.gym.find_actor_rigid_body_index(
            env_ptr, self.robot_handles[0], "link6", gymapi.DOMAIN_ENV
        )

        # 获取所有环境中末端执行器的位置
        ee_positions = body_states.view(self.num_envs, rigid_body_count, 13)[:, ee_index, :3]

        return ee_positions

    def _get_ee_velocity(self):
        # 使用张量 API 获取所有刚体状态
        body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        body_states = gymtorch.wrap_tensor(body_states)  # 形状 [num_rigid_bodies * num_envs, 13]

        # 重塑为 [num_envs, num_rigid_bodies, 13]
        body_states = body_states.view(self.num_envs, -1, 13)

        # 获取所有环境中末端执行器的速度（假设 ee_link_index 已正确设置）
        ee_vel = body_states[:, self.ee_link_index, 7:10]  # 线性速度在 [7:10], 结果形状 [32, 3]

        return ee_vel

    def _get_fingers_width(self):
        # 计算夹爪宽度
        # 根据关节位置计算夹爪宽度
        drive_jnt = self.dof_pos[:, 8]  # 假设第8个关节是驱动关节
        return 0.018 + 0.11 * torch.sin(0.069 - drive_jnt)

    def _width_to_angle(self, width):
        # 将夹爪宽度转换为关节角度
        asin_val = torch.clamp((width - 0.018) / 0.11, -1.0, 1.0)
        return 0.69 - torch.asin(asin_val)


    def _set_joint_position_targets(self, targets):  # 设置关节的目标角度
        # 确保关节限制是张量
        if isinstance(self.dof_limits_lower, list):
            self.dof_limits_lower = torch.tensor(self.dof_limits_lower, device=self.device)
        if isinstance(self.dof_limits_upper, list):
            self.dof_limits_upper = torch.tensor(self.dof_limits_upper, device=self.device)

        # 传入机械臂
        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(targets)
        )