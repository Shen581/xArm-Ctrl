# World Model Planning for Screw

这份笔记整理一个用世界模型做 screw 任务 planning 的最小清晰版本。核心问题是：先训练一个模型预测环境动力学，再在推理时用这个模型搜索未来动作序列，选出当前要执行的第一个动作。

## 1. 你的思路是否正确

你的基本想法是正确的。

真实仿真环境里，当前观测是：

$$
s_0^{sim}
$$

执行动作：

$$
a_0
$$

仿真环境真实转移到下一步：

$$
s_1^{sim}
$$

世界模型预测下一步：

$$
\hat{s}_1 = W_\theta(s_0^{sim}, a_0)
$$

训练世界模型时，可以用预测误差作为损失：

$$
L_{wm} = \|\hat{s}_1 - s_1^{sim}\|^2
$$

所以你说的：

$$
\|s_1^{estimated} - s_1^{sim}\|
$$

就是最基础的 one-step dynamics learning。这个方向没问题。

但要补充三点：

1. screw 任务通常不建议只预测完整原始 state。更推荐预测“对规划有用的状态”，比如 screw progress、contact stable、slip、done、proprio latent。
2. 训练时世界模型只是在学动力学，不直接决定动作。
3. 推理时动作来自 planning：在世界模型里搜索一串候选动作，选 cost 最低的序列，只执行第一个动作。

## 2. 训练阶段

训练阶段的目标是学一个世界模型：

$$
W_\theta(s_t, a_t) \rightarrow \hat{s}_{t+1}
$$

也可以写成 latent 形式：

$$
z_t = E_\theta(s_t)
$$

$$
\hat{z}_{t+1} = F_\theta(z_t, a_t)
$$

$$
\hat{s}_{t+1} = D_\theta(\hat{z}_{t+1})
$$

### 2.1 数据

从仿真里收集轨迹：

$$
\tau = \{s_t, a_t, s_{t+1}, r_t, done_t\}_{t=0}^{T-1}
$$

动作可以来自：

- teacher policy
- student policy
- random exploration
- CEM/MPC 早期探索
- 人类 teleop 数据

对 screw 任务，数据需要覆盖：

- 未接触
- 刚接触
- 稳定旋转
- 打滑
- 卡住
- 脱离
- 成功旋入
- 失败 reset

### 2.2 世界模型输入输出

最简单版本：

$$
\hat{s}_{t+1} = W_\theta(s_t, a_t)
$$

这里的 $s_t$ 可以包含：

- 关节位置
- 关节速度
- 上一步动作或目标位置
- tactile/contact
- screw progress，如果训练时 sim 可读
- nut/contact 状态，如果训练时 sim 可读

更推荐的版本是预测 task-relevant state：

$$
W_\theta(s_t, a_t) \rightarrow
\{\hat{s}_{t+1}^{prop}, \hat{p}_{t+1}, \hat{c}_{t+1}, \hat{d}_{t+1}, \hat{r}_t\}
$$

其中：

- $\hat{s}_{t+1}^{prop}$：下一步 proprio state
- $\hat{p}_{t+1}$：下一步 screw progress
- $\hat{c}_{t+1}$：稳定接触概率
- $\hat{d}_{t+1}$：失败或 done 概率
- $\hat{r}_t$：预测 reward

### 2.3 训练损失

最小 one-step loss：

$$
L_{state} = \|\hat{s}_{t+1} - s_{t+1}\|^2
$$

如果额外预测 screw progress：

$$
L_{progress} = \|\hat{p}_{t+1} - p_{t+1}\|^2
$$

如果预测 progress 增量：

$$
L_{\Delta p} = \|\widehat{\Delta p_t} - (p_{t+1} - p_t)\|^2
$$

如果预测接触稳定性：

$$
L_{contact} = BCE(\hat{c}_{t+1}, c_{t+1})
$$

如果预测失败或 done：

$$
L_{done} = BCE(\hat{d}_{t+1}, done_{t+1})
$$

如果预测 reward：

$$
L_{reward} = \|\hat{r}_t - r_t\|^2
$$

总损失可以写成：

$$
L =
\lambda_s L_{state}
+ \lambda_p L_{progress}
+ \lambda_{\Delta p} L_{\Delta p}
+ \lambda_c L_{contact}
+ \lambda_d L_{done}
+ \lambda_r L_{reward}
$$

如果使用 latent dynamics，还可以加：

$$
L_{latent} = \|\hat{z}_{t+1} - stopgrad(z_{t+1})\|^2
$$

## 3. 推理阶段

推理阶段不是直接让世界模型输出动作。世界模型只负责预测：“如果我执行这个动作，未来会怎样”。

动作来自 planning。

当前真实观测是：

$$
s_0^{sim}
$$

我们要找一段未来动作序列：

$$
A = \{a_0, a_1, ..., a_{H-1}\}
$$

让这段动作在世界模型中 rollout 后 cost 最小：

$$
A^* = \arg\min_A J(A; s_0)
$$

最后只执行第一个动作：

$$
a_0^{estimated} = A^*_0
$$

下一步拿到新的真实观测 $s_1^{sim}$ 后，再重新 planning。这叫 MPC 或 receding horizon control。

### 3.1 世界模型内 rollout

给定候选动作序列 $A$：

$$
\hat{s}_{1} = W_\theta(s_0, a_0)
$$

$$
\hat{s}_{2} = W_\theta(\hat{s}_1, a_1)
$$

$$
...
$$

$$
\hat{s}_{H} = W_\theta(\hat{s}_{H-1}, a_{H-1})
$$

然后根据预测的整段未来状态算 cost。

### 3.2 Cost Function

如果有目标状态 $s_{goal}$，最基础的 cost 是：

$$
J(A; s_0) = \sum_{k=1}^{H} \|\hat{s}_k - s_{goal}\|_Q^2
+ \sum_{k=0}^{H-1} \|a_k\|_R^2
$$

但 screw 任务更建议用 task cost：

$$
J(A; s_0) =
- w_p \sum_{k=0}^{H-1} \Delta \hat{p}_k
- w_c \sum_{k=1}^{H} \hat{c}_k
+ w_s \sum_{k=1}^{H} \hat{slip}_k
+ w_d \sum_{k=1}^{H} \hat{done}_k
+ w_a \sum_{k=0}^{H-1} \|a_k\|^2
+ w_{smooth} \sum_{k=1}^{H-1} \|a_k - a_{k-1}\|^2
$$

含义：

- 最大化 screw progress
- 奖励稳定接触
- 惩罚打滑
- 惩罚失败或脱离
- 惩罚动作太大
- 惩罚动作变化太剧烈

如果目标是达到某个最终 screw depth，可以加终端 cost：

$$
J_{terminal} = \|\hat{p}_{H} - p_{goal}\|^2
$$

最终：

$$
J_{total} = J + w_T J_{terminal}
$$

### 3.3 怎么从 plan 得到动作

常见做法有三种。

第一种：random shooting。

随机采样很多条动作序列：

$$
A^1, A^2, ..., A^N
$$

用世界模型分别 rollout，计算每条序列的 cost，选择 cost 最小的序列：

$$
i^* = \arg\min_i J(A^i; s_0)
$$

然后执行：

$$
a_0 = A^{i^*}_0
$$

第二种：CEM。

初始化动作序列分布：

$$
A \sim \mathcal{N}(\mu, \Sigma)
$$

每轮采样多条序列，选 top-k elite，再用 elite 更新 $\mu, \Sigma$。重复几轮后，取均值序列的第一个动作：

$$
a_0 = \mu_0
$$

第三种：gradient planning。

如果世界模型可微，可以把动作序列 $A$ 当成待优化变量，直接对 cost 反向传播：

$$
A \leftarrow A - \alpha \nabla_A J(A; s_0)
$$

优化若干步后执行 $A_0$。

灵巧手 screw 任务里，优先推荐 CEM 或 MPPI，因为接触动力学复杂，梯度可能不稳定。

## 4. 完整算法

### 4.1 Train

1. 在 sim 中采集轨迹：

$$
(s_t, a_t, s_{t+1}, r_t, done_t)
$$

2. 用世界模型预测下一步：

$$
\hat{s}_{t+1} = W_\theta(s_t, a_t)
$$

3. 计算损失：

$$
L =
\lambda_s L_{state}
+ \lambda_p L_{progress}
+ \lambda_c L_{contact}
+ \lambda_d L_{done}
+ \lambda_r L_{reward}
$$

4. 反向传播更新世界模型参数 $\theta$。

5. 用新的策略或 planner 继续采更多数据，扩充数据集。

### 4.2 Inference

1. 读取当前真实观测：

$$
s_t^{sim}
$$

2. 采样或优化多条候选动作序列：

$$
A^i = \{a_t^i, ..., a_{t+H-1}^i\}
$$

3. 对每条动作序列，用世界模型 rollout：

$$
\hat{s}_{t+k+1}^i = W_\theta(\hat{s}_{t+k}^i, a_{t+k}^i)
$$

4. 计算每条序列的 cost：

$$
J(A^i; s_t)
$$

5. 选择最优序列：

$$
A^* = \arg\min_i J(A^i; s_t)
$$

6. 只执行第一个动作：

$$
a_t = A^*_0
$$

7. 环境前进一步，得到新观测，再重复 planning。

## 5. 需要完善的地方

你的原始思路里，最需要补齐的是下面这些点。

### 5.1 不要只做 one-step state loss

只用：

$$
\|\hat{s}_{t+1} - s_{t+1}\|^2
$$

可以训练出一个基础世界模型，但 planning 时可能不够。因为 screw 任务关心的不是所有 state 维度都预测得一样准，而是下面这些量预测得准：

- screw progress
- 接触是否稳定
- 是否打滑
- 是否会失败
- 动作是否会让物体脱离

所以需要加 progress/contact/done/reward head。

### 5.2 需要 multi-step loss

planning 会连续 rollout 很多步。如果世界模型只在 one-step 上准，rollout 多步后可能漂移。可以加 multi-step prediction：

$$
L_{multi} = \sum_{k=1}^{K} \|\hat{s}_{t+k} - s_{t+k}\|^2
$$

其中：

$$
\hat{s}_{t+k+1} = W_\theta(\hat{s}_{t+k}, a_{t+k})
$$

### 5.3 需要动作约束

planner 可能会找到模型里看起来很强、真实环境里很怪的动作。需要加入：

- action bound
- action smoothness
- joint limit cost
- torque cost
- policy prior cost

policy prior 可以来自 teacher/student：

$$
J_{prior} = \|a_t - \pi_{prior}(s_t)\|^2
$$

这样 planner 不会离已有策略太远。

### 5.4 需要模型不确定性

如果世界模型在某些状态动作上没见过，planner 可能会利用模型错误。可以训练 ensemble：

$$
W_1, W_2, ..., W_M
$$

用不同模型预测的方差作为不确定性惩罚：

$$
J_{uncertainty} = w_u Var(\hat{s}_{t+1}^{1:M})
$$

### 5.5 要区分 sim state 和 deploy observation

训练时可以读到 sim 的真值，例如 object pose、nut dof、contact force。部署时不一定有。建议：

- 世界模型输入只用部署时可观测的信息
- sim 特权量可以作为训练标签
- 不要让 inference 依赖真实机器人拿不到的 state

## 6. 推荐的最小可行版本

最小版本可以这样做：

输入：

$$
s_t = [proprio\_history, tactile/contact, previous\_action]
$$

动作：

$$
a_t = 12D\ DOF\ target/action
$$

世界模型输出：

$$
W_\theta(s_t, a_t) \rightarrow
\{\hat{s}_{t+1}^{prop}, \widehat{\Delta p_t}, \hat{contact}_{t+1}, \hat{done}_{t+1}\}
$$

训练损失：

$$
L =
\lambda_s \|\hat{s}_{t+1}^{prop} - s_{t+1}^{prop}\|^2
+ \lambda_p \|\widehat{\Delta p_t} - (p_{t+1}-p_t)\|^2
+ \lambda_c BCE(\hat{contact}_{t+1}, contact_{t+1})
+ \lambda_d BCE(\hat{done}_{t+1}, done_{t+1})
$$

planning cost：

$$
J =
- w_p \sum_{k=0}^{H-1} \widehat{\Delta p_k}
- w_c \sum_{k=1}^{H} \hat{contact}_k
+ w_d \sum_{k=1}^{H} \hat{done}_k
+ w_a \sum_{k=0}^{H-1} \|a_k\|^2
+ w_{smooth} \sum_{k=1}^{H-1} \|a_k-a_{k-1}\|^2
$$

推理时：

$$
a_t = A^*_0
$$

其中：

$$
A^* = \arg\min_A J(A; s_t)
$$

这就是一个完整的 world model + MPC screw 框架。
