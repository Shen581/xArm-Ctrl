# LeWorldModel 训练、rollout 与 play step 全流程

这份文档按当前仓库代码解释 LeWorldModel 的训练与评估流程。核心代码在 `train.py`、`jepa.py`、`module.py`、`utils.py` 和 `eval.py`，外部训练循环由 `stable_pretraining.Module/Manager` 承接，环境交互与规划由 `stable_worldmodel.World/Policy/Solver` 承接。

先说最重要的一点：本仓库的 `train.py` 训练世界模型时不是在线和环境交互，也没有在训练中调用 `play_step`。训练阶段只从 HDF5 里的离线专家轨迹切出短片段，做 latent next-embedding prediction。`rollout` 和类似 `play_step` 的环境交互发生在评估/规划阶段，也就是 `eval.py` 调用 `World.step()` 时。

## 1. 训练入口在做什么

运行命令通常是：

```bash
python train.py data=pusht
```

Hydra 会读取 `config/train/lewm.yaml`，默认再读取 `config/train/data/pusht.yaml`。默认关键配置是：

```yaml
loader:
  batch_size: 128

wm:
  history_size: 3
  num_preds: 1
  embed_dim: 192

data:
  dataset:
    num_steps: ${wm.num_preds + wm.history_size}  # 4
    frameskip: 5
```

因此对 PushT 默认训练来说：

| 名称 | 默认值 | 含义 |
| --- | ---: | --- |
| `batch_size` | 128 | 每个 minibatch 有 128 个短轨迹片段 |
| `history_size` | 3 | predictor 每次看 3 个 latent context |
| `num_preds` | 1 | 预测下一步 embedding |
| `num_steps` | 4 | 一个训练样本包含 4 个下采样时间点 |
| `frameskip` | 5 | 相邻两个训练时间点之间隔 5 个原始环境 step |
| `img_size` | 224 | 图像被 resize 到 224 |
| `embed_dim` | 192 | latent embedding 维度 |

## 2. 离线轨迹怎样切成训练样本

`train.py` 用下面这行创建数据集：

```python
dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
```

HDF5 里每个 episode 有连续的 step。`HDF5Dataset` 会枚举所有合法的 `(episode_id, start_step)`，每个索引取一个短窗口。

默认 `num_steps=4`、`frameskip=5`，所以一个样本覆盖原始轨迹中的 20 个连续环境 step：

```text
原始 step:  start, start+1, ..., start+19
图像/状态:  start, start+5, start+10, start+15      -> 4 个时间点
动作:       start 到 start+19 全部保留，然后 reshape -> 4 个动作块
```

对非 action 列，比如 `pixels`、`proprio`、`state`，数据会按 `frameskip` 下采样，因此得到 4 个时间点。对 `action` 列，代码会先保留 20 个原始动作，再 reshape 成：

```text
(num_steps, frameskip * raw_action_dim)
```

如果 PushT 原始动作维度是 2，那么一个动作块就是 `5 * 2 = 10` 维。于是一个训练样本大致是：

| key | 单样本形状 | minibatch 后形状 | 用途 |
| --- | --- | --- | --- |
| `pixels` | `(4, 3, 224, 224)` | `(128, 4, 3, 224, 224)` | 世界模型训练的视觉输入 |
| `action` | `(4, 10)` | `(128, 4, 10)` | 每个训练时间点对应 5 个原始动作拼成的 action block |
| `proprio` | `(4, proprio_dim)` | `(128, 4, proprio_dim)` | 会被归一化，但当前 JEPA forward 不使用 |
| `state` | `(4, state_dim)` | `(128, 4, state_dim)` | PushT 评估重置/目标设置会用，训练 forward 不使用 |

`train.py` 还会对数据做预处理：

1. `pixels` 经过 `ToImage`、ImageNet normalize、resize 到 `224x224`。
2. 非图像列通过 `get_column_normalizer` 按整列均值/方差标准化。
3. `cfg.wm.action_dim = dataset.get_dim("action")` 保存原始动作维度。
4. 模型里实际 action encoder 输入维度是：

```python
effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
```

对 PushT 原始动作 2 维、`frameskip=5` 的例子，`effective_act_dim=10`。

## 3. 真实网络结构

`train.py` 里构造的真实 LeWM 包含这些模块：

| 模块 | 代码类/函数 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- | --- |
| 图像 encoder | `spt.backbone.utils.vit_hf(...)` | `(B*T, 3, 224, 224)` | ViT hidden states | `encoder_scale=tiny` 时 hidden size 通常是 192 |
| projector | `MLP` | `(B*T, hidden_dim)` | `(B*T, embed_dim)` | 把 ViT CLS token 投影成 latent embedding |
| action encoder | `Embedder` | `(B, T, effective_act_dim)` | `(B, T, embed_dim)` | 对 action block 做 Conv1d + MLP embedding |
| predictor | `ARPredictor` | `emb: (B, 3, 192)`, `act_emb: (B, 3, 192)` | `(B, 3, 192)` | 带 AdaLN conditioning 的 causal Transformer |
| pred projector | `MLP` | `(B*T, hidden_dim)` | `(B*T, embed_dim)` | predictor 输出再投影回 embedding 空间 |
| regularizer | `SIGReg` | `(T, B, D)` | 标量 | 约束 latent 分布接近 isotropic Gaussian |

默认 predictor 配置是：

```yaml
predictor:
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
```

所以这里不是 PPO 常见的 `[4, 2, 2]` 小 MLP。下面第 6 节会用 `[4, 2, 2]` 做一个同构小例子，方便看懂输入输出和反传。

## 4. 一个 minibatch 的 forward 过程

训练时真正执行的是 `train.py` 里的 `lejepa_forward(self, batch, stage, cfg)`。

设默认 PushT minibatch 为：

```text
B = 128
T = 4
D = 192
raw_action_dim = 2
frameskip = 5
effective_act_dim = 10
```

输入 batch：

```python
batch["pixels"]  # (128, 4, 3, 224, 224)
batch["action"]  # (128, 4, 10)
```

### 4.1 处理 NaN action

轨迹边界附近 action 可能有 NaN，代码先替换为 0：

```python
batch["action"] = torch.nan_to_num(batch["action"], 0.0)
```

### 4.2 encode 图像和动作

调用：

```python
output = self.model.encode(batch)
```

`JEPA.encode` 内部做两件事。

第一，把图像时间维展平后送进 ViT：

```text
pixels:       (B, T, C, H, W)
flatten:      (B*T, C, H, W)
ViT CLS:      (B*T, hidden_dim)
projector:    (B*T, embed_dim)
reshape:      (B, T, embed_dim)
```

默认例子里就是：

```text
(128, 4, 3, 224, 224)
-> (512, 3, 224, 224)
-> (512, 192)
-> (128, 4, 192)
```

得到：

```python
emb = output["emb"]  # (128, 4, 192)
```

第二，把 action block 编码到同一个 embedding 维度：

```text
action:          (128, 4, 10)
action_encoder:  (128, 4, 192)
```

得到：

```python
act_emb = output["act_emb"]  # (128, 4, 192)
```

### 4.3 构造 context、label 和 prediction

默认 `history_size=3`、`num_preds=1`：

```python
ctx_emb = emb[:, :3]       # (128, 3, 192)
ctx_act = act_emb[:, :3]   # (128, 3, 192)
tgt_emb = emb[:, 1:]       # (128, 3, 192)
pred_emb = model.predict(ctx_emb, ctx_act)  # (128, 3, 192)
```

可以把它理解成三组 next-embedding prediction：

```text
pred_emb[:, 0]  对齐  emb[:, 1]
pred_emb[:, 1]  对齐  emb[:, 2]
pred_emb[:, 2]  对齐  emb[:, 3]
```

也就是用当前 latent 和当前 action block 预测下一帧 latent。由于 predictor 是 causal Transformer，它在第 2、3 个位置还能看到更长的历史 context。

### 4.4 计算损失

预测损失是 latent MSE：

```python
pred_loss = (pred_emb - tgt_emb).pow(2).mean()
```

SIGReg 正则项输入是把 batch 和 time 换一下：

```python
sigreg_loss = SIGReg(emb.transpose(0, 1))  # (4, 128, 192)
```

总损失：

```python
loss = pred_loss + cfg.loss.sigreg.weight * sigreg_loss
```

默认 `cfg.loss.sigreg.weight = 0.09`。

### 4.5 训练公式总览

把代码里的训练过程写成公式，可以分成五步。设一个 minibatch 为 $\{(o_{b,t}, a_{b,t})\}$，其中 $b=1,\ldots,B$，$t=0,\ldots,T-1$。这里的 $a_{b,t}$ 不是单个原始环境动作，而是 `frameskip` 个原始动作拼成的 action block。

第一，图像 encoder 和 projector 得到 latent：

```text
h_{b,t} = Encoder(o_{b,t})[CLS]
z_{b,t} = Projector(h_{b,t})
```

第二，action encoder 把 action block 变成和 latent 同维度的条件向量：

```text
u_{b,t} = ActionEncoder(a_{b,t})
```

第三，causal predictor 根据历史 latent 和 action embedding 做 next-embedding prediction：

```text
\hat z_{b,t+1} = Predictor(z_{b,0:t}, u_{b,0:t})_t
```

在默认 `history_size=3`、`num_preds=1` 时，代码实际对齐为：

```text
输入: z_0,z_1,z_2 和 u_0,u_1,u_2
目标: z_1,z_2,z_3
预测: \hat z_1,\hat z_2,\hat z_3
```

第四，预测损失是所有 batch、时间和维度上的 MSE：

```text
L_pred = mean((\hat z_{b,t+1} - z_{b,t+1})^2)
```

对应代码：

```python
pred_loss = (pred_emb - tgt_emb).pow(2).mean()
```

第五，SIGReg 正则项对应 `module.py` 里的 `SIGReg.forward(proj)`。输入是：

```text
Z = emb.transpose(0, 1)  # (T, B, D)
```

代码每次 forward 采样 `M=num_proj` 个单位随机方向 $r_m \in \mathbb{R}^D$，把 latent 投影成一维：

```text
y_{t,b,m} = z_{t,b}^T r_m
```

然后在 `knots=17` 个 $\tau_k \in [0, 3]$ 上比较经验特征函数和标准高斯特征函数。标准高斯的一维特征函数是：

```text
phi(\tau_k) = exp(-\tau_k^2 / 2)
```

代码中的误差项等价于：

```text
err_{t,m,k} = (mean_b cos(\tau_k y_{t,b,m}) - phi(\tau_k))^2
           + (mean_b sin(\tau_k y_{t,b,m}))^2
```

再用梯形积分权重加权，并乘以 batch size：

```text
SIGReg(Z) = mean_{t,m} B * sum_k w_k * err_{t,m,k}
```

所以总训练目标是：

```text
L = L_pred + lambda * SIGReg(Z)
```

默认代码里 `lambda = cfg.loss.sigreg.weight = 0.09`。这个 loss 的梯度会同时更新 encoder、projector、action encoder、predictor 和 pred projector；SIGReg 自身只保存积分点和权重 buffer，没有需要学习的参数。

## 5. minibatch 训练和反向传播

`train.py` 先用 `random_split` 把所有短片段随机分成训练/验证，默认训练集占 90%。训练 loader 是：

```python
DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    ...
)
```

所以每次 optimizer update 使用一个随机 minibatch，包含 128 个短轨迹片段。这里的“轨迹”不是完整 episode，而是从完整 episode 里滑窗切出来的 4 个训练时间点。

外部训练循环由 `stable_pretraining.Module` 和 PyTorch Lightning 执行。实际逻辑可以简化成：

```python
for epoch in range(max_epochs):
    for batch_idx, batch in enumerate(train_loader):
        state = lejepa_forward(batch, stage="fit", cfg=cfg)
        loss = state["loss"]

        manual_backward(loss)
        clip_gradients(gradient_clip_val=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

优化器配置是：

```yaml
optimizer:
  type: AdamW
  lr: 5e-5
  weight_decay: 1e-3
```

`train.py` 里只配置了一个 optimizer group：

```python
"modules": "model"
```

因此它更新 `self.model` 下的所有可训练参数，也就是：

```text
encoder + predictor + action_encoder + projector + pred_proj
```

`SIGReg` 自身只有 buffer，没有可训练参数；它参与 loss 计算，梯度会通过 `emb` 回传到 encoder/projector，但不会更新 SIGReg 本身。

训练结束或每个 epoch 结束时，`ModelObjectCallBack` 会把完整模型对象保存到 `$STABLEWM_HOME/<run_id>/lewm_epoch_<n>_object.ckpt`。Lightning checkpoint 也会保存权重文件。

## 6. 用 `[4, 2, 2]` 小网络举一个完整例子

这里用一个小例子帮助理解形状。这个例子不是仓库真实网络，只是把大 ViT/Transformer 换成小 MLP，类似你说的 PPO 里用 `[4, 2, 2]` 代替大网络。

假设：

```text
观测 o_t 是 4 维向量，而不是图片
原始 action 是 2 维
frameskip = 1
history_size = 3
num_preds = 1
num_steps = 4
embed_dim = 2
batch_size = 2
```

定义一个小 encoder：

```text
encoder: [4, 2, 2]

输入 o_t: 4 维
hidden: 2 维
输出 z_t: 2 维 embedding
```

再定义一个小 predictor。为了直观看 shape，我们让 predictor 输入拼接后的 `[z_t, a_t]`：

```text
z_t: 2 维
a_t: 2 维
concat(z_t, a_t): 4 维

predictor: [4, 2, 2]

输出 pred_z_{t+1}: 2 维
```

一个 minibatch 输入是：

```text
obs:    (B=2, T=4, obs_dim=4)
action: (B=2, T=4, action_dim=2)
```

对第一个样本，轨迹可以写成：

```text
obs:    o0, o1, o2, o3
action: a0, a1, a2, a3
```

先编码每个 observation：

```text
z0 = encoder(o0)  # 2 维
z1 = encoder(o1)
z2 = encoder(o2)
z3 = encoder(o3)

emb shape: (2, 4, 2)
```

再构造训练对：

```text
context embedding: z0, z1, z2       -> (2, 3, 2)
context action:    a0, a1, a2       -> (2, 3, 2)
target embedding:  z1, z2, z3       -> (2, 3, 2)
```

predictor 做三次 next prediction：

```text
p1 = predictor([z0, a0])  # 预测 z1
p2 = predictor([z1, a1])  # 预测 z2
p3 = predictor([z2, a2])  # 预测 z3
```

于是：

```text
pred_emb = [p1, p2, p3]  # (2, 3, 2)
tgt_emb  = [z1, z2, z3]  # (2, 3, 2)
```

预测损失就是：

```text
pred_loss = mean((pred_emb - tgt_emb)^2)
```

如果加上 SIGReg：

```text
sigreg_loss = SIGReg([z0, z1, z2, z3] over batch)
loss = pred_loss + lambda * sigreg_loss
```

反向传播会从两个方向更新网络：

1. `pred_loss` 让 predictor 学会根据当前 latent 和 action 预测下一 latent。
2. `sigreg_loss` 让 encoder/projector 产出的 latent 不塌缩到常数，保持接近高斯分布。

这和真实代码的差别是：真实代码不是简单 concat MLP，而是：

```text
图片 -> ViT -> projector -> z_t
action block -> action_encoder -> action embedding
(z_0:z_2, action_0:action_2) -> conditional Transformer predictor -> pred_z_1:pred_z_3
```

但训练目标和 tensor 对齐方式完全一样。

## 7. rollout 在哪里用

`jepa.py` 里的 `JEPA.rollout` 是推理/规划用的，不是 `train.py` 的训练 forward。

评估时，`eval.py` 会创建：

```python
world = swm.World(...)
model = swm.policy.AutoCostModel(cfg.policy)
solver = CEMSolver(model=model, ...)
policy = WorldModelPolicy(solver=solver, config=PlanConfig(...))
world.set_policy(policy)
```

然后 `world.evaluate_from_dataset(...)` 会从数据集中取初始状态和目标状态，重置环境，再循环执行：

```python
for i in range(eval_budget):
    world.step()
```

这个 `World.step()` 就是本仓库里最接近你说的 `play_step` 的位置。它做：

```python
actions = policy.get_action(world.infos)
states, rewards, terminateds, truncateds, infos = envs.step(actions)
```

也就是先让 policy 根据当前 `infos` 选 action，再把 action 送入真实环境前进一步。

## 8. 评估时一次 play step 怎么产生 action

以 `config/eval/pusht.yaml` 和 `config/eval/solver/cem.yaml` 为例：

```yaml
plan_config:
  horizon: 5
  receding_horizon: 5
  action_block: 5

solver:
  num_samples: 300
  n_steps: 30
  topk: 30
```

如果 raw action dim 是 2，那么：

```text
action_block = 5
block action dim = 5 * 2 = 10
horizon = 5
num_samples = 300
```

CEM 每轮采样的候选动作序列形状是：

```text
candidates: (B, S, H, A_block)
          = (num_envs, 300, 5, 10)
```

`CEMSolver.solve` 每次迭代会：

1. 从当前高斯分布采样 300 条 action sequence。
2. 调用 `model.get_cost(current_info, candidates)` 评估每条序列。
3. 选择 cost 最低的 top 30 条 elite。
4. 用 elite 的均值和方差更新采样分布。
5. 重复 30 次。

最后输出：

```text
actions: (num_envs, horizon, block_action_dim)
       = (num_envs, 5, 10)
```

`WorldModelPolicy` 会取前 `receding_horizon=5` 个 block action，并 reshape 回真实环境 step：

```text
(num_envs, 5, 10)
-> (num_envs, 5 * 5, 2)
-> (num_envs, 25, 2)
```

然后放进 action buffer。之后每次 `World.step()` 只弹出一个 `(num_envs, 2)` action 给环境执行。buffer 用完后再重新规划。

## 9. `get_cost` 和 `rollout` 的完整形状

CEM 需要的是每条候选 action sequence 的 cost。LeWM 的 cost 来自 latent rollout 后和 goal latent 的距离。

调用链是：

```text
CEMSolver.solve
  -> model.get_cost(info_dict, candidates)
      -> encode(goal)
      -> rollout(info_dict, candidates)
      -> criterion(info_dict)
```

### 9.1 输入给 `get_cost`

评估时 `WorldModelPolicy._prepare_info` 会把图像转成 channel-first 并 normalize。之后 CEM 会给 info 加 sample 维并 expand：

```text
info["pixels"]: (B, S, history, 3, 224, 224)
info["goal"]:   (B, S, history, 3, 224, 224)
candidates:     (B, S, horizon, block_action_dim)
```

PushT 默认 eval 里 `world.history_size=1`，所以常见是：

```text
pixels:     (B, 300, 1, 3, 224, 224)
goal:       (B, 300, 1, 3, 224, 224)
candidates: (B, 300, 5, 10)
```

这里的 `B` 不是整个评估的 `num_envs=50`，而是 `CEMSolver` 当前正在处理的小批量大小 `current_bs`。默认 `config/eval/solver/cem.yaml` 里 `batch_size: 1`，所以每次 `get_cost` 实际常见是：

```text
pixels:     (1, 300, 1, 3, 224, 224)
goal:       (1, 300, 1, 3, 224, 224)
candidates: (1, 300, 5, 10)
```

### 9.2 encode goal

`get_cost` 先构造 goal dict，把 `goal` 当成 `pixels`：

```python
goal["pixels"] = goal["goal"]
goal = self.encode(goal)
info_dict["goal_emb"] = goal["emb"]
```

于是：

```text
goal_emb: (B, history, D)
```

注意这里 `goal_emb` 代码里没有显式保留 sample 维 `S`，因为 `get_cost` 先取了 `v[:, 0]`。随后 `criterion` 里用 `goal_emb[..., -1:, :].expand_as(pred_emb)` 把目标 embedding 广播到所有候选 action sample 上。默认 CEM `batch_size=1` 时这个广播和当前实现匹配；如果以后把 CEM `batch_size` 改成大于 1，需要特别检查这里的广播维度，必要时应显式插入 sample 维。

### 9.3 rollout 候选动作

`JEPA.rollout(info, action_sequence, history_size=3)` 做 autoregressive latent rollout。

设：

```text
B = current CEM batch size，默认是 1
S = 300
T = horizon = 5
D = 192
H0 = 当前已有历史帧数 = info["pixels"].size(2) = 1
HS = predictor history_size = 3
```

先切 action：

```text
act_0:      (B, S, H0, 10)      # 已有历史对应的 action block
act_future: (B, S, T-H0, 10)    # 未来 action block
```

再 encode 初始图像历史：

```text
init pixels: (B, H0, 3, 224, 224)
emb:         (B, H0, 192)
expand S:    (B, S, H0, 192)
flatten:     (B*S, H0, 192)
```

然后循环预测未来 latent：

```python
for t in range(T - H0):
    act_emb = action_encoder(act)          # (B*S, current_len, 192)
    emb_trunc = emb[:, -HS:]               # 最多取最近 3 个 latent
    act_trunc = act_emb[:, -HS:]           # 最多取最近 3 个 action embedding
    pred_emb = predict(emb_trunc, act_trunc)[:, -1:]
    emb = torch.cat([emb, pred_emb], dim=1)
    act = torch.cat([act, next_act], dim=1)
```

循环之后代码还会再预测一次 last state，因此输出：

```text
predicted_emb: (B, S, T+1, 192)
```

在默认例子里是：

```text
predicted_emb: (B, 300, 6, 192)
```

### 9.4 cost 怎么算

`criterion` 只看最后一步 predicted embedding 和 goal embedding 的 MSE：

```python
cost = mse(predicted_emb[..., -1:, :], goal_emb[..., -1:, :]).sum(...)
```

输出：

```text
cost: (B, S)
```

也就是每个环境、每条候选 action sequence 一个标量 cost。CEM 就按这个 cost 选 top-k。

## 10. 最终推理流程的代码级调用链

最终推理不是直接调用 `JEPA.rollout` 就结束，而是 `World`、`WorldModelPolicy`、`CEMSolver` 和 `JEPA.get_cost` 一起组成一个 MPC 控制环。

### 10.1 加载模型

`eval.py` 里这行会从 checkpoint 里找出带 `get_cost` 方法的模块：

```python
model = swm.policy.AutoCostModel(cfg.policy)
```

`AutoCostModel` 在 `.venv` 的 `stable_worldmodel.policy` 里实现。它会加载 `*_object.ckpt`，递归扫描 Lightning 模块，找到当前仓库的 `JEPA` 对象，因为 `JEPA` 实现了：

```python
get_cost(info_dict, action_candidates)
```

评估时模型会被设置成：

```python
model.eval()
model.requires_grad_(False)
```

因此推理阶段只做 cost evaluation，不更新世界模型参数。

### 10.2 policy 预处理当前观测

`World.step()` 每次执行时先调用：

```python
actions = policy.get_action(world.infos)
```

`WorldModelPolicy.get_action` 首先执行 `_prepare_info(info_dict)`。这个函数来自 `.venv` 的 `stable_worldmodel.policy.BasePolicy`，会做三件事：

1. 对 `action`、`state`、`proprio` 等非图像列应用 `StandardScaler`。
2. 对 `pixels` 和 `goal` 转成 channel-first 图像。
3. 应用 `torchvision` transform：`ToImage`、float、ImageNet normalize、resize 到 `224x224`。

所以传给 `JEPA.get_cost` 的 `pixels/goal` 已经是模型训练时同分布的 tensor。

### 10.3 action buffer、warm start 和重新规划

`WorldModelPolicy` 有一个 `_action_buffer`。如果 buffer 里还有动作，当前 `World.step()` 不会重新跑 CEM，只弹出一个动作执行：

```python
action = self._action_buffer.popleft()
```

如果 buffer 为空，才会重新规划：

```python
outputs = self.solver(info_dict, init_action=self._next_init)
actions = outputs["actions"]  # (num_envs, horizon, action_block * raw_action_dim)
```

默认 PushT：

```text
horizon = 5
receding_horizon = 5
action_block = 5
raw_action_dim = 2
solver action_dim = 10
```

CEM 返回的是 block action plan：

```text
actions: (num_envs, 5, 10)
```

policy 取前 `receding_horizon` 个 block 作为当前要执行的 plan：

```python
plan = actions[:, :keep_horizon]
rest = actions[:, keep_horizon:]
self._next_init = rest if self.cfg.warm_start else None
```

这里 `warm_start=True` 时，剩余 plan 会作为下一次 CEM 的初始化均值前缀。当前默认 `horizon == receding_horizon`，所以 `rest` 长度为 0；如果以后把 `receding_horizon` 设得比 `horizon` 小，warm start 才会明显保留未执行的尾部计划。

接着 policy 把 block action 展开成真实环境动作：

```text
(num_envs, 5, 10)
-> (num_envs, 25, 2)
```

这 25 个动作依次进入 `_action_buffer`。之后每个 `World.step()` 执行一个 `(num_envs, 2)` 原始动作，并通过 `process["action"].inverse_transform` 还原到环境动作尺度。

### 10.4 CEM 的优化公式

CEM 每一轮维护一个高斯动作分布：

```text
a^{(s)}_{0:T-1} ~ N(mu, sigma^2)
```

其中每条样本都是一条候选 action sequence。对每条候选序列，`CEMSolver` 调用：

```python
costs = self.model.get_cost(current_info, candidates)
```

LeWM 的规划目标可以写成：

```text
J(a_{0:T-1}) = || z_goal - \hat z_final(a_{0:T-1}) ||_2^2
```

其中 $\hat z_\text{final}$ 来自 `JEPA.rollout` 的最后一个 predicted embedding。CEM 选择 cost 最低的 `topk=30` 条 elite：

```text
E = topk_lowest({J(a^{(s)}_{0:T-1})}_{s=1}^S)
```

然后更新采样分布：

```text
mu    = mean(E)
sigma = std(E)
```

重复 `n_steps=30` 次后，最终返回最后一轮的 `mu` 作为动作计划。

### 10.5 外部包和本仓库的职责边界

`.venv` 里的几个包承担的是框架职责，当前仓库承担的是 LeWM 世界模型本体：

| 位置 | 职责 | 和 LeWM 的关系 |
| --- | --- | --- |
| `stable_worldmodel.data.HDF5Dataset` | 枚举 episode clip、按 `frameskip` 切片、把 action reshape 成 block | 决定训练样本的时间窗口和 action block 形状 |
| `stable_pretraining.Module` | Lightning manual optimization、`manual_backward`、梯度裁剪、optimizer/scheduler step | 真正执行反向传播和参数更新 |
| `stable_worldmodel.World` | 包装 vectorized Gymnasium env，维护 `infos/rewards/dones` | `World.step()` 是 play-step 等价位置 |
| `stable_worldmodel.policy.WorldModelPolicy` | 预处理 observation/goal、管理 action buffer、触发 CEM | 把 LeWM cost model 接成可执行 policy |
| `stable_worldmodel.solver.CEMSolver` | 采样候选 action sequence、调用 `get_cost`、用 top-k 更新分布 | 用 LeWM 的 latent cost 做 MPC 规划 |
| `jepa.JEPA` | `encode/predict/rollout/get_cost` | 世界模型本体，定义 latent dynamics 和 cost |

## 11. 从训练到 play 的一条完整故事线

用 PushT 默认配置串起来：

1. 离线数据里有很多专家 episode。
2. 训练集把每个 episode 切成长度 20 原始 step 的窗口。
3. 每个窗口变成 4 个图像时间点和 4 个 action block。
4. DataLoader 每次随机拿 128 个窗口组成 minibatch。
5. ViT encoder 把每帧图片编码成 `z_t: 192` 维。
6. action encoder 把每个 10 维 action block 编成 192 维。
7. predictor 用 `z_0,z_1,z_2` 和 `a_0,a_1,a_2` 预测 `z_1,z_2,z_3`。
8. loss 是 next-embedding MSE 加 SIGReg。
9. Lightning manual optimization 对 loss 做 backward，AdamW 更新 encoder、predictor、action_encoder、projector、pred_proj。
10. 训练完保存模型对象。
11. 评估时加载模型作为 cost model。
12. CEM 为当前状态和目标采样 300 条 action sequence。
13. LeWM 在 latent 空间 rollout 每条 action sequence，得到最后 predicted embedding。
14. predicted embedding 和 goal embedding 越近，cost 越低。
15. CEM 反复选 top-k 更新动作分布，最后给出一条 action plan。
16. policy 把 block action 展开成真实环境 action，`World.step()` 每次执行一个。
17. action buffer 用完后重新规划，直到到达目标或 eval budget 用完。

## 12. 常见容易混淆的点

1. `num_steps=4` 不是 rollout horizon。训练时它表示一个短片段有 4 个下采样时间点；评估时 CEM 的 `horizon=5` 才是规划 horizon。
2. `frameskip=5` 会让 action 维度变大。原始动作 2 维时，训练模型看到的是 10 维 action block。
3. 训练时没有在线 environment step。训练只做离线 clip 的 supervised/self-supervised next latent prediction。
4. `World.step()` 是环境交互位置，等价于很多 RL 代码里说的 `play_step`。
5. `proprio/state` 会被加载和归一化，但当前 `JEPA.encode` 只使用 `pixels` 和 `action`。
6. LeWM 不使用 stop-gradient、EMA teacher 或预训练 encoder；encoder、predictor 和 projector 都通过同一个 loss 端到端更新。
7. 训练 loss 不是 reward loss，也不是 PPO policy loss；它训练的是一个 latent world model，评估时再用 CEM 在这个 world model 上做规划。
8. 默认 CEM `batch_size=1` 不等于评估环境数 `num_envs=50`。CEM 会把 50 个环境分成小批量逐个规划；当前 `goal_emb` 的广播写法也依赖这个默认设置最稳。