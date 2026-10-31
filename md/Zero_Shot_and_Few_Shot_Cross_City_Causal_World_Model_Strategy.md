# Zero-shot 与 few-shot 跨城市公交因果世界模型迁移路线

本文档固定当前论文下一阶段的目标、术语和实验路线，避免后续把 zero-shot、target-label-free、few-shot/offline adaptation、calibrated transfer 混在一起。

## 1. 目标重新定义

当前论文结果主要是：

> 多个 source cities 训练 residual/world-model correction，held-out target city 只用于评价；target city 不用 transition labels 做训练或选择。

这已经是 cross-city transfer validation，但还不是最强形式的跨城市部署论文。

下一阶段更有冲击力的目标应当是：

> 一个在 source city 历史数据和不标定仿真上训练出的因果机制世界模型，可以在 unseen target city 中用不同等级的目标城市信息逐步实例化，并随着 target 历史数据和标签增加而稳定变好。

最终主线建议改成：

> Current version: cross-city residual validation.  
> Next version: zero-shot / few-shot cross-city causal world-model transfer.

## 2. 术语边界

必须严格区分以下设置。

| 设定 | Target city 使用什么 | 推荐术语 |
|---|---|---|
| 只用 target GTFS、路网、静态属性，不用历史 next-state labels | 路网、时刻表、静态 speed/demand summary | zero-shot / target-label-free |
| 使用 target 无标签状态分布或车辆 snapshot，但不用 next-state labels | realtime/archived vehicle positions, occupancy snapshot, unlabeled state distribution | transductive zero-shot / target-label-free |
| 使用少量 target AVL/APC passive logs 的 next-state labels | 1 小时、6 小时、1 天、1 周或若干 target route groups | target offline adaptation / few-shot |
| 使用 target 仿真 calibration 或大量 target route labels | calibrated target adaptation | calibrated / not zero-shot |
| 使用 target 在线交互 | online adaptation | offline-to-online / online adaptation |

公交问题里，完全没有 target 历史数据就直接部署通常不现实。因此本文最终应当把 zero-shot 作为最低信息量对照，把 target offline adaptation / few-shot 作为主要实用设置。

## 3. 方法路线

### 3.1 Invariant local coordinate system

不要让模型直接依赖 city-specific route id 或固定城市拓扑维度。所有城市统一映射到局部公交控制坐标：

- forward/backward headway ratio
- route position / station fraction
- segment speed quantile
- demand rate / waiting pressure
- dwell pressure
- same-line neighboring stop demand
- nearby vehicle snapshot if available
- co-line / shared-stop local neighborhood

这样 policy/world model 可以跨城市实例化。

### 3.2 Causal mechanism world model

把公交状态转移拆成机制：

- headway propagation
- passenger accumulation
- dwell time
- segment speed / travel time
- reward / passenger cost

MC-WM 负责学习 city/context latent mechanism factor。  
CFCMT 负责限制 parent set、residual path 和机制级 source transfer，避免 dense residual 把 headway、demand、dwell、speed 的城市差异混在同一个参数块里。

### 3.3 Target instantiation

到新城市时，不同信息等级使用不同实例化方式：

1. Zero-shot：只用 GTFS route/stop/trip/shape、静态 demand/speed summary、不标定 SUMO 或 rule simulator。
2. Transductive zero-shot：再加入 target 的无标签 vehicle position / headway / occupancy snapshots。
3. Few-shot：再加入少量 target AVL/APC passive next-state labels，只更新 mechanism latent、bias adapter 或 source gate，不重训完整模型。
4. Calibrated upper bound：允许 target route labels 或 calibrated simulator，作为上限而不是主方法。

策略上更推荐 model + MPC，而不是直接 policy transfer。Bus holding 的动作空间小，模型可解释；跨城市 topology 变化时，MPC 比直接 policy head 更容易防守。

## 4. 实验路线

### 4.1 Single-source zero-shot A -> B/C/D

只用一个 source city 的历史数据训练，直接迁移到其他城市。

必须报告完整 source-target matrix：

- Singapore -> Austin / Halifax / MBTA
- Austin -> Singapore / Halifax / MBTA
- Halifax -> Singapore / Austin / MBTA
- MBTA -> Singapore / Austin / Halifax

这个结果可能不稳定，但必须有，因为它最符合“一个城市训练，另一个城市直接用”的直觉。

### 4.2 Multi-source leave-one-city-out zero-shot

用除 target 外的所有 source cities 训练，target 只提供路网和无标签 summary。

这是当前 strict leave-one-city-out 的升级版，需要在论文里明确为 target-label-free cross-city transfer。

### 4.3 Few-shot target offline adaptation curve

给 target 不同数量的 passive historical labels，只更新轻量机制参数：

- 0 target labels
- 1 hour target logs
- 6 hours target logs
- 1 day target logs
- 1 week target logs
- or 1%, 5%, 10%, 25% target route groups

对比：

- H2O+-style dense residual with same target labels
- AdaRL-style adaptation gate
- FANS-RL-style factored latent adaptation
- MC-WM
- CFCMT-MCWM
- target-calibrated oracle

核心叙事是：target data 越多，机制 latent / bias / gate 越准；CFCMT-MCWM 应该以更少 target labels 接近或超过 dense H2O+-style adaptation。

### 4.4 Policy-level target-city rollout

用 target city 全量路网生成 BusSimEnv/SUMO rollout。Target 不标定。

比较：

- no holding
- Daganzo / classical holding
- H2O+-style dense residual
- AdaRL/FANS-RL-style adaptation
- MC-WM
- CFCMT-MCWM
- target-calibrated oracle

现实边界必须写清楚：真实 AVL/APC 只能验证 passive state prediction 或 OPE；hold-action counterfactual 仍需要 SUMO/BusSimEnv、自然实验、dispatch intervention logs 或 field trial。

## 5. 数据路线

现有四城继续作为主工程基线：

- Singapore LTA
- Austin / CapMetro
- Halifax Transit
- MBTA

下一步应增强 real AVL snapshot validation：

- NYC MTA Bus / Bus Time / GTFS-RT archive
- SEPTA bus vehicle positions
- MBTA GTFS-RT / API archive or self-archive

这些数据优先用于 passive state prediction、headway propagation validation、unlabeled target snapshot encoder，而不是直接声称有 hold-action counterfactual labels。

## 6. 近期工程任务

1. 在现有四城上明确产出 single-source A -> B/C/D zero-shot transfer matrix。
2. 把 current strict leave-one-city-out 重新标注为 target-label-free multi-source cross-city transfer。
3. 把 existing calibration-budget experiment 改写成 target offline adaptation / few-shot curve，并补同信息量 baselines。
4. 把 full-module CFCMT-MCWM evaluation 从 residual-only 扩展到 target-label-free model + MPC / one-step policy evaluation。
5. 加入 GTFS-RT / AVL snapshot 数据路径，做 real passive state prediction validation。

## 7. 论文 claim 边界

可以逐步走向的强 claim：

> A causal-mechanism world model trained on source-city historical data and an uncalibrated simulator can be instantiated in unseen target cities with only public network/schedule/realtime summaries, and can improve rapidly with a small amount of target passive historical data.

现在不能过早声称：

- field-ready zero-calibration control
- true counterfactual policy effect from passive AVL/APC only
- universal cross-city generalization from four cities

更稳的表述是：

> The results support CFCMT-MCWM as a target-label-efficient cross-city bus holding world-model transfer framework, with zero-shot transfer as the minimum-information baseline and few-shot target offline adaptation as the practical deployment regime.
