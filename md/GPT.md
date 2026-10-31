我：
因果迁移方向，这两年最新的论文是什么，我记得huang 他们组后面又做了几篇深入的，帮我找出来
GPT：
你记得没错。以你上传的 **AdaRL / ICLR 2022** 为起点，Huang / Zhang / Fan Feng 这条线后面确实往更深处推进了：从“固定 latent space 里的少数机制变化”，扩展到 **非平稳变化、环境空间变化、组合式 causal components、networked / MARL 场景**。AdaRL 本身的核心是用 DBN/结构图显式刻画 observation、transition、reward 中哪里变了，目标域只估计少量 domain-specific change factors，尽量不重新做 policy optimization。

## 最相关的最新几篇

**1. Towards Generalizable Reinforcement Learning via Causality-Guided Self-Adaptive Representations，ICLR 2025**
作者：Yupei Yang, Biwei Huang, Fan Feng, Xinyue Wang, Shikui Tu, Lei Xu。这个基本就是 AdaRL 后面最直接的一篇：AdaRL 更多假设 source/target 共享环境变量空间，而这篇 CSR 明确处理 **distribution shift + environment space shift**，比如目标环境出现以前没见过的新敌人、新变量。它用 causal representation 判断变化是“分布变了”还是“空间变了”，再选择更新 change factor、加新变量或做 causal graph pruning。这个你应该优先读。([OpenReview][1])

**2. Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning，ICLR 2025**
作者：Xinyue Wang, Biwei Huang。方法名 WM3C。这个方向更像“因果迁移 + 组合泛化”：用 language 作为 compositional modality，把 latent space 拆成可组合的 causal components，然后在未见过的环境组合上迁移。实验包含 numerical simulation 和真实机器人 manipulation task。([arXiv][2])

**3. Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems，NeurIPS 2025 Spotlight**
作者：Hao Liang, Shuqing Shi, Yudi Zhang, Biwei Huang, Yali Du。这个对你做交通/网络系统可能最有用：它把 causal representation learning 和 meta actor-critic 结合，用 sparse local causal mask 找每个 agent 需要看的最小邻域变量，同时估计 compact domain factors；测试时用少量轨迹估计新 domain factor 来适配。场景明确包括 traffic、power、wireless grids 这类 large-scale networked systems。([OpenReview][3])

**4. Identifying Latent State-Transition Processes for Individualized Reinforcement Learning，NeurIPS 2024**
作者包括 Yuewen Sun, Biwei Huang, Peter Spirtes, Kun Zhang 等。这个不是标准 transfer RL，但和“不同个体/环境有不同 latent transition mechanism”高度相关。它关注 individual-specific latent factors 如何影响 state transition，并证明这些 latent factors 可识别，用于 individualized policy learning。([NeurIPS Proceedings][4])

**5. ACAMDA: Improving Data Efficiency in Reinforcement Learning Through Guided Counterfactual Data Augmentation，AAAI 2024**
作者包括 Yuewen Sun, Biwei Huang, Chaochao Lu, Kun Zhang。这个是因果反事实数据增强路线：恢复 temporal causal relationships，然后生成 realistic yet unobserved counterfactual data，目标是减少真实交互数据需求，并在 unseen domains 上更好。

**6. Boosting Efficiency in Task-Agnostic Exploration through Causal Knowledge，IJCAI 2024**
作者：Yupei Yang, Biwei Huang, Shikui Tu, Lei Xu。它不是 transfer policy 的核心论文，而是“先学一个因果 world model，再对下游任务 zero-shot/快速适配”的 task-agnostic exploration 路线。重点是 online causal discovery + causal structural constraints 来提升 world model 的样本效率。([ar5iv][5])

## 如果你说的“因果迁移”不限 RL，还要看这两篇

**7. Latent Covariate Shift: Unlocking Partial Identifiability for Multi-Source Domain Adaptation，TMLR 2025**
作者包括 Yuhang Liu, Mingming Gong, Biwei Huang, Kun Zhang。这个是 supervised/domain adaptation 线：提出 latent covariate shift，让 label distribution 可以跨域变化，并给出 latent content variable 的 partial/block identifiability。不是 RL，但属于更纯的 causal transfer / domain adaptation。([OpenReview][6])

**8. Identifiable Latent Polynomial Causal Models through the Lens of Change，ICLR 2024**
作者包括 Yuhang Liu, Mingming Gong, Biwei Huang, Kun Zhang。这个偏理论，是“利用跨环境变化识别 latent causal model”的基础论文之一：从线性高斯扩展到 polynomial nonlinear causal relationships 和 exponential-family noise。对理解后面 CSR/WM3C 这种 latent causal representation 的 identifiability 很有用。

## 不是近两年，但应该作为前置脉络补上

**Factored Adaptation for Non-Stationary Reinforcement Learning，NeurIPS 2022**
这个是 AdaRL 后面非常直接的一篇，方法名 FANS-RL。它自己明说是 build upon AdaRL：把 AdaRL 中“每个 domain 内固定的 change factor”扩展成 **time-evolving change factors**，处理 episode 内和 episode 间的 non-stationarity。

**Learning World Models with Identifiable Factorization，NeurIPS 2023**
方法名 IFactor。它把 RL latent state 分成四类：reward-relevant/irrelevant、controllable/uncontrollable，并给出 block-wise identifiability。这个是从“哪些状态对 policy 必要”进一步走到“world model 中哪些 latent block 可识别”的一篇。

**Action-Sufficient State Representation Learning for Control with Structural Constraints，ICML 2022**
这是 AdaRL 同一时期的基础论文，核心是 ASR：在 POMDP/high-dimensional observation 下，用结构约束找对 control 足够且最小的 state representation。它和 AdaRL 的 compact sufficient representation 思路是一脉相承的。([ar5iv][7])

## 推荐阅读顺序

你要抓这条线，直接按这个顺序读：

**AdaRL → FANS-RL → IFactor → CSR → WM3C → GSAC**

如果你的目标是交通/公交/信号控制里的因果迁移，最该重点读的是 **CSR + GSAC**。CSR 解决“目标环境出现新变量/新机制”的问题；GSAC 把这个思路推进到 networked systems / multi-agent 场景，和交通网络更贴近。

[1]: https://openreview.net/forum?id=bMvqccRmKD "Towards Generalizable Reinforcement Learning via Causality-Guided Self-Adaptive Representations | OpenReview"
[2]: https://arxiv.org/abs/2505.08361 "[2505.08361] Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning"
[3]: https://openreview.net/forum?id=dfcQFL89OM&referrer=%5Bthe+profile+of+Biwei+Huang%5D%28%2Fprofile%3Fid%3D~Biwei_Huang1%29 "Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems | OpenReview"
[4]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e1d40e928125c515099a3fdb75f06a50-Abstract-Conference.html "Identifying Latent State-Transition Processes for Individualized Reinforcement Learning"
[5]: https://ar5iv.org/pdf/2407.20506 "[2407.20506] Boosting Efficiency in Task-Agnostic Exploration through Causal Knowledge"
[6]: https://openreview.net/forum?id=9kFlOyLwyf "Latent Covariate Shift: Unlocking Partial Identifiability for Multi-Source Domain Adaptation | OpenReview"
[7]: https://ar5iv.org/pdf/2110.05721 "[2110.05721] Action-Sufficient State Representation Learning for Control with Structural Constraints"

我：
https://github.com/erzhu419/H2Oplus
你觉得这个项目如果想加上因果，哪个方法最合适，我目前用的single agent，当然不局限于这几个方法
以及还有个问题，就是这几个方法里是否都前置因果关系DAG识别？还是手动识别？我自己的理解是AdaRL是手动设置的因果关系，而这几个方法里的因果关系识别都是如何做的？

GPT：
直说：**你这个 H2Oplus 项目不适合直接塞完整 AdaRL / CSR / WM3C。最合适的是在现有 H2O+ 的 dynamics-ratio / discriminator 位置，加一个“因果分解的 dynamics gap estimator”**。也就是：

> **H2O+ backbone 不动；把 `D(real vs sim)` 从普通判别器改成 causal-masked / factored dynamics discriminator。**

原因很简单：H2O+ 本来就是为 **offline real data + imperfect simulator** 的 dynamics gap 设计的，论文核心就是用 dynamics-aware mixed value update，把真实离线数据和有偏仿真数据混起来，同时用 dynamics ratio 修正 simulator 的 next-state 偏差。([ar5iv][1]) 你的 repo 也已经往这个方向改了：bus 版本里用了 SUMO 当 Real World、简化 bus sim 当 Dynamics Gap Simulator，并且设计了 snapshot reset、context-aware discriminator、early truncation。([GitHub][2])

## 我会选的方法

**第一选择：FANS/GSAC 思路，但不是照搬，而是做一个轻量版 causal-factored discriminator。**

你现在是 single agent，但交通系统状态不是单体状态。你的 action 是一个 ego bus 的 holding time，可是环境里有前车、后车、站点等待人数、局部路段速度、全局密度。这里的“因果图”不是 multi-agent 图，而是 **temporal DBN：哪些变量会影响下一步哪些变量**。

你现在代码里已经有雏形了：`h2oplus_bus.py` 里 bus 版本默认用 `ZOnlyDiscriminator(z_t, z_t1)`，再通过 `compute_z_importance_weight` 计算 simulator transition 的权重；这个权重进入 simulated Q loss，而不是直接喂给 actor。([GitHub][3]) `data_utils.py` 里也已经有 `ZOnlyDiscriminator`、`TransitionDiscriminator`、`DynamicsDiscriminator`、`FactoredDynamicsDiscriminator` 等候选结构，其中 `FactoredDynamicsDiscriminator` 已经明确在做 “D_sa - D_sas isolates dynamics from policy”，并只用 fwd headway、bwd headway、gap 这些 dynamics-sensitive features。([GitHub][4])

所以最自然的升级不是 AdaRL，而是：

**Causal-Masked Factored Dynamics Discriminator**

形式上可以是：

```text
D_i = D_i( child_i,t+1 | Pa_i,t, action_t, context_t )
log w = sum_i log D_i(real | transition_i) - log D_i(sim | transition_i)
```

其中 child 可以是：

```text
next_forward_headway
next_backward_headway
next_gap
next_load / waiting
next_segment_speed / density
reward-relevant next state
```

父节点 Pa 不要全连接，而是按交通机理给 mask：

```text
holding_time -> departure_time -> forward/backward headway -> reward
local traffic speed/density -> travel time -> arrival time -> headway
waiting_count / arrival_rate -> load / dwell time -> delay / reward
front/back bus state -> headway / bunching risk
```

这比“把所有 obs/action/z 全塞进 MLP 判别器”更像因果，因为它强制 discriminator 只看真正可能产生机制差异的局部转移模块，而不是利用 line_id、bus_id、时间段、状态分布差异这些 shortcut 去区分 SUMO 和 Sim。

## 不推荐直接上 AdaRL 的原因

AdaRL 的思想很适合写 related work，但直接嵌进你这个项目成本太高。它要学一个 DBN over observation/latent state/reward/action/change factors，然后估计 minimal sufficient representations，再用 target samples 估计 domain-specific factor。上传的 AdaRL 论文里说得很清楚：它学习 DBN、识别最小 policy-transfer 表示，并在目标域只估计低维 change factor；模型估计部分还用 MiSS-VAE、binary masks、sparsity regularization。

你的 H2Oplus 当前问题不是“从多个 source domains 学一个可泛化 policy family”，而是更具体的：

```text
Real SUMO offline data 少
Simplified simulator 便宜但会漂移
需要知道哪些 sim transition 值得信
```

这正好是 H2O+ / DARC / dynamics-ratio 那条线的问题，不是完整 AdaRL 的问题。H2O+ 论文自己也说 H2O+ 用 dynamics-aware mixed value update，并采用 dynamics ratio reweighting 来修正 simulator dynamics 产生的问题 next state。([ar5iv][1])

## 如果你要写成论文贡献，我会这样包装

不要说“我们做了 causal discovery”。那很危险，审稿人会追着问 identifiability、intervention、confounding、behavior policy bias。

更稳的表述是：

> We incorporate causal-structured dynamics gap estimation into hybrid offline-and-online RL. Instead of estimating the real-vs-sim discrepancy over the full transition, we factorize the discriminator according to a temporal causal structure of bus operations, and use the resulting factor-wise dynamics ratio for simulated transition reweighting and early truncation.

也就是：**causal structure as inductive bias**，不是 claim 自动发现完整 DAG。

## 这几个方法到底是不是都要先识别 DAG？

不是。区别很大。

| 方法          | DAG/因果关系怎么来                                                                                                                                                                                      | 对你项目的适配性                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| **AdaRL**   | 不是纯手动。它先规定 DBN 模板和变量类型，然后通过 MiSS-VAE 学 binary masks；在 MDP 情况下给了 structural identifiability，POMDP 下也用条件独立关系定位变化。实际实现里仍然依赖强结构假设和候选变量设计，不是完全自动发现。                                                   | 理论漂亮，但太重                                                   |
| **FANS-RL** | 联合学习 factored MDP causal structure 和 time-varying change factors；不是先跑一个独立 DAG discovery 再做 RL。它声称在标准假设下可恢复 transition/reward 的 causal graph 和 change-factor 到 state component 的部分结构。([arXiv][5]) | 适合你，因为你的 sim drift / demand / traffic 状态本质是 non-stationary |
| **IFactor** | 不是学完整 DAG，而是把 latent state 分成 reward-relevant / reward-irrelevant、controllable / uncontrollable 等 identifiable blocks。([arXiv][6])                                                               | 如果你用高维 latent world model 才有必要；你现在 engineered state 更不需要   |
| **CSR**     | 学 latent causal variables 和结构关系，用于判断 target 是 distribution shift 还是 environment-space shift。([arXiv][7])                                                                                         | 如果目标线路/站点/车辆类型出现“新变量”，可以考虑；当前阶段太重                          |
| **WM3C**    | 用语言引导 decomposition，学习 composable causal components；重点是 unseen dynamics / unseen task composition。([arXiv][8])                                                                                   | 除非你要跨城市、跨线路语义泛化，否则不优先                                      |
| **GSAC**    | 学 sparse local causal mask 和 compact domain factors，面向 traffic/power/wireless 这种 networked systems；不是手写完整 DAG，但候选邻域/locality 仍是先验。([arXiv][9])                                                   | 你以后做多线路、多agent时非常相关；single-agent 阶段可借它的 local mask 思想      |

所以你的理解“**AdaRL 是手动设置因果关系**”不完全准确。更准确是：

> AdaRL 手动规定了变量模板、时间方向、候选结构和模型形式；具体边/mask 理论上是学出来的。但它不是那种完全无先验、从原始数据自动发现完整 DAG 的方法。

这点很关键。交通控制里更现实的做法是 **半手动、半学习**：交通机理给候选 DAG / mask，模型只学 mask 强度、domain factor、transition discrepancy。

## 我建议你当前版本这样改

先做最小可行版本：

1. **保留 H2O+ 训练框架**：real offline + sim online + mixed value update 不动。
2. **替换/增强 discriminator**：从 `ZOnlyDiscriminator(z_t,z_t1)` 升级到 `CausalFactoredDynamicsDiscriminator`。
3. **不要把 z、w 喂给 actor**：你自己的文档也写了，`w`、`z_t`、`z_t1` 只能进 buffer weighting 和 early truncation，不能泄露给 actor。([GitHub][2])
4. **先手工定义 causal parents**：比如 action 只允许影响 headway/gap/dwell/arrival-time 相关模块；bus_id、line_id 只能作为分层变量，不让 discriminator 用它们做 real/sim shortcut。
5. **再加 soft mask 学习作为增强版**：每个 child head 前面放 Concrete / Gumbel-sigmoid mask，加 L1 penalty。初始化用手工图，训练后看 mask 有没有恢复合理结构。

最适合你现在代码的落点，其实就是把现有 `FactoredDynamicsDiscriminator` 做严肃：从“选 fwd_hw/bwd_hw/gap 三个特征”升级成“按公交运行因果机制分解 transition module”。这个工作量小、和 H2O+ 主线一致、论文叙事也最稳。

[1]: https://ar5iv.org/abs/2309.12716 "[2309.12716] H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps"
[2]: https://github.com/erzhu419/H2Oplus/blob/master/H2O%2B.md "H2Oplus/H2O+.md at master · erzhu419/H2Oplus · GitHub"
[3]: https://raw.githubusercontent.com/erzhu419/H2Oplus/master/SimpleSAC/h2oplus_bus.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/erzhu419/H2Oplus/master/bus_h2o/common/data_utils.py "raw.githubusercontent.com"
[5]: https://arxiv.org/abs/2203.16582?utm_source=chatgpt.com "Factored Adaptation for Non-Stationary Reinforcement Learning"
[6]: https://arxiv.org/abs/2306.06561?utm_source=chatgpt.com "Learning World Models with Identifiable Factorization"
[7]: https://arxiv.org/abs/2407.20651?utm_source=chatgpt.com "Towards Generalizable Reinforcement Learning via Causality-Guided Self-Adaptive Representations"
[8]: https://arxiv.org/abs/2505.08361?utm_source=chatgpt.com "Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning"
[9]: https://arxiv.org/abs/2510.21427?utm_source=chatgpt.com "Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems"

我：
如果不考虑工作量，而是理论契合度+贡献，你觉得我该怎么做？
后面换城市迁移肯定是个方向，以及现在H2O+其实对后面接的sim还是需求了非常强的手工calibration过程，并不是随便搭的sim（不是说SUMO，offline data是SUMO产生的，sim是另一个自建仿真），最完美的就是不再需要从客流到路段速度的calibration，只要站点数量和线路数量一致就可以有效的迁移，你看这个角度下，哪个因果迁移更合适，当然不只局限于因果

GPT：
结论：**如果不考虑工作量，最值得做的不是“给 H2O+ 加一个因果判别器”，而是把问题升级成：**

> **Causal latent system identification + graph-local adaptive H2O+**
> 也就是：不用手工校准客流、路段速度、红绿灯等价速度，而是从少量目标城市轨迹里自动推断一组低维 latent mechanism factors，然后让 policy 和 H2O+ 的 sim weighting 都条件化在这些 factors 上。

最接近的理论底座是：

**AdaRL 的“domain-specific change factor” + FANS-RL 的“time-varying factored mechanism” + GSAC 的“traffic network locality”。**

如果只能选一个主轴，我会选 **FANS-RL / factored non-stationary causal adaptation**。AdaRL 是根；FANS 更贴合公交，因为客流、路况、信号延误不是一个固定 domain parameter，而是按时间段、路段、天气、拥堵状态变化的 latent mechanism。FANS-RL 明确把非平稳性建模成多个 individual latent change factors，并联合学习 factored MDP 的 causal structure 和 time-varying change factors。([arXiv][1])

## 先说一个硬边界

你说的“只要站点数量和线路数量一致就可以有效迁移”，如果意思是 **完全没有目标城市轨迹/日志/少量运行数据**，那不可能。因为同样的站点数、线路数，可以对应完全不同的客流强度、站间运行时间、信号延误、上下客分布、车队规模，最优 holding policy 会完全不同。

但如果意思是：

> 不再手工 calibration；只需要目标城市的一小段 offline trajectories / AVL / AFC / SUMO-like logs，由模型自己推断 latent demand / speed / dwell / delay factors。

那是合理且很有贡献的。这个方向比普通 H2O+ 强得多。

## 为什么当前 H2O+ 不够

H2O+ 本身适合“有限 offline data + imperfect simulator”的混合训练；原论文也说它的目标是同时利用 limited offline data 和 imperfect simulator，并处理 real/sim dynamics gap。([arXiv][2])

但你现在 repo 里的公交版本，实际上还依赖很强的工程对齐：文档里写了站点 ID、线路 ID、站点距离、发车时间要一一对应，route length 和 station positions 误差要小于 1%，还要从 SUMO 历史运行里统计每条 edge 的平均通过时间，再填成简化仿真的等效速度。([GitHub][3]) 这说明现在的 sim 不是“随便搭就能用”，而是一个经过静态和动力学校准的 low-fidelity sim。

你已经在往正确方向走了：`extract_structured_context` 里把速度改成 moving fraction，把密度改成 fractional density，这两个都是 scale-invariant 的设计，目的是减少不同 bus count、不同速度尺度造成的不可比性。([GitHub][4]) 但这还只是 **representation-level 去标定化**，不是 **mechanism-level 去标定化**。

真正有贡献的是后者。

## 我建议的完整方案

我会把项目定义成：

> **Calibration-Free Causal Adaptive Hybrid RL for Bus Holding Control**

核心不是发现一个完整 DAG，而是学习一个 **可迁移的机制分解世界模型**。

### 1. 把公交系统拆成几个 causal mechanism

不要让模型直接学：

```text
s_t, a_t → s_{t+1}
```

而是拆成：

```text
passenger arrival mechanism
dwell / boarding / alighting mechanism
running time / road speed mechanism
headway propagation mechanism
holding intervention mechanism
reward mechanism
```

对应公交场景就是：

```text
客流到达 λ_i,t  →  waiting_i,t
waiting_i,t, load_t, action_t  →  dwell_t
segment traffic θ_speed,i,t  →  travel_time_i,t
holding_t, dwell_t, travel_time_t  →  arrival_time_next
arrival_time_next, front/back bus state  →  forward/backward headway
headway, waiting, load, delay  →  reward
```

这比一个大 MLP 判别 real/sim 高级很多，因为你不再要求整个 sim 都像真实城市，而是问：

> 哪些机制可信？哪些机制要由目标城市数据推断？哪些机制对 policy 真正重要？

### 2. 用 latent factors 替代手工 calibration

你现在手工校准的是：

```text
客流 arrival_rate
路段速度 / 等效红灯延误
站点距离 / 发车时刻
```

升级后变成学习：

```text
θ_city = {
  θ_demand,
  θ_speed,
  θ_dwell,
  θ_signal_delay,
  θ_schedule_compliance,
  θ_behavior_noise
}
```

目标城市来了以后，不再人工填 Excel / config，而是用 encoder：

```text
qφ(θ_city,t | recent trajectories)
```

从最近若干条轨迹里估计这些 latent factors。

这就是把 calibration 变成 **amortized system identification**。非因果版本类似 UP-OSI：用大量 simulator randomization 训练 universal policy，再用 recent state-action history 预测 dynamics parameters。([arXiv][5]) 但你的贡献点应该是：这些 latent parameters 不是黑盒一坨，而是按公交 causal mechanisms 分解。

### 3. policy 不直接适配城市，而是适配机制

AdaRL 的精髓正好在这里：policy 写成：

```text
π(a | s_min, θ_min)
```

它不是重新训练 target policy，而是估计目标域低维 change factor，然后直接条件化 policy。你上传的 AdaRL 里明确说，目标域只需要少量 rollout 估计低维 domain-specific representation，其他参数固定，就能做 target adaptation。

你的版本可以改成：

```text
π(a_t | local causal state, inferred mechanism factors)
```

比如：

```text
π(holding_time | headway_state, station_state, θ_demand,t, θ_speed,t, θ_dwell,t)
```

这样“换城市”不再是重新调 simulator，而是估计新城市的 θ。

### 4. H2O+ 里的 discriminator 也要机制分解

现在你的代码里已经有几类 discriminator：`ZOnlyDiscriminator` 用 `z_t,z_t1`；`TransitionDiscriminator` 用 obs/action/next_obs/z；`DynamicsDiscriminator` 只预测 fwd headway、bwd headway、gap；`FactoredDynamicsDiscriminator` 用 `D_sas - D_sa` 消掉 policy distribution difference，保留 dynamics signal。([GitHub][4])

我会进一步升级为：

```text
w = Π_i w_i
```

每个机制一个权重：

```text
w_demand
w_dwell
w_speed
w_headway
w_reward
```

含义是：

```text
这个 sim transition 的“客流机制”像不像目标城市？
这个 sim transition 的“运行时间机制”像不像目标城市？
这个 sim transition 的“头距传播机制”像不像目标城市？
```

这样就不需要整个 simulator 都校准。哪怕简化 sim 的路段速度很假，只要它在某些局部机制上可用，H2O+ 仍然能利用；不可用的机制被 downweight 或被 learned world model 替换。

## 哪个 existing 方法最合适？

### 第一名：FANS-RL

FANS-RL 最贴你这个目标，因为你真正想解决的是：

```text
客流/路况/信号延误随城市、线路、时间段变化
但 policy 希望迁移
```

FANS-RL 的设定就是 factored non-stationary MDP：多个 latent change factors 随时间变化，且影响不同 state components。它比 AdaRL 更适合公交，因为公交不是“城市 A 有固定 gravity，城市 B 有固定 gravity”，而是 morning peak、off-peak、局部拥堵、站点突发客流不断变。([arXiv][6])

### 第二名：GSAC / causality + locality

GSAC 更适合你未来的 city transfer / network transfer。它明确面向 traffic、power、wireless 这种 large-scale networked systems，并把 causal representation learning 和 meta actor-critic 结合，核心是学习 sparse local causal mask 和 domain-conditioned policy。([OpenReview][7])

你现在虽然是 single agent，但公交 holding 本质上不是普通 single-agent MDP，而是：

```text
single controlled ego bus
+
networked uncontrolled environment
```

所以 GSAC 的 locality 很有用。你可以不做 MARL，但可以做：

```text
ego bus local causal neighborhood
```

也就是只让 policy 看：

```text
前车、后车、本站、下一站、局部路段、局部客流
```

而不是整张路网。这样换城市时泛化性更强。

### 第三名：AdaRL

AdaRL 是理论根基。它的优点是 clean：DBN、change factors、minimal sufficient representation、少量 target samples。AdaRL 论文里也明确通过 DBN over observations/latent states/reward/actions/change factors 来刻画跨域变化。

但它对你来说太“静态 domain adaptation”。如果只做从一个城市到另一个城市，且每个城市一个固定 θ，AdaRL 可以；但如果你想覆盖早高峰/晚高峰/路段速度/客流波动，FANS 更合适。

### CSR / WM3C 暂时不是主线

CSR 处理的是 target 环境不只是 distribution shift，还可能有 environment-space shift，比如出现新变量、新敌人、新环境元素。([OpenReview][8]) 如果你未来要从“同样站点数/线路数”扩展到“站点数量不同、线路拓扑不同、城市规则不同”，CSR 很有价值。

WM3C 更偏 compositional causal components，甚至引入 language 作为 decomposition modality，用于 unseen dynamics / unseen tasks。([arXiv][9]) 对公交来说，除非你想把“公交线、地铁接驳、BRT、环线、支线、潮汐客流”等描述文本也纳入模型，否则它现在不是最优先。

## 最有论文贡献的版本

我会这样设计：

**Causal-FANS-H2O+**

1. **Graph-local causal state encoder**
   用站点图/线路图做 GNN 或 Transformer。节点是站点、车辆、路段；边是前后站、同线前后车、换乘/共线关系。输入用 normalized / dimensionless features，不依赖绝对速度、绝对客流标定。

2. **Factored latent mechanism encoder**
   从目标城市少量 offline trajectories 推断：

   ```text
   θ_demand,t
   θ_speed,t
   θ_dwell,t
   θ_signal,t
   ```

   这些 θ 是 time-varying 的，不是城市级常数。

3. **Causal mechanism world model**
   不预测完整 next state，而是预测每个机制模块：

   ```text
   P(waiting_next | waiting, θ_demand)
   P(dwell | waiting, load, action, θ_dwell)
   P(travel_time | segment_state, θ_speed)
   P(headway_next | headway, dwell, travel_time, holding)
   ```

4. **Policy conditioning**
   policy 只依赖 reward-relevant local state 和 policy-relevant θ：

   ```text
   a = π(s_local_min, θ_policy_min)
   ```

5. **H2O+ mixed update with factor-wise confidence**
   simulator transition 不再整体接受/拒绝，而是按机制加权：

   ```text
   Q update weight = f(w_demand, w_speed, w_dwell, w_headway)
   ```

6. **Calibration-free transfer protocol**
   源城市训练时用大量 randomized uncalibrated simulators + SUMO/offline data。目标城市测试时不给手工 calibration，只给：

   ```text
   station/route schema + 少量目标轨迹
   ```

   然后模型自动 infer θ，再运行 policy。

## 这比普通 H2O+ 的贡献在哪里？

普通 H2O+ 说：

> simulator 有 dynamics gap，我用 offline data 和 online sim 混合，并用 dynamics ratio 修正。

你的新贡献可以说：

> simulator 不需要事先 calibration；我们把 dynamics gap 分解成可解释的 causal mechanism gaps，并通过 latent mechanism inference 自动完成 city-level adaptation。

这比“加一个因果图”强很多。

## DAG 到底要不要自动识别？

我建议：**不要宣称完全自动识别完整 DAG。**

最稳的是：

```text
交通物理先验给 candidate causal graph / parent sets
模型学习 soft masks 和 latent factors
```

也就是半手动、半学习。

原因很现实：公交系统里有大量未观测混杂，比如天气、事故、信号配时、节假日、临时管制。你从 offline trajectory 自动发现完整 DAG，很容易被审稿人打穿。更稳的 claim 是：

> We use causal structure as a constrained inductive bias and learn sparse mechanism masks under this candidate graph.

这和你的目标更一致。你不是要证明“从数据中发现了真 DAG”，你是要证明：

> 在弱物理结构约束下，模型可以不手工校准客流/速度，也能迁移到新城市。

## 最后一句判断

如果你只想加一点因果：做 **causal factored discriminator**。

但如果你要理论契合度和论文贡献最大：做 **FANS-style time-varying causal mechanism adaptation + GSAC-style local graph representation + H2O+ hybrid training**。

这条线最像一个完整顶刊/顶会故事：**从 calibrated sim-to-real，推进到 calibration-free mechanism-level transfer。**

[1]: https://arxiv.org/abs/2203.16582?utm_source=chatgpt.com "Factored Adaptation for Non-Stationary Reinforcement Learning"
[2]: https://arxiv.org/abs/2309.12716 "[2309.12716] H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps"
[3]: https://raw.githubusercontent.com/erzhu419/H2Oplus/master/H2O%2B.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/erzhu419/H2Oplus/master/bus_h2o/common/data_utils.py "raw.githubusercontent.com"
[5]: https://arxiv.org/abs/1702.02453?utm_source=chatgpt.com "Preparing for the Unknown: Learning a Universal Policy with Online System Identification"
[6]: https://arxiv.org/pdf/2203.16582?utm_source=chatgpt.com "[PDF] Factored Adaptation for Non-Stationary Reinforcement Learning"
[7]: https://openreview.net/forum?id=dfcQFL89OM&referrer=%5Bthe+profile+of+Biwei+Huang%5D%28%2Fprofile%3Fid%3D~Biwei_Huang1%29&utm_source=chatgpt.com "Causality Meets Locality: Provably Generalizable and ..."
[8]: https://openreview.net/forum?id=bMvqccRmKD&utm_source=chatgpt.com "Towards Generalizable Reinforcement Learning via ..."
[9]: https://arxiv.org/abs/2505.08361?utm_source=chatgpt.com "Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning"

我：
你提到世界模型，https://github.com/erzhu419/MC-WM 这是我自己做的一个残差贝尔曼世界模型，你看对 FANS-style time-varying causal mechanism adaptation + GSAC-style local graph representation + H2O+ hybrid training 有帮助么?
GPT：
有帮助，而且**很大**。但不是“直接把 MC-WM 接到 H2O+ 后面”这么简单。更准确地说：

> **MC-WM 可以成为你那个 FANS-style + GSAC-style + H2O+ 框架里的“免校准 residual mechanism adapter”。**

你现在 MC-WM 的核心已经很接近这个目标：先用 simulator transitions 训练 (M_{sim})，再学一个小 residual (\delta) 去适配 real dynamics，也就是 (M_{real}=M_{sim}+\delta)，再用 corrected world model 做 MBPO-style rollouts。repo README 里也明确写了这个三阶段结构：pretrain (M_{sim})、self-hypothesis residual loop、再做 MBPO 训练。([GitHub][1]) 代码里的 `WorldModelEnsemble` 也是典型 probabilistic ensemble，预测 next state 和 reward，并用 ensemble disagreement 表示 epistemic uncertainty。([GitHub][2])

所以直说：**MC-WM 比单纯 H2O+ 更适合解决你说的“sim 不再需要强手工 calibration”问题。**

但它现在还不够，需要从 **global residual world model** 升级成 **causal-factored residual world model**。

---

## 为什么 MC-WM 对这个方向特别有用

你现在 H2O+ 的瓶颈是：自建简化 sim 仍然要校准客流、路段速度、站点位置、发车间隔，否则 sim dynamics gap 太大。MC-WM 正好可以把这个问题改写成：

> 不再手工校准 simulator，而是学习 simulator 和 target city 之间的 residual mechanism。

也就是：

```text
粗糙 simulator + learned residual = target-city-adapted simulator
```

这个比 H2O+ 原来的 discriminator weighting 更强。H2O+ 更像是问：

```text
这个 sim transition 能不能信？
```

MC-WM 更像是问：

```text
这个 sim transition 哪里错了？错多少？能不能被结构化 residual 修正？
```

这是更高一级的问题。

你 MC-WM 里 SINDy+NAU 这部分尤其有价值。代码注释里写得很清楚：SINDy 负责从 paired data 中发现 symbolic features，NAU/NMU head 负责映射到 residual correction，并且强调 interpretable、OOD bounded、sparse。([GitHub][3]) 对公交系统来说，这正好可以对应：

```text
路段速度 residual
dwell time residual
passenger arrival residual
headway propagation residual
reward residual
```

这比一个黑盒 MLP residual 更像“机制迁移”。

---

## 但现在的 MC-WM 最大问题：它还是整体 residual，不是机制 residual

你现在的 residual adapter 是：

```text
M_real(s,a) = M_sim(s,a) + δ(s,a)
```

代码里也明确写了 residual δ 是一个小网络，输入是整体 state-action，输出整体 state/reward correction，并且训练依赖 paired sim/real data。([GitHub][2])

这在 MuJoCo gravity gap 里可以，因为状态变量连续、系统封闭、sim-real 对齐比较容易。但换到公交城市迁移，会有几个硬问题：

第一，**真实城市里没有严格 paired counterfactual transition**。你很难拿到同一个 state-action 下的 sim next state 和 real next state。代码当前 `fit` 接口需要 `sim_next_states` 和 `real_next_states`，并显式把 `real_deltas - sim_deltas` 当 residual target。([GitHub][2]) 这对城市迁移太强。

第二，**公交系统的 gap 不是一个整体函数，而是多个机制错位**：

```text
客流错
速度错
红灯/拥堵延误错
dwell time 错
车辆间相互作用错
reward shaping 错
```

一个全局 (\delta(s,a)) 能拟合，但理论贡献不够干净，也不容易解释“免校准”。

第三，**现在还没有 GSAC-style locality**。公交 holding 的状态应该是 local graph object，不应该是一个平铺 state vector。前车、后车、本站、下一站、局部路段、局部客流才是 policy-relevant causal neighborhood。GSAC 这类工作强调 sparse local causal mask 和 compact domain factors，用于 networked systems 的跨域泛化；它明确面向 traffic/power/wireless grids 这类系统。([arXiv][4])

---

## 我会怎么把 MC-WM 接进你那个大框架

我会把最终方法写成：

> **Causal-Factored Residual Bellman World Model for Calibration-Free Hybrid RL**

核心结构如下：

```text
Graph-local encoder
        ↓
Time-varying mechanism factor inference
        ↓
Causal-factored residual world model
        ↓
Bellman model-trust critic
        ↓
H2O+ hybrid value update / rollout weighting
```

更具体：

### 1. 用 GSAC-style local graph 表示替代 flat state

每个 controlled bus 不看全局状态，只看 local causal neighborhood：

```text
ego bus
front bus
back bus
current station
next station
current segment
next segment
local waiting passengers
local schedule/headway deviation
```

如果以后从 single agent 扩到 multi-agent，这个结构天然可以复用。现在即使是 single agent，也应该把它当成 **local graph-conditioned single agent**，而不是普通 MDP。

---

### 2. 用 FANS-style time-varying latent factors 替代手工 calibration

FANS-RL 的核心是：不要用一个 shared embedding 表示所有变化，而是用 individual latent change factors + causal graphs 来建模非平稳性，并联合学习 factored MDP 的 causal structure 和 time-varying change factors。([arXiv][5])

公交里这些 factors 可以是：

```text
θ_demand,i,t      站点 i 的客流强度机制
θ_speed,e,t       路段 e 的速度/拥堵机制
θ_dwell,i,t       站点 i 的上下客/dwell 机制
θ_signal,e,t      路段/交叉口延误机制
θ_bunch,t         车队聚集传播机制
```

目标城市来了以后，不再手工填 passenger rate 和 edge speed，而是从一小段目标轨迹里估计：

```text
q(θ_t | recent local trajectory)
```

这才是你想要的“免校准”。

---

### 3. 把 MC-WM 的 residual 拆成 mechanism-level residual

不要学一个：

```text
δ(s,a)
```

而是学多个：

```text
δ_demand
δ_speed
δ_dwell
δ_headway
δ_reward
```

每个 residual 只看自己的 causal parents。例如：

```text
δ_speed_e:
  input = segment_e features, time, recent travel time, θ_speed,e,t

δ_dwell_i:
  input = waiting_i, onboard_load, boarding/alighting proxy, holding action, θ_dwell,i,t

δ_headway:
  input = front/back headway, dwell, travel time, holding action, θ_bunch,t
```

这一步是最关键的。它把 MC-WM 从“sim-to-real residual model”升级成“causal mechanism adapter”。

---

### 4. 把 QΔ-Bellman 改成 mechanism-wise trust critic

你现在 `QDeltaBellman` 已经很有意思：它不是简单一步 MSE confidence，而是把 model trust 写成一个带 (\gamma>0) 的 Bellman residual critic；代码注释明确说它学习 log-confidence Bellman operator，最终 rollout weight 是 (w=\exp(Q_\Delta(s,a)))。([GitHub][6])

这对 H2O+ 非常有用，因为 H2O+ 原本需要决定“什么时候信 simulator”。你可以进一步做成：

```text
QΔ_demand
QΔ_speed
QΔ_dwell
QΔ_headway
QΔ_reward
```

最终 simulator/model rollout 的权重不是一个整体判别器给的，而是：

```text
w = exp(α1 QΔ_demand + α2 QΔ_speed + α3 QΔ_dwell + α4 QΔ_headway)
```

或者更保守：

```text
w = min(w_demand, w_speed, w_dwell, w_headway)
```

这样你可以说：

> 我们不是整体判断 simulator 是否可信，而是判断 simulator 在不同 causal mechanisms 上分别是否可信。

这个贡献比普通 H2O+ discriminator 更强。

---

## 这能不能实现“只要站点数量和线路数量一致就能迁移”？

严格说，**只靠站点数和线路数不够**。这在信息论上不可能。同样 20 个站、2 条线，可以有完全不同的客流、拥堵、站距、信号、车队密度，最优 holding policy 不可识别。

但你可以把目标改成一个合理且很强的版本：

> 不需要手工 calibration；只需要目标城市的 route schema + 少量 offline trajectories，模型自动推断 latent mechanism factors，然后迁移。

也就是说，你至少需要：

```text
站点/线路拓扑一致或可映射
状态变量 schema 一致
少量目标城市轨迹
```

不需要：

```text
手工 passenger arrival rate
手工 edge speed
手工 signal delay
手工 dwell-time formula
```

这个目标是合理的，而且很有论文价值。

---

## 最推荐的最终架构

我会这样定：

```text
Causal-Factored MC-WM-H2O+
```

组件分工：

| 组件                                 | 作用                                                           |
| ---------------------------------- | ------------------------------------------------------------ |
| **GSAC-style local graph encoder** | 解决不同城市/线路下的局部可迁移表示                                           |
| **FANS-style time-varying θ**      | 自动估计 demand/speed/dwell 等非平稳机制                               |
| **MC-WM residual adapter**         | 修正 uncalibrated simulator 的机制误差                              |
| **SINDy+NAU/self-hypothesis loop** | 发现可解释 residual features，而不是黑盒 calibration                    |
| **QΔ-Bellman trust critic**        | 判断 model/sim rollout 在长 horizon 下是否可信                        |
| **H2O+ hybrid update**             | 把 real offline data、sim rollout、corrected-model rollout 混合训练 |

这不是简单拼接。它的理论故事是：

> H2O+ handles hybrid training, MC-WM handles residual model correction, FANS handles time-varying mechanism adaptation, GSAC handles graph-local generalization.

---

## 最小实验设计也很清楚

你可以按这个 ablation 做：

```text
A0: calibrated simplified sim + H2O+
A1: uncalibrated simplified sim + H2O+
A2: uncalibrated sim + monolithic MC-WM residual
A3: uncalibrated sim + causal-factored MC-WM residual
A4: A3 + time-varying θ inference
A5: A4 + graph-local encoder / city transfer
```

如果 A3/A4 明显超过 A1/A2，你的主张就站住了：

> 不是 residual model 本身有用，而是 causal-factored residual mechanism 有用。
> 不是 simulator 被校准了，而是 calibration 被 latent mechanism inference 替代了。

---

## 最后判断

**有帮助，而且应该变成核心。**

但别把 MC-WM 当成一个“附加模块”。你应该把它改造成：

> **从 uncalibrated simulator 到 target-city-adapted simulator 的机制级 residual adaptation 层。**

当前 MC-WM 已经有三个非常好的资产：symbolic residual、self-hypothesis loop、Bellman trust critic。缺的是三个东西：

```text
机制分解
图局部性
time-varying latent factors
```

补上这三个，它就不是“残差世界模型 + H2O+”，而是一个完整的 **calibration-free causal transfer RL framework**。

[1]: https://github.com/erzhu419/MC-WM/tree/master/md "MC-WM/md at master · erzhu419/MC-WM · GitHub"
[2]: https://github.com/erzhu419/MC-WM/blob/master/mc_wm/residual/world_model.py "MC-WM/mc_wm/residual/world_model.py at master · erzhu419/MC-WM · GitHub"
[3]: https://github.com/erzhu419/MC-WM/blob/master/mc_wm/residual/sindy_nau_adapter.py "MC-WM/mc_wm/residual/sindy_nau_adapter.py at master · erzhu419/MC-WM · GitHub"
[4]: https://arxiv.org/abs/2510.21427?utm_source=chatgpt.com "Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems"
[5]: https://arxiv.org/abs/2203.16582?utm_source=chatgpt.com "Factored Adaptation for Non-Stationary Reinforcement Learning"
[6]: https://github.com/erzhu419/MC-WM/blob/master/mc_wm/policy/qdelta_bellman.py "MC-WM/mc_wm/policy/qdelta_bellman.py at master · erzhu419/MC-WM · GitHub"

我：
因果关系DAG在这个calibration-free causal transfer RL framework 框架下怎么解决
GPT:
直说：**这个框架里的 DAG 不应该靠“完全自动因果发现”解决，也不应该完全手画死。最合理的是：**

> **手工定义高层机制 DAG / candidate parent sets，模型学习 sparse masks 和 domain-change edges。**

也就是：

```text
物理/业务先验给候选图
+
跨城市/跨 simulator 数据学习哪些边真的有用
+
目标城市只估计 latent mechanism factors
```

不要把论文写成“我们从公交轨迹里自动发现完整 DAG”。那会被打爆。更稳、更强的 claim 是：

> 我们用公交运行机理构造一个可验证的 temporal causal template，并在该 template 内学习机制级稀疏 mask 和跨域变化因子。

AdaRL 其实也是这个路子：它不是无先验从零发现任意 DAG，而是先规定 dynamic Bayesian network 的变量类型、时间方向、结构 mask、change factors，然后通过模型估计和稀疏正则学习哪些边/哪些 change factor 有效；论文里明确把结构关系编码成 binary masks，并把 θ 作为 observation、transition、reward 的 domain-specific change factors。

---

## 1. 你的 DAG 应该是 temporal DBN，不是普通静态 DAG

公交系统有循环：车沿线路循环、站点相邻、前后车相互影响。静态图很容易变成有环。但 RL 里真正需要的是：

```text
X_t, A_t, θ_t  →  X_{t+1}, R_{t+1}
```

所以时间展开以后天然是 DAG：

```text
state_t  →  mechanism_t  →  state_{t+1}
action_t →  mechanism_t  →  state_{t+1}
θ_t      →  mechanism_t  →  state_{t+1}
```

线路拓扑可以有环，但因果图按时间展开以后没有环。你不需要 NOTEARS 那种任意 DAG 约束作为主方法；你需要的是 **temporal parent mask**。

---

## 2. DAG 分三层解决

### 第一层：route / topology graph

这个不是因果 DAG，是空间邻接图：

```text
station_i ↔ station_{i+1}
segment_i connects station_i to station_{i+1}
bus_b has front bus and back bus
```

它的作用是限制 local neighborhood：

```text
ego bus 只看前车、后车、当前站、下一站、附近路段
```

这对应 GSAC-style locality。它不是你要学习的因果图，而是 causal graph 的候选边来源。

---

### 第二层：mechanism DAG

这是主图，建议手工给出高层结构：

```text
passenger arrival → waiting passengers
waiting passengers + onboard load + action → dwell time
road condition / speed factor → travel time
holding action + dwell time + travel time → next arrival/departure time
front/back bus state + own arrival/departure → next headway
headway + waiting + delay + load → reward
```

具体到公交 holding：

```text
θ_demand,i,t  → passenger_arrival_i,t → waiting_i,t+1
θ_dwell,i,t   → dwell_i,t
waiting_i,t   → dwell_i,t
load_b,t      → dwell_i,t
holding_t     → departure_time_t
dwell_i,t     → departure_time_t
θ_speed,e,t   → travel_time_e,t
departure_time_t + travel_time_e,t → arrival_time_next
arrival_time_next + front_bus_state + back_bus_state → headway_next
headway_next + waiting_next + delay_next + load_next → reward_next
```

这个图可以手工写成 candidate graph。重点是：**它只定义“可能有边”，不是强行规定每条边一定存在。**

---

### 第三层：domain-change graph

这个是最关键的迁移部分。你要学的是：

```text
θ_demand 影响哪些机制？
θ_speed 影响哪些机制？
θ_dwell 影响哪些机制？
θ_signal 影响哪些机制？
θ_bunching 影响哪些机制？
```

比如：

```text
θ_demand → passenger_arrival module
θ_speed  → travel_time module
θ_dwell  → dwell_time module
θ_signal → travel_time / delay module
θ_bunch  → headway_propagation module
```

这里可以学习 mask：

```text
M_{θ→module}
```

换城市时，不需要手工校准 passenger rate / edge speed，而是从少量目标轨迹估计这些 θ。

---

## 3. DAG 里哪些边手动，哪些边学习？

我会分三类。

### A. 固定边：物理必然成立

这些边不需要学，直接写死：

```text
holding action → departure time
departure time → next arrival time
travel time → next arrival time
arrival time → headway
headway → reward
waiting passengers → dwell time
```

这些是公交系统定义级别的因果关系。学它们反而浪费样本。

---

### B. 候选边：允许模型学习 mask

这些边有可能存在，但强度随城市、线路、时段不同：

```text
waiting passengers → reward
load → dwell time
front headway → action value
back headway → action value
segment density → travel time
time-of-day → demand factor
time-of-day → speed factor
station embedding → demand factor
```

这些边用 soft mask 学：

```text
m_ij ∈ [0,1]
```

训练时加稀疏正则：

```text
prediction loss + λ * |mask|
```

最后得到一个稀疏 causal mask。

---

### C. 禁止边：直接屏蔽

这类边必须禁止，不然模型会学 shortcut：

```text
city_id → action
bus_id → real/sim discriminator
line_id → reward
future state → current state
reward → transition dynamics
target-domain label → policy action
```

尤其是 H2O+/MC-WM 里，discriminator 很容易偷看 city_id、bus_id、line_id、时间分布，然后学到“区分域”的 shortcut，而不是 dynamics gap。这个必须靠 DAG/mask 禁掉。

---

## 4. 你真正要学习的不是完整 DAG，而是四个 mask

我建议定义四组 mask：

```text
M_state→state
M_action→state
M_theta→state
M_state/action/theta→reward
```

对应模块：

```text
x_{t+1}^j = f_j(Pa_j(x_t, a_t), θ_{j,t}) + δ_j(Pa_j, θ_{j,t})
```

其中 `Pa_j` 由 mask 决定。

更具体：

```text
waiting_next    = f_demand(waiting, arrival_proxy, θ_demand)
dwell_time      = f_dwell(waiting, load, holding, θ_dwell)
travel_time     = f_speed(segment_state, θ_speed)
headway_next    = f_headway(front_hw, back_hw, dwell, travel_time, holding, θ_bunch)
reward_next     = f_reward(headway, waiting, delay, load, holding, θ_reward)
```

每个模块只允许看自己的 parents。这样 residual world model 不是一个黑盒：

```text
δ(s,a)
```

而是：

```text
δ_demand, δ_dwell, δ_speed, δ_headway, δ_reward
```

---

## 5. 训练时怎么学 DAG / mask？

流程可以这样：

### Step 1：用交通机理给 candidate parent set

比如：

```text
Pa(waiting_next) = {waiting_now, time, station_context, θ_demand}
Pa(dwell)        = {waiting_now, load_now, holding, station_context, θ_dwell}
Pa(travel_time)  = {segment_context, recent_speed_proxy, θ_speed}
Pa(headway_next) = {front_hw, back_hw, dwell, travel_time, holding, θ_bunch}
Pa(reward)       = {headway_next, waiting_next, load_next, delay_next, holding}
```

这一步是手工的。

### Step 2：每个 module 内学习 soft mask

每个 parent 前面乘一个 mask：

```text
input_j = m_j ⊙ Pa_j
```

然后用：

```text
L_pred + λ_sparse ||m||_1 + λ_group group_lasso(m)
```

学习边是否保留。

### Step 3：跨 source cities / randomized sims 学 invariant modules

你用多个 simulator randomization：

```text
demand randomized
speed randomized
dwell randomized
signal delay randomized
dispatch noise randomized
```

让模型看到不同机制如何变化。共享的函数 `f_j` 跨域不变，变化部分由 `θ_j` 吸收。

### Step 4：domain factor edge 通过 residual variation 学

如果某个机制 residual 在不同 source domain 之间稳定变化，就给它一个 θ：

```text
θ_speed → travel_time residual
θ_demand → waiting residual
θ_dwell → dwell residual
```

如果 θ 对 reward 无路径，就不进入 policy；只用于 world model correction。这一点和 AdaRL 的 minimal representation 思路一致：不是所有 change factors 都应该进 policy，只保留会影响未来 reward 的部分。

---

## 6. simulator randomization 是你的“弱干预”

这是你框架里最关键的识别资源。

真实城市 offline data 很难做干预，但自建 simulator 可以随机化：

```text
客流强度
路段速度
上下客时间
红灯延误
车辆初始间隔
发车扰动
holding policy
```

这等价于给不同机制做 weak intervention。你可以利用它识别：

```text
变 passenger arrival rate，只应该影响 waiting/dwell/reward；
变 road speed，只应该影响 travel_time/arrival/headway/reward；
变 dwell parameter，只应该影响 dwell/departure/headway/reward；
变 holding action，只应该影响 departure/headway/reward，不应该影响 exogenous demand。
```

这比纯 observational causal discovery 稳得多。

所以你的 DAG 识别主张可以是：

> We do not infer an unrestricted causal DAG from observational trajectories. Instead, we exploit simulator-side mechanism randomization as weak interventions to identify sparse mechanism masks within a temporal causal template.

这句话很稳。

---

## 7. 目标城市怎么用 DAG？

目标城市不重新学图。目标城市只做：

```text
freeze candidate DAG / learned masks
freeze shared mechanism functions
infer θ_target,t
optionally fine-tune residual heads with strong regularization
```

也就是：

```text
target trajectory → encoder → θ_demand,t, θ_speed,t, θ_dwell,t, θ_bunch,t
```

然后：

```text
corrected world model = uncalibrated sim + causal residual modules conditioned on θ_target
policy = π(local_state, θ_policy_relevant)
H2O+ weight = factor-wise trust from mechanism residual / QΔ
```

这才是 calibration-free。

---

## 8. 你论文里可以这样定义最终 DAG 方案

我建议命名成：

> **Template-Constrained Causal Mechanism Graph**

或者：

> **Graph-local Factored Dynamic Causal Model**

不要叫 “DAG discovery”。叫 “template-constrained causal structure learning”。

方法描述可以是：

```text
We construct a temporal causal template from bus-operation principles and route topology.
The template defines admissible parent sets for each mechanism module.
Within each parent set, we learn sparse differentiable masks shared across source domains.
Domain-specific and time-varying latent factors are connected to modules through another sparse mask.
The learned graph is used to factorize residual world-model correction, simulator trust estimation, and policy conditioning.
```

这比“自动发现 DAG”靠谱，也比“手工画图”有贡献。

---

## 9. 最推荐的 DAG 版本

你最终应该用这个：

```text
                 θ_demand
                     ↓
time, station → passenger_arrival → waiting_next
                                      ↓
                 θ_dwell             ↓
                     ↓                ↓
waiting, load, holding → dwell_time → departure_time
                                           ↓
                 θ_speed                  ↓
                     ↓                    ↓
segment, traffic_proxy → travel_time → arrival_next
                                           ↓
front/back bus state ----------------→ headway_next
                                           ↓
headway, waiting, load, delay, holding → reward
```

其中：

```text
θ_demand, θ_dwell, θ_speed, θ_bunch
```

是目标城市自动估计的 latent factors；边的存在由候选图约束，边的强度由 mask 学习。

---

## 最后一句

在这个 framework 里，DAG 的解决方案不是：

```text
我用某个算法自动发现公交系统 DAG
```

而是：

```text
我把公交系统拆成可迁移机制，
用时间方向和物理先验确定 candidate DAG，
用多源随机化和目标轨迹学习 sparse masks 与 change-factor edges，
再把这个图用于 residual world model、H2O+ weighting 和 policy adaptation。
```

这条路理论上最稳，也最符合 calibration-free transfer 的目标。


我：
GFlow综述里反复提到的贝叶斯优化可以用到这里么？用一遍遍的数据把相关性塌缩成因果性的思路 https://chatgpt.com/share/6a081ce4-3168-83a4-a4f5-0d9a0d7a9d56 在因果性优化Transformer 这个对话里也提到了这方面的问题

GPT:
可以用，但要分清两件事：

**GFlowNet 解决的是“在很多可能 DAG/机制里维持一个 posterior”；Bayesian optimization 更适合解决“下一批该采什么数据 / 做什么扰动，才能最快缩小这个 posterior”。**

你发的 ChatGPT share 链接我这边只能打开到登录页，看不到对话正文，所以只能按你描述的“相关性逐步塌缩为因果性”这个思想来接。([ChatGPT][1])

## 1. 你的“塌缩”理解对一半

严格说，不是“数据越来越多，相关性自然塌缩成因果性”。应该是：

> **posterior over causal hypotheses 在有信息量的数据下逐步塌缩。**

如果只是同一个城市、同一个策略、同一种 observational data，再多数据也可能只把不确定性压到 Markov equivalence class，不能唯一确定方向。GFlow 综述里也强调，纯 observational data 下 ground-truth 往往只能识别到 MEC；Bayesian approach 的好处不是强行给一个 DAG，而是保留 causal model distribution。

真正让 posterior 从“相关性”往“因果性”塌缩的，是 **干预、扰动、跨环境变化、非平稳机制变化**。综述里明确说，GFlowNet posterior 在有足够合适的 interventional data 时才会在极限下塌缩到真实 causal graph。

所以你的表述要改成：

> 不是一遍遍被动看数据，而是一遍遍主动制造/选择能区分候选因果图的数据。

这正是 Bayesian optimization / Bayesian experimental design 可以进来的位置。

## 2. 在你的框架里，BO 最适合放在“主动机制辨识层”

你现在的目标是 calibration-free causal transfer RL。核心问题不是“调一个最优 simulator 参数”，而是：

```text
哪些机制错了？
哪些机制对 policy 有用？
下一次应该扰动 demand、speed、dwell 还是 signal delay，才能最快识别这些机制？
```

所以我会这样接：

```text
GFlowNet / Bayesian posterior:
    q(G, θ, δ | D)

Bayesian optimization / active design:
    choose next simulator randomization or target-city data query e
    to maximize information gain about G, θ, δ
```

形式上：

```text
e* = argmax_e I((G, θ, δ); D_e | D)
```

其中：

```text
G = causal mechanism graph / mask
θ = time-varying mechanism factors
δ = MC-WM residual mechanisms
D_e = 在实验/扰动 e 下采到的新 transition data
```

这比“用 BO 找最优客流率/速度参数”高级得多。后者只是自动 calibration；前者是 **主动因果结构/机制辨识**。

## 3. GFlowNet 和 BO 在你这里的分工

**GFlowNet 不负责选下一次实验。**
它负责从当前数据中维护一个多峰 posterior：

```text
q(G | D) 或 q(G, θ | D)
```

这个很适合你的 DAG/mask 不确定性，因为 GFlowNet 本来就是为了从组合空间里采样对象，综述里说它可以学习 Bayesian posterior over DAGs，而不是返回一个单点 DAG。

**BO / active design 负责选下一步数据。**
比如现在有几个候选图都能解释历史数据：

```text
G1: demand → waiting → dwell → headway
G2: speed → travel_time → headway
G3: time_of_day 同时影响 demand 和 speed
G4: holding action 只影响 departure，不影响 passenger arrival
```

你下一次不该随机采数据，而应该选最能区分这些图的扰动：

```text
只变 demand，不变 speed
只变 speed，不变 demand
只变 dwell parameter，不变 passenger arrival
只改变 holding policy 的 exploration 强度
```

如果扰动 demand 后 waiting/dwell/reward 变，但 travel_time 不变，那么 posterior 会压低“demand → travel_time”这类边。如果扰动 speed 后 travel_time/headway 变，但 waiting 不变，那么 speed mechanism 被单独识别出来。这就是你说的“相关性塌缩”，但塌缩来自 **独立扰动机制**，不是来自单纯增加样本量。

## 4. 对你的公交场景，最自然的 active variables 是这些

在自建 sim 里，你可以主动扰动：

```text
passenger arrival intensity
passenger burstiness
segment speed / travel time
signal delay
dwell-time coefficient
dispatch interval noise
holding action exploration range
front/back bus initialization
```

每次 BO 选择一个实验配置：

```text
e = {demand_scale, speed_scale, dwell_scale, signal_delay, policy_noise}
```

目标不是让 simulator 更像 SUMO，而是最大化：

```text
posterior entropy reduction over policy-relevant mechanisms
```

也就是说：

```text
选下一批仿真实验，让模型最快知道：
θ_demand 是否只影响 waiting/dwell？
θ_speed 是否只影响 travel_time/headway？
θ_dwell 是否真是 headway instability 的关键？
holding action 的因果路径是否被正确识别？
```

这和 AdaRL 的思想也能对上：AdaRL 关心哪些 change factors 真正影响未来 reward / policy，不是所有环境变化都需要进入 policy；论文里用 compact domain-specific representations θ_min 和 compact shared state 表示 policy 所需的最小信息。

## 5. 最适合你的组合：GFlowNet posterior + BO active simulator design + MC-WM residual

我会把这个模块叫：

> **Active Bayesian Causal Mechanism Discovery**

整体流程：

```text
1. 先给公交机制 template DAG
   demand, speed, dwell, holding, headway, reward

2. GFlowNet 在候选 mask 空间采样图
   G ~ qψ(G | D)

3. 对每个 G 训练 causal-factored MC-WM residual
   δ_demand, δ_speed, δ_dwell, δ_headway, δ_reward

4. 用 BO 选择下一批 simulator randomization
   e* = argmax information gain

5. 更新 posterior q(G, θ, δ | D)

6. 用 posterior-averaged world model / conservative trust 给 H2O+ 供数据
```

这里 GFlowNet 的 reward 可以设计成：

```text
R(G, θ) = exp[-L_pred(G, θ) - λ_sparsity |G| - λ_inv L_invariance - λ_policy L_policy]
```

含义是：图既要能预测 transition，又要稀疏，还要在跨随机化 simulator / 跨城市数据中保持 mechanism invariance，并且最好解释 policy-relevant value error。

## 6. 这个方向比“直接用 BO 做 calibration”强在哪里？

如果你用 BO 去找：

```text
best passenger_rate
best edge_speed
best dwell_time
```

那论文会变成：

> 我们用 BO 自动校准 simulator。

这不够新，而且和你的 calibration-free 目标冲突。

更好的写法是：

> 我们不优化 simulator 参数本身，而是主动选择机制扰动，使 posterior over causal mechanisms 快速收缩；目标城市只需少量轨迹来 infer latent mechanism factors，而不是手工校准 passenger arrival 和 road speed。

这才是 causality + transfer 的贡献。

## 7. GFlowNet 是否值得真加？

如果变量是原始 state 维度，**不值得**，太大。综述也说 DAG 空间 super-exponential，结构学习 NP-complete。

但如果你把图压到机制层，值得：

```text
demand module
speed module
dwell module
holding module
headway module
reward module
```

也就是 5 到 8 个 mechanism nodes，再加局部 graph mask。这个规模下 GFlowNet 有意义，因为它可以保留多种候选机制图，而不是像 NOTEARS/普通 mask learning 一样给一个过早确定的图。

对公交这种 temporal/cyclic 系统，也可以借 DynGFN 的思路：静态上有反馈环，但展开到时间上就是 dynamic causal discovery；综述里 DynGFN 部分也正是处理 time-varying behavior 和 feedback loops。

## 8. 最后判断

**能用，而且如果做得好，是你这个方向里最有理论味的一层。**

但不要说：

```text
我们用 BO 把相关性变成因果性
```

应该说：

```text
我们维护 causal mechanism posterior；
通过 active simulator intervention / Bayesian experimental design 主动采集最有信息量的数据；
posterior 在足够有区分度的干预数据下向 policy-relevant causal mechanisms 收缩；
再用这个 posterior 指导 MC-WM residual correction 和 H2O+ hybrid training。
```

一句话：

> **GFlowNet 给你“不确定的因果图分布”，BO 给你“下一步最该做什么扰动来消除不确定性”。**

这比单纯 DAG discovery 更契合你的 calibration-free transfer RL。

[1]: https://chatgpt.com/share/6a081ce4-3168-83a4-a4f5-0d9a0d7a9d56 "ChatGPT - 因果性优化Transformer"

我：
不一定需要用BO 我只是提出一个设想，不一定对 所以总览一下，我们提到了很多路线，很多边界问题，你大概也知道我想做的是什么了，你看看最合适的框架应该怎么走

GPT：
我会把你现在这条线收束成一个主框架：

# **Calibration-Free Causal-Mechanism Transfer RL**

不要把它定义成“因果发现 + H2O+ + 世界模型 + GFlowNet + BO”这种拼盘。真正主线应该是：

> **用机制级 latent factor 和残差世界模型，替代人工 simulator calibration；再用 H2O+ 式 hybrid RL 训练 policy。**

也就是说，你的核心问题不是“怎么发现完整 DAG”，也不是“怎么调最优 sim 参数”，而是：

> **在目标城市没有手工校准客流、路段速度、dwell 参数的情况下，只用少量目标轨迹，自动估计哪些公交运行机制变了，并把 uncalibrated simulator 修正成 policy 可用的训练源。**

---

## 1. 最合适的总体架构

我建议最后框架长这样：

```text
Route / local graph encoder
        ↓
Causal mechanism template
        ↓
Time-varying latent mechanism inference
        ↓
Causal-factored residual world model
        ↓
Factor-wise simulator/model trust
        ↓
H2O+ hybrid policy training
```

核心公式可以写成：

```text
x_{t+1}^{(m)}
=
f_m(Pa_m(x_t, a_t), θ_{m,t})
+
δ_m(Pa_m(x_t, a_t), θ_{m,t})
```

其中：

```text
m ∈ {demand, dwell, speed, headway, reward}
```

意思是：每个公交运行机制单独建模，单独估计目标城市的 latent factor，单独学习 residual correction，而不是用一个黑盒 world model 预测完整 next state。

这条路比单纯 H2O+ 更强，因为 H2O+ 主要回答“这个 sim transition 能不能信”；你的框架要回答的是：

> **simulator 哪个机制错了，错多少，怎么修正，以及这个修正对 policy 是否重要。**

---

## 2. 方法里的各部分怎么定位

### **AdaRL：提供理论骨架，不直接照搬**

AdaRL 最有用的是两个思想：

```text
1. domain-specific change factors θ
2. policy 只需要 reward-relevant 的最小 θ_min 和 state_min
```

它的目标域适配是：源域学共享结构，目标域只估计低维 change factor，然后 policy 条件化在这些 factor 上。

你这里对应的是：

```text
θ_demand,t
θ_speed,t
θ_dwell,t
θ_signal,t
θ_bunching,t
```

但 AdaRL 偏静态 domain shift，你的公交问题是城市 + 时段 + 局部路况的动态变化，所以不能只用一个 city-level θ。

---

### **FANS-style：作为主轴**

你真正需要的是：

```text
time-varying latent mechanism factors
```

因为公交系统不是“城市 A 一个固定参数，城市 B 一个固定参数”，而是：

```text
早高峰客流变
局部路段速度变
站点 dwell 变
车辆 bunching 状态变
```

所以主轴应该是 FANS-style：

> **每个机制一个 latent factor，且 factor 随时间/局部状态变化。**

例如：

```text
θ_demand,i,t  控制站点 i 的客流机制
θ_speed,e,t   控制路段 e 的运行时间机制
θ_dwell,i,t   控制站点 i 的 dwell 机制
θ_bunch,t     控制头距传播/车队聚集机制
```

目标城市来了之后，不做人工 calibration，而是：

```text
recent target trajectory → encoder → θ_target,t
```

---

### **GSAC-style：作为表示层，不一定要变成 MARL**

你现在是 single agent，这没问题。但公交 holding 的状态天然是 networked system。

所以你不需要马上做 multi-agent，而是做：

```text
single-agent local graph control
```

也就是 ego bus 只看：

```text
前车
后车
当前站
下一站
局部路段
局部等待人数
局部 headway / delay
```

这就是 GSAC-style locality 的价值：不是为了炫图网络，而是为了让跨城市迁移不依赖全局状态维度。

如果以后变多线路、多车、多 agent，这个 local graph encoder 可以直接扩展。

---

### **MC-WM：应该成为核心，不是辅助**

你自己的 MC-WM 最有价值。它本来就是 residual Bellman world model，天然适合解决：

```text
uncalibrated simulator → target-city-adapted simulator
```

但要改造：

从：

```text
M_real = M_sim + δ(s,a)
```

变成：

```text
M_real^m = M_sim^m + δ_m(Pa_m, θ_m)
```

也就是：

```text
δ_demand
δ_speed
δ_dwell
δ_headway
δ_reward
```

每个 residual 对应一个机制。

这一步是整个框架的“实质贡献”。没有它，你只是因果版 H2O+；有它，你是在做 **calibration-free residual mechanism adaptation**。

---

### **H2O+：作为训练 backbone**

H2O+ 不应该被替代。它的角色是：

```text
real offline data
+
uncalibrated sim data
+
corrected world-model rollout
+
factor-wise trust weighting
```

也就是 H2O+ 负责 hybrid RL 训练，但 simulator transition 的权重不再由一个整体 discriminator 给，而是由机制级 trust 给：

```text
w = ψ(w_demand, w_speed, w_dwell, w_headway, w_reward)
```

这样你可以解释：

```text
这个 transition 的 speed mechanism 不可信，但 dwell/headway mechanism 仍然可用
```

这比整体 accept/reject sim transition 更细。

---

## 3. DAG 怎么处理

最合适的做法不是自动发现完整 DAG。

应该是：

> **手工定义高层 causal mechanism template，模型学习 sparse mask 和 θ→mechanism 的变化边。**

也就是：

```text
物理先验给 candidate graph
模型学习哪些边真的有用
目标城市只估计 latent mechanism factors
```

推荐机制图：

```text
θ_demand → passenger arrival → waiting
waiting + load + holding + θ_dwell → dwell
θ_speed / θ_signal → travel time
holding + dwell + travel time → arrival/departure
front/back bus state + own arrival/departure → headway
headway + waiting + delay + load → reward
```

你不要 claim：

```text
we discover the true DAG from bus trajectories
```

应该 claim：

```text
we learn sparse mechanism masks within a template-constrained temporal causal graph
```

这更稳，也更符合交通系统。

---

## 4. GFlowNet 和 BO 放哪里？

**不要放在主线第一版。**

GFlowNet 可以作为后续增强，用来维护：

```text
posterior over causal mechanism graphs / masks
```

GFlowNet 综述里强调，它的优势不是给一个单点 DAG，而是学习 Bayesian posterior over DAGs，并保留多峰不确定性；纯 observational data 下很多时候只能识别到 Markov equivalence class，所以 posterior 比单点图更稳。

但你的第一篇最好不要把 GFlowNet 搬进来。原因：

```text
1. 工程量大
2. 审稿人会转移火力到 causal discovery 是否可靠
3. 你的核心贡献会被稀释
```

BO 也是一样。它可以以后用于 active simulator randomization：

```text
下一轮该扰动 demand 还是 speed，才能最快缩小 mechanism posterior？
```

但它不是主框架必要部件。

---

## 5. 最终框架应该怎么命名

我建议用这类名字：

```text
Causal-Factored Residual H2O+
```

或者：

```text
Calibration-Free Mechanism-Adaptive Hybrid RL
```

再强一点：

```text
CF-MARL / CF-MAHRL
Calibration-Free Mechanism-Adaptive Hybrid Reinforcement Learning
```

但如果你现在还是 single agent，不要在名字里写 MARL。

我更推荐：

> **CF-H2O: Calibration-Free Hybrid RL via Causal-Factored Residual World Models**

这个名字干净，能看出你是在 H2O+ 上做升级。

---

## 6. 最合理的论文贡献点

不要写太散。就写三个贡献：

### **贡献 1：Calibration-free transfer formulation**

传统 H2O+ 仍需要 calibrated low-fidelity simulator。你提出一个更难的问题：

```text
只有 route schema + 少量目标轨迹
没有手工 passenger arrival / speed / dwell calibration
policy 仍然要迁移
```

这是问题定义贡献。

---

### **贡献 2：Causal-factored residual world model**

把 simulator gap 分解成机制级 residual：

```text
demand residual
speed residual
dwell residual
headway residual
reward residual
```

每个 residual 只看 causal parents 和对应 latent factor。

这是方法核心。

---

### **贡献 3：Mechanism-aware H2O+ training**

H2O+ 的 simulator trust 从整体 dynamics ratio 变成 factor-wise mechanism trust：

```text
不是判断整个 sim transition 是否可信
而是判断每个机制是否可信
```

然后用于：

```text
sim transition reweighting
model rollout truncation
critic update weighting
policy training
```

这是 RL 贡献。

---

## 7. 最小可行路线

我会按三步走。

### **第一阶段：不引入 GFlowNet，不引入 BO**

先做：

```text
hand-crafted causal template
+
learned soft masks
+
causal-factored MC-WM residual
+
H2O+ weighting
```

实验目标：

```text
uncalibrated sim 下，明显超过 H2O+
calibrated sim 下，不输 H2O+
跨 city/route randomization 下，比 monolithic residual world model 稳
```

这是最容易成型的一版。

---

### **第二阶段：加入 time-varying θ**

第一阶段可能只是 city-level θ。第二阶段做：

```text
θ_t = encoder(recent local trajectory)
```

这样能处理：

```text
早高峰
晚高峰
局部拥堵
站点突发客流
```

这个阶段就是 FANS-style 的核心。

---

### **第三阶段：再考虑 GFlowNet posterior / active randomization**

等前面站住以后，再加：

```text
GFlowNet over mechanism masks
active simulator randomization
Bayesian model averaging over mechanism graphs
```

这个可以是第二篇或扩展版，不要一开始就塞进去。

---

## 8. 最关键的边界条件

你要非常清楚地写：

```text
我们不是说只知道站点数和线路数就能迁移。
```

更准确是：

```text
我们不需要手工校准客流、速度、dwell 参数；
但需要 route schema、状态变量 schema，以及少量目标城市轨迹。
```

否则会被审稿人抓住不放。

也不要写：

```text
we discover the true causal DAG
```

要写：

```text
we use a template-constrained temporal causal graph and learn sparse mechanism masks
```

---

## 9. 最后给你一个最短版本

最适合你的框架不是 GFlowNet，也不是 BO，也不是纯 AdaRL。

而是：

> **以 H2O+ 为 hybrid RL backbone，以 MC-WM 为 residual world model 核心，以 FANS-style time-varying mechanism factors 解决免校准迁移，以 GSAC-style local graph encoder 解决跨城市局部泛化，以 template-constrained causal DAG 作为机制分解先验。**

一句话：

> **从“校准一个 simulator”转成“自动识别并修正 simulator 的机制级偏差”。**

这就是你这条线最稳、最有贡献、也最贴合你现有代码资产的走法。

我：
我自己认为啊，还是需要自动发现DAG，因为本来这篇的目的就是不去手工calibration H2O+后面接的仿真，为的就是方便很多从业者。现在如果还要他们设置DAG（大部分人完全没概念DAG是什么，跟别说如何设置了，当然这部分LLM可以代劳，但是需要在流程里留出这个功能），我都不知道怎么设置，更别说从业者了
写个完整的开发手册md吧，细到拿给codex开发不会产生歧义的那种，大概遵循的流程是：首先应该是把H2O+里面的相关部分用MC-WM替换，因为两者都收敛了。然后一个模块一个模块的替换，直至最终完全成为Calibration-Free Causal-Mechanism Transfer RL的
Route / local graph encoder
        ↓
Causal mechanism template
        ↓
Time-varying latent mechanism inference
        ↓
Causal-factored residual world model
        ↓
Factor-wise simulator/model trust
        ↓
H2O+ hybrid policy training
当然这里没有自主发现DAG（其实真的可以考虑用LLM来辅助完成，我自己感觉这个牵一发动全身）

GPT：
你这个判断是对的：**默认流程里不能要求从业者手写 DAG**。手工 DAG 最多只能作为 debug / ablation / expert override，主流程必须是自动发现 **temporal mechanism graph posterior**。LLM 可以参与，但只能做 weak prior，不能直接定图；最终图要由数据驱动 mask learning / posterior sampling / stability selection 决定。这个和 GFlowNet 综述里强调的“不是单点 DAG，而是 Bayesian posterior over DAGs”是一致的。

我已经按你说的迁移路线写成完整开发手册 MD 了，里面细化到目录结构、类名、API、CLI、测试、阶段验收标准、LLM-DAG prior JSON schema、自动 DAG 发现流程，以及如何先把 H2O+ 的相关 trust 部分替换成 MC-WM。现有 H2O+ Bus 版本确实已经有 snapshot reset、context discriminator、early truncation 这条路线；MC-WM 也已经有 `M_sim + residual adapter` 的世界模型基础，所以手册按“先桥接，再机制化”的方式写。([GitHub][1])

下载这里：

[CF_H2O_Calibration_Free_Causal_Mechanism_Transfer_RL_Dev_Manual.md](/md/CF_H2O_Calibration_Free_Causal_Mechanism_Transfer_RL_Dev_Manual.md)

我在手册里把主线定成：

```text
H2O+ current discriminator/trust
    → MC-WM monolithic trust
    → route-local representation
    → automatic temporal graph posterior
    → time-varying mechanism factors
    → causal-factored residual world model
    → factor-wise trust
    → full calibration-free CF-H2O
```

关键改动是：**DAG 不再是用户输入，而是系统输出**。用户只需要给 route schema / data / sim env，系统自动生成 `graph_posterior.json`、`edge_report.md`、`mechanism_report.md`。

[1]: https://raw.githubusercontent.com/erzhu419/H2Oplus/master/H2O%2B.md "raw.githubusercontent.com"
