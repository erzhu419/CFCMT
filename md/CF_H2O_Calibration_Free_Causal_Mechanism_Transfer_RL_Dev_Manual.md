# CF-H2O 开发手册：Calibration-Free Causal-Mechanism Transfer RL

版本：v0.1  
目标读者：Codex / 开发者 / 论文原型实现者  
当前依赖资产：`erzhu419/H2Oplus`、`erzhu419/MC-WM`  
核心目标：把现有 H2O+ Bus Sim-to-Real 框架逐步替换为一个不依赖手工校准客流、路段速度、dwell 参数的 **Calibration-Free Causal-Mechanism Transfer RL** 框架。

---

## 0. 一句话目标

现有 H2O+ Bus 版本依赖一个已经做过较强工程标定的简化仿真器。CF-H2O 的目标是：

> 给定 route/schema、少量目标域轨迹、一个未充分校准的简化仿真器，系统自动发现/估计关键动态因果机制，用机制级 residual world model 修正仿真器，并用 H2O+ 式 hybrid RL 训练可迁移 policy。

最终框架：

```text
Route / local graph encoder
        ↓
Automatic temporal DAG / mechanism graph discovery
        ↓
Time-varying latent mechanism inference
        ↓
Causal-factored residual world model
        ↓
Factor-wise simulator/model trust
        ↓
H2O+ hybrid policy training
```

注意：这里的 “DAG 自动发现” 不等于从 observational data 中保证恢复真实因果图。工程实现必须输出 **posterior / soft graph / edge probabilities**，而不是强制输出唯一真图。训练时默认使用 graph posterior averaging 或 soft masks。

---

## 1. 设计原则

### 1.1 必须坚持的原则

1. **不要求从业者手写 DAG。** 运行主流程时，用户只需要提供数据路径、route schema、仿真环境入口。DAG 由自动模块生成。
2. **允许 LLM 辅助，但 LLM 不作为真值来源。** LLM 只能生成 prior / candidate grouping / edge hint；最终图由数据驱动的 mask learner、posterior sampler、稳定性选择决定。
3. **先替换 H2O+ 里最相关的 sim-trust 部分，再逐步机制化。** 第一阶段不要一次性改成最终框架，否则无法定位问题。
4. **先 monolithic MC-WM，后 causal-factored MC-WM。** 因为 MC-WM 和 H2O+ 都已收敛，第一步应做稳定桥接。
5. **禁止 source/domain shortcut。** city_id、line_id、bus_id、real/sim label 不能直接进入 policy，也不能让 discriminator/trust 模型偷看标签。
6. **训练路径必须兼容 single agent。** 即使将来可扩展到 MARL，目前默认是 single-agent local graph control。
7. **所有模块都必须可 ablation。** 每个替换阶段都要能单独开关。

### 1.2 不做的事情

第一版不要做：

- 不做完整 unrestricted DAG discovery over raw state dimensions。
- 不要求用户手写 passenger arrival rate / edge speed / dwell coefficient。
- 不把 LLM 生成的图直接当作最终图。
- 不把 graph posterior 的不确定性压成一个无解释的单点图。
- 不一开始就接 GFlowNet 和 BO。GFlowNet 可作为 posterior sampler 的后续可选增强。

---

## 2. 当前代码资产与迁移定位

### 2.1 H2Oplus 当前可复用部分

保留：

- `SimpleSAC/h2oplus_bus.py`
  - SAC/H2O+ 混合训练骨架
  - offline real + online/sim batch 混合
  - blended TD target
  - automatic entropy tuning
  - Q/V/policy optimizer
- `bus_h2o/common/data_utils.py`
  - snapshot → structured context `z`
  - route length / edge linear map 工具
  - current discriminator / dynamics-ratio 相关工具
- `bus_h2o/sim_core/*`
  - 简化 bus simulator
- `bus_h2o/sumo_env/*`
  - SUMO bridge / snapshot conversion
- replay buffer / offline data collection 协议

逐步替换：

- `ZOnlyDiscriminator`
- `compute_z_importance_weight`
- single scalar dynamics ratio `w`
- 强依赖人工校准的 simulator 参数

### 2.2 MC-WM 当前可复用部分

保留并迁移：

- `mc_wm/residual/world_model.py`
  - `WorldModelEnsemble`
  - `ResidualAdapter`
  - ensemble disagreement / uncertainty
- `mc_wm/policy/qdelta_bellman.py`，如果存在
  - QΔ-Bellman trust critic
- `mc_wm/residual/sindy_nau_adapter.py`，如果存在
  - symbolic / sparse residual adapter

逐步改造：

- 从 monolithic residual：

```text
M_real(s,a) = M_sim(s,a) + δ(s,a)
```

改为 causal-factored residual：

```text
M_real^m(Pa_m, θ_m) = M_sim^m(Pa_m, θ_m) + δ_m(Pa_m, θ_m)
```

其中：

```text
m ∈ {demand, speed, dwell, headway, reward, optional_signal}
```

---

## 3. 新代码包结构

在 `H2Oplus/` 根目录下新建包：

```text
H2Oplus/
├── cf_h2o/
│   ├── __init__.py
│   ├── schemas.py
│   ├── registry.py
│   ├── config/
│   │   ├── default.yaml
│   │   ├── stage0_mcwm_h2o.yaml
│   │   ├── stage1_route_graph.yaml
│   │   ├── stage2_auto_dag.yaml
│   │   ├── stage3_latent_factors.yaml
│   │   └── stage4_full_cf_h2o.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── h2o_buffer_adapter.py
│   │   ├── transition_dataset.py
│   │   ├── snapshot_features.py
│   │   └── normalization.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── route_graph.py
│   │   ├── local_neighborhood.py
│   │   ├── feature_registry.py
│   │   ├── mechanism_discovery.py
│   │   ├── auto_dag.py
│   │   ├── llm_prior.py
│   │   ├── graph_posterior.py
│   │   └── validation.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── local_graph_encoder.py
│   │   ├── temporal_encoder.py
│   │   └── mechanism_encoder.py
│   ├── latent/
│   │   ├── __init__.py
│   │   ├── factor_encoder.py
│   │   ├── factor_prior.py
│   │   └── factor_regularizers.py
│   ├── world_model/
│   │   ├── __init__.py
│   │   ├── mcwm_adapter.py
│   │   ├── monolithic_residual.py
│   │   ├── mechanism_modules.py
│   │   ├── causal_factored_residual.py
│   │   └── rollout.py
│   ├── trust/
│   │   ├── __init__.py
│   │   ├── factor_trust.py
│   │   ├── qdelta_adapter.py
│   │   └── weight_composer.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── h2o_mcwm_bridge.py
│   │   ├── cf_h2o_trainer.py
│   │   └── policy_inputs.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── train_stage0.py
│   │   ├── train_stage1.py
│   │   ├── discover_dag.py
│   │   ├── train_full.py
│   │   └── evaluate_transfer.py
│   └── tests/
│       ├── test_schemas.py
│       ├── test_no_future_leakage.py
│       ├── test_route_graph.py
│       ├── test_auto_dag_synthetic.py
│       ├── test_mcwm_bridge_shapes.py
│       ├── test_factor_trust_shapes.py
│       └── test_full_smoke.py
└── docs/
    ├── CF_H2O_DEV_MANUAL.md
    └── CF_H2O_USER_GUIDE.md
```

Codex 要求：除非本手册明确要求，不要大规模重构 `SimpleSAC/h2oplus_bus.py` 和 `bus_h2o/*` 原文件。先通过 adapter 接入。

---

## 4. 数据结构定义

文件：`cf_h2o/schemas.py`

### 4.1 TransitionBatch

```python
from dataclasses import dataclass
from typing import Any, Optional
import torch

@dataclass
class TransitionBatch:
    observations: torch.Tensor        # [B, obs_dim]
    actions: torch.Tensor             # [B, act_dim]
    rewards: torch.Tensor             # [B] or [B, 1]
    next_observations: torch.Tensor   # [B, obs_dim]
    dones: torch.Tensor               # [B] or [B, 1]
    z_t: Optional[torch.Tensor] = None     # [B, z_dim]
    z_t1: Optional[torch.Tensor] = None    # [B, z_dim]
    domain_id: Optional[torch.Tensor] = None   # [B], int64; source city/sim/randomization id
    line_id: Optional[list[str]] = None
    route_id: Optional[list[str]] = None
    global_time: Optional[torch.Tensor] = None # [B]
    snapshot_t: Optional[list[dict[str, Any]]] = None
    snapshot_t1: Optional[list[dict[str, Any]]] = None
    source: Optional[list[str]] = None  # each item in {"real", "sim", "model"}
```

### 4.2 FeatureTensor

```python
@dataclass
class FeatureTensor:
    names: list[str]
    values: torch.Tensor          # [B, D]
    groups: dict[str, list[int]]  # e.g. {"headway": [0,1], "demand": [2,3]}
    metadata: dict[str, Any]
```

### 4.3 GraphSpec

```python
@dataclass
class GraphSpec:
    node_names: list[str]
    adjacency: torch.Tensor          # [N, N], float in [0,1], edge i -> j
    hard_mask: torch.Tensor          # [N, N], bool; False means edge forbidden
    edge_probs: torch.Tensor         # [N, N], posterior probability
    node_groups: dict[str, list[int]]
    graph_type: str                  # "temporal_mechanism_graph"
    version: str
```

### 4.4 GraphPosterior

```python
@dataclass
class GraphPosterior:
    graphs: list[GraphSpec]
    log_weights: torch.Tensor        # [K]
    edge_marginals: torch.Tensor     # [N, N]
    diagnostics: dict[str, Any]
```

### 4.5 MechanismSpec

```python
@dataclass
class MechanismSpec:
    name: str                         # demand/speed/dwell/headway/reward
    child_names: list[str]
    parent_names: list[str]
    latent_dim: int
    output_dim: int
    loss_type: str                    # gaussian_nll, mse, poisson_nll, bce
```

### 4.6 MechanismBatch

```python
@dataclass
class MechanismBatch:
    mechanism_name: str
    parents: torch.Tensor       # [B, P_m]
    child_target: torch.Tensor  # [B, O_m]
    actions: torch.Tensor       # [B, act_dim]
    theta: Optional[torch.Tensor] = None  # [B, latent_dim]
    mask: Optional[torch.Tensor] = None   # [P_m]
```

---

## 5. 迁移阶段总览

必须按阶段开发，不允许直接上最终版。

```text
Stage 0: H2O+ ↔ MC-WM bridge
Stage 1: Route / local graph encoder
Stage 2: Automatic temporal DAG / mechanism graph discovery
Stage 3: Time-varying latent mechanism inference
Stage 4: Causal-factored residual world model
Stage 5: Factor-wise simulator/model trust
Stage 6: Full CF-H2O hybrid RL training
Stage 7: LLM-assisted zero-DAG user workflow
Stage 8: Experiments, ablations, reports
```

---

# Stage 0：用 MC-WM 替换 H2O+ 的 sim trust 相关部分

## 6. Stage 0 目标

先不做因果图，不做机制分解。目标是把 H2O+ 中的 real/sim mixed training 与 MC-WM 的 world model / residual correction 接起来。

当前 H2O+ 使用：

```text
real_batch + sim_batch
sim transition weight = compute_z_importance_weight(D, z_t, z_t1)
```

Stage 0 替换为：

```text
real_batch + sim_batch + optional model_rollout_batch
sim/model weight = MC-WM uncertainty or QΔ trust
```

## 6.1 新建文件

`cf_h2o/world_model/mcwm_adapter.py`

### 6.1.1 MCWMAdapter API

```python
class MCWMAdapter:
    def __init__(self, obs_dim: int, act_dim: int, config: dict, device: str):
        ...

    def fit_sim_model(self, sim_dataset: TransitionBatch) -> dict:
        """Train M_sim on sim transitions."""
        ...

    def fit_residual(self, paired_or_aligned_dataset: TransitionBatch) -> dict:
        """Train residual adapter. Stage 0 allows approximate aligned pairs.
        If true paired sim/real data are unavailable, use nearest-neighbor alignment
        in (obs, action, local context) space and record alignment quality.
        """
        ...

    def predict(self, observations: torch.Tensor, actions: torch.Tensor, deterministic: bool=False) -> dict:
        """Return next_observations, rewards, uncertainty."""
        return {
            "next_observations": next_obs,   # [B, obs_dim]
            "rewards": rewards,             # [B]
            "epistemic": epistemic,         # [B]
            "aleatoric": aleatoric,         # [B] or None
        }

    def rollout(self, init_observations: torch.Tensor, policy, horizon: int) -> TransitionBatch:
        ...

    def trust_weight(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return [B] weights in [w_min, w_max]. Default: decreasing function of ensemble disagreement."""
        ...
```

## 6.2 新建 H2O bridge

文件：`cf_h2o/rl/h2o_mcwm_bridge.py`

### 6.2.1 目的

不要直接改 `H2OPlusBus.train()` 的大量逻辑。创建一个 wrapper/trainer class，内部持有原 `H2OPlusBus`，但覆盖 sim weight 来源。

```python
class H2OMCWMBridge:
    def __init__(self, h2o_algo, mcwm_adapter, replay_buffer, config):
        self.h2o = h2o_algo
        self.mcwm = mcwm_adapter
        self.replay_buffer = replay_buffer
        self.config = config

    def train_step(self, batch_size: int, pretrain_steps: int = 0) -> dict:
        """Call H2O+ train but replace discriminator-based sim weights.
        Implementation option A: subclass H2OPlusBus and override _compute_sim_weight.
        Implementation option B: monkey-patch only inside bridge. Prefer A if clean.
        """
        ...
```

### 6.2.2 必须抽象出的 H2O+ hook

在 `SimpleSAC/h2oplus_bus.py` 中添加最小 hook：

```python
class H2OPlusBus:
    ...
    def compute_sim_weight(self, sim_batch: dict) -> torch.Tensor:
        """Default implementation uses old discriminator weight.
        Subclasses/bridges may override.
        Returns [B] tensor on config.device.
        """
        ...
```

要求：

- 不改变原有默认行为。
- 如果没有 external weight provider，仍使用原 discriminator。
- 所有 sim Q loss 中使用的 dynamics ratio 都改为调用 `self.compute_sim_weight(sim_batch)`。

## 6.3 Stage 0 acceptance tests

新建：`cf_h2o/tests/test_mcwm_bridge_shapes.py`

测试内容：

1. `MCWMAdapter.predict()` 输入 `[B, obs_dim]`, `[B, act_dim]`，输出 shape 正确。
2. `trust_weight()` 输出 `[B]`，无 NaN，范围在 `[0.05, 5.0]`。
3. `H2OPlusBus.compute_sim_weight()` 默认旧逻辑不变。
4. Bridge 替换后，训练 3 step 不报错。
5. 不要求性能提升，只要求 loss finite。

---

# Stage 1：Route / local graph encoder

## 7. Stage 1 目标

当前 `z_t` 是整条 route 的 30 维 fingerprint。Stage 1 增加 local graph representation，但不删除 `z_t`。

核心思想：single agent 仍然只控制 ego bus，但输入应是 local causal neighborhood：

```text
ego bus
front bus
back bus
current station
next station
current segment
next segment
local waiting passengers
local headway/delay
```

## 7.1 新建 RouteGraph

文件：`cf_h2o/graph/route_graph.py`

```python
class RouteGraph:
    def __init__(self, stations: list[dict], segments: list[dict], lines: list[str]):
        ...

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "RouteGraph":
        ...

    def get_station_neighbors(self, station_id: str, radius: int = 1) -> list[str]:
        ...

    def get_segment_neighbors(self, segment_id: str, radius: int = 1) -> list[str]:
        ...

    def to_pyg_data(self) -> object:
        """Optional torch_geometric Data. If torch_geometric unavailable, return dict."""
        ...
```

## 7.2 新建 LocalNeighborhoodExtractor

文件：`cf_h2o/graph/local_neighborhood.py`

```python
class LocalNeighborhoodExtractor:
    def __init__(self, max_front_back: int = 1, station_radius: int = 1, segment_radius: int = 1):
        ...

    def extract(self, snapshot: dict, ego_bus_id: str | None = None) -> dict:
        """Return local neighborhood dict with deterministic keys.
        If ego_bus_id is None, use snapshot['ego_bus_id'].
        """
        return {
            "ego": {...},
            "front_buses": [...],
            "back_buses": [...],
            "current_station": {...},
            "next_station": {...},
            "segments": [...],
            "global_context": {...},
        }
```

## 7.3 新建 LocalGraphEncoder

文件：`cf_h2o/encoders/local_graph_encoder.py`

```python
class LocalGraphEncoder(torch.nn.Module):
    def __init__(self, input_dims: dict, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        ...

    def forward(self, local_graph_batch: dict) -> torch.Tensor:
        """Return local embedding h_local [B, out_dim]."""
        ...
```

第一版不强制使用 GNN，可以用 deterministic feature flatten + MLP。后续再替换成 GNN。

## 7.4 Stage 1 acceptance tests

1. 同一个 snapshot 多次提取 local neighborhood，输出字段顺序一致。
2. ego/front/back 不存在时用 zero padding + mask。
3. 不允许用 future snapshot_t1 的字段构造 t 时刻 policy input。
4. `LocalGraphEncoder` 输出 `[B, out_dim]`。

---

# Stage 2：Automatic temporal DAG / mechanism graph discovery

## 8. Stage 2 目标

用户不需要手动写 DAG。系统自动输出：

```text
graph_posterior.json
edge_report.md
discovered_mechanisms.yaml
```

### 8.1 重要边界

自动 DAG discovery 不能保证真实因果图。实现必须满足：

- 输出 edge probability / posterior，不输出 “truth”。
- 默认用 temporal constraints 保证无未来泄漏。
- 允许 LLM 给 prior，但最终 edge score 由数据决定。
- 对低置信边保留 uncertainty，不硬删。

## 8.2 DAG 发现的三层结构

### 层 1：Temporal variable graph

节点形式：

```text
feature_name@t
feature_name@t1
reward@t1
action@t
latent_theta_m@t
```

允许边：

```text
feature@t  -> feature@t1
action@t   -> feature@t1
feature@t  -> reward@t1
action@t   -> reward@t1
theta@t    -> feature@t1
theta@t    -> reward@t1
```

禁止边：

```text
feature@t1 -> feature@t
reward@t1  -> feature@t
source_label -> policy action
real_sim_label -> policy action
city_id -> policy action
```

### 层 2：Mechanism graph

系统自动把 features 聚合成 mechanism candidates：

```text
demand
speed
signal_delay
dwell
headway
load
reward
```

如果无法识别公交语义，则 fallback 为 generic modules：

```text
state_group_0
state_group_1
...
reward
```

### 层 3：Graph posterior

最终训练不使用单图，而使用：

```text
edge_marginals: p(edge_i_j | data)
K sampled graphs: G_1 ... G_K
soft parent masks: m_ij ∈ [0,1]
```

## 8.3 FeatureRegistry

文件：`cf_h2o/graph/feature_registry.py`

```python
class FeatureRegistry:
    def __init__(self):
        self.features = []

    def register(self, name: str, group: str, time_index: str, source: str, dim: int = 1):
        ...

    @classmethod
    def from_transition_dataset(cls, dataset: TransitionBatch, obs_names: list[str] | None = None) -> "FeatureRegistry":
        """If obs_names absent, auto-generate obs_0...obs_n.
        Then infer groups using names and snapshot metadata.
        """
        ...

    def infer_groups(self) -> dict[str, list[str]]:
        """Rule-based grouping plus optional LLM hints."""
        ...
```

### 8.3.1 Group inference rules

Use deterministic rules first:

```python
KEYWORD_GROUPS = {
    "headway": ["headway", "hw", "forward", "backward", "gap"],
    "demand": ["waiting", "arrival_rate", "passenger", "queue"],
    "dwell": ["dwell", "boarding", "alighting", "stop_time"],
    "speed": ["speed", "travel_time", "runtime", "segment"],
    "load": ["load", "onboard", "capacity"],
    "reward": ["reward"],
    "action": ["action", "holding", "hold"],
}
```

If no keyword matches, assign to `unknown_state`.

## 8.4 LLM prior provider

文件：`cf_h2o/graph/llm_prior.py`

### 8.4.1 目的

LLM 只生成 prior，不直接决定 DAG。

### 8.4.2 API

```python
class LLMDAGPriorProvider:
    def __init__(self, backend: str = "none", model: str | None = None):
        ...

    def build_prompt(self, registry: FeatureRegistry, route_schema: dict | None) -> str:
        ...

    def propose_prior(self, registry: FeatureRegistry, route_schema: dict | None = None) -> dict:
        """Return JSON-compatible prior.
        If backend == 'none', return empty neutral prior.
        """
        return {
            "edge_priors": [
                {"src": "waiting@t", "dst": "dwell@t1", "log_prior": 1.0, "reason": "..."},
                {"src": "reward@t1", "dst": "waiting@t", "log_prior": -999.0, "reason": "future leakage"},
            ],
            "group_priors": {...},
            "warnings": [...]
        }
```

### 8.4.3 LLM 输出 JSON schema

```json
{
  "edge_priors": [
    {
      "src": "string",
      "dst": "string",
      "log_prior": 0.0,
      "confidence": 0.0,
      "reason": "string"
    }
  ],
  "forbidden_edges": [
    {"src": "string", "dst": "string", "reason": "string"}
  ],
  "suggested_groups": {
    "group_name": ["feature_a", "feature_b"]
  },
  "warnings": ["string"]
}
```

### 8.4.4 LLM prompt template

```text
You are assisting a causal mechanism discovery module for bus holding control.
You are NOT allowed to decide the final causal graph.
Return only JSON.
Given the feature list and route schema, propose weak priors over temporal causal edges.
Use these rules:
1. Never allow future-to-past edges.
2. Never allow reward@t1 to cause state@t.
3. Never allow city_id/source_label/real_sim_label to directly cause policy action.
4. Prefer temporal edges from state@t/action@t to state@t1/reward@t1.
5. Provide log_prior in [-3, 3], except forbidden edges use -999.
6. If uncertain, use log_prior=0.
```

## 8.5 Masked parent learner

文件：`cf_h2o/graph/mechanism_discovery.py`

### 8.5.1 目标

对每个 child variable 学一个 parent mask。

```python
class MaskedParentLearner(torch.nn.Module):
    def __init__(self, parent_dim: int, child_dim: int, hidden_dim: int, mask_type: str = "concrete"):
        ...

    def forward(self, parents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prediction and mask probabilities."""
        ...

    def loss(self, pred, target, mask_probs, prior_logits=None) -> torch.Tensor:
        """NLL/MSE + sparsity + prior regularization."""
        ...
```

Loss：

```text
L = L_pred
  + λ_sparse * ||mask||_1
  + λ_prior * BCE(mask_logits, prior_edge_probs)
  + λ_stability * L_domain_stability
  + λ_no_shortcut * L_shortcut_penalty
```

### 8.5.2 Domain stability loss

如果同一机制在多个 randomized source domains 中存在，要求 shared mechanism residual 稳定，domain factor 解释变化。

```text
L_domain_stability = variance_domain( residual_error_after_theta_conditioning )
```

第一版可实现为：每个 domain 的 child prediction residual 均值差异最小化。

## 8.6 AutoDAGDiscoverer

文件：`cf_h2o/graph/auto_dag.py`

```python
class AutoDAGDiscoverer:
    def __init__(self, config: dict):
        ...

    def fit(self, dataset: TransitionBatch, registry: FeatureRegistry, route_schema: dict | None = None) -> GraphPosterior:
        """Main entry. No manual DAG input required."""
        # 1. build temporal nodes
        # 2. build hard masks from time direction and leakage rules
        # 3. get optional LLM priors
        # 4. train masked parent learners per child
        # 5. bootstrap stability selection
        # 6. build graph posterior
        ...

    def save(self, posterior: GraphPosterior, out_dir: str):
        ...
```

## 8.7 Graph posterior output files

`graph_posterior.json`:

```json
{
  "version": "0.1",
  "node_names": ["waiting@t", "action@t", "dwell@t1", "reward@t1"],
  "edge_marginals": [[0.0, 0.1], [0.0, 0.0]],
  "hard_mask": [[false, true], [false, false]],
  "graphs": [
    {"weight": 0.37, "edges": [["waiting@t", "dwell@t1"], ["dwell@t1", "reward@t1"]]}
  ],
  "diagnostics": {
    "bootstrap_runs": 20,
    "mean_validation_nll": 0.123,
    "low_confidence_edges": []
  }
}
```

`edge_report.md` must include:

- high-confidence edges `p > 0.8`
- uncertain edges `0.3 <= p <= 0.8`
- rejected edges `p < 0.3`
- possible shortcut warnings
- features with no stable parent

## 8.8 Stage 2 CLI

文件：`cf_h2o/cli/discover_dag.py`

命令：

```bash
python -m cf_h2o.cli.discover_dag \
  --offline-buffer path/to/offline.pkl \
  --sim-buffer path/to/sim.pkl \
  --route-schema path/to/route_schema.json \
  --out-dir runs/dag_discovery/exp001 \
  --llm-backend none \
  --bootstrap 20 \
  --max-parents 8
```

默认 `--llm-backend none`。如果未来接 LLM，只能影响 prior。

## 8.9 Stage 2 acceptance tests

### Synthetic DAG test

创建 synthetic bus mechanism 数据：

```text
θ_demand -> waiting_next
waiting + load + holding -> dwell
θ_speed -> travel_time
holding + dwell + travel_time -> headway_next
headway_next + waiting_next -> reward
```

测试：

- temporal forbidden edges 概率必须 < 0.05。
- true direct edges AUC > 0.75。
- bootstrap edge probabilities shape 正确。
- 没有用户 DAG 输入时可完整跑完。

---

# Stage 3：Time-varying latent mechanism inference

## 9. Stage 3 目标

用少量目标城市轨迹自动推断机制因子：

```text
θ_demand,t
θ_speed,t
θ_dwell,t
θ_headway,t
θ_reward,t
```

这些 θ 替代人工 calibration 参数。

## 9.1 FactorEncoder

文件：`cf_h2o/latent/factor_encoder.py`

```python
class TimeVaryingFactorEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, mechanism_names: list[str], latent_dims: dict[str, int], hidden_dim: int = 128):
        super().__init__()
        ...

    def forward(self, history_features: torch.Tensor, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """history_features: [B, T, D]
        Return {mechanism_name: theta [B, latent_dim]}.
        """
        ...
```

### 9.1.1 History window

默认：

```yaml
latent:
  history_len: 8
  mechanism_latent_dims:
    demand: 2
    speed: 2
    dwell: 2
    headway: 2
    reward: 1
```

## 9.2 Factor regularization

文件：`cf_h2o/latent/factor_regularizers.py`

必须实现：

```python
def temporal_smoothness_loss(theta_seq):
    return ((theta_seq[:, 1:] - theta_seq[:, :-1]) ** 2).mean()

def mechanism_independence_loss(theta_dict):
    """Penalize correlation among mechanism factors."""
    ...

def domain_contrast_loss(theta_dict, domain_id):
    """Optional: factors should explain domain differences without collapsing."""
    ...
```

## 9.3 Stage 3 acceptance tests

1. `TimeVaryingFactorEncoder` 输出 dict keys 与 mechanism_names 一致。
2. `θ` 不得直接包含 domain_id / real-sim label。
3. history_len 不足时使用 padding mask。
4. temporal smoothness loss finite。

---

# Stage 4：Causal-factored residual world model

## 10. Stage 4 目标

把 MC-WM 从整体 residual world model 改成机制级 residual world model。

## 10.1 Mechanism modules

文件：`cf_h2o/world_model/mechanism_modules.py`

```python
class MechanismModule(torch.nn.Module):
    def __init__(self, parent_dim: int, theta_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = ...

    def forward(self, parents: torch.Tensor, theta: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> dict:
        """Return mean/logvar or deterministic output."""
        ...
```

## 10.2 CausalFactoredResidualWorldModel

文件：`cf_h2o/world_model/causal_factored_residual.py`

```python
class CausalFactoredResidualWorldModel(torch.nn.Module):
    def __init__(self, mechanism_specs: list[MechanismSpec], graph_posterior: GraphPosterior, config: dict):
        super().__init__()
        ...

    def fit_sim_modules(self, sim_data: TransitionBatch) -> dict:
        """Train base mechanism modules on uncalibrated sim data."""
        ...

    def fit_residual_modules(self, real_data: TransitionBatch, sim_data: TransitionBatch | None = None) -> dict:
        """Train residual modules. If paired sim-real data unavailable, use distributional/aligned loss."""
        ...

    def predict(self, batch: TransitionBatch, theta_dict: dict[str, torch.Tensor], graph_sample: GraphSpec | None = None) -> dict:
        return {
            "next_observations": next_obs,
            "reward": reward,
            "mechanism_outputs": {...},
            "mechanism_uncertainty": {...},
        }

    def rollout(self, init_batch: TransitionBatch, policy, horizon: int, theta_encoder) -> TransitionBatch:
        ...
```

## 10.3 Residual training modes

### Mode A：paired residual

如果有 paired sim-real：

```text
target_residual = real_next - sim_next
```

### Mode B：aligned residual

如果没有严格 paired data：

1. 用 nearest neighbor 在 `(local_state, action, z_t)` 中对齐 sim 和 real。
2. 记录 alignment distance。
3. 只对 alignment distance 小于阈值的样本训练 residual。
4. 低质量对齐样本只用于 distributional loss。

### Mode C：distributional residual

用 MMD / adversarial / energy distance 约束机制输出分布接近真实。

第一版必须实现 A 和 B；C 可留接口。

## 10.4 Graph posterior 使用方式

默认不要采单图，而是 soft mask：

```text
parent_input_m = edge_marginals_m ⊙ parent_features_m
```

可选：sample K graphs 并 average prediction：

```text
p(x_next | x,a,D) ≈ Σ_k w_k p(x_next | x,a,G_k)
```

## 10.5 Stage 4 acceptance tests

1. monolithic MC-WM 和 causal-factored model 在同一 batch 上输出 shape 相同。
2. 每个 mechanism module 的 parent input 不包含 forbidden features。
3. graph posterior soft mask 可正常广播。
4. residual loss 下降，至少在 synthetic data 上能恢复已知 residual。

---

# Stage 5：Factor-wise simulator/model trust

## 11. Stage 5 目标

替换 H2O+ 单一 discriminator ratio：

```text
w = D(real | z_t,z_t1) / D(sim | z_t,z_t1)
```

为机制级 trust：

```text
w_m = trust_m(parent_m, child_m, θ_m, uncertainty_m)
w_total = compose(w_demand, w_speed, w_dwell, w_headway, w_reward)
```

## 11.1 FactorTrustEstimator

文件：`cf_h2o/trust/factor_trust.py`

```python
class FactorTrustEstimator(torch.nn.Module):
    def __init__(self, mechanism_names: list[str], config: dict):
        super().__init__()
        ...

    def forward(self, mechanism_outputs: dict, real_targets: dict | None = None) -> dict[str, torch.Tensor]:
        """Return per-mechanism trust weights [B]."""
        ...
```

Trust 来源：

- ensemble disagreement
- residual magnitude
- QΔ-Bellman confidence
- graph posterior uncertainty
- alignment quality

## 11.2 WeightComposer

文件：`cf_h2o/trust/weight_composer.py`

```python
class WeightComposer:
    def __init__(self, mode: str = "geometric_mean", w_min: float = 0.05, w_max: float = 5.0):
        ...

    def compose(self, trust_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if mode == "min":
            return torch.stack(list(trust_dict.values()), dim=0).min(dim=0).values
        if mode == "geometric_mean":
            ...
        if mode == "reward_path_weighted":
            ...
```

默认使用 `geometric_mean`，避免单机制误差过度支配。

## 11.3 H2O+ hook 替换

`H2OPlusBus.compute_sim_weight()` 最终调用：

```python
sim_weight = cf_h2o_trust.compute_weight(sim_batch)
```

要求：

- pretrain 阶段不用 sim weight。
- trust warmup 之前返回全 1。
- weight detach 后进入 Q loss，不反向更新 trust 模块，除非显式设置 `joint_train_trust=true`。

## 11.4 Stage 5 acceptance tests

1. 每个机制 trust shape `[B]`。
2. `w_total` finite，范围 `[w_min, w_max]`。
3. graph uncertainty 高时，trust 下降。
4. model rollout horizon 越长，trust 不应无约束上升。

---

# Stage 6：Full CF-H2O hybrid policy training

## 12. Stage 6 目标

整合：

```text
real offline batch
sim batch
corrected model rollout batch
factor-wise trust
H2O+ SAC training
```

## 12.1 CFH2OTrainer

文件：`cf_h2o/rl/cf_h2o_trainer.py`

```python
class CFH2OTrainer:
    def __init__(self, h2o_algo, world_model, factor_encoder, graph_posterior, trust_estimator, replay_buffer, config):
        ...

    def pretrain_policy_on_real(self, steps: int):
        ...

    def train_world_model(self):
        ...

    def train_step(self) -> dict:
        """One full update step:
        1. sample real batch
        2. sample sim batch
        3. optionally generate model rollout batch
        4. infer θ
        5. compute factor-wise trust
        6. call H2O+ update with weights
        """
        ...
```

## 12.2 Training schedule

默认：

```yaml
training:
  pretrain_real_steps: 5000
  train_sim_model_epochs: 100
  train_residual_epochs: 50
  main_train_steps: 200000
  world_model_update_interval: 1000
  factor_encoder_update_interval: 500
  model_rollout_start_step: 20000
  model_rollout_horizon: 1
  max_model_rollout_horizon: 5
  rollout_horizon_schedule: linear
```

## 12.3 Policy input

默认 policy 输入：

```text
policy_input = concat(
    normalized_obs,
    local_graph_embedding,
    selected_theta_policy_relevant
)
```

严禁输入：

```text
real_sim_label
city_id
raw domain_id
future snapshot
reward@t1
```

## 12.4 Stage 6 acceptance tests

1. full trainer smoke test 10 steps。
2. policy input 不包含 forbidden fields。
3. model rollout batch 进入 replay buffer 时 `source="model"`。
4. H2O+ 原有 metrics 仍输出。
5. 新增 metrics：
   - `theta_norm/demand`
   - `trust/speed_mean`
   - `graph/entropy`
   - `world_model/residual_loss_dwell`

---

# Stage 7：LLM-assisted zero-DAG user workflow

## 13. Stage 7 目标

从业者不懂 DAG，也不想写 DAG。用户流程必须是：

```bash
python -m cf_h2o.cli.train_full \
  --real-data data/real.pkl \
  --sim-env bus_h2o.sim_core.sim:MultiLineEnv \
  --route-schema data/route_schema.json \
  --out-dir runs/cf_h2o_exp001
```

系统自动：

1. 读取数据 schema。
2. 构造 feature registry。
3. 可选调用 LLM 生成 weak prior。
4. 自动发现 temporal DAG / mechanism graph posterior。
5. 训练 latent factor encoder。
6. 训练 causal-factored MC-WM。
7. 训练 CF-H2O policy。
8. 输出 report。

## 13.1 User-facing route_schema.json

用户只需要提供事实结构，不需要提供因果边：

```json
{
  "city": "example_city",
  "lines": [
    {
      "line_id": "7X",
      "stations": [
        {"station_id": "S1", "position_m": 0.0},
        {"station_id": "S2", "position_m": 500.0}
      ],
      "segments": [
        {"segment_id": "E1", "from_station": "S1", "to_station": "S2", "length_m": 500.0}
      ]
    }
  ],
  "action": {
    "name": "holding_time",
    "unit": "seconds",
    "min": 0.0,
    "max": 120.0
  }
}
```

## 13.2 Auto-generated reports

每次训练必须输出：

```text
runs/exp/
├── graph_posterior.json
├── edge_report.md
├── mechanism_report.md
├── trust_report.md
├── training_curves.json
├── config_resolved.yaml
└── warnings.md
```

`warnings.md` 包含：

- 低置信 DAG 区域
- 可能无法识别的边
- possible shortcut
- route schema 缺失字段
- target data 太少的提醒

---

# Stage 8：实验与 ablation

## 14. 必须实现的实验组

### 14.1 Baselines

```text
B0: 原 H2O+ + calibrated sim
B1: 原 H2O+ + uncalibrated sim
B2: H2O+ + monolithic MC-WM residual
B3: H2O+ + route/local graph encoder + monolithic MC-WM
B4: CF-H2O without automatic DAG，使用 fully-connected soft mask
B5: CF-H2O with automatic DAG posterior
B6: CF-H2O with LLM prior + automatic DAG posterior
B7: Oracle calibrated sim / target-trained upper bound
```

### 14.2 关键问题

要回答：

1. 不手工校准 sim 时，CF-H2O 是否明显优于 H2O+？
2. causal-factored residual 是否优于 monolithic residual？
3. automatic DAG posterior 是否优于 fully-connected mask？
4. LLM prior 是否只是加速/稳定，而不是决定最终图？
5. 少量目标数据下，time-varying θ 是否能替代 passenger/speed/dwell calibration？

### 14.3 指标

RL 指标：

```text
average return
headway variance
average passenger waiting time
holding time total
bunching frequency
on-time performance
```

模型指标：

```text
one-step transition NLL
multi-step rollout error
reward prediction error
mechanism residual error
trust calibration curve
```

图发现指标：

```text
edge stability
graph posterior entropy
synthetic DAG AUC
edge calibration under known randomized interventions
```

---

## 15. 配置文件规范

文件：`cf_h2o/config/stage4_full_cf_h2o.yaml`

```yaml
seed: 42
device: cuda

paths:
  real_buffer: null
  sim_buffer: null
  route_schema: null
  out_dir: runs/cf_h2o/default

h2o:
  batch_size: 2048
  batch_sim_ratio: 0.5
  discount: 0.80
  pretrain_steps: 5000

mcwm:
  ensemble_size: 5
  hidden_dim: 200
  lr: 0.001
  residual_hidden_dim: 64
  residual_lr: 0.001
  use_qdelta: false

route_graph:
  max_front_back: 1
  station_radius: 1
  segment_radius: 1
  encoder_dim: 128

auto_dag:
  enabled: true
  llm_backend: none
  bootstrap_runs: 20
  max_parents: 8
  lambda_sparse: 0.001
  lambda_prior: 0.1
  lambda_stability: 0.1
  min_edge_prob_for_hard: 0.8
  use_soft_masks: true

latent:
  enabled: true
  history_len: 8
  hidden_dim: 128
  mechanism_latent_dims:
    demand: 2
    speed: 2
    dwell: 2
    headway: 2
    reward: 1
  lambda_smooth: 0.01
  lambda_independence: 0.01

world_model:
  mode: causal_factored
  train_epochs_sim: 100
  train_epochs_residual: 50
  rollout_horizon: 1
  max_rollout_horizon: 5

trust:
  mode: geometric_mean
  w_min: 0.05
  w_max: 5.0
  warmup_steps: 5000
  use_graph_uncertainty: true

policy_input:
  use_raw_obs: true
  use_local_graph_embedding: true
  use_theta: true
  forbidden_fields:
    - city_id
    - domain_id
    - real_sim_label
    - source
```

---

## 16. CLI 总入口

文件：`cf_h2o/cli/train_full.py`

```python
def main():
    # 1. parse args
    # 2. load config and override by CLI
    # 3. load real/sim buffers
    # 4. build FeatureRegistry
    # 5. run AutoDAGDiscoverer if graph_posterior not supplied
    # 6. build RouteGraph / LocalGraphEncoder
    # 7. build FactorEncoder
    # 8. build CausalFactoredResidualWorldModel
    # 9. build TrustEstimator
    # 10. build H2OPlusBus and CFH2OTrainer
    # 11. train
    # 12. evaluate
    # 13. save reports/checkpoints
```

命令：

```bash
python -m cf_h2o.cli.train_full \
  --config cf_h2o/config/stage4_full_cf_h2o.yaml \
  --real-buffer data/real_buffer.pkl \
  --sim-buffer data/sim_buffer.pkl \
  --route-schema data/route_schema.json \
  --out-dir runs/cf_h2o_exp001
```

---

## 17. 关键实现细节与禁止事项

### 17.1 禁止 future leakage

任何训练 policy、factor encoder、DAG discovery 的 `t` 时刻输入不得包含：

```text
next_obs
reward
snapshot_t1
z_t1
future station waiting
future vehicle positions
```

例外：world model / DAG discovery 训练 target 可以使用 next variables，但必须标记为 target，不可进入 policy input。

### 17.2 禁止 domain shortcut

以下字段不得作为 policy input：

```text
source
real_sim_label
domain_id
city_id
raw line_id embedding if it reveals target/source split
```

line/route 信息只能通过 route graph topology 和 normalized local features 表示。

### 17.3 自动 DAG 的硬约束

无论 LLM 或数据如何，以下边永远 forbidden：

```text
reward@t1 -> any@t
next_obs@t1 -> obs@t
done@t1 -> action@t
real_sim_label -> action@t
city_id -> action@t
```

### 17.4 用户没有 route_schema 时

允许 fallback：

```text
flat temporal graph + obs_i groups + no route graph
```

但必须在 `warnings.md` 写明：route schema missing，local transfer ability degraded。

---

## 18. 开发顺序任务清单

### Sprint 1：Stage 0 bridge

- [ ] 新建 `cf_h2o/schemas.py`
- [ ] 新建 `cf_h2o/data/h2o_buffer_adapter.py`
- [ ] 新建 `cf_h2o/world_model/mcwm_adapter.py`
- [ ] 在 `H2OPlusBus` 加 `compute_sim_weight()` hook
- [ ] 新建 `cf_h2o/rl/h2o_mcwm_bridge.py`
- [ ] 通过 `test_mcwm_bridge_shapes.py`

### Sprint 2：local graph

- [ ] 新建 `route_graph.py`
- [ ] 新建 `local_neighborhood.py`
- [ ] 新建 `local_graph_encoder.py`
- [ ] 接入 policy input，但默认可关闭
- [ ] 通过 no future leakage tests

### Sprint 3：auto DAG

- [ ] 新建 `feature_registry.py`
- [ ] 新建 `llm_prior.py`，默认 backend none
- [ ] 新建 `mechanism_discovery.py`
- [ ] 新建 `auto_dag.py`
- [ ] 新建 `graph_posterior.py`
- [ ] 生成 `edge_report.md`
- [ ] 通过 synthetic DAG test

### Sprint 4：latent factors

- [ ] 新建 `factor_encoder.py`
- [ ] 新建 `factor_regularizers.py`
- [ ] 接入 world model input
- [ ] 输出 theta metrics

### Sprint 5：causal-factored world model

- [ ] 新建 `mechanism_modules.py`
- [ ] 新建 `causal_factored_residual.py`
- [ ] 实现 paired/aligned residual training
- [ ] 接入 graph posterior soft masks

### Sprint 6：factor trust + full trainer

- [ ] 新建 `factor_trust.py`
- [ ] 新建 `weight_composer.py`
- [ ] 新建 `cf_h2o_trainer.py`
- [ ] 新建 `train_full.py`
- [ ] 跑 smoke + ablation

---

## 19. 最小可运行路径

第一个可运行版本只需完成：

```text
Stage 0 + Stage 2 的 simplified AutoDAG + Stage 5 的 trust composer
```

不需要完整 time-varying θ，也不需要 GFlowNet。

最小训练路径：

```bash
python -m cf_h2o.cli.train_stage0 \
  --real-buffer data/real.pkl \
  --sim-buffer data/sim.pkl \
  --out-dir runs/stage0

python -m cf_h2o.cli.discover_dag \
  --real-buffer data/real.pkl \
  --sim-buffer data/sim.pkl \
  --out-dir runs/dag

python -m cf_h2o.cli.train_full \
  --config cf_h2o/config/stage4_full_cf_h2o.yaml \
  --graph-posterior runs/dag/graph_posterior.json \
  --out-dir runs/full
```

---

## 20. 论文叙事建议

### 20.1 不要这样写

```text
We automatically discover the true causal DAG of bus systems.
```

### 20.2 应该这样写

```text
We remove manual simulator calibration by learning a posterior over temporal mechanism graphs and by adapting an uncalibrated simulator through causal-factored residual world models.
```

### 20.3 贡献点

1. **Calibration-free hybrid RL formulation**：不需要手工校准客流、速度、dwell 参数。
2. **Automatic temporal mechanism graph discovery**：从数据和弱先验自动生成 graph posterior，不要求从业者写 DAG。
3. **Causal-factored residual world model**：把 simulator gap 分解到机制层，而不是整体黑盒 residual。
4. **Factor-wise trust H2O+**：按机制估计 sim/model trust，用于 hybrid Bellman update。

---

## 21. 风险与防线

### 21.1 自动 DAG 不可识别

防线：输出 posterior，不输出单点真图；用 synthetic known-DAG 验证可恢复性；真实实验只 claim performance 和 stability。

### 21.2 LLM prior 被质疑

防线：LLM prior 可关闭；LLM 只影响 prior；报告 ablation：without LLM / with LLM。

### 21.3 residual model 被认为只是自动 calibration

防线：强调不是估计 passenger rate / edge speed，而是机制级 residual + factor-wise trust + policy-relevant adaptation。

### 21.4 工程复杂度过高

防线：所有模块可关闭；Stage 0 是可发表/可复现的基础版本；最终版逐步增强。

---

## 22. Codex 开发约束

Codex 执行本手册时必须遵守：

1. 每次只实现一个 Sprint。
2. 每个 Sprint 必须新增或更新 tests。
3. 不删除原 H2Oplus 功能。
4. 不把 LLM API key 写入代码。
5. 不把 `domain_id`、`source`、`real_sim_label` 输入 policy。
6. 不让 DAG discovery 使用 future variables 作为 parent。
7. 所有输出文件必须写入 `out_dir`。
8. 所有随机过程必须接收 seed。
9. 所有 tensor shape 必须在 docstring 写清楚。
10. 所有 ablation 开关必须进 yaml config。

---

## 23. 最终检查清单

项目完成时，应能做到：

- [ ] 用户不提供 DAG，系统自动生成 graph posterior。
- [ ] 用户不提供 passenger arrival rate / edge speed / dwell coefficient，系统仍可训练。
- [ ] 可在 uncalibrated sim 上超过原 H2O+。
- [ ] 可输出机制级 residual 和 trust report。
- [ ] 可关闭 LLM prior 并仍能运行。
- [ ] 可关闭 auto DAG 改用 fully-connected soft mask 做 ablation。
- [ ] 可关闭 causal-factored residual 改用 monolithic MC-WM 做 ablation。
- [ ] 可复现实验曲线和 edge stability report。

---

## 24. 附录：推荐默认机制变量

若系统能识别公交字段，默认 mechanisms：

```yaml
mechanisms:
  demand:
    children: [waiting_next, passenger_arrival_proxy]
    candidate_parents: [waiting, time_of_day, station_context, local_history]
    latent_dim: 2
    loss_type: poisson_nll

  speed:
    children: [travel_time_next, moving_fraction_next]
    candidate_parents: [segment_context, current_speed, density, time_of_day]
    latent_dim: 2
    loss_type: gaussian_nll

  dwell:
    children: [dwell_time_next]
    candidate_parents: [waiting, load, action, station_context]
    latent_dim: 2
    loss_type: gaussian_nll

  headway:
    children: [forward_headway_next, backward_headway_next, gap_next]
    candidate_parents: [forward_headway, backward_headway, dwell_time, travel_time, action]
    latent_dim: 2
    loss_type: gaussian_nll

  reward:
    children: [reward_next]
    candidate_parents: [headway_next, waiting_next, delay_next, load_next, action]
    latent_dim: 1
    loss_type: gaussian_nll
```

这些只是自动初始化，不是用户手写 DAG。系统仍需通过 data-driven mask learning 生成最终 graph posterior。

---

## 25. 附录：最小 graph discovery 伪代码

```python
def discover_graph(real_batch, sim_batch, route_schema=None, llm_backend="none"):
    dataset = merge_batches(real_batch, sim_batch)
    registry = FeatureRegistry.from_transition_dataset(dataset)

    hard_mask = build_temporal_hard_mask(registry)
    route_prior = build_route_locality_prior(registry, route_schema)
    llm_prior = LLMDAGPriorProvider(llm_backend).propose_prior(registry, route_schema)

    prior_logits = combine_priors(route_prior, llm_prior, hard_mask)

    edge_probs_accum = []
    graphs = []
    for b in range(num_bootstrap):
        boot_dataset = bootstrap(dataset)
        learners = []
        for child in registry.child_nodes_t1_and_reward():
            parents = registry.allowed_parents(child, hard_mask)
            learner = MaskedParentLearner(len(parents), child.dim)
            train_masked_parent_learner(learner, boot_dataset, parents, child, prior_logits)
            learners.append(learner)
        edge_probs = collect_edge_probs(learners)
        edge_probs_accum.append(edge_probs)
        graphs.append(sample_graph_from_edge_probs(edge_probs, hard_mask))

    edge_marginals = mean(edge_probs_accum)
    posterior = GraphPosterior(graphs=graphs, edge_marginals=edge_marginals, ...)
    return posterior
```

---

## 26. 附录：Stage 0 H2O+ hook 修改示例

原逻辑中找到 sim Q loss 权重位置。不要改 loss 结构，只改权重来源：

```python
# before
sim_ratio = compute_z_importance_weight(self.discriminator, sim_z_t, sim_z_t1)

# after
sim_ratio = self.compute_sim_weight(sim_batch)
```

默认实现：

```python
def compute_sim_weight(self, sim_batch):
    if getattr(self, "external_sim_weight_provider", None) is not None:
        return self.external_sim_weight_provider(sim_batch)
    return compute_z_importance_weight(self.discriminator, sim_batch["z_t"], sim_batch["z_t1"])
```

这样旧实验不受影响，新框架可以接入 MC-WM / factor-wise trust。

---

## 27. 附录：graph report 示例

```markdown
# Edge Report

## High-confidence edges

| Edge | Probability | Type | Reason |
|---|---:|---|---|
| waiting@t -> dwell@t1 | 0.91 | learned | stable across bootstrap |
| action@t -> headway@t1 | 0.87 | learned | improves validation NLL |

## Uncertain edges

| Edge | Probability | Warning |
|---|---:|---|
| speed@t -> dwell@t1 | 0.42 | may be confounded by time_of_day |

## Rejected / forbidden edges

| Edge | Probability | Reason |
|---|---:|---|
| reward@t1 -> action@t | 0.00 | future leakage forbidden |
| city_id -> action@t | 0.00 | shortcut forbidden |
```

---

## 28. 结论

开发主线是：

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

最终卖点不是“我们手工写了一个公交 DAG”，而是：

> 用户不写 DAG、不校准客流/速度/dwell。系统从 route schema 和少量轨迹中自动生成机制图 posterior，学习目标城市 latent mechanism factors，并用机制级 residual world model 修正 uncalibrated simulator，最后通过 H2O+ hybrid training 训练策略。
