# CFCMT 论文手术式审计记录

审计对象：`paper_tsc/main.tex` 及其纳入的章节、图表、附录和参考文献。审计后的论文保留既有负结果：压力控制仍是闭环上限，V159 关闭最后一条端点校准路线；本轮不再继续算法调参。

## 1. 编号、交叉引用与附录闭合

- 57 个标签均有正文引用，无重复标签、无未定义引用、无孤立标签。
- 16 个编号公式连续为 (1)--(16)，每个公式都有唯一标签且在正文中被解释或调用。
- 4 幅图、17 张表均有唯一编号和正文引用；原有的硬编码 “Figure 1--3” 与错误的表图混合引用已改为可自动更新的交叉引用。
- 3 个命题、1 个假设、1 个推论均采用定理环境，正文引用其编号，证明紧随相应陈述。
- 附录 A.1--A.12 均有标签并被正文明确指向。网络清单、开发历史、城市结果、来源诊断、V98、来源门槛、补救路线、指标与安全、废弃 guard、冻结后压力测试、原生 RL 和制品链不再是与正文割裂的材料。

## 2. 数学符号一义性

| 符号 | 唯一涵义 |
|---|---|
| $c,n,g$ | 城市组、网络、匹配动作组 |
| $s_g,\mathcal A_g,a,r_g$ | 完整状态、可行动作集、候选动作、参考动作 |
| $\tau,H,L_n$ | 一秒采样索引、60 个样本的视界、网络受控车道数 |
| $Y_g(a)$ | 候选动作的反事实成本 |
| $\eta_g,\sigma_g,z_g(a)$ | 数值下限、组内尺度、归一化参考动作对比 |
| $b_g,q_g(a),e_g(a)$ | 动作共同项、可迁移动作项、剩余动作误差 |
| $\theta,\psi$ | 刚性锚点参数、成对模型参数 |
| $A_g(a),\widetilde P_g(a),P_g(a),S_g(a)$ | 锚点评分、未平移成对评分、参考平移成对评分、部署评分 |
| $\widehat\delta_{gij},K_g,\overline z_g$ | 成对预测差、动作数、组内平均归一化对比 |
| $u_{A,g}(a),u_{P,g}(a)$ | 锚点与成对模型的动作级不确定度 |
| $\alpha_g,\kappa,\zeta$ | 混合权重、局部扰动上限、置信乘数 |
| $h_g,D_g,M_g,U_g$ | 局部尺度、分歧、偏好边际、不确定度和 |
| $m,k,W^\pi_{c,m,k},\rho_k$ | 需求场景、随机种子、策略等待时间、CFCMT 相对 legacy comparator 的种子级效应 |
| $p_{\mathrm{sign}}$ | 单侧精确符号检验的概率值 |

原来相互冲突的用途已经拆开：种子使用 $k$，不再与状态混用；场景使用 $m$，不再与队列项混用；局部上限使用 $\kappa$，不再与索引混用；正则化写为 $\ell_2$，不再与车道数 $L_n$ 混用。正文现在明确说明每个算子、索引、集合、尺度和不确定度。

## 3. 简称与专名

首次出现处现已给出全称：CFCMT、SUMO、RL、H2O、H2O$^+$、FRAP、GESA、UGAT、OOF、MPC、RESCO、MC-WM、B100、DQN、IDQN、PCG64、CSV、SHA-256、JSON、DOI，以及附录中的 B20/B5。后文只使用简称。

PressLight、MPLight、MetaLight、X-Light、CrossLight、CoLight、CityFlow、MaxPressure 和 PhasePressure 是论文或项目采用的专名，不被臆造为不存在的缩写展开。V 前缀在引言中统一定义为冻结实验协议的溯源编号，而不是方法名。

## 4. 文献与引用语义

参考文献共 19 条，19 条均在正文出现；无缺失条目、无未引用条目。引用位置按其实际支持内容重新核对：

- Varaiya 仅支持 MaxPressure 在其 store-and-forward 假设下的吞吐与稳定性结论；正文不再把该保证扩展到项目的 PhasePressure。
- PressLight/MPLight 支持压力结构，FRAP 支持相位竞争；不再互换这两类机制的文献。
- H2O/H2O$^+$ 支持真实系统与模拟器动力学缺口下的混合离线/在线学习，不再被写成跨城市源数据迁移证据。
- MetaLight、GESA、X-Light、CrossLight 与 UGAT 分别放在元适应、异构结构统一、跨城市轨迹表征、离线到在线跨城市学习和 sim-to-real 动作变换的对应陈述后。
- Bengio、Schölkopf 与 Factored Adaptation 仅用于支持模块化因果机制和因子化适应的动机；正文明确限制了本文“causal”的含义。
- Los Angeles 文件来源归于 LibSignal；CityFlow 只支持其 roadnet/flow 表示。Jinan 文件来源归于 CoLight。SUMO 与 RESCO 分别支持模拟器和基准网络说明。

按“存在 arXiv 就使用 arXiv”的要求，以下 11 条采用 arXiv 版本：H2O (2206.13464)、H2O$^+$ (2309.12716)、Bengio 等 (1901.10912)、Factored Adaptation (2203.16582)、Towards Causal Representation Learning (2102.11107)、X-Light (2404.12090)、UGAT (2307.12388)、FRAP (1905.04722)、LibSignal (2211.10649)、CoLight (1905.05717) 和 CityFlow (1905.05217)。

对其余 8 条按精确题名核验未找到对应 arXiv 版本，因此保留正式出版信息：SUMO、MaxPressure、RESCO、MetaLight、GESA、CrossLight、PressLight 和 Thousand Lights。H2O$^+$ 的标题已与当前 arXiv 版本一致；Schölkopf、Flötteröd 和 Lücken 的变音符号已修正。

## 5. 构建与验证结果

- LaTeX/BibTeX 完整构建成功，输出 31 页 PDF。
- 构建日志无未定义引用、未定义文献、重复标签、overfull/underfull box 或过大浮动体警告。
- 参考文献闭合：19 个正文 citation key 对应 19 个 BibTeX 条目。
- 交叉引用闭合：57 个唯一标签对应 57 个被引用标签。
- 来源补救路线生成器测试通过：`1 passed`。
- 逐页检查了公式页、四幅主图、附录长表、制品链和参考文献页；未见截断、重叠或错误编号。

## 6. 已知投稿事项

数据与代码可用性段落仍明确写着公共仓库 DOI 尚未生成。论文内容和引用审计已经闭合，但正式投稿前仍需将仓库固化到可引用版本，并用 DOI 替换该占位说明。
