**有，建议做一轮针对性修订后再提交。**我逐页检查了这份 CFCMT_Gemini_V93_TITS.pdf[CFCMT_Gemini_V93_TITS.pdf](sandbox:/workspace/scratch/378ce4a71627/upload/CFCMT_Gemini_V93_TITS.pdf)，也对照了仓库最新提交 `12e7955` 中的代码、实验记录和种子级结果。

先说明一个重要判断：**这版主结果能核对上，不能再沿用上次对旧控制器的判断。**相对 MaxPressure 的平均相对变化 **−2.4200%**、95% 区间 **[−4.5814%, −0.6618%]**、Holm 校正后 **p = 0.005772**，均与当前数据一致。论文也正确说明了：相对 PhasePressure 的区间跨零，不能宣称显著优于它。[对应结果记录](https://github.com/erzhu419/CFCMT/blob/12e7955d0350e4cac2a344d04954cfb5efea70a2/experiments/tsc_v93_max_pressure_20261030/analysis.json)

剩下的问题中，**最重要的是测试种子的历史复用、缺少实际回退策略的对照，以及十二页超出你原定的免费篇幅目标。**

## 一、最需要认真处理：56 个测试种子并非整个开发过程中从未使用过

**位置：第 7 页 V.B，以及第 11 页图 5 的“Frozen confirmation”。**

我把本轮种子名单与此前实验逐一对比，结果是：

| 核对项 | 结果 |
|---|---:|
| 本轮开发种子 | 8 个 |
| 本轮测试种子 | 56 个 |
| 本轮开发集与测试集的交集 | 0 |
| 56 个测试种子中，此前已用于旧控制器实验的种子 | **56 个** |
| 本轮 8＋56 个种子的并集 | **恰好等于此前实验的 64 个种子** |

而且，归档时间显示，旧控制器的 64 种子结果已经产生，随后才完成这轮来源选择和评估。[本轮种子划分](https://github.com/erzhu419/CFCMT/blob/12e7955d0350e4cac2a344d04954cfb5efea70a2/cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json)；[此前 64 种子记录](https://github.com/erzhu419/CFCMT/blob/12e7955d0350e4cac2a344d04954cfb5efea70a2/cf_h2o/results/cluster/tsc_v91_external_source_contribution_20260830/source_contribution_audit_v1.json)

这里必须准确区分两件事：

- **成立的说法**：这 56 个种子没有用于本轮的来源、权重和 guard 选择。
- **目前不能让读者误以为成立的说法**：这 56 个交通随机实现，在整个方法开发过程中从未被使用或观察过。

这**不等于已经证明你拿本轮测试结果直接调参**，也不意味着新结果无效。但是，目前“held-out”“confirmation”的措辞需要交代这个背景。

**具体改法：**在 V.B 明确写出，这些种子与本轮八个选择种子不重叠，但曾在较早的不同控制器实验中使用；因此这是对复用基准场景的评估。若要保留“整个开发历史之外的独立确认”这一更强定位，才需要在完全冻结当前流程后，使用此前未使用过的种子重新验证。

这是比英文润色更重要的一项。

## 二、实验里缺少一个很有判别力的对照：控制器实际使用的回退策略

**位置：第 4 页参考相位定义、第 6 页执行规则、第 8 页表 II。**

正文明确写了，CFCMT 回退到的参考策略是：

\[
\Pi(a)=q_{\mathrm{green}}(a)-0.02\,o_{\mathrm{down}}(a).
\]

但表 II 的 PhasePressure 只最大化：

\[
q_{\mathrm{green}}(a).
\]

**这两个策略并不相同。**

同时，学习动作会被 guard、空间冲突和冷却规则拦截；被拦截时，系统执行的是前面那个广义压力参考策略。因此审稿人很容易问：

> 改善究竟来自学到的相位干预，还是主要来自底层参考策略和限制干预频率的规则？

目前比较结果支持的是**整套系统**的表现，论文对这一点已经比较克制；但它还没有直接回答“学习层给自己的底层参考策略增加了多少价值”。

**最值得补的实验是：**

1. 将这个**完全相同的广义压力参考策略单独运行**，使用相同的相位执行器。
2. 报告学习动作实际执行的比例，以及因 guard、冷却和空间冲突而回退的比例。

这一项直接对应方法贡献。如果还能补，再考虑同容量的目标域训练对照，以及参考预测器、成对预测器和协调器的消融。论文已经承认缺少同容量对照和现代跨城方法的同协议比较，但这些仍然是实质性的审稿风险。

## 三、方法部分有一处明确的逻辑表述错误，以及几处复现细节不足

### 1. 第 6 页公式（10）前的“if and only if”应当改

现在的意思是：只要且只有满足公式（10），学习相位就替换参考相位。

但下一段马上说明，**即便通过公式（10），协调器仍可能因为冷却或空间冲突而拒绝执行。**

因此，通过公式（10）意味着**获得候选资格**，不意味着必然执行。这里应改成：

> “is eligible to replace the reference phase”

随后说明，候选动作还要经过稀疏协调器，最终未获准执行的路口仍采用参考相位。这是一个可以直接修好的文字与执行逻辑不一致问题。

### 2. 目标锚点的“close to unity”不够具体

**位置：第 5–6 页 IV-D。**

洛杉矶最终就是使用这个 target anchor，因此它不是可以略写的次要模型。现在只说目标头权重“接近 1”，源头保留用于处理平局，读者无法准确复现。

需要补出：

- 目标头与源头分别用哪些数据训练；
- 插值系数的确切计算方式；
- 列出的 80 次迭代、7 个叶节点等参数属于哪个头；
- 极小的源头权重如何影响相同或近似相同的目标分数。

当前实现中的权重实际上近似为 \(N_T/(N_T+10^{-12})\)，远比通常理解的“接近 1”更极端。[目标锚点实现](https://github.com/erzhu419/CFCMT/blob/12e7955d0350e4cac2a344d04954cfb5efea70a2/cf_h2o/traffic_signal/action_ranker.py)

### 3. 稀疏协调器还需要几句操作定义

**位置：第 6 页 IV-D 最后一段。**

“在冲突图上贪心选择”还缺三个影响结果的细节：

- 两个路口在什么条件下被视为冲突；
- 候选动作按什么分数排序；
- 分数相同时如何决定先后。

这些在代码里有明确规则，应放进正文或简短伪代码。尤其是这个协调器会直接改变最终执行动作，不能只作为一句实现细节带过。[协调器实现](https://github.com/erzhu419/CFCMT/blob/12e7955d0350e4cac2a344d04954cfb5efea70a2/cf_h2o/traffic_signal/intervention_graph.py)

### 4. “unseen traffic regimes”说得偏强

**位置：第 5 页 IV-C。**

这里的距离项主要衡量**静态路网上下文**的差异，不是直接检测当前交通状态是否超出训练分布。新随机种子在同一目标路网上运行，并不自动形成新的路网上下文。

建议把动机改为“对预测误差和路网上下文差异施加启发式惩罚”。正文已经正确说明它不是校准后的概率区间，这一点应保留。

## 四、格式方面：十二页需要压缩，摘要要去掉缩略语

### 1. 当前是 12 页，不符合你此前的费用目标

T-ITS 目前仍规定 Regular Paper 标准篇幅为 **10 页**，最终超出部分每页 **175 美元**。所以：

- **12 页可以提交**，并未超过绝对投稿上限；
- 但如果最终仍是 12 页，对应 **350 美元超页费**；
- 这不符合你此前“控制在免费页数内”的要求。[IEEE ITSS](https://ieee-itss.org/pub/t-its/?utm_source=chatgpt.com)

这份有足够的内容压缩空间，我建议按以下顺序处理：

1. **合并图 1 和图 2**：两张都在解释离线拟合、选择和在线执行，重复较多。
2. **把图 3 的单一种子 departure-cohort 展示移到补充材料**：它有解释价值，但不是主结论的核心证据。
3. **压缩附录中的三个证明**：推导是正确的，但都比较短，可以更紧凑。
4. 删除引言末尾的整篇章节导航，以及第 3 页方法开头再次逐小节导航的段落。

另外，**图 5、图 6 现在排在结论、数据声明、致谢和附录开头之后**，打断了阅读顺序。应将它们放回结果部分附近。不要通过缩小正文字号或页边距解决页数问题。

### 2. 摘要长度合格，但缩略语不符合官网要求

摘要约 **228 词**，在 150–250 词范围内。但里面有 **TSC、CFCMT、MP、PP**，官网要求摘要不使用缩略语。直接用“the proposed method”“MaxPressure”“PhasePressure”等替换即可。[IEEE ITSS](https://ieee-itss.org/pub/t-its/?utm_source=chatgpt.com)

作者信息现在已经补齐，六个关键词的类别与数量也没有发现问题。

## 五、参考文献还需要一次正式出版信息清理

**位置：第 12 页。**

这一版有九条引用仍采用 arXiv 记录：**[4]、[5]、[8]、[10]、[11]、[12]、[14]、[17]、[18]**。其中不少已经有正式出版版本。

三个尤其明显的例子：

| 条目 | 当前写法 | 应采用的正式记录 |
|---|---|---|
| **[8] X-Light** | arXiv 2024 | **IJCAI 2024，94–102 页**。[IJCAI](https://www.ijcai.org/proceedings/2024/11?utm_source=chatgpt.com) |
| **[14] scikit-learn** | arXiv 2012 | **JMLR，12(85):2825–2830，2011**；作者名单也应对应正式版本。[jmlr.org](https://www.jmlr.org/papers/v12/pedregosa11a.html?utm_source=chatgpt.com) |
| **[17] LibSignal** | arXiv 2022 | **Machine Learning，113:5235–5271，2024**。[Springer Nature Link](https://link.springer.com/article/10.1007/s10994-023-06412-y?utm_source=chatgpt.com) |

FRAP 应更新为 CIKM 2019，Bengio 的 meta-transfer 论文应更新为 ICLR 2020，Factored Adaptation 应更新为 NeurIPS 2022。[Penn State](https://pure.psu.edu/en/publications/learning-phase-competition-for-traffic-signal-control/?utm_source=chatgpt.com)

引用预印本本身不是违规，但最终投稿稿件里大量保留已过时的预印本信息，会显得文献整理没有完成。

## 六、还应补报已有的碰撞与 teleport 结果

**位置：第 7 页指标说明、第 8 页主结果表。**

方法用碰撞和 teleport 数量筛选 guard，却没有在最终结果中报告这些指标。当前归档已经记录：

| 控制器 | 汇总碰撞事件数 | Starting / ending teleports |
|---|---:|---:|
| CFCMT | 13 | 0 / 0 |
| MaxPressure | 18 | 0 / 0 |

这些是模拟器中的事件计数，不应解释成已经证明安全优势。[对应汇总记录](https://github.com/erzhu419/CFCMT/blob/12e7955d0350e4cac2a344d04954cfb5efea70a2/experiments/tsc_v93_max_pressure_20261030/analysis.json)

建议给六个控制器统一补一张小表，或者在主表增加相应列。**这不需要新实验，只需要把已有结果报告完整。**

## 七、语言上仍有几处可以直接改掉

整体已经能读，但以下地方仍带有内部版本语言或过强表述：

| 位置 | 当前表达 | 建议 |
|---|---|---|
| 图 1，第 3 页 | `V93: ...` | 删除版本号 |
| 图 2，第 4 页 | `CFCMT V93 ...` | 删除版本号 |
| 第 3 页方法开头 | `selects the optimal source mixture` | 改为根据开发运行结果选择配置；当前分阶段搜索不保证全局最优 |
| 第 2 页贡献列表 | `Comprehensive Closed-Loop Validation` | 改为 `Paired Closed-Loop Evaluation`，更符合两个外部路网的范围 |
| 第 5 页 | 排除静态几何是为了 `prevent overfitting` | 改为 `limit dependence on source-specific layouts` |
| 第 11 页图 6 | `Risk = 0.5` | 改为 `Guard multiplier β = 0.5`，避免被理解为风险概率 |

还有一句典型的绕口表达：

> A key distinction … lies in the timing and nature of target-domain information utilization.

可以直接改成：

> Transfer methods differ in how much target data they require and when those data are collected.

致谢中的 AI 辅助说明已经具体列出了系统、涉及部分和用途，不需要为了减少“AI 味”而删除。

---

**我的提交建议是：先完成种子历史披露、公式（10）的执行条件修正、复现细节补充、参考文献更新和十页压缩。**这些大多可以基于现有材料完成。

如果还能补一项实验，优先补**“实际广义压力回退策略单独运行”及学习动作执行率**。它最直接关系到审稿人会如何判断这套学习方法的贡献。当前主结果数值有依据，但测试独立性的表述和学习层贡献的证据，仍需要认真处理。