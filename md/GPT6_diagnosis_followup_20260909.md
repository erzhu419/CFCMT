**2026-09-09：对 Boston v18 与 V150–V153 的后续分析**

本文保留最初的修复前诊断。随后执行的代码纠正、V153 v2 七城结果和
Cologne 服务停滞重放，见[当前进展](/home/erzhu419/mine_code/CFCMT/paper/tsc_v150_gpt6_remediation_progress.md)
及[Cologne 机制诊断](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154_cologne_rigid_service_diagnostic.md)。

首轮代码修复结果：发现并修复 rigid 的 29 个输入中 9 个错列。原模型、原种子的
Cologne 3600 秒重跑，平均等待从 1484.36 秒降到 19.71 秒，到达车辆从 210
增至 1997，待入网从 1606 降为零；仍有 18 起碰撞事件，尚未优于
PhasePressure。见[列映射修复结果](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154_rigid_feature_alignment_correction.md)。
同时修复嵌套验证和列映射后的 [V153 v3](/home/erzhu419/mine_code/CFCMT/paper/tsc_v153a_target_calibrated_hierarchical_prior_v3_result.md)
最终为 0/7 准入，全部回退到修正后的 rigid；中间 v2 的 Ingolstadt 阳性已被取代。
进一步的[四城复测](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154b_feature_aligned_rigid_smoke_result.md)
已经完成。RESCO 等待下降 51.7%、系统车辆小时下降 24.7%；少到达的 3 辆仅占
需求 0.2%，按用户接受的取舍判断为明显改善。Atlanta、New York 的等待分别下降
2.4%、3.8%，仍有严重积压。此次四城只纠正列绑定，未混入新的占用率公式。

[Cologne 首碰撞重放](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154_cologne_first_collision_diagnostic_result.md)
精确复现了原 23 条动作和 2 条碰撞报告：左转车已经停住，但车身侵入直行车辆
通道 5.65 厘米。侵入在绿灯时已经形成，下一相位尚未开启；具体问题是内部
等待位置与车身几何不兼容。冲突和让行关系正确，继续增加全红时长无法消除
这对已经进入路口的车辆之间的几何冲突。

占用率修复已接入实际压力特征、完整解析先验、cache v21/v22 和 V150K/V150L
模型及输入合同。新运行路径拒收旧单位缓存与旧模型；固定小规模新数据试验
及其适用范围见[单位合同说明](/home/erzhu419/mine_code/CFCMT/paper/tsc_occupancy_fraction_contract.md)。
独立的等待点净空修正包已完成同一窗口的配对验证，见
[V154C 结果](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154c_cologne_waiting_geometry_repair_result.md)。
原网精确复现；修正侧另一辆接近静止的左转车实测净空为正 3.565 厘米，
支持这处局部几何修正。但未修改的对侧等待点仍有 1 起碰撞，原直行焦点车
也未在固定窗口内驶出路口，整体判定保留 FAIL。

随后完成的 [V154D 双侧修正](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154d_cologne_bilateral_waiting_geometry_result.md)
按同一规则把对侧等待段从 8.62 米缩至 8.37 米。原网再次精确复现；修正网
466 秒内碰撞为零、无 teleport，对侧两辆焦点车均实际通过。对侧同一辆左转车
接近静止时，车身净空从侵入 4.963 厘米变为正 3.637 厘米。整体仍保留 FAIL，
唯一未满足条件是原直行焦点车未在窗口结束前通过；最后观测时它已在绿灯前
以 5.575 m/s 接近路口，并非静止。当前窗口到达数为 241，原网为 256，
尚无服务改善结论。

随后完成的 [V154E 原时长验证](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154e_cologne_bilateral_full_validation_result.md)
已经通过：保留 V154D 几何不变，3600 秒内碰撞从 18 起降至 **0 起**，无
teleport，四辆焦点车全部通过。原网全部 77 个指标、175 条动作以及原有碰撞
记录精确复现，修正网前 466 秒也与 V154D 完全一致。此前未通过的直行车在
25667 秒进入路口、25681 秒驶出；V154D 的短窗 FAIL 保留，V154E 为独立 PASS。
代价是到达数 1997→1993、平均等待 19.71→20.96 秒，系统车辆小时仅增加
0.17%。这支持该场景下的碰撞改善，并明确了较小的通行代价。新完整清单
确认原 18 起中 17 起属于修正的两个运动对，另 1 起为 16_0↔22_0；后者未在
新轨迹重现，尚不能归因为该运动对本身几何得到修复。V154E 本身的证据限于一个种子。

最新的 [V154F 同网比较](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154f_cologne_repaired_controller_comparison_result.md)
固定修复网和原模型，使用既有三个种子 52282、41242、63792，已完成全部六个
组合；其中 rigid/41242 复用 V154E，只新增五次仿真。六次均完成 3600 秒且无
teleport、车辆总数记账完整。**rigid 为 3/3 零碰撞，PhasePressure 为 2/3**；
后者在 52282 有 1 起事件，保留 FAIL，并纳入所有预定服务均值。

rigid 平均等待 **20.13 秒，PhasePressure 为 13.35 秒**，差 6.77 秒（+50.7%）；
系统车辆小时为 **40.52 对 32.42**（+25.0%），三个种子方向一致。到达数平均
仅少 1.67 辆，不能把主要问题归结为这几辆车；主要差距是过程中累计的行驶、
等待和待入网时间。预先单列的两个补充种子也有等待 +53.3%、系统车辆小时
+25.9%，结论不依赖几何开发种子 41242。

剩余碰撞发生在 27481 秒：直行车仍在 16_1，另一车已到共同出口；后者路线
结合唯一静态连接指向 9→23 掉头路径。两车速度分别为 9.61、11.99 m/s，
均非原先静止左转侵入的记录，涉及的 movement 也不属于两处等待点修正。
其中掉头车的车尾仍跨越上一段，当前记录不能完整重建车身，原因尚需限定窗口重放。
当前可确认 rigid 在这三条轨迹中没有碰撞，PhasePressure 通行更快；尚无
同时证明普遍安全和性能优越的结果。旧单位公式仍保持冻结，本轮没有训练或
source 选择；新单位公式和 source 贡献仍需各自独立验证。

当前证据支持继续做有边界的改善，但优先级需要调整：先修正诊断与验证实现，再解释 rigid 主控的通行退化，最后决定是否值得继续 source 迁移。继续增加候选先验、utility gate 或目标预算，缺少足够依据。Boston 是独立的仿真准入问题。

下文保留最初分析阶段的判断：当时对照用户给出的三份材料，检查本地结果 JSON、标签与运行代码，并核实 SUMO 官方文档及 1.22.0 源码。文档中的建议和授权措辞作为历史材料分析；最初分析阶段没有修改既有实验、重跑 SUMO、提交任务或从服务器下载数据。后续获用户同意执行的工作已在上文逐项更新。

**先纠正两处会改变解释的实际问题。**

Boston 报告把 TraCI `getLeader` 的 distance 当成车身之间的物理间距。该接口返回的是物理间距减去后车 minGap；摘要脚本却又将返回值与 minGap 比较，重复扣除了最低间距。见 [摘要脚本](/home/erzhu419/mine_code/CFCMT/scripts/data/summarize_sumo_collision_trace.py:141)、[官方接口说明](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html#extended-retrieval-messages)及 [SUMO 1.22.0 实现](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSVehicle.cpp#L6358)。

原始位置和车长可独立还原：

| 仿真时刻 | 前车位置减车长、再减后车位置 | 物理间距 | getLeader 返回值 | 真实解释 |
|---|---|---:|---:|---|
| 17264 s | 41.6988557842 − 5 − 34.4105173028 | 2.288338 m | 0.788338 m | 高于 minGap=1.5 m |
| 17265 s | 52.0055705228 − 5 − 46.0627786385 | 0.942792 m | −0.557208 m | 低于 minGap，但没有车身重叠 |

因此，应把“换道即形成低于最低间距的 0.788 m 间隙”改为“换道后仍有 0.788 m 最低间距余量，下一秒前车强刹，物理间距降至 0.943 m”。SUMO 默认会将低于 minGap 的事件登记为 collision，所以冻结零碰撞准入的 FAIL 完全成立。需要纠正的是触发序列的解释。证据见 [原始 trace](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/boston_v18_collision_diagnostic_v1/result.json:5576)、[碰撞定义](https://sumo.dlr.de/docs/Simulation/Safety.html#collisions)。

V153A 的“完整嵌套选择”也与实现不符。代码先把 B100 分成五折，每次以其他 80 组拟合；随后在这五份已缓存 OOF 结果上，再留一折选择候选。例如把 A 折当外层测试时，用于选候选的 B 折预测，其基础模型此前训练过 A 折标签。外层标签因此能够间接影响候选选择。

正确的完整嵌套应先排除外层 A，再在剩余 80 组内形成候选选择数据。项目 V150K 已实现这一路径：外层 80 组内用 60 组拟合、20 组生成内层记录。见 [V153 单层拟合](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_target_calibrated_hierarchical_prior.py:306)、[缓存 OOF 再选择](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_target_calibrated_hierarchical_prior.py:196)、[V150K 真正的内层拟合](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_state_conditioned_source_utility.py:727)。V153 的 B100 之外评估数据仍然独立；此问题限于宣称的完整嵌套审计。现有全拒绝与精确回退保留，修正不能预期带来通过，也不能把当前实现当作对所有 B100 可辨识方法的彻底否证。

**控制效果的首要问题，在 source 之前就已出现。**

V150L 的 Cologne smoke，三臂需求都是 2015 辆：

| 控制器 | 平均 tripinfo 等待/s | 已入网 | 已到达 | 结束时待入网 | 碰撞事件数（incident） |
|---|---:|---:|---:|---:|---:|
| PhasePressure | 14.991 | 2015 | 1999 | 0 | 20 |
| Rigid target-only | 1484.364 | 409 | 210 | 1606 | 0 |
| Selected source | 2047.510 | 363 | 128 | 1652 | 0 |

Rigid 的 352 次决策全部偏离 PhasePressure，只有约 20.3% 需求入网。Source 又将入网比例降至约 18.0%。这说明需要解释的是持续控制下的放行能力丧失，source 误干预进一步加重了问题。精确回退保证的是恢复 rigid；它不保证 rigid 本身有良好的闭环控制效果。先前优于 dense H2O+-style residual 的结构结果仍成立，其比较范围不能扩大为已验证的稳定控制器。

证据：[rigid 原始结果](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/state_conditioned_source_closed_loop_smoke_v3_paired_safety/cologne/cologne1/seed_41242/rigid_target_only/result.json:10000)、[source 原始结果](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/state_conditioned_source_closed_loop_smoke_v3_paired_safety/cologne/cologne1/seed_41242/selected_source_corrected_rigid/result.json:10000)、[PhasePressure 原始结果](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/state_conditioned_source_closed_loop_smoke_v3_paired_safety/cologne/cologne1/seed_41242/phase_pressure/result.json:1328)。

这里有明确的监督与部署差别。反事实标签的含义是：焦点路口执行一个候选动作，其他路口遵循 PhasePressure，此后所有路口继续 PhasePressure；V150K 使用其中 450 秒成本。V150L 则每 10 秒继续用 rigid/source 决策，采用 direct 模式且没有 cooldown。候选首动作之后的策略、状态分布和全网联合动作均不同。代码见 [反事实延续过程](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py:1206)、[V150L 运行配置](/home/erzhu419/mine_code/CFCMT/cf_h2o/config/traffic_signal_tsc_v150l_state_conditioned_source_closed_loop.json:85)。这使标签对实际长期策略的预测力成为首要待验证问题；它本身尚不足以证明 Cologne 退化的唯一原因。

另一个差别是优化量。离线标签按每个动作组的成本范围归一化，降低了城市量纲差异，但也改变了不同状态的实际损失权重；较大的交通损失与较小的交通损失可能贡献相近的归一化误差。均值改善和 0.75 误差分位数余量，均不是单次干预的安全保证，更不是全轨迹保证。见 [标签归一化](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_target_calibrated_source_gate.py:206)、[误差余量拟合](/home/erzhu419/mine_code/CFCMT/cf_h2o/traffic_signal/state_conditioned_source_utility.py:682)。

当前 tripinfo 主指标包含已入网但未完成车辆，处理这一部分是正确的；它不包含尚未入网的 1606/1652 辆。因此应同时解释已有 pending/system-load 指标。仅比较已完成车辆的等待会得出另一种、失真的印象。

**Source 路线的主要困难，是增量弱、条件异质，以及迁移对象尚未真正分离局部机制。**

V150A 的 16/21 与七城均值改善表明 source 信号有一定重复性，但选择过程借用了另外两个评估种子的结果。它回答“有信息时能找到吗”，没有回答 B100 部署条件下能否选到。V150F 的已通过分支中，未来随机性方差占比仅约 0.00905；把主要预算用于多抽未来种子，没有现成证据支持。见 [V150A](/home/erzhu419/mine_code/CFCMT/paper/tsc_v150a_source_benefit_repeatability_result.md:15)、[V150F](/home/erzhu419/mine_code/CFCMT/paper/tsc_v150f_future_randomness_identifiability_result.md:17)。

V150H/I 已说明，增加目标组数与减弱先验确实能改变负迁移，但不能保证 source 的独立价值。V150O 把记录数增至 1512–3240 后仍未通过真实 source 对 placebo/source-blind 的比较；这些记录只有 42–90 个不同状态组，每组标签用于 36 个候选，不能当成同等数量的独立交通状态。New York 保留了局部正信号，Atlanta 的好处由对照同样或更好地解释。见 [V150I](/home/erzhu419/mine_code/CFCMT/paper/tsc_v150i_sample_coherent_source_prior_result.md:14)、[V150O](/home/erzhu419/mine_code/CFCMT/paper/tsc_v150o_dense_pressure_pairwise_source_utility_result.md:10)。

V153A 同样如此。强制 prior 相对 rigid 的均值差为 −0.005919，但最大贡献来自 Atlanta（−0.036579）；Atlanta 的真实 source、matched placebo、zero-centred 三臂评估均值相同。剔除 Atlanta 后，其余六城的描述性均值差约 −0.000809。这个计算不是新的显著性检验，而是说明“改善五城”的读法掩盖了 source-specific 效应较弱这一事实。见 [V153 结果](/home/erzhu419/mine_code/CFCMT/paper/tsc_v153a_target_calibrated_hierarchical_prior_result.md:12)、[Atlanta 三对照](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v153_target_calibrated_hierarchical_prior_20260908/atlanta_preflight_v1/atlanta/result.json:3)。

更值得继续研究的区别是：这些 mechanism prior 用机制名称划分特征块，但其系数都拟合同一个归一化 450 秒长期成本，并未分别监督实际放行率、接受容量等局部响应。长期价值包含目标城市需求、网络分布和后续控制策略的影响；将系数按机制分块，不会自动移除这些影响。因此，当前失败约束的是这类分块价值先验，并没有穷尽真实局部响应的迁移。见 [机制分块](/home/erzhu419/mine_code/CFCMT/cf_h2o/traffic_signal/mechanism_parameter_prior.py:59)、[source 拟合目标](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_state_conditioned_source_utility.py:309)。

Atlanta 的对照还提示一个成本较低的改善候选：单独研究 target-only 的结构化正则化。需给它真正由 target 确定的 scale/precision；现有 zero-centred control 仍沿用 source-derived precision，不能直接改名为完全不使用 source 的方法。当前结果严格限制的是真实 source 均值/标签的额外贡献。见 [zero-centred 实现](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_hierarchical_mechanism_prior.py:334)。

**Boston 仍有改善可能，现有证据不足以确定修复成本。**

静态与全量路线检查通过是实质进展；它们分别解决连接/优先权和路线可行性问题，不能保证微观车流交互无碰撞。当前 v18 在冻结协议下拒绝是正确结论。

不过，SUMO 官方支持没有内部连接的简化微观仿真；它仍遵守信号和次路让行。缺少内部边限制了路口内部排队、阻塞及碰撞建模，但不等同于微观仿真必然不可用。因此“当前包未达到当前严格准入”有证据，“只能重建整个微观网络”则尚无充分证据。[官方说明](https://sumo.dlr.de/docs/Simulation/Intersections.html#internal-links)

更具体的待验证机制是：前车从下游主路状态 M 的车道切入次路状态 m 的车道，随后接近 9 m visibility 范围时强刹。官方说明，次路车辆可以在尚未到达可见距离时预先减速，即使冲突主路暂时为空。故 watched foe lane 为空，不能排除让行模型的预防性减速。[路口速度规则](https://sumo.dlr.de/docs/Simulation/VehicleSpeed.html)

适合的下一项工作是限定于这次事件的刹车来源诊断，保留车辆参数和原包，以决策前后少量 trace 区分换道跟驰、次路预防性减速、下游阻塞及其他速度约束。如果证实编码错误，再定义新修复；如果是行为标定问题，则另立微观标定协议，并用未参与标定的窗口验证。完整内部几何重建和原生 meso 评估都可以成为独立方向，但其成本与科学用途应在机制诊断后判断。

**建议下一步只做一个控制失效定位，再做一个最小机制验证。**

先纠正 Boston gap 摘要、V153 嵌套实现及其表述。另有状态输出不一致：[V150L aggregate](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v150l_state_conditioned_source_closed_loop_smoke_v3_paired_safety.json:65)仍写 `authorize_v150l_full_development_matrix`，实际只检查运行安全、`performance_checks` 为空；论文已记录性能拒绝。应补上明确的最终裁定，避免把 operational PASS 用作下一阶段依据。它不是性能差的原因，但会影响后续实验选择。

研究主线优先定位 Cologne rigid 第一次持续失去服务能力的窗口。先用已有本地日志；若缺字段，在服务器做限定重放，仅返回小型摘要。需要分清三种实际可处置的原因：连续保相导致其他 movement 长期未服务；频繁切换导致清空损失；有绿灯却因接受空间或让行条件无法出车。观察量应连接“选了什么→实际执行多少绿/黄/全红→各 movement 实际通过多少车→队列和待入网怎样增长”。第一类指向服务历史表示，第二类指向动作执行与价值估计，第三类指向局部接受/放行响应。机制未明前不扩大训练矩阵。

若出现可重复的局部响应缺口，再做同表示、同目标预算的 target-only/真实 source/placebo 最小比较，验证 source 能否改善实际服务响应与后续动作选择。所需新信息可能是各 movement 实出流、未服务时长或下游可接收空间，而不是再加一个已有的平均 occupancy/pressure 特征。只有该中间关系能在未用于拟合的路口/窗口预测正确，才值得把它接入完整闭环。

旧版本已经测试过 waiting-aligned 标签、多 horizon、基线延续/cooldown 和机制代理；这些名称本身不构成新路线。新的区别必须是针对当前失效状态、直接验证实际放行机制或修复已证实的执行错误。不能预先假定它们一定存在，也不能承诺再做一次 cooldown 就能改善。

例如 V81 已限制同时最多一次压力覆盖并对齐 60 秒延续；V83 的 uncertainty/h120 曾在开发集改善，但 V84 新种子确认仍未通过。V134 的 served_movement 代理实际是 one_step_green_queue，red_accumulation 实际是 one_step_red_queue，尚不是实出流或逐 movement 服务年龄。见 [V81 配置](/home/erzhu419/mine_code/CFCMT/cf_h2o/config/traffic_signal_tsc_v81_external_v9_multihorizon_estimand_aligned_closed_loop_development.json:74)、[V84 确认](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v84r80_external_v9_runtime_queue_trust_confirmation_20260830/uncertainty_horizon_confirmation_audit_v2.json:13)、[V134 代理定义](/home/erzhu419/mine_code/CFCMT/cf_h2o/eval/traffic_signal_local_mechanism_surrogate_oracle.py:43)。

**证据边界。** Cologne 表格来自一个场景和一个 smoke 种子，足以确认这次运行的严重退化，不能估计七城普遍退化幅度；PhasePressure 同次运行有 20 个碰撞 incident，其低等待不能当作安全部署基准。Source 的两次直接覆盖计数是 source 轨迹上与 rigid 在同状态决策的比较，不代表两条完整轨迹仅有两处动作不同，也尚未单独证明具体队列传播链。Boston 的最后刹车内部原因尚未确定。嵌套实现修正、真实局部响应迁移及新微观标定都还没有新的运行结果。现有 FAIL 保留；可改善的是明确可修的诊断/验证错误，以及尚待实证的控制与机制表示问题，普遍跨城 source 收益目前仍无法承诺。
