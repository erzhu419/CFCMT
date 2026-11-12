9 个固定案例、18 条有效原生轨迹已完成；全部检查点物理状态、候选对比数据、PP 成本曲线及两臂 450 秒成本复现。7 项测试通过。逐秒原始数据留在服务器，本地案例 JSON 合计约 114 KiB。

九例本路口黄灯/全红 mode 逐秒一致，首次目标绿灯均在第 7–40 秒执行。不能将这些案例的损失归因为额外清空时间。

短期和本地指标会误判：case 2 前 60 秒多损失 244 车辆·秒，最终改善 2789；case 8 前 120 秒损失 379，最终改善 985；case 1 在 120 秒改善 19，最终损失 162。case 4 本路口改善 435，其他路口恶化 1057，最终损失 622；case 3 则主要是本地损失（993），因此问题同时包含本地服务排序与后续网络反馈。

下一步从已有 100 组动作前快照恢复“候选动作关联的下游相位状态”，先确认相对 PP 存在可辨识差异，再作一次固定五折增量对照；复用 450 秒标签、history80、模型参数及原选择规则。无需新增仿真，最多 5 个新拟合。完整数值见 [summary.json](summary.json)，下一步定义见 [next_phase_context_spec.json](next_phase_context_spec.json)。

范围与记录：九例按结果类型选取，只用于机制辨别。首个脚本曾遗漏 reference/delta 列，在动作窗口前中断；修复后18条有效轨迹完成，失败前缀单独保留。[记录](first_attempt_failure.json)。v2 原始占用字段误标 pct，数值实际为长度占用比例；最终 summary 已更正字段名，未重跑或改动数值。[单位说明](raw_units.json)，依据 [SUMO 1.22 Lane 实现](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_22_0/src/libsumo/Lane.cpp) 与 [长度占用定义](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_22_0/src/microsim/MSLane.cpp)。
