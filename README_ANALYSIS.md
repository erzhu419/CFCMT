# CFCMT 精简分析包

本包保留算法源码、测试、配置、公交实验、当前信控论文和版本级关键证据。
文件来自完整项目归档；没有重训、重跑实验或改写结果。

## 阅读入口

- **当前信控投稿稿：** `paper/GPT_TITS_paper/main.pdf`；正文、证明、图表脚本和源数据位于同目录。该稿对应 V93。
- **V93 完整配对比较：** `paper/GPT_TITS_paper/source_data/source_selector_with_mp_*.csv`，56 个种子、四个场景、六个控制器，共 1,344 次运行。原五方法数据也保留，供 MP 补充分析使用。
- **V93 方法与配置：** `cf_h2o/eval/`、`cf_h2o/config/`、`cf_h2o/results/cluster/tsc_v93_source_city_components_20260831/`；MP 补充位于 `experiments/tsc_v93_max_pressure_20261030/`。
- **公交方向：** `cf_h2o/`、`mc_wm/`、`H2Oplus/`，以及 `experiments/bus_cross_city_native_v1/`、`experiments/bus_thesis_rule_confirmation_v1/`，均保留。
- **历史与失败证据：** `paper/` 中的版本进展和结果报告、`paper_tsc/`、`paper/development_history_20261030/`、各版本的关键结果与审计记录。
- **目录索引：** `ANALYSIS_CONTENTS.csv` 给出本包实际包含的文件数量和字节数。

V43 与 V93 是不同控制器版本，结果分别保留。当前 V93 对 MaxPressure 的等待时间改善有统计支持；对 PhasePressure 的均值较低，但 95% 区间跨零。

## 精简范围

省略外部参考论文 PDF（文献表仍保留）、两个已展开的重复论文源码 ZIP、第三方可视化静态资源，以及重复的逐种子/分片中间记录和较大的旧结果明细。
所有算法源码和当前 V93 论文的完整小型验证数据均保留。191 个 cluster 实验目录各保留关键记录；仅有启动快照的目录仍保留其快照，没有补造实验结果。

完整版本仍保存在 GitHub 的 `codex/cross-city-validation` 分支：
https://github.com/erzhu419/CFCMT/tree/codex/cross-city-validation

本包用于代码与结果分析。完整原始数据、模型和中间执行记录保留在原归档，原生仿真复现仍需要相应输入与运行环境。
