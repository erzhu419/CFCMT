# 公交固定强规则隔离补评

复用 v13 全部20个新训练种子的 fixed0250 最终策略，补充固定 `forward_proportional` 规则：增益0.6，保持上限60秒，使用相同 early60 执行器。既定5个测试日×两种需求，共10次24000秒评估，另有一次同日duet规则完整复现确认运行环境；不新增训练。v13测试日已有分析，本次是更强规则补评。

唯一主比较为 fixed0250 对新规则，按20种子×3源条件×2需求等权汇总。判定、训练种子、规则参数及统计范围见 `protocol.json`。原有 duet/H2O+/其他系数比较仅描述，直接复用既有摘要。

本轮代码和结果归 CFCMT 所有。运行器只使用自己的 `thesis_snapshot/sumo_eval` 冻结代码和 `reference_snapshot` 中600份既有测试记录；`runtime.json` 固定Python及环境。原 v13 输出来源记在协议中，执行时不读取其他对话正在修改的 Thesis 代码或输出。

本地验证入口为 `python3 -m unittest -v test_confirmation.py`、`python3 launch.py plan`。`python3 launch.py stage` 仅上传本轮控制代码、协议和小型runtime配置。服务器单独执行一次 `freeze_runtime.py` 冻结所需代码及参考JSON，数据使用服务器只读链接；之后运行 `collect_results.py --inventory`。

`python3 launch.py smoke` 提交两种需求的1200秒接口检查；`python3 launch.py submit` 提交11次完整评估。运行环境确认将冻结代码的duet结果与原同日cost、departed人数和completion比较；不一致时输出INVALID_RUNTIME并停止旧结果配对，与科学STOP分开。提交记录留在本目录。`python3 launch.py watch` 在服务器完成比较，仅回收 `analysis_summary.json`。逐日结果、数据与权重留在服务器。`previous_thesis_location` 保留迁移前的取消任务和接口检查记录。

逐日客流分母必须完全一致。缺失输出保留PENDING，协议或配对不符直接中止汇总；STOP不调整增益、种子或判定条件。
