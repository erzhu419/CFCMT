# 全网递推路线：状态与首轮测量方案

**设计完成，首轮仅提议8条济南测量分支；本轮0仿真、0拟合。** 状态采用逐车packet：位置、速度、参数、已走过的车身路径、持续的转向belief；另保留边界待插入库存和全网完整执行器。15米cell只作索引。这是近似模型的闭合状态设计，尚未验证SUMO动态可由其充分预测。

模型每秒推进车辆与库存，再更新占用清空和信号时钟；每10秒由预测状态重新运行原PhasePressure（PP）。成本由原受控车道的预测停车车辆数逐秒累积。来源以后提供局部通行动态先验，目标用本地单轨观测适配；当前不训练新的450秒scalar读出。

首轮固定候选为济南seed76116801、intersection_2_2、t600起60秒内首个八动作时刻。首PP计入8条，所有分支原生从零开始、各测450秒；包括前缀最多8880模拟秒。只验收全网每秒packet/车道/连接流/执行器、车辆守恒、原PP请求及成本口径。该seed和8臂永不进入后续训练或验证，失败保留记录并停止。

已确认固定1.22运行时27个必要getter符号存在，尚未调用其仿真状态接口。占用率fraction由两条旧实测记录与[1.22 getter源码](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_22_0/src/libsumo/Lane.cpp)、[车道实现](https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_22_0/src/microsim/MSLane.cpp)交叉核对；PP原系数保持。

后续先验收纯目标模型的递推/成本技术校准，再另定预算比较真实来源S、纯目标T、置换来源P及PP。T无需先胜PP；S的真实450秒成本须同时胜T/P/PP。训练量、学习器和后续评估样本数尚未定，不能由这8条测量先导推功效。

当前只定义‘首动作后PP续行450秒’的模型评价。整小时采用450秒option还是10秒滚动选相，尚须独立协议；真实未来状态、路线、到达计划和PP请求不能作为初始规划输入。当前getter和参数来自全观测仿真，实路传感可得性尚未建立。

**下一步：实现被动前缀记录、每秒全网采集与账本验收，再按8条提案执行测量；本轮尚未启动。** 原结果与Thesis不改。

[状态递推](state_transition_design.json) · [字段合同](data_contract.json) · [8条先导提案](measurement_protocol_proposed.json) · [分阶段预算](budget_and_gates.json)
