# 咖啡流程协作交接

更新：2026-11-09。此页是协作者的快速入口；更详细的设计过程和失败记录见[咖啡开发记录](coffee-service-development-20261102.md)及[主计划](coffee-plan.md)。

## 当前可复现结果

- v58 从仿真直立下垂起点连续完成准备、取壶、三杯服务、一次低余量还壶/补液、再次取壶、最终还壶和退出。结果为 `standing_final_hold`，仿真时长 994.28 s。
- 独立连续验收通过；48 段搬运速度审计通过，其中 12 段无接触采用 4 倍参考速度，36 段接触/有支撑采用 2 倍。
- 关键接触阶段使用 250 µs。1 ms 带载倒水曾出现夹爪单侧卸载，不纳入合格配置；500 µs 只在末段壶座入座/释放通过。
- v58 的完整轨迹、检查点、力跟踪和验收输出保留在本地 `artifacts/realman/coffee_service_20261102/`，不包含在精简协作提交中。连续结果和速度倍率见本页摘要；需要重新核验运行时，应从当前代码重新运行并保存当次模块快照。
- v58 全流程视频可在本地生成的文件中查看：`artifacts/realman/coffee_service_20261102/lifecycle_coffee/service_lifecycle_staff_collection_clearance_v31_cycle_speed2_v58/coffee_service_v58_4x.mp4`。它是 4 倍播放、约 4 分 9 秒，未纳入版本管理；视频不是机器人执行时间测量。

## 重要边界与当前首要工作

v58 使用的准备姿态将 `dt_joint` 扩展到 `+1.018853 rad`，超过当前厂商 URDF 中 `+0.17 rad` 的上限。该结果只说明扩展范围下的离线仿真流程通过，不能作为真机动作或限位依据。

在源 URDF 关节范围内已找到一个双臂预抓可达且两侧进臂路径采样无碰撞的身体姿态候选：`xt_joint_1=-1.5`、`xt_joint_2=+1.5`，其他身体关节为零。对应筛选摘要和全身路径检查留在本地 artifacts，未随精简提交；候选尚未用于咖啡任务端到端仿真，也没有伺服动力学证据。

建议的下一步：

1. 以该限位内姿态重定向 `initial_standing → prepare` 的身体和手臂目标，重新计算咖啡机/杯架相对可达性与两侧进臂路径。
2. 用固定物体几何和原有夹爪、载荷、接触验收条件，重新跑准备—完整服务/补液—退出；不能把旧 v58 的范围扩展姿态或其结果直接继承为新姿态的通过证据。
3. 在运行入口捕获实际使用的模块快照，再运行连续流程、杯子交付、释放、速度策略审计；保留 4 倍无接触/2 倍接触的逐段核验。
4. 完成场景和仿真策略验证后，再进入 ASPIRE 策略训练/独立验证和真机迁移评审。

## 代码入口

- `src/cap_harness/realman/coffee_service.py`：取壶、单杯服务、还壶、余量门槛、等待补液及退出的业务状态机。
- `src/cap_harness/realman/service_lifecycle.py`：直立工作位准备、退出路线及逐级扩大物理步长的状态检查。
- `src/cap_harness/realman/coffee_speed.py`：按独立清空/接触或支撑证据选择 4 倍和 2 倍参考速度。
- `tools/realman/audit_coffee_speed_policy.py`、`tools/realman/compare_coffee_service_speed_profiles_v2.py`：速度策略审计与速度目标汇总。
- `tests/realman/test_coffee_speed.py`、`test_coffee_speed_policy_audit.py`、`test_coffee_speed_comparison.py`：速度选择和报告逻辑回归。
- `ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf`：源 URDF；它的关节限制优先于扩展范围仿真参数。

所有已验收流程均为固定场景几何真值控制和规定质量转移；没有模拟真实液流、学习策略输出、工作人员真实运动，也没有完成真机资格确认。
