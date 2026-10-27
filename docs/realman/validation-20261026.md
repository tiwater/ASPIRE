# 首轮验证记录：2026-10-26

## 软件与模型

使用当前 `.venv-libero`、MuJoCo 3.3.0：

```text
pytest -q tests/realman tests/test_contracts.py tests/test_api.py
98 passed in 21.06s

ruff check src/cap_harness/realman tools/realman tests/realman
All checks passed!

git diff --check
通过
```

覆盖状态单位/错误数据、图像字节序和行跨度、七轴契约、10 个夹爪 mimic 约束、
中空杯射线检查、无效控制批次无副作用、过期起点拒绝、离线 IK 不改变仿真状态、
身体冻结姿态改变 FK、现有 CAP 程序调用、阶段证据时效与完整性。

生成模型包含 74 个 body、35 个 joint、341 个 geom、16 个 actuator。
保留机器人原始 STL，引入参数化中空对象和接触代理；哈希、装配碰撞排除及限制在
`artifacts/realman/coffee/build_report.json`。

## 名义保持测试

空壶与 4 杯在无机器人运动的模型中演化 10 秒：

| 物体 | 相对初始位置的位移 |
|---|---:|
| 壶 | 1.02 mm |
| 最下层杯 | 4.83 mm |
| 其余三杯 | 3.21 / 2.39 / 2.52 mm |

状态有限，MuJoCo 警告计数均为零；未出现杯子落到杯架底座。
这是代理参数下的初始保持测试，不是出杯、倒水、夹持和放回成功测试。
最初的低刚度版本会在数秒后掉杯，已保留这一发现于开发记录；
当前摩擦/刚度参数是仿真调试值，必须通过真实出杯力和杯间摩擦测量校准。

`examples/realman/observe_and_move_sim.py` 经 ASPIRE ProgramExecutor 执行，
右臂第七关节小幅运动成功，输出两臂各 7 轴。
结果不赋予咖啡任务成功标记，每次运行记录位于 `artifacts/realman/runs/`。

## 现场只读核对

读取 `/robot_slave/states` 两个样本，左右臂关节错误码均为零。
取得四路深度、对应四路彩色，以及胸部广角彩色，共 9 张图；
同时保留 CameraInfo、TF 和时间戳。图片可以看到真实操作台和咖啡器具。

胸部驱动查询返回 `enable_depth_scale=true`、`depth_registration=false`、
`align_mode=SW`，`depth_precision` 为空字符串；没有据此猜测精确深度比例。
数据仍需要深度单位确认、同步和外参验证，尚未转换成实时 CAP 世界坐标观测。

未发送机器人运动/夹爪命令，未运行历史动作脚本，未修改机器人现有控制服务。
没有验证真实咖啡流程、液体量、热液安全、力阈值、手眼精度或落座精度。

## 明确未通过/未完成的项目

- 默认 URDF 零位、默认场景站位尚未证明咖啡操作可达。
  左臂安装末端对 `[0.55, 0.25, 1.0] m` 的位置探针残差约 107 mm，返回失败。
- RealMan 真机运动后端及原生 CAP 工具尚未实现；YAM 工具不能用于该机器人的七轴控制。
- 当前控制器↔URDF、工具、身体和相机变换未完成认证。
- 杯架/弹簧/液体与材质模型没有实测参数；未实现真实液体仿真。
- 未实现端到端咖啡策略、视觉阶段判据或真实倒水量闭环。

这些事项不会被单元测试或仿真示例通过所覆盖。后续顺序见 [实施计划](coffee-plan.md)。
