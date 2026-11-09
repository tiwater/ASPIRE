# RealBot-01 咖啡开发入口

当前工作基线：[咖啡工作流程 v1](coffee-workflow-baseline.md)，Git 标签 `coffee-sim-baseline-v1`。后续开发以冻结配置及本次连续运行结果为对照。

最新连续仿真状态与后续工作：[咖啡流程协作交接页](coffee-collaboration-handoff.md)。[v58 复现说明](v58-reproduction.md)包含运行脚本、固定输入和审计方式；原始结果不能逐字节回放，因为当时没有保存完整模块快照。v58从几何定义的站立下垂起点，连续完成两侧抬臂准备、三杯服务、一次低余量还壶/模拟补液/重取壶、最终还壶与退出；独立验收和4倍自由/2倍接触速度策略审计通过。v61另验证最后约31 mm壶座入座/释放可用500 µs步长。结果来自固定场景几何真值和规定质量转移；开机姿态未由真机测量，真实液流、学习策略及硬件执行尚未验证。

当前抓姿调整：[侧面垂直接近、上横梁承托与柔性退出](upper-handle-validation-20261101.md)。用户要求取消不利于倒水的 45° 换抓，允许约 135° 朝向微调及出水口约束期间短时单侧接触；新抓姿已完成上述单次连续名义几何流程；该链接内的 33 N 局部退出结果为历史诊断。参数入口为 `configs/realman/coffee_upper_handle_candidate.json`。

历史完整倾倒基线：[降低腿部站姿、45° 换抓与完整动作衔接](pouring-path-validation-20261101.md)。原指垫、两杯水负载保持；取杯—取壶—倾倒—放杯—还壶已通过一次 673.4 秒的完整连续名义仿真，包含双手撤离；视频、指标与复现命令见报告。[直接打开完整回放](../../artifacts/realman/pour_completion_20261101/leg_low_pour_cold_full_axial_overview_4x.mp4)。开发配置为 `configs/realman/coffee_pour_candidate.json`；规定负载转移不等于真实液流验证或学习策略训练完成。

## 历史迭代记录

前一轮站姿试验：[腿部调整、上身约 2.9°、肩部距咖啡机约 54.5 cm](leg-stance-return-validation-20261031.md)。原指垫、两杯水负载下，连续取杯、退出壶 180 mm、回壶及松爪稳定检查已通过；回推绕行峰值 6 mm、朝向纠偏范围 ±2°，抓持及碰撞检查保持。该轮完整倒咖啡流程尚未通过；[旧站姿的局部诊断](return-yaw-validation-20261031.md)单独留档。

前一轮控制试验：[将平台支撑滑动的倾角验收上限调为 5°](supported-tilt-validation-20261031.md)。按用户要求单独放宽该项，其他抓持、受力和碰撞判据保留；180 mm 支撑滑动退出通过，回推至 20 mm 时仍有偏航误差待处理。

最新几何修正：[加热盘低于平台 2 mm、圆滑过渡、盘宽不小于壶底](recessed-platform-validation-20261031.md)。默认场景与负载准备器已更新；下面的 200 mm 滑动结果属于旧平台几何，不能代替新场景复验。

上一版取放方式（旧平台几何）：[沿平台支撑滑动取放](supported-platform-slide-20261031.md)。平台擦底属于预期接触；冷启动已验证至 200 mm 滑动退出，最终悬空承重与完整还壶仍待通过。

当前诊断与控制改进：[分级接触力、接触短暂丢失、轨迹接续及碰撞精度修正](diagnostic-margin-validation-20261031.md)。试验配置为 `coffee_workload_diagnostic_ccd.json`，保留原指垫和两杯水；新诊断窗口仅用于仿真，完整带载流程尚未通过。

当前出口模型：[圆锥台配合、安装预压与进出验证](conical-outlet-validation-20261031.md)。原规格指垫和两杯水保持不变；默认负载准备器使用该机构假设，机构通过不等于机器人咖啡流程通过。

当前默认：[两杯水负载、原规格指垫](two-cups-validation-20261031.md)。装水量随杯容量计算；禁止改变指垫规格。工作负载配置为 `configs/realman/coffee_workload.json`，较早的 1 L / 改型指垫候选只作历史记录。

弹簧力后续对照：[两杯水、出水口弹簧再次减半及接触力诊断线说明](two-cups-spring-half-20261031.md)。保持指垫原规格，单独检查机构阻力的影响。

历史接触改进：[数值稳定性、扶正分段复验及指垫改型](contact-refinement-20261030.md)。多接触点设置通过了 30° 至直立分段；低力矩改型通过微抬，尚未通过退出或完整咖啡流程。

历史 1 L 负载工作：[1 L 负载与夹爪承载验证](payload-gripper-validation-20261030.md)。包括空壶接触检查修正后的完整复验、加力/加刚度对照及剩余限制；空壶通过不代表 1 L 完整流程通过。已核对官方 CTAG2F90-D 最大夹持力 25 N；当前强化模型仅作诊断参考，尚不能映射为真机能力。

最新候选：[135° 初始壶柄与完整连续空壶动作验证](initial135-validation-20261030.md)，包含视频、复现命令、证据与未验证项。当前开发场景由 `configs/realman/coffee_stance_reference.json.active_scene_candidate` 指向。

最新路线优化与取壶诊断见[右腕朝上接近](camera-up-pregrasp-20261029.md)。

当前杯架模型与取杯验证见[四片弹性卡扣](dispenser-elastic-tabs-20261029.md)。

## 开发与迁移入口

真机控制与标定的可复用代码、证据和部署边界见 [部署单元说明](deployment.md)。
恢复真机适配前先查 [历史标定与控制证据索引](hardware-evidence-index.md)，其中明确
区分已提交资料、未入库的本地原始资料、历史验证范围和缺失项。

**当前主线（2026-10-29）：仿真建模/适配 → ASPIRE 技能迭代与独立验证 → 真机迁移。**
此前真机观察/标定保留为迁移资料，停止以真机抓取为下一步。当前不要求继续摆位。
主计划以 [coffee-plan.md](coffee-plan.md) 为准；旧现场文档中的“下一步”不是当前执行任务。

可重复的仿真初态检查（不是咖啡策略评分）：

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv-libero/bin/python -m cap_harness.realman \
  benchmark-check --split development --seeds 51 52 53 \
  --output artifacts/realman/COFFEE_NEW_FIXTURE_RUN
```

详细状态、实施顺序和现场所需数据见 [coffee-plan.md](coffee-plan.md)。
最新仿真结果见 [工作姿态、避碰路径与物体判分](simulation-workspace-20261029.md)。
后续接触开发、杯间/杯架摩擦分离和可重复入口见 [接触机构进展](simulation-contact-20261029.md)。
当前按操作员要求先做[多视角几何核查](geometry-audit-20261029.md)：两轮共 24 组 RGB-D 已归档，并修正黑色加热盘尺寸。
当前具备只读硬件采集、参数化 MuJoCo 场景、ASPIRE 仿真适配及默认只读的
[RealMan MCP 控制服务](live-control.md)。运动实现仍待实机验收和工作空间审查，尚无完整咖啡策略。

## 在当前环境使用

在项目根目录运行；现有 `.venv-libero` 已包含本轮所需依赖。
新环境可安装项目的 `realman` extra，ROS 采集脚本则在机器人 ROS2 Humble 环境运行。

```bash
export PYTHONPATH=src
.venv-libero/bin/python -m cap_harness.realman build
MUJOCO_GL=egl .venv-libero/bin/python -m cap_harness.realman render
.venv-libero/bin/python -m cap_harness.realman viewer
```

`viewer` 在有图形桌面的终端运行，可旋转/缩放查看；它不驱动物理机器人。
查看器默认仅展示模型，避免交互查看时物体自行演化。
`artifacts/realman/coffee/scene.xml` 可在 MuJoCo 中加载，依赖原始 `ref/realman_robot` 网格路径。
`overview.png`、`front.png`、`side.png` 是可直接打开的预览。

```bash
# 接口回归示例：只在仿真中让右臂第七关节小幅运动。
MUJOCO_GL=egl .venv-libero/bin/python -m cap_harness.realman run examples/realman/observe_and_move_sim.py

# 初始物体保持与数值稳定性，不代表取杯/倒水成功。
.venv-libero/bin/python -m cap_harness.realman validate --seconds 5

# 离线安装坐标系 IK 探针；输出不能发送给真机。
.venv-libero/bin/python -m cap_harness.realman ik --arm left --xyz 0.55 0.25 1.0

# 查看未完成的硬件标定项。
.venv-libero/bin/python -m cap_harness.realman doctor
```

CAP 程序中可以显式调用 `solve_ik(pose, arm="left", backend="realman_sim")`，
结果是 URDF 安装末端的数值 IK，并检查终点碰撞；全路径规划尚未接入。
当前 `move_to_pose` 默认的 PyRoki 服务未为该机器人配置，不能假定可用。
七轴关节、双臂轨迹、夹爪、合成 RGB-D 与现有 CAP 数据契约一致。
原有 `cap-harness run --benchmark ...` 和 YAM MCP 启动器未被改成 RealMan；使用这里的专用入口。

每次 `run` 在 `artifacts/realman/runs/` 创建独立记录，保存程序、模型/参数副本、构建哈希、
初末 RGB-D、工具调用摘要和结果。`coffee_task_success` 保持未知，不把接口示例通过算成咖啡任务完成。

## 只读 ROS 采集

`tools/realman/capture_ros.py` 只有订阅，不创建运动服务客户端/发布器。
每次使用一个新的输出目录，保留原始证据。示例在机器人上执行：

```bash
source /opt/ros/humble/setup.bash
python3 capture_ros.py --output /tmp/coffee_capture_NEW_ID --timeout 20
```

将采集目录复制回本机后：

```bash
.venv-libero/bin/python -m cap_harness.realman capture-summary artifacts/realman/aspire_coffee_capture_20261026_01
```

原始图像保留编码、字节序、行跨度和时间戳。`16UC1` 不独立证明深度单位是毫米。
本次诊断帧不保证同步，深度和彩色也未配准；不能用未验证外参拼装成虚假的世界坐标观测。

## 修改与检查

- `configs/realman/robot.json`：硬件身份、话题、单位、七轴命名、待标定项。
- `configs/realman/coffee_scene.json`：尺寸、物体关系、代理接触参数及来源。
- `src/cap_harness/realman/`：构建、仿真适配、IK、状态解码、采集读取、任务判据。
- `tests/realman/`：单位/错误数据/过期状态/中空杯/初态稳定/离线 IK 回归。
- `examples/realman/observe_and_move_sim.py`：ASPIRE 策略入口最小示例。

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv-libero/bin/python -m pytest -q tests/realman tests/test_contracts.py
.venv-libero/bin/python -m ruff check src/cap_harness/realman tools/realman tests/realman
```

模型测试需要用户提供的 `ref/realman_robot`，缺失时会明确跳过，不伪造测试通过。
生成的模型、预览和现场采集写入已忽略的 `artifacts/`；源码及选定标定证据已提交，
完整原始采集并未全部入库，详见上述证据索引。

用户确认后的[容量/力/倾倒初值](parameter-priors.md)和[标定配合流程](calibration.md)已单独记录。
可运行 `python -m cap_harness.realman estimate` 查看可重算参数，
用 `calibration-check CAPTURE_DIR` 做板与相机元数据的只读预检。
