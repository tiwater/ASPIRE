# RealBot-01 咖啡仿真存档

保存双 RM75、CTAG2F90D 夹爪和咖啡操作台的第一版仿真成果。模型来自用户提供
的 URDF/STL 与多角度照片；咖啡机、壶、纸杯、摩擦杯架为可调参数的几何模型，
不是高精度扫描件。当前没有经过验证的端到端咖啡操作策略。

## 已保存的内容

- `ref/realman_robot/`：原始机器人 URDF 与 STL 网格。
- `ref/scenes/coffee/`：建模参考照片。
- `configs/realman/coffee_scene.json`：物体尺寸、质量、接触参数及来源。
- `configs/realman/robot_sim.json`：独立仿真配置；不包含真机连接或标定结果。
- `src/cap_harness/realman/`：场景构建、七轴双臂仿真适配、离线 IK、任务判据，
  以及只读采集解析与标定预检。
- `assets/realman/coffee/`：已有模型、预览、参数估算、构建和稳定性记录的存档。
  XML 网格引用已转为仓库相对路径；原始运行记录仍保留当时本机路径和哈希，
  不能将其原始 XML 哈希当作路径转换后文件的哈希。
- `examples/realman/observe_and_move_sim.py`、`tests/realman/`：仿真调用示例与回归测试。

## 重建与查看

在仓库根目录、Python 3.10–3.12 环境中安装 `pip install -e '.[realman,dev]'`，
然后执行：

```bash
python -m cap_harness.realman build
MUJOCO_GL=egl python -m cap_harness.realman render
python -m cap_harness.realman viewer
python -m cap_harness.realman validate --seconds 5
MUJOCO_GL=egl python -m cap_harness.realman run examples/realman/observe_and_move_sim.py
```

构建默认使用 `robot_sim.json`，输出到已忽略的 `artifacts/realman/coffee/`。
也可直接查看存档：

```bash
python -m cap_harness.realman viewer --model assets/realman/coffee/scene.xml
python -m cap_harness.realman doctor --robot-config configs/realman/robot_sim.json
```

图形查看器需桌面显示；无显示环境使用 EGL 渲染。预览图可以直接在 GitHub 中查看。
如需重复打包该模型，必须同时保留 `ref/realman_robot` 的网格。

## 模型边界

杯和壶均有中空结构，出水口用弹簧滑动接触代理，杯架用摩擦触点及独立杯体。
空壶质量为用户给出的 396 g；八成杯按几何内腔容量的 80% 计算，
弹簧力与杯架力仍为待实测初值，详见 [parameter-priors.md](parameter-priors.md)。
液体行为未模拟。初态稳定、接口运动成功不能证明取杯、倒水或真机任务完成。
URDF 安装末端不等于夹爪 TCP，离线 IK 的终点碰撞检查不等于全路径规划。

[首轮计划](coffee-plan.md)和[当时验证记录](validation-20261026.md)保留历史状态；
后续真机控制与相机标定工作另行保存，本次提交集中于可恢复的仿真成果。

## 本次归档核验（2026-10-27）

在仅含待提交文件的独立目录中运行仿真/任务/参数/只读解析相关测试与既有
CAP API、契约回归：109 项通过。默认命令可从源 URDF 重建模型；相对路径
存档可直接加载，5 秒保持检查通过，MuJoCo 警告计数为零。存档文件的新校验和
见 `assets/realman/coffee/snapshot_manifest.json`。原始厂商 URDF 字节保持不变。
