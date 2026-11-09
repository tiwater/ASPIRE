# v58 咖啡流程复现

本页提供一条从初始站姿重新运行 v58 参数流程的可复现路径。材料在 [`assets/realman/coffee/v58_reproduction/`](../../assets/realman/coffee/v58_reproduction/)，运行工具为 [`run_coffee_v58_reproduction.py`](../../tools/realman/run_coffee_v58_reproduction.py)。

## 复现历史结果与重新运行的区别

原始运行记录确认 v58 连续完成准备、三杯服务、一次模拟补液/重取壶、最终还壶和退出，终态为 `standing_final_hold`，仿真时间 994.28 s；48 段速度策略审计通过（12 段 4 倍自由运动，36 段 2 倍接触/支撑运动）。历史摘要、命令参数和验收报告随仓库保存。

原始运行目录没有保存 `implementation_snapshots/captured_during_run.json`，因此不能对原始 v58 做逐字节、原环境回放。本工具使用保存的原始实验脚本、命令参数、初始姿态、生命周期计划、场景配置和模型，从起始状态重新运行；运行时将当前仓库中的完整 `cap_harness` Python 源码复制为不可变快照，并写入新 trial。新 trial 可使用 `replay_coffee_trial.py` 再次回放。新运行的结果应单独审计，不能继承原 v58 的验收结论。

场景 XML 原来使用 `/home/sawyer/aspire/...` 绝对网格路径。包内包含仓库中的 RealBot STL 网格，以及原始 1,320 个 base-housing OBJ 网格的压缩包和逐文件哈希；启动器会把 OBJ 解包到被忽略的 `artifacts/realman/coffee_service_20261102/base_housing_parts/`，再通过仓库根相对路径加载。网格几何未替换或简化，XML 字节哈希与历史模型不同。历史模型哈希、移植模型哈希及输入文件哈希记录在包内清单中。

## 环境

使用 Python 3.12，安装本项目 RealMan 仿真依赖：

```bash
python -m venv .venv-v58
source .venv-v58/bin/activate
python -m pip install -e '.[realman]' numpy==1.26.4 scipy==1.17.1
```

历史环境记录为 Python 3.12.14、MuJoCo 3.3.0、NumPy 1.26.4、SciPy 1.17.1。复现工具强制 MuJoCo 3.3.0，并检查其余关键版本。模型引用的 RealBot 网格位于 `ref/realman_robot/LSRX_V0.0.13.SLDASM/meshes/`；base-housing 网格由复现包自动解压。

## 运行

```bash
python tools/realman/run_coffee_v58_reproduction.py --name v58_reproduction_001
```

第一次运行会在被 `.gitignore` 忽略的 `artifacts/realman/coffee_service_20261102/` 下准备输入和结果。准备结果但暂不启动仿真：

```bash
python tools/realman/run_coffee_v58_reproduction.py --name v58_reproduction_001 --prepare-only
```

每次必须使用新的 `--name`。流程仅允许仿真，输出目录保留原始轨迹、检查点和接触诊断；一次完整结果约占 2 GB 磁盘空间。视频和渲染帧不生成、不提交。

完成后工具自动运行连续服务/杯交付验收和速度策略审计。成功条件是实验 `result.json` 报告 `experiment_success=true`、`error=null`，且两个审计命令都返回 0。结果摘要、日志和审计报告保存在该 trial 目录。

对新运行做一次快照回放，可使用：

```bash
python tools/realman/replay_coffee_trial.py \
  artifacts/realman/coffee_service_20261102/v58_reproduction/v58_reproduction_001 \
  --name v58_replay_001 --execute
```

回放也必须使用新名称。它验证的是刚才新运行的快照，不会补回原始 v58 缺失的历史模块快照。

## 验收边界

此流程复现的是固定场景几何真值控制和规定质量转移代理，不含真实液流、学习策略输出、真实工作人员运动或真机验证。原始 v58 的 `dt_joint=+1.018853 rad` 超出当前厂商 URDF 的 `+0.17 rad` 上限；成功重跑仍不能作为硬件动作资格证明。
