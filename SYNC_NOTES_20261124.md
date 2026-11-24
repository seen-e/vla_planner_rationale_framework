# 1015 实例同步说明

本分支同步自 `/mnt/workspace/vla_planner_rationale_framework`，日期 2026-11-24。
推理模块、提示词、调度器及配置按实例文件同步，不迁入虚拟环境、缓存、真实预测数据或 API 凭据。

## 当前运行入口

从仓库根目录执行：

```bash
python -m pip install -e '.[test]'
python -m scheduling.run \
  --config config.robodojo.yaml \
  --credentials /path/to/private/credentials.yaml \
  --input /path/to/inputs/manifest.json \
  --output /path/to/new/predictions
```

私有凭据文件提供 `base_url`、`api_key`、`model`。不要将真实凭据提交到仓库。
`config.robodojo.yaml` 中的 `configured_from_credentials` 是运行时替换占位符。

当前配置为 A=24、B=32、C=24 独立并发，三个阶段启用思考模式；
调度入口使用 SSE 流式传输。A/B 需要图像，C 为纯文本整合审核。

每个 episode 完成 A 后即可提交自己的 B；该 episode 的 B 全部完成后即可进入 C，
不等待其他 episode。详情见 [scheduling/ReadMe.md](scheduling/ReadMe.md)。

`planner_rationale.batch` 是保留的旧整批阶段入口。
`ROBODY_ADAPTATION.md` 中关于阶段屏障的描述针对这个旧入口，不能据此理解当前独立队列调度。
运行 `python -m scheduling.run` 时请位于仓库根目录；当前 setuptools 包配置只包含 `planner_rationale`。

## RoboDojo 数据适配与现有结果

`planner_rationale/robodojo.py` 负责核验并转换原 Stage3 输入，按实际 sample 映射转换时间。
该适配器默认选择已通过审核或矫正核验的版本；后来补跑的 238 条原标注是经单独授权、
由实例外部脚本构建的输入，不是把未通过的数据自动标记为通过。

实例 `/mnt/workspace/script/` 下的数据迁移、批次启动和兜底修复脚本不属于本次框架快照。
本仓库不声称包含所有历史实验调度脚本；标准独立队列运行入口已包含在仓库中。

已完成的 3500 条结果目前在：

```text
/mnt/data/chachaxu/dataset/robodojo/planner_rationale_result/
```

该目录是完成后合并的数据发布目录，不是两个旧批次的原始缓存路径。
使用 `result_index.json` 读取当前结果。完整三类标注数据说明位于实例：

```text
/mnt/data/chachaxu/dataset/robodojo/RoboDojo_Annotation_Data_Guide.md
```

## 测试

从仓库根目录运行：

```bash
python -m pytest tests scheduling -q
```

必须显式包含 `scheduling`，因为默认 pytest testpaths 仅包含 `tests`。
这些是接口、采样、调度、流式与缓存测试，不验证真实模型的标注语义质量。
