# VLA Planner Rationale Annotation Framework

基于 **Stage3 已完成的 Subtask 标注 + Episode 视频**，生成每个 Subtask 的任务规划依据：**为什么在前一段之后要执行当前段？**

该项目是 **Stage3 后独立的 Planner 标注流程**，不改 Stage1/2/3，不改已有 Subtask 边界/动作文本，也不承担视频动作标注纠错。

## 1. 三个阶段的实际实现

| 阶段 | 输入 | 视频/图片具体采样 | 单个推理单元 | 输出 |
|---|---|---|---|---|
| **Stage A**：全局任务与视觉关注规划 | Task、完整 Stage3 Subtasks、全局图像 | **每个 Subtask 的起始边界帧、结束边界帧 + 内部 1 FPS 候选帧均匀选最多 4 张** | 一个 Episode | `stage_a.json`：全局目标、约束、目标组、逐 Subtask Stage B 关注点 |
| **Stage B**：相邻操作规划依据 | Stage A 输出、**前一 Subtask + 当前 Subtask** 的文本/图像 | **两段各自 2 FPS**，含边界帧，重合帧只编码一次；第一个 Subtask 无前驱 | 一个相邻 Subtask 对（S1 首段单独） | `stage_b/subtask_XXXXXX.json`：为什么接着做、关联状态、规划证据、成功条件 |
| **Stage C**：纯文本整合审核 | Task、所有 Stage3 Subtasks、Stage A/B 所有输出 | **不读视频、不传任何图片** | 一个 Episode | `stage_c.json`、`planner.json`：与每段一一对应的 Planner 标注 |

**重要：Stage B 并非视频文本标注纠错器。** 它使用前一段与当前段场景信息，解释当前操作和任务目标、场景、上一段结果之间的规划关联。它不能声称知道示范机器人的真实内部推理。

### 三个独立 batch size

`config.example.yaml` 中：

```yaml
stage_a_batch_size: 2    # 同时处理 2 个 Episode 的 Stage A
stage_b_batch_size: 4    # 同时处理 4 组相邻 Subtask 对的 Stage B
stage_c_batch_size: 2    # 同时处理 2 个 Episode 的 Stage C
```

这三个是 **最大并行任务数**（in-flight inference units），不是一个请求中打包几个样本，也不是帧数。模型推理仍为单 Episode / 单相邻对的 JSON 输出，batch size 不改变 Prompt、FPS、时间窗口和结果一一对应关系。

对单个任务进行多次模型调用的情况：当 A 的 Episode 全部图片超过 `stage_a_max_images_per_request`、或 B 的一对相邻 Subtask 的图片超过 `stage_b_max_images_per_request`，框架保留**全部已抽帧图像与指定 1/2 FPS**，按时间分批 VLM 观察后做一次纯文本归纳。上述 A/B 的拆分归纳均被纳入各自批次并发控制。**不会偷偷降采样、丢后半段视频或混合不同 Episode 的图像。**

## 2. 安装

Python 3.10+，建议新建环境：

```bash
conda create -n vla-planner python=3.11 -y
conda activate vla-planner
pip install -e .
```

`opencv-python-headless` 用于读取源视频和按帧索引抽帧，`httpx` 直接调用 OpenAI 兼容的 `POST /v1/chat/completions`。支持典型 vLLM / SGLang 模型服务；是否支持 `response_format=json_object` 取决于后端，可在 YAML 中按阶段设置 `json_mode`。

设置服务地址与模型名称：

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml 里的 api.base_url 以及 models.stage_a/b/c.model
export OPENAI_API_KEY=EMPTY     # 本地无需鉴权的服务可用 EMPTY
```

示例配置默认 Stage A/B 视觉模型、Stage C 文本模型；请改成自己 `/v1/models` 返回的真实模型 ID。如果三个阶段运行在不同的 vLLM/SGLang 服务（不同端口），可在 `models.stage_a`、`models.stage_b`、`models.stage_c` 下分别配置 `base_url`（以及可选的 `api_key_env`、`timeout_seconds`），会覆盖全局 `api` 中的对应项。请求超时、重试、温度、max_tokens、后端专用 `extra_body` 均可配置。

## 3. 输入数据格式

支持单个 Episode JSON、多个 Episode 的 manifest JSON（根字段 `episodes`），或包含多个 Episode JSON 的目录。

```json
{
  "episode_id": "episode_000027",
  "task": "将纸箱搬运到托盘上",
  "video_path": "/data/episode_000027.mp4",
  "annotation_fps": 1,
  "instruction_expansion": "可选的指令扩写，不应当成确定事实",
  "subtasks": [
    {
      "subtask_index": 1,
      "start_frame_index": 0,
      "end_frame_index": 30,
      "segment_summary": "抓取纸箱",
      "arm_1": {"description": "...", "outcome": "..."},
      "arm_2": {"description": "...", "outcome": "..."},
      "boundary_evidence": {}
    }
  ]
}
```

**必须区分 Stage3 的时间索引帧率和原视频帧率。** 上例 `annotation_fps: 1` 意味着 `start_frame_index: 30` 是 30 秒。若 Stage3 帧号来自 25 FPS 原视频，设 `annotation_fps: 25`。如果已知精确时间，更推荐每段直接提供 `start_sec`、`end_sec`（单位秒），此时不需要 `annotation_fps`。

也支持 `start_time_sec`/`end_time_sec`、`start_time`/`end_time` 和 `time_sec: [start,end]`。未识别的原始标注字段会完整原样保留在 Prompt 中。`video_path` 相对路径相对于该 JSON 所在目录。

对于不同的 Stage3 JSON schema，可用一个预处理脚本把每个 Episode 包成上面的标准结构；原则是保留完整 `subtasks` 文本原字段，不要预先生成 Planner 的结果。

## 4. 运行

先生成小型测试视频和配套 JSON（**仅用于接口与抽帧冒烟测试，不作为真实标注质量依据**）：

```bash
python examples/create_demo.py
vla-planner dry-run --input examples/episode.json --config config.yaml --output ./outputs
```

`dry-run` **不调用模型**，只读取原视频元数据、计算 A/B 将抽取的帧数与 B 的相邻配对，保存 `outputs/_dry_run_plan.json`。

实际运行：

```bash
vla-planner run \
  --input /data/planner_input/ \
  --config config.yaml \
  --output /data/planner_outputs/
```

按需覆盖 batch size：

```bash
vla-planner run --input /data/planner_input/ --config config.yaml --output /data/planner_outputs/ \
  --stage-a-batch-size 2 --stage-b-batch-size 8 --stage-c-batch-size 4
```

单独运行阶段（依赖阶段必须已有有效缓存）：

```bash
vla-planner run --input examples/episode.json --config config.yaml --output outputs --stage a
vla-planner run --input examples/episode.json --config config.yaml --output outputs --stage b
vla-planner run --input examples/episode.json --config config.yaml --output outputs --stage c
```

`--force` 强制重跑所选阶段，不使用已有缓存。默认按 `Stage3 文本 + 源视频的路径/大小/修改时间 + 当前阶段模型/采样配置 + Prompt` 的摘要做断点续跑；只改变 batch size 不会触发不必要的重新推理。如果前一阶段结果或本阶段配置改变，下游缓存失效并重算。

## 5. 输出文件

```text
outputs/
  _run_summary.json
  _dry_run_plan.json
  episode_000027_<hash>/
    stage_a.json              # _meta + result
    stage_b/
      subtask_000001.json     # 对首段的任务启动依据
      subtask_000002.json     # S1 -> S2 的任务规划依据
      subtask_000003.json     # S2 -> S3 的任务规划依据
    stage_c.json              # _meta + result
    planner.json              # 直接可用的纯 Planner JSON
```

`planner.json` 包含：

- `episode_id`、`task_goal`；
- `planner_annotations`：每个 `subtask_index` 恰好一条，含 `goal_id`、`previous_state`、`remaining_goal`、`motivation`、`decision_basis`、`visual_basis`、`decision`、`support_level`、`uncertainty`、`evidence_frame_refs`；
- `decision`：保留参考 HTML 中的 `type`、`instruction`、`success_criteria`；
- `global_review`：Planner 依据自身的全局一致性审核，不是 Stage3 动作纠错报告。

`evidence_frame_refs` 中的 `f120@4.800s` 表示 **原始视频第 120 帧、源视频时间 4.8 秒**，不是 Stage3 采样索引。模型只应引用 Prompt 中真实传入的帧 ID。

## 6. 质量约束与界限

- A 看的是全局稀疏图像（每段边界+内部最多4张）；B 在相邻两段内按原始时间 2 FPS 取帧；C **严格纯文本**。
- 所有核心结果均与原始 `subtask_index` 对齐；模型漏输出/多输出索引时，会报错而**不伪造缺失的结果**。
- 选择某个物体、位置或顺序的内在理由不一定能从示范里唯一推断。区分 `supported`、`plausible`、`undetermined`，禁止伪称机器人采用最优策略。
- Stage A 可以总结完整任务，但 B/C 不能把未来操作当成当前动作之前的已知条件。
- OpenAI 兼容服务要求支持图像 `image_url`，本工程没有启动或部署 VLM 的代码。开始正式标注前需要提供真实服务及真实 Episode/Stage3 标注。
- 长视频分批会增加一个归纳步骤；每段仍在原采样率下完整覆盖。视频抽帧使用 CPU 解码，速度与视频编码格式/GOP 结构有关。

## 7. 测试

```bash
pip install -e '.[test]'
pytest -q
```

测试包含：Stage A 帧数上限、Stage B 2 FPS 与相邻边界去重、索引/时间基准、端到端三阶段执行、分阶段缓存、以及三个独立 batch size 的并发上限。
