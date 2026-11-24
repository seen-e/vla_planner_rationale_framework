"""Chinese prompts; output keys English, descriptions Chinese."""
from __future__ import annotations

from typing import Any
import base64
import json

from .dataset import Episode, Subtask
from .sampling import SampledFrame


def pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


SYSTEM = """你是 VLA 机器人高层 Planner Rationale 标注员。任务是对已经完成的 Stage3 Subtask 轨迹进行回顾性任务规划依据标注，而不是修改、纠错或重新切分视频动作标注。
必须严格区分：原始指令中的任务约束、Stage3 的动作描述、视频中可见的场景事实、以及合理但无法证实的事后推断。
不声称知道机器人真实内部思考，不虚构最优策略、效率、安全性、成功率等原因；证据不足时输出 unknown/undetermined。
不能将当前动作结束后的结果或未来动作视为当前动作开始前已知的决策依据。
图像的 frame_index、f<frame>@<sec>s 和 timestamp 均为 episode 内的编号与时间；source_frame_index 仅用于定位共享源视频。旧标注中的 sample 引用若无明确映射，不得视为当前图像编号。
审核通过不代表任务必然成功。原标注保留的失败或不确定性必须保留，不得为解释规划动机而补写成功结果。
只输出合法 JSON 对象，不要 Markdown、不要思考过程。JSON key 使用英文、描述使用中文。"""


A_DIRECT = """【Stage A：全局任务理解与 Stage B 关注点规划】
你将得到整个 Episode 的原始任务、Stage3 全部 Subtask 文本，以及从每个 Subtask 边界帧和内部采样的图像。
目的：建立任务级概览，判断每段的任务角色与相邻段的规划联系，给 Stage B 生成问题清单。不是动作标注质量审核。
要求：
1. task_goal 和 task_constraints 只从原始 Task 的明确含义提取；instruction_expansion 只能作为假设，不是事实。
2. episode_overview 可总结完整轨迹，但不要把后续轨迹当成前面决策时的已知事实。
3. goal_groups 允许多个 Subtask 共用一个 goal_id；不同物体的任务可区分。
4. 对【每个】原始 subtask_index 生成一条 subtask_plans，不能增删索引。
5. planning_questions：重点解释为什么完成前一段后要执行当前段。visual_focus 只写与规划动机有关的场景布局、对象/目标空间关系、顺序约束等，不把 Stage B 变成动作核验纠错。
6. 视觉不可确定的选择原因标注 uncertain，不要推测效率最优等。
输出严格 JSON：
{
  "task_goal":"...",
  "task_constraints":["..."],
  "episode_overview":"...",
  "goal_groups":[{"goal_id":"goal_1","subtask_indices":[1,2],"purpose":"..."}],
  "subtask_plans":[{
     "subtask_index":1,"goal_id":"goal_1", "role":"initiate|continue|switch|adjust|wait|finish|other|unknown",
     "planning_questions":["为什么此时执行这一步？"],
     "visual_focus":["..."],"uncertainties":["..."]
  }]
}
这里的 index 只是示例，请使用输入中真实的 subtask_index。"""


A_CHUNK = """【Stage A 的分批视觉观察】
因为本 Episode 图像较多，以下只是一组连续 subtask 的抽帧图像。
仅提取与任务规划相关的可见场景：操作对象、目标区域、空间关系、当前操作与前后任务关联的可见线索。
不要纠错动作文本，不要编造任务策略，不要生成最终 Planner Rationale。
输出 JSON：{"visual_observations":[{"subtask_index":1,"facts":["..."],"uncertainties":["..."]}]}
请覆盖本组全部 subtask_index。"""


A_MERGE = """【Stage A：利用已提取的视觉事实生成全局任务与观察计划】
以下包含完整 Stage3 文本，以及对完整 Episode 稀疏图像的分批视觉观察。
遵循 Stage A 的相同要求，输出 task_goal, task_constraints, episode_overview, goal_groups, subtask_plans。
每个真实 subtask_index 必须恰好出现一次。只能将 Task 明确内容写入 task_constraints。
仅根据已提供的视觉摘要推断场景关系；不能假称看到了未提供的细节。"""


B_DIRECT = """【Stage B：相邻 Subtask 的任务规划依据推导】
根据 Stage A 的全局目标和关注计划、前一个与当前 Subtask 的 Stage3 文本、两段视频的 2 FPS 抽帧，重点回答：
**为什么在前一个 Subtask 之后，接着执行当前 Subtask？当前操作如何推进任务目标？**
注意：
- 这是回顾性规划依据解释，不是视频/动作文本纠错，不核验标注是否正确，不修改任何 subtask。
- first subtask 没有前驱时，解释为何启动此任务。
- 关注前一个操作后的任务进度、当前的目标需求、场景布局和对象关系。
- 区分计划目标与实际结果；不能把当前段结束后的结果用作该段开始前的已知条件。
- 对当前动作的目的可作任务逻辑解释，但对"为什么选择某一个具体对象/路径/位置"，没有约束与视频线索就说明无法唯一确定。
- success_criteria 是完成当前操作应满足的条件，不等于视频真实结果。
- 仅依据可见事实和给定文本；不要声称机器人采用最优策略，不要输出冗长泛化的动机。
- evidence_frame_refs 只引用输入图像对应的 f<frame>@<sec>s 标识；若无直接证据，用空数组。
输出 JSON：
{
 "subtask_index":1,
 "goal_id":"...",
 "previous_state":"上一步后的可用任务状态/若首段则初始状态",
 "remaining_goal":"...",
 "motivation":"为什么此时执行当前操作",
 "decision_basis":"当前操作与前一状态、任务目标的因果连接",
 "visual_basis":"与规划决策有关的可见场景线索，不确定可写无法确认",
 "causal_relation":"initiate|continue|switch|adjust|retry|wait|cooperate|finish|unknown",
 "success_criteria":"...",
 "evidence_frame_refs":["f120@4.800s"],
 "support_level":"supported|plausible|undetermined",
 "uncertainty":"..."
}
使用输入中真实的 current subtask_index；不要增加其他 subtask。"""


B_CHUNK = """【Stage B：相邻操作的局部视觉事实摘录】
因为相邻视频的 2 FPS 图像超过单次图片预算，以下是其中按时间排序的一部分图像。
只记录对相邻 Subtask 的任务规划有用的场景事实，例如操作前后对象与目标的空间关系、任务进展、可见约束。不要重新标注动作，也不要判断 Stage3 是否标错。
不要单独生成完整动机，后续会将所有图像块的事实合并推导。
输出 JSON：{"visual_facts":["..."],"evidence_frame_refs":["..."],"uncertainties":["..."]}。只引用本批图像中出现的帧 ID。"""


B_MERGE = """【Stage B：汇总相邻操作的 2 FPS 分批视觉观察】
输入包含完整相邻 Subtask 文本、Stage A 的规划上下文、以及两段 2 FPS 图像各批次的可见场景事实。
回答为什么前一段之后执行当前段；不要做动作标注纠错，不把后续结果当成当前执行前已经知道的信息。
严格输出与 Stage B 单次视觉分析相同的 JSON schema；给出可证实的因果解释，理由不充分则降级 support_level。"""


C_FINAL = """【Stage C：纯文本 Planner Rationale 全局整合与审核】
本阶段只使用纯文本输入，不得要求视频。已给定 Task、Stage3 全部 Subtask、Stage A 全局目标与规划、Stage B 对每个 Subtask 的推导。
请整合得到整个 Episode 的最终 Planner 标注，并审核不同 Subtask 的目标与因果逻辑是否连续。不要纠错或修改 Stage3 已有的动作、对象、边界。
要求：
1. 一条 planner_annotations 对应一个原始 subtask_index，不增删，不重新编号。
2. 保留 HTML Planner 风格的 decision.type / decision.instruction / decision.success_criteria。
3. motivation 说明为什么此时做；decision_basis 描述由前一状态向当前操作的逻辑连接。
4. 若无法说明具体位置/对象选择的独特原因，不能编造；标注 plausible/undetermined。
5. 多段可共享 goal_id，区分子步骤结束与目标完成；只有确实属于任务终结的阶段才使用 task_complete。
6. success_criteria 为预期目标，不是实际结果；只据文字提供可观察的完成条件。
7. 严格避免把未来状态写成过去决策时已知信息。
8. global_review.issues 只记录 Planner 依据自身的因果/目标矛盾与证据不足，不把这项工作变成视频标注错误筛查。
9. 每条 evidence_frame_refs 只能从 allowed_evidence_frame_refs_by_subtask 中该 subtask_index 对应的列表逐字选取；不能新增、换算帧号，不能从原Stage3文本、其他subtask或自由描述中提取新引用。无合适引用时输出空数组。该列表才是本阶段可以引用的已验证证据帧集合。
严格输出 JSON：
{
 "episode_id":"...",
 "task_goal":"...",
 "planner_annotations":[{
   "subtask_index":1,
   "goal_id":"...",
   "motivation":"...",
   "previous_state":"...",
   "remaining_goal":"...",
   "decision_basis":"...",
   "visual_basis":"...",
   "decision":{
      "type":"next_subtask|task_complete",
      "instruction":"对当前 subtask 的简洁动作指令（包含必要任务依据）",
      "success_criteria":"..."
   },
   "support_level":"supported|plausible|undetermined",
   "uncertainty":"...",
   "evidence_frame_refs":[]
 }],
 "global_review":{"status":"passed|needs_review","issues":["..."]}
}
全部字段请结合真实输入填写。"""


def episode_context(ep: Episode) -> dict[str, Any]:
    return ep.summary()


def pair_context(ep: Episode, a_output: dict[str, Any], previous: Subtask | None,
                 current: Subtask) -> dict[str, Any]:
    plans = a_output.get("subtask_plans") or []
    plans = [p for p in plans if p.get("subtask_index") in
             ({current.index} if previous is None else {previous.index, current.index})]
    return {
        "episode_id": ep.episode_id,
        "task": ep.task,
        "task_goal": a_output.get("task_goal"),
        "task_constraints": a_output.get("task_constraints", []),
        "goal_groups": a_output.get("goal_groups", []),
        "stage_a_subtask_plans": plans,
        "previous_subtask": None if previous is None else {
            "subtask_index": previous.index,
            "start_sec": previous.start_sec, "end_sec": previous.end_sec,
            "annotation": previous.annotation},
        "current_subtask": {
            "subtask_index": current.index,
            "start_sec": current.start_sec, "end_sec": current.end_sec,
            "annotation": current.annotation},
    }


def visual_messages(system_instruction: str, text_payload: str,
                    frames: list[SampledFrame]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text_payload}]
    for frame in frames:
        req = frame.request
        content.append({"type": "text", "text":
                        f"[IMAGE {req.ref}] role={req.label}; episode_frame_index={req.frame_index}; episode_time_sec={req.time_sec:.3f}; source_frame_index={req.source_frame_index if req.source_frame_index is not None else req.frame_index}"})
        uri = "data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return [{"role": "system", "content": SYSTEM + "\n" + system_instruction},
            {"role": "user", "content": content}]


def text_messages(system_instruction: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM + "\n" + system_instruction},
            {"role": "user", "content": pretty(payload)}]
