# AI 辅助 ADS 仿真与独立复核记录

记录日期：2026 年 11 月 22 日

本记录用于准备最终提交中的 AI-use reflection。它将作者提供的使用经历与项目文件中可核对的验证结果分开记录。

## 作者确认的 AI 使用经历

作者使用 AI 辅助操作 ADS 的 Python Console，实现批量化、并行仿真。

作者还使用 AI 帮助查找并解释所比较的两种仿真器的算法逻辑与数学基础，用于理解它们的求解方式、建模假设以及仿真结果差异。

作者希望 reflection 围绕以下复核过程展开：使用 openEMS 对模型相关结果进行全波复核，发现电路模型中成立的调谐结果在全波模型中未必成立，再结合仿真软件的算法说明分析差异。

当前未记录具体 AI 产品、模型版本或完整提示词，不补写未经确认的工具名称，也不将其他环节自动归为 AI 完成。

## 可以支撑反思的项目验证记录

以下内容由现有代码和结果提供依据，描述的是项目已保存的检查，不额外推断每一步由谁操作。

- `scripts/ads_sim.py` 实现 ADS 批量仿真，包括并发任务、结果保存及恢复运行的支持。
- `docs/algorithm_plan_v2.md` 的第 3 步记录了 4,788 次独立仿真，以及并发许可证拒绝后的单进程补跑。该数量属于该阶段记录，不代表整个项目全部仿真的总数。
- `results/step3/ads_eval.json` 记录完整联盟的响应与对应数据集响应一致，最大复数 S 参数差异为 0。此项支持 ADS 批处理流程在这些检查案例中的可复现性。
- `results/fullwave_openems/report.json` 记录三个在 ADS 中通过的 SHAP 调谐设计，在 openEMS 复核中均未通过原规格掩模。其中两个案例改善，一个案例恶化。
- `docs/reference/shap_lime_filter_paper.txt`、求解器算法说明以及网格研究可以用于说明比较背景。数值方法差异提供解释框架，具体误差来源仍需结合网格、端口、边界条件和带宽定义分析。

## Reflection 的核心论点

AI 帮助自动化了重复的仿真执行工作，使批量实验和解释评估能够规模化进行。自动化流程运行成功之后，仍需要检查网表参数、求解器输出、缓存与失败任务，并通过已知案例的一致性验证流程。

AI 也参与了技术资料检索与原理解释，帮助作者理解两种仿真器的算法和数学基础。这部分帮助构成分析结果差异的理解框架，具体解释仍需与求解器资料及实际数值结果对应。

同一电路求解器中的重复检查主要检验执行与数据一致性。openEMS 的独立全波复核进一步检验调谐结果能否在另一建模层级下保持成立。三个案例的结果提示：AI 辅助建立的高效工作流，以及代理解释与 ADS 的高度一致，都不足以单独保证工程建议在全波模型下通过规格。

最终反思可以强调，作者对自动化输出的采纳建立在检查和独立复核之上。openEMS 在这里属于独立全波复核或压力测试；它不是用于模型训练、早停或超参数选择的 validation split。

## 可用于最终英文反思的段落

I used AI assistance to operate the ADS Python Console and automate batch and parallel simulations. I also used AI to help locate and explain technical descriptions of the algorithms and underlying mathematics of the two simulators being compared. This supported my understanding of their solution methods and modeling assumptions when interpreting differences between their results. Batch automation made repeated simulator evaluations more manageable, but successful execution alone did not establish that the resulting design advice was reliable. The project therefore included checks against existing ADS responses and a separate full-wave evaluation using openEMS. In the three cases examined, designs that passed the specification mask after tuning in ADS did not pass in openEMS. I interpreted this discrepancy using the solvers' numerical-method descriptions alongside the recorded mesh and response comparisons. This experience showed the value of combining AI-assisted automation and technical explanation with independent verification: AI supported execution and understanding, while cross-model testing exposed a limitation that agreement within the circuit model could not reveal.
