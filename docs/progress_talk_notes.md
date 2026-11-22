# Progress Talk: Explainable Surrogate Modeling for Microwave Filter Design

> 当前作业要求为 5 分钟、stakeholder evidence 与首个 faithfulness result。请使用 [progress_talk_5min.md](progress_talk_5min.md) 作为当前主稿；本文件保留为较长的技术备用材料。

准备日期：2026-11-05。默认场景：DSCI498 课程进展汇报，约 9 分 30 秒，9 页。英文用于 slide，中文用于讲解准备。以下数字引用项目已保存的报告，本次未重跑训练、XAI 或 ADS。

## 汇报主线

Can explanations of a learned filter surrogate be faithful to the model, consistent with the simulator, and useful for tuning?

不要把故事写成“训练模型，然后画 SHAP 图”。主线是：验证预测 → 验证解释 → 核对 ADS → 检验调谐用途。

## Slide 1 — Motivation and Research Questions（40 秒）

### Slide text

- Fast surrogate predictions can support filter design.
- Prediction accuracy alone does not validate an explanation.
- RQ1: Is the explanation faithful to the surrogate?
- RQ2: Does it agree with ADS-based reference attributions?
- RQ3: Does explanation-based parameter selection help tuning?

### 讲解稿

我的项目研究微带带通滤波器的可解释代理建模。代理模型能够快速预测频率响应，但预测准，不代表我们已经知道模型依据什么作出判断。我把这个问题拆成三个层次：解释是否忠实于模型、模型归因是否与 ADS 一致，以及这些解释是否真的能帮助设计调谐。这样，当某个解释表现不好时，可以进一步判断问题来自模型、解释方法，还是解释到调谐的使用方式。

### 这一页希望听众理解

研究贡献是对解释进行分层验证，而不只是比较模型误差或画重要性排名。

### 建议画面

Geometry → Surrogate → Response；下方标出 explanation、ADS check、tuning 三个验证环节。

## Slide 2 — Dataset and Experimental Scope（50 秒）

### Slide text

- 1,440 ADS circuit-simulated designs.
- Six geometry variables: L1, L2, S1, S2, Wc1, Wc2.
- Three negative-control inputs.
- Core response: 9–15 GHz, 601 frequency points, complex S21 and S11.
- Design-level split: 1,008 train / 216 validation / 216 test.
- Train-only normalization.

### 讲解稿

目前的数据包含 1440 个设计，每个设计有六个可调几何参数，对应 9 到 15 GHz 的 S 参数曲线。我另外加入三个不参与 ADS 几何仿真的负对照输入，用于观察模型是否依赖无关变量，以及解释器能否反映这种依赖。划分以完整设计为单位，同一个设计的频点不会分散到训练集和测试集。这里的物理参考来自 ADS 电路仿真，尚不能表述为全波或实测验证。

### 这一页希望听众理解

负对照是可检验的诊断工具。若模型实际使用了噪声，解释器正确显示噪声的重要性，也是忠实解释。

### 来源

- ../README.md
- training_xai_protocol_v2.json

## Slide 3 — Surrogate Models and Prediction Accuracy（55 秒）

### Slide text

- PR-Net: geometry → shared poles and residues → complex response.
- M1: frequency-conditioned complex-output MLP.
- M0: literature-inspired direct dB-output baseline.
- B1: PCA + Gaussian process regression.
- PR-Net provides an interpretable intermediate representation.

| Test metric | PR-Net | M1 | M0 |
|---|---:|---:|---:|
| S21 full-curve MAE, dB | 0.084 | 0.021 | 0.266 |
| S11 full-curve MAE, dB | 0.078 | 0.025 | 0.811 |
| Band-edge MAE, MHz | 3.220 | 0.585 | 42.258 |

注：216 个 test 设计；表中为三个训练种子的均值。带边沿仅在真值单通带设计上评价。

### 讲解稿

我比较了四类代理模型。PR-Net 的特点是先预测极点和留数，再重建响应，因此可以进一步解释几何如何影响谐振频率和 Q。M1 是没有这层结构的黑箱对照，M0 是对文献网络结构的适配，B1 是传统机器学习基线。从测试精度看，M1 比 PR-Net 更准确；我保留 PR-Net 为主模型，是因为它通过了预设精度门槛，并且能够支持概念层解释，而不是因为它精度最高。

### 这一页希望听众理解

区分“预测最准确的模型”和“适合研究概念解释的模型”；不要声称 PR-Net 在所有指标上最优。

### 来源

- ../results/step3/step3_report.md，test 精度表

## Slide 4 — What Do SHAP and LIME Explain?（90 秒）

### Slide text

We explain a scalar target F(x), rather than the entire curve at once.

- Targets: frequency-point response, passband metrics, mask violation, and pole concepts.
- SHAP: allocate F(x) − F(reference) across the input features.
- LIME: approximate F near the current design using a weighted linear model.
- IG: integrate gradients along the reference-to-design path.

SHAP: F(x) − F(reference) = Σ φ_j

LIME: F(z) ≈ β_0 + Σ β_j (z_j − x_j)

LIME-delta: β_j (x_j − reference_j)

### 讲解稿

首先，我解释的是一个明确的工程目标，比如 mask 违反量，而不是含糊地问“哪个参数最重要”。SHAP 问的是：当前设计相对于基线的预测差异，应该怎样分摊给各个输入。这里九个特征可以完整枚举 512 个组合，所以主版本使用精确 Shapley 计算。

LIME 问的是另一个问题：在当前设计附近，改变哪个输入会使预测变化较大。它扰动附近的设计、查询模型，再拟合一个距离加权的线性近似。斜率是局部敏感度；我还用斜率乘上与基线的距离，得到 LIME-delta，便于与归因量比较。这个乘积不是精确分解，在强非线性区域不保证与 SHAP 相同。

对于 mask 违反量，正的 SHAP 值表示相对于基线增加了预测违反量，负值表示减少。它不直接表示“把参数增大就更差”；局部增减方向应看梯度或 LIME 斜率。

### 这一页希望听众理解

相同设计可以有不同目标的解释；贡献符号与局部导数符号不同；SHAP 与 LIME 的差异不自动表示其中一个算法出错。

### 建议画面

左侧：基线 → 当前预测的 waterfall；右侧：当前设计附近的采样点和局部拟合线。若使用示意数字，必须标明 illustrative。

### 来源

- ../src/filterxai/xai.py
- ../src/filterxai/targets.py
- ../scripts/step2_xai.py

## Slide 5 — Validation Beyond Explainer Agreement（60 秒）

### Slide text

- Internal checks: additivity, IG completeness, LIME held-out R².
- Negative controls and intervention sensitivity.
- Stability across seeds, backgrounds, and neighborhoods.
- ADS reference game: six geometry variables, 64 coalitions per design.
- 40 validation + 40 test designs for attribution comparisons.

Explanation → Surrogate attribution → ADS reference attribution

### 讲解稿

两个解释器给出相似排名，只能说明它们意见相近，不能单独说明它们符合物理。因此我既做内部检查，也直接在 ADS 上计算参考归因。对于六个几何参数，每个设计需要 64 个组合；缺席参数都取相同 ADS 基线。这样可以分别检查解释器和模型之间的差异，以及模型和仿真器之间的差异。比较时需要匹配目标、参考点和参数集合。九特征解释的几何部分与六特征 ADS 博弈仍有负对照处理的区别，所以项目另算了代理模型的六特征博弈来分解误差。

### 这一页希望听众理解

这里的 ADS 参考是真实运行仿真器得到的、在指定博弈下的归因；它不等于参考点无关的唯一物理因果真值。

### 来源

- ../results/step3/step3_report.md
- ../scripts/step3_ads_eval.py

## Slide 6 — Current Results: Attribution Reliability（80 秒）

### Slide text

PR-Net, test designs, nonlinear T2 targets:

| Attribution comparison | Mean rank agreement |
|---|---:|
| Surrogate six-feature game vs ADS | 0.99 |
| SHAP geometry attribution vs ADS | 0.99 |
| IG vs ADS | 0.82 |
| LIME-delta vs ADS | 0.84 |

- Accurate response prediction supports accurate simulator-reference attribution.
- Local sensitivity and baseline-relative attribution are different evaluation tasks.
- Explainer agreement alone is insufficient evidence of physical agreement.

注：指标为六个几何变量上绝对归因的 Spearman；40 个 test 被解释设计，按报告对设计、三个 T2 目标和三个训练种子汇总。不是“99% 的解释正确”。

### 讲解稿

在测试集的非线性 T2 目标上，PR-Net 的六特征归因与 ADS 参考排序相关约为 0.99。SHAP 的几何归因也接近这个水平。IG 和 LIME-delta 分别约为 0.82 和 0.84。

这里不能简单说 LIME 没有用，因为它原本拟合的是局部关系，而 ADS Shapley 分摊的是相对基线的变化。作为另一个检验，PR-Net 的 LIME 斜率与小步长 ADS 有限差分，在测试集上的排序相关约为 0.92，符号一致率约为 0.98。因此它在局部敏感度这个任务上表现更好。模型正确、解释问题定义匹配、以及数值近似可靠，需要分开讨论。

### 这一页希望听众理解

不要从当前 Shapley 参考评价推出“SHAP 在所有解释任务上优于 LIME”。精确枚举消除了联盟抽样误差，但没有消除代理模型误差或参考点选择的影响。

### 来源

- ../results/step3/step3_report.md，test 分解表与局部有限差分表

## Slide 7 — What Engineering Interpretation Do We Obtain?（65 秒）

### Slide text

- Centroid target: L2 > L1 > Wc2.
- Nonlinear T2 targets: L2 > L1 > S2 in ADS global attribution rankings.
- At 12 GHz, L1 can become the leading contributor.
- Middle resonant-mode Q3: S1 and Wc1 are prominent.
- Interpretation depends on the target, design, and reference.

### 讲解稿

我希望解释给出的不只是一个统一排名，而是不同工程指标对应哪些几何因素。ADS 参考结果显示，通带质心目标的全局前三是 L2、L1、Wc2；几个非线性通带目标通常是 L2、L1、S2。但如果解释 12 GHz 的响应，L1 可以排在最前面。概念层又给出不同信息：中间谐振模态 Q3 的重要参数是 S1 和 Wc1。

这些结果支持不同几何变量在频率位置、耦合和响应形状上承担不同作用的解释方向。但排名本身不告诉我们增大参数后一定往哪个方向变化，也不能直接把每个拟合极点等同于某个具体金属谐振器；方向和更细的物理对应关系需要额外干预或场分布验证。

### 这一页希望听众理解

解释的工程价值是形成可验证的目标相关诊断，而不是给所有频率、所有设计套同一条参数口诀。

### 来源

- ../results/step3/step3_report.md，ADS 全局重要性表
- reference/baseline_review.md，物理假设说明

## Slide 8 — Does Explanation-Based Selection Help Tuning?（90 秒）

### Slide text

30 initially failing, single-band test designs; ADS-confirmed success:

| Method | Successful designs |
|---|---:|
| SHAP-selected two parameters | 25/30 |
| IG-selected two parameters | 25/30 |
| LIME-delta-selected two parameters | 24/30 |
| One random pair per case | 6/30 |
| Fixed L1 + L2 — exploratory comparison | 21/30 |
| All six parameters, surrogate optimization | 30/30 |

- Explanations help select a useful restricted search space.
- Better attribution agreement did not yield clearly better tuning success.
- Full-parameter surrogate optimization achieved the highest observed success.

### 讲解稿

调谐部分，我从 30 个初始未通过规格的单通带测试设计出发。解释方法选择两个参数，代理模型负责搜索新值，最后全部用 ADS 确认。SHAP 和 IG 分别修好了 25 个，LIME-delta 修好了 24 个，一次随机选参只修好了 6 个。

但我也加入了更强对照。事后探索中，固定调 L1 和 L2 就能修好 21 个；穷举全部 15 个参数对得到的随机成功数精确期望为 8.3，而不是那次随机抽样的 6。因此解释相对随机的收益，有相当一部分来自识别了总体重要参数。逐设计解释在固定 L1/L2 之上的增益约为三到四个案例。这个固定参数对比较是探索性的，不把它表述成新的确认性显著结论。

同时，直接在代理模型上优化六个参数可以 30 个全部通过。因此目前得到的结论是：解释有助于限制需要调整的参数数量，但没有证据表明它提高了相对于六参数优化的成功率。解释“当前偏离由谁贡献”和寻找“哪些参数可以修好它”，是相关但不同的问题。

### 这一页希望听众理解

归因质量和决策效果需要分别验证。不要把 25/30 与 24/30 宣传为解释器之间已存在显著优劣。

### 来源

- ../results/step4/step4_report.md
- ../results/step4b/step4b_report.md，明确为事后探索性分析

## Slide 9 — Progress, Limitations, and Next Steps（40 秒）

### Slide text

Completed:

- Dataset pipeline and surrogate baselines.
- SHAP / IG / LIME evaluation and ADS reference comparisons.
- ADS-confirmed tuning experiments and exploratory controls.

Next:

- Audit metric edge cases and artifact provenance.
- Present representative success and failure cases.
- Test explanation value under practical parameter constraints.
- Extend validation beyond the current ADS circuit-simulation setting.

### 讲解稿

目前数据、训练、归因评价和 ADS 调谐实验已经跑通。下一步首先补齐指标边界处理和实验版本追溯，再把典型成功和失败设计整理成清晰案例。在研究问题上，我想进一步测试：如果某些参数不可改、可调整数量有限，或制造代价不同，解释的价值是否更明显。当前所有结论都限定在这套 ADS 电路仿真、参数范围和规格目标内；全波或实测验证仍是后续工作。

### 结束句（英文）

The main finding so far is that attribution agreement, simulator consistency, and tuning utility need to be evaluated separately. Explanations can help identify a small set of useful parameters, but their practical benefit depends on the design constraints.

## 预期解释、实际观察、可支持的结论

| 原先可检验的预期 | 当前观察 | 汇报时的解释 |
|---|---|---|
| 无关输入应当得到较小重要性 | PR-Net/M1 的实际负对照干预占比约不超过 1%；M0 约 4–8% | 先核查模型是否使用噪声，再判断解释是否忠实；不能要求解释器替模型隐藏噪声依赖 |
| 长度类参数可能主导频率位置 | 质心的 ADS 全局前三为 L2、L1、Wc2 | 与物理假设相容，但排序与符号、局部方向是不同信息 |
| 耦合相关几何可能影响模式 Q | Q3 的 ADS 排名突出 S1 和 Wc1 | 为概念解释提供可核对证据；不从归因单独推出唯一电磁机理 |
| 不同解释器在简单目标上更接近 | T0 接近线性；T2 下 IG/LIME-delta 与 ADS Shapley 差异更大 | 解释器定义和非线性会改变结果；不能只凭互相一致判断正确 |
| 更准确的归因可能带来更好调谐 | SHAP 对 ADS 的归因一致度更高，但成功数与 IG/LIME 接近 | 同一设计可能有多个可行参数对，归因差异不必转化为成功率差异 |
| 解释选参应优于随机选参 | 25/30、25/30、24/30，对比一次随机 6/30 | 支持有用的选参信息；结合固定 L1/L2 21/30 和六参数 30/30，限定收益范围 |

## Q&A 准备

### Why use PR-Net if M1 predicts more accurately?

M1 是更强的纯预测基线。PR-Net 的研究价值在于可核对的极点、留数中间表示；它通过精度门槛后，可以把解释推进到谐振频率和 Q。不要声称它精度最优。

### Does a positive SHAP value mean that increasing that parameter makes the filter worse?

不直接意味着。对于违反量，正 SHAP 表示该设计输入相对于参考值增加了预测违反量；增大参数的局部方向要看导数、LIME 斜率或有限差分。

### Why do SHAP and LIME disagree?

首先确认它们是否解释相同目标、相同参考和相同特征集合。SHAP 分摊参考到当前的差异，LIME 拟合当前附近的关系。再检查局部 R²、邻域大小、采样稳定性和模型非线性。

### Is ADS Shapley the physical ground truth?

它是在当前电路仿真器、当前基线和联盟定义下的参考归因。它支持 simulator consistency；不等于对真实器件的最终因果验证。

### Why not just optimize all six parameters?

如果目标仅为当前设置下的通过率，现有结果支持直接优化六参数。解释的潜在价值在于受限调参、诊断和设计沟通；对约束与代价的收益还需要实验。

### What is the role of a simple linear baseline?

很重要。事后检查中，train OLS 在近似线性的 T0 和 T3 上已经很强，而 T2 的排序相关约 0.59，代理 + 精确 SHAP 约 0.99。这表明复杂解释的增益主要应在非线性目标上讨论。该数字来自 validation+test 共 80 个设计的补充分析，不能与 Slide 6 的纯 test 指标混作同一总体。

### What happens when LIME has poor local R²?

该解释只能视作不充分的线性摘要，应标记拟合质量；必要时调整邻域并报告敏感性。不能在低 R² 情况下把系数当成可靠的局部规律。

## 发表前的口径检查

- 当前报告中的一个原句称“PR-Net test 边沿 MAE 3.2 MHz 大于 P95 4.1 MHz”，文字不成立；本讲稿只引用表中正确数值，不引用该比较句。
- 第一轮 review 发现边沿指标跳过非有限边沿；已核查当前保存的 PR-Net/M1/M0 验证集和 test 预测无缺失边沿，因此未发现这批预测受该问题影响。仍应补充失败计数及明确失败规则。
- freeze.json 记录哈希，但数据入口只检查文件存在。30 个清单文件中当前有 4 个哈希不同：两份方案文件与 step2_xai.py、step2_eval.py。现有记录不能单独证明或否定事前冻结被遵守；需要保存版本和变更依据。
- M0 是对论文结构的适配：输入、输出任务和数据源不同，层宽等设置由本项目补充。避免声称严格复现了原论文全部实验。
- 当前数据近似无损。训练中的功率平衡软惩罚不等于证明完整两端口、所有频率上的无源性或因果性。
- 极点是共享有理函数的谐振模态表征；不要未经验证就宣称每个极点对应一个具体物理谐振器。
- Slide 8 的随机对照、固定参数对和六参数优化应同时展示；主检验与事后探索应明确区分。

## 如果只有 5 分钟

保留 Slide 1、Slide 2/3 合并、Slide 4、Slide 5/6 合并、Slide 8、Slide 9。将详细物理排名、全部模型表和 Q&A 放到备用页。重点保留三个层次的研究问题、SHAP/LIME 的不同语义，以及“解释优于随机但六参数优化更强”的结果。

## 证据索引

1. ../README.md：数据、目录和当前完成状态。
2. algorithm_plan_v2.md：模型、目标、流程与阶段性发现。
3. ../results/step2/xai_report.md：负对照、稳定性与解释器内部评价。
4. ../results/step3/step3_report.md：test 精度、ADS 排名、归因比较及局部有限差分。
5. ../results/step3/controls.md：事后朴素线性基线和 bootstrap 检查。
6. ../results/step4/step4_report.md：30 个 test 案例的调谐结果。
7. ../results/step4b/step4b_report.md：固定参数对、随机期望与穷举等探索性对照。
