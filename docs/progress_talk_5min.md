# Five-minute Progress Talk — Stakeholder Evidence and First Faithfulness Result

日期：2026-11-05。使用 5 页，每页约一分钟。替代此前 9–10 分钟版本作为当前汇报主稿。

当前 stakeholder 材料：项目作者于 2026-11-05 通过网络电话直接访谈了一位下游 RF 接收前端评估/排障从业者。英文编辑版见 [stakeholder_interview_notes_en.md](stakeholder_interview_notes_en.md)。材料是根据作者笔记整理的转述，不是逐字记录，因此可称为 interview findings，但不使用引号、不作为原话引用。

## Slide 1 — Who Is the Explanation For?（60 秒）

英文 slide：

- Stakeholder perspective: receive-side RF front-end evaluation and troubleshooting.
- Start with the affected frequency range, failed specification, and violation magnitude.
- Provide case-level explanation-quality diagnostics.
- Separate diagnostic attribution from validated tuning advice.
- Geometry attribution may support communication with component designers.

中文讲解：

我在 11 月 5 日通过网络电话访谈了一位从事接收侧射频前端评估和排障工作的工程师。这个角色主要通过响应、规格和实测来缩小问题范围，不直接修改微带几何。访谈中提出三个需求：先展示哪个频段、哪项规格、超标多少；提供单个案例的解释质量信息；区分诊断归因与已验证的调参建议。因此，我会先展示响应与规格，再展示 SHAP/LIME，并注明质量诊断和行动边界。几何归因对这个角色更可能用于与器件工程师沟通。这是一位从业者的访谈，不是工具已经提高排障效率的证据。

证据边界：上述需求来自一次单人访谈（网络电话，2026-11-05，作者笔记整理），不代表所有 RF 工程师；访谈前介绍过项目，材料中的项目术语可能来自该介绍。厂商资料支持背景，不证明 XAI 的实际收益。

页脚来源：

- R&S, Understanding error vector magnitude: https://www.rohde-schwarz.com/us/products/test-and-measurement/analyzers/signal-and-spectrum-analyzers/understanding-error-vector-magnitude_258370.html
- ADI, AN-1354: https://www.analog.com/en/resources/app-notes/an-1354.html

英文口播与 slide 摘要见 stakeholder_interview_notes_en.md 的第 7、8 节。没有逐字记录，不使用引号。

## Slide 2 — Task and Three Explanation Methods（60 秒）

英文 slide：

- 1,440 ADS designs; six geometry variables and three negative controls.
- Explain a scalar target: e.g., smooth specification-mask violation.
- SHAP: allocate the prediction difference from a fixed reference.
- IG: integrate model gradients along the reference-to-design path.
- LIME: fit a local weighted linear approximation.

中文讲解：

模型从六个几何参数和三个负对照输入预测 S 参数曲线，数据有 1440 个 ADS 电路仿真设计。我把响应转换成明确的标量，例如规格 mask 违反量，再应用三种解释方法。SHAP 把相对基线的差异分配给输入；九个特征可以枚举 512 个联盟。IG 积分从基线到当前设计路径上的梯度。LIME 在当前设计附近采样并拟合局部线性模型。我另外计算局部斜率乘参考距离的 LIME-delta，以便做贡献量比较，但它不是精确的差异分解。

建议图：Geometry → response → scalar target，下方并列三个解释器。

## Slide 3 — Where and Why Do Methods Disagree?（50 秒）

英文 slide：

- Different questions: reference-relative contribution vs local sensitivity.
- Nonlinear response and interactions can produce different rankings.
- IG can be numerically unstable near deep dB notches.
- Check LIME local R² and sensitivity to neighborhood choice.

中文讲解：

在 PR-Net 的验证集 T2 目标上，SHAP 与 IG 的绝对归因排序相关约为 0.93，与 LIME-delta 约为 0.85。这些分歧有不同来源：SHAP 考虑不同联盟中的边际变化，IG 沿特定路径分配变化，而 LIME 近似当前附近的关系，非线性下不应期待三者完全相同。另一个具体问题是 S11 的深 dB 谷：PR-Net 在 12 GHz S11 目标上，256 步积分后仍约有 45% 的解释完整性残差超过 1%，与路径上的近奇异响应一致。因此我把方法定义造成的差异与数值不稳定分开讨论。

注意：0.93/0.85 来自参考种子 2026，在 40 个验证集设计和三个 T2 目标上求平均，不是三个训练种子的平均，也不是某个单独目标或单个设计的数值；45% 是另一个 S11 点目标，不能混为 T2 的失败率。

来源：../results/step2/xai_report.md，A1 与 A4。

## Slide 4 — First Faithfulness Result: Model Interventions（90 秒）

英文 slide：

Question: Do highly attributed inputs account for the model's reference-to-design change?

Test design:

1. Rank inputs by attribution aligned with the model-output gap.
2. Delete inputs by replacing them with reference values; insert in reverse construction.
3. Compute normalized curve AUC.
4. Compare against 100 random feature orders.

PR-Net reference seed 2026, 40 validation designs, three T2 targets:

| Method | Deletion: fraction in best 5% of random orders | Insertion: same measure |
|---|---:|---:|
| SHAP | 80% | 89% |
| IG | 69% | 86% |
| LIME-delta | 64% | 93% |

中文讲解：

第一个忠实性结果来自模型输入干预。我按照解释排序，将输入逐步替换成参考值，观察模型相对于参考的输出差异如何消失；插入则从参考开始，逐步恢复当前设计的特征。排序使用归因乘预测差异的符号，而不是简单按绝对值。输出用参考到当前的差异归一化，再比较曲线面积与 100 个随机顺序。差异很小的条目会跳过，避免不稳定归一化。

在 PR-Net 的验证集 T2 目标中，SHAP 的删除顺序约有 80% 落在随机顺序最优的 5%，IG 为 69%，LIME-delta 为 64%；插入时则是 89%、86% 和 93%。这支持解释识别了模型的重要输入，但结果并不支持 SHAP 在所有指标上占优，也不代表“80% 的解释正确”。目前只有九个特征，且重要性较集中，这个检验相对容易，需要补充控制实验。

记录要求：现有 algorithm_plan_v2.md §5A 和 training_xai_protocol_v2.json 已描述该检验，但当前可见方案文件在实验后有更新。尚未核实到可证明“运行前写定”的原始版本，因此只能报告为已有测试及结果，不能把现在的文件当作独立事前注册证据。

建议图：显示一个真实设计的删除/插入曲线及随机基准分布；如果只展示以上汇总表，也必须说明分母是参考种子 2026 下、排除小 gap 后的有效设计–T2目标条目。代码对这些有效条目直接合并求比例；不是三个模型种子的平均。

来源：../results/step2/xai_report.md A2；../scripts/step2_eval.py；../src/filterxai/xai.py。

## Slide 5 — Domain Grounding and Next Steps（40 秒）

英文 slide：

- Known simulator truth: nuisance controls do not change ADS outputs.
- Physical hypotheses: lengths affect frequency position; coupling geometry affects response shape.
- Check against ADS interventions and matched reference games.
- Next: confirm the practitioner requirements on a concrete case and prospectively record a controlled-input test.

中文讲解：

领域依据分成两类。第一类是确定的：三个负对照不参与 ADS，因而对仿真器应当没有影响。若代理模型却使用了它们，忠实解释应该显示这个问题。第二类是物理假设：长度和耦合相关几何可能主要影响不同响应特征，这些需要 ADS 干预验证，而不能只凭重要性图判断正确。下一步我会用具体案例确认从业者提出的需求，并在看到结果之前保存一个新的受控输入测试协议，进一步检查解释是否错误地给完全不被模型使用的输入分配贡献。

## 对课程四项要求的覆盖

| Required component | 本次可讲内容 | 当前状态 |
|---|---|---|
| Stakeholder evidence | 接收前端从业者访谈（网络电话，2026-11-05）；R&S/ADI 实践资料 | 已提炼需求；单人访谈，作者笔记转述，不作逐字引用 |
| Multiple methods | SHAP、IG、LIME；定义差异、路径数值问题、局部拟合质量 | 已有方法和对比结果 |
| Faithfulness test | 删除/插入 vs 100 随机顺序；首个汇总结果 | 已有结果；运行前记录的原始版本待核查 |
| Domain grounding | ADS 不使用负对照；物理假设与 ADS 对照 | 可说明仿真器依据；真实链路和实测验证未完成 |

## 下一轮测试应怎样事前写定（未运行）

可用真实冻结 PR-Net 构造受控包装 F_controlled(x)=F_PRNet(x_1,...,x_6,0,0,0)，使三个负对照在函数定义上严格无效。事前记录：模型与代码哈希、验证集取样规则、解释目标、SHAP/IG/LIME 设置、随机种子、误差容限、评价指标及所有失败结果的保留规则。

明确预期：该包装函数对三个对照没有依赖；精确 SHAP 和 IG 的对照归因应在浮点容限内为零。LIME 的有限样本局部拟合可能产生非零系数，应报告其大小、重复采样变化和留出 R²。不要把未运行协议说成已通过测试，也不要在看到结果后改容限。

这项受控测试检验解释器的无关特征行为，不足以单独证明原始代理模型的全部解释正确。

## 访谈总结三句话

1. On November 5, I interviewed (internet voice call) one RF communications engineer who works on receive-side RF front-end evaluation and troubleshooting.
2. This changed my explanation design by putting the failed specification, frequency range, and violation magnitude before attribution, adding case-level quality diagnostics, and separating diagnostic attribution from validated tuning advice; it does not yet establish that the tool improves troubleshooting.
