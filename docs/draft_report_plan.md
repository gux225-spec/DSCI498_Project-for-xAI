# Draft report 计划（截止 2026-12-02）

写于 2026-11-17。所有数字来自 `results/` 下现有报告，括号里注明来源；写正文时直接引用这些文件，不重新计算。

## 0. 一句话定位（framing）

> For a **receive-side RF troubleshooter** deciding **which failed specification / frequency band to check first and what to hand to the component designer**, does a **contrastive (reference-relative) explanation** of a surrogate filter model improve **decision quality** without losing **faithfulness** — and does it still hold when the physics model changes?

- 原则：Contrastive（Lipton）。单一参考 SHAP 本身就是“为什么是这个设计的响应，而不是基线设计的响应”，与排障人员“为什么这里没过、正常的应该是什么样”一致。
- Data：1440 个 ADS 电路仿真设计，6 个几何参数 + 3 个负对照，S 参数 9–15 GHz。
- Model：PR-Net 代理模型（另有 M1 强基线、M0 弱模型做对照）。
- Stakeholder：一位接收前端评估/排障工程师（2026-11-05 访谈）。
- Decision：先查哪个规格/频段；交给器件工程师什么（几何调整建议需另行验证）。

## 1. 目前的发现（按课程三个问题组织）

### Q1 解释是否真实（faithful）

**对模型忠实——成立。**
- SHAP 加和误差最大 1.6e-4；删除测试中 SHAP 有 80% 的条目落在 100 个随机顺序的前 5%，IG 69%，LIME-delta 64%（`step2/xai_report.md` A1/A2）。
- 负对照：PR-Net 对三个对照的归因占比 ≈ 0，与模型实际干预敏感度（≈ 0–0.01）一致。M0 对对照有 3–7% 的归因，也与它实际使用对照的程度（4–8%）一致，说明解释如实暴露了弱模型的捷径（A3）。

**对物理真值忠实——成立，且误差主要来自解释器，不是网络。**
- test 集冻结后（`step3/freeze.json`，读 test 标签前写入）用 ADS 计算真值 Shapley（80 个设计 × 64 个联盟）。
- PR-Net 的 SHAP 与 ADS 真值的 Spearman 为 0.98–0.99，符号一致率 1.00；IG 为 0.82–0.99；LIME-delta 为 0.84–0.97（`step3/step3_report.md`）。
- 分解：PR-Net“网络 vs ADS”= 0.98–0.99，“SHAP vs 网络”= 0.99，所以 IG/LIME 的偏差来自解释方法本身。M0 的“网络 vs ADS”只有 0.80–0.91，这时偏差来自模型。
- 朴素基线（train OLS 斜率）在 T1b/T2 上只有 0.56–0.59，解释器明显优于“不用模型”。

**方法之间的分歧有原因可讲。**
- SHAP 与 IG 在 T2 上的一致性 0.93，SHAP 与 LIME-delta 0.85。三者回答的问题不同：参考相对贡献 vs 路径积分 vs 局部线性。
- IG 在 S11 深谷处数值不稳定，45% 的条目残差 > 1%。
- SHAP 依赖背景选择：单一参考与 lhs32 的一致性只有 0.70–0.78（A4）。

### Q2 解释是否有用（useful）

证据有两层，都只是代理证据，没有真人决策实验。

1. **预注册调谐实验**（`step4/prereg.json`、`step4_report.md`）：30 个未过掩模的 test 设计，只允许改 2 个参数。
   - SHAP 选参数 25/30 成功，随机选 6/30，McNemar p = 4e-5（Holm）。IG 25/30，LIME-delta 24/30。
   - SHAP 选出的参数组与 ADS 真值 Shapley 选出的有 87% 相同，而 ADS 真值本身要多花 1992 次仿真。
   - **不利于解释的事实（必须写）**：不看解释、直接用代理模型搜索 6 个参数，30/30 成功，中位数只需 1 次仿真。固定调 L1+L2 也有 21/30（`step4b`）。所以解释的价值不在“更快修好”，而在**告诉人该看哪里、交给谁**。
2. **Stakeholder 需求**（`docs/stakeholder_interview_notes_en.md`），三条：
   - 先给出失败的规格、频段和超标量；
   - 给出单个案例的解释质量信息；
   - 把诊断归因和已验证的调参建议分开。
   这三条直接决定了报告里解释的呈现顺序。

### Q3 条件变化后是否站得住（hold up）

**模型偏移（电路 → 全波）：调谐决策站不住**（`results/fullwave_openems/report.md`）。
- 3 个 SHAP 调谐后在 ADS 里通过的设计，用 openEMS 全波复核：**0/3 通过**。2/3 违反大幅下降（约 13 dB），test_0005 反而变差 3.8 dB。
- 机制：电路模型比全波宽出约 0.13 GHz 带宽，并且漏掉约 17% 的通带辐射（约 0.8 dB）。Momentum 独立给出同样的带宽和辐射，所以不是 openEMS 的个别问题。
- 附带发现：数据集 README 中 Momentum 与电路之间约 122 MHz 的 f0 偏差，有一部分是 Momentum 网格误差。openEMS 与 Momentum 的 f0 从两侧向约 12.04–12.12 GHz 收敛。

**模型质量下降（PR-Net → M0）：解释平滑退化。**
- M0 的 SHAP 调谐仍有 22/30 成功（`step4b`）。
- M0 的解释与真值一致性降到 0.80–0.91，下降幅度与模型自身误差相当，没有被放大。

### 失败或放弃的解释（final report 必需一节）

| 项目 | 发生了什么 | 来源 |
|---|---|---|
| 论文的信任判据（SHAP–LIME 一致 → 可信） | 判据与真值一致性的 Spearman 只有 0.12–0.27，**不能**用来判断某个解释是否可信 | step3 最后一节 |
| IG 在 S11 深谷 | 256 步积分后仍有 45% 的残差 > 1% | step2 A1 |
| 基线处 SHAP 当局部灵敏度用 | 与 ADS 单参数扰动的排序一致性只有 0.33，因为它回答的是另一个问题 | step2 B |
| “SHAP 前 2 直接改回基线、不搜索” | 只有 17/30 成功，解释给出了方向，但没给出步长 | step4 |
| SHAP 引导的调谐 → 全波 | 0/3 通过，1/3 变差 | fullwave_openems |
| 放弃的 stakeholder 框架 | 最初设想由器件设计者挑选两个几何尺寸，但唯一可访谈的人不改几何，因此改为排障人员 | 访谈笔记 |
| 删除/插入测试的预注册 | 找不到能证明“运行前写定”的版本，只能作为事后检验报告 | progress_talk_5min |

## 2. 与课程要求的差距

| 要求 | 现状 | 差距 |
|---|---|---|
| Stakeholder evidence | 1 次访谈 | 无具体案例；**可选**：用 test_0005 做一张“先规格后归因”的案例卡，再找同一人谈 15 分钟 |
| ≥3 种方法 + 分歧分析 | 已完成 | 无 |
| 运行前写定的忠实性测试 | step3 冻结 + ADS 真值（有哈希记录）；step4 prereg | 已满足。planted-equation 测试（路线图 Phase 1）**可选**加强 |
| Domain grounding | ADS 真值 Shapley；负对照；全波 | 已满足 |
| Q3 | 全波复核 + M0 | 已满足。planted-shortcut 消融**可选** |
| Failed explanations 一节 | 素材齐全 | 无 |
| repo：README、decision log、AI-use reflection | 均未写 | draft 不需要，final（12-09）需要 |

结论：**不补任何新实验也能写出完整的 draft。** 可选实验只用来加强论证，不是必需。

## 3. Draft 结构（会议格式 6–8 页，英文）

| § | 标题 | 页数 | 核心内容 / 图表 |
|---|---|---|---|
| 1 | Introduction | 0.5 | framing 句；三问题；贡献列表 |
| 2 | Stakeholder and Decision | 0.5 | 访谈三条需求 → 设计约束 |
| 3 | Data and Surrogate Model | 0.75 | 1440 ADS 设计，负对照；PR-Net 精度（S21 MAE 0.084 dB，f0 误差 2.7 MHz）；**Fig 1** 流程图 |
| 4 | Explanation Methods and Principle | 0.5 | SHAP（单一参考 = contrastive）、IG、LIME-delta；为什么会分歧 |
| 5 | Q1 Faithfulness | 1.5 | 5.1 对模型（删除/插入、负对照）；5.2 对 ADS 真值（冻结、分解表）；**Fig 2** 解释→网络→物理分解；**Tab 1** |
| 6 | Q2 Usefulness | 1.0 | 预注册调谐；6 参数搜索这个“反例”；对排障人员意味着什么；**Fig 3** 成功率 |
| 7 | Q3 Robustness under Model Shift | 1.0 | 全波复核；机制；M0 退化；**Fig 4** = `summary_violation.png`，**Fig 5** = test_0005 曲线 |
| 8 | Failed or Abandoned Explanations | 0.75 | 上表 |
| 9 | Limitations and Conclusion | 0.5 | 单人访谈、n=3 全波、ADS 理想化；回答 framing 句 |

## 4. 时间表（15 天）

| 日期 | 任务 |
|---|---|
| 11-17 → 11-19 | 写 `docs/decision_log.md`（补记关键决策）；从现有结果生成 Fig 1–3（脚本放 `scripts/`） |
| 11-19 → 11-22 | （可选）test_0005 案例卡 + 同一 stakeholder 回访；或 planted-equation 测试（先写协议并记哈希，再运行） |
| 11-22 → 11-28 | 写正文 §1–9 |
| 11-29 → 12-01 | 逐个核对数字与来源文件；压缩到 8 页内 |
| 12-02 | 提交 draft |
| 12-02 → 12-09 | 根据反馈修改；README、AI-use reflection、repo 整理 |
