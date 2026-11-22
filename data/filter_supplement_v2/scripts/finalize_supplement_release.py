"""Finalize data-only deliverables, scientific figures and portable archives."""
from pathlib import Path
import csv
import json
import hashlib
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import supplement_filter_dataset as helper
import run_momentum_diagnostics as em

ROOT=helper.ROOT
MIRROR=Path('C:/Users/Administrator/Documents/Codex/2026-10-23/referenced-chatgpt-conversation-this-is-an/outputs')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    directory=Path(load(ROOT/'datasets/latest_supplement.json')['directory']);release=directory/'release'
    summary=load(release/'validation_summary.json');plan=load(directory/'plan.json')
    emsource=em.EMROOT/'momentum_diagnostics';manifest=load(emsource/'manifest.json')
    if manifest['status']!='complete':raise RuntimeError('Wait for EM diagnostic completion before final delivery')
    if not (emsource/'focused_mesh_checks.json').exists():raise RuntimeError('Focused mesh follow-up has not completed')
    focus=load(emsource/'focused_mesh_checks.json')
    for cells in [60,90]:
        metadata_path=emsource/f'baseline_focused_mesh{cells}/parameters.json'
        metadata=load(metadata_path);metadata['frequency_points']=20
        metadata['actual_frequency_plan']='9 GHz; 11.3-13.0 GHz at 100 MHz; 15 GHz'
        save(metadata_path,metadata)
    emdest=release/'em_diagnostics'
    if emdest.exists():
        shutil.copytree(emsource,emdest,dirs_exist_ok=True)
    else:shutil.copytree(emsource,emdest)
    native=Path(manifest['native_dense_baseline'])
    native_dest=emdest/'native_baseline_inputs';native_dest.mkdir(exist_ok=True)
    for name in ['proj_a','proj.ltd','proj.pin','proj.prt','proj.opt','proj.sti','proj.cfg','proj.cti','proj.stm']:
        shutil.copy2(native/name,native_dest/name)
    comparisons=list(csv.DictReader((emdest/'circuit_vs_em_diagnostics.csv').open(encoding='utf-8-sig',newline='')))
    assert len(comparisons)==14 and len(manifest['runs'])==14
    nominal=load(emdest/'native_baseline_metrics.json')
    base=load(directory/'runs/diagnostic_baseline/attempt_001/metrics.json')
    mesh30=np.load(emdest/'baseline/response.npz',allow_pickle=False)
    mesh45=np.load(emdest/'baseline_mesh45/response.npz',allow_pickle=False)
    mdiff=float(abs(mesh30['S_complex']-mesh45['S_complex']).max())
    fm=mesh30['frequency_Hz'];pb=(fm>=11.6e9)&(fm<=12.4e9)
    amplitude_diff=float(abs(mesh30['Y_dB'][pb,0]-mesh45['Y_dB'][pb,0]).max())
    m30=load(emdest/'baseline/metrics.json');m45=load(emdest/'baseline_mesh45/metrics.json')
    meshcheck={'baseline_only':True,'cells_per_wavelength':[30,45],'frequency_step_MHz':100,
               'max_complex_S_difference':mdiff,'fixed_passband_max_S21_difference_dB':amplitude_diff,
               'center_difference_MHz':abs(m30['f0_arithmetic_GHz']-m45['f0_arithmetic_GHz'])*1000,
               'qualification':'Two mesh settings on one geometry; does not establish convergence of every EM design or dense notches.'}
    save(emdest/'baseline_mesh_comparison.json',meshcheck)
    center_delta=(nominal['f0_arithmetic_GHz']-base['f0_arithmetic_GHz'])*1000
    summary['EM_diagnostics']={'status':'completed_and_kept_separate','unique_geometries':13,
           'coarse_sweeps_in_parameter_table':14,'coarse_frequency_step_MHz':100,
           'native_baseline_points':601,'native_baseline_step_MHz':10,
           'baseline_circuit_f0_GHz':base['f0_arithmetic_GHz'],'baseline_EM_f0_GHz':nominal['f0_arithmetic_GHz'],
           'baseline_EM_minus_circuit_f0_MHz':center_delta,'baseline_mesh_check':meshcheck,
           'focused_mesh_followup':focus,
           'quality_status':'preliminary: baseline mesh sensitivity observed; domain-wide EM convergence not established',
           'qualification':'Not a 1440-design EM training set. Parameterized conductor outlines verified against native nominal export. Coarse sweeps are diagnostic, not dense interpolated labels.'}
    summary['pending']=['EM mesh convergence across parameter domain','dense/full-domain EM surrogate data','calibrated modal features Z0e/Z0o and Qe','sufficient changed-material learning data']
    summary['EM_source_geometry_reproduction']=load(emdest/'template_verification.json')
    save(release/'validation_summary.json',summary)
    # Verify loaders and all per-design nuisance assignments without fitting models.
    import load_supplement_data as loader
    checks=[];all_geometries=set();all_ids=set()
    for split,n in [('train',1008),('validation',216),('test',216)]:
        for band,points in [('core',601),('wideband',2101)]:
            with np.load(release/f'data/{band}_{split}.npz',allow_pickle=False) as data:
                assert data['S_complex'].shape==(n,points,2,2)
                assert data['X_candidate'].shape==(n,25)
                assert data['negative_controls_replicates'].shape==(3,n,3)
                assert np.array_equal(data['X_candidate'][:,-3:],data['negative_controls_replicates'][0])
                assert np.isfinite(data['S_complex']).all() and np.isfinite(data['X_candidate']).all()
                if band=='core':
                    for did,g in zip(data['design_id'],data['geometry_mil']):
                        assert str(did) not in all_ids and tuple(g) not in all_geometries
                        all_ids.add(str(did));all_geometries.add(tuple(g))
                    assert np.array_equal(data['mask_pass_sampled_grid'],np.all(data['mask_violation_dB']<=1e-9,axis=1))
            checks.append({'split':split,'band':band,'designs':n,'points':points,'passed':True})
        for replicate in [0,1,2]:
            item=loader.load_split(release,split,'core','wide_candidate_experiment',replicate,normalize=True)
            assert item['X_design'].shape==(n,25) and np.isfinite(item['X_design']).all()
            if split=='train':
                assert np.max(abs(item['X_design'].mean(0)))<1e-10
    assert len(all_ids)==len(all_geometries)==1440
    # Check full-band held-out package content against core at shared frequencies.
    shared_error=0.
    for split in ['train','validation','test']:
        with np.load(release/f'data/core_{split}.npz',allow_pickle=False) as core,np.load(release/f'data/wideband_{split}.npz',allow_pickle=False) as wide:
            assert np.array_equal(core['design_id'],wide['design_id'])
            assert np.array_equal(core['X_candidate'],wide['X_candidate'])
            shared_error=max(shared_error,float(abs(core['S_complex']-wide['S_complex'][:,:601]).max()))
    assert shared_error<1e-10
    summary['final_package_checks']={'arrays':checks,'loaders_all_control_replicates_passed':True,
        'unique_designs_and_splits_verified':True,'core_wide_shared_S_max_error':shared_error,
        'model_fitting_performed':False}
    save(release/'validation_summary.json',summary)
    # Static scientific plots backed by actual simulation files.
    figures=release/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11.5,7.4),layout='constrained')
    colors=['#245A81','#B66A28','#6C7040']
    for did,label,color,style in [('diagnostic_baseline','Baseline',colors[0],'-'),
                 ('diagnostic_Wc1_low','CLin1.W -10%',colors[1],'--'),('diagnostic_Wc1_high','CLin1.W +10%',colors[2],'-.')]:
        item=np.load(directory/f'runs/{did}/attempt_001/response.npz',allow_pickle=False)
        f=item['frequency_Hz']/1e9;y=item['Y_dB'][:,0]
        axes[0].plot(f,y,color=color,linestyle=style,label=label)
        axes[1].plot(f,y,color=color,linestyle=style,label=label)
    axes[0].set(xlim=(9,30),ylim=(-65,2),xlabel='Frequency (GHz)',ylabel='S21 (dB)',title='Actual ADS circuit sweeps: 9-30 GHz')
    axes[0].axvspan(18,30,color='#EEEEEE',zorder=-1)
    axes[1].set(xlim=(22.5,25.5),ylim=(-30,1),xlabel='Frequency (GHz)',ylabel='S21 (dB)',title='Circuit-model parasitic passband candidates; EM verification remains separate')
    axes[1].axhline(-3,color='#555555',linestyle=':',label='Absolute -3 dB threshold')
    for ax in axes:ax.grid(alpha=.18);ax.legend(fontsize=9)
    fig.savefig(figures/'wideband_circuit_diagnostics.png',dpi=160);plt.close(fig)
    source=np.load(emdest/'native_baseline_601.npz',allow_pickle=False)
    circuit=np.load(directory/'runs/diagnostic_baseline/attempt_001/response.npz',allow_pickle=False)
    fig,axes=plt.subplots(2,1,figsize=(11.5,7.4),sharex=True,layout='constrained')
    for ax,name,(i,j) in zip(axes,['S21','S11'],[(1,0),(0,0)]):
        ax.plot(circuit['frequency_Hz'][:601]/1e9,20*np.log10(np.maximum(abs(circuit['S_complex'][:601,i,j]),1e-15)),color=colors[0],label='ADS circuit model')
        ax.plot(source['frequency_Hz']/1e9,20*np.log10(np.maximum(abs(source['S_complex'][:,i,j]),1e-15)),color=colors[1],linestyle='--',label='Momentum mesh30 (preliminary)')
        ax.axvspan(11.6,12.4,color='#EEEEEE',zorder=-1)
        ax.set(xlim=(9,15),ylim=(-65,1),ylabel=name+' (dB)');ax.grid(alpha=.18);ax.legend(fontsize=9)
    axes[-1].set_xlabel('Frequency (GHz)')
    fig.suptitle('Baseline comparison: 10 MHz grids; preliminary EM mesh\nFurther mesh refinement changes the response; see mesh checks')
    fig.savefig(figures/'baseline_circuit_vs_momentum.png',dpi=160);plt.close(fig)
    counts={'independent_physical_design_features':6,'fixed_primary_physical_metadata_features':7,
            'deterministic_geometry_transforms':9,'synthetic_negative_controls':3,'candidate_columns_total':25}
    readme=f'''# 补充数据集 v2：使用说明

日期：2026-10-24。数据已准备并核验，**未训练模型**。这是工程数据使用记录，不是课程提交用的学术报告正文。

## 本次补足的内容

| 数据 | 数量与范围 |
|---|---|
| 主数据独立几何设计 | **1,440 组**：原四维 480 组＋新增六维 960 组 |
| 六个可变化的尺寸 | L1、L2、S1、S2、CLin1.W、CLin2.W；单位 mil |
| 核心扫频 | 9–15 GHz、步长 10 MHz、每组 601 点；合计 **865,440** 点 |
| 扩频扫频 | 全部 1,440 组都实际仿真到 30 GHz，每组 2,101 点；合计 **3,025,440** 点 |
| 固定划分 | train 1,008；validation 216；test 216。原始 480 组的划分保持不变 |
| 候选输入 | **25 列**：6 个独立尺寸＋7 个固定物理参数＋9 个几何变换＋3 个负对照 |
| 负对照 | 独立均匀随机量、正态随机量、未接入求解器的虚拟标记尺寸；3 套固定随机分配 |
| 物理诊断 | 25 组独立场景，包括尺寸扰动、Er/H、馈线、介质损耗和有限电导率场景；不混入主训练集 |
| 加密频率 | 5 组 2 MHz 核心频段检查；不作为新增独立设计 |
| 全波对照 | Momentum **13 个几何设计**的 100 MHz 扰动对照，另有基线 10 MHz 扫频及网格对照，单独存储 |

新增六维范围：L1 131.575–145.425；L2 129.2–142.8；S1 2.64–3.96；S2 14.4–21.6；CLin1.W 22.95–28.05；CLin2.W 28.35–34.65 mil。它们作用于同一个 halffil 定义，所以两半同步变化。电气 W 与布局属性 W1/W2 不同。

原 480 组位于固定宽度切片；新增 960 组是六维 LHS。合并后有 1,440 个不同的六坐标设计，但不能称为 1,440 个六维 LHS 样本。核心频点与扩频数据的前 601 点重叠，不能相加当作独立样本。

## 文件与输入预设

- `data/core_train.npz`、`core_validation.npz`、`core_test.npz`：核心频段主数据。
- `data/wideband_train.npz`、`wideband_validation.npz`、`wideband_test.npz`：完整扩频主数据。
- `data/candidate_inputs.csv`：每设计一行的全部 25 列候选输入、固定划分和来源。
- `data/design_metrics.csv`：中心、带宽、纹波、回波损耗、插损、群时延、功率/无源性/互易性、扩频候选通带与临时规格违反量。
- `data/diagnostic_scenarios.npz` 和 `diagnostic_scenario_metrics.csv`：单独保存材料/损耗等场景。
- `em_diagnostics/`：完整保存这轮全波对照的输入、输出和电路配对；不混入 1,440 组电路标签。
- `feature_roles.json`、`schema.json`、`scalers_train_only.json`：字段角色、单位、数组形状和训练集专用统计量。
- `scripts/load_supplement_data.py`：只读加载器，使用 NumPy 即可，不要求 ADS，不会训练。

NPZ 中 `geometry_mil` 是 N×6；`X_candidate` 是 N×25；频率另存 `frequency_Hz`。`S_complex` 是 N×F×2×2，后两维顺序为输出端口、输入端口。四个复数 S 全部保留；`Y_dB` 顺序为 S21、S11。另存 S21 展开相位及群时延（ns）。深阻带传输零点附近的相位导数可能未解析，核心文件提供高传输区域的诊断掩码；它不是训练输入，也不是严格收敛证明。

三个输入预设：

1. `geometry_only`：六个尺寸。
2. `geometry_plus_negative_controls`：六尺寸＋三负对照，适合直接做无关变量实验。
3. `wide_candidate_experiment`：全部 25 列，包含固定列、冗余几何变换与人工对照。

固定参数列不会提供独立变化的信息。负对照每个设计只生成一组值，所有频点共用；三套随机分配不是新增几何设计，不能拆分进入不同集合。同一设计的所有频点必须留在同一个划分中。负对照未根据输出相关性重新抽选；有限样本相关性记录在 `negative_control_train_diagnostics.json`。网络可能误用随机量，解释它们不自动意味着解释器错误。

完整曲线、由曲线提取的中心/带宽/群时延，以及 mask 违反量均是目标或诊断信息，**没有放入候选输入**。若任务改为用部分测试响应补全其他响应，应另定义输入和评价方式，不能直接把同一条目标曲线泄漏回模型。

## 已观察到的工程差异

基线电路模型的算术中心为 **{base['f0_arithmetic_GHz']:.6f} GHz**，原生版图 Momentum 为 **{nominal['f0_arithmetic_GHz']:.6f} GHz**，相差 **{center_delta:.3f} MHz**；带宽分别为 **{base['BW_GHz']:.6f} GHz** 和 **{nominal['BW_GHz']:.6f} GHz**。差异已经明显，不能把电路标签称为全波真值，也不应将少量全波标签不加区分地拼进同一个输出表。此处不把差异归因于某一个未经隔离验证的机制。

全波 13 几何对照采用 100 MHz 网格，适合初步参数趋势和来源差异检查；不能用插值把它伪装成 601 个实际 EM 频点，也不能据此认证所有窄回波谷值。基线有额外的 10 MHz 原生扫频。网格两级对照：通带 S21 最大差异 {amplitude_diff:.6f} dB、中心差异 {meshcheck['center_difference_MHz']:.6f} MHz，仅覆盖基线。

**EM 网格精度限制：**30→45 cells/wavelength 的变化明显，12.175649 GHz 是原始网格结果。另实际补做 60、90 cells/wavelength 的基线重点频率检查，包含 9、15 GHz 端点以及 11.3–13 GHz 的 100 MHz 频点。最后一对网格是否达到本轮预设 0.02 dB／2 MHz 基线检查范围：**{focus['last_pair_within_focus_limits']}**。范围是工程检查起点，不是行业标准；不能据此声称整个参数域已收敛。EM 数据标记为诊断／待确认，不当作完整精度认证的训练真值，详见 `em_diagnostics/focused_mesh_checks.json`。

参数化平面轮廓与 ADS 原生基线导出的五个金属轮廓逐一核对。其名义复数响应未达到最初的 1e-6 严格重复门槛；工程复现检查保留实际差异、频带指标差异与阈值，详见 `em_diagnostics/template_verification.json`。这不是数值完全相等的声明。

宽候选中的几何变换依赖原始尺寸。工程反事实改变尺寸时须同步重算变换；单独 mask 它们属于模型输入空间的消融，不能直接当成可制造的物理修改。

电路模型扩频基线在 **23.355–23.566 GHz** 及 **24.179–24.911 GHz** 出现绝对 -3 dB 候选通带。该结论来自实际电路扫频，尚不是该频段的全波或硬件验证。

主电路数据仍是近似无损、理想导体模型。材料/损耗场景只是明确标注的数值探索，并非厂家公差或某块真实板材的实测参数。实际损耗、辐射和端口处理都会改变两端口功率平衡，不能把主电路的无损等式直接强制到全波数据。

## 核验范围

原始四维版本和基线未覆盖。所有新电路案例均有独立网表、原生 .ds、完整 .s2p、压缩数值数组及日志；已经核对网表参数、原始哈希、数据拼接和按设计划分。选定的 17 个案例还独立读取原生 .ds 与 .s2p 交叉检查。训练统计只来自 train；三个负对照版本各有 train-only 统计。

这轮主电路数据与诊断共完成 **1,470 次原生电路仿真**（960 新主设计＋480 旧设计扩频＋25 诊断＋5 加密）。另外的 EM 对照及其电路配对单独保存，不计为新增独立主设计。

Qe、谐振器 k、可靠的奇偶模阻抗/电长度尚未校准，因此没有编造这些字段。当前没有模型预测结果、训练准确率或 XAI 分析结果。

原始完整目录：`{directory}`。两种便携数据包只是它的发布子集；原生求解器文件仍保存在 D 盘目录。
'''
    (release/'README.md').write_text(readme,encoding='utf-8')
    save(release/'input_column_counts.json',counts)
    for name in ['load_supplement_data.py','finalize_supplement_release.py','run_momentum_diagnostics.py','probe_rfpro_baseline.py','refine_momentum_baseline.py']:
        shutil.copy2(ROOT/'scripts'/name,release/'scripts'/name)
    # Hashes will be sealed after actual final-figure inspection.
    save(release/'delivery_status.json',{'status':'ready_for_final_visual_inspection','no_training':True})
    print(json.dumps({'release':str(release),'EM_center_difference_MHz':center_delta,'EM_mesh_check':meshcheck,
                      'unique_geometries':1440,'candidate_columns':25,'source_arrays_and_loader_checks':'passed'},indent=2))


if __name__=='__main__':main()
