"""Release supplemented ADS data and reproducible synthetic XAI controls.

Core set merges old/new distinct geometries without changing legacy splits.
Wide-band set contains ONLY new actual 9-30 GHz simulations. No labels filled in.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import zipfile
import numpy as np

ROOT=Path('D:/Downloads/ADS_XAI_runs')
MIRROR=Path('C:/Users/Administrator/Documents/Codex/2026-10-23/referenced-chatgpt-conversation-this-is-an/outputs')


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mod(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def stats(a):
    mean=a.mean(0);sd=a.std(0);constant=sd<1e-12
    # Relative round-off on fixed physical constants is not real variation.
    constant|=np.ptp(a,axis=0)<1e-12
    sd[constant]=0
    mean[constant]=a[0,constant]
    return {'mean':mean.tolist(),'std_population':sd.tolist(),'effective_std':np.where(constant,1,sd).tolist(),
            'constant_indices':np.flatnonzero(constant).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path);args=ap.parse_args()
    directory=args.dataset or Path(load(ROOT/'datasets/latest_supplement.json')['directory'])
    plan=load(directory/'plan.json');state=load(directory/'state.json')
    legacy_wide_records=load(directory/'legacy_wide_records.json')['records']
    if state['status'] not in ['simulations_complete','complete']:raise RuntimeError('Simulation stage incomplete')
    helper=mod(ROOT/'scripts/supplement_filter_dataset.py','supplement')
    audit=mod(ROOT/'scripts/validate_pretraining_physics.py','physics')
    oldroot=Path(plan['legacy_dataset']);oldrelease=oldroot/'release'
    release=directory/'release';data=release/'data';data.mkdir(parents=True,exist_ok=True)
    feature_names=helper.PHYSICAL+helper.DERIVED+helper.CONTROLS
    old_metrics={r['design_id']:r for r in csv.DictReader((ROOT/'analysis/pretraining_physics_20261024/all_design_physical_metrics.csv').open(encoding='utf-8-sig',newline=''))}
    all_metrics=[];features=[];split_ids={};scalers={}
    raw_integrity={'complete_run_count':len(state['runs']),'all_raw_hashes_match':True,'netlist_parameters_match':True,
                  'original_release_hashes_match':True,'source_export_alignment_sample_ids':[]}
    original_hashes=load(oldrelease/'artifact_hashes.json')
    for key,digest in original_hashes.items():
        if sha(oldrelease/key)!=digest:raise ValueError('Original release changed: '+key)
    assert sha(ROOT/'baseline'/Path(plan['source_netlist']).name)==plan['source_sha256']
    # Independent literal netlist extraction for all native cases, including material/loss cases.
    for record in plan['records']+legacy_wide_records:
        entry=state['runs'][record['design_id']]
        if entry['status']!='complete':raise ValueError('Failed run in release')
        for key in ['response_npz','netlist','dataset']:
            if sha(directory/entry[key])!=entry[key+'_sha256']:raise ValueError('Raw source changed')
        text=(directory/entry['netlist']).read_text(encoding='utf-8')
        import re
        links={'L1':('MCFIL:CLin1','L'),'L2':('MCFIL:CLin2','L'),'S1':('MCFIL:CLin1','S'),
               'S2':('MCFIL:CLin2','S'),'Wc1':('MCFIL:CLin1','W'),'Wc2':('MCFIL:CLin2','W'),
               'Lfeed':('MLIN2:TL1','L'),'H':('model MSub1 MSUB','H'),'Er':('model MSub1 MSUB','Er'),
               'TanD':('model MSub1 MSUB','TanD'),'Cond':('model MSub1 MSUB','Cond'),'T':('model MSub1 MSUB','T')}
        for k,(component,param) in links.items():
            line=re.findall(rf'(?m)^{re.escape(component)}\s+[^\r\n]*$',text)
            assert len(line)==1
            actual=float(re.search(rf'(?<!\w){param}=([-+\d.eE]+)',line[0])[1])
            assert np.isclose(actual,record['parameters'][k],rtol=1e-11,atol=1e-11)
    export_samples=['diagnostic_baseline','diagnostic_Wc1_low','diagnostic_Wc2_high','diagnostic_TanD_2','diagnostic_conductor_loss_2']
    for split in ['train','validation','test']:
        ordered=[r['design_id'] for r in plan['records'] if r['split']==split]
        export_samples.extend([ordered[0],ordered[-1]])
        ordered=[r['design_id'] for r in legacy_wide_records if r['original_split']==split]
        export_samples.extend([ordered[0],ordered[-1]])
    export_max=0.;native_max=0.
    for did in export_samples:
        entry=state['runs'][did];case=(directory/entry['response_npz']).parent
        raw=np.loadtxt(case/'response.s2p',comments=('!','#'))
        z=raw[:,1::2]+1j*raw[:,2::2]
        full=np.empty((len(raw),2,2),complex)
        full[:,0,0]=z[:,0];full[:,1,0]=z[:,1];full[:,0,1]=z[:,2];full[:,1,1]=z[:,3]
        with np.load(directory/entry['response_npz'],allow_pickle=False) as response:
            export_max=max(export_max,float(abs(full-response['S_complex']).max()))
            assert np.max(abs(raw[:,0]-response['frequency_Hz']))<1.
            _,native,_=helper.read_ds(directory/entry['dataset'],30.,.01)
            native_max=max(native_max,float(abs(native-response['S_complex']).max()))
    assert export_max<1e-12 and native_max<1e-12
    raw_integrity.update(source_export_alignment_sample_ids=export_samples,
                         s2p_vs_response_npz_max_error=export_max,
                         native_ds_vs_response_npz_max_error=native_max)
    for split in ['train','validation','test']:
        old=np.load(oldrelease/f'data/{split}.npz',allow_pickle=False)
        new_records=[r for r in plan['records'] if r['split']==split]
        new_responses=[np.load(directory/state['runs'][r['design_id']]['response_npz'],allow_pickle=False) for r in new_records]
        if len(new_records)!=plan['new_split_sizes'][split]:raise ValueError('New split size mismatch')
        cf=old['frequency_Hz'];wf=new_responses[0]['frequency_Hz']
        assert len(wf)==2101 and np.max(abs(wf[:601]-cf))<1.
        for response in new_responses:
            assert np.array_equal(response['frequency_Hz'],wf)
        ns=np.stack([r['S_complex'] for r in new_responses]);ny=np.stack([r['Y_dB'] for r in new_responses])
        ng=np.array([[r['parameters'][k] for k in helper.GEOM] for r in new_records])
        ni=np.array([r['design_id'] for r in new_records])
        nx=np.array([helper.candidates(r['parameters'],r['design_id']) for r in new_records])
        nc=np.array([[helper.control_values(did,k) for did in ni] for k in range(3)])
        oldg=np.c_[old['geometry_mil'],np.tile([25.5,31.5],(len(old['design_id']),1))]
        oldx=np.array([helper.candidates(dict(zip(helper.GEOM,g)),str(did)) for g,did in zip(oldg,old['design_id'])])
        oldc=np.array([[helper.control_values(str(did),k) for did in old['design_id']] for k in range(3)])
        g=np.r_[oldg,ng];ids=np.r_[old['design_id'],ni];x=np.r_[oldx,nx]
        controls=np.concatenate([oldc,nc],axis=1)
        oldwide_records=[r for r in legacy_wide_records if r['original_split']==split]
        assert [r['original_design_id'] for r in oldwide_records]==old['design_id'].tolist()
        oldwide=[np.load(directory/state['runs'][r['design_id']]['response_npz'],allow_pickle=False) for r in oldwide_records]
        ows=np.stack([r['S_complex'] for r in oldwide]);owy=np.stack([r['Y_dB'] for r in oldwide])
        assert float(abs(ows[:,:601]-old['S_complex']).max())<1e-10
        ws=np.concatenate([ows,ns],axis=0);wy=np.concatenate([owy,ny],axis=0)
        np.savez_compressed(data/f'wideband_{split}.npz',frequency_Hz=wf,S_complex=ws,Y_dB=wy,
             geometry_mil=g,design_id=ids,split=np.array([split]*len(ids)),feature_names=np.array(helper.GEOM),
             X_candidate=x,candidate_feature_names=np.array(feature_names),negative_controls_replicates=controls,
             negative_control_names=np.array(helper.CONTROLS),
             S21_phase_unwrapped_rad=np.concatenate([np.stack([r['S21_phase_unwrapped_rad'] for r in oldwide]),np.stack([r['S21_phase_unwrapped_rad'] for r in new_responses])]),
             S21_group_delay_ns=np.concatenate([np.stack([r['S21_group_delay_ns'] for r in oldwide]),np.stack([r['S21_group_delay_ns'] for r in new_responses])]))
        s=np.concatenate([old['S_complex'],ns[:,:601]],axis=0)
        y=np.concatenate([old['Y_dB'],ny[:,:601]],axis=0)
        pb=(cf>=11.6e9)&(cf<=12.4e9);low=(cf>=9e9)&(cf<=10.8e9);high=(cf>=13.2e9)&(cf<=15e9)
        mask_violation=np.maximum(0,np.stack([-.5-y[:,pb,0].min(1),y[:,pb,1].max(1)+10,
                                            y[:,low,0].max(1)+20,y[:,high,0].max(1)+15],axis=-1))
        mask_pass=np.all(mask_violation<=1e-9,axis=1)
        phase=np.unwrap(np.angle(s[:,:,1,0]),axis=1)
        tau=-np.gradient(phase,2*np.pi*cf,axis=1,edge_order=2)*1e9
        valid=(y[:,:,0]>=-30)
        valid[:,1:-1]&=(y[:,:-2,0]>=-30)&(y[:,2:,0]>=-30)
        phase_step=np.abs(np.angle(s[:,1:,1,0]*s[:,:-1,1,0].conj()))
        valid[:,1:-1]&=(phase_step[:,:-1]<np.pi/2)&(phase_step[:,1:]<np.pi/2)
        valid[:,[0,-1]]=False
        np.savez_compressed(data/f'core_{split}.npz',frequency_Hz=cf,S_complex=s,Y_dB=y,geometry_mil=g,
             design_id=ids,split=np.array([split]*len(ids)),feature_names=np.array(helper.GEOM),
             X_candidate=x,candidate_feature_names=np.array(feature_names),negative_controls_replicates=controls,
             negative_control_names=np.array(helper.CONTROLS),S21_phase_unwrapped_rad=phase,
             S21_group_delay_ns=tau,S21_group_delay_high_transmission_mask=valid,
             mask_violation_dB=mask_violation,mask_pass_sampled_grid=mask_pass,
             mask_component_names=np.array(['passband_transmission','passband_matching','low_stopband','high_stopband']))
        split_ids[split]=ids.tolist()
        for i,(did,geo,xx) in enumerate(zip(ids,g,x)):
            legacy=i<len(old['design_id']);params={**helper.BASE,**dict(zip(helper.GEOM,geo))}
            features.append({'design_id':str(did),'split':split,'source':'legacy_4d_actual' if legacy else 'new_6d_actual',
                             **dict(zip(feature_names,map(float,xx)))})
            if legacy:
                lr=oldwide_records[i]
                m=load(directory/state['runs'][lr['design_id']]['metrics'])
                m={k:v for k,v in m.items() if k not in ['design_id','split','physical_parameters']}
                m['wideband_available']=True
            else:
                m=load(directory/state['runs'][str(did)]['metrics'])
                m={k:v for k,v in m.items() if k not in ['design_id','split','physical_parameters']}
                m['wideband_available']=True
            all_metrics.append({'design_id':str(did),'split':split,'source':'legacy_4d_actual' if legacy else 'new_6d_actual',
                                **{k:params[k] for k in helper.GEOM},**m,'mask_violation_dB':mask_violation[i].tolist(),
                                'mask_pass_sampled_grid':bool(mask_pass[i])})
        if split=='train':
            scalers={'fit_split':'train only','geometry_order':helper.GEOM,'candidate_order':feature_names,
                     'geometry':stats(g),'candidates':stats(x),
                     'negative_control_replicates':{str(k):stats(controls[k]) for k in range(3)},
                     'frequency_core_Hz':stats(cf[:,None]),'frequency_wideband_Hz':stats(wf[:,None]),
                     'outputs_dB':stats(y.reshape(-1,2)),
                     'outputs_wideband_dB':stats(wy.reshape(-1,2)),
                     'outputs_complex_order':['S11_real','S11_imag','S21_real','S21_imag','S12_real','S12_imag','S22_real','S22_imag'],
                     'outputs_complex':stats(np.stack([s[:,:,0,0].real,s[:,:,0,0].imag,s[:,:,1,0].real,s[:,:,1,0].imag,
                                                     s[:,:,0,1].real,s[:,:,0,1].imag,s[:,:,1,1].real,s[:,:,1,1].imag],axis=-1).reshape(-1,8)),
                     'constant_columns':'Kept for provenance; effective std=1 prevents divide by zero; constants do not provide independent variation.'}
            # Describe finite-sample control correlations; never optimize generation against them.
            measures=np.c_[g, np.ptp(y[:,:,0],axis=1),y[:,:,0].mean(1),y[:,:,1].mean(1)]
            correlations=[]
            for k in range(3):
                combined=np.c_[controls[k],measures]
                correlation=np.corrcoef(combined,rowvar=False)[:3,3:]
                correlations.append({'replicate':k,'max_abs_control_vs_geometry_or_basic_response_corr_train':float(abs(correlation).max()),
                                     'correlation_matrix':correlation.tolist()})
            save(release/'negative_control_train_diagnostics.json',{'no_response_based_regeneration':True,
                 'all_control_columns_are_synthetic':True,'finite_sample_correlations':correlations,
                 'causal_truth':'Controls do not enter ADS; fitted networks can nevertheless learn spurious dependencies. Attribution to them is not automatically explainer failure.'})
        # Exact final-package/source reconciliation, not approximate metric comparison.
        packed=np.load(data/f'core_{split}.npz',allow_pickle=False)
        assert np.array_equal(packed['S_complex'][:len(oldg)],old['S_complex'])
        assert np.array_equal(packed['S_complex'][len(oldg):],ns[:,:601])
        assert np.max(np.abs(20*np.log10(np.maximum(abs(s[:,:,1,0]),1e-15))-y[:,:,0]))<1e-10
        assert np.isfinite(x).all() and np.isfinite(s).all()
        assert np.array_equal(x[:,-3:],controls[0])
        for obj in new_responses:obj.close()
        for obj in oldwide:obj.close()
        old.close();packed.close()
    # Diagnostics kept separately: never smuggle sparse changed-material cases into main learning splits.
    diags=[r for r in plan['records'] if r['split']=='diagnostic']
    dr=[np.load(directory/state['runs'][r['design_id']]['response_npz'],allow_pickle=False) for r in diags]
    np.savez_compressed(data/'diagnostic_scenarios.npz',frequency_Hz=dr[0]['frequency_Hz'],
         design_id=np.array([r['design_id'] for r in diags]),physical_feature_names=np.array(helper.PHYSICAL+['Cond']),
         physical_parameters=np.array([[r['parameters'][k] for k in helper.PHYSICAL+['Cond']] for r in diags]),
         S_complex=np.stack([r['S_complex'] for r in dr]),Y_dB=np.stack([r['Y_dB'] for r in dr]))
    diagmetrics=[load(directory/state['runs'][r['design_id']]['metrics']) for r in diags]
    write_csv(data/'diagnostic_scenario_metrics.csv',diagmetrics)
    fine_rows=[]
    for did in plan['fine_ids']:
        coarse=np.load(directory/state['runs'][did]['response_npz'],allow_pickle=False)
        fine=np.load(directory/state['fine_runs'][did]['response_npz'],allow_pickle=False)
        f=fine['frequency_Hz'];core=coarse['frequency_Hz'][:601]
        assert np.max(abs(f[::5]-core))<1.
        alignment=float(abs(coarse['S_complex'][:601]-fine['S_complex'][::5]).max())
        pb=(core>=11.6e9)&(core<=12.4e9)
        delta=coarse['S21_group_delay_ns'][:601][pb]-fine['S21_group_delay_ns'][::5][pb]
        fine_rows.append({'design_id':did,'aligned_complex_S_max_abs_error':alignment,
                          'fixed_passband_group_delay_max_abs_difference_ps':float(abs(delta).max()*1000)})
        assert alignment<1e-10
    write_csv(data/'fine_grid_checks.csv',fine_rows)
    write_csv(data/'candidate_inputs.csv',features)
    write_csv(data/'design_metrics.csv',all_metrics)
    save(release/'scalers_train_only.json',scalers)
    save(release/'split_manifest.json',split_ids)
    flat=[tuple(r[k] for k in helper.GEOM) for r in features]
    assert len(flat)==1440 and len(set(flat))==1440
    assert {k:len(v) for k,v in split_ids.items()}=={'train':1008,'validation':216,'test':216}
    # Full-input constants and derived ratios are not new physical degrees of freedom.
    save(release/'feature_roles.json',{'input_presets':{'geometry_only':helper.GEOM,
             'geometry_plus_negative_controls':helper.GEOM+helper.CONTROLS,
             'wide_candidate_experiment':feature_names},'frequency':'Query coordinate appended when needed; never an actionable geometry parameter.',
             'physical_independent_in_primary_data':helper.GEOM,
             'fixed_primary_physical_columns':[k for k in helper.PHYSICAL if k not in helper.GEOM],
             'deterministic_geometry_transforms':helper.DERIVED,'synthetic_negative_controls':helper.CONTROLS,
             'targets_not_inputs':['S_complex','Y_dB','phase','group_delay','f0','BW','ripple','return_loss','attenuation','parasitic_passbands'],
             'provenance_not_inputs':['design_id','split','file_path','source','run_order'],
             'unavailable_not_fabricated':['Z0e','Z0o','theta_e','theta_o','resonator_k','Qe','fullwave_S_parameters'],
             'warnings':['Derived ratios are redundant with geometry and may share attribution; they are not independent tuning actions.',
                         'Do not infer Er/H/TanD effects from constant primary columns; sparse scenario data are diagnostics.',
                         'Cond=1e306 means ideal conductor in this model; do not standardize it as a varying numeric learning feature.']})
    save(release/'schema.json',{'core':'core_{train,validation,test}.npz: 1440 actual distinct six-coordinate geometries, 601 points 9-15 GHz, 1008/216/216 designs',
             'wideband':'wideband_{train,validation,test}.npz: all 1440 actual designs, 2101 points 9-30 GHz, 1008/216/216 designs; legacy 480 actually resimulated over the wider range',
             'geometry_mil':'N x 6, order L1 L2 S1 S2 Wc1 Wc2','X_candidate':f'N x {len(feature_names)}, physical parameters + deterministic geometry transforms + 3 synthetic controls; no target statistics',
             'negative_controls_replicates':'3 x N x 3; replicate 0 is present in X_candidate. Alternative replicates are experimental nuisance assignments, not new designs.',
             'S_complex':'N x F x 2 x 2, output port then input port, all complex channels retained',
             'Y_dB':'N x F x 2, S21_dB then S11_dB; log floor only at -300 dB',
             'phase_and_group_delay':'S21 unwrapped radians and ns; 10 MHz sampling. Deep stopband zeros can invalidate derivatives.',
             'S21_group_delay_high_transmission_mask':'Core-only diagnostic target mask: center and neighbors >=-30 dB, adjacent phase steps <90 degrees, endpoints excluded; not a mathematical convergence certificate or a model input.',
             'mask_targets':'Core mask_violation_dB N x 4 and mask_pass_sampled_grid N; inherited provisional mask, not an external engineering specification and not an input.',
             'diagnostics':'25 independent exploratory physical scenarios + 5 fine-grid reruns; excluded from primary splits',
             'frequency_rows':'Core 865440; wideband 3025440. Core rows overlap the first 601 points of wideband; never count them as independent observations.',
             'split_policy':'Preserve all legacy splits; new designs preassigned before simulation. No random per-frequency splitting.'})
    summary={'status':'complete','source_type':'ADS 2026 circuit simulation, not EM or hardware',
             'legacy_actual_designs':480,'new_actual_primary_designs':960,'core_unique_designs':1440,
             'core_frequency_rows':1440*601,'core_split_counts':{k:len(v) for k,v in split_ids.items()},
             'wideband_designs':1440,'wideband_frequency_rows':1440*2101,'wideband_range_GHz':[9,30],
             'new_diagnostic_scenarios':len(diags),'new_fine_runs':len(plan['fine_ids']),
             'new_native_simulation_total':len(state['runs'])+len(state['fine_runs']),
             'candidate_input_columns':len(feature_names),'negative_control_replicates':3,
             'new_main_LHS_rank':int(np.linalg.matrix_rank(np.array([[r['parameters'][k] for k in helper.GEOM] for r in plan['records'] if r['split']!='diagnostic'])-np.array([helper.BASE[k] for k in helper.GEOM]))),
             'original_data_preserved':True,'raw_integrity':raw_integrity,
             'fine_grid_checks':fine_rows,'no_neural_training':True,
             'main_lossless_max_unitarity_error':max(r['unitarity_error'] for r in state['runs'].values() if r['split']!='diagnostic'),
             'all_runs_max_passivity_excess':max(r['passivity_excess'] for r in state['runs'].values()),
             'all_runs_max_reciprocity_error':max(r['reciprocity_error'] for r in state['runs'].values()),
             'provisional_mask_pass_count':sum(r['mask_pass_sampled_grid'] for r in all_metrics),
             'legacy_wideband_completion':'All 480 legacy geometries actually resimulated over 9-30 GHz; no model-filled labels.',
             'pending':['fullwave comparison','calibrated modal features Z0e/Z0o and Qe','sufficient changed-material learning data']}
    save(release/'validation_summary.json',summary)
    for name in ['plan.json','baseline_repeat_check.json','legacy_wide_records.json']:shutil.copy2(directory/name,release/name)
    scripts=release/'scripts';scripts.mkdir(exist_ok=True)
    for name in ['supplement_filter_dataset.py','package_supplement_dataset.py','validate_pretraining_physics.py','complete_legacy_wideband.py','load_supplement_data.py']:
        shutil.copy2(ROOT/'scripts'/name,scripts/name)
    state['status']='complete';save(directory/'state.json',state)
    save(ROOT/'datasets/latest_supplement.json',{'directory':str(directory),'release':str(release),**summary})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
