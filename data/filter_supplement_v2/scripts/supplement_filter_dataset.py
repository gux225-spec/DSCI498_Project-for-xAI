"""Supplement real ADS circuit data, wide-band diagnostics and XAI controls.

No model fitting. Original data stay immutable. All changed physical inputs are
actually netlisted and simulated. Synthetic negative controls never enter ADS.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback
import numpy as np

ROOT=Path('D:/Downloads/ADS_XAI_runs')
ADS=Path('E:/ADS2026')
GEOM=['L1','L2','S1','S2','Wc1','Wc2']
BASE={'L1':138.5,'L2':136.,'S1':3.3,'S2':18.,'Wc1':25.5,'Wc2':31.5,
      'Lfeed':100.,'Er':3.78,'H':15.,'TanD':0.,'Cond':1e306,'T':.1,'Wfeed':32.,'Rough':0.}
BOUNDS={'L1':[131.575,145.425],'L2':[129.2,142.8],'S1':[2.64,3.96],'S2':[14.4,21.6],
        'Wc1':[22.95,28.05],'Wc2':[28.35,34.65]}
SPLITS={'train':(672,202610241),'validation':(144,202610242),'test':(144,202610243)}
PHYSICAL=GEOM+['Lfeed','Er','H','TanD','T','Wfeed','Rough']
DERIVED=['L1_over_H','L2_over_H','S1_over_H','S2_over_H','Wc1_over_H','Wc2_over_H',
         'Lfeed_over_H','Lsum_over_H','S1_over_S2']
CONTROLS=['control_uniform','control_normal','control_unused_marker_mm']


def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(p)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def patch(source,params,stop=30.,step=.01):
    p={**BASE,**params}
    text=source
    specs=[('MCFIL:CLin1',{'L':(p['L1'],' mil'),'S':(p['S1'],' mil'),'W':(p['Wc1'],' mil')}),
           ('MCFIL:CLin2',{'L':(p['L2'],' mil'),'S':(p['S2'],' mil'),'W':(p['Wc2'],' mil')}),
           ('MLIN2:TL1',{'L':(p['Lfeed'],' mil'),'W':(p['Wfeed'],' mil')}),
           ('model MSub1 MSUB',{'H':(p['H'],' mil'),'Er':(p['Er'],''),'TanD':(p['TanD'],''),
                              'Cond':(p['Cond'],''),'T':(p['T'],' mil'),'Rough':(p['Rough'],' meter')})]
    verified={}
    for prefix,changes in specs:
        pattern=rf'(?m)^{re.escape(prefix)}\s+[^\r\n]*$'
        hits=list(re.finditer(pattern,text))
        if len(hits)!=1:raise ValueError('Ambiguous component: '+prefix)
        line=hits[0].group(0)
        for key,(value,unit) in changes.items():
            token=rf'(?<!\w){key}=([-+\d.eE]+){re.escape(unit)}(?=\s|$)'
            line,n=re.subn(token,lambda m:f'{key}={value:.12g}{unit}',line)
            if n!=1:raise ValueError('Cannot change '+prefix+'.'+key)
            found=re.search(token,line)
            actual=float(found[1])
            if not np.isclose(actual,value,rtol=1e-11,atol=1e-11):raise ValueError('Emitted parameter mismatch')
            verified[prefix+'.'+key]=actual
        text=text[:hits[0].start()]+line+text[hits[0].end():]
    text,n=re.subn(r'(?m)^SweepPlan: SP1_stim Start=[^\r\n]+$',
                   f'SweepPlan: SP1_stim Start=9.0 GHz Stop={stop:.12g} GHz Step={step:.12g} GHz ',text)
    if n!=1:raise ValueError('Ambiguous sweep statement')
    return text,p,verified


def read_ds(path,stop,step):
    from keysight.ads import dataset
    with dataset.open(path) as ds:
        blocks=[k for k in ds.varblock_names if {'S[1,1]','S[1,2]','S[2,1]','S[2,2]'}.issubset({v.name for v in ds[k].dvars})]
        if len(blocks)!=1:raise ValueError('Ambiguous S parameter block')
        frame=ds[blocks[0]].to_dataframe().reset_index()
    f=frame['freq'].to_numpy(float)
    expected=np.linspace(9e9,stop*1e9,round((stop-9)/step)+1)
    if f.shape!=expected.shape or not np.allclose(f,expected,rtol=0,atol=1):raise ValueError('Sweep grid mismatch')
    s=np.empty((len(f),2,2),complex)
    for i in range(2):
        for j in range(2):s[:,i,j]=frame[f'S[{i+1},{j+1}]'].to_numpy(complex)
    for port in [1,2]:
        if not np.allclose(frame[f'PortZ[{port}]'],50+0j,rtol=0,atol=1e-9):raise ValueError('Port impedance mismatch')
    if not np.isfinite(s).all():raise ValueError('Nonfinite S data')
    return f,s,blocks[0]


def control_values(design_id,replicate=0):
    # Per-design stream independent of geometry, labels, frequencies and run order.
    seed=int.from_bytes(hashlib.sha256(f'XAI_null_v1|{20261024+replicate}|{design_id}'.encode()).digest()[:16],'little')
    rng=np.random.default_rng(seed)
    return np.array([rng.uniform(-1,1),rng.normal(),rng.uniform(0,10)])


def candidates(params,design_id):
    p={**BASE,**params};h=p['H']
    derived=[p[k]/h for k in ['L1','L2','S1','S2','Wc1','Wc2','Lfeed']]+[(p['L1']+p['L2'])/h,p['S1']/p['S2']]
    return np.r_[[p[k] for k in PHYSICAL],derived,control_values(design_id)]


def lhs(n,seed):
    rng=random.Random(seed);cols={}
    for name in GEOM:
        lo,hi=BOUNDS[name]
        vals=[round(lo+(hi-lo)*(j+rng.random())/n,9) for j in range(n)]
        rng.shuffle(vals);cols[name]=vals
    return [{k:cols[k][i] for k in GEOM} for i in range(n)]


def initialize():
    legacy=Path(load(ROOT/'datasets/latest_full_dataset.json')['directory'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory=ROOT/'datasets'/f'filter_supplement_v2_{stamp}'
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'source').mkdir()
    source_path=next((ROOT/'baseline').glob('*.ckt'))
    shutil.copy2(source_path,directory/'source/baseline.ckt')
    records=[{'design_id':'diagnostic_baseline','split':'diagnostic','kind':'baseline','parameters':BASE.copy()}]
    for name in GEOM:
        for side,value in zip(['low','high'],BOUNDS[name]):
            records.append({'design_id':f'diagnostic_{name}_{side}','split':'diagnostic','kind':'one_factor_geometry',
                            'parameters':{**BASE,name:value}})
    for name,factors in [('Er',[.95,1.05]),('H',[.95,1.05]),('Lfeed',[.9,1.1])]:
        for i,factor in enumerate(factors):
            records.append({'design_id':f'diagnostic_{name}_{i}','split':'diagnostic','kind':'exploratory_material_or_feed',
                            'parameters':{**BASE,name:BASE[name]*factor}})
    for i,value in enumerate([.001,.003,.005]):
        records.append({'design_id':f'diagnostic_TanD_{i}','split':'diagnostic','kind':'dielectric_loss_only',
                        'parameters':{**BASE,'TanD':value}})
    for i,p in enumerate([{'Cond':5.8e7},{'Cond':5.8e7,'TanD':.003},{'Cond':5.8e7,'TanD':.003,'T':1.378} ]):
        records.append({'design_id':f'diagnostic_conductor_loss_{i}','split':'diagnostic','kind':'hypothetical_copper_loss',
                        'parameters':{**BASE,**p}})
    for split,(n,seed) in SPLITS.items():
        for i,g in enumerate(lhs(n,seed)):
            records.append({'design_id':f'v2_{split}_{i:04d}','split':split,'kind':'six_dimensional_LHS','parameters':{**BASE,**g}})
    fine_ids=['diagnostic_baseline','diagnostic_Wc1_low','diagnostic_Wc1_high','diagnostic_Wc2_low','diagnostic_Wc2_high']
    plan={'version':2,'created_utc':utc(),'directory':str(directory),'legacy_dataset':str(legacy),
          'source_netlist':str(source_path),'source_sha256':sha(source_path),'source_snapshot_sha256':sha(directory/'source/baseline.ckt'),
          'native_solver':'ADS 2026 circuit S-parameter simulation; not Momentum/HFSS and not measured hardware',
          'independent_main_geometry':GEOM,'bounds_mil':BOUNDS,'new_split_sizes':{k:v[0] for k,v in SPLITS.items()},
          'new_split_seeds':{k:v[1] for k,v in SPLITS.items()},'fixed_main':{k:v for k,v in BASE.items() if k not in GEOM},
          'main_wide_grid':{'start_GHz':9.,'stop_GHz':30.,'step_GHz':.01,'n_points':2101},
          'core_grid':{'start_GHz':9.,'stop_GHz':15.,'step_GHz':.01,'n_points':601},
          'fine_core_grid':{'start_GHz':9.,'stop_GHz':15.,'step_GHz':.002,'n_points':3001},
          'fine_ids':fine_ids,'records':records,
          'synthetic_controls':{'names':CONTROLS,'replicates':3,'frequency_behavior':'constant for each design',
                'generator':'SHA256-derived independent RNG stream per design ID and replicate; independent of labels and geometry',
                'finite_sample_correlations':'Report on train only; never regenerate controls to reduce observed correlations.',
                'solver_connection':'none; clearly synthetic XAI negative controls, not physical measurements'},
          'main_eligibility':'Retain every successfully simulated numerically valid design, irrespective of mask success or disconnected bands.',
          'diagnostic_ranges':'Exploratory numeric scenarios, not manufacturer tolerances or validated material properties. Excluded from all primary learning splits.',
          'not_started':['neural training','XAI fitting','counterfactual search'],
          'original_data_policy':'Read-only; do not overwrite original 4D data, source schematics or baseline.'}
    save(directory/'plan.json',plan)
    save(directory/'state.json',{'status':'prepared','runs':{},'fine_runs':{},'created_utc':utc()})
    save(ROOT/'datasets/latest_supplement.json',{'directory':str(directory),'status':'prepared'})
    print('PREPARED '+str(directory),flush=True)
    return directory


def run_case(record,directory,simulator,audit,fine=False):
    state=load(directory/'state.json');key='fine_runs' if fine else 'runs';did=record['design_id']
    old=state[key].get(did)
    if old and old['status']=='complete':
        for field in ['response_npz','netlist','dataset']:
            if sha(directory/old[field])!=old[field+'_sha256']:raise ValueError('Completed source changed: '+did)
        return old
    base=directory/key/did
    attempt=len(list(base.glob('attempt_*')))+1
    case=base/f'attempt_{attempt:03d}';case.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();step=.002 if fine else .01;stop=15. if fine else 30.
    result={'status':'running','split':record['split'],'kind':record['kind'],'started_utc':utc()}
    state[key][did]=result;save(directory/'state.json',state)
    try:
        source=(directory/'source/baseline.ckt').read_text(encoding='utf-8')
        netlist,p,actual=patch(source,record['parameters'],stop,step)
        save(case/'parameters.json',{'requested':record['parameters'],'actual_netlist':actual,'all_physical_parameters':p})
        simulator.run_netlist(netlist,output_dir=str(case),working_dir=str(case),netlist_file=str(case/'design.ckt'),
                              output_file=str(case/'simulator.log'),dataset_name='response')
        log=(case/'simulator.log').read_text(encoding='utf-8',errors='replace')
        ds=list(case.glob('*.ds'))
        if len(ds)!=1 or 'Simulation finished' not in log or 'Error detected by hpeesofsim' in log:
            raise ValueError('Native solver completion not confirmed')
        f,s,block=read_ds(ds[0],stop,step)
        m,traces=audit.measure(f,s)
        singular=m['sampled_passivity_positive_excess'];recip=m['reciprocity_max_abs_error']
        lossless=p['TanD']==0 and p['Cond']>=1e100
        if singular>1e-7 or recip>1e-7:raise ValueError('Passivity/reciprocity numeric check failed')
        if lossless and m['full_twoport_unitarity_max_abs_error']>1e-7:raise ValueError('Lossless matrix check failed')
        m.update({'design_id':did,'split':record['split'],'physical_parameters':p,'lossless_material_model':lossless,
                  'physics_check_scope':'sampled circuit data only','group_delay_validity':'Use in-band high-transmission values; wide-band phase derivatives near transmission zeros can be underresolved.'})
        if not fine:
            # Absolute -3 dB parasitic-passband candidates; not a relative threshold on a weak stopband peak.
            ff,yy=audit.window(f/1e9,traces[0],18.,30.)
            m['wideband_18_30_absolute_minus3dB_components']=audit.intervals(ff,yy,-3.)
            m['S21_18_30_peak_dB']=float(yy.max())
            m['S21_18_30_peak_frequency_GHz']=float(ff[np.argmax(yy)])
            for ghz in [20.,24.,25.,30.]:m[f'S21_at_{ghz:g}GHz_dB']=float(np.interp(ghz,f/1e9,traces[0]))
        save(case/'metrics.json',m)
        np.savez_compressed(case/'response.npz',frequency_Hz=f,S_complex=s,Y_dB=np.stack(traces[:2],axis=-1),
                            S21_phase_unwrapped_rad=traces[3],S21_group_delay_ns=traces[2])
        with (case/'response.s2p').open('w',encoding='ascii') as stream:
            stream.write('! Real ADS circuit simulation, not EM or hardware; synthetic XAI controls do not enter solver\n# Hz S RI R 50\n')
            for freq,sm in zip(f,s):
                vals=[sm[0,0],sm[1,0],sm[0,1],sm[1,1]]
                stream.write(' '.join([f'{freq:.12g}']+[f'{v:.16g}' for z in vals for v in [z.real,z.imag]])+'\n')
        result.update(status='complete',parameters=p,n_points=len(f),block=block,
             response_npz=str((case/'response.npz').relative_to(directory)),netlist=str((case/'design.ckt').relative_to(directory)),
             dataset=str(ds[0].relative_to(directory)),metrics=str((case/'metrics.json').relative_to(directory)),
             lossless=lossless,unitarity_error=m['full_twoport_unitarity_max_abs_error'],
             passivity_excess=singular,reciprocity_error=recip)
        for field in ['response_npz','netlist','dataset']:result[field+'_sha256']=sha(directory/result[field])
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
        (case/'python_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result['elapsed_seconds']=time.monotonic()-start;result['completed_utc']=utc()
    state[key][did]=result;save(directory/'state.json',state)
    with (directory/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps({'design_id':did,'fine':fine,**result})+'\n')
    return result


def run(directory,limit=None):
    os.environ['HPEESOF_DIR']=str(ADS)
    from keysight.edatoolbox.ads import CircuitSimulator
    simulator=CircuitSimulator(hpeesof_dir=str(ADS))
    audit=module(ROOT/'scripts/validate_pretraining_physics.py','physical_audit')
    plan=load(directory/'plan.json')
    state=load(directory/'state.json');state['status']='running';save(directory/'state.json',state)
    completed_now=0
    for i,record in enumerate(plan['records']):
        if load(directory/'state.json')['runs'].get(record['design_id'],{}).get('status')=='complete':continue
        r=run_case(record,directory,simulator,audit)
        if r['status']!='complete':
            r=run_case(record,directory,simulator,audit)
            if r['status']!='complete':raise RuntimeError('Persistent failure retained: '+record['design_id']+' '+r.get('error',''))
        if record['design_id']=='diagnostic_baseline':
            data=np.load(directory/r['response_npz'],allow_pickle=False)
            _,original,_=module(ROOT/'scripts/stage_a_sweep.py','stage_helper').read_dataset(ROOT/'baseline/spar_sim.ds')
            diff=float(np.abs(data['S_complex'][:601]-original).max())
            if diff>1e-10:raise RuntimeError('Baseline repeat mismatch')
            save(directory/'baseline_repeat_check.json',{'max_complex_S_difference_on_9_15':diff,'passed':True})
        completed_now+=1
        if i<26 or (i+1)%20==0 or i==len(plan['records'])-1:
            print(f'PROGRESS {i+1}/{len(plan["records"])} {record["design_id"]} {r["elapsed_seconds"]:.2f}s',flush=True)
        if limit and completed_now>=limit:return
    for did in plan['fine_ids']:
        record=next(r for r in plan['records'] if r['design_id']==did)
        r=run_case(record,directory,simulator,audit,fine=True)
        if r['status']!='complete':raise RuntimeError('Fine run failed: '+did)
        print('FINE_COMPLETE '+did,flush=True)
    state=load(directory/'state.json');state['status']='simulations_complete';save(directory/'state.json',state)
    save(ROOT/'datasets/latest_supplement.json',{'directory':str(directory),'status':'simulations_complete'})
    print('SIMULATIONS_COMPLETE '+str(directory),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    directory=args.resume or initialize()
    run(directory,args.limit)


if __name__=='__main__':main()
