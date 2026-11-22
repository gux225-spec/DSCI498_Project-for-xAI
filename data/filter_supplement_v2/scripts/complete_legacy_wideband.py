"""Actually rerun every legacy geometry over 9-30 GHz, keeping original splits."""
import os
from pathlib import Path
import json
import numpy as np
import supplement_filter_dataset as helper


def main():
    directory=Path(helper.load(helper.ROOT/'datasets/latest_supplement.json')['directory'])
    plan=helper.load(directory/'plan.json');state=helper.load(directory/'state.json')
    if state['status'] not in ['simulations_complete','complete','running_legacy_wideband']:
        raise RuntimeError('Finish primary simulation writer first; do not race state.json')
    oldroot=Path(plan['legacy_dataset']);records=[]
    for split in ['train','validation','test']:
        with np.load(oldroot/f'release/data/{split}.npz',allow_pickle=False) as old:
            for did,g in zip(old['design_id'],old['geometry_mil']):
                records.append({'design_id':'wide_completion_'+str(did),'original_design_id':str(did),'original_split':split,
                                'split':'legacy_wide','kind':'legacy_wideband_completion',
                                'parameters':{**helper.BASE,**dict(zip(['L1','L2','S1','S2'],map(float,g)))}})
    helper.save(directory/'legacy_wide_records.json',{'source':str(oldroot),'records':records,'count':len(records),
                'selection':'All legacy designs, without inspecting response quality. Preserve original design IDs and splits.'})
    state['status']='running_legacy_wideband';helper.save(directory/'state.json',state)
    os.environ['HPEESOF_DIR']=str(helper.ADS)
    from keysight.edatoolbox.ads import CircuitSimulator
    sim=CircuitSimulator(hpeesof_dir=str(helper.ADS))
    audit=helper.module(helper.ROOT/'scripts/validate_pretraining_physics.py','audit')
    for i,record in enumerate(records):
        result=helper.run_case(record,directory,sim,audit)
        if result['status']!='complete':
            result=helper.run_case(record,directory,sim,audit)
            if result['status']!='complete':raise RuntimeError(result.get('error'))
        if (i+1)%20==0 or i in [0,len(records)-1]:
            print(f'LEGACY_WIDE {i+1}/{len(records)} {record["original_design_id"]}',flush=True)
    state=helper.load(directory/'state.json');state['status']='simulations_complete';helper.save(directory/'state.json',state)
    print('ALL_LEGACY_WIDEBAND_COMPLETE',flush=True)


if __name__=='__main__':main()
