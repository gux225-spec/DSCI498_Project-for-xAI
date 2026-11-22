"""Load a released split without ADS, fitting anything, or changing data splits."""
from pathlib import Path
import argparse
import json
import numpy as np


def load_split(release,split='train',band='core',preset='geometry_plus_negative_controls',control_replicate=0,normalize=False):
    release=Path(release)
    if split not in ['train','validation','test'] or band not in ['core','wideband']:raise ValueError('Invalid split/band')
    roles=json.loads((release/'feature_roles.json').read_text(encoding='utf-8'))
    names=roles['input_presets'][preset]
    with np.load(release/f'data/{band}_{split}.npz',allow_pickle=False) as data:
        all_names=data['candidate_feature_names'].tolist()
        idx=[all_names.index(name) for name in names]
        x=data['X_candidate'][:,idx].copy()
        if control_replicate not in [0,1,2]:raise ValueError('Control replicate must be 0, 1 or 2')
        if control_replicate:
            for j,name in enumerate(data['negative_control_names'].tolist()):
                if name in names:x[:,names.index(name)]=data['negative_controls_replicates'][control_replicate,:,j]
        if normalize:
            all_scalers=json.loads((release/'scalers_train_only.json').read_text(encoding='utf-8'))
            scaler=all_scalers['candidates'];means=np.array(scaler['mean'])[idx].copy();sds=np.array(scaler['effective_std'])[idx].copy()
            if control_replicate:
                nuisance=all_scalers['negative_control_replicates'][str(control_replicate)]
                for j,name in enumerate(data['negative_control_names'].tolist()):
                    if name in names:
                        means[names.index(name)]=nuisance['mean'][j];sds[names.index(name)]=nuisance['effective_std'][j]
            x=(x-means)/sds
        return {'X_design':x,'feature_names':names,'frequency_Hz':data['frequency_Hz'].copy(),
                'S_complex':data['S_complex'].copy(),'Y_dB':data['Y_dB'].copy(),
                'design_id':data['design_id'].copy(),'split':split,'band':band,
                'S21_group_delay_ns':data['S21_group_delay_ns'].copy()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path,required=True)
    ap.add_argument('--split',default='train',choices=['train','validation','test'])
    ap.add_argument('--band',default='core',choices=['core','wideband'])
    ap.add_argument('--preset',default='geometry_plus_negative_controls')
    args=ap.parse_args()
    item=load_split(args.release,args.split,args.band,args.preset)
    print(json.dumps({'split':args.split,'band':args.band,'features':item['feature_names'],
          'X_design_shape':list(item['X_design'].shape),'frequency_shape':list(item['frequency_Hz'].shape),
          'S_complex_shape':list(item['S_complex'].shape),'Y_dB_shape':list(item['Y_dB'].shape),
          'warning':'Frequency is a query coordinate; this does not count frequency rows as independent geometries.'},indent=2))


if __name__=='__main__':main()
