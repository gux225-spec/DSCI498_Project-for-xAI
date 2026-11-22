"""Focused real EM mesh checks; keep them separate from dense response labels."""
from pathlib import Path
import subprocess
import json
import time
import numpy as np
import run_momentum_diagnostics as em


def main():
    out=em.EMROOT/'momentum_diagnostics'
    manifest=em.helper.load(out/'manifest.json');native=Path(manifest['native_dense_baseline'])
    from keysight.edatoolbox.ads import ADSEnvironment
    env=ADSEnvironment(hpeesof_dir=str(em.helper.ADS))._env
    engine=Path('E:/ADS2026/Momentum/2026.00/win32_64/bin/MomEngine.exe')
    cases=[]
    for cells in [60,90]:
        case=out/f'baseline_focused_mesh{cells}'
        if not (case/'response.npz').exists():
            em.prepare(case,native,em.helper.BASE,cells,61)
            # Keep 15 GHz in the stimulus so the mesh upper-frequency basis is unchanged.
            (case/'proj.sti').write_text('START 9 STOP 9 LIN 1,\nSTART 11.3 STOP 13.0 LIN 18,\nSTART 15 STOP 15 LIN 1,\nEND;\n',encoding='ascii')
            metadata=em.helper.load(case/'parameters.json');metadata['frequency_points']=20
            metadata['actual_frequency_plan']='9 GHz; 11.3-13.0 GHz at 100 MHz; 15 GHz'
            em.helper.save(case/'parameters.json',metadata)
            started=time.monotonic()
            with (case/'solver_stdout.txt').open('w',encoding='utf-8') as stream:
                process=subprocess.run([str(engine),'-O','-3D','proj','proj'],cwd=case,env=env,
                                       stdout=stream,stderr=subprocess.STDOUT,timeout=360)
            if process.returncode!=0:raise RuntimeError('Refined Momentum failed')
            f,s=em.read_citi(case/'proj.cti')
            assert len(f)==20 and f[0]==9e9 and f[-1]==15e9
            y=20*np.log10(np.maximum(abs(s[:,1,0]),1e-15))
            np.savez_compressed(case/'response.npz',frequency_Hz=f,S_complex=s,S21_dB=y)
            em.helper.save(case/'focused_result.json',{'status':'complete','mesh_cells_per_wavelength':cells,
                'actual_frequency_points':20,'focus_GHz':[11.3,13.0],'max_mesh_frequency_GHz':15,
                'elapsed_seconds':time.monotonic()-started,
                'scope':'Focused baseline mesh diagnostic, not a dense 9-15 GHz response'})
            print(f'FOCUSED_EM_COMPLETE mesh={cells}',flush=True)
        cases.append(case)
        metadata=em.helper.load(case/'parameters.json');metadata['frequency_points']=20
        metadata['actual_frequency_plan']='9 GHz; 11.3-13.0 GHz at 100 MHz; 15 GHz'
        em.helper.save(case/'parameters.json',metadata)
    results=[]
    previous=out/'baseline/response.npz'
    prev_cells=30
    for cells,path in [(45,out/'baseline_mesh45/response.npz'),(60,cases[0]/'response.npz'),(90,cases[1]/'response.npz')]:
        a=np.load(previous,allow_pickle=False);b=np.load(path,allow_pickle=False)
        fa=a['frequency_Hz'];fb=b['frequency_Hz']
        shared=[]
        for j,f in enumerate(fb):
            i=int(np.argmin(abs(fa-f)))
            if abs(fa[i]-f)<1:shared.append((i,j))
        ia=np.array([i for i,j in shared]);ib=np.array([j for i,j in shared])
        pb=(fb[ib]>=11.6e9)&(fb[ib]<=12.4e9)
        sa=a['S_complex'][ia];sb=b['S_complex'][ib]
        dbdiff=float(abs(20*np.log10(abs(sa[pb,1,0]))-20*np.log10(abs(sb[pb,1,0]))).max())
        def edges(item):
            f=item['frequency_Hz']/1e9;y=20*np.log10(np.maximum(abs(item['S_complex'][:,1,0]),1e-15))
            focus=(f>=11.3)&(f<=13.)
            comp=em.helper.module(em.ROOT/'scripts/validate_pretraining_physics.py','audit').intervals(f[focus],y[focus],float(y[focus].max()-3))
            if len(comp)==1 and not(comp[0]['left_clipped'] or comp[0]['right_clipped']):
                v=comp[0];return (v['lower_GHz']+v['upper_GHz'])/2,v['bandwidth_GHz']
            return None,None
        ca,ba=edges(a);cb,bb=edges(b)
        results.append({'mesh_from':prev_cells,'mesh_to':cells,'shared_frequency_points':len(shared),
            'fixed_passband_shared_S21_max_difference_dB':dbdiff,'shared_max_complex_difference':float(abs(sa-sb).max()),
            'center_from_GHz':ca,'center_to_GHz':cb,
            'center_difference_MHz':abs(cb-ca)*1000 if ca is not None and cb is not None else None,
            'BW_difference_MHz':abs(bb-ba)*1000 if ba is not None and bb is not None else None})
        previous=path;prev_cells=cells
    last=results[-1]
    quality={'comparisons':results,'provisional_focus_limits':{'S21_difference_dB':.02,'center_difference_MHz':2.},
         'last_pair_within_focus_limits':last['fixed_passband_shared_S21_max_difference_dB']<=.02 and last['center_difference_MHz'] is not None and last['center_difference_MHz']<=2,
         'scope':'One baseline, shared in-band frequencies. Does not certify all 13 geometries, narrow S11 minima or unobserved frequency intervals.'}
    em.helper.save(out/'focused_mesh_checks.json',quality)
    print(json.dumps(quality,indent=2),flush=True)


if __name__=='__main__':main()
