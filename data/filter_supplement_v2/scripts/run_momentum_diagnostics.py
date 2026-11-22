"""13 real Momentum diagnostic sweeps, using a baseline-verified planar template.

The template reconstructs the five conductor polygons exported by ADS PCells.
Nominal polygons must exactly match the native export before any changed case.
61-point sweeps are diagnostics, not filled/interpolated 601-point EM labels.
"""
from pathlib import Path
import csv
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import supplement_filter_dataset as helper

ROOT=helper.ROOT
EMROOT=ROOT/'experiments/em_probe_20261024'


def read_citi(path):
    lines=[line.strip() for line in path.read_text(encoding='utf-8',errors='replace').splitlines()]
    declared=next(re.match(r'VAR freq MAG (\d+)',x) for x in lines if x.startswith('VAR freq MAG '))
    n=int(declared[1]);a=lines.index('VAR_LIST_BEGIN');b=lines.index('VAR_LIST_END')
    f=np.array([float(x) for x in lines[a+1:b] if x and not x.startswith('#')])
    names=[x.split()[1] for x in lines if x.startswith('DATA ')]
    blocks=[];active=None
    for line in lines[b+1:]:
        if line=='BEGIN':active=[]
        elif line=='END' and active is not None:
            blocks.append(np.array(active));active=None
        elif active is not None and line and not line.startswith('#'):
            vals=[float(x) for x in re.split(r'[,\s]+',line) if x]
            if len(vals)!=2:raise ValueError('Unexpected CITI RI record')
            active.append(complex(*vals))
    assert len(f)==n and len(blocks)==len(names) and all(len(x)==n for x in blocks)
    fields=dict(zip(names,blocks));s=np.empty((n,2,2),complex)
    for i in range(2):
        for j in range(2):s[:,i,j]=fields[f'S[{i+1},{j+1}]']
    for p in [1,2]:assert np.allclose(fields[f'PORTZ[{p}]'],50+0j,atol=1e-8,rtol=0)
    assert np.isfinite(s).all() and np.all(np.diff(f)>0)
    return f,s


def pair_outline(a,b):
    # Rectangles touch on a vertical boundary with positive overlap.
    x0,x1,y0,y1=a;xx0,x2,yy0,yy1=b
    assert abs(x1-xx0)<1e-9 and min(y1,yy1)>max(y0,yy0)
    return [(x1,y1),(x0,y1),(x0,y0),(x1,y0),(x1,yy0),(x2,yy0),(x2,yy1),(x1,yy1)]


def polygons(p):
    lf=p['Lfeed'];l1=p['L1'];l2=p['L2'];w1=p['Wc1'];w2=p['Wc2']
    # Actual PCell connection widths W1=32 and W2=31.5 stay fixed, as in the example.
    pin_in=32.;pin_out=31.5
    x1=lf+l1;x2=x1+l2
    bottom1=-pin_in/2;top1=bottom1+w1
    lower1top=bottom1-p['S1'];pin1y=lower1top-pin_out/2
    upper2bottom=pin1y-pin_in/2;upper2top=upper2bottom+w2
    lower2top=upper2bottom-p['S2'];lower2bottom=lower2top-w2
    pin2y=lower2top-pin_out/2;xr=2*x2;yr=2*pin2y
    one=[(x1,bottom1),(x1,top1),(lf,top1),(lf,16.),(0.,16.),(0.,0.),(0.,-16.)]
    two=pair_outline((lf,x1,lower1top-w1,lower1top),(x1,x2,upper2bottom,upper2top))
    three=pair_outline((x1,x2,lower2bottom,lower2top),(x2,xr-x1,yr-lower2top,yr-lower2bottom))
    three=three[6:]+three[:6]
    # Preserve the native polygon traversal at nominal geometry. Redundant
    # collinear vertices can alter mesher ordering even for an equal outline.
    changed=True
    while changed and len(three)>3:
        changed=False;clean=[]
        for i,b in enumerate(three):
            a=three[i-1];c=three[(i+1)%len(three)]
            cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
            if b==a or abs(cross)<1e-9:changed=True
            else:clean.append(b)
        three=clean
    mirror=lambda poly:[(xr-x,yr-y) for x,y in poly]
    four=mirror(two);four=four[4:]+four[:4]
    five=mirror(one);five=five[6:]+five[:6]
    return [one,two,three,four,five],(xr,yr)


def canonical(points):
    points=[tuple(round(float(v),8) for v in p) for p in points]
    if points[0]==points[-1]:points.pop()
    changed=True
    while changed and len(points)>3:
        changed=False;out=[]
        for i,b in enumerate(points):
            a=points[i-1];c=points[(i+1)%len(points)]
            cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
            if b==a or abs(cross)<1e-8:changed=True
            else:out.append(b)
        points=out
    return sorted(points)


def verify_template(native):
    source=[]
    for line in (native/'proj_a').read_text().splitlines():
        if line.startswith('ADD P1 :W0 '):
            source.append([(float(a),float(b)) for a,b in re.findall(r'([-+\d.]+),([-+\d.]+)',line)])
    generated,_=polygons(helper.BASE)
    assert len(source)==len(generated)==5
    for original,new in zip(source,generated):
        assert canonical(original)==canonical(new),(canonical(original),canonical(new))
    return {'native_exported_conductor_count':5,'parametric_conductor_count':5,
            'nominal_polygon_vertices_after_collinear_simplification_match':True,
            'physical_model':'Same ideal substrate and PCell pad-width conventions as native export; changed geometry is encoded directly into verified planar outlines.'}


def prepare(case,native,p,cells=30,points=61):
    case.mkdir(parents=True,exist_ok=True)
    for name in ['proj.ltd','proj.opt','proj.prt','proj.cfg']:
        shutil.copy2(native/name,case/name)
    poly,port2=polygons(p)
    with (case/'proj_a').open('w',encoding='ascii') as stream:
        stream.write('UNITS MIL,100;\n')
        for i,points_xy in enumerate(poly):
            points_xy=points_xy+[points_xy[0]]
            stream.write('ADD P1 :W0 '+' '.join(f'{x:.8f},{y:.8f}' for x,y in points_xy)+';\n')
            net='P1' if i==0 else 'P2' if i==4 else 'Unlabeled Interconnect'
            stream.write("  BEGIN_ASSOC\n    ADD N1 :F1.000000 :R0 :T12345 'net="+net+"' 0.00 0.00;\n  END_ASSOC\n")
    pins=ET.parse(native/'proj.pin')
    for pin in pins.getroot().findall('pin'):
        if pin.findtext('name')=='P2':
            pin.find('layout/shape/point/x').text=f'{port2[0]*25.4e-6:.16g}'
            pin.find('layout/shape/point/y').text=f'{port2[1]*25.4e-6:.16g}'
    pins.write(case/'proj.pin',encoding='utf-8',xml_declaration=True)
    text=(case/'proj.opt').read_text()
    text,n=re.subn(r'no_cells_per_wavelength \d+;',f'no_cells_per_wavelength {cells};',text)
    assert n==1
    (case/'proj.opt').write_text(text,encoding='ascii')
    (case/'proj.sti').write_text(f'START 9 STOP 15 LIN {points},\nEND;\n',encoding='ascii')
    helper.save(case/'parameters.json',{'geometry':p,'template_version':2,'port2_mil':port2,'mesh_cells_per_wavelength':cells,
                 'frequency_points':points,'purpose':'EM diagnostic only; no interpolation into dense EM labels'})


def main():
    result=helper.load(EMROOT/'rfpro_probe_result.json')
    completed=[r for r in result['run']['simulations'] if r['status']=='Completed']
    assert len(completed)==1
    native=Path(completed[0]['path'])/'work'
    f0,s0=read_citi(native/'proj.cti')
    assert len(f0)==601 and np.allclose(f0,np.linspace(9e9,15e9,601),rtol=0,atol=1.)
    template=verify_template(native)
    out=EMROOT/'momentum_diagnostics';out.mkdir(exist_ok=True)
    helper.save(out/'template_verification.json',template)
    audit=helper.module(ROOT/'scripts/validate_pretraining_physics.py','audit')
    metrics0,_=audit.measure(f0,s0)
    np.savez_compressed(out/'native_baseline_601.npz',frequency_Hz=f0,S_complex=s0)
    helper.save(out/'native_baseline_metrics.json',metrics0)
    engine=Path('E:/ADS2026/Momentum/2026.00/win32_64/bin/MomEngine.exe')
    from keysight.edatoolbox.ads import ADSEnvironment,CircuitSimulator
    env=ADSEnvironment(hpeesof_dir=str(helper.ADS))._env
    circuit=CircuitSimulator(hpeesof_dir=str(helper.ADS))
    # Match the frequency plan as well as geometry when testing the template.
    # Different sweep plans may change solver preprocessing/interpolation.
    reference=out/'native_reference_61'
    if not (reference/'response.npz').exists():
        prepare(reference,native,helper.BASE,30,61)
        shutil.copy2(native/'proj_a',reference/'proj_a')
        shutil.copy2(native/'proj.pin',reference/'proj.pin')
        with (reference/'solver_stdout.txt').open('w',encoding='utf-8') as stream:
            process=subprocess.run([str(engine),'-O','-3D','proj','proj'],cwd=reference,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
        if process.returncode!=0:raise RuntimeError('Native reference sweep failed')
        rf,rs=read_citi(reference/'proj.cti')
        assert len(rf)==61
        np.savez_compressed(reference/'response.npz',frequency_Hz=rf,S_complex=rs)
    reference_data=np.load(reference/'response.npz',allow_pickle=False)
    reference_s=reference_data['S_complex']
    template['native_61_vs_native_601_aligned_max_complex_S_difference']=float(abs(reference_s-s0[::10]).max())
    helper.save(out/'template_verification.json',template)
    records=[('baseline',helper.BASE.copy())]
    for name in helper.GEOM:
        values=[helper.BASE[name]*.98,helper.BASE[name]*1.02] if name in ['L1','L2'] else helper.BOUNDS[name]
        for side,value in zip(['low','high'],values):records.append((name+'_'+side,{**helper.BASE,name:round(value,8)}))
    # Mesh variation is diagnostic evidence, not a universal convergence certificate.
    records.append(('baseline_mesh45',helper.BASE.copy()))
    manifest={'status':'running','native_dense_baseline':str(native),'template':template,'runs':[]}
    helper.save(out/'manifest.json',manifest)
    comparisons=[]
    for did,p in records:
        cells=45 if did=='baseline_mesh45' else 30
        case=out/did
        oldpath=case/'result.json'
        if oldpath.exists() and helper.load(oldpath).get('status')=='complete':
            result=helper.load(oldpath)
        else:
            start=time.monotonic()
            reuse=(case/'response.npz').exists() and (case/'parameters.json').exists() and helper.load(case/'parameters.json')['geometry']==p and helper.load(case/'parameters.json').get('template_version')==2
            if not reuse:
                prepare(case,native,p,cells,61)
                with (case/'solver_stdout.txt').open('w',encoding='utf-8') as stream:
                    process=subprocess.run([str(engine),'-O','-3D','proj','proj'],cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=240)
                if process.returncode!=0:raise RuntimeError('Momentum failed: '+did)
            f,s=read_citi(case/'proj.cti')
            assert len(f)==61 and np.allclose(f,np.linspace(9e9,15e9,61),rtol=0,atol=1.)
            log=(case/'proj.log').read_text(encoding='utf-8',errors='replace')
            assert 'S-parameter simulation finished' in log or 'S-parameter simulation' in log
            m,tr=audit.measure(f,s)
            np.savez_compressed(case/'response.npz',frequency_Hz=f,S_complex=s,Y_dB=np.stack(tr[:2],axis=-1))
            helper.save(case/'metrics.json',m)
            if did=='baseline':
                delta=float(abs(s-reference_s).max())
                reference_metrics,_=audit.measure(f,reference_s)
                pb=(f>=11.6e9)&(f<=12.4e9)
                db_delta=float(abs(20*np.log10(abs(s[pb,1,0]))-20*np.log10(abs(reference_s[pb,1,0]))).max())
                center_delta=abs(m['f0_arithmetic_GHz']-reference_metrics['f0_arithmetic_GHz'])*1000
                bw_delta=abs(m['BW_GHz']-reference_metrics['BW_GHz'])*1000
                template['original_strict_1e_minus6_complex_repeatability_met']=delta<=1e-6
                template['engineering_reproduction']={'max_complex_S_difference':delta,'passband_max_S21_difference_dB':db_delta,
                    'center_difference_MHz':center_delta,'bandwidth_difference_MHz':bw_delta,
                    'limits':{'max_complex_S_difference':.001,'passband_difference_dB':.01,'center_and_BW_difference_MHz':1.},
                    'scope':'Geometrically identical nominal outlines, adequate for coarse diagnostic comparisons; not exact numerical equality. Limits are engineering choices, not published standards.'}
                if delta>.001 or db_delta>.01 or center_delta>1 or bw_delta>1:
                    helper.save(out/'template_verification.json',template)
                    raise RuntimeError('Parametric nominal reproduction fails engineering-level checks')
                template['parametric_61_vs_native_61_max_complex_S_difference']=delta
                helper.save(out/'template_verification.json',template)
            else:delta=None
            result={'status':'complete','design_id':did,'parameters':p,'n_frequency_points':61,'frequency_step_MHz':100,
                    'mesh_cells_per_wavelength':cells,'elapsed_seconds':time.monotonic()-start,'reused_completed_solver_data':reuse,
                    'native_baseline_aligned_max_complex_S_difference':delta,
                    'response':str(case/'response.npz'),'response_sha256':helper.sha(case/'response.npz'),
                    'geometry_sha256':helper.sha(case/'proj_a')}
            helper.save(case/'result.json',result)
        response=np.load(case/'response.npz',allow_pickle=False);f=response['frequency_Hz'];s=response['S_complex']
        m=helper.load(case/'metrics.json')
        # Actual paired circuit run with identical requested electrical parameters and reference impedances.
        pair=case/'circuit_pair';pair.mkdir(exist_ok=True)
        if not (pair/'response.npz').exists():
            source=(ROOT/'baseline'/next((ROOT/'baseline').glob('*.ckt')).name).read_text(encoding='utf-8')
            net,_,_=helper.patch(source,p,stop=15.,step=.01)
            circuit.run_netlist(net,output_dir=str(pair),working_dir=str(pair),netlist_file=str(pair/'design.ckt'),
                                output_file=str(pair/'simulator.log'),dataset_name='response')
            fc,sc,_=helper.read_ds(pair/'response.ds',15.,.01)
            np.savez_compressed(pair/'response.npz',frequency_Hz=fc,S_complex=sc)
        paired=np.load(pair/'response.npz',allow_pickle=False);fc=paired['frequency_Hz'];sc=paired['S_complex']
        cm,_=audit.measure(fc,sc)
        dbem=20*np.log10(np.maximum(abs(s[:,1,0]),1e-15));dbc=20*np.log10(np.maximum(abs(sc[::10,1,0]),1e-15))
        comparisons.append({'design_id':did,'mesh_cells_per_wavelength':cells,
                 'EM_points':len(f),'EM_frequency_step_MHz':100.,'EM_single_band_f0_GHz':m['f0_arithmetic_GHz'],
                 'circuit_single_band_f0_GHz':cm['f0_arithmetic_GHz'],
                 'f0_EM_minus_circuit_MHz':(m['f0_arithmetic_GHz']-cm['f0_arithmetic_GHz'])*1000 if m['f0_arithmetic_GHz'] is not None and cm['f0_arithmetic_GHz'] is not None else None,
                 'EM_BW_GHz':m['BW_GHz'],'circuit_BW_GHz':cm['BW_GHz'],
                 'sampled_S21_dB_MAE_over_9_15':float(abs(dbem-dbc).mean()),
                 'EM_max_singular_value':m['sampled_passivity_max_singular_value'],
                 'EM_reciprocity_max_error':m['reciprocity_max_abs_error'],
                 'circuit_pair_actual_simulation':True})
        manifest['runs'].append(result);helper.save(out/'manifest.json',manifest)
        print('EM_COMPLETE '+did+' '+str(round(result['elapsed_seconds'],2))+'s',flush=True)
    with (out/'circuit_vs_em_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    manifest.update(status='complete',unique_geometry_designs=13,coarse_sweeps=14,dense_native_baseline_points=601,
             note='Coarse 100 MHz sweeps diagnose parameter trends; not sufficient alone for narrow-notch or final band-edge certification. EM and circuit labels remain separate.')
    helper.save(out/'manifest.json',manifest)
    print('MOMENTUM_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':main()
