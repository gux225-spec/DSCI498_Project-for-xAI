"""Read-only pretraining S-parameter/metric audit. No ADS runs or model fitting.

All reported bandwidths/threshold crossings are sampled-grid estimates.
Uses an independent implementation, not the release metric extractor.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np

ROOT = Path('D:/Downloads/ADS_XAI_runs')
DATASET = Path(json.loads((ROOT/'datasets/latest_full_dataset.json').read_text())['directory'])
RELEASE = DATASET/'release'
OUT = ROOT/'analysis/pretraining_physics_20261024'
FIXED_PASS = (11.6, 12.4)  # Existing provisional engineering evaluation window.
SEARCH = (10.5, 13.5)
TOL = 1e-10  # Numerical tolerance for this ideal double-precision circuit data.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def window(f, y, lo, hi):
    """Include exact boundary positions; interpolate in dB when necessary."""
    ix = (f > lo) & (f < hi)
    return np.r_[lo, f[ix], hi], np.r_[np.interp(lo, f, y), y[ix], np.interp(hi, f, y)]


def intervals(f, y, threshold):
    """All connected y>=threshold components; never bridge rejected gaps."""
    good = y >= threshold
    starts = np.flatnonzero(good & ~np.r_[False, good[:-1]])
    ends = np.flatnonzero(good & ~np.r_[good[1:], False])
    result = []
    for a, b in zip(starts, ends):
        left = float(f[a]) if a == 0 else float(f[a-1] + (threshold-y[a-1])*(f[a]-f[a-1])/(y[a]-y[a-1]))
        right = float(f[b]) if b == len(f)-1 else float(f[b] + (threshold-y[b])*(f[b+1]-f[b])/(y[b+1]-y[b]))
        result.append({'lower_GHz': left, 'upper_GHz': right, 'bandwidth_GHz': right-left,
                       'left_clipped': bool(a == 0), 'right_clipped': bool(b == len(f)-1)})
    return result


def group_delay(f_Hz, s21):
    phase = np.unwrap(np.angle(s21))
    return -np.gradient(phase, 2*np.pi*f_Hz, edge_order=2)*1e9, phase


def csv_s(path):
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
    f = np.array([float(r['frequency']) for r in rows])
    s = np.empty((len(rows), 2, 2), dtype=complex)
    for o, i, name in [(0,0,'S11'), (1,0,'S21'), (0,1,'S12'), (1,1,'S22')]:
        s[:,o,i] = [complex(float(r[name+'_real']), float(r[name+'_imag'])) for r in rows]
    return f, s, rows


def measure(f_Hz, s):
    f = f_Hz/1e9
    d21 = 20*np.log10(np.maximum(np.abs(s[:,1,0]), 1e-15))
    d11 = 20*np.log10(np.maximum(np.abs(s[:,0,0]), 1e-15))
    fs, ys = window(f, d21, *SEARCH)
    peak = float(ys.max())
    bands = intervals(fs, ys, peak-3)
    one = len(bands) == 1 and not (bands[0]['left_clipped'] or bands[0]['right_clipped'])
    b = bands[0] if one else None
    f0 = (b['lower_GHz']+b['upper_GHz'])/2 if b else None
    fp, yp = window(f, d21, *FIXED_PASS)
    fr, rl = window(f, -d11, *FIXED_PASS)
    rl_bands = intervals(fr, rl, 15.0)
    # Both reference impedances are real 50 ohms; power-wave S conventions apply.
    gram = np.swapaxes(s.conj(), -1, -2) @ s
    sigma = np.linalg.svd(s, compute_uv=False)
    tau, phase = group_delay(f_Hz, s[:,1,0])
    phase_adj = np.angle(s[1:,1,0]*s[:-1,1,0].conj())
    pb = (f >= FIXED_PASS[0]) & (f <= FIXED_PASS[1])
    i = np.flatnonzero(pb)
    # Independent local-ratio implementation of the same central phase secant.
    tau_local = -np.angle(s[i+1,1,0]*s[i-1,1,0].conj())/(2*np.pi*(f_Hz[i+1]-f_Hz[i-1]))*1e9
    return {
        'n_connected_3dB_bands': len(bands),
        'single_band_center_valid': one,
        'f_peak_sampled_GHz': float(fs[np.argmax(ys)]),
        'f_lower_GHz': b['lower_GHz'] if b else None,
        'f_upper_GHz': b['upper_GHz'] if b else None,
        'f0_arithmetic_GHz': f0,
        'f0_geometric_GHz': math.sqrt(b['lower_GHz']*b['upper_GHz']) if b else None,
        'BW_GHz': b['bandwidth_GHz'] if b else None,
        'FBW_arithmetic_percent': 100*b['bandwidth_GHz']/f0 if b else None,
        'IL_at_arithmetic_center_dB': float(-np.interp(f0, f, d21)) if b else None,
        'IL_best_search_window_sampled_dB': -peak,
        'IL_fixed_pass_best_dB': float(-yp.max()),
        'IL_fixed_pass_worst_dB': float(-yp.min()),
        'ripple_fixed_pass_dB': float(np.ptp(yp)),
        'return_loss_fixed_pass_worst_dB': float(rl.min()),
        'return_loss_15dB_intervals_fixed_pass': rl_bands,
        'return_loss_15dB_total_coverage_fixed_pass_GHz': sum(v['bandwidth_GHz'] for v in rl_bands),
        'attenuation_at_10GHz_dB': float(-np.interp(10, f, d21)),
        'attenuation_at_14GHz_dB': float(-np.interp(14, f, d21)),
        'low_stopband_worst_attenuation_dB': float(-d21[(f>=9)&(f<=10.8)].max()),
        'high_stopband_worst_attenuation_dB': float(-d21[(f>=13.2)&(f<=15)].max()),
        'group_delay_fixed_pass_min_ns': float(tau[pb].min()),
        'group_delay_fixed_pass_max_ns': float(tau[pb].max()),
        'group_delay_fixed_pass_mean_ns': float(tau[pb].mean()),
        'group_delay_fixed_pass_ripple_ns': float(np.ptp(tau[pb])),
        'group_delay_local_ratio_vs_unwrapped_max_error_ns': float(np.max(np.abs(tau_local-tau[pb]))),
        'adjacent_S21_phase_step_max_deg': float(np.rad2deg(np.abs(phase_adj).max())),
        'port1_power_balance_max_abs_error': float(np.abs(np.abs(s[:,0,0])**2+np.abs(s[:,1,0])**2-1).max()),
        'full_twoport_unitarity_max_abs_error': float(np.abs(gram-np.eye(2)).max()),
        'sampled_passivity_max_singular_value': float(sigma.max()),
        'sampled_passivity_positive_excess': float(np.maximum(sigma-1, 0).max()),
        'reciprocity_max_abs_error': float(np.abs(s[:,0,1]-s[:,1,0]).max()),
        'symmetry_S11_minus_S22_max_abs_error': float(np.abs(s[:,0,0]-s[:,1,1]).max()),
        'all_3dB_intervals': bands,
    }, (d21, d11, tau, phase)


def formula_controls():
    f = np.linspace(9e9, 15e9, 601)
    delay = 0.5e-9
    t, _ = group_delay(f, np.exp(-2j*np.pi*f*delay))
    gd_error = float(np.max(np.abs(t-delay*1e9)))
    # Unitary reciprocal twoport with reflection: insertion loss can be 3.0103 dB.
    s = np.array([[1j, 1], [1, 1j]], dtype=complex)/np.sqrt(2)
    lossless_IL = float(-20*np.log10(abs(s[1,0])))
    unitary = float(np.max(np.abs(s.conj().T@s-np.eye(2))))
    # A two-port isolator is passive but not reciprocal.
    iso = np.array([[0., 0.], [1., 0.]])
    assert np.linalg.svd(iso, compute_uv=False).max() <= 1
    assert iso[0,1] != iso[1,0]
    # Test disconnected intervals, interpolation and clipped boundaries.
    comps = intervals(np.arange(5.), np.array([0.,2.,0.,2.,0.]), 1)
    assert len(comps) == 2 and comps[0]['lower_GHz'] == .5 and comps[1]['upper_GHz'] == 3.5
    clipped = intervals(np.arange(3.), np.array([2.,2.,0.]), 1)
    assert clipped[0]['left_clipped'] and clipped[0]['upper_GHz'] == 1.5
    assert gd_error < 1e-10 and unitary < 1e-12 and abs(lossless_IL-3.010299956639812)<1e-12
    return {'synthetic_delay_known_ns': .5, 'synthetic_delay_max_abs_error_ns': gd_error,
            'lossless_reflecting_network_IL_dB': lossless_IL,
            'lossless_reflecting_network_unitarity_error': unitary,
            'passive_nonreciprocal_counterexample_singular_values': np.linalg.svd(iso, compute_uv=False).tolist(),
            'disconnected_and_truncated_band_controls': 'passed'}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controls = formula_controls()
    old = {r['design_id']:r for r in csv.DictReader((RELEASE/'data/design_summary.csv').open(encoding='utf-8-sig', newline=''))}
    rows = []
    sources = []
    comparison_errors = {}
    agreement = {'f0_arithmetic_GHz':'f0_arithmetic_GHz', 'BW_GHz':'BW_GHz',
                 'ripple_fixed_pass_dB':'passband_ripple_dB',
                 'return_loss_fixed_pass_worst_dB':'passband_return_loss_worst_dB',
                 'low_stopband_worst_attenuation_dB':'low_stopband_attenuation_worst_dB',
                 'high_stopband_worst_attenuation_dB':'high_stopband_attenuation_worst_dB'}
    null_status_agreement = True
    db_max_error = 0.
    for split in ['train','validation','test']:
        path = RELEASE/f'data/{split}.npz'
        sources.append({'path':str(path), 'sha256':sha(path)})
        with np.load(path, allow_pickle=False) as data:
            f = data['frequency_Hz']
            assert f.shape == (601,) and np.allclose(f, np.linspace(9e9,15e9,601), rtol=0, atol=1e-5)
            for n, did in enumerate(data['design_id']):
                did = str(did)
                s = data['S_complex'][n]
                assert np.isfinite(s).all()
                m, traces = measure(f, s)
                raw_dB = np.stack([traces[0],traces[1]],axis=-1)
                db_max_error = max(db_max_error,float(np.abs(raw_dB-data['Y_dB'][n]).max()))
                row = {'design_id':did,'split':split,**dict(zip(['L1_mil','L2_mil','S1_mil','S2_mil'], map(float,data['geometry_mil'][n]))),**m}
                rows.append(row)
                ref = old[did]
                null_status_agreement &= bool(ref['f0_arithmetic_GHz']) == m['single_band_center_valid']
                assert int(ref['n_connected_3dB_bands']) == m['n_connected_3dB_bands']
                for name, other in agreement.items():
                    if m[name] is not None:
                        err = abs(float(ref[other])-m[name])
                        comparison_errors[name] = max(comparison_errors.get(name,0.),err)
    assert len(rows) == 480 and len(set(r['design_id'] for r in rows)) == 480
    assert null_status_agreement and max(comparison_errors.values()) < 1e-10 and db_max_error < 1e-10
    write_csv(OUT/'all_design_physical_metrics.csv',rows)
    base_path = DATASET/'runs/b_ofat_000/attempt_001/sparams.csv'
    bf, bs, _ = csv_s(base_path)
    baseline, bt = measure(bf, bs)
    sources.append({'path':str(base_path),'sha256':sha(base_path)})
    dump(OUT/'baseline_physical_metrics.json',baseline)
    write_csv(OUT/'baseline_phase_group_delay.csv',[
        {'frequency_GHz':float(f/1e9), 'S21_dB':float(bt[0][i]), 'S11_dB':float(bt[1][i]),
         'S21_unwrapped_phase_rad':float(bt[3][i]),'group_delay_ns':float(bt[2][i]),
         'inside_fixed_evaluation_passband':bool(FIXED_PASS[0]<=f/1e9<=FIXED_PASS[1])}
        for i,f in enumerate(bf)])
    fine_checks = []
    for directory in sorted((DATASET/'fine_runs').iterdir()):
        coarse_path = DATASET/f'runs/{directory.name}/attempt_001/sparams.csv'
        fine_path = directory/'attempt_001/sparams.csv'
        cf, cs, _ = csv_s(coarse_path)
        ff, ss, _ = csv_s(fine_path)
        cm, ct = measure(cf,cs)
        fm, ft = measure(ff,ss)
        assert len(ff) == 3001
        pb = (cf/1e9 >= FIXED_PASS[0]) & (cf/1e9 <= FIXED_PASS[1])
        nearest = np.searchsorted(ff,cf)
        assert np.max(np.abs(ff[nearest]-cf)) < 1.
        dt = ct[2][pb]-ft[2][nearest[pb]]
        edge_diff = []
        if len(cm['all_3dB_intervals']) == len(fm['all_3dB_intervals']):
            for a,b in zip(cm['all_3dB_intervals'],fm['all_3dB_intervals']):
                edge_diff.extend([abs(a[k]-b[k])*1e3 for k in ['lower_GHz','upper_GHz']])
        fine_checks.append({'design_id':directory.name,
             'coarse_step_MHz':10.,'fine_step_MHz':2.,
             'group_delay_coarse_central_aperture_MHz':20.,'group_delay_fine_central_aperture_MHz':4.,
             'group_delay_fixed_pass_max_abs_difference_ps':float(np.abs(dt).max()*1000),
             'group_delay_fixed_pass_RMS_difference_ps':float(np.sqrt(np.mean(dt**2))*1000),
             'group_delay_ripple_coarse_ns':cm['group_delay_fixed_pass_ripple_ns'],
             'group_delay_ripple_fine_ns':fm['group_delay_fixed_pass_ripple_ns'],
             'ripple_fixed_pass_coarse_dB':cm['ripple_fixed_pass_dB'],
             'ripple_fixed_pass_fine_dB':fm['ripple_fixed_pass_dB'],
             'RL_worst_fixed_pass_coarse_dB':cm['return_loss_fixed_pass_worst_dB'],
             'RL_worst_fixed_pass_fine_dB':fm['return_loss_fixed_pass_worst_dB'],
             'all_sorted_3dB_edges_max_abs_difference_MHz':max(edge_diff) if edge_diff else None})
        sources.append({'path':str(fine_path),'sha256':sha(fine_path)})
    write_csv(OUT/'fine_grid_physics_comparison.csv',fine_checks)
    summary = {
        'created_UTC':datetime.now(timezone.utc).isoformat(),
        'assessment':'Share with caveats: verified within existing lossless ADS circuit model and sampled frequency domain',
        'source_dataset':str(DATASET),'n_designs':len(rows),'n_frequency_points_per_design':601,
        'split_counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},
        'single_connected_passband_designs':sum(r['single_band_center_valid'] for r in rows),
        'multiple_passband_designs':sum(r['n_connected_3dB_bands']>1 for r in rows),
        'numerical_tolerance':TOL,
        'maxima':{k:max(r[k] for r in rows) for k in [
           'port1_power_balance_max_abs_error','full_twoport_unitarity_max_abs_error',
           'sampled_passivity_max_singular_value','sampled_passivity_positive_excess',
           'reciprocity_max_abs_error','symmetry_S11_minus_S22_max_abs_error',
           'adjacent_S21_phase_step_max_deg','group_delay_local_ratio_vs_unwrapped_max_error_ns']},
        'passes':{k:sum(r[k]<=TOL for r in rows) for k in [
           'port1_power_balance_max_abs_error','full_twoport_unitarity_max_abs_error',
           'sampled_passivity_positive_excess','reciprocity_max_abs_error']},
        'fixed_passband_window_GHz':list(FIXED_PASS),'main_response_search_window_GHz':list(SEARCH),
        'baseline':baseline,
        'independent_reconciliation_max_abs_errors':comparison_errors,
        'complex_vs_stored_dB_max_abs_error_dB':db_max_error,
        'multiband_null_status_matches_release':null_status_agreement,
        'fine_grid_checks':fine_checks,
        'fine_grid_max_group_delay_difference_ps':max(r['group_delay_fixed_pass_max_abs_difference_ps'] for r in fine_checks),
        'fine_grid_max_edge_difference_MHz':max(r['all_sorted_3dB_edges_max_abs_difference_MHz'] for r in fine_checks),
        'formula_controls':controls,
        'spurious_near_2f0_status':'not assessed: no source data beyond 15 GHz; extend to at least 25 GHz in a distinct simulation version',
        'phase_prediction_status':'current 2-dB-output MLP cannot predict group delay; need complex/phase outputs before claiming a group-delay surrogate',
        'not_verified':['layout full-wave accuracy','measured hardware','lossy-material behavior','new independent variables',
                        'LineCalc modal-feature calibration','resonator coupling k identification','external Q identification',
                        'continuous-frequency passivity or out-of-domain causality','fine-grid convergence of all 480 designs'],
        'sources':sources,
        'no_training_or_new_simulations':True,
    }
    dump(OUT/'validation_summary.json',summary)
    definitions = {
       'f0_and_bandwidth':'Relative sampled S21 peak minus 3.000 dB, crossings interpolated in dB in 10.5-13.5 GHz. Arithmetic center retained; geometric center separately reported. No single overall f0/BW for disconnected bands.',
       'FBW':'100*(fH-fL)/((fL+fH)/2); arithmetic convention explicit, not an interchangeable geometric convention.',
       'return_loss':'RL=-20log10(abs(S11)); worst RL = minimum RL in fixed 11.6-12.4 GHz. RL>=15 dB intervals reported separately without bridging gaps. Width coverage is clipped to this window.',
       'ripple':'max S21_dB - min S21_dB in fixed 11.6-12.4 GHz, not in the self-selected -3 dB interval. Fixed window is a provisional existing evaluation policy, not an external engineering specification.',
       'attenuation':'A(f)=-S21_dB(f) at 10 and 14 GHz, plus existing low/high-stopband worst values.',
       'insertion_loss':'IL(f)=-S21_dB(f), meaningful even if lossless through reflection. Report best, worst and center separately. Not a measure of dissipation alone.',
       'group_delay':'tau=-d unwrap(arg S21)/d(2*pi*f); numpy central difference on interior points, second-order endpoint rule. Passband statistics use only 11.6-12.4 GHz. No smoothing. Coarse aperture 20 MHz; fine aperture 4 MHz. Fine-grid check covers five existing diagnostic designs only.',
       'energy':'For current ideal lossless circuit and 50-ohm power waves, |S11|^2+|S21|^2=1; also check full S^H S=I. Future loss/radiation can reduce retained port power.',
       'passivity':'max singular value of full complex S <=1+tolerance at sampled frequencies, with real equal 50-ohm references. Not a proof between samples or outside the sampled band.',
       'reciprocity':'S12=S21 is expected for the present reciprocal microstrip network; passivity alone does not imply reciprocity.',
       'coupled_line_proxy':'(Z0e-Z0o)/(Z0e+Z0o) may be used as a coupled-line coupling proxy under its stated assumptions; do not equate it automatically to resonator coupling-matrix k.',
       'external_Q':'Requires a documented external-coupling extraction topology or constrained circuit calibration; S1 alone is insufficient. Do not replace it with full-filter f0/BW.',
       'dB_floor':'Only log evaluation uses magnitude floor 1e-15 (-300 dB). Raw complex S retained. No target clipping or deletion.',
    }
    dump(OUT/'metric_definitions.json',definitions)
    print(json.dumps({k:summary[k] for k in ['n_designs','single_connected_passband_designs','multiple_passband_designs','maxima','passes','fine_grid_max_group_delay_difference_ps','fine_grid_max_edge_difference_MHz']},indent=2))
    print('Baseline: '+json.dumps(baseline,ensure_ascii=False))
    print('Saved: '+str(OUT))


if __name__ == '__main__':
    main()
