"""Re-analyze saved outputs without running or modifying the simulations.

Run from any directory: python review_2026-11-23/analyze_results.py
Requires numpy, scipy and matplotlib. Derived outputs stay in this folder.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import platform
import re

import numpy as np
import scipy
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
PHI0 = 2.067833848e-15
FP = np.sqrt(2*np.pi*1e6/(PHI0*.06))/(2*np.pi)


def read_csv(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def write_csv(name, rows):
    with (HERE/name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


inputs = sorted([*ROOT.glob('*.py'), *ROOT.glob('*.png'),
                 *(ROOT/'iv_fluxon_results').glob('*'),
                 *(ROOT/'ljj_capacitive_jtl_results').glob('*/*')])
inputs = [p for p in inputs if p.is_file() and p.name != '.DS_Store']
manifest = [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
write_csv('input_manifest.csv', manifest)
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text())

iv = []
iv_arrays = {}
for path in sorted((ROOT/'iv_fluxon_results').glob('*.csv')):
    if '_spectrum_peaks_' in path.name:
        continue
    d = read_csv(path)
    assert all(np.all(np.isfinite(d[k])) for k in d.dtype.names)
    n = int(re.search(r'_N(\d+)_', path.name)[1])
    i = d['I_normalized'] if 'I_normalized' in d.dtype.names else d['J_bias_A_per_cm2']/100
    v = d['V_uV_Nfl_1'] if 'V_uV_Nfl_1' in d.dtype.names else d['V_normalized_Nfl_1'] * PHI0*FP*1e6
    fj = d['josephson_frequency_GHz_Nfl_1']
    assert np.allclose(v, fj*PHI0*1e15, rtol=1e-10, atol=1e-8)
    if 'radiation_frequency_GHz_Nfl_1' in d.dtype.names:
        assert np.allclose(d['radiation_frequency_GHz_Nfl_1'], fj/2)
    target = 10 if n == 65 else 5
    j = int(np.argmin(abs(fj-target)))
    active = np.flatnonzero(v > 1)
    iv.append(dict(file=path.name, N=n, points=len(d), di=float(np.median(np.diff(i))),
                   onset_i_at_V_gt_1uV=float(i[active[0]]) if len(active) else None,
                   target_fJ_GHz=target, closest_i=float(i[j]),
                   closest_V_uV=float(v[j]), closest_fJ_GHz=float(fj[j]),
                   max_V_uV=float(v.max())))
    iv_arrays[path.name] = (d,i,v,fj)
write_csv('iv_summary.csv', iv)

jtl = []
jtl_arrays = {}
for path in sorted((ROOT/'ljj_capacitive_jtl_results').glob('*/jtl_output.csv')):
    meta = json.loads((path.parent/'parameters.json').read_text())
    p,s = meta['parameters'], meta['physical_scales']
    d = read_csv(path)
    assert all(np.all(np.isfinite(d[k])) for k in d.dtype.names)
    t,v,q = d['tau'], d['V_out_normalized'],d['Delta_theta_N_over_2pi']
    assert np.all(np.diff(t)>0)
    assert np.allclose(d['V_out_mV'],v*s['V_SCALE_V']*1e3)
    peaks,_ = find_peaks(v, height=.5, prominence=.5, distance=60)
    tp = t[peaks]
    mid = (tp[1:]+tp[:-1])/2
    cycle_areas = np.diff(np.interp(mid,t,q))
    dt = np.diff(tp)
    label = f"M{p['M']}, cell={p['fluxon_cells'][0]}, {path.parent.name[-6:]}"
    jtl.append(dict(run=path.parent.name, M=p['M'], initial_cell=p['fluxon_cells'][0],
                    kappa=p['kappa'], Ic_uA=s['IC_L_PHYS_A']*1e6,
                    C_pF=s['C_L_PHYS_F']*1e12, fp_GHz=s['OMEGA_PL_rad_s']/(2*np.pi*1e9),
                    duration_ns=t[-1]/s['OMEGA_PL_rad_s']*1e9,
                    full_local_peaks=len(peaks), final_Q=float(q[-1]),
                    max_V_uV=float(d['V_out_mV'].max()*1000),
                    period_tau=float(np.mean(dt)),
                    repetition_GHz=float(s['OMEGA_PL_rad_s']/np.mean(dt)/1e9),
                    cycle_area_min_Phi0=float(cycle_areas.min()),
                    cycle_area_max_Phi0=float(cycle_areas.max()),
                    total_integral_minus_Q=float(np.trapezoid(v,t)/(2*np.pi)-q[-1]),
                    peak_tau=tp.tolist()))
    jtl_arrays[path.parent.name] = (d,p,s,label)
write_csv('jtl_summary.csv', [{k:v for k,v in row.items() if k!='peak_tau'} for row in jtl])

spectra = []
for path in sorted((ROOT/'iv_fluxon_results').glob('*.npz')):
    with np.load(path, allow_pickle=False) as z:
        f = z['frequency_GHz']
        power = z['power_W'] if 'power_W' in z else z['power']
        assert power.shape == (len(z['currents_norm']),len(f))
        assert np.all(np.isfinite(power)) and np.all(power>=0)
        a,b = np.unravel_index(np.argmax(power),power.shape)
        spectra.append(dict(file=path.name, rows=power.shape[0], bins=power.shape[1],
                            df_GHz=float(f[1]-f[0]), max_frequency_GHz=float(f[-1]),
                            units='W_equivalent_50ohm' if 'power_W' in z else 'normalized_voltage_squared',
                            strongest_i=float(z['currents_norm'][a]),
                            strongest_frequency_GHz=float(f[b]), strongest_value=float(power[a,b]),
                            signal_mode=str(z['spectrum_signal_mode'])))
write_csv('spectrum_summary.csv',spectra)

groups = {}
for row in manifest:
    if row['path'].endswith('.csv'):
        groups.setdefault(row['sha256'], []).append(row['path'])
duplicates = [v for v in groups.values() if len(v)>1]
latest = next(k for k in iv_arrays if k.endswith('151709.csv'))
d,i,v,fj = iv_arrays[latest]
zpath = ROOT/'iv_fluxon_results'/latest.replace('.csv','_spectrum_Nfl1.npz')
with np.load(zpath) as z:
    freq,power = z['frequency_GHz'],z['power_W']
    peakfreq = z['peak_frequency_GHz']
    peakpower = z['peak_power_W']
    fine_power = power.copy()
    spectrum_currents = z['currents_norm'].copy()
    current_uA = z['I_cell_uA'].copy()

selected = []
for target in [.27,.4,.5,.6,.62,.7,.71,.75,.77,.775,.8]:
    a = int(np.argmin(abs(i-target)))
    selected.append(dict(i=float(i[a]), I_cell_uA=float(d['I_cell_uA'][a]),
                         V_uV=float(v[a]), fJ_GHz=float(fj[a]),
                         exported_fJ_over_2_GHz=float(fj[a]/2),
                         strongest_FFT_GHz=float(peakfreq[a]),
                         strongest_FFT_W=float(peakpower[a])))
write_csv('selected_operating_points.csv',selected)

coarse_path = next((ROOT/'iv_fluxon_results').glob('*N65_*144123_spectrum_Nfl1.npz'))
with np.load(coarse_path) as z:
    common_current_match = bool(np.allclose(z['currents_norm'], spectrum_currents[::2]))
    coarse_difference = float(np.max(abs(z['power_W']-fine_power[::2])))

mu0 = 4*np.pi*1e-7
lambda_j = np.sqrt(PHI0/(2*np.pi*mu0*200e-9*1e6))
dx_um = np.sqrt(.1)*lambda_j*1e6
scales = dict(fp_GHz=FP/1e9, omega_p_rad_s=FP*2*np.pi,
              voltage_scale_uV=PHI0*FP*1e6, lambda_J_um=lambda_j*1e6,
              dx_um=dx_um, L_N65_um=64*dx_um, area_cell_um2=10*dx_um,
              target_voltage_uV=2*PHI0*5e9*1e6,
              R_shunt_ohm=1/(2*np.pi*FP*.2*(10*dx_um*.06e-12)),
              L_L_pH=.1*PHI0/(2*np.pi*(10*dx_um*1e-6))*1e12,
              L_J_pH=9*PHI0/(2*np.pi*(10*dx_um*1e-6))*1e12,
              C_c_pF=2*10*dx_um*.06,
              initial_q_cell1=float((4*np.arctan(np.exp((64-1)*np.sqrt(.1)/np.sqrt(1-.9**2)))-4*np.arctan(np.exp(-np.sqrt(.1)/np.sqrt(1-.9**2))))/(2*np.pi)))
metrics = dict(date='2026-11-23', source_files=len(inputs), source_bytes=sum(x['bytes'] for x in manifest),
               versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
               scales=scales, iv=iv,jtl=jtl,spectra=spectra,selected_points=selected,
               exact_duplicate_csv_groups=duplicates,
               N65_coarse_fine_common_currents_match=common_current_match,
               N65_coarse_fine_common_power_max_abs_difference=coarse_difference)
(HERE/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False,default=serial)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.dpi':190})
colors = ['#176b87','#c05640','#627742','#9063a8']
fig,ax=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
ax[0].plot(v,d['I_cell_uA'],'.-',ms=2,lw=.6,color=colors[0])
ax[0].set(xlabel='Voltage (µV)',ylabel='Bias per LJJ element (µA)',title='N=65: full saved I–V curve')
ax[0].axvline(scales['target_voltage_uV'],ls='--',lw=.8,color=colors[1])
mask=i<=.775
ax[1].plot(v[mask],d['I_cell_uA'][mask],'.-',ms=3,lw=.7,color=colors[0])
ax[1].set(xlim=(-1,90),xlabel='Voltage (µV)',ylabel='Bias per LJJ element (µA)',title='Fluxon branches and target voltage')
ax[1].axvline(scales['target_voltage_uV'],ls='--',lw=.9,color=colors[1],label='20.678 µV: fJ=10 GHz')
for row in selected:
    if row['i'] in [.62,.71,.75,.77]:
        ax[1].plot(row['V_uV'],row['I_cell_uA'],'o',color=colors[1],ms=4)
        ax[1].annotate(f"i={row['i']:g}",(row['V_uV'],row['I_cell_uA']),xytext=(5,-13),textcoords='offset points',fontsize=8)
ax[1].legend(fontsize=8)
for a in ax:a.grid(alpha=.2)
fig.savefig(FIG/'iv_overview.png');plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
f_edges=np.r_[freq-(freq[1]-freq[0])/2,freq[-1]+(freq[1]-freq[0])/2]
dy=np.median(np.diff(current_uA))
y_edges=np.r_[current_uA-dy/2,current_uA[-1]+dy/2]
im=ax[0].pcolormesh(f_edges,y_edges,power,shading='flat',norm=LogNorm(vmin=1e-13,vmax=power.max()),cmap='viridis')
ax[0].set(xlim=(0,25),ylim=(25,95),xlabel='Frequency (GHz)',ylabel='Bias per element (µA)',title='Edge-voltage spectrum (50 Ω equivalent)')
fig.colorbar(im,ax=ax[0],label='Spectrum value (W)',shrink=.9)
for bias,c in zip([.6,.62,.77],colors):
    j=np.argmin(abs(i-bias));ax[1].semilogy(freq,power[j],label=f'i={i[j]:g}',color=c)
ax[1].set(xlim=(0,25),ylim=(1e-16,5e-10),xlabel='Frequency (GHz)',ylabel='Spectrum value (W)',title='Fundamental and higher harmonics')
ax[1].axvline(5,color='#777777',ls='--',lw=.8);ax[1].legend();ax[1].grid(alpha=.2)
fig.savefig(FIG/'spectral_analysis.png');plt.close(fig)

fig,ax=plt.subplots(2,1,figsize=(10.5,6),sharex=True,layout='constrained')
for idx,M in enumerate([65,131,143]):
    key=next(k for k,(d0,p0,s0,l0) in jtl_arrays.items() if p0['M']==M and (M!=65 or p0['fluxon_cells']==[10]))
    d0,p0,s0,label=jtl_arrays[key];tns=d0['tau']/s0['OMEGA_PL_rad_s']*1e9
    ax[0].plot(tns,d0['V_out_mV']*1000,lw=1,color=colors[idx],label=f'M={M}')
    ax[1].plot(tns,d0['Delta_theta_N_over_2pi'],lw=1.4,color=colors[idx])
ax[0].set(ylabel='JTL output (µV)',title='Saved coupled runs: repeated output pulses');ax[0].legend(ncol=3)
ax[1].set(xlabel='Physical time (ns)',ylabel='Accumulated phase / 2π')
for a in ax:a.grid(alpha=.2)
fig.savefig(FIG/'jtl_comparison.png');plt.close(fig)
print(json.dumps({'inputs':len(inputs),'iv_curves':len(iv),'coupled_runs':len(jtl),'spectra':len(spectra),'scales':scales,'duplicate_groups':duplicates,'common_grid_power_difference':coarse_difference},indent=2,ensure_ascii=False))
