"""Targeted replay and controls, with all derived files isolated here.

Never calls the original main block, save functions or plotting functions.
"""
from pathlib import Path
import importlib.util
import json
import time
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('ljj_reference',ROOT/'ljj_capacitive_jtl_final_calculated_params.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
folder=next((ROOT/'ljj_capacitive_jtl_results').glob('M65__*cells10*'))
reference=np.genfromtxt(folder/'jtl_output.csv',delimiter=',',names=True)
params=json.loads((folder/'parameters.json').read_text())['parameters']
results=[]
arrays={}
for case in ['replay','tighter','no_fluxon','no_coupling']:
    p=dict(params)
    if case=='tighter':p.update(max_step=.025,rtol=1e-8,atol=1e-10)
    if case=='no_fluxon':p.update(n_fluxons=0,fluxon_cells=[])
    if case=='no_coupling':p['kappa']=0.
    _,y0=mod.make_initial_state(p)
    t0=time.monotonic()
    sol=solve_ivp(lambda t,y:mod.system_ljj_cap_jtl(t,y,p),(0.,p['tau_end']),y0,
                  t_eval=reference['tau'],method=p['method'],rtol=p['rtol'],atol=p['atol'],max_step=p['max_step'])
    assert sol.success,sol.message
    tau,phi,theta,psi,omega=mod.unpack_solution(sol,p)
    v=omega[:,-1]
    q=(theta[:,-1]-theta[0,-1])/(2*np.pi)
    peaks,_=find_peaks(v,height=.5,prominence=.5,distance=60)
    periods=np.diff(tau[peaks])
    row=dict(case=case,final_Q=float(q[-1]),max_abs_Vnorm=float(abs(v).max()),
              detected_peaks=len(peaks),elapsed_s=time.monotonic()-t0,
              repetition_GHz=float(mod.OMEGA_PL/periods.mean()/1e9) if len(periods) else None,
              max_abs_Vnorm_difference_from_saved=float(np.max(abs(v-reference['V_out_normalized']))),
              max_abs_Q_difference_from_saved=float(np.max(abs(q-reference['Delta_theta_N_over_2pi']))))
    results.append(row)
    arrays[case+'_Vnorm']=v
    arrays[case+'_Q']=q
    print(json.dumps(row),flush=True)
    (HERE/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
np.savez_compressed(HERE/'validation_signals.npz',tau=tau,**arrays)
