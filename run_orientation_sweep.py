"""Receiver-orientation sensitivity sweep (Sec. 7.6 of the manuscript).

Re-runs the full protocol at the baseline (delta, Umacro, Pt) for both
scheduling regimes ('full-load' = every user active, 'rr1' = round-robin-1) at
several values of the polar-tilt standard deviation ``rx_tilt_sigma_deg``.
The upward-facing baseline (sigma = 0 deg) is included so the table in
Sec. 7.6 can be read directly against Table 2.

Output: ``results/sensitivity_rx_tilt.csv`` with columns
    protocol, scheduler, rx_tilt_sigma_deg, trials,
    mean_sinr_db, median_sinr_db, avg_se_user_bpshz,
    jain_fairness, cov_p10db, cov_p20db, time_s.

Reproduction command (matches Table 4 in the manuscript):
    python run_orientation_sweep.py --trials 10000 --Pt 3.0 --users_per_macro 42
"""

import argparse
import math
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

SIM_FILE = 'simulator.py'

spec = importlib.util.spec_from_file_location('simmod', SIM_FILE)
simmod = importlib.util.module_from_spec(spec)
sys.modules['simmod'] = simmod
spec.loader.exec_module(simmod)


def cov_prob(user_mean_sinr_linear: np.ndarray, thr_db: float) -> float:
    if user_mean_sinr_linear.size == 0:
        return float('nan')
    thr_lin = 10.0 ** (thr_db / 10.0)
    return float(np.mean(user_mean_sinr_linear >= thr_lin))


def run_case(trials, Pt, scheduler, sigma_deg, users_per_macro,
             delta_val=2.0/3.0, beam_half_deg=20.0, rx_FOV_semi_deg=40.0,
             led_efficiency=0.3, boost_factor_edge=0.7, rings=2, B_e=3,
             power_control_alpha=0.0):
    """Single configuration: re-patches the module-level delta (identical
    bookkeeping to run_experiments.py) so the strict-FFR partition is
    constructed the same way as in the baseline."""
    simmod.delta = float(delta_val)
    simmod.N_data = simmod.K // 2 - 1
    data_idxs = np.arange(1, simmod.K // 2)
    simmod.Nc = math.floor((delta_val ** 2) * simmod.N_data)
    simmod.N_edge = (simmod.N_data - simmod.Nc) // 3
    simmod.center_subs = data_idxs[:simmod.Nc]
    simmod.edge_subs_list = [
        data_idxs[simmod.Nc : simmod.Nc + simmod.N_edge],
        data_idxs[simmod.Nc + simmod.N_edge : simmod.Nc + 2 * simmod.N_edge],
        data_idxs[simmod.Nc + 2 * simmod.N_edge : simmod.Nc + 3 * simmod.N_edge],
    ]

    res = simmod.simulate(
        trials=trials,
        users_per_macro=users_per_macro,
        Pt_electrical=Pt,
        scheduler=scheduler,
        rings=rings,
        B_e=B_e,
        beam_half_deg=beam_half_deg,
        rx_FOV_semi_deg=rx_FOV_semi_deg,
        led_efficiency=led_efficiency,
        boost_factor_edge=boost_factor_edge,
        power_control_alpha=power_control_alpha,
        rx_tilt_sigma_deg=sigma_deg,
        debug_print=False,
    )

    mean_sinr_db = 10.0 * math.log10(res['mean_sinr_linear'] + 1e-30)
    median_sinr_db = 10.0 * math.log10(res['median_sinr_linear'] + 1e-30)
    user_se = res.get('user_se_samples', np.array([]))
    user_sinr = res.get('user_mean_sinr_samples', np.array([]))

    return {
        'protocol': 'ADT+OFDMA+strictFFR',
        'scheduler': scheduler,
        'rx_tilt_sigma_deg': float(sigma_deg),
        'trials': int(trials),
        'mean_sinr_db': float(mean_sinr_db),
        'median_sinr_db': float(median_sinr_db),
        'avg_se_user_bpshz': float(np.mean(user_se)) if user_se.size else float('nan'),
        'jain_fairness': float(res.get('fairness_jain', float('nan'))),
        'cov_p10db': cov_prob(user_sinr, 10.0),
        'cov_p20db': cov_prob(user_sinr, 20.0),
        'time_s': float(res.get('time_s', float('nan'))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=10000)
    ap.add_argument('--Pt', type=float, default=3.0)
    ap.add_argument('--users_per_macro', type=int, default=42)
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--sigmas', type=float, nargs='+',
                    default=[0.0, 5.0, 15.0, 29.0])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for sigma in args.sigmas:
        for sched in ['full-load', 'rr1']:
            print(f'Running RX-tilt sweep | sigma={sigma:>5.1f} deg | scheduler={sched}')
            rows.append(run_case(args.trials, args.Pt, sched, sigma,
                                 args.users_per_macro))

    out = os.path.join(args.outdir, 'sensitivity_rx_tilt.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('Done. Wrote', out)


if __name__ == '__main__':
    main()
