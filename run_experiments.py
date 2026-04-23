"""Full-protocol experiment driver implementing Algorithm 2 of the manuscript.

Runs the proposed ADT + OFDMA (disjoint subcarriers) + strict-FFR protocol
under the two interference-loading regimes (``'full-load'`` = every user
active in every trial, i.e. worst-case interference; ``'rr1'`` = round-robin
one user per region), plus the two sensitivity sweeps
over the center-radius fraction delta and user density.

Produces the three CSV summaries referenced by Tables 2-3 and Figures 4-8:
  ``results/full_protocol_summary.csv``
  ``results/sensitivity_delta.csv``
  ``results/sensitivity_users.csv``

Reproduction command used for all reported numbers (10,000 Monte Carlo trials):
  python run_full_protocol_sensitivity_v3.py --trials 10000 --Pt 3.0 --users_per_macro 42
"""

import argparse
import os
import math
import importlib.util
import sys
import numpy as np
import pandas as pd

SIM_FILE = 'simulator.py'

spec = importlib.util.spec_from_file_location('simmod', SIM_FILE)
simmod = importlib.util.module_from_spec(spec)
sys.modules['simmod'] = simmod
spec.loader.exec_module(simmod)


def patch_full_protocol(delta_val: float):
    """Override the simulator module's global delta and recompute the strict-FFR
    subcarrier pools, so the delta sensitivity sweep exercises exactly the same
    partition rule (Nc = floor(delta^2 * N_data), three equal edge pools) that
    the baseline run uses.
    """
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


def cov_prob(user_mean_sinr_linear: np.ndarray, thr_db: float) -> float:
    if user_mean_sinr_linear.size == 0:
        return float('nan')
    thr_lin = 10.0 ** (thr_db / 10.0)
    return float(np.mean(user_mean_sinr_linear >= thr_lin))


def run_case(trials, Pt, scheduler, delta_val, users_per_macro,
             beam_half_deg=20.0, rx_FOV_semi_deg=40.0,
             led_efficiency=0.3, boost_factor_edge=0.7, rings=2, B_e=3,
             power_control_alpha=0.0):

    patch_full_protocol(delta_val)

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
        debug_print=False,
    )

    mean_sinr_db = 10.0 * math.log10(res['mean_sinr_linear'] + 1e-30)
    median_sinr_db = 10.0 * math.log10(res['median_sinr_linear'] + 1e-30)

    user_se = res.get('user_se_samples', np.array([]))
    user_sinr = res.get('user_mean_sinr_samples', np.array([]))

    return {
        'protocol': 'ADT+OFDMA+strictFFR',
        'scheduler': scheduler,
        'delta': float(delta_val),
        'users_per_macro': int(users_per_macro),
        'trials': int(trials),
        'Pt_electrical_macro_W': float(Pt),
        'mean_sinr_db': float(mean_sinr_db),
        'median_sinr_db': float(median_sinr_db),
        'avg_se_user_bpshz': float(np.mean(user_se)) if user_se.size else float('nan'),
        'avg_se_per_data_subcarrier_bpshz': (float(np.mean(user_se)) / float(simmod.N_data)) if user_se.size else float('nan'),
        'jain_fairness': float(res.get('fairness_jain', float('nan'))),
        'conflicts': int(res.get('conflicts', -1)),
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
    ap.add_argument('--delta', type=float, default=2.0/3.0)
    ap.add_argument('--beam_half_deg', type=float, default=20.0)
    ap.add_argument('--rx_FOV_semi_deg', type=float, default=40.0)
    ap.add_argument('--power_control_alpha', type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Baseline summary (Algorithm 2, step 2): run the full protocol at the
    #    baseline (delta, users_per_macro) under each scheduler regime.
    rows = []
    for sched in ['full-load', 'rr1']:
        print(f'Running FULL protocol | scheduler={sched}')
        rows.append(run_case(args.trials, args.Pt, sched, args.delta, args.users_per_macro,
                             beam_half_deg=args.beam_half_deg, rx_FOV_semi_deg=args.rx_FOV_semi_deg,
                             power_control_alpha=args.power_control_alpha))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, 'full_protocol_summary.csv'), index=False)

    # 2) Delta sweep (Algorithm 2, step 3): vary the center-radius fraction
    #    over {0.5, 2/3, 0.8}; each value reshapes the strict-FFR partition.
    deltas = [0.5, 2.0/3.0, 0.8]
    rows = []
    for dval in deltas:
        for sched in ['full-load', 'rr1']:
            print(f'Running sensitivity delta={dval:.3f} | scheduler={sched}')
            rows.append(run_case(args.trials, args.Pt, sched, dval, args.users_per_macro,
                                 beam_half_deg=args.beam_half_deg, rx_FOV_semi_deg=args.rx_FOV_semi_deg,
                                 power_control_alpha=args.power_control_alpha))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, 'sensitivity_delta.csv'), index=False)

    # 3) User-density sweep (Algorithm 2, step 4): vary the offered load at the
    #    baseline delta so scheduler-driven trends can be separated from reuse.
    users_list = [14, 28, 42, 56]
    rows = []
    for upm in users_list:
        for sched in ['full-load', 'rr1']:
            print(f'Running sensitivity users_per_macro={upm} | scheduler={sched}')
            rows.append(run_case(args.trials, args.Pt, sched, 2.0/3.0, upm,
                                 beam_half_deg=args.beam_half_deg, rx_FOV_semi_deg=args.rx_FOV_semi_deg,
                                 power_control_alpha=args.power_control_alpha))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, 'sensitivity_users.csv'), index=False)

    print('Done. CSV written to', args.outdir)


if __name__ == '__main__':
    main()
