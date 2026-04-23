"""Ablation / prior-scheme comparison driver (Sec. 8 of the manuscript).

Under matched baseline assumptions (P_t = 3 W electrical, 42 users/macrocell,
10,000 Monte-Carlo trials, identical hex topology), this script runs four
resource-management variants and writes a single CSV summary so that reviewers
can verify the row-by-row quantitative gains of the proposed joint scheme
against its closest prior-art building blocks:

    1) JOINT         : ADT + OFDMA + strict-FFR                       (proposed)
    2) ADT-SDMA      : ADT + OFDMA, no strict-FFR                     (proxy for [1])
    3) FFR-ONLY      : strict-FFR + OFDMA, no ADT spatial selectivity (proxy for [2])
    4) OFDMA-ONLY    : plain OFDMA, no FFR, no ADT spatial selectivity (proxy for [4]-[6])

Implementation-level ablation knobs (no topology refactoring required):

  * Disabling strict-FFR is modelled by patching delta := 1 so that every user
    is a center-pool user, which merges all three edge pools into a single
    cluster-wide OFDMA pool and removes the adjacency-aware reuse protection.
  * Disabling ADT spatial selectivity is modelled by relaxing the angular
    filter gates to near-hemispherical (beam_half_deg := 89, rx_FOV_semi_deg
    := 89). The physical concentrator gain is recomputed consistently via
    psi_c = rx_FOV_semi_deg (lines 354, 247, 228 of simulator.py), so a
    wider-FOV receiver is paired with a proportionally lower concentrator
    gain, as expected from optical receiver design.

Output: ``results/comparison_prior_schemes.csv`` with columns
    mode, scheduler, trials, Pt_W, mean_sinr_db, median_sinr_db,
    avg_se_user_bpshz, per_user_rate_Mbps, jain_fairness,
    cov_p10db, cov_p20db, time_s.
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


def patch_delta(delta_val: float) -> None:
    """Override the module-level delta (and dependent pool sizes) so the
    strict-FFR partition is rebuilt exactly as in run_experiments.py."""
    simmod.delta = float(delta_val)
    simmod.N_data = simmod.K // 2 - 1
    data_idxs = np.arange(1, simmod.K // 2)
    simmod.Nc = math.floor((delta_val ** 2) * simmod.N_data)
    simmod.N_edge = (simmod.N_data - simmod.Nc) // 3
    simmod.center_subs = data_idxs[: simmod.Nc]
    simmod.edge_subs_list = [
        data_idxs[simmod.Nc : simmod.Nc + simmod.N_edge],
        data_idxs[simmod.Nc + simmod.N_edge : simmod.Nc + 2 * simmod.N_edge],
        data_idxs[simmod.Nc + 2 * simmod.N_edge : simmod.Nc + 3 * simmod.N_edge],
    ]


# Knob preset per comparison mode (see module docstring).
MODES = {
    'joint':     dict(disable_ffr=False, disable_adt=False),
    'adt_sdma':  dict(disable_ffr=True,  disable_adt=False),
    'ffr_only':  dict(disable_ffr=False, disable_adt=True),
    'ofdma_only':dict(disable_ffr=True,  disable_adt=True),
}

BASELINE_DELTA = 2.0 / 3.0
NO_FFR_DELTA = 1.0
ADT_BEAM_HALF_DEG = 20.0
ADT_RX_FOV_DEG = 40.0
WIDE_BEAM_HALF_DEG = 89.0
WIDE_RX_FOV_DEG = 89.0


def cov_prob(user_mean_sinr_linear: np.ndarray, thr_db: float) -> float:
    if user_mean_sinr_linear.size == 0:
        return float('nan')
    thr_lin = 10.0 ** (thr_db / 10.0)
    return float(np.mean(user_mean_sinr_linear >= thr_lin))


def run_case(mode: str, scheduler: str, trials: int, Pt: float,
             users_per_macro: int, rings: int = 2, B_e: int = 3,
             led_efficiency: float = 0.3, boost_factor_edge: float = 0.7,
             power_control_alpha: float = 0.0):
    flags = MODES[mode]

    if flags['disable_ffr']:
        patch_delta(NO_FFR_DELTA)
    else:
        patch_delta(BASELINE_DELTA)

    beam_half = WIDE_BEAM_HALF_DEG if flags['disable_adt'] else ADT_BEAM_HALF_DEG
    rx_fov = WIDE_RX_FOV_DEG if flags['disable_adt'] else ADT_RX_FOV_DEG

    res = simmod.simulate(
        trials=trials,
        users_per_macro=users_per_macro,
        Pt_electrical=Pt,
        scheduler=scheduler,
        rings=rings,
        B_e=B_e,
        beam_half_deg=beam_half,
        rx_FOV_semi_deg=rx_fov,
        led_efficiency=led_efficiency,
        boost_factor_edge=boost_factor_edge,
        power_control_alpha=power_control_alpha,
        debug_print=False,
    )

    mean_sinr_db = 10.0 * math.log10(res['mean_sinr_linear'] + 1e-30)
    median_sinr_db = 10.0 * math.log10(res['median_sinr_linear'] + 1e-30)

    user_se = res.get('user_se_samples', np.array([]))
    user_sinr = res.get('user_mean_sinr_samples', np.array([]))

    avg_se = float(np.mean(user_se)) if user_se.size else float('nan')

    return {
        'mode': mode,
        'scheduler': scheduler,
        'trials': int(trials),
        'Pt_W': float(Pt),
        'disable_ffr': bool(flags['disable_ffr']),
        'disable_adt': bool(flags['disable_adt']),
        'beam_half_deg': float(beam_half),
        'rx_FOV_semi_deg': float(rx_fov),
        'delta_used': float(simmod.delta),
        'mean_sinr_db': float(mean_sinr_db),
        'median_sinr_db': float(median_sinr_db),
        'avg_se_user_bpshz': avg_se,
        'per_user_rate_Mbps': (avg_se / simmod.N_data) * simmod.B / 1e6 if user_se.size else float('nan'),
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
    ap.add_argument('--modes', nargs='+',
                    default=['joint', 'adt_sdma', 'ffr_only', 'ofdma_only'])
    ap.add_argument('--schedulers', nargs='+', default=['full-load', 'rr1'])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for mode in args.modes:
        for sched in args.schedulers:
            print(f'Running comparison | mode={mode:<10} | scheduler={sched}')
            rows.append(run_case(mode, sched, args.trials, args.Pt,
                                 args.users_per_macro))

    out = os.path.join(args.outdir, 'comparison_prior_schemes.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('Done. Wrote', out)


if __name__ == '__main__':
    main()
