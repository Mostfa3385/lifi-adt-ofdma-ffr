"""Monte Carlo simulator for the proposed ADT + OFDMA + strict-FFR protocol.

Implements Algorithm 1 of the associated manuscript. Each macrocell is served
by a single co-located 7-element angle-diversity transmitter (ADT). The seven
microcells are logical floor regions used for user placement, center/edge
classification (Sec. 3.2), and adjacency-aware edge-pool assignment
(Sec. 4.1); the physical transmitter position for all seven beams is the
macrocell center.

Public entry point: ``simulate(...)`` -- returns per-user SINR, spectral
efficiency, and fairness samples used to build Table 2 and Figures 4-8.
"""

import math
import time
import numpy as np
from scipy.spatial import KDTree
from numpy.fft import ifft
import networkx as nx

# ===================== System Parameters =====================
# Notation follows Section 3 of the manuscript; values match Table 1.
R_micro = 3.3                 # microcell region radius R [m]
hf = 2.15                     # transmitter-receiver vertical separation [m]
delta = 2.0/3.0               # center-radius fraction (0 < delta < 1)
K = 512                       # OFDM FFT size
N_data = K//2 - 1             # usable data subcarriers (Hermitian symmetry)
Nc = math.floor((delta**2) * N_data)   # strict-FFR center pool size
N_edge = (N_data - Nc) // 3            # per-pool edge size (3 protected edge pools)

phi_half = math.radians(60)                          # LED half-power semi-angle
m = -math.log(2)/math.log(math.cos(phi_half))        # Lambertian order (Eq. 1a)

responsivity = 0.4                 # photodiode responsivity eta [A/W]
optical_filter = 0.95              # optical filter transmission Ts
eta_base = responsivity * optical_filter

n_refr = 1.5                       # concentrator refractive index
A_pd_phys = 5e-5                   # photodiode active area Ar [m^2]

B = 5e6                            # system bandwidth [Hz]
N0 = 1e-21                         # thermal noise PSD [W/Hz]

# Data-subcarrier index sets used by the strict-FFR partition (Sec. 3.3).
data_idxs = np.arange(1, K//2)
center_subs = data_idxs[:Nc]
edge_subs_list = [
    data_idxs[Nc : Nc + N_edge],
    data_idxs[Nc + N_edge : Nc + 2*N_edge],
    data_idxs[Nc + 2*N_edge : Nc + 3*N_edge]
]

def noise_variance_total():
    """Total in-band thermal noise power N0 * B (Sec. 3.5)."""
    return N0 * B

# ===================== Topology / ADT =====================
class Microcell:
    """Logical microcell region served by one ADT beam.

    region_center: where users are dropped and classified center/edge.
    tx_center: macrocell center (co-located ADT array) for ALL beams.
    """
    def __init__(self, region_center, tx_center, parent_macro_idx, idx):
        self.region_center = region_center
        self.tx_center = tx_center
        self.parent_macro_idx = parent_macro_idx
        self.idx = idx               # 0=center region, 1..6 edge regions
        self.adjacent = []
        self.edge_band = None
        self.users = []
        self.beam_idx = None
        self.beam_az = None


def hex_grid_centers(rings, spacing):
    centers = []
    for q in range(-rings, rings+1):
        r1 = max(-rings, -q - rings)
        r2 = min(rings, -q + rings)
        for r in range(r1, r2+1):
            x = spacing * (3/2.0 * q)
            y = spacing * (math.sqrt(3)/2.0 * q + math.sqrt(3) * r)
            centers.append((x, y))
    return centers


def build_topology(rings=2, cutoff_factor=1.01, NLED=7):
    """Construct macrocell centers and seven logical regions per macrocell.

    All seven beams of a macrocell share the same physical ``tx_center``
    (co-located ADT at the macrocell centroid). ``region_center`` is used
    for user placement, center/edge classification, and adjacency only
    (Sec. 3.1).
    """
    R_macro = 2 * R_micro * math.cos(math.pi/6)
    spacing = R_macro * math.sqrt(3)
    macro_centers = hex_grid_centers(rings, spacing)

    all_mcs = []

    # Seven region centers per macrocell: one at the macrocell center plus six
    # placed at the vertices of a regular hexagon one region-to-region apart.
    dist = 2 * R_micro * math.cos(math.pi/6)
    for mi, (cx, cy) in enumerate(macro_centers):
        tx_center = (cx, cy)
        all_mcs.append(Microcell(region_center=(cx, cy), tx_center=tx_center, parent_macro_idx=mi, idx=0))
        for rid, th in enumerate(np.linspace(0, 2*math.pi, 6, endpoint=False), start=1):
            rx = cx + dist * math.cos(th)
            ry = cy + dist * math.sin(th)
            all_mcs.append(Microcell(region_center=(rx, ry), tx_center=tx_center, parent_macro_idx=mi, idx=rid))

    # Adjacency is derived from region-center distances (Sec. 4.1);
    # it drives the strict-FFR edge-pool graph coloring in color_edge_bands().
    pts = np.array([mc.region_center for mc in all_mcs])
    tree = KDTree(pts)
    cutoff = 2.0 * R_micro * cutoff_factor
    for i, mc in enumerate(all_mcs):
        nbrs = tree.query_ball_point(pts[i], r=cutoff)
        mc.adjacent = [all_mcs[j] for j in nbrs if j != i]

    # Beam azimuths per macrocell (NLED co-located beams, uniform 360/NLED spacing).
    beams_az_per_macro = []
    for _mi in range(len(macro_centers)):
        azs = [(360.0 / NLED) * b for b in range(NLED)]
        beams_az_per_macro.append(azs)

    # Beam-to-region assignment: region 0 uses beam 0 by definition; regions 1..6
    # use the beam axis closest to their azimuth relative to the macrocell centroid.
    for mc in all_mcs:
        mi = mc.parent_macro_idx
        macro_cx, macro_cy = macro_centers[mi]
        azs = beams_az_per_macro[mi]

        if mc.idx == 0:
            mc.beam_idx = 0
            mc.beam_az = azs[0]
        else:
            vx = mc.region_center[0] - macro_cx
            vy = mc.region_center[1] - macro_cy
            az_reg = math.degrees(math.atan2(vy, vx))
            if az_reg < 0:
                az_reg += 360.0
            diffs = [abs((az_reg - a + 180) % 360 - 180) for a in azs]
            best = int(np.argmin(diffs))
            mc.beam_idx = best
            mc.beam_az = azs[best]

    return all_mcs, macro_centers, beams_az_per_macro


def color_edge_bands(all_mcs, B_e=3):
    edge_nodes = [i for i, mc in enumerate(all_mcs) if mc.idx > 0]
    G = nx.Graph()
    G.add_nodes_from(edge_nodes)
    for i in edge_nodes:
        for nbr in all_mcs[i].adjacent:
            j = all_mcs.index(nbr)
            if j in edge_nodes:
                G.add_edge(i, j)

    greedy = nx.coloring.greedy_color(G, strategy='largest_first')
    mapping = {i: color % B_e for i, color in greedy.items()}
    for i, mc in enumerate(all_mcs):
        if mc.idx > 0:
            mc.edge_band = mapping.get(i, 0)

    conflicts = 0
    for i, mc in enumerate(all_mcs):
        if mc.idx > 0:
            for nbr in mc.adjacent:
                j = all_mcs.index(nbr)
                if j > i and nbr.idx > 0 and mc.edge_band == nbr.edge_band:
                    conflicts += 1
    return mapping, conflicts

# ===================== OFDM / Clipping =====================
clip_cache = {}

def ofdm_clipping_stats(num_sc):
    """DCO-OFDM bias-plus-clip statistics (Sec. 4.3).

    Returns ``(clip_factor, clip_noise_td)``: the multiplicative reduction in
    mean optical amplitude caused by DC-bias + zero-clipping, and the
    time-domain clipping-noise variance. Cached per subcarrier count.
    """
    if num_sc <= 0:
        return 1.0, 0.0
    if num_sc in clip_cache:
        return clip_cache[num_sc]

    ph = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=num_sc)
    X = np.zeros(K, dtype=complex)
    pos = np.arange(1, 1+num_sc)
    X[pos] = ph
    X[-pos] = np.conj(ph[::-1])
    x = np.real(ifft(X))
    bias = -np.min(x)
    xb = x + bias
    xc = xb.copy()
    xc[xc < 0] = 0.0

    mean_unclipped = np.mean(np.maximum(x, 0))
    mean_clipped = np.mean(xc)
    clip_factor = mean_clipped / (mean_unclipped + 1e-12)

    clip_noise_td = np.mean((xc - (x + bias))**2)
    clip_cache[num_sc] = (clip_factor, clip_noise_td)
    return clip_cache[num_sc]

# ===================== Power Allocation =====================
def uniform_alloc(P_total, num_sc):
    if num_sc <= 0:
        return np.zeros(1)
    return np.full(num_sc, P_total / float(num_sc))

# ===================== Channel gain =====================
def angle_diff_deg(a, b):
    return abs((a - b + 180) % 360 - 180)


def concentrator_gain(psi, psi_c):
    if 0 <= psi <= psi_c:
        return (n_refr**2) / (math.sin(psi_c)**2 + 1e-30)
    return 0.0


def channel_gain_los(d_horiz, hf, A_pd, m, psi_c, Ts=1.0):
    """Lambertian LOS DC channel gain H(r) per Eq. (1); 0 if psi > psi_c."""
    d = math.sqrt(d_horiz*d_horiz + hf*hf)
    if d <= 0:
        return 0.0
    cos_phi = hf / d
    psi = math.acos(max(-1.0, min(1.0, cos_phi)))
    if psi > psi_c:
        return 0.0
    g = concentrator_gain(psi, psi_c)
    H = ((m+1) * A_pd / (2 * math.pi * d*d)) * (cos_phi**m) * Ts * g * cos_phi
    return max(0.0, H)

# ===================== OFDMA subcarrier partitioning =====================
def partition_subcarriers(subs, num_users):
    if num_users <= 0:
        return []
    parts = [[] for _ in range(num_users)]
    for i, s in enumerate(subs):
        parts[i % num_users].append(int(s))
    return [np.array(p, dtype=int) for p in parts]

# ===================== Main Simulation =====================
def simulate(
    trials=10000,
    users_per_macro=42,
    Pt_electrical=3.0,
    B_e=3,
    power_control_alpha=0.0,
    scheduler='none',
    rings=2,
    NLED=7,
    beam_half_deg=20.0,
    rx_FOV_semi_deg=40.0,
    led_efficiency=0.3,
    boost_factor_edge=1.0,
    debug_print=True
):
    P_optical_macro = led_efficiency * Pt_electrical

    all_mcs, macro_centers, _ = build_topology(rings=rings, NLED=NLED)
    _, conflicts = color_edge_bands(all_mcs, B_e)

    noise_total = noise_variance_total()
    noise_sub = noise_total / float(N_data)          # per-subcarrier noise Ns (Eq. 12)

    # Optical-power split across the seven beams. With boost_factor_edge == 1.0
    # this reproduces the uniform allocation described in Sec. 4.3; values < 1.0
    # (e.g. 0.7) give the center beam the residual budget and leave 6 * P_edge_req
    # W across the six edge beams. The else-branch is a numerical safety net when
    # an extreme boost_factor_edge would otherwise drive the center budget below 0.
    P_cell_base = P_optical_macro / 7.0
    P_edge_req = boost_factor_edge * P_cell_base
    P_center_calc = P_optical_macro - 6.0 * P_edge_req
    if P_center_calc > 0:
        P_cell_center = P_center_calc
        P_cell_edge = P_edge_req
    else:
        eps = 1e-6 * max(1.0, Pt_electrical)
        P_cell_center = eps
        P_cell_edge = (P_optical_macro - P_cell_center) / 6.0

    if debug_print:
        print(f"DEBUG: Pt_electrical(per-macro)={Pt_electrical} W -> P_optical_macro={P_optical_macro:.6f} W")
        print(f"DEBUG: per-beam optical power: center={P_cell_center:.6e} W, edge={P_cell_edge:.6e} W")

    users_per_region = users_per_macro // 7

    # Reference channel gain used only by the optional distance-based power-control
    # mode (power_control_alpha > 0). With alpha == 0 (the value used throughout
    # the manuscript) this block has no effect on the reported results.
    psi_c = math.radians(rx_FOV_semi_deg)
    H_vals = []
    for mc in all_mcs:
        tx_cx, tx_cy = mc.tx_center
        rx, ry = mc.region_center
        d_h = math.hypot(rx - tx_cx, ry - tx_cy)
        H_vals.append(channel_gain_los(d_h, hf, A_pd_phys, m, psi_c, Ts=1.0))
    Href = np.median(H_vals) if len(H_vals) else 1.0
    if Href <= 0:
        Href = 1e-30

    # Interferer-set construction (Sec. 4.2). Each region's neighbor set includes
    # its geometric neighbors and all six sibling regions of the same macrocell;
    # center/edge filtering of this set is applied later when SINR is evaluated.
    mc_neighbors = []
    for i, mc in enumerate(all_mcs):
        nbr_idxs = set([all_mcs.index(n) for n in mc.adjacent])
        same_macro = [j for j, mm in enumerate(all_mcs) if mm.parent_macro_idx == mc.parent_macro_idx]
        nbr_idxs.update(same_macro)
        mc_neighbors.append(sorted(nbr_idxs))

    sinr_sc = []
    user_mean_sinr = []
    user_total_se = []

    start = time.time()

    for _tr in range(trials):
        # Step 8 of Algorithm 1: uniformly drop users within each region; the
        # sqrt-sampled radius preserves uniform area density inside a disk of
        # radius R_micro.
        for mc in all_mcs:
            mc.users = []
            rcx, rcy = mc.region_center
            for _ in range(users_per_region):
                r = R_micro * math.sqrt(np.random.rand())
                th = 2 * math.pi * np.random.rand()
                x = rcx + r * math.cos(th)
                y = rcy + r * math.sin(th)
                mc.users.append((x, y))

        # Interference-loading regime: 'none' activates every user (full-activity),
        # 'rr1' activates exactly one user per region per trial (round-robin-1).
        active_map = {}
        for i, mc in enumerate(all_mcs):
            if scheduler == 'rr1':
                active_map[i] = [np.random.randint(len(mc.users))] if len(mc.users) > 0 else []
            else:
                active_map[i] = list(range(len(mc.users)))

        # Step 10 of Algorithm 1: OFDMA allocation. Center users draw disjoint
        # subsets of the common center pool; edge users draw from the protected
        # edge pool selected by the region's graph-coloring band.
        user_subs = {}
        for i, mc in enumerate(all_mcs):
            act = active_map[i]
            if not act:
                continue
            centers = []
            edges = []
            rcx, rcy = mc.region_center
            for ui in act:
                x, y = mc.users[ui]
                is_center = (math.hypot(x - rcx, y - rcy) <= delta * R_micro)
                (centers if is_center else edges).append(ui)

            if centers:
                parts = partition_subcarriers(center_subs, len(centers))
                for ui, part in zip(centers, parts):
                    user_subs[(i, ui)] = part

            if edges:
                band = mc.edge_band if mc.edge_band is not None else 0
                subs_edge = edge_subs_list[band % 3] if B_e == 3 else data_idxs[Nc:Nc+3*N_edge]
                parts = partition_subcarriers(subs_edge, len(edges))
                for ui, part in zip(edges, parts):
                    user_subs[(i, ui)] = part

        # Per-(region, subcarrier) squared received-signal amplitude cache,
        # reused below when forming both desired-signal and interference terms.
        tx_amp2 = {}

        for i, mc in enumerate(all_mcs):
            tx_cx, tx_cy = mc.tx_center

            center_users = [ui for ui in active_map[i] if (i, ui) in user_subs and user_subs[(i, ui)].size and user_subs[(i, ui)][0] in center_subs]
            edge_users = [ui for ui in active_map[i] if (i, ui) in user_subs and user_subs[(i, ui)].size and user_subs[(i, ui)][0] not in center_subs]

            def add_group(users, P_cell_group):
                if not users:
                    return
                used = np.unique(np.concatenate([user_subs[(i, ui)] for ui in users]))
                num_sc = max(1, len(used))
                clip_factor, _ = ofdm_clipping_stats(num_sc)
                P_eff = P_cell_group * clip_factor
                P_alloc = uniform_alloc(P_eff, num_sc)
                P_map = {int(s): float(P_alloc[k]) for k, s in enumerate(used)}

                for ui in users:
                    x, y = mc.users[ui]
                    az_user = math.degrees(math.atan2(y - tx_cy, x - tx_cx))
                    if az_user < 0:
                        az_user += 360.0
                    theta_deg = angle_diff_deg(az_user, mc.beam_az or 0.0)
                    if theta_deg > beam_half_deg or theta_deg > rx_FOV_semi_deg:
                        continue
                    d_h = math.hypot(x - tx_cx, y - tx_cy)
                    H = channel_gain_los(d_h, hf, A_pd_phys, m, psi_c, Ts=1.0)
                    if H <= 0:
                        continue

                    user_scale = 1.0
                    if power_control_alpha > 0.0:
                        user_scale = (H / (Href + 1e-30))**(-power_control_alpha)
                        user_scale = max(0.1, min(user_scale, 10.0))

                    for s in user_subs[(i, ui)]:
                        Ps = P_map.get(int(s), 0.0) * user_scale
                        amp = eta_base * H * Ps
                        tx_amp2[(i, int(s))] = (amp * amp)

            add_group(center_users, P_cell_center)
            add_group(edge_users, P_cell_edge)

        # Evaluate SINR/SE for users in the central macrocell only. Outer
        # macrocells are included as interference sources but are excluded from
        # the performance averages to avoid finite-layout edge effects; this
        # matches the "central macrocell users" scope stated in Table 2.
        for i, mc in enumerate(all_mcs):
            if mc.parent_macro_idx != 0:
                continue

            tx_cx, tx_cy = mc.tx_center
            my_idx = i

            for ui in active_map[i]:
                subs_u = user_subs.get((i, ui), None)
                if subs_u is None or subs_u.size == 0:
                    continue

                x, y = mc.users[ui]
                is_center_user = (subs_u.size and int(subs_u[0]) in set(center_subs))

                az_user = math.degrees(math.atan2(y - tx_cy, x - tx_cx))
                if az_user < 0:
                    az_user += 360.0
                theta_deg = angle_diff_deg(az_user, mc.beam_az or 0.0)
                if theta_deg > beam_half_deg or theta_deg > rx_FOV_semi_deg:
                    continue

                d_h = math.hypot(x - tx_cx, y - tx_cy)
                H = channel_gain_los(d_h, hf, A_pd_phys, m, psi_c, Ts=1.0)
                if H <= 0:
                    continue

                if is_center_user:
                    interferers = set(mc_neighbors[my_idx])
                    interferers.discard(my_idx)
                else:
                    band = mc.edge_band if mc.edge_band is not None else 0
                    candidate = mc_neighbors[my_idx]
                    interferers = set([j for j in candidate if all_mcs[j].idx > 0 and all_mcs[j].edge_band == band])
                    interferers.discard(my_idx)

                used_sc = len(subs_u)
                _cf, clip_noise_td = ofdm_clipping_stats(max(1, used_sc))
                clipping_noise_eq = clip_noise_td * (eta_base * H)**2 / (max(1, used_sc))

                se_u = 0.0
                sinrs_u = []
                for s in subs_u:
                    S = tx_amp2.get((my_idx, int(s)), 0.0)
                    I = 0.0
                    for j in interferers:
                        I += tx_amp2.get((j, int(s)), 0.0)
                    sinr = S / (I + noise_sub + clipping_noise_eq + 1e-30)
                    sinr_sc.append(sinr)
                    sinrs_u.append(sinr)
                    se_u += math.log2(1 + sinr)

                if sinrs_u:
                    user_mean_sinr.append(float(np.mean(sinrs_u)))
                    user_total_se.append(float(se_u))

    elapsed = time.time() - start

    sinr_sc = np.array(sinr_sc) if len(sinr_sc) else np.array([0.0])
    user_mean_sinr = np.array(user_mean_sinr) if len(user_mean_sinr) else np.array([])
    user_total_se = np.array(user_total_se) if len(user_total_se) else np.array([])

    fairness = 0.0
    if user_total_se.size:
        Ssum = np.sum(user_total_se)
        S2 = np.sum(user_total_se**2)
        n = float(user_total_se.size)
        fairness = (Ssum*Ssum)/(n*S2) if S2 > 0 else 0.0

    return {
        'time_s': float(elapsed),
        'conflicts': int(conflicts),
        'mean_sinr_linear': float(np.mean(sinr_sc)) if sinr_sc.size else 0.0,
        'median_sinr_linear': float(np.median(sinr_sc)) if sinr_sc.size else 0.0,
        'user_mean_sinr_samples': user_mean_sinr,
        'user_se_samples': user_total_se,
        'fairness_jain': float(fairness),
        'P_cell_center': float(P_cell_center),
        'P_cell_edge': float(P_cell_edge),
    }


if __name__ == '__main__':
    # Standalone run of the simulator. Trial count matches the value reported
    # in the paper (10,000 Monte Carlo trials). For the full protocol summary
    # and sensitivity sweeps used to produce Tables 2-3 and Figures 4-8, use
    # run_full_protocol_sensitivity_v3.py instead.
    res = simulate(trials=10000, users_per_macro=42, Pt_electrical=3.0, scheduler='none',
                   rings=2, NLED=7, beam_half_deg=20.0, rx_FOV_semi_deg=40.0,
                   boost_factor_edge=0.7, led_efficiency=0.3, debug_print=True)
    print('\nBaseline run results:')
    print('Mean SINR (dB)=', 10*math.log10(res['mean_sinr_linear'] + 1e-30))
    print('Median SINR (dB)=', 10*math.log10(res['median_sinr_linear'] + 1e-30))
    print('Jain fairness=', res['fairness_jain'])
