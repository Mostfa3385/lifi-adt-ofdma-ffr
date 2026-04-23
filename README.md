# Joint ADT-enabled SDMA, OFDMA, and Strict FFR for Dense LiFi Attocell Networks

Reference implementation and Monte Carlo simulation code for the manuscript submitted to
*Scientific Reports*. The code reproduces every numerical result reported in Table 2 and
Section 7.5 (δ sensitivity and user-density sensitivity) of the paper.

## Repository contents

```
.
├── simulator.py              # Algorithm 1: per-trial SINR / SE / fairness simulator
├── run_experiments.py        # Algorithm 2: driver that produces the three CSV outputs
├── results/
│   ├── full_protocol_summary.csv   # Table 2 (baseline δ = 2/3, 42 users/macrocell)
│   ├── sensitivity_delta.csv       # §7.5 δ sweep over {0.5, 2/3, 0.8}
│   └── sensitivity_users.csv       # §7.5 user-density sweep over {14, 28, 42, 56}
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

`simulator.py` implements the topology, ADT beam model, strict-FFR partition, scheduler
regimes (`'none'` = full activity, `'rr1'` = round-robin one user per region), DCO-OFDM
clipping bookkeeping, and the per-user SINR / spectral-efficiency / Jain-fairness
estimators. `run_experiments.py` dynamically imports `simulator.py`, patches the
strict-FFR partition for each δ value, and writes the three CSV summaries used by the
paper.

## System requirements

- Python 3.10 or newer (tested on 3.11)
- The Python packages listed in `requirements.txt`: `numpy`, `scipy`, `pandas`,
  `networkx`, `matplotlib`
- No GPU required. Typical run time for a full reproduction (all three CSVs,
  10,000 Monte Carlo trials per row) is 30–60 minutes on a modern laptop CPU.

## Installation

```bash
git clone https://github.com/Mostfa3385/lifi-adt-ofdma-ffr.git
cd lifi-adt-ofdma-ffr
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper

All three CSVs that back Table 2 and §7.5 are produced by a single command:

```bash
python run_experiments.py --trials 10000 --Pt 3.0 --users_per_macro 42
```

This writes the three files into `results/`, overwriting any prior run. The expected
baseline outputs (δ = 2/3, 42 users/macrocell, 10,000 trials) are:

| scheduler | mean SINR [dB] | median SINR [dB] | per-user SE [bps/Hz] | Jain's fairness | cov @ 10 dB | cov @ 20 dB |
|-----------|---------------:|-----------------:|---------------------:|----------------:|------------:|------------:|
| `none`    | ≈ 28.11        | ≈ 27.10          | ≈ 375.8              | ≈ 0.793         | 1.00        | 1.00        |
| `rr1`     | ≈ 28.28        | ≈ 27.15          | ≈ 1028.9             | ≈ 0.990         | 1.00        | 1.00        |

These exact values are archived in `results/full_protocol_summary.csv` and correspond
to Table 2 of the manuscript. A re-run on a different machine / BLAS configuration is
expected to agree within Monte Carlo variability (typically ±0.1 dB for SINR at
10,000 trials).

### Command-line options (`run_experiments.py`)

| flag | default | meaning |
|------|---------|---------|
| `--trials` | `10000` | Monte Carlo trials per configuration |
| `--Pt` | `3.0` | macrocell electrical transmit power [W] |
| `--users_per_macro` | `42` | offered user density at the baseline |
| `--delta` | `2/3` | center-radius fraction for the baseline summary |
| `--outdir` | `results` | output directory for the CSV files |
| `--beam_half_deg` | `20.0` | ADT beam half-angle [deg] |
| `--rx_FOV_semi_deg` | `40.0` | receiver field-of-view semi-angle [deg] |
| `--power_control_alpha` | `0.0` | reserved knob; `0.0` matches the paper |

## File-to-paper cross-reference

| File / function | Paper section / equation |
|-----------------|--------------------------|
| `simulator.py :: channel_gain_los` | Eq. (1a)–(1c), Sec. 3.1 |
| `simulator.py :: build_topology` | Sec. 3.1 (co-located ADT) |
| `simulator.py :: ofdm_clipping_stats` | Sec. 4.3 (DCO-OFDM clipping) |
| `simulator.py :: noise_variance_total` | Sec. 3.5 |
| `simulator.py :: simulate` (main body) | Algorithm 1 |
| `run_experiments.py :: patch_full_protocol` | strict-FFR partition, Sec. 3.3 |
| `run_experiments.py :: main` | Algorithm 2 |

## Data and code availability

The code and results in this repository are mirrored as a versioned archive on Zenodo
with a persistent DOI. The Zenodo record is what the manuscript cites under *Data
Availability* and *Code Availability*.

