"""Synthetic multi-barcode reads + library for annotate_reads / call_cells tests.

Layout mirrors TDP-GWS: 15 cycles, MAP = cycles 1-12 (iBar2), RECOMB = cycles
13-15 (iBar1). Library min pairwise distance is 12, so Hamming-1 correction is
unambiguous. Each cell exercises one branch of the per-read state.
"""

import pandas as pd

N_CYCLES = 15

LIBRARY = pd.DataFrame(
    {
        "prefix_map": ["AAAAAAAAAAAA", "CCCCCCCCCCCC", "GGGGGGGGGGGG", "TTTTTTTTTTTT"],
        "gene_symbol": ["GENE_A", "GENE_C", "GENE_G", "GENE_T"],
        "prefix_recomb": ["ACG", "CGT", "GTA", "TAC"],
    }
)
A, C, G, T = LIBRARY["prefix_map"]

MULTI_PARAMS = dict(
    q_min=0,
    map_start=1,
    map_end=12,
    prefix_map="prefix_map",
    recomb_start=13,
    recomb_end=15,
    prefix_recomb="prefix_recomb",
    recomb_filter_col="Q_recomb",
    recomb_q_thresh=0.1,
    error_correct=True,
    max_distance=1,
    sort_calls="peak",
    n_barcodes=2,
)

SIMPLE_PARAMS = dict(
    q_min=0,
    prefix_col="prefix",
    error_correct=False,
    sort_calls="count",
    n_barcodes=2,
)


def _mut(seq, pos0, base):
    """Substitute `base` at 0-indexed position `pos0`."""
    return seq[:pos0] + base + seq[pos0 + 1 :]


# Named reads so tests can address them by read id.
READ_CLEAN = 1  # cell 1, exact A + ACG
READ_MAP_MISCALL = 4  # cell 2, A with cycle-5 miscall, recomb intact -> corrected
READ_RECOMB_OFF = 6  # cell 3, exact C + CGA (one base off CGT) -> no_recomb False
READ_LOW_Q_RECOMB = 8  # cell 4, exact G + GTA but Q_12 = 0.05 -> no_recomb <NA>
READ_UNMAPPED = 9  # cell 5, two map miscalls -> mapped False
READ_LOW_Q_MIN = 12  # cell 6, Q = 0.2 everywhere -> dropped when q_min > 0.2


def make_reads():
    """One row per read; columns mirror call_reads output plus `plate`."""
    rows = []

    def add(cell, read, barcode, peak=100.0, q=0.9, q_override=None):
        qs = [q] * N_CYCLES
        for k, v in (q_override or {}).items():
            qs[k] = v
        rows.append(
            {
                "plate": "1",
                "well": "A1",
                "tile": 1,
                "cell": cell,
                "read": read,
                "i": 10 * read,
                "j": 20 * read,
                "barcode": barcode,
                "peak": peak,
                **{f"Q_{k}": qs[k] for k in range(N_CYCLES)},
                "Q_min": min(qs),
            }
        )

    # cell 1: three clean reads of A -> no_recomb True, uncorrected
    add(1, READ_CLEAN, A + "ACG")
    add(1, 2, A + "ACG")
    add(1, 3, A + "ACG", peak=80)
    # cell 2: cycle-5 miscall in MAP (0-idx 4) -> corrected, correction_cycle 5
    add(2, READ_MAP_MISCALL, _mut(A, 4, "G") + "ACG", peak=120)
    add(2, 5, A + "ACG")
    # cell 3: RECOMB 3-mer one base off expected -> no_recomb False (never corrected)
    add(3, READ_RECOMB_OFF, C + "CGA")
    add(3, 7, C + "CGA")
    # cell 4: Q_recomb below threshold -> no_recomb <NA>, indeterminant True
    add(4, READ_LOW_Q_RECOMB, G + "GTA", q_override={12: 0.05})
    # cell 5: unmapped read (peak 100) + one mapped T read (peak 50)
    #   -> rank 0 = T (mapped sorts first), rank 1 = the unmapped read
    add(5, READ_UNMAPPED, _mut(_mut(G, 0, "T"), 1, "T") + "GTA")
    add(5, 10, T + "TAC", peak=50)
    # cell 6: low quality everywhere -> passed_q_min False once q_min > 0.2
    add(6, READ_LOW_Q_MIN, A + "ACG", q=0.2)
    # cell 0: background read -> removed by call_cells' `cell > 0`
    add(0, 13, C + "CGT")
    return pd.DataFrame(rows)


def make_simple_inputs():
    """Single-barcode variant: 12-mer barcodes, library keyed by `prefix`."""
    reads = make_reads()
    reads["barcode"] = reads["barcode"].str[:12]
    lib = LIBRARY.rename(columns={"prefix_map": "prefix"}).drop(columns="prefix_recomb")
    return reads, lib
