"""Construct- and gene-level bootstrap p-values for one (cell_class, channel_combo).

Nulls are drawn in memory per gene (beta-quantile shortcut, no resampling, nothing
persisted); the two result tables keep the schema of the old per-construct fan-out.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from lib.aggregate.bootstrap import (
    apply_multiple_hypothesis_correction,
    bootstrap_gene,
    prepare_control_pools,
)
from lib.aggregate.cell_data_utils import get_feature_table_cols

for _param_name in ["perturbation_name_col", "control_key"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

pert_col = snakemake.params.perturbation_name_col
pert_id_col = snakemake.params.perturbation_id_col or pert_col
control_key = snakemake.params.control_key
exclusion_string = snakemake.params.exclusion_string
num_sims = snakemake.params.num_sims
construct_out, gene_out = snakemake.output[0], snakemake.output[1]

construct_table = pd.read_csv(snakemake.input.construct_table, sep="\t")
gene_table = pd.read_csv(snakemake.input.gene_table, sep="\t")

# Same feature selection as before: an explicit list, else the default nucleus/cell filter.
all_features = [
    c for c in construct_table.columns if c not in (pert_id_col, pert_col, "cell_count")
]
if snakemake.params.bootstrap_features_fp is not None:
    with open(snakemake.params.bootstrap_features_fp) as f:
        requested = [line.strip() for line in f if line.strip()]
    features = [c for c in requested if c in all_features]
    if missing := set(requested) - set(features):
        print(
            f"Warning: {len(missing)} requested features not in data: {sorted(missing)}"
        )
else:
    features = get_feature_table_cols(all_features)
print(f"Using {len(features)} of {len(all_features)} features")

# Control cells only, feature columns only; the pool id is the control's perturbation name.
controls = (
    ds.dataset(snakemake.input.features_singlecell)
    .to_table(
        columns=[pert_col] + features,
        filter=pc.match_substring(ds.field(pert_col), control_key),
    )
    .to_pandas()
)
print(f"Control cells: {len(controls)} in {controls[pert_col].nunique()} pools")

if exclusion_string is not None:
    construct_table = construct_table[
        ~construct_table[pert_col].str.contains(exclusion_string, na=False)
    ]

if len(controls) == 0 or len(construct_table) == 0 or len(features) == 0:
    print("WARNING: nothing to bootstrap, writing empty outputs")
    pd.DataFrame().to_csv(construct_out, sep="\t", index=False)
    pd.DataFrame().to_csv(gene_out, sep="\t", index=False)
    exit(0)

POOLS, SIZES = prepare_control_pools(controls.values)
del controls

gene_observed = gene_table.set_index(pert_col)[features].astype(float)
tasks = []
for gene, rows in construct_table.groupby(pert_col, sort=False):
    if gene not in gene_observed.index:
        print(f"Warning: {gene} has constructs but no gene-table row, skipped")
        continue
    constructs = [
        (str(r[pert_id_col]), int(r["cell_count"]), r[features].to_numpy(dtype=float))
        for _, r in rows.iterrows()
    ]
    tasks.append((len(tasks), gene, constructs, gene_observed.loc[gene].to_numpy()))
print(f"{len(tasks)} genes, {len(construct_table)} constructs, {num_sims} sims")


def _work(task):
    i, gene, constructs, observed = task
    rows, gene_p, n_used = bootstrap_gene(
        POOLS, SIZES, constructs, observed, num_sims, np.random.default_rng([0, i])
    )
    return gene, rows, gene_p, n_used


# fork: workers only read the module-level pools; nothing has been written yet.
with mp.get_context("fork").Pool(snakemake.threads) as pool:
    results = list(pool.imap_unordered(_work, tasks, chunksize=16))

construct_rows, gene_rows = [], []
for gene, rows, gene_p, n_used in results:
    for cid, n, p in rows:
        construct_rows.append(
            [gene, cid, n, num_sims]
            + (list(p) if p is not None else [np.nan] * len(features))
        )
    total_cells = int(sum(n for _, n, _ in rows))
    gene_rows.append(
        [gene, n_used, total_cells, num_sims]
        + (list(gene_p) if gene_p is not None else [np.nan] * len(features))
    )

construct_df = pd.DataFrame(
    construct_rows, columns=["gene", "construct", "sample_size", "num_sims"] + features
).sort_values(["gene", "construct"])
gene_df = pd.DataFrame(
    gene_rows, columns=["gene", "num_constructs", "total_cells", "num_sims"] + features
).sort_values("gene")

apply_multiple_hypothesis_correction(construct_df, features).to_csv(
    construct_out, sep="\t", index=False
)
apply_multiple_hypothesis_correction(gene_df, features).to_csv(
    gene_out, sep="\t", index=False
)
print(f"Wrote {len(construct_df)} construct rows and {len(gene_df)} gene rows")
