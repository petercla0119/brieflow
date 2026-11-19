import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

_REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (_REPO_ROOT, _REPO_ROOT / "workflow"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.aggregate.align import (  # noqa: E402
    centerscale_by_batch,
    embed_files,
    prepare_alignment_data,
)

META = ["plate", "well", "tile", "cell_0", "gene_symbol_0", "cell_barcode_0"]


def _well(rng, plate, well, n, n_feat):
    genes = rng.choice(["nontargeting", "TARDBP", "STMN2", "FUS"], size=n)
    df = pd.DataFrame(
        {
            "plate": plate,
            "well": well,
            "tile": rng.integers(0, 5, n),
            "cell_0": np.arange(n),
            "gene_symbol_0": genes,
            "cell_barcode_0": [f"{g}_{i % 3}" for i, g in enumerate(genes)],
        }
    )
    feats = pd.DataFrame(
        rng.normal(size=(n, n_feat)).astype(np.float32) + np.arange(n_feat, dtype=np.float32),
        columns=[f"cell_f{j}_mean" for j in range(n_feat)],
    )
    return pd.concat([df, feats], axis=1)


@pytest.mark.unit
def test_embed_files_parallel_matches_serial(tmp_path):
    rng = np.random.default_rng(0)
    frames = [_well(rng, 1, w, n, 12) for w, n in (("A1", 150), ("A2", 90), ("B1", 120))]
    paths = []
    for i, f in enumerate(frames):
        p = tmp_path / f"well{i}.parquet"
        f.to_parquet(p, index=False)
        paths.append(str(p))

    pooled = pd.concat(frames, ignore_index=True)
    meta, feats = pooled[META].copy(), pooled.drop(columns=META)
    meta, feats = prepare_alignment_data(
        meta, feats, ["plate", "well"], "gene_symbol_0", "nontargeting", "cell_barcode_0"
    )
    pca = PCA(n_components=5).fit(centerscale_by_batch(feats, meta, "batch_values"))

    kwargs = dict(
        columns=list(pooled.columns),
        metadata_cols=META + ["perturbation_score", "perturbation_auc"],
        pca=pca,
        batch_cols=["plate", "well"],
        pert_col="gene_symbol_0",
        control_key="nontargeting",
        pert_id_col="cell_barcode_0",
    )
    m1, e1 = embed_files(paths, n_jobs=1, **kwargs)
    m2, e2 = embed_files(paths, n_jobs=2, **kwargs)

    assert e1.dtype == np.float32 and e1.shape == (360, 5)
    np.testing.assert_array_equal(e1, e2)
    pd.testing.assert_frame_equal(m1, m2)
    # file order is preserved whatever n_jobs is
    assert m1["well"].tolist() == ["A1"] * 150 + ["A2"] * 90 + ["B1"] * 120
