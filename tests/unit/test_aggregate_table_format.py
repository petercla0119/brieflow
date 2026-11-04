"""Wide aggregate tables may be parquet or tsv; every reader accepts both (issue #96).

Three layers: the shared ``read_df`` / ``write_df`` dispatch, the
``aggregate.table_format`` switch in the aggregate targets, and the Snakemake
scripts that consume the tables, run against a tsv and a parquet copy of the
same inputs.
"""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.shared.parquet_io import read_df, write_df  # noqa: E402

FORMATS = ("tsv", "parquet")


# --------------------------------------------------------------------------- #
# read_df / write_df
# --------------------------------------------------------------------------- #
def _wide_frame(n_rows=25, n_feats=40):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "gene_symbol_0": [f"G{i % 9}" for i in range(n_rows)],
            "cell_count": rng.integers(1, 500, n_rows),
            "is_control": rng.integers(0, 2, n_rows).astype(bool),
            "with_nan": rng.normal(size=n_rows),
        }
    )
    df.loc[df.index[::7], "with_nan"] = np.nan
    feats = pd.DataFrame(
        rng.normal(size=(n_rows, n_feats)),
        columns=[f"cell_feature_{i}" for i in range(n_feats)],
    )
    return pd.concat([df, feats], axis=1)


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_values(tmp_path, fmt):
    df = _wide_frame()
    path = tmp_path / f"table.{fmt}"
    write_df(df, path)
    back = read_df(path)
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    assert list(back.columns) == list(df.columns)


def test_parquet_keeps_dtypes(tmp_path):
    df = _wide_frame()
    path = tmp_path / "table.parquet"
    write_df(df, path)
    back = read_df(path)
    assert back["cell_count"].dtype == df["cell_count"].dtype
    assert back["is_control"].dtype == np.bool_
    assert back["cell_feature_0"].dtype == np.float64


def test_parquet_preserves_float_precision(tmp_path):
    df = pd.DataFrame({"f": [0.1 + 0.2, np.pi, 1e-300, 1.0 / 3.0]})
    write_df(df, tmp_path / "t.parquet")
    assert np.array_equal(read_df(tmp_path / "t.parquet")["f"].to_numpy(), df["f"])


@pytest.mark.parametrize("fmt", FORMATS)
def test_empty_frame_round_trips(tmp_path, fmt):
    # The aggregate scripts write pd.DataFrame() on empty input; a bare tsv of
    # that used to raise EmptyDataError in every downstream reader.
    path = tmp_path / f"empty.{fmt}"
    write_df(pd.DataFrame(), path)
    back = read_df(path)
    assert back.empty and back.shape == (0, 0)


@pytest.mark.parametrize("fmt", FORMATS)
def test_accepts_str_and_path(tmp_path, fmt):
    df = _wide_frame(5, 3)
    write_df(df, str(tmp_path / f"t.{fmt}"))
    pd.testing.assert_frame_equal(
        read_df(str(tmp_path / f"t.{fmt}")), df, check_dtype=False
    )


def test_mixed_object_columns_write_as_parquet(tmp_path):
    # `.values` of a str + float frame gives an all-object array; the bootstrap
    # prep used to build its export tables that way.
    arr = pd.DataFrame({"id": ["a", "b"], "f1": [1.5, 2.5], "f2": [0.0, 1.0]}).values
    df = pd.DataFrame(arr, columns=["id", "f1", "f2"]).infer_objects()
    write_df(df, tmp_path / "t.parquet")
    back = read_df(tmp_path / "t.parquet")
    assert back["f1"].tolist() == [1.5, 2.5]


@pytest.mark.parametrize("name", ["t.csv", "t.txt", "t"])
def test_unknown_suffix_raises(tmp_path, name):
    with pytest.raises(ValueError, match="parquet or .tsv"):
        write_df(pd.DataFrame({"a": [1]}), tmp_path / name)
    with pytest.raises(ValueError, match="parquet or .tsv"):
        read_df(tmp_path / name)


def test_read_df_does_not_probe_siblings(tmp_path):
    # Unlike read_table: a stale parquet next to the declared tsv must be ignored.
    write_df(pd.DataFrame({"a": [1]}), tmp_path / "t.parquet")
    write_df(pd.DataFrame({"a": [2]}), tmp_path / "t.tsv")
    assert read_df(tmp_path / "t.tsv")["a"].tolist() == [2]
    assert read_df(tmp_path / "t.parquet")["a"].tolist() == [1]


# --------------------------------------------------------------------------- #
# aggregate targets: `aggregate.table_format`
# --------------------------------------------------------------------------- #
def _aggregate_outputs(table_format=None):
    cfg = {"aggregate": {}}
    if table_format is not None:
        cfg["aggregate"]["table_format"] = table_format
    combos = pd.DataFrame(
        {
            "plate": [1],
            "well": ["A1"],
            "cell_class": ["all"],
            "channel_combo": ["DAPI"],
        }
    )
    g = {
        "ROOT_FP": Path("/root"),
        "config": cfg,
        "aggregate_wildcard_combos": combos,
        "IMG_FMT": "tiff",
    }
    exec((_WORKFLOW / "targets" / "aggregate.smk").read_text(), g)
    return g


WIDE_TABLES = [
    ("AGGREGATE_OUTPUTS", "generate_feature_table", 1),  # features_constructs
    ("AGGREGATE_OUTPUTS", "generate_feature_table", 2),  # features_genes
    ("AGGREGATE_OUTPUTS", "aggregate", 0),  # aggregated
    ("BOOTSTRAP_OUTPUTS", "controls_arr", None),
    ("BOOTSTRAP_OUTPUTS", "construct_features_arr", None),
    ("BOOTSTRAP_OUTPUTS", "sample_sizes", None),
    ("BOOTSTRAP_OUTPUTS", "combined_construct_results", None),
    ("BOOTSTRAP_OUTPUTS", "combined_gene_results", None),
]


def _path(g, table, key, idx):
    p = g[table][key]
    return Path(p if idx is None else p[idx])


@pytest.mark.parametrize("table,key,idx", WIDE_TABLES)
def test_wide_tables_default_to_parquet(table, key, idx):
    assert _path(_aggregate_outputs(), table, key, idx).suffix == ".parquet"


@pytest.mark.parametrize("table,key,idx", WIDE_TABLES)
def test_wide_tables_follow_table_format_tsv(table, key, idx):
    assert _path(_aggregate_outputs("tsv"), table, key, idx).suffix == ".tsv"


def test_table_format_moves_feature_tables_between_dirs():
    parq = _aggregate_outputs()
    tsv = _aggregate_outputs("tsv")
    assert _path(parq, "AGGREGATE_OUTPUTS", "aggregate", 0).parent.name == "parquets"
    assert _path(tsv, "AGGREGATE_OUTPUTS", "aggregate", 0).parent.name == "tsvs"


def test_unchanged_outputs_keep_their_format():
    g = _aggregate_outputs()
    assert (
        _path(g, "AGGREGATE_OUTPUTS", "generate_feature_table", 0).suffix == ".parquet"
    )
    assert _path(g, "AGGREGATE_OUTPUTS", "eval_aggregate", 0).suffix == ".tsv"
    assert _path(g, "MONTAGE_OUTPUTS", "montage_data", None).suffix == ".tsv"
    assert _path(g, "BOOTSTRAP_OUTPUTS", "construct_data", None).suffix == ".tsv"
    assert _path(g, "BOOTSTRAP_OUTPUTS", "bootstrap_gene_pvals", None).suffix == ".tsv"


def test_invalid_table_format_raises():
    with pytest.raises(ValueError, match="table_format"):
        _aggregate_outputs("csv")


# --------------------------------------------------------------------------- #
# Snakemake scripts, run against tsv and parquet inputs
# --------------------------------------------------------------------------- #
_SCRIPTS = _WORKFLOW / "scripts"


class _IO(list):
    """Snakemake-style input/output: indexable and attribute-addressable."""

    def __init__(self, *positional, **named):
        super().__init__(list(positional) + list(named.values()))
        self.__dict__.update(named)


def _run_script(rel, **snakemake_attrs):
    sm = SimpleNamespace(**snakemake_attrs)
    try:
        runpy.run_path(str(_SCRIPTS / rel), init_globals={"snakemake": sm})
    except SystemExit as exc:  # scripts `exit(0)` on empty input
        assert exc.code in (0, None), exc.code


def _aggregated_table():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "gene_symbol_0": ["nontargeting", "TARDBP", "FUS", "G3"],
            "cell_count": [400, 120, 3, 0],
            **{f"PC_{i}": rng.normal(size=4) for i in range(6)},
        }
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_clean_aggregate_reads_either_format(tmp_path, fmt):
    agg = _aggregated_table()
    inp = tmp_path / f"aggregated.{fmt}"
    write_df(agg, inp)
    out = tmp_path / f"aggregate_cleaned_{fmt}.tsv"
    _run_script(
        "cluster/clean_aggregate.py",
        input=_IO(str(inp)),
        output=_IO(str(out)),
        params=SimpleNamespace(cell_class="all", min_cell_cutoffs={"all": 10}),
    )
    cleaned = pd.read_csv(out, sep="\t")
    expected = agg[agg["cell_count"] > 10].reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, expected, check_dtype=False)


def _bootstrap_inputs(tmp_path, fmt):
    tmp_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(2)
    feats = [f"cell_feature_{i}" for i in range(5)]
    controls = pd.DataFrame(
        {
            "gene_symbol_0": ["nontargeting"] * 60,
            **{f: rng.normal(size=60) for f in feats},
        }
    )
    constructs = pd.DataFrame(
        {
            "cell_barcode_0": ["sg1", "sg2"],
            **{f: rng.normal(size=2) + 1.0 for f in feats},
        }
    )
    sizes = pd.DataFrame({"cell_barcode_0": ["sg1", "sg2"], "cell_count": [12, 7]})
    construct_data = pd.DataFrame({"construct_id": ["sg2"], "gene": ["TARDBP"]})
    paths = {}
    for name, df in [
        ("controls_arr", controls),
        ("construct_features_arr", constructs),
        ("sample_sizes", sizes),
    ]:
        paths[name] = tmp_path / f"{name}.{fmt}"
        write_df(df, paths[name])
    paths["construct_data"] = tmp_path / "construct_data.tsv"
    write_df(construct_data, paths["construct_data"])
    return paths, feats


def _run_bootstrap_construct(tmp_path, fmt):
    paths, feats = _bootstrap_inputs(tmp_path, fmt)
    nulls = tmp_path / f"nulls_{fmt}.npy"
    pvals = tmp_path / f"pvals_{fmt}.tsv"
    _run_script(
        "aggregate/bootstrap_construct.py",
        input=_IO(**{k: str(v) for k, v in paths.items()}),
        output=_IO(str(nulls), str(pvals)),
        params=SimpleNamespace(num_sims=40),
    )
    return pd.read_csv(pvals, sep="\t"), np.load(nulls), feats


@pytest.mark.parametrize("fmt", FORMATS)
def test_bootstrap_construct_reads_either_format(tmp_path, fmt):
    # The numba bootstrap kernel has its own RNG state, so p-values are not
    # reproducible across runs; check the structure that depends on the inputs.
    pvals, nulls, feats = _run_bootstrap_construct(tmp_path / fmt, fmt=fmt)
    assert (
        list(pvals.columns) == ["gene", "construct", "sample_size", "num_sims"] + feats
    )
    assert pvals.loc[0, ["gene", "construct", "sample_size", "num_sims"]].tolist() == [
        "TARDBP",
        "sg2",
        7,
        40,
    ]
    assert ((pvals[feats] >= 0) & (pvals[feats] <= 1)).all().all()
    assert nulls.shape == (40, len(feats))


@pytest.mark.parametrize("fmt", FORMATS)
def test_prepare_bootstrap_data_empty_input_writes_readable_tables(tmp_path, fmt):
    features = tmp_path / "features_singlecell.parquet"
    pd.DataFrame().to_parquet(features, index=False)
    meta_cols = tmp_path / "metadata_cols.tsv"
    meta_cols.write_text("gene_symbol_0\ncell_barcode_0\n")
    outs = {
        k: tmp_path / f"{k}.{fmt}"
        for k in ("controls_arr", "construct_features_arr", "sample_sizes")
    }
    _run_script(
        "aggregate/prepare_bootstrap_data.py",
        input=_IO(
            features_singlecell=str(features),
            construct_table=str(tmp_path / f"features_constructs.{fmt}"),
            gene_table=str(tmp_path / f"features_genes.{fmt}"),
        ),
        output=_IO(
            str(tmp_path / "bootstrap_data"), **{k: str(v) for k, v in outs.items()}
        ),
        params=SimpleNamespace(
            perturbation_name_col="gene_symbol_0",
            perturbation_id_col="cell_barcode_0",
            control_key="nontargeting",
            metadata_cols_fp=str(meta_cols),
            use_classifier=False,
            exclusion_string=None,
            bootstrap_features_fp=None,
        ),
    )
    for path in outs.values():
        assert read_df(path).empty


def _features_constructs_and_genes(tmp_path, fmt):
    tmp_path.mkdir(exist_ok=True)
    """Minimal construct + gene tables plus a features_singlecell parquet for prepare_bootstrap_data."""
    rng = np.random.default_rng(3)
    # names must pass get_feature_table_cols (compartment prefix + intensity/shape tag)
    feats = ["cell_DAPI_mean", "nucleus_DAPI_integrated", "cell_area"]
    cells = pd.DataFrame(
        {
            "gene_symbol_0": ["nontargeting"] * 30 + ["TARDBP"] * 10,
            "cell_barcode_0": ["nt1"] * 30 + ["sg1"] * 10,
            "batch_values": ["1_A1"] * 40,
            **{f: rng.normal(size=40) for f in feats},
        }
    )
    constructs = pd.DataFrame(
        {
            "cell_barcode_0": ["nt1", "sg1"],
            "gene_symbol_0": ["nontargeting", "TARDBP"],
            "cell_count": [30, 10],
            **{f: rng.normal(size=2) for f in feats},
        }
    )
    genes = constructs.drop(columns="cell_barcode_0")
    features = tmp_path / "features_singlecell.parquet"
    write_df(cells, features)
    ct = tmp_path / f"features_constructs.{fmt}"
    gt = tmp_path / f"features_genes.{fmt}"
    write_df(constructs, ct)
    write_df(genes, gt)
    meta_cols = tmp_path / "metadata_cols.tsv"
    meta_cols.write_text("gene_symbol_0\ncell_barcode_0\n")
    return features, ct, gt, meta_cols, feats


def _run_prepare_bootstrap_data(tmp_path, fmt):
    features, ct, gt, meta_cols, feats = _features_constructs_and_genes(tmp_path, fmt)
    outs = {
        k: tmp_path / f"{k}.{fmt}"
        for k in ("controls_arr", "construct_features_arr", "sample_sizes")
    }
    _run_script(
        "aggregate/prepare_bootstrap_data.py",
        input=_IO(
            features_singlecell=str(features),
            construct_table=str(ct),
            gene_table=str(gt),
        ),
        output=_IO(
            str(tmp_path / "bootstrap_data"), **{k: str(v) for k, v in outs.items()}
        ),
        params=SimpleNamespace(
            perturbation_name_col="gene_symbol_0",
            perturbation_id_col="cell_barcode_0",
            control_key="nontargeting",
            metadata_cols_fp=str(meta_cols),
            use_classifier=False,
            exclusion_string=None,
            bootstrap_features_fp=None,
        ),
    )
    return {k: read_df(v) for k, v in outs.items()}, feats


def test_prepare_bootstrap_data_identical_for_tsv_and_parquet(tmp_path):
    out_tsv, feats = _run_prepare_bootstrap_data(tmp_path / "tsv", "tsv")
    out_parq, _ = _run_prepare_bootstrap_data(tmp_path / "parquet", "parquet")
    assert list(out_tsv["controls_arr"].columns) == ["gene_symbol_0"] + feats
    assert out_tsv["controls_arr"].shape == (30, 4)
    assert out_tsv["sample_sizes"]["cell_count"].tolist() == [30, 10]
    for k in out_tsv:
        pd.testing.assert_frame_equal(out_tsv[k], out_parq[k], check_dtype=False)
