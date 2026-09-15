"""Issue #10: phenotype OME-Zarr channels surface biological marker names.

Verifies that ``biological_annotation.marker`` (falling back to
``full_label``) from ``channels_metadata`` becomes the ``omero.channels[].label``
written into zarr.json, while the flat ``channel_names`` remains the fallback
and SBS stores are unaffected.
"""

from types import SimpleNamespace

from lib.shared.hcs import (
    _resolve_channel_names_for_store,
    _build_and_set_omero,
)


def _fake_pos(n_channels):
    """Minimal stand-in for an iohub Position for name/omero resolution."""
    return SimpleNamespace(
        channel_names=[f"c{i}" for i in range(n_channels)],
        metadata=SimpleNamespace(omero=None),
    )


PHENO_META = [
    {"name": "DAPI", "index": 0, "color": "0000FF",
     "biological_annotation": {"marker": "DAPI", "full_label": "nucleus, DAPI"}},
    {"name": "COXIV", "index": 1, "color": "00FF00",
     "biological_annotation": {"marker": "pTDP43", "full_label": "TDP43 (phospho)"}},
    {"name": "CENPA", "index": 2, "color": "FF0000",
     "biological_annotation": {"marker": "PGRN", "full_label": "progranulin"}},
    {"name": "WGA", "index": 3, "color": "FF00FF",
     "biological_annotation": {"marker": "TDP43", "full_label": "cell membrane, WGA"}},
]

FLAT_NAMES = ["DAPI", "COXIV", "CENPA", "WGA"]


def _omero_labels(resolved, channels_metadata):
    pos = _fake_pos(len(resolved))
    _build_and_set_omero(pos, resolved, channels_metadata=channels_metadata)
    return [ch.label for ch in pos.metadata.omero.channels]


def test_phenotype_marker_becomes_omero_label():
    resolved = _resolve_channel_names_for_store(
        _fake_pos(4), FLAT_NAMES, "aligned", channels_metadata=PHENO_META
    )
    assert resolved == ["DAPI", "pTDP43", "PGRN", "TDP43"]
    # omero labels track resolved names (what the picker reads)
    assert _omero_labels(resolved, PHENO_META) == ["DAPI", "pTDP43", "PGRN", "TDP43"]


def test_full_label_fallback_when_no_marker():
    meta = [
        {"index": 0, "biological_annotation": {"full_label": "nucleus, DAPI"}},
        {"index": 1, "biological_annotation": {"full_label": "mito, COXIV"}},
        # index 2/3 have no bio marker at all -> flat name kept
        {"index": 2},
        {"index": 3, "biological_annotation": {}},
    ]
    resolved = _resolve_channel_names_for_store(
        _fake_pos(4), FLAT_NAMES, "aligned", channels_metadata=meta
    )
    assert resolved == ["nucleus, DAPI", "mito, COXIV", "CENPA", "WGA"]


def test_fallback_to_flat_names_without_metadata():
    # Current behavior preserved: no channels_metadata -> flat channel_names.
    resolved = _resolve_channel_names_for_store(
        _fake_pos(4), FLAT_NAMES, "aligned"
    )
    assert resolved == FLAT_NAMES


def test_sbs_store_unaffected():
    # Call site passes channels_metadata=None for SBS, so even with SBS marker
    # metadata present the base names (DAPI/G/T/A/C) survive.
    sbs_flat = ["DAPI", "G", "T", "A", "C"]
    resolved = _resolve_channel_names_for_store(
        _fake_pos(5), sbs_flat, "aligned", channels_metadata=None
    )
    assert resolved == sbs_flat
