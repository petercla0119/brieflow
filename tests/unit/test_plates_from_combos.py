"""Unit tests for plates_from_combos + the multi-plate benchmark tag (issue #29).

A multi-plate run has no --plate-filter, so every benchmark row was stamped with a
blank plate and you could not tell which plates a row covered. These pin the two
halves of the fix: deriving the plate set from the combo TSVs, and accepting it as
a tag.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflow"))
from lib.shared.resource_monitor import (  # noqa: E402
    plates_from_combos,
    set_benchmark_context,
)


def _combo(tmp_path, name, plates):
    fp = tmp_path / name
    pd.DataFrame({"plate": plates, "well": ["A1"] * len(plates)}).to_csv(
        fp, sep="\t", index=False
    )
    return str(fp)


def test_union_across_files_deduped_and_sorted(tmp_path):
    a = _combo(tmp_path, "sbs.tsv", [4, 4, 5])
    b = _combo(tmp_path, "phenotype.tsv", [5, 1])
    assert plates_from_combos(a, b) == "1,4,5"


def test_single_plate_still_reads_as_one_value(tmp_path):
    assert plates_from_combos(_combo(tmp_path, "c.tsv", [4, 4, 4])) == "4"


def test_numeric_sort_not_lexicographic(tmp_path):
    # plain string sort would give "1,10,2"; benchmark rows should read "1,2,10"
    assert plates_from_combos(_combo(tmp_path, "c.tsv", [10, 2, 1])) == "1,2,10"


def test_missing_and_none_paths_are_skipped_not_raised(tmp_path):
    good = _combo(tmp_path, "good.tsv", [7])
    # benchmark bookkeeping must never be able to fail a pipeline run
    assert plates_from_combos(good, None, str(tmp_path / "nope.tsv")) == "7"


def test_file_without_plate_column_is_skipped(tmp_path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"well": ["A1"]}).to_csv(bad, sep="\t", index=False)
    assert plates_from_combos(str(bad)) == ""


def test_no_inputs_gives_empty_tag():
    assert plates_from_combos() == ""


@pytest.mark.parametrize(
    "plate,expected",
    [
        (None, ""),  # genuinely unknown
        (4, "4"),  # --plate-filter 4
        ("4,5", "4,5"),  # already-joined, e.g. from plates_from_combos
        ([5, 4], "4,5"),  # iterable, sorted
        ({10, 2}, "2,10"),  # set, numeric order
    ],
)
def test_set_benchmark_context_tag_forms(tmp_path, plate, expected):
    set_benchmark_context("sbs", str(tmp_path), plate=plate)
    assert os.environ["BRIEFLOW_PLATE"] == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
