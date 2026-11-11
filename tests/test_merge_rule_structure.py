"""Structural guard on workflow/rules/merge.smk.

format_merge only emits x_pos/y_pos/pixel_size/global_i/global_j when the rule hands
it both preprocess metadata files. Dropping one of those inputs is a one-line edit
that leaves the DAG valid; the script would then raise at run time, but only after a
long pipeline. Catch it at collection time instead, like test_ci_workflow_structure.py
does for the CI matrix.
"""

import re
from pathlib import Path

RULES_FP = Path(__file__).resolve().parents[1] / "workflow/rules/merge.smk"
TEXT = RULES_FP.read_text()


def _rule_block(name):
    m = re.search(rf"^rule {name}:\n(.*?)(?=^rule |\Z)", TEXT, re.S | re.M)
    assert m, f"rule {name} not found in {RULES_FP}"
    return m.group(1)


def test_format_merge_declares_both_metadata_inputs():
    block = _rule_block("format_merge")
    for name in ("phenotype_metadata=", "sbs_metadata="):
        assert name in block, f"format_merge lost its {name} input"
    assert "combine_metadata_phenotype" in block and "combine_metadata_sbs" in block


def test_format_merge_passes_tile_dimensions():
    block = _rule_block("format_merge")
    for param in ("phenotype_dimensions=", "sbs_dimensions=", "approach="):
        assert param in block, f"format_merge lost its {param} param"
    assert "../scripts/merge/format_merge.py" in block
