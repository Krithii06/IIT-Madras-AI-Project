"""Checks on the manifest and, most importantly, on the split being leak-free.

The leaf-grouped split is the one part of this project where a silent regression
would quietly inflate every number in the report, so it gets tested directly.
"""

import pytest

from src import config
from src.data.dataset import (check_leaf_disjoint, leaf_grouped_split, load_manifest,
                              random_image_split)


@pytest.fixture(scope="module")
def rows():
    if not config.MANIFEST_PATH.exists():
        pytest.skip("manifest not built; run src.data.prepare first")
    return load_manifest()


def test_manifest_covers_the_four_selected_classes(rows):
    assert {r["source_class"] for r in rows} == set(config.SOURCE_CLASSES)


def test_binary_label_matches_the_source_class(rows):
    for row in rows:
        assert row["binary_label"] == config.to_binary_label(row["source_class"])


def test_every_image_has_a_leaf_id(rows):
    assert all(r["leaf_id"] for r in rows)


def test_leaf_grouped_split_shares_no_leaf_between_splits(rows):
    train, val, test = leaf_grouped_split(rows)
    overlap = check_leaf_disjoint(train, val, test)
    assert overlap == {
        "train_val_shared_leaves": 0,
        "train_test_shared_leaves": 0,
        "val_test_shared_leaves": 0,
    }


def test_split_is_a_partition_of_the_manifest(rows):
    train, val, test = leaf_grouped_split(rows)
    paths = [r["rel_path"] for r in train + val + test]
    assert len(paths) == len(rows)
    assert set(paths) == {r["rel_path"] for r in rows}


def test_test_split_is_the_official_held_out_set(rows):
    _, _, test = leaf_grouped_split(rows)
    assert {r["official_split"] for r in test} == {"test"}
    assert len(test) == sum(1 for r in rows if r["official_split"] == "test")


def test_split_is_reproducible_for_a_fixed_seed(rows):
    first = leaf_grouped_split(rows, seed=config.SEED)
    second = leaf_grouped_split(rows, seed=config.SEED)
    for a, b in zip(first, second):
        assert [r["rel_path"] for r in a] == [r["rel_path"] for r in b]


def test_both_binary_classes_present_in_every_split(rows):
    for subset in leaf_grouped_split(rows):
        assert {r["binary_label"] for r in subset} == set(config.BINARY_CLASSES)


def test_random_split_really_does_leak_leaves(rows):
    """Guards the leakage experiment itself.

    If this ever came back clean the comparison in the report would be meaningless,
    so assert that the naive split does what it is supposed to demonstrate.
    """
    train, val, _ = random_image_split(rows)
    shared = {r["leaf_id"] for r in train} & {r["leaf_id"] for r in val}
    assert len(shared) > 0
