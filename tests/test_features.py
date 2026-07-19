# -*- coding: utf-8 -*-
"""End-to-end smoke test for FeatureEngineer on synthetic data.

Asserts structural invariants only (shapes, columns, no-NaN, force-kept
features, saved artefacts) — not canonical numbers, which require the real
UCI dataset that is not shipped in the repo.
"""

import json

import pytest

from clustering_library import FeatureEngineer


@pytest.fixture
def fitted_engineer(cleaned_csv, raw_csv):
    fe = FeatureEngineer(str(cleaned_csv), raw_data_path=str(raw_csv))
    fe.load_data()
    fe.create_customer_features()
    fe.transform_features()
    fe.scale_features()
    return fe


def test_create_customer_features_structure(fitted_engineer):
    fe = fitted_engineer
    feats = fe.customer_features

    assert "CustomerID" in feats.columns
    # One row per cleaned customer, no missing values.
    assert feats["CustomerID"].is_unique
    assert not feats.isnull().values.any()
    # Selected features are a subset of the declared candidates.
    assert set(fe.feature_customer).issubset(set(FeatureEngineer.CANDIDATES))
    # Force-kept anchors always survive the correlation filter.
    assert set(FeatureEngineer.FORCE_KEEP).issubset(set(fe.feature_customer))


def test_transform_and_scale_preserve_shape(fitted_engineer):
    fe = fitted_engineer
    n_customers = fe.customer_features.shape[0]
    n_features = len(fe.feature_customer)

    assert fe.customer_features_transformed.shape == (n_customers, n_features)
    assert fe.customer_features_scaled.shape == (n_customers, n_features)
    assert list(fe.customer_features_scaled.columns) == fe.feature_customer
    # Every family weight is strictly positive.
    assert fe.family_weights
    assert all(w > 0 for w in fe.family_weights.values())


def test_save_pipeline_writes_artifacts(fitted_engineer, tmp_path):
    fe = fitted_engineer
    out = tmp_path / "models"
    fe.save_pipeline(str(out))

    assert (out / "scaler.pkl").is_file()
    assert (out / "quantile_transformer.pkl").is_file()
    meta_path = out / "pipeline_metadata.json"
    assert meta_path.is_file()

    meta = json.loads(meta_path.read_text())
    assert meta["feature_order"] == fe.feature_customer
    assert meta["n_candidates"] == len(FeatureEngineer.CANDIDATES)
