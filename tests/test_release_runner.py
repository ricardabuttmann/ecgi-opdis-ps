from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from ecgi.forward.measurement_whitening import (
    ConditionalApproximationError,
    MeasurementWhitening,
)
from opdis_ps.data import (
    OPERATOR_DISCREPANCIES,
    PUBLIC_EXAMPLES,
    load_example_case,
    public_example_path,
)
from opdis_ps.results import METHOD_LABELS, refresh_results_table
from opdis_ps.runner import IMPLEMENTED_METHODS, run_method


def _example(path: Path) -> None:
    rng = np.random.default_rng(7)
    n_electrodes, n_state, n_time = 5, 8, 7
    forward = rng.normal(size=(n_electrodes, n_state))
    truth = rng.normal(size=(n_state, n_time))
    measurement = forward @ truth
    laplacian = sp.diags(
        [-np.ones(n_state - 1), 2 * np.ones(n_state), -np.ones(n_state - 1)],
        [-1, 0, 1],
        format="csr",
    )
    np.savez_compressed(
        path,
        schema_version=np.asarray(2),
        measurement=measurement,
        forward=forward,
        truth=truth,
        points_mm=np.column_stack(
            (np.arange(n_state, dtype=float), np.zeros(n_state), np.zeros(n_state))
        ),
        laplacian_data=laplacian.data,
        laplacian_indices=laplacian.indices,
        laplacian_indptr=laplacian.indptr,
        laplacian_shape=np.asarray(laplacian.shape),
        template_z=np.repeat([0.25, 0.75], 4),
        template_phi=np.tile(np.linspace(-np.pi, np.pi, 4, endpoint=False), 2),
        template_shape=np.asarray((2, 4)),
        lambda_min=np.asarray(0.1),
        dt_ms=np.asarray(1.0),
        target_snr_db=np.asarray(40.0),
        realized_snr_db=np.asarray(40.0),
        noise_norm=np.asarray(0.1),
        case_id=np.asarray("example-001"),
        operator_discrepancy=np.asarray("lung-conductivity"),
    )


def test_tracked_public_examples_are_loadable_and_stable():
    root = Path(__file__).parents[1]
    forbidden = ("/home/", "/Users/")
    expected_digests = {
        "lung-conductivity": "302b4749f76b656cdea5ecf08931cfa5d7f0896d79a2d6a7a779df3f07200e0b",
        "subcutaneous-fat": "d495eeadd346a701a693367551237a11bd884b24328ab6ce996f6d4288df35bb",
        "lbbb-lung-conductivity": "81f0367db1c9480a51e6892452d6274c0a43476f05ad0c88aa8036490e32a4d3",
    }
    discrepancies = {
        "lung-conductivity": "lung-conductivity",
        "subcutaneous-fat": "subcutaneous-fat",
        "lbbb-lung-conductivity": "lung-conductivity",
    }
    cases = {}
    for name in PUBLIC_EXAMPLES:
        path = root / public_example_path(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as bundle:
            strings = [
                str(value)
                for key in bundle.files
                if bundle[key].dtype.kind in "US"
                for value in bundle[key].reshape(-1)
            ]
        case = load_example_case(path)
        cases[name] = case

        assert digest == expected_digests[name]
        assert not any(token in value for token in forbidden for value in strings)
        assert case.case_id == f"{name}-001"
        assert case.operator_discrepancy == discrepancies[name]
        assert case.measurement.shape == (176, 91)
        assert case.forward.shape == (176, 1800)
        assert case.truth.shape == (1800, 91)
        assert case.points_mm.shape == (1800, 3)
        assert case.target_snr_db == 40.0
        assert 40.0 <= case.realized_snr_db < 40.1
        assert case.template_shape == (30, 60)
        assert case.noise_norm > 0

    assert np.array_equal(cases["lung-conductivity"].truth, cases["subcutaneous-fat"].truth)
    assert np.array_equal(cases["lung-conductivity"].forward, cases["subcutaneous-fat"].forward)
    assert not np.array_equal(
        cases["lung-conductivity"].measurement,
        cases["subcutaneous-fat"].measurement,
    )
    assert not np.array_equal(
        cases["lung-conductivity"].truth,
        cases["lbbb-lung-conductivity"].truth,
    )


def test_review_terms_and_data_notice_do_not_grant_public_reuse():
    root = Path(__file__).parents[1]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    public_docs = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("README.md", "THIRD_PARTY_DATA.md", "docs/simulation_protocol.md")
    )

    assert license_text.startswith("Review-Only Terms\n")
    assert "solely to evaluate the associated anonymous" in license_text
    assert "No permission is granted to redistribute" in license_text
    assert "CC BY 4.0" not in public_docs
    assert "heartchambers_highres" in public_docs


def test_every_frozen_public_method_is_connected():
    root = Path(__file__).parents[1]
    configured = {
        path.stem
        for path in (root / "configs" / "paper").glob("*.toml")
        if "method" in tomllib.loads(path.read_text(encoding="utf-8"))
    }
    assert set(IMPLEMENTED_METHODS) == configured


def test_diffusion_contract_keeps_explicit_parameterizations_but_has_no_legacy_default():
    from ecgi.models.diffusion_baseline.diffusion import resolve_parameterization

    assert resolve_parameterization({"parameterization": "eps"}) == "eps"
    assert resolve_parameterization({"parameterization": "v"}) == "v"
    with pytest.raises(ValueError, match="model_config"):
        resolve_parameterization(None)
    with pytest.raises(ValueError, match="parameterization"):
        resolve_parameterization({})


def test_palette_builder_only_accepts_the_released_bsp_conditioning():
    from ecgi.models.conditional_diffusion.diffusion import build_conditional_diffusion

    with pytest.raises(ValueError, match="released Palette checkpoint"):
        build_conditional_diffusion("unused-conditioning")


def test_likelihood_artifacts_are_minimal_and_path_free():
    root = Path(__file__).parents[1]
    expected = {
        "opdis-ps.npz": {
            "conditional_covariance",
            "conditional_regression",
            "signal_mean",
            "mu_d",
            "reduced_basis",
            "conditional_model",
            "conditional_rank",
        },
        "opdis-ps-marginal-isotropic.npz": {"covariance", "mu_d", "reduced_basis"},
        "opdis-ps-marginal.npz": {"covariance", "mu_d", "reduced_basis"},
    }
    configured = {
        tomllib.loads((root / "configs/paper/opdis-ps.toml").read_text())["settings"][
            "conditional_artifact"
        ],
        tomllib.loads((root / "configs/paper/opdis-ps-marginal.toml").read_text())["settings"][
            "discrepancy"
        ],
        tomllib.loads(
            (root / "configs/paper/opdis-ps-marginal-isotropic.toml").read_text()
        )["settings"]["discrepancy"],
    }
    assert configured == set(expected)
    forbidden = ("/home/", "/Users/")
    for case_name in OPERATOR_DISCREPANCIES:
        for filename, keys in expected.items():
            with np.load(root / "artifacts" / case_name / filename, allow_pickle=False) as artifact:
                assert set(artifact.files) == keys
                strings = [
                    str(value)
                    for key in artifact.files
                    if artifact[key].dtype.kind in "US"
                    for value in artifact[key].reshape(-1)
                ]
            assert not any(token in value for token in forbidden for value in strings)

        with np.load(
            root / "artifacts" / case_name / "opdis-ps-marginal.npz"
        ) as structured, np.load(
            root / "artifacts" / case_name / "opdis-ps-marginal-isotropic.npz"
        ) as isotropic:
            assert np.isclose(
                np.trace(structured["covariance"]), np.trace(isotropic["covariance"])
            )
            off_diagonal = isotropic["covariance"] - np.diag(np.diag(isotropic["covariance"]))
            assert np.count_nonzero(off_diagonal) == 0


def test_published_likelihood_artifacts_load_through_inference_classes():
    root = Path(__file__).parents[1]
    for case_name in OPERATOR_DISCREPANCIES:
        artifact_root = root / "artifacts" / case_name
        marginal = MeasurementWhitening.from_npz(
            artifact_root / "opdis-ps-marginal.npz", ridge=0.0
        )
        isotropic = MeasurementWhitening.from_npz(
            artifact_root / "opdis-ps-marginal-isotropic.npz", ridge=0.0
        )
        conditional = ConditionalApproximationError.from_npz(
            artifact_root / "opdis-ps.npz", ridge=0.0
        )
        assert marginal.n_reduced == isotropic.n_reduced == conditional.n_reduced
        assert np.isfinite(marginal.discrepancy_covariance).all()
        assert np.isfinite(isotropic.discrepancy_covariance).all()
        assert np.isfinite(conditional.regression).all()


def test_three_cpu_methods_create_combined_csv_and_html(tmp_path):
    case_path = tmp_path / "case.npz"
    _example(case_path)
    output = tmp_path / "outputs"
    config_root = Path(__file__).parents[1] / "configs" / "paper"

    for method in ("tikhonov", "tv", "st-tikhonov"):
        record = run_method(method, case_path, output, config_root)
        assert record["method"] == method
        assert (output / method / "reconstruction.npz").is_file()
        assert json.loads((output / method / "metrics.json").read_text())["case_id"] == "example-001"
        if method == "tv":
            assert record["alpha"] == 0.01
            assert record["converged"] is True
            assert record["iterations"] > 0

    csv_path, html_path = refresh_results_table(output, open_browser=False)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "Method",
        "RE ↓",
        "ATE [ms] ↓",
        "LE [mm] ↓",
        "sCC ↑",
        "tCC ↑",
    ]
    assert [row["Method"] for row in rows] == [
        METHOD_LABELS["tikhonov"],
        METHOD_LABELS["st-tikhonov"],
        METHOD_LABELS["tv"],
    ]
    assert "ECGI example results" in html_path.read_text()
    assert METHOD_LABELS["st-tikhonov"] in html_path.read_text()
    assert METHOD_LABELS["tv"] in html_path.read_text()


def test_results_table_opens_in_browser_when_enabled(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    method_output = output / "tikhonov"
    method_output.mkdir(parents=True)
    (method_output / "metrics.json").write_text(
        json.dumps({"method": "tikhonov", "case_id": "lung-conductivity-001"})
    )
    opened = []
    monkeypatch.setattr("opdis_ps.results.webbrowser.open", opened.append)

    _, html_path = refresh_results_table(output, open_browser=True)

    assert opened == [html_path.resolve().as_uri()]


def test_reference_results_cover_fast_methods_and_public_examples():
    path = Path(__file__).parents[1] / "docs" / "reference_results.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == [
        "Case",
        "Method",
        "RE ↓",
        "ATE [ms] ↓",
        "LE [mm] ↓",
        "sCC ↑",
        "tCC ↑",
    ]
    assert {(row["Case"], row["Method"]) for row in rows} == {
        (case, method)
        for case in PUBLIC_EXAMPLES
        for method in (METHOD_LABELS["tikhonov"], METHOD_LABELS["transformer"])
    }
