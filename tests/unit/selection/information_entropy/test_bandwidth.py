"""Pool weighting, frozen bandwidth, and calibration reference tests."""

import logging

import numpy as np
import pytest

from nepflow.stages.selection.algorithms.information_entropy.bandwidth import (
    BandwidthCalibrationError,
    build_entropy_pool,
    calculate_frozen_bandwidths,
    calibrate_bandwidth,
)
from nepflow.stages.selection.algorithms.information_entropy.models import (
    EntropyBandwidthSettings,
    EntropyPool,
)


def _pool() -> EntropyPool:
    descriptors = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    return build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "a", "b", "c", "c"],
        candidate_ids=["a", "b", "c"],
    )


def test_candidate_weights_are_equal_by_candidate_then_by_row() -> None:
    pool = _pool()

    np.testing.assert_allclose(
        pool.probabilities,
        [1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    )
    assert float(np.sum(pool.probabilities)) == pytest.approx(1.0)
    assert pool.row_candidate_indices.tolist() == [0, 0, 1, 2, 2]


def test_frozen_bandwidths_are_read_only_and_chunk_independent() -> None:
    pool = _pool()
    first = calculate_frozen_bandwidths(pool, 1, 2.0, chunk_size=1)
    second = calculate_frozen_bandwidths(pool, 1, 2.0, chunk_size=64)

    np.testing.assert_array_equal(first.radii, second.radii)
    np.testing.assert_array_equal(first.bandwidths, second.bandwidths)
    assert first.fingerprint == second.fingerprint
    assert first.radii.flags.writeable is False
    assert first.bandwidths.flags.writeable is False
    with pytest.raises(ValueError):
        first.bandwidths[0] = 99.0


def test_manual_and_single_pair_automatic_calibration_match() -> None:
    pool = _pool()
    manual = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(mode="manual", k=1, c=2.0),
    )
    automatic = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(mode="automatic", k_candidates=(1,), c_candidates=(2.0,)),
    )

    assert manual.selected.k == automatic.selected.k == 1
    assert manual.selected.c == automatic.selected.c == 2.0
    np.testing.assert_array_equal(manual.selected.bandwidths, automatic.selected.bandwidths)
    assert manual.objective == pytest.approx(automatic.objective)
    assert [attempt.status for attempt in automatic.attempts] == ["valid"]


def test_zero_leave_one_out_support_is_an_invalid_pair() -> None:
    descriptors = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "b", "c"],
        candidate_ids=["a", "b", "c"],
    )

    with pytest.raises(BandwidthCalibrationError, match="no valid") as error:
        calibrate_bandwidth(
            pool,
            EntropyBandwidthSettings(mode="automatic", k_candidates=(1,), c_candidates=(1.0,)),
        )
    assert error.value.attempts[0].status == "invalid"
    assert "support" in (error.value.attempts[0].reason or "")


def test_calibration_logs_every_pair_and_final_counts(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(
        logging.INFO,
        logger="nepflow.stages.selection.algorithms.information_entropy.bandwidth",
    )
    pool = _pool()

    result = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(
            mode="automatic",
            k_candidates=(1, 2),
            c_candidates=(2.0, 4.0),
        ),
    )

    pair_messages = [
        record.getMessage()
        for record in caplog.records
        if "Bandwidth calibration pair completed:" in record.getMessage()
    ]
    assert len(pair_messages) == 4
    assert all(
        f"{index}/4 combinations" in message for index, message in enumerate(pair_messages, 1)
    )
    assert result.evaluation_count == 4
    assert "Bandwidth calibration started: mode=automatic, ordered combinations=4" in caplog.text
    assert "Bandwidth calibration completed: selected k=" in caplog.text


def test_rejected_k_is_logged_as_one_invalid_pair_per_scale(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="nepflow.stages.selection.algorithms.information_entropy.bandwidth",
    )
    pool = _pool()

    result = calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(
            mode="automatic",
            k_candidates=(1, 8),
            c_candidates=(2.0, 4.0),
        ),
    )

    invalid_messages = [
        record.getMessage()
        for record in caplog.records
        if "Bandwidth calibration pair completed:" in record.getMessage()
        and "J(k,c)=" not in record.getMessage()
    ]
    assert len(invalid_messages) == 2
    assert result.evaluation_count == 4
    assert [attempt.status for attempt in result.attempts] == [
        "valid",
        "valid",
        "invalid",
        "invalid",
    ]
    assert "4/4 combinations" in caplog.text


def test_all_invalid_calibration_logs_failure_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(
        logging.INFO,
        logger="nepflow.stages.selection.algorithms.information_entropy.bandwidth",
    )
    descriptors = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=["a", "b", "c"],
        candidate_ids=["a", "b", "c"],
    )

    with pytest.raises(BandwidthCalibrationError):
        calibrate_bandwidth(
            pool,
            EntropyBandwidthSettings(mode="automatic", k_candidates=(1,), c_candidates=(1.0,)),
        )

    assert "Bandwidth calibration pair completed:" in caplog.text
    assert "Bandwidth calibration failed: no valid pair" in caplog.text
    assert "Bandwidth calibration completed:" not in caplog.text


def test_long_pair_logs_source_progress_at_work_thresholds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.INFO,
        logger="nepflow.stages.selection.algorithms.information_entropy.bandwidth",
    )
    descriptors = np.arange(21, dtype=np.float64).reshape(-1, 1)
    identifiers = [str(index) for index in range(len(descriptors))]
    pool = build_entropy_pool(
        descriptors,
        row_candidate_ids=identifiers,
        candidate_ids=identifiers,
    )

    calibrate_bandwidth(
        pool,
        EntropyBandwidthSettings(mode="automatic", k_candidates=(1,), c_candidates=(2.0,)),
    )

    source_messages = [
        record.getMessage()
        for record in caplog.records
        if "Bandwidth calibration pair source progress:" in record.getMessage()
    ]
    completed = [
        int(message.split("source=", 1)[1].split("/", 1)[0]) for message in source_messages
    ]
    assert completed == sorted(set(completed))
    assert completed[-1] == len(descriptors)
    assert all(
        "k=1, c=2" in message and "estimated remaining=" in message for message in source_messages
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "manual", "k": 0, "c": 1.0},
        {"mode": "manual", "k": 1, "c": 0.0},
        {"mode": "manual", "k": 1, "c": np.nan},
        {"mode": "manual", "k": 1, "c": np.inf},
    ],
)
def test_bad_manual_bandwidth_inputs_fail(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EntropyBandwidthSettings(**kwargs)  # type: ignore[arg-type]
