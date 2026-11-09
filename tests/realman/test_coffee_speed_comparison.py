from tools.realman.compare_coffee_service_speed_profiles_v2 import (
    summarize_reference_factor_attainment,
    summarize_speed_targets,
)


def test_reference_factor_attainment_counts_segments_by_factor():
    report = summarize_reference_factor_attainment({
        "free_transport": {
            "matched": [
                {"candidate_reference_factor": 4, "observed_at_least_reference_factor": True},
                {"candidate_reference_factor": 4, "observed_at_least_reference_factor": False},
            ]
        },
        "contact": {
            "matched": [
                {"candidate_reference_factor": 2, "observed_at_least_reference_factor": True},
                {"candidate_reference_factor": 2, "observed_at_least_reference_factor": None},
                {"candidate_reference_factor": None, "observed_at_least_reference_factor": None},
            ]
        },
        "unmatched": {"matched": []},
    })

    assert report == {
        "4": {"segments": 2, "at_least_reference_factor": 1},
        "2": {"segments": 2, "at_least_reference_factor": 1},
    }


def test_speed_target_summary_separates_met_below_and_unmatched_segments():
    report = summarize_speed_targets(
        {
            "free_transfer": {
                "matched": [
                    {"occurrence": 0, "candidate_reference_factor": 4, "observed_at_least_reference_factor": True},
                    {"occurrence": 1, "candidate_reference_factor": 4, "observed_at_least_reference_factor": False},
                ]
            },
            "contact_transfer": {
                "matched": [
                    {"occurrence": 0, "candidate_reference_factor": 2, "observed_at_least_reference_factor": None},
                ]
            },
        },
        {
            "free_transfer": [{"factor": 4}, {"factor": 4}],
            "contact_transfer": [{"factor": 2}, {"factor": 2}],
        },
    )

    assert report["targets"]["4"] == {
        "target_speedup": 4,
        "expected_segments": 2,
        "measured_segments": 2,
        "segments_at_or_above_target": 1,
        "segments_below_target": 1,
        "segments_unverified": 0,
        "unmatched_segments": 0,
        "all_segments_met": False,
        "below_target": [
            {
                "stage": "free_transfer",
                "occurrence": 1,
                "candidate_reference_factor": 4,
                "observed_at_least_reference_factor": False,
            }
        ],
        "unverified": [],
    }
    assert report["targets"]["2"]["measured_segments"] == 1
    assert report["targets"]["2"]["segments_unverified"] == 1
    assert report["targets"]["2"]["unmatched_segments"] == 1
    assert report["targets"]["2"]["all_segments_met"] is False
    assert report["all_measured_targets_met"] is False
