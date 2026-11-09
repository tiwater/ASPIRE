import pytest

from cap_harness.realman.coffee_speed import pot_transport_speed_factor


@pytest.mark.parametrize("support_n", [0.0, 0.01, 0.02])
def test_pot_uses_free_factor_only_when_clear_and_unsupported(support_n):
    assert pot_transport_speed_factor(
        4,
        2,
        independently_clear=True,
        upward_support_n=support_n,
    ) == 4


@pytest.mark.parametrize(
    ("clear", "support_n"),
    [(False, 0.0), (True, 0.020001), (False, 5.0)],
)
def test_pot_retains_contact_factor_until_clear_and_unsupported(clear, support_n):
    assert pot_transport_speed_factor(
        4,
        2,
        independently_clear=clear,
        upward_support_n=support_n,
    ) == 2


@pytest.mark.parametrize(
    ("clear", "expected"),
    [(True, 1), (False, 1)],
)
def test_minimum_speed_factors_are_valid(clear, expected):
    assert pot_transport_speed_factor(
        1,
        1,
        independently_clear=clear,
        upward_support_n=0,
    ) == expected


@pytest.mark.parametrize(
    ("free", "contact", "clear", "support", "tolerance"),
    [
        (0, 2, True, 0, 0.02),
        (5, 2, True, 0, 0.02),
        (4, 2.1, True, 0, 0.02),
        (0.99, 2, True, 0, 0.02),
        (4, 0.99, True, 0, 0.02),
        (True, 2, True, 0, 0.02),
        (4, False, True, 0, 0.02),
        (4, 2, True, False, 0.02),
        ("4", 2, True, 0, 0.02),
        (4, 2, True, "0", 0.02),
        (4, float("inf"), True, 0, 0.02),
        (4, 2, True, -0.1, 0.02),
        (4, 2, True, float("nan"), 0.02),
        (4, 2, True, 0, -0.01),
        (4, 2, 1, 0, 0.02),
    ],
)
def test_invalid_speed_policy_inputs_are_rejected(free, contact, clear, support, tolerance):
    with pytest.raises(ValueError):
        pot_transport_speed_factor(
            free,
            contact,
            independently_clear=clear,
            upward_support_n=support,
            support_tolerance_n=tolerance,
        )
