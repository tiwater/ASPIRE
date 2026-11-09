import itertools

import pytest

from cap_harness.realman.contact_guards import prohibited_pot_contacts


def contact(other, geom, force):
    return dict(body1="pot", body2=other, geom1="lid", geom2=geom, normal_force_n=force)


def test_housing_contact_is_detected_in_every_list_position_and_body_order():
    bad = contact("coffee_machine", "machine_head", 0.1)
    good = [
        contact("coffee_machine", "hotplate", 15),
        contact("spring_outlet", "tip", 5),
        contact("coffee_machine", "machine_platform_ramp_00_001", 2),
    ]
    for records in itertools.permutations([bad, *good]):
        assert prohibited_pot_contacts(records) == [bad]
        reversed_records = [
            dict(
                body1=r["body2"],
                body2=r["body1"],
                geom1=r["geom2"],
                geom2=r["geom1"],
                normal_force_n=r["normal_force_n"],
            )
            for r in records
        ]
        assert len(prohibited_pot_contacts(reversed_records)) == 1


def test_retains_all_violations_and_rejects_invalid_measurements():
    contacts = [contact("cup_0", "rim", 0.3), contact("table", "tabletop", 1)]
    assert prohibited_pot_contacts(contacts) == contacts
    assert not prohibited_pot_contacts([contact("coffee_machine", "head", 0.02)])
    assert not prohibited_pot_contacts([])
    with pytest.raises(ValueError):
        prohibited_pot_contacts([contact("coffee_machine", "head", float("nan"))])
