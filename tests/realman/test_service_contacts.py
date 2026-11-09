import pytest

from cap_harness.realman.service_contacts import prohibited_pot_contacts


def contact(body, geom, force=1.0):
    return dict(body1="pot", body2=body, geom1="pot_bottom", geom2=geom, normal_force_n=force)


@pytest.mark.parametrize("body", ["cup_0", "cup_1", "cup_3", "cup_dispenser", "retention_finger_0"])
def test_service_guard_covers_every_cup_and_rack_with_original_identity(body):
    row = contact(body, body + "_geom")
    assert prohibited_pot_contacts([row]) == [row]
    assert not prohibited_pot_contacts([contact(body, body + "_geom", 0.01)])


def test_machine_support_and_handle_tool_contacts_keep_original_contract():
    allowed = [
        contact("coffee_machine", "hotplate"),
        contact("spring_outlet", "spring_outlet_contact"),
        contact("ltool_Right_Support_Link", "ltool_Right_Support_Link_collision_0"),
    ]
    assert not prohibited_pot_contacts(allowed)
    with pytest.raises(ValueError, match="nonfinite"):
        prohibited_pot_contacts([contact("cup_2", "cup_2_wall", float("nan"))])
