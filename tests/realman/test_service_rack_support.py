from cap_harness.realman.service_evaluation import rack_support_paths


def graph(*pairs):
    edges = {}
    for a, b in pairs:
        edges.setdefault(a, set()).add(b)
        edges.setdefault(b, set()).add(a)
    return edges


def test_intact_stack_can_transmit_support_through_bottom_cup():
    contacts = graph(("cup_2", "cup_1"), ("cup_1", "cup_0"), ("cup_0", "retention_finger_0"))
    assert all(rack_support_paths(contacts, ["cup_0", "cup_1", "cup_2"]).values())
    assert not any(rack_support_paths(contacts, ["cup_1", "cup_2"]).values())
    contacts = graph(("cup_2", "cup_1"), ("cup_1", "retention_finger_1"))
    assert all(rack_support_paths(contacts, ["cup_1", "cup_2"]).values())


def test_robot_table_and_already_served_cup_cannot_anchor_remaining_stack():
    for bridge in ["rtool_Base_Link", "table", "cup_0"]:
        contacts = graph(("cup_2", "cup_1"), ("cup_1", bridge), (bridge, "cup_dispenser"))
        assert not any(rack_support_paths(contacts, ["cup_1", "cup_2"]).values())


def test_one_supported_member_does_not_certify_entire_inventory():
    contacts = graph(("cup_0", "cup_dispenser"), ("cup_1", "cup_2"))
    assert rack_support_paths(contacts, ["cup_0", "cup_1", "cup_2"]) == {
        "cup_0": True,
        "cup_1": False,
        "cup_2": False,
    }
