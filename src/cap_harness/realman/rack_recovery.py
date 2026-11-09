"""Select cups eligible for a bounded pause; never certify recovered stability."""

from collections.abc import Mapping


def rack_settle_members(lost, retained, inventory):
    """Return every cup to monitor, or None when settling cannot explain loss.

    Both rack predicates can report the same velocity transient. Recovery must
    retain a fresh contact path for every affected cup, including the selected
    bottom cup when it is still inventory. The caller keeps the existing 0.5 s
    budget, checks position/tilt and support during the pause, and rechecks all
    original predicates before resuming.
    """
    rack_keys = {"remaining_cups_retained", "rack_inventory_supported"}
    lost = set(lost)
    if not lost & rack_keys or not lost <= rack_keys | {"single_cup_separated"}:
        return None
    groups = []
    if "remaining_cups_retained" in lost:
        groups.append(retained)
    if "rack_inventory_supported" in lost:
        groups.append(inventory)
    members = set()
    for group in groups:
        if not isinstance(group, Mapping) or not group:
            return None
        for name, state in group.items():
            if (
                not isinstance(name, str)
                or not name.startswith("cup_")
                or not isinstance(state, Mapping)
                or state.get("anchored_now") is not True
            ):
                return None
            members.add(name)
    if len(groups) == 2 and not set(retained) <= set(inventory):
        return None
    return tuple(sorted(members))
