import dataclasses

import params_ranges as pr

from transmon_sim import DeviceParams

FIELDS = [f.name for f in dataclasses.fields(DeviceParams)]


def test_groups_cover_every_field_exactly_once():
    grouped = [n for names in pr.GROUPS.values() for n in names]
    assert sorted(grouped) == sorted(FIELDS)
    assert len(grouped) == len(set(grouped))


def test_published_notes_name_real_fields_and_a_source():
    assert set(pr.PUBLISHED) <= set(FIELDS)
    assert all(note and source for note, source in pr.PUBLISHED.values())


def test_published_preset_is_a_valid_device():
    assert set(pr.PUBLISHED_PRESET) <= set(FIELDS)
    p = DeviceParams(**pr.PUBLISHED_PRESET)
    assert p != DeviceParams()
