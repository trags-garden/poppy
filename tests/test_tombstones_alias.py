import poppy.tombstones
import poppy.ui.tombstones


def test_ui_tombstones_is_canonical_module() -> None:
    assert poppy.ui.tombstones is poppy.tombstones
