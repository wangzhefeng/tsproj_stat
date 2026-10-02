"""面板 ID 与切片契约，不涉及批任务编排。"""
import pandas as pd
import pytest
from data_provider.panel import SeriesPanel


@pytest.mark.parametrize("ids", [[2, 1, 2], ["b", "a", "b"]])
def test_panel_normalizes_ids_and_isolates_frames(ids):
    source = pd.DataFrame({"id": ids, "y": [2., 1., 3.]})
    panel = SeriesPanel(source, "id")
    key = str(ids[0])
    assert panel.series_frame(key)["y"].tolist() == [2., 3.]
    source.loc[0, "y"] = 99.
    sliced = panel.series_frame(key)
    sliced.loc[0, "y"] = 100.
    assert panel.series_frame(key)["y"].tolist() == [2., 3.]


@pytest.mark.parametrize("ids", [[], [None], [""], ["  "], [1, "1"]])
def test_panel_rejects_empty_or_ambiguous_identifiers(ids):
    with pytest.raises(ValueError, match="identifiers"):
        SeriesPanel(pd.DataFrame({"id": ids, "y": [1.] * len(ids)}), "id")
