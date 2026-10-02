"""长表面板数据容器；任务配置、执行和产物收集归 pipeline。"""
import pandas as pd


class SeriesPanel:
    """ID 统一为字符串，拥有隔离副本，不承诺 pandas 零拷贝视图。"""

    def __init__(self, source: pd.DataFrame, id_col: str):
        if id_col not in source or source.empty or source[id_col].isna().any():
            raise ValueError("panel requires non-empty, non-null series identifiers")
        identifiers = source[id_col].astype(str)
        if identifiers.str.strip().eq("").any():
            raise ValueError("panel requires non-blank series identifiers")
        if source[id_col].nunique() != identifiers.nunique():
            raise ValueError("panel identifiers collide after string conversion")
        self.source = source.copy(deep=True)
        self.source[id_col] = identifiers
        self.id_col = id_col
        self._groups = {str(key): group for key, group in self.source.groupby(id_col, sort=True)}

    @property
    def series_ids(self) -> list[str]:
        """字符串 ID 的字典序，与 CSV 入口的字符串读取一致。"""
        return list(self._groups)

    def series_frame(self, series_id: str) -> pd.DataFrame:
        """返回去掉 ID 列的隔离单序列表，保留原行序。"""
        return self._groups[series_id].drop(columns=[self.id_col]).copy(deep=True)

    def __len__(self) -> int:
        return len(self._groups)
