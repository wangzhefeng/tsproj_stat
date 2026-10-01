"""应用配置到数据聚合服务的适配；通用数据模块不依赖 AppConfig。"""
from config import AppConfig
from data_provider.resampling.service import AggregationResult, aggregate_csv


def resolve_config_aggregation(cfg: AppConfig) -> AggregationResult | None:
    """按 AppConfig 生成派生文件，并把本次有效 data_path 切换到派生文件。"""
    if not cfg.aggregation_enabled:
        return None
    if cfg.data_path is None or cfg.aggregation_source_freq is None:
        raise ValueError("aggregation requires data_path and aggregation_source_freq")
    result = aggregate_csv(
        source_path=cfg.data_path,
        time_col=cfg.time_col,
        target_col=cfg.target_col,
        source_freq=cfg.aggregation_source_freq,
        target_freq=cfg.freq,
        method=cfg.aggregation_method,
        fill_method=cfg.aggregation_fill_method,
        fill_weeks=cfg.aggregation_fill_weeks,
        output_path=cfg.aggregation_output_path,
    )
    cfg.data_path = str(result.data_path)
    return result
