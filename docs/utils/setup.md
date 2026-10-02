# 安装与依赖

## 环境

```bash
uv venv .venv --python 3.12
uv sync --extra dev
```

- Python 基线 `3.12`（以 `.python-version` 与当前开发环境为准）
- 统一使用项目根目录 `.venv`；不使用系统 Python、conda 或其他分散虚拟环境

## 依赖管理

依赖来源：`pyproject.toml` + `uv.lock`。

```bash
uv add <package>          # 新增/更新依赖（不用 pip install）
uv add --dev <package>
uv sync --extra dev       # 环境同步
```

- 运行、测试、脚本一律直接调用 `.venv/bin/python`，不经过 `uv run`，不设置 `UV_CACHE_DIR`
- MC/Hermes 会话需加 `env -u PYTHONPATH` 前缀；普通 shell 可省略
- 若 `.venv` 与 `pyproject.toml` / `uv.lock` 不一致，重新执行 `uv sync --extra dev`

## 缓存纪律

- `matplotlib` 默认使用用户级缓存目录（`~/.matplotlib`）；仅在该目录不可写时回退系统临时目录（`utils/runtime_env.py`）
- 仓库根目录不得产生 `.uv_cache` / `.pytest_cache` / `.mplconfig` 缓存目录；pytest 经 `pyproject.toml` addopts `-p no:cacheprovider` 从源头禁用缓存

## 依赖锁定

- StatsForecast 锁定 2.0.1、pandas 锁定 3.0.1；上游新版本声明 pandas<3，不直接升级。实际解析版本以项目 `uv.lock` 为准。
