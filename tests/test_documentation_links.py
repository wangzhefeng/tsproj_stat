"""模块文档迁移的持久守卫：目录、链接及运行时报告引用。"""
from pathlib import Path
import re

from eda.report_generator import generate_eda_report

ROOT = Path(__file__).resolve().parents[1]


def test_generated_report_points_to_existing_module_guide(tmp_path):
    (tmp_path / "eda_summary.json").write_text('{"n_samples":20}', encoding="utf-8")
    report = generate_eda_report(tmp_path)
    assert report is not None
    text = Path(report).read_text(encoding="utf-8")
    guides = re.findall(r"`(docs/[^`]+\.md)`", text)
    assert guides == ["docs/eda/eda_report_guide.md"]
    assert all((ROOT / path).is_file() for path in guides)


def test_module_document_layout_and_relative_links():
    docs = ROOT / "docs"
    assert not (docs / "README.md").exists(), "根 README 是唯一总目录"
    pages = {ROOT / "README.md", *docs.rglob("*.md")}
    edges: dict[Path, set[Path]] = {}
    for path in pages:
        text = path.read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 60, f"需拆成小章节: {path}"
        if path != ROOT / "README.md":
            assert (ROOT / path.relative_to(docs).parts[0]).is_dir()
        edges[path] = set()
        for link in re.findall(r"\]\(([^)]+)\)", text):
            target = link.split("#", 1)[0]
            if not target or re.match(r"^[A-Za-z][\w+.-]*:", target):
                continue
            destination = (path.parent / target).resolve()
            assert destination.is_file(), (path, link)
            assert '.hermes' not in destination.relative_to(ROOT).parts, (path, link)
            if destination in pages:
                edges[path].add(destination)
    reached, pending = set(), [ROOT / "README.md"]
    while pending:
        page = pending.pop()
        if page not in reached:
            reached.add(page)
            pending.extend(edges[page] - reached)
    assert reached == pages, f"主目录无法到达的文档: {pages - reached}"
