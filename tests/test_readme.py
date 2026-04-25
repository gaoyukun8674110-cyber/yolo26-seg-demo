from pathlib import Path


def test_readme_mentions_api_web_and_scripts() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "scripts/" in content
    assert "api/" in content
    assert "web/" in content
    assert "python -m uvicorn api.app.main:app --reload" in content
    assert "npm --prefix web run dev" in content
    assert "scripts\\convert_mvtec_to_yolo_seg.py" in content
