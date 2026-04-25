from scripts.export_eval_gallery import build_gallery_payload


def test_build_gallery_payload_normalizes_paths() -> None:
    payload = build_gallery_payload(
        [
            {
                "id": "sample",
                "category": "bottle",
                "status": "success",
                "overlay_path": "artifacts/examples/sample_overlay.png",
            }
        ]
    )

    assert payload["items"][0]["image"] == "/artifacts/examples/sample_overlay.png"
