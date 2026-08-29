import io

import numpy as np

import cloudinary_storage


class FakeUploader:
    def __init__(self):
        self.calls = []

    def upload(self, file_object, **options):
        assert isinstance(file_object, io.BytesIO)
        assert file_object.getbuffer().nbytes > 0
        self.calls.append(options)
        resource_type = options.get("resource_type", "image")
        return {
            "secure_url": f"https://example.test/{options['public_id']}",
            "public_id": f"{options['folder']}/{options['public_id']}",
            "resource_type": resource_type,
            "format": options.get("format", "png"),
            "bytes": file_object.getbuffer().nbytes,
        }


def test_cloudinary_is_optional(monkeypatch):
    for name in (
        "CLOUDINARY_URL",
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    assert cloudinary_storage.cloudinary_enabled() is False

    monkeypatch.setenv("CLOUDINARY_URL", "cloudinary://key:secret@example")
    assert cloudinary_storage.cloudinary_enabled() is True


def test_uploads_all_scan_assets_and_evaluation_json(monkeypatch):
    uploader = FakeUploader()
    monkeypatch.setattr(cloudinary_storage, "_load_cloudinary", lambda: uploader)
    monkeypatch.setenv("CLOUDINARY_OMR_FOLDER", "school/omr")
    image = np.full((80, 60, 3), 255, dtype=np.uint8)

    images = cloudinary_storage.upload_scan_images(
        scan_id="scan-123",
        original_bytes=b"original-image",
        corrected_image=image,
        evaluated_image=image,
    )
    evaluation = cloudinary_storage.upload_evaluation_json(
        scan_id="scan-123",
        evaluation={"score": 42, "series": "P"},
    )

    assert set(images) == {"original", "corrected", "evaluated"}
    assert images["original"]["url"] == "https://example.test/scan-123"
    assert evaluation["resource_type"] == "raw"
    assert evaluation["public_id"].endswith("/scan-123.json")
    assert [call["folder"] for call in uploader.calls] == [
        "school/omr/originals",
        "school/omr/corrected",
        "school/omr/evaluated",
        "school/omr/evaluations",
    ]
