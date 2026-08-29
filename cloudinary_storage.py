"""Optional Cloudinary persistence for OMR scan artifacts."""

from __future__ import annotations

import io
import json
import os
from typing import Any

import cv2


def cloudinary_enabled() -> bool:
    """Return true when either supported Cloudinary credential form exists."""
    if os.environ.get("CLOUDINARY_URL"):
        return True

    return all(
        os.environ.get(name)
        for name in (
            "CLOUDINARY_CLOUD_NAME",
            "CLOUDINARY_API_KEY",
            "CLOUDINARY_API_SECRET",
        )
    )


def _load_cloudinary():
    try:
        import cloudinary
        import cloudinary.uploader
    except ImportError as error:
        raise RuntimeError(
            "Cloudinary credentials are configured, but the cloudinary "
            "Python package is not installed."
        ) from error

    config = {"secure": True}
    if not os.environ.get("CLOUDINARY_URL"):
        config.update(
            {
                "cloud_name": os.environ["CLOUDINARY_CLOUD_NAME"],
                "api_key": os.environ["CLOUDINARY_API_KEY"],
                "api_secret": os.environ["CLOUDINARY_API_SECRET"],
            }
        )
    cloudinary.config(**config)
    return cloudinary.uploader


def _asset_summary(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": response.get("secure_url"),
        "public_id": response.get("public_id"),
        "resource_type": response.get("resource_type"),
        "format": response.get("format"),
        "bytes": response.get("bytes"),
    }


def _encode_jpeg(image) -> io.BytesIO:
    if image is None:
        raise ValueError("OMR image is unavailable for Cloudinary upload.")

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise ValueError("Could not encode OMR image for Cloudinary upload.")

    return io.BytesIO(encoded.tobytes())


def upload_scan_images(
    *,
    scan_id: str,
    original_bytes: bytes,
    corrected_image,
    evaluated_image,
) -> dict[str, dict[str, Any]]:
    """Upload the original, corrected, and evaluated/debug OMR images."""
    uploader = _load_cloudinary()
    root_folder = os.environ.get("CLOUDINARY_OMR_FOLDER", "omr-scanner").strip("/")

    uploads = {
        "original": uploader.upload(
            io.BytesIO(original_bytes),
            folder=f"{root_folder}/originals",
            public_id=scan_id,
            resource_type="image",
            overwrite=True,
        ),
        "corrected": uploader.upload(
            _encode_jpeg(corrected_image),
            folder=f"{root_folder}/corrected",
            public_id=scan_id,
            resource_type="image",
            overwrite=True,
            format="jpg",
        ),
        "evaluated": uploader.upload(
            _encode_jpeg(evaluated_image),
            folder=f"{root_folder}/evaluated",
            public_id=scan_id,
            resource_type="image",
            overwrite=True,
            format="jpg",
        ),
    }
    return {name: _asset_summary(response) for name, response in uploads.items()}


def upload_evaluation_json(
    *,
    scan_id: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Upload the serializable evaluation result as a raw JSON asset."""
    uploader = _load_cloudinary()
    root_folder = os.environ.get("CLOUDINARY_OMR_FOLDER", "omr-scanner").strip("/")
    payload = json.dumps(
        evaluation,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    response = uploader.upload(
        io.BytesIO(payload),
        folder=f"{root_folder}/evaluations",
        public_id=f"{scan_id}.json",
        resource_type="raw",
        overwrite=True,
    )
    return _asset_summary(response)
