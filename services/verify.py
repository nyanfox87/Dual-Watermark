import hashlib
import io
from typing import Dict

import numpy as np
from PIL import Image

from schemas.metadata import parse_metadata_json


def _fingerprint_png_sha256(image_rgb_uint8: np.ndarray) -> str:
    image = Image.fromarray(image_rgb_uint8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def run_verify_pipeline(image_rgb_uint8, metadata_json, editguard_adapter, stegastamp_adapter, detector_model_dir, stegastamp_model_dir) -> Dict[str, object]:
    metadata = parse_metadata_json(metadata_json)

    edit_result = editguard_adapter.reveal(image_rgb_uint8, metadata.editguard_bits_expected)

    decoded = stegastamp_adapter.decode(image_rgb_uint8, model_dir=stegastamp_model_dir)
    decoded_text = (decoded.get("decoded_text") or "").strip()
    detected_texts = [decoded_text] if decoded_text else []

    detect_result = {
        "annotated": None,
        "mask": None,
        "found_codes": detected_texts,
    }

    stega_mode = "decode-only-pytorch"
    fallback_note = "StegaStamp verification runs in decode-only mode via stegastamp-pytorch."

    copyright_match = metadata.stegastamp_secret_expected in detected_texts
    fingerprint_match = _fingerprint_png_sha256(image_rgb_uint8) == metadata.final_image_sha256

    summary = {
        "stegastamp_mode": stega_mode,
        "note": fallback_note,
        "editguard_intact": bool(edit_result["bits_match"]),
        "copyright_match": bool(copyright_match),
        "fingerprint_match": bool(fingerprint_match),
        "overall_pass": bool(edit_result["bits_match"] and copyright_match),
        "detected_stegastamp_texts": detected_texts,
        "editguard_accuracy": edit_result["accuracy"],
    }

    return {
        "editguard_mask": edit_result["mask"],
        "editguard_recovered_bits": edit_result["recovered_bits"],
        "editguard_accuracy": edit_result["accuracy"],
        "stegastamp_annotated": detect_result["annotated"],
        "stegastamp_mask": detect_result["mask"],
        "stegastamp_found_codes": detect_result["found_codes"],
        "summary": summary,
    }
