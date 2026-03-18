import hashlib
import io
from datetime import datetime, timezone
from typing import Dict

import numpy as np
from PIL import Image

from schemas.metadata import WatermarkMetadata, metadata_to_json


def random_bits_64() -> str:
    rng = np.random.default_rng()
    return "".join(str(int(bit)) for bit in rng.integers(0, 2, size=64))


def _fingerprint_png_sha256(image_rgb_uint8: np.ndarray) -> str:
    image = Image.fromarray(image_rgb_uint8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    return hashlib.sha256(content).hexdigest()


def run_embed_pipeline(image_rgb_uint8, editguard_bits, stegastamp_secret, editguard_adapter, stegastamp_adapter, stegastamp_model_dir) -> Dict[str, object]:
    stega_result = stegastamp_adapter.encode(
        image_rgb_uint8,
        stegastamp_secret,
        model_dir=stegastamp_model_dir,
    )
    stegastamp_embedded = stega_result["hidden"]
    final_watermarked = editguard_adapter.embed(stegastamp_embedded, editguard_bits)

    metadata = WatermarkMetadata(
        version="dual-watermark-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        editguard_bits_expected=editguard_bits,
        stegastamp_secret_expected=stegastamp_secret,
        final_image_sha256=_fingerprint_png_sha256(final_watermarked),
    )

    return {
        "stegastamp_image": stegastamp_embedded,
        "editguard_image": final_watermarked,
        "final_image": final_watermarked,
        "stegastamp_residual": stega_result["residual"],
        "metadata_json": metadata_to_json(metadata),
    }
