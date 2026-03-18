import json
from dataclasses import asdict, dataclass


@dataclass
class WatermarkMetadata:
    version: str
    created_at: str
    editguard_bits_expected: str
    stegastamp_secret_expected: str
    final_image_sha256: str


def metadata_to_json(metadata: WatermarkMetadata) -> str:
    return json.dumps(asdict(metadata), ensure_ascii=False, indent=2)


def parse_metadata_json(raw: str) -> WatermarkMetadata:
    if not raw or not raw.strip():
        raise ValueError("Metadata JSON is empty.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON: {exc}") from exc

    required = [
        "version",
        "created_at",
        "editguard_bits_expected",
        "stegastamp_secret_expected",
        "final_image_sha256",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Metadata missing required fields: {', '.join(missing)}")

    edit_bits = str(payload["editguard_bits_expected"]).strip()
    if len(edit_bits) != 64 or any(char not in {"0", "1"} for char in edit_bits):
        raise ValueError("metadata.editguard_bits_expected must be 64 bits of 0/1.")

    return WatermarkMetadata(
        version=str(payload["version"]),
        created_at=str(payload["created_at"]),
        editguard_bits_expected=edit_bits,
        stegastamp_secret_expected=str(payload["stegastamp_secret_expected"]),
        final_image_sha256=str(payload["final_image_sha256"]),
    )
