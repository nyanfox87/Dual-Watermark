import argparse
import base64
import json
from pathlib import Path

import requests


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def decode_base64_to_file(data: str, out_path: Path):
    out_path.write_bytes(base64.b64decode(data))


def post_json(base_url: str, endpoint: str, payload: dict) -> dict:
    response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Dual-WaterMark FastAPI client example")
    parser.add_argument("--base-url", default="http://127.0.0.1:7867", help="API base url")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--mask",
        default="",
        help="Optional mask image path for /inpaint (white region will be inpainted)",
    )
    parser.add_argument("--bits", default="0101010101010101010101010101010101010101010101010101010101010101")
    parser.add_argument("--secret", default="Stega!")
    parser.add_argument("--prompt", default="repair tampered region")
    parser.add_argument("--out-dir", default="./output_api_client")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    image_path = Path(args.image)
    mask_path = Path(args.mask) if args.mask else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print("[1/3] Calling /embed ...")
    embed_payload = {
        "image_base64": encode_file_base64(image_path),
        "editguard_bits": args.bits,
        "stegastamp_secret": args.secret,
    }
    embed_resp = post_json(base_url, "/embed", embed_payload)

    metadata_json = embed_resp["metadata_json"]
    (out_dir / "metadata.json").write_text(metadata_json, encoding="utf-8")
    decode_base64_to_file(embed_resp["stegastamp_image_base64"], out_dir / "stegastamp_stage.png")
    decode_base64_to_file(embed_resp["final_image_base64"], out_dir / "final_watermarked.png")
    decode_base64_to_file(embed_resp["stegastamp_residual_base64"], out_dir / "stegastamp_residual.png")

    image_for_verify_b64 = embed_resp["final_image_base64"]

    if mask_path and mask_path.is_file():
        print("[2/3] Calling /inpaint ...")
        inpaint_payload = {
            "image_base64": image_for_verify_b64,
            "mask_base64": encode_file_base64(mask_path),
            "prompt": args.prompt,
        }
        inpaint_resp = post_json(base_url, "/inpaint", inpaint_payload)
        image_for_verify_b64 = inpaint_resp["inpainted_image_base64"]
        decode_base64_to_file(image_for_verify_b64, out_dir / "inpainted.png")
    else:
        print("[2/3] Skip /inpaint (no --mask provided)")

    print("[3/3] Calling /verify ...")
    verify_payload = {
        "image_base64": image_for_verify_b64,
        "metadata_json": metadata_json,
    }
    verify_resp = post_json(base_url, "/verify", verify_payload)

    decode_base64_to_file(verify_resp["editguard_mask_base64"], out_dir / "verify_mask.png")
    (out_dir / "verify_response.json").write_text(json.dumps(verify_resp, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Stega found codes: {verify_resp.get('stegastamp_found_codes', [])}")
    print(f"EditGuard accuracy: {verify_resp.get('editguard_accuracy')}")
    print(f"Overall pass: {verify_resp.get('summary', {}).get('overall_pass')}")


if __name__ == "__main__":
    main()
