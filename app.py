import argparse
import json
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from adapters import EditGuardAdapter, StegaStampAdapter
from services.pipeline import random_bits_64, run_embed_pipeline
from services.verify import run_verify_pipeline


DEFAULT_EDITGUARD_ROOT = Path("/home/project/Documents/EditGuard")
DEFAULT_STEGASTAMP_ROOT = Path("/home/project/Documents/StegaStamp-pytorch")
DEFAULT_STEGASTAMP_MODEL_DIR = DEFAULT_STEGASTAMP_ROOT / "asset" / "best.pth"
DEFAULT_DETECTOR_MODEL_DIR = Path("")
DEFAULT_STEGASTAMP_ENV = "unused"
DEFAULT_INPAINT_PROMPT = "repair tampered region"
CUSTOM_PROMPT_OPTION = "Custom..."
INPAINT_PROMPT_TEMPLATES = [
    CUSTOM_PROMPT_OPTION,
    "repair tampered region",
    "restore the missing details naturally",
    "remove the tampered artifact and preserve original texture",
    "recover realistic face details in the masked area",
    "remove unwanted edited area and keep background consistent",
]

_ADAPTER_CACHE = {}


def _read_image_rgb_uint8(image_input) -> np.ndarray:
    if image_input is None:
        raise ValueError("Please upload an image.")
    if isinstance(image_input, np.ndarray):
        image = image_input
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    path = Path(image_input)
    if not path.is_file():
        raise ValueError(f"Image file not found: {path}")
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _get_adapters(editguard_root: str, stegastamp_root: str, stegastamp_env: str):
    key = (str(Path(editguard_root).resolve()), str(Path(stegastamp_root).resolve()), stegastamp_env)
    cached = _ADAPTER_CACHE.get(key)
    if cached is not None:
        return cached

    editguard_adapter = EditGuardAdapter(editguard_root=editguard_root)
    stegastamp_adapter = StegaStampAdapter(stegastamp_root=stegastamp_root, stegastamp_env=stegastamp_env)
    _ADAPTER_CACHE[key] = (editguard_adapter, stegastamp_adapter)
    return editguard_adapter, stegastamp_adapter


def on_random_bits():
    return random_bits_64()


def on_run_embed(
    image_path,
    editguard_bits,
    stegastamp_secret,
    editguard_root,
    stegastamp_root,
    stegastamp_env,
    stegastamp_model_dir,
):
    try:
        image = _read_image_rgb_uint8(image_path)
        editguard_adapter, stegastamp_adapter = _get_adapters(editguard_root, stegastamp_root, stegastamp_env)

        result = run_embed_pipeline(
            image_rgb_uint8=image,
            editguard_bits=editguard_bits,
            stegastamp_secret=stegastamp_secret,
            editguard_adapter=editguard_adapter,
            stegastamp_adapter=stegastamp_adapter,
            stegastamp_model_dir=stegastamp_model_dir,
        )

        status = "Pipeline success: StegaStamp embedded, then EditGuard embedded."
        return (
            result.get("stegastamp_image", result["final_image"]),
            result["final_image"],
            result["stegastamp_residual"],
            result["metadata_json"],
            status,
        )
    except Exception:
        return None, None, None, "", f"Embed pipeline failed:\n\n{traceback.format_exc()}"


def on_run_verify(
    image_path,
    metadata_json,
    editguard_root,
    stegastamp_root,
    stegastamp_env,
    stegastamp_model_dir,
    detector_model_dir,
):
    try:
        image = _read_image_rgb_uint8(image_path)
        editguard_adapter, stegastamp_adapter = _get_adapters(editguard_root, stegastamp_root, stegastamp_env)

        result = run_verify_pipeline(
            image_rgb_uint8=image,
            metadata_json=metadata_json,
            editguard_adapter=editguard_adapter,
            stegastamp_adapter=stegastamp_adapter,
            detector_model_dir=detector_model_dir,
            stegastamp_model_dir=stegastamp_model_dir,
        )

        summary_json = json.dumps(result["summary"], ensure_ascii=False, indent=2)
        found_codes_text = "\n".join(result["stegastamp_found_codes"]) if result["stegastamp_found_codes"] else "No valid code detected"

        return (
            result["editguard_mask"],
            result["editguard_recovered_bits"],
            result["editguard_accuracy"],
            found_codes_text,
            summary_json,
        )
    except Exception:
        return None, "", "", "", f"Verify failed:\n\n{traceback.format_exc()}"


def on_send_to_next_pages(final_image_value, metadata_json):
    if final_image_value is None:
        raise ValueError("Please run page 1 embed first.")
    if not metadata_json or not str(metadata_json).strip():
        raise ValueError("Metadata JSON is empty. Please run page 1 embed first.")

    return final_image_value, metadata_json, final_image_value, metadata_json, "Sent page 1 result to page 2 and page 3."


def on_send_inpaint_to_verify(inpainted_image_value, metadata_json):
    if inpainted_image_value is None:
        raise ValueError("Please run page 2 inpaint first.")
    if not metadata_json or not str(metadata_json).strip():
        raise ValueError("Metadata JSON is empty. Please provide metadata from page 1.")

    return inpainted_image_value, metadata_json, "Sent inpaint output to page 3 verification."


def on_run_inpaint(
    image_input,
    metadata_json,
    prompt,
    editguard_root,
    stegastamp_root,
    stegastamp_env,
):
    try:
        if isinstance(image_input, dict):
            base_image = image_input.get("image")
            user_mask = image_input.get("mask")
        else:
            base_image = image_input
            user_mask = None

        image = _read_image_rgb_uint8(base_image)
        editguard_adapter, _ = _get_adapters(editguard_root, stegastamp_root, stegastamp_env)

        if user_mask is None:
            raise ValueError("Please draw an inpaint region on page 2 image (sketch mask).")

        mask = np.array(user_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if np.max(mask) <= 0:
            raise ValueError("Mask is empty. Please draw a region to inpaint.")

        inpainted = editguard_adapter.inpaint(image=image, prompt=prompt, mask=mask)
        status = "Inpaint done using user-drawn region."
        return mask, inpainted, status
    except Exception:
        return None, None, f"Inpaint failed:\n\n{traceback.format_exc()}"


def on_select_inpaint_prompt(template_value):
    if not template_value:
        return DEFAULT_INPAINT_PROMPT
    if template_value == CUSTOM_PROMPT_OPTION:
        return gr.update()
    return template_value


def build_app():
    with gr.Blocks(title="Dual-WaterMark") as demo:
        gr.Markdown("# Dual-WaterMark Demo")
        gr.Markdown("Two-stage watermarking: **StegaStamp-pytorch → EditGuard** for embedding, then dual verification without removing watermark.")

        with gr.Accordion("Path Settings", open=False):
            editguard_root = gr.Textbox(label="EditGuard root", value=str(DEFAULT_EDITGUARD_ROOT))
            stegastamp_root = gr.Textbox(label="StegaStamp-pytorch root", value=str(DEFAULT_STEGASTAMP_ROOT))
            stegastamp_env = gr.Textbox(label="StegaStamp env (unused)", value=DEFAULT_STEGASTAMP_ENV)
            stegastamp_model_dir = gr.Textbox(label="StegaStamp checkpoint (.pth)", value=str(DEFAULT_STEGASTAMP_MODEL_DIR))
            detector_model_dir = gr.Textbox(
                label="StegaStamp detector model dir (unused in pytorch mode)",
                value=str(DEFAULT_DETECTOR_MODEL_DIR),
            )

        with gr.Tabs():
            with gr.TabItem("1) Add Watermark Pipeline"):
                gr.Markdown("Embed order: **StegaStamp first, then EditGuard**. You can forward page 1 result to page 2/3 with one click.")

                image_input = gr.Image(type="filepath", label="Input image")
                with gr.Row():
                    editguard_bits = gr.Textbox(label="EditGuard bits (64 bits)", value=random_bits_64())
                    random_btn = gr.Button("Random 64-bit")
                stegastamp_secret = gr.Textbox(label="StegaStamp secret (<= 7 UTF-8 bytes)", value="Hello")
                run_embed_btn = gr.Button("Run embed pipeline")

                with gr.Row():
                    editguard_image = gr.Image(label="After StegaStamp")
                    final_image = gr.Image(label="Final output (StegaStamp + EditGuard)")
                stega_residual = gr.Image(label="StegaStamp residual")

                metadata_json = gr.Textbox(label="Encrypted metadata JSON", lines=12)
                embed_status = gr.Textbox(label="Status", lines=4)
                send_next_btn = gr.Button("Send result to page 2 and page 3")
                send_status = gr.Textbox(label="Send status", lines=2)

                random_btn.click(fn=on_random_bits, inputs=[], outputs=[editguard_bits])
                run_embed_btn.click(
                    fn=on_run_embed,
                    inputs=[
                        image_input,
                        editguard_bits,
                        stegastamp_secret,
                        editguard_root,
                        stegastamp_root,
                        stegastamp_env,
                        stegastamp_model_dir,
                    ],
                    outputs=[
                        editguard_image,
                        final_image,
                        stega_residual,
                        metadata_json,
                        embed_status,
                    ],
                )

            with gr.TabItem("2) EditGuard Inpaint"):
                gr.Markdown("Draw your tampered region on the image, then run EditGuard inpaint with prompt.")
                inpaint_image_input = gr.Image(source="upload", tool="sketch", type="numpy", label="Watermarked image (draw mask here)")
                inpaint_metadata = gr.Textbox(label="Metadata JSON from page 1", lines=10)
                inpaint_prompt_template = gr.Dropdown(
                    label="Inpaint prompt template",
                    choices=INPAINT_PROMPT_TEMPLATES,
                    value=CUSTOM_PROMPT_OPTION,
                )
                inpaint_prompt = gr.Textbox(label="Inpaint prompt", value=DEFAULT_INPAINT_PROMPT)
                run_inpaint_btn = gr.Button("Run EditGuard inpaint")

                with gr.Row():
                    inpaint_mask = gr.Image(label="EditGuard mask used for inpaint")
                    inpaint_output = gr.Image(label="Inpaint output")
                inpaint_status = gr.Textbox(label="Inpaint status", lines=3)
                send_inpaint_btn = gr.Button("Send inpaint output to page 3")
                send_inpaint_status = gr.Textbox(label="Send-to-verify status", lines=2)

                run_inpaint_btn.click(
                    fn=on_run_inpaint,
                    inputs=[
                        inpaint_image_input,
                        inpaint_metadata,
                        inpaint_prompt,
                        editguard_root,
                        stegastamp_root,
                        stegastamp_env,
                    ],
                    outputs=[
                        inpaint_mask,
                        inpaint_output,
                        inpaint_status,
                    ],
                )

                inpaint_prompt_template.change(
                    fn=on_select_inpaint_prompt,
                    inputs=[inpaint_prompt_template],
                    outputs=[inpaint_prompt],
                )

            with gr.TabItem("3) Detect & Verify"):
                gr.Markdown("Decode StegaStamp text and verify EditGuard tamper mask from watermarked image. Paste metadata JSON from page 1.")

                verify_image_input = gr.Image(type="numpy", label="Watermarked image to verify")
                verify_metadata = gr.Textbox(label="Metadata JSON from page 1", lines=12)
                run_verify_btn = gr.Button("Run dual verification")

                editguard_mask = gr.Image(label="EditGuard tamper mask")
                with gr.Row():
                    recovered_bits = gr.Textbox(label="EditGuard recovered bits")
                    recovered_accuracy = gr.Textbox(label="EditGuard bit accuracy")

                found_codes = gr.Textbox(label="StegaStamp detected text(s)", lines=4)
                verify_summary = gr.Textbox(label="Verification summary JSON", lines=10)

                run_verify_btn.click(
                    fn=on_run_verify,
                    inputs=[
                        verify_image_input,
                        verify_metadata,
                        editguard_root,
                        stegastamp_root,
                        stegastamp_env,
                        stegastamp_model_dir,
                        detector_model_dir,
                    ],
                    outputs=[
                        editguard_mask,
                        recovered_bits,
                        recovered_accuracy,
                        found_codes,
                        verify_summary,
                    ],
                )

                send_next_btn.click(
                    fn=on_send_to_next_pages,
                    inputs=[final_image, metadata_json],
                    outputs=[
                        inpaint_image_input,
                        inpaint_metadata,
                        verify_image_input,
                        verify_metadata,
                        send_status,
                    ],
                )

                send_inpaint_btn.click(
                    fn=on_send_inpaint_to_verify,
                    inputs=[inpaint_output, inpaint_metadata],
                    outputs=[
                        verify_image_input,
                        verify_metadata,
                        send_inpaint_status,
                    ],
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Dual-WaterMark Gradio app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
