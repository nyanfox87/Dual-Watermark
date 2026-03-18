import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image


class EditGuardAdapter:
    def __init__(self, editguard_root: str):
        self.editguard_root = Path(editguard_root).resolve()
        self.code_dir = self.editguard_root / "code"
        self.checkpoint_path = self.editguard_root / "checkpoints" / "clean.pth"
        self.options_path = self.code_dir / "options" / "test_editguard.yml"

        self._model = None
        self._load_image_fn = None
        self._image_editing_fn = None

    def _ensure_imports(self):
        code_dir_str = str(self.code_dir)
        if code_dir_str not in sys.path:
            sys.path.insert(0, code_dir_str)

    def _ensure_model(self):
        if self._model is not None and self._load_image_fn is not None:
            return

        if not self.code_dir.is_dir():
            raise RuntimeError(f"EditGuard code directory not found: {self.code_dir}")
        if not self.checkpoint_path.is_file():
            raise RuntimeError(f"EditGuard checkpoint not found: {self.checkpoint_path}")
        if not self.options_path.is_file():
            raise RuntimeError(f"EditGuard options not found: {self.options_path}")

        self._ensure_imports()

        import torch
        import options.options as option
        from models import create_model as create_model_editguard
        from test_gradio import image_editing, load_image

        opt = option.parse(str(self.options_path), is_train=True)
        opt["sdinpaint"] = False
        opt["controlnetinpaint"] = False
        opt["sdxl"] = False
        opt["repaint"] = False
        opt["dist"] = False
        opt = option.dict_to_nonedict(opt)
        torch.backends.cudnn.benchmark = True

        model = create_model_editguard(opt)
        model.load_test(str(self.checkpoint_path))

        self._model = model
        self._load_image_fn = load_image
        self._image_editing_fn = image_editing

    @staticmethod
    def _validate_bits(bits: str) -> str:
        if bits is None:
            raise ValueError("EditGuard bits cannot be empty.")
        bits = bits.strip()
        if not re.fullmatch(r"[01]{64}", bits):
            raise ValueError("EditGuard bits must be exactly 64 characters of 0/1.")
        return bits

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image is required.")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and np.max(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    @staticmethod
    def _similarity_percentage(expected: str, recovered: str) -> str:
        if len(expected) == 0:
            return "原始版权水印未知"
        if len(expected) != len(recovered):
            return "输入输出水印长度不同"
        same_count = sum(1 for a, b in zip(expected, recovered) if a == b)
        similarity = (same_count / len(expected)) * 100.0
        return f"{similarity}%"

    def embed(self, image: np.ndarray, bits: str) -> np.ndarray:
        self._ensure_model()
        bits = self._validate_bits(bits)
        image = self._to_uint8_rgb(image)

        message = np.array([int(char) for char in bits], dtype=np.float32)
        message = message - 0.5

        val_data = self._load_image_fn(image, message)
        self._model.feed_data(val_data)
        container = self._model.image_hiding()
        return self._to_uint8_rgb(container)

    def reveal(self, image: np.ndarray, expected_bits: str) -> Dict[str, object]:
        self._ensure_model()
        expected_bits = self._validate_bits(expected_bits)
        image = self._to_uint8_rgb(image)

        container_data = self._load_image_fn(image)
        self._model.feed_data(container_data)
        mask, recovered_bits = self._model.image_recovery(0.2)

        recovered_bits_array = recovered_bits.cpu().numpy()[0]
        recovered_bits_str = "".join(str(int(x)) for x in recovered_bits_array)
        accuracy = self._similarity_percentage(expected_bits, recovered_bits_str)

        if isinstance(mask, Image.Image):
            mask_np = np.array(mask, dtype=np.uint8)
        else:
            mask_np = np.array(mask, dtype=np.uint8)

        if mask_np.ndim == 2:
            mask_np = np.stack([mask_np] * 3, axis=-1)

        return {
            "mask": mask_np,
            "recovered_bits": recovered_bits_str,
            "accuracy": accuracy,
            "bits_match": recovered_bits_str == expected_bits,
        }

    def inpaint(self, image: np.ndarray, prompt: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._ensure_model()
        if self._image_editing_fn is None:
            raise RuntimeError("EditGuard inpaint function is unavailable.")

        image = self._to_uint8_rgb(image)
        if mask is None:
            raise ValueError("Mask is required for EditGuard inpaint.")

        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        if mask_np.dtype != np.uint8:
            if np.issubdtype(mask_np.dtype, np.floating) and np.max(mask_np) <= 1.0:
                mask_np = mask_np * 255.0
            mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)

        prompt = (prompt or "").strip()
        if not prompt:
            prompt = "repair tampered region"

        inpainted = self._image_editing_fn(image, mask_np, prompt)
        return self._to_uint8_rgb(np.array(inpainted))
