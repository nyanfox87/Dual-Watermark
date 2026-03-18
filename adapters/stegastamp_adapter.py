import importlib
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageOps


class StegaStampAdapter:
    def __init__(self, stegastamp_root: str, stegastamp_env: str = "stegastamp"):
        self.stegastamp_root = Path(stegastamp_root).resolve()
        self.default_model_path = self.stegastamp_root / "asset" / "best.pth"
        self.stegastamp_env = stegastamp_env
        self.secret_size = 100
        self.width = 400
        self.height = 400

        self._ready = False
        self._torch = None
        self._bch = None
        self._bits_to_bytes = None
        self._prepare_deployment_hiding = None
        self._encoder = None
        self._decoder = None

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image is required.")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def _resolve_model_path(self, model_dir: str = None) -> Path:
        model_path = Path(model_dir).resolve() if model_dir else self.default_model_path
        if not model_path.is_file():
            raise RuntimeError(f"StegaStamp checkpoint not found: {model_path}")
        return model_path

    @staticmethod
    def _load_checkpoint(path: Path, torch_module):
        with open(path, "rb") as f:
            header = f.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"StegaStamp checkpoint is a Git LFS pointer, not real weights: {path}. "
                "Run `git lfs install && git lfs pull` in StegaStamp-pytorch."
            )
        try:
            return torch_module.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch_module.load(str(path), map_location="cpu")

    def _ensure_imports(self):
        root_str = str(self.stegastamp_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def _ensure_models(self, model_path: Path):
        if self._ready:
            return

        if not self.stegastamp_root.is_dir():
            raise RuntimeError(f"StegaStamp-pytorch root not found: {self.stegastamp_root}")

        self._ensure_imports()

        torch_module = importlib.import_module("torch")
        bchlib_module = importlib.import_module("bchlib")
        models_module = importlib.import_module("stegastamp.models")
        decode_module = importlib.import_module("stegastamp.decode_image")

        device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
        encoder = models_module.StegaStampEncoder(
            height=self.height,
            width=self.width,
            secret_size=self.secret_size,
        ).to(device)
        decoder = models_module.StegaStampDecoder(
            secret_size=self.secret_size,
            height=self.height,
            width=self.width,
        ).to(device)

        ckpt = self._load_checkpoint(model_path, torch_module)
        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"])
        else:
            encoder.load_state_dict(ckpt)

        if "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"])
        else:
            decoder.load_state_dict(ckpt)

        encoder.eval()
        decoder.eval()

        self._torch = torch_module
        self._bch = bchlib_module.BCH(5, 137)
        self._bits_to_bytes = decode_module.bits_to_bytes
        self._prepare_deployment_hiding = models_module.prepare_deployment_hiding
        self._encoder = encoder
        self._decoder = decoder
        self._device = device
        self._ready = True

    def _secret_to_tensor(self, secret: str):
        if secret is None:
            secret = ""
        if len(secret) > 7:
            raise ValueError("StegaStamp secret must be <= 7 characters.")

        data = bytearray(secret + " " * (7 - len(secret)), "utf-8")
        ecc = self._bch.encode(data)
        packet = data + ecc
        packet_binary = "".join(format(x, "08b") for x in packet)
        secret_bits = [int(x) for x in packet_binary]
        secret_bits.extend([0, 0, 0, 0])

        secret_tensor = self._torch.tensor(
            secret_bits,
            dtype=self._torch.float32,
            device=self._device,
        ).unsqueeze(0)

        if secret_tensor.shape[1] < self.secret_size:
            pad = self._torch.zeros(1, self.secret_size - secret_tensor.shape[1], device=self._device)
            secret_tensor = self._torch.cat([secret_tensor, pad], dim=1)
        else:
            secret_tensor = secret_tensor[:, :self.secret_size]
        return secret_tensor

    def _image_to_tensor(self, image: np.ndarray):
        image = self._to_uint8_rgb(image)
        fitted = ImageOps.fit(Image.fromarray(image), (self.width, self.height))
        arr = np.array(fitted, dtype=np.float32) / 255.0
        return self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def encode(self, image: np.ndarray, secret: str, model_dir: str = None) -> Dict[str, np.ndarray]:
        model_path = self._resolve_model_path(model_dir)
        self._ensure_models(model_path)

        image_tensor = self._image_to_tensor(image)
        secret_tensor = self._secret_to_tensor(secret)

        with self._torch.no_grad():
            hidden_img, residual = self._prepare_deployment_hiding(self._encoder, secret_tensor, image_tensor)

        hidden = (hidden_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        residual_viz = ((residual[0].permute(1, 2, 0).cpu().numpy() + 0.5) * 255).clip(0, 255).astype(np.uint8)

        return {
            "hidden": hidden,
            "residual": residual_viz,
            "secret": secret,
            "log": "encoded by stegastamp-pytorch",
        }

    def decode(self, image: np.ndarray, model_dir: str = None) -> Dict[str, object]:
        model_path = self._resolve_model_path(model_dir)
        self._ensure_models(model_path)

        image_tensor = self._image_to_tensor(image)

        with self._torch.no_grad():
            logits = self._decoder(image_tensor)
            bits = (
                self._torch.round(self._torch.sigmoid(logits))
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.uint8)
                .tolist()
            )

        raw = self._bits_to_bytes(bits[:-4])
        data, ecc = raw[:7], raw[7:]
        try:
            self._bch.decode(bytearray(data), bytearray(ecc))
            msg = bytes(data).decode("utf-8").strip()
            return {
                "decoded_text": msg,
                "bitflips": None,
                "success": True,
                "log": "decoded by stegastamp-pytorch",
            }
        except Exception:
            return {
                "decoded_text": "",
                "bitflips": None,
                "success": False,
                "log": "decode failed",
            }

    def detect_decode(self, image: np.ndarray, detector_model_dir: str, decoder_model_dir: str = None) -> Dict[str, object]:
        decoded = self.decode(image=image, model_dir=decoder_model_dir)
        found_codes: List[str] = []
        decoded_text = (decoded.get("decoded_text") or "").strip()
        if decoded_text:
            found_codes = [decoded_text]

        return {
            "annotated": None,
            "mask": None,
            "found_codes": found_codes,
            "log": decoded.get("log", "decode-only"),
        }
