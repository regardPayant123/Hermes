"""Image input normalization helpers; Pillow stays an optional dependency."""

from io import BytesIO
from pathlib import Path


def load_pil_image(image):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow with: pip install Pillow") from exc

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(image))).convert("RGB")
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as source:
            return source.convert("RGB")
    raise TypeError("image must be a path, bytes, or PIL image")


def model_id_or_error(model_id: str | None, capability: str) -> str:
    if not model_id or not model_id.strip():
        raise ValueError(f"A model_id or model_path must be configured for {capability}")
    return model_id
