"""Turn an arbitrary uploaded image into a Fashion-MNIST style 28x28 input."""
import io

import numpy as np
from PIL import Image, ImageOps

import config


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Return a (28, 28) float32 array in [0, 1] with a dark background.

    Fashion-MNIST items are light objects on a black background, while most
    real photos are dark objects on a light background, so we invert when the
    border of the image is bright.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)

    # Flatten transparency onto white before converting to grayscale.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L")

    arr = np.asarray(img, dtype=np.uint8)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:
        img = ImageOps.invert(img)

    # Pad to a square (keeps aspect ratio), then resize to 28x28.
    side = max(img.size)
    square = Image.new("L", (side, side), 0)
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    img = square.resize((config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)

    return np.asarray(img, dtype=np.float32) / 255.0


def to_png(arr: np.ndarray) -> bytes:
    """Encode the preprocessed 28x28 array as PNG (to show what the model saw)."""
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST).save(buf, "PNG")
    return buf.getvalue()
