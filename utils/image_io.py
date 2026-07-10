from pathlib import Path

import numpy as np
import torch


def make_stitch_coords(height, width, patch_size, stride):
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"image size {(height, width)} is smaller than patch_size={patch_size}."
        )

    ys = list(range(0, height - patch_size + 1, stride))
    xs = list(range(0, width - patch_size + 1, stride))
    if ys[-1] != height - patch_size:
        ys.append(height - patch_size)
    if xs[-1] != width - patch_size:
        xs.append(width - patch_size)

    return [(y, x) for y in ys for x in xs]


def load_image_tensor(path, channels, dtype=torch.float32):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        image = np.load(path)
        if channels == 1:
            if image.ndim == 2:
                tensor = torch.as_tensor(image, dtype=dtype).unsqueeze(0)
            elif image.ndim == 3 and image.shape[-1] == 1:
                tensor = torch.as_tensor(image[..., 0], dtype=dtype).unsqueeze(0)
            elif image.ndim == 3 and image.shape[0] == 1:
                tensor = torch.as_tensor(image, dtype=dtype)
            else:
                raise ValueError(f"Unsupported grayscale npy shape {image.shape} from {path}")
        else:
            if image.ndim == 3 and image.shape[-1] == 3:
                tensor = torch.as_tensor(image, dtype=dtype).permute(2, 0, 1)
            elif image.ndim == 3 and image.shape[0] == 3:
                tensor = torch.as_tensor(image, dtype=dtype)
            else:
                raise ValueError(f"Unsupported RGB npy shape {image.shape} from {path}")

        if np.issubdtype(image.dtype, np.integer):
            tensor = tensor / 255.0
        return tensor.clamp(0, 1).contiguous()

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image tensor loading for non-npy files requires Pillow.") from exc

    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode)).copy()

    if channels == 1:
        tensor = torch.as_tensor(array, dtype=dtype).unsqueeze(0)
    else:
        tensor = torch.as_tensor(array, dtype=dtype).permute(2, 0, 1)

    return (tensor / 255.0).clamp(0, 1).contiguous()


def tensor_to_image_array(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.dim() != 3:
        raise ValueError(f"Expected [C,H,W] tensor, got {tuple(tensor.shape)}")

    if tensor.size(0) == 1:
        array = tensor.squeeze(0).numpy()
    elif tensor.size(0) == 3:
        array = tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.size(0)}")

    return (array * 255.0).round().clip(0, 255).astype(np.uint8)


def save_stitched_tensor(tensor, path_base, output_format):
    path_base.parent.mkdir(parents=True, exist_ok=True)

    if output_format in {"npy", "both"}:
        np.save(path_base.with_suffix(".npy"), tensor.detach().cpu().numpy())

    if output_format in {"png", "both"}:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Saving stitched PNG outputs requires Pillow.") from exc

        Image.fromarray(tensor_to_image_array(tensor)).save(path_base.with_suffix(".png"))


load_clean_image_tensor = load_image_tensor
