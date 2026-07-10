import math


def as_tuple(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


def parse_channel_mults(text):
    return as_tuple(text)


def normalize_image_shape_arg(image_shape):
    image_shape = as_tuple(image_shape)
    if image_shape is None:
        return None
    if len(image_shape) == 2:
        return (1, *image_shape)
    if len(image_shape) == 3:
        return image_shape
    raise ValueError("image_shape must be H W or C H W")


def infer_image_shape_from_data(clean, input_dim, ckpt_args=None, override=None):
    ckpt_args = {} if ckpt_args is None else ckpt_args

    image_shape = normalize_image_shape_arg(override)
    if image_shape is not None:
        return image_shape

    image_shape = normalize_image_shape_arg(ckpt_args.get("image_shape"))
    if image_shape is not None:
        return image_shape

    if clean is not None:
        if clean.ndim == 4:
            if clean.shape[1] in (1, 3):
                return tuple(int(v) for v in clean.shape[1:])
            if clean.shape[-1] in (1, 3):
                return (int(clean.shape[-1]), int(clean.shape[1]), int(clean.shape[2]))

        if clean.ndim == 3:
            return (1, int(clean.shape[1]), int(clean.shape[2]))

    if input_dim is not None:
        side = int(round(math.sqrt(float(input_dim))))
        if side * side == int(input_dim):
            return (1, side, side)

    raise ValueError("Could not infer image_shape. Pass --image-shape C H W.")
