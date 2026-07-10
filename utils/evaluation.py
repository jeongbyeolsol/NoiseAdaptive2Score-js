from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from data import (
    StreamingImagePatchDataset,
    find_image_paths,
    load_array,
    preprocess_ardae_data,
)
from utils.image_shape import normalize_image_shape_arg


def parse_float_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def infer_path_data_mode(path, fallback="array"):
    if path is None:
        return fallback
    path = Path(path)
    if path.is_dir():
        return "image-folder"
    return fallback


def normalize_eval_data_modes(args, include_ardae=False):
    eval_data_path = args.noisy_data or args.clean_data
    if args.data_mode == "array":
        args.data_mode = infer_path_data_mode(eval_data_path, fallback=args.data_mode)

    if not include_ardae:
        return

    if getattr(args, "ardae_train_data_mode", None) is None:
        args.ardae_train_data_mode = infer_path_data_mode(
            args.ardae_train_data or args.clean_data,
            fallback=args.data_mode,
        )

    if getattr(args, "ardae_test_data_mode", None) is None:
        args.ardae_test_data_mode = infer_path_data_mode(
            args.ardae_test_data or args.clean_data,
            fallback=args.data_mode,
        )


def normalize_noise_param_aliases(args):
    if not hasattr(args, "poisson_peak"):
        args.poisson_peak = None
    if not hasattr(args, "noise_param_min"):
        args.noise_param_min = None
    if not hasattr(args, "noise_param_max"):
        args.noise_param_max = None
    if not hasattr(args, "linear_noise_param"):
        args.linear_noise_param = False

    if args.poisson_peak is not None:
        if args.noise_type != "poisson":
            raise ValueError("--poisson-peak can only be used with --noise-type poisson.")
        args.noise_param = float(args.poisson_peak)

    if args.noise_param_min is not None or args.noise_param_max is not None:
        if args.noise_param_min is None:
            args.noise_param_min = args.noise_param_max
        if args.noise_param_max is None:
            args.noise_param_max = args.noise_param_min
        args.noise_param_min = float(args.noise_param_min)
        args.noise_param_max = float(args.noise_param_max)
        if args.noise_param_min <= 0 or args.noise_param_max <= 0:
            raise ValueError("Observation noise parameter range must be positive.")
        if args.noise_param_min > args.noise_param_max:
            args.noise_param_min, args.noise_param_max = (
                args.noise_param_max,
                args.noise_param_min,
            )
        args.noise_param = 0.5 * (args.noise_param_min + args.noise_param_max)

    if args.noise_type == "poisson":
        if args.noise_param <= 0:
            raise ValueError("Poisson peak must be positive.")
        args.poisson_peak = float(args.noise_param)
        args.poisson_lam = 1.0 / float(args.noise_param)
        if args.noise_param_min is not None:
            args.poisson_peak_min = float(args.noise_param_min)
            args.poisson_peak_max = float(args.noise_param_max)
            args.poisson_lam_min = 1.0 / float(args.poisson_peak_max)
            args.poisson_lam_max = 1.0 / float(args.poisson_peak_min)
        else:
            args.poisson_peak_min = None
            args.poisson_peak_max = None
            args.poisson_lam_min = None
            args.poisson_lam_max = None
    else:
        args.poisson_lam = None
        args.poisson_peak_min = None
        args.poisson_peak_max = None
        args.poisson_lam_min = None
        args.poisson_lam_max = None


def infer_stream_shape(args, ckpt_image_shape=None):
    image_shape = normalize_image_shape_arg(args.image_shape)
    if image_shape is None:
        image_shape = normalize_image_shape_arg(ckpt_image_shape)
    if image_shape is None:
        if args.patch_size is None:
            raise ValueError(
                "--data-mode image-folder requires --image-shape or --patch-size."
            )
        image_shape = (args.channels, args.patch_size, args.patch_size)

    channels, height, width = image_shape
    if height != width:
        raise ValueError("Streamed patch evaluation currently requires square patches.")
    if args.patch_size is None:
        args.patch_size = height
    if args.patch_size != height or args.patch_size != width:
        raise ValueError(
            f"--patch-size {args.patch_size} does not match image_shape {image_shape}."
        )
    if args.stride is None:
        args.stride = args.patch_size
    args.channels = channels
    return image_shape


def make_eval_loader(args, backbone, image_shape, device, raw_clean=None):
    data_path = args.noisy_data or args.clean_data
    data_key = args.noisy_key if args.noisy_data is not None else args.key

    if args.data_mode == "array":
        if raw_clean is None:
            raw_clean = load_array(data_path, key=data_key)
        raw_clean = raw_clean.float()
        if raw_clean.max() > 1.5:
            raw_clean = raw_clean / 255.0

        clean = preprocess_ardae_data(
            data=raw_clean.clamp(0, 1),
            input_dim=args.input_dim,
            normalize=None,
            flatten=(backbone != "unet"),
            image_shape=image_shape,
        )

        loader = DataLoader(
            TensorDataset(clean),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        return loader, len(clean), str(clean.dtype)

    clean_paths = find_image_paths(data_path, recursive=args.recursive_images)
    if len(clean_paths) == 0:
        raise ValueError(f"No image/npy files found in {data_path}.")

    dataset = StreamingImagePatchDataset(
        image_paths=clean_paths,
        patch_size=args.patch_size,
        stride=args.stride,
        channels=args.channels,
        max_patches_per_image=args.max_patches_per_image,
        seed=0,
        flatten=(backbone != "unet"),
        dtype=torch.float32,
        shuffle_images=False,
        shuffle_patches=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    args.num_clean_files = len(clean_paths)
    args.num_patches = len(dataset)
    return loader, len(dataset), "torch.float32"


def info_dir_for(path):
    path = Path(path)
    return path if path.is_dir() else path.parent


def quality_image_shape(args, image_shape):
    if image_shape is not None:
        return image_shape
    parsed = normalize_image_shape_arg(args.image_shape)
    if parsed is not None:
        return parsed
    if args.patch_size is not None:
        return (args.channels, args.patch_size, args.patch_size)
    return None
