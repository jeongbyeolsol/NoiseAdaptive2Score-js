from pathlib import Path

import torch

from models.ardae import ARDAE
from models.noise2score import Noise2Score, Noise2ScoreBlind
from utils.image_shape import as_tuple, infer_image_shape_from_data, normalize_image_shape_arg


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_noise2score_checkpoint(checkpoint):
    return checkpoint.get("checkpoint_type") in {
        "noise2score",
        "noise2score_blind",
    }


def get_ardae_checkpoint(checkpoint):
    if is_noise2score_checkpoint(checkpoint):
        return checkpoint.get("ardae_checkpoint", {})
    return checkpoint


def get_checkpoint_args(checkpoint):
    return get_ardae_checkpoint(checkpoint).get("args", {})


def _load_ardae_from_checkpoint_obj(
    ckpt,
    input_dim,
    device,
    clean=None,
    image_shape_override=None,
):
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt_args.get("backbone", "mlp")

    image_shape = None
    if backbone == "unet":
        if clean is None:
            image_shape = normalize_image_shape_arg(ckpt_args.get("image_shape"))
        else:
            image_shape = infer_image_shape_from_data(
                clean,
                input_dim,
                ckpt_args,
                override=image_shape_override,
            )

    model = ARDAE(
        input_dim=input_dim,
        h_dim=ckpt_args.get("h_dim", 1000),
        noise_param=ckpt_args.get("noise_param", 0.1),
        num_hidden_layers=ckpt_args.get("num_hidden_layers", 1),
        nonlinearity=ckpt_args.get("nonlinearity", "tanh"),
        noise_type=ckpt_args.get("noise_type", "gaussian"),
        use_metric=False,
        backbone=backbone,
        image_shape=image_shape,
        base_channels=ckpt_args.get("base_channels", 64),
        channel_mults=as_tuple(ckpt_args.get("channel_mults"), default=(1, 2, 4, 8)),
        use_norm=not ckpt_args.get("no_norm", False),
        use_gaussian_smoothing=ckpt_args.get("use_gaussian_smoothing", False),
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, backbone, image_shape


def load_ardae_from_checkpoint(path, input_dim, device, clean=None, image_shape_override=None):
    ckpt = _torch_load(path)
    return _load_ardae_from_checkpoint_obj(
        get_ardae_checkpoint(ckpt),
        input_dim=input_dim,
        device=device,
        clean=clean,
        image_shape_override=image_shape_override,
    )


def _noise2score_config_from_args(args, blind=False):
    config = {
        "noise_type": args.noise_type,
        "noise_param": args.noise_param,
        "noise_param_min": getattr(args, "noise_param_min", None),
        "noise_param_max": getattr(args, "noise_param_max", None),
        "linear_noise_param": getattr(args, "linear_noise_param", False),
        "score_sigma": args.score_sigma,
        "clamp": not getattr(args, "no_clamp", False),
    }
    if hasattr(args, "smoothing"):
        config["smoothing"] = args.smoothing
        config["score_smoothing"] = args.smoothing
    if hasattr(args, "smoothing_samples"):
        config["smoothing_samples"] = args.smoothing_samples
        config["score_smoothing_samples"] = args.smoothing_samples
    if getattr(args, "noise_type", None) == "poisson":
        config["poisson_peak"] = getattr(args, "poisson_peak", args.noise_param)
        config["poisson_lam"] = getattr(args, "poisson_lam", None)
    if blind:
        config["blind"] = True
    return config


def load_noise2score_from_checkpoint(
    path,
    input_dim,
    device,
    clean=None,
    image_shape_override=None,
    args=None,
    blind=False,
):
    checkpoint = _torch_load(path)
    ardae_checkpoint = get_ardae_checkpoint(checkpoint)
    ardae, backbone, image_shape = _load_ardae_from_checkpoint_obj(
        ardae_checkpoint,
        input_dim=input_dim,
        device=device,
        clean=clean,
        image_shape_override=image_shape_override,
    )

    if is_noise2score_checkpoint(checkpoint):
        n2s_config = dict(checkpoint.get("noise2score", {}))
    else:
        if args is None:
            raise ValueError("args is required when loading an ARDAE-only checkpoint.")
        n2s_config = _noise2score_config_from_args(args, blind=blind)

    cls = Noise2ScoreBlind if blind else Noise2Score
    n2s = cls(
        ardae=ardae,
        noise_type=n2s_config.get("noise_type", "gaussian"),
        noise_param=n2s_config.get("noise_param", 0.1),
        score_sigma=n2s_config.get("score_sigma", None),
        clamp=n2s_config.get("clamp", True),
    )

    return n2s, backbone, image_shape, checkpoint


def save_noise2score_checkpoint(
    path,
    n2s,
    ardae_checkpoint=None,
    ardae_checkpoint_path=None,
    args=None,
    blind=False,
    extra=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_type = "noise2score_blind" if blind else "noise2score"

    if ardae_checkpoint is None:
        ardae_args = vars(args) if args is not None else {}
        ardae_checkpoint = {
            "model_state_dict": n2s.ardae.state_dict(),
            "args": ardae_args,
        }

    if args is not None:
        n2s_config = _noise2score_config_from_args(args, blind=blind)
    else:
        n2s_config = {
            "noise_type": n2s.noise_type,
            "noise_param": n2s.noise_param,
            "score_sigma": n2s.score_sigma,
            "clamp": n2s.clamp,
        }

    payload = {
        "checkpoint_type": checkpoint_type,
        "noise2score": n2s_config,
        "noise2score_state_dict": n2s.state_dict(),
        "ardae_checkpoint": ardae_checkpoint,
        "ardae_checkpoint_path": str(ardae_checkpoint_path) if ardae_checkpoint_path else None,
        "args": vars(args) if args is not None else {},
    }
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)
    return path
