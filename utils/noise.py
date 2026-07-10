import torch



def make_noise_param(x, sigma_min=0.001, sigma_max=0.5, use_log_scale=True):
    if sigma_min == sigma_max:
        return torch.full(
            (x.size(0), 1),
            float(sigma_min),
            device=x.device,
            dtype=x.dtype,
        )

    if use_log_scale:
        log_sigma_min = torch.log(torch.tensor(sigma_min, device=x.device, dtype=x.dtype))
        log_sigma_max = torch.log(torch.tensor(sigma_max, device=x.device, dtype=x.dtype))

        log_sigma = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype).uniform_(
            log_sigma_min,
            log_sigma_max,
        )
        return log_sigma.exp()

    return torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype).uniform_(
        sigma_min,
        sigma_max,
    )


def make_observation_noise_param(args, x):
    noise_param_min = getattr(args, "noise_param_min", None)
    noise_param_max = getattr(args, "noise_param_max", None)

    if noise_param_min is None and noise_param_max is None:
        return getattr(args, "noise_param", None)

    if noise_param_min is None:
        noise_param_min = noise_param_max
    if noise_param_max is None:
        noise_param_max = noise_param_min

    use_log_scale = not getattr(args, "linear_noise_param", False)
    return make_noise_param(
        x,
        sigma_min=float(noise_param_min),
        sigma_max=float(noise_param_max),
        use_log_scale=use_log_scale,
    )


def summarize_noise_param(noise_param):
    if noise_param is None or not torch.is_tensor(noise_param):
        return None

    values = noise_param.detach().float().cpu().view(-1)
    if values.numel() == 0:
        return None

    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
    }


def add_gaussian_noise(input, std):
    std = _view_param(std, input)
    eps = torch.randn_like(input)
    x_bar = input + std * eps
    return x_bar, eps


def add_poisson_noise(input, peak=30.0, clamp=True):
    peak = _view_param(peak, input)

    rate = input.clamp_min(0) * peak
    noisy_counts = torch.poisson(rate)
    x_bar = noisy_counts / peak

    if clamp:
        x_bar = x_bar.clamp(0, 1)

    eps = x_bar - input
    return x_bar, eps


def add_gamma_noise(input, concentration=2.0, clamp=True):
    concentration = _view_param(concentration, input)

    alpha = concentration.expand_as(input)
    rate = concentration.expand_as(input)

    gamma_noise = torch.distributions.Gamma(alpha, rate).sample()
    x_bar = input * gamma_noise

    if clamp:
        x_bar = x_bar.clamp(0, 1)

    eps = x_bar - input
    return x_bar, eps


def add_observation_noise(x, noise_type, noise_param):
    if noise_type == "gaussian":
        y, _ = add_gaussian_noise(x, std=noise_param)
        return y.clamp(0, 1)

    if noise_type == "poisson":
        y, _ = add_poisson_noise(x, peak=noise_param)
        return y

    if noise_type == "gamma":
        y, _ = add_gamma_noise(x, concentration=noise_param)
        return y

    raise NotImplementedError(noise_type)


def denoise_from_score(y, score, noise_type, noise_param, clamp=True, smoothing=0.0):
    smoothing = float(smoothing or 0.0)

    if smoothing > 0.0 and noise_type != "gaussian":
        x_hat = y + smoothing ** 2 * score

    elif noise_type == "gaussian":
        sigma = noise_param
        x_hat = y + sigma ** 2 * score

    elif noise_type == "poisson":
        peak = noise_param
        x_hat = (y + 1.0 / (2.0 * peak)) * torch.exp(score / peak)

    elif noise_type == "gamma":
        alpha = noise_param
        denom = (alpha - 1.0) - y * score
        denom = denom.clamp_min(1e-6)
        x_hat = alpha * y / denom

    else:
        raise NotImplementedError(f"Unknown noise_type: {noise_type}")

    if clamp:
        x_hat = x_hat.clamp(0, 1)

    return x_hat
  
  
def _view_param(param, input):
    """
    param을 input에 broadcast 가능한 shape으로 바꾼다.

    input:
        [B, D] 또는 [B, C, H, W]
    param:
        scalar, [B], [B, 1], [B, 1, 1, 1] 가능

    반환값은 input과 곱셈/나눗셈이 가능한 형태가 된다.
    """
    if not torch.is_tensor(param):
        param = input.new_tensor(float(param))

    param = param.to(device=input.device, dtype=input.dtype)

    if param.ndim == 0:
        return param

    batch_size = input.size(0)

    if param.ndim == 1:
        param = param.view(batch_size, 1)

    if input.ndim <= 2:
        return param

    if param.ndim == 2:
        return param.view(batch_size, 1, *([1] * (input.ndim - 2)))

    return param
