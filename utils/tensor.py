from pathlib import Path

import numpy as np

import torch

'''
miscellaneous functions: learning
'''

def expand_tensor(input, sample_size, do_unsqueeze):
    """
    설명:
        배치 텐서의 각 샘플을 sample_size번 복제한 뒤,
        복제 차원을 배치 차원으로 합친 flattened tensor를 함께 반환한다.

        do_unsqueeze=True인 경우:
            입력 텐서가 [B, ...] 형태라고 가정한다.
            먼저 dim=1 위치에 sample dimension을 추가하여 [B, 1, ...]로 만든 뒤,
            이를 [B, sample_size, ...]로 확장한다.
            이후 [B * sample_size, ...] 형태로 flatten한다.

        do_unsqueeze=False인 경우:
            입력 텐서가 이미 [B, 1, ...] 형태라고 가정한다.
            dim=1의 singleton dimension을 sample_size로 확장하여
            [B, sample_size, ...]로 만든 뒤,
            이를 [B * sample_size, ...] 형태로 flatten한다.

    Args:
        input (torch.Tensor):
            확장할 입력 텐서.
            do_unsqueeze=True이면 [B, ...] 형태,
            do_unsqueeze=False이면 [B, 1, ...] 형태여야 한다.

        sample_size (int):
            각 배치 샘플을 복제할 횟수.

        do_unsqueeze (bool):
            True이면 dim=1에 새 sample dimension을 추가한 뒤 확장한다.
            False이면 input의 dim=1이 이미 크기 1이라고 가정하고 확장한다.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            input_expanded:
                sample dimension이 유지된 확장 텐서.
                do_unsqueeze=True: [B, sample_size, ...]
                do_unsqueeze=False: [B, sample_size, ...]

            input_expanded_flattened:
                sample dimension을 batch dimension으로 합친 텐서.
                shape: [B * sample_size, ...]

    예시:
        input.shape = [B, C, H, W], do_unsqueeze=True인 경우

            input_expanded.shape = [B, sample_size, C, H, W]
            input_expanded_flattened.shape = [B * sample_size, C, H, W]

        input.shape = [B, 1, C, H, W], do_unsqueeze=False인 경우

            input_expanded.shape = [B, sample_size, C, H, W]
            input_expanded_flattened.shape = [B * sample_size, C, H, W]
    """
    batch_size = input.size(0)
    if do_unsqueeze:
        sz_from = [-1]*(input.dim()+1)
        sz_from[1] = sample_size
        input_expanded = input.unsqueeze(1).expand(*sz_from).contiguous()

        sz_to = list(input.size())
        sz_to[0] = batch_size*sample_size
    else:
        assert input.size(1) == 1
        sz_from = [-1]*(input.dim())
        sz_from[1] = sample_size
        input_expanded = input.expand(*sz_from).contiguous()

        _sz_to = list(input.size())
        sz_to = _sz_to[0:1]+_sz_to[2:]
        sz_to[0] = batch_size*sample_size
    input_expanded_flattened = input_expanded.view(*sz_to)
    return input_expanded, input_expanded_flattened



def load_array(path, key=None):
    """
    csv/txt/tsv, npy/npz, pt/pth 파일을 torch.Tensor로 읽는다.

    npz 또는 dict 형태의 pt/pth 파일은 key가 필요할 수 있다.
    key가 없고 항목이 하나뿐이면 그 항목을 자동으로 사용한다.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.loadtxt(path, delimiter=delimiter)
        return torch.as_tensor(array)

    if suffix == ".npy":
        return torch.as_tensor(np.load(path))

    if suffix == ".npz":
        archive = np.load(path)
        if key is None:
            if len(archive.files) != 1:
                raise ValueError(f"key must be given for {path}; found keys: {archive.files}")
            key = archive.files[0]
        return torch.as_tensor(archive[key])

    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            if key is None:
                if len(obj) != 1:
                    raise ValueError(f"key must be given for {path}; found keys: {list(obj.keys())}")
                key = next(iter(obj))
            return torch.as_tensor(obj[key])
        return torch.as_tensor(obj)

    raise ValueError(f"Unsupported data file type: {suffix}")



def normalize_tensor(x, method="standard", eps=1e-8):
    # MLP [N, D]뿐 아니라 이미지 [N, C, H, W]도 sample 차원 기준으로 정규화한다.
    reduce_dims = (0,)

    if method == "standard":
        mean = x.mean(dim=reduce_dims, keepdim=True)
        std = x.std(dim=reduce_dims, keepdim=True).clamp_min(eps)
        return (x - mean) / std

    if method in {"minmax", "zero_one"}:
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
        return (x - x_min) / (x_max - x_min).clamp_min(eps)

    raise ValueError(f"Unknown normalize method: {method}")