from pathlib import Path
from tqdm import tqdm
import json
import shutil
from collections import Counter
import argparse

import numpy as np
import torch


def make_unique_save_path(base_dir):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return base_dir

    for index in range(1, 10000):
        candidate = base_dir.with_name(f"{base_dir.name}_{index:03d}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an unused save directory for {base_dir}")


def log_message(message, log_path):
    tqdm.write(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_config(path, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(args, dict):
        config_obj = args

    elif isinstance(args, argparse.Namespace):
        config_obj = vars(args)

    elif isinstance(args, (list, tuple)):
        config_obj = args

    elif hasattr(args, "__dict__"):
        config_obj = vars(args)

    else:
        config_obj = args

    config_obj = _make_json_serializable(config_obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            config_obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )


def _make_json_serializable(obj):
    """JSON 직렬화가 어려운 객체들을 재귀적으로 변환합니다."""

    if isinstance(obj, Path):
        return obj.as_posix()

    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            _make_json_serializable(k): _make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]

    return obj

def make_unique_file_path(path):
    path = Path(path)
    if not path.exists():
        return path

    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find an unused file path for {path}")


def file_from_to(src, dst, *, new_name=None, overwrite=False, on_conflict="error"):
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    if not src.is_file():
        raise ValueError(f"src must be a file, got: {src}")

    # dst를 디렉토리로 취급하는 경우
    if dst.exists() and dst.is_dir():
        dst_file = dst / (new_name or src.name)

    elif dst.suffix == "":
        dst.mkdir(parents=True, exist_ok=True)
        dst_file = dst / (new_name or src.name)

    # dst를 파일 경로로 취급하는 경우
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst_file = dst

    if dst_file.exists() and not overwrite:
        if on_conflict == "suffix":
            dst_file = make_unique_file_path(dst_file)
        elif on_conflict != "error":
            raise ValueError("on_conflict must be 'error' or 'suffix'.")
        else:
            raise FileExistsError(
                f"Destination already exists: {dst_file}\n"
                f"Use overwrite=True to replace it."
            )

    if dst_file.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {dst_file}\n"
            f"Use overwrite=True to replace it."
        )

    shutil.copy2(src, dst_file)
    return dst_file


def config_all_from_to(
    src_dir,
    dst_dir,
    *,
    new_name=None,
    prefix=None,
    suffix=None,
    dst_subdir=None,
    recursive=False,
    overwrite=False,
    on_conflict="suffix",
    strict=False,
    is_csv=False
):
    """
    src_dir 안의 모든 .json 파일을 dst_dir로 복사합니다.
    내부적으로 config_from_to()를 재사용합니다.

    Parameters
    ----------
    src_dir : str | Path
        JSON config들이 들어 있는 원본 디렉토리.

    dst_dir : str | Path
        복사 대상 디렉토리.

    new_name : str | None
        복사할 JSON이 정확히 하나일 때만 사용할 새 이름.
        여러 JSON이 있을 때는 충돌 방지를 위해 사용 불가.

    prefix : str | None
        복사되는 파일명 앞에 붙일 prefix.
        예: prefix="data_"이면 config.json -> data_config.json
    suffix : str | None
        복사되는 파일명 stem 뒤에 붙일 suffix.
        예: suffix="_data"이면 config.json -> config_data.json

    dst_subdir : str | None
        dst_dir 아래에 따로 config들을 모을 하위 폴더명.
        예: dst_subdir="data_configs"

    recursive : bool
        True면 하위 디렉토리까지 재귀적으로 .json을 찾습니다.

    overwrite : bool
        대상 파일이 이미 있을 때 덮어쓸지 여부.

    on_conflict : {"suffix", "error"}
        overwrite=False이고 대상 파일이 이미 있을 때 처리 방식.
        "suffix"이면 _001, _002처럼 자동 suffix를 붙입니다.

    strict : bool
        True면 json 파일이 없을 때 에러를 냅니다.
        
    is_csv: bool
        True면 위 로직은 csv파일에 적용

    Returns
    -------
    list[Path]
        복사된 파일 경로 목록.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if dst_subdir is not None:
        dst_dir = dst_dir / dst_subdir

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    if not src_dir.is_dir():
        raise NotADirectoryError(f"src_dir must be a directory: {src_dir}")

    extension = '.json' if not is_csv else '.csv'
    pattern = "**/*"+extension if recursive else "*" + extension
    json_paths = sorted(p for p in src_dir.glob(pattern) if p.is_file())
    
    if recursive and _has_duplicate_filenames(json_paths):
        raise ValueError(
            "Duplicate filenames found during recursive copy. "
            "Use prefix=..., dst_subdir=..., or disable recursive=True."
        )

    if len(json_paths) == 0:
        msg = f"No {extension} files found in: {src_dir}"
        if strict:
            raise FileNotFoundError(msg)
        return []

    if new_name is not None and len(json_paths) > 1:
        raise ValueError(
            f"new_name can only be used when exactly one {extension} file is found. "
            f"Use prefix=... or dst_subdir=... for multiple {extension} files."
        )

    copied_paths = []

    for src_path in json_paths:
        if new_name is not None:
            copied_path = file_from_to(
                src_path,
                dst_dir,
                new_name=new_name,
                overwrite=overwrite,
                on_conflict=on_conflict,
            )

        elif prefix is not None or suffix is not None:
            stem_suffix = "" if suffix is None else suffix
            copied_path = file_from_to(
                src_path,
                dst_dir,
                new_name=f"{prefix or ''}{src_path.stem}{stem_suffix}{src_path.suffix}",
                overwrite=overwrite,
                on_conflict=on_conflict,
            )

        else:
            copied_path = file_from_to(
                src_path,
                dst_dir,
                new_name=src_path.name,
                overwrite=overwrite,
                on_conflict=on_conflict,
            )

        copied_paths.append(copied_path)

    return copied_paths


def _has_duplicate_filenames(paths):
    """
    Path 목록에서 파일명(name)이 겹치는지 확인합니다.

    Parameters
    ----------
    paths : iterable[Path]
        검사할 파일 경로 목록.

    Returns
    -------
    bool
        같은 파일명이 2개 이상 있으면 True.
    """
    names = [Path(p).name for p in paths]
    counts = Counter(names)
    return any(count > 1 for count in counts.values())
