#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 VocalSeparator 需要的本地资源：FFmpeg 与模型权重。

仓库不含这些大文件（GitHub 单文件上限 100 MB），首次使用按需拉取。

用法：
    python scripts/fetch_assets.py --list          # 看要下什么、多大、去向
    python scripts/fetch_assets.py ffmpeg          # 只下 FFmpeg 到 bin\
    python scripts/fetch_assets.py models          # 下载内置模型到 models\
    python scripts/fetch_assets.py all             # 全下

FFmpeg 走 gyan.dev 的 Windows 构建；模型走 HuggingFace 上的公开权重仓库。
若你的网络访问 HuggingFace 有困难，可用 --mirror 指定镜像前缀，
或手工按 --list 打印的文件名放进 models\ 目录。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
MODELS = ROOT / "models"

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

# 内置模型：文件名 -> 公开下载地址。地址失效时请手动获取同名文件。
# 这里只列 app.py 里 BUILTIN_MODELS 必须的那 8 个。
MODEL_SOURCES: dict[str, str] = {
    # UVR / audio-separator 官方权重仓库
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt":
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt":
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
}

# 其余模型需要按对应作者的仓库获取，这里给出可读的查找提示。
MODEL_HINTS = {
    "bs_roformer_vocals_revive_v3e_unwa.ckpt": "unwa 的 vocals revive 系列，见 HuggingFace unwa/ 仓库",
    "melband_roformer_instvox_duality_v2.ckpt": "unwa InstVoc Duality V2",
    "mel_band_roformer_instrumental_fv7z_gabox.ckpt": "Gabox instrumental fv7z",
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": "aufr33/viperx karaoke",
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": "anvuew dereverb SDR 19.1729",
    "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt":
        "anvuew dereverb less-aggressive SDR 18.8050",
}


def builtin_models() -> list[str]:
    import re
    src = (ROOT / "app.py").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"BUILTIN_MODELS\s*=\s*\[(.*?)\]", src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def download(url: str, dest: Path, mirror: str = "") -> bool:
    if mirror:
        for prefix, repl in (("https://huggingface.co", mirror.rstrip("/")),):
            url = url.replace(prefix, repl)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  -> {dest.name}")
    print(f"     {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r     {pct:3d}%  {human(done)} / {human(total)}", end="")
            print()
        tmp.replace(dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"     失败：{type(e).__name__}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def cmd_list(args) -> int:
    print("FFmpeg（约 90 MB 压缩 / 解出约 200 MB）")
    print(f"  目标：{BIN / 'ffmpeg.exe'}")
    print(f"  来源：{FFMPEG_URL}")
    print()
    print(f"内置模型（{len(builtin_models())} 个）")
    for name in builtin_models():
        dest = MODELS / name
        state = "已具备" if dest.exists() else ("可自动下载" if name in MODEL_SOURCES else "需手动获取")
        print(f"  [{state}] {name}")
        if name not in MODEL_SOURCES and name in MODEL_HINTS:
            print(f"            {MODEL_HINTS[name]}")
    print()
    print(f"模型目录：{MODELS}")
    return 0


def cmd_ffmpeg(args) -> int:
    if (BIN / "ffmpeg.exe").exists() and not args.force:
        print("bin\\ffmpeg.exe 已存在（加 --force 重新下载）")
        return 0
    zip_path = BIN / "ffmpeg.zip"
    if not download(FFMPEG_URL, zip_path):
        print("FFmpeg 下载失败。可手工下载后把 ffmpeg.exe / ffprobe.exe 放进 bin\\")
        return 1
    print("  解压 ...")
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            base = os.path.basename(member)
            if base in ("ffmpeg.exe", "ffprobe.exe"):
                with z.open(member) as s, open(BIN / base, "wb") as d:
                    shutil.copyfileobj(s, d)
                print(f"     -> bin\\{base}")
    zip_path.unlink(missing_ok=True)
    BIN.joinpath(".gitkeep").touch()
    return 0


def cmd_models(args) -> int:
    names = builtin_models()
    todo = [n for n in names if not (MODELS / n).exists()]
    if not todo:
        print("内置模型已齐全")
        return 0
    manual = []
    failed = 0
    for name in todo:
        url = MODEL_SOURCES.get(name)
        if not url:
            manual.append(name)
            continue
        if not download(url, MODELS / name, args.mirror):
            failed += 1
    if manual:
        print()
        print(f"以下 {len(manual)} 个模型没有公开直链，需要手动获取后放进 models\\：")
        for n in manual:
            print(f"  - {n}")
            if n in MODEL_HINTS:
                print(f"      {MODEL_HINTS[n]}")
        print()
        print("提示：完整内置模型集合随 Release 的「模型包」提供，")
        print("      见 https://github.com/456erddgr/VocalSeparator/releases")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 VocalSeparator 本地资源")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="列出需要的资源")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("ffmpeg", help="下载 FFmpeg 到 bin/")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_ffmpeg)

    p = sub.add_parser("models", help="下载内置模型到 models/")
    p.add_argument("--mirror", help="HuggingFace 镜像前缀，如 https://hf-mirror.com")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("all", help="下载全部")
    p.add_argument("--force", action="store_true")
    p.add_argument("--mirror")
    p.set_defaults(func=None)
    args = ap.parse_args()

    if args.cmd == "all":
        rc = cmd_ffmpeg(args)
        rc2 = cmd_models(args)
        return rc or rc2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
