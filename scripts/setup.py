#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VocalSeparator 环境自检 / 依赖安装助手（跨平台，命令行版）。

用途：
    python scripts/setup.py              # 自检当前环境并给出修复建议
    python scripts/setup.py --install    # 自动创建 venv 并安装依赖
    python scripts/setup.py --check-models  # 校验 models 目录与 catalog.json 是否对得上

与 `一键安装.bat` 的区别：本脚本是命令行 / 可脚本化入口，
便于 CI 与自动化部署；Windows 普通用户直接双击 bat 即可。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MODELS = ROOT / "models"
BIN = ROOT / "bin"
CATALOG = ROOT / "catalog.json"

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple/",
]

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def say(mark: str, text: str) -> None:
    print(f"{mark} {text}")


def run(cmd: list[str], **kw) -> int:
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], cwd=str(ROOT), **kw)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check_python() -> bool:
    v = sys.version_info
    if (v.major, v.minor) in ((3, 10), (3, 11), (3, 12)):
        say(OK, f"Python 解释器 {v.major}.{v.minor}.{v.micro}")
        return True
    say(WARN, f"Python {v.major}.{v.minor} 不在推荐范围 3.10~3.12 内，可能有依赖不兼容。")
    return False


def check_venv() -> bool:
    if PY.is_file():
        say(OK, f"虚拟环境存在：{PY}")
        return True
    say(WARN, f"没有虚拟环境：{PY}（运行 --install 可自动创建）")
    return False


def check_deps() -> bool:
    if not PY.is_file():
        say(WARN, "跳过依赖自检（虚拟环境还不存在）")
        return False
    code = (
        "import sys\n"
        "mods=[('torch','torch'),('gradio','gradio'),('audio_separator','audio_separator'),"
        "('onnxruntime','onnxruntime'),('soundfile','soundfile'),('librosa','librosa')]\n"
        "bad=[]\n"
        "for name,imp in mods:\n"
        "    try:\n"
        "        m=__import__(imp); print('  OK ',name,getattr(m,'__version__','?'))\n"
        "    except Exception as e:\n"
        "        bad.append(name); print('  BAD',name,type(e).__name__,e)\n"
        "try:\n"
        "    import torch; print('  cuda',torch.cuda.is_available(),"
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
        "except Exception: pass\n"
        "sys.exit(1 if bad else 0)\n"
    )
    rc = run([PY, "-c", code])
    if rc == 0:
        say(OK, "关键依赖齐全")
        return True
    say(FAIL, "有依赖缺失，请重新运行 --install")
    return False


def check_bin() -> bool:
    need = ["ffmpeg.exe"] if os.name == "nt" else ["ffmpeg"]
    missing = [n for n in need if not (BIN / n).exists()]
    if not missing:
        say(OK, f"本地二进制就位：{', '.join(need)}")
        return True
    say(FAIL, f"缺少 {', '.join(missing)}（请放入 {BIN}）—— 没有它无法解码音频")
    return False


def check_models() -> bool:
    if not CATALOG.is_file():
        say(FAIL, "缺少 catalog.json")
        return False
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    present = {p.name for p in MODELS.glob("*") if p.is_file()} if MODELS.is_dir() else set()
    # 内置模型：app.py 里的 BUILTIN_MODELS 才是运行时真正要求的集合
    builtin = _read_builtin_models()
    missing_builtin = [m for m in builtin if m not in present]
    all_files = {it["file"] for it in data.get("all", [])}
    have = sorted(f for f in all_files if f in present)
    if not present:
        say(FAIL, f"models 目录是空的（catalog.json 登记 {len(all_files)} 个模型，"
                  f"其中 {len(builtin)} 个是内置必须项）")
        return False
    if missing_builtin:
        say(WARN, f"内置模型缺 {len(missing_builtin)}/{len(builtin)} 个，"
                  f"缺的档位不会出现在界面下拉里：")
        for m in missing_builtin:
            print("        - " + m)
    else:
        say(OK, f"内置模型齐全（{len(builtin)} 个）")
    say(OK, f"models 目录共 {len(present)} 个文件，"
            f"其中 catalog 登记且已具备的有 {len(have)} 个")
    return not missing_builtin


def _read_builtin_models() -> list[str]:
    """从 app.py 里解析 BUILTIN_MODELS，避免在两处维护清单。"""
    try:
        import re
        src = (ROOT / "app.py").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"BUILTIN_MODELS\s*=\s*\[(.*?)\]", src, re.S)
        if not m:
            return []
        return re.findall(r'"([^"]+)"', m.group(1))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 安装
# ---------------------------------------------------------------------------
def install() -> int:
    if PY.is_file():
        say(OK, f"虚拟环境已存在，跳过创建：{PY}")
    else:
        say(OK, f"创建虚拟环境 {VENV} ...")
        if run([sys.executable, "-m", "venv", str(VENV)]) != 0:
            say(FAIL, "创建 venv 失败")
            return 1

    ok = True
    if run([PY, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]) != 0:
        ok = False

    say(OK, f"安装 PyTorch（CUDA 12.8）：{TORCH_INDEX}")
    if run([PY, "-m", "pip", "install", "torch", "torchaudio", "torchvision",
            "--index-url", TORCH_INDEX]) != 0:
        say(WARN, "CUDA 版 torch 失败，回退 CPU 版（能用但慢很多）")
        run([PY, "-m", "pip", "install", "torch", "torchaudio", "torchvision"])

    installed = False
    for mirror in MIRRORS:
        say(OK, f"安装 requirements.txt（{mirror}）")
        if run([PY, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"),
                "-i", mirror]) == 0:
            installed = True
            break
        say(WARN, "该镜像失败，换下一个")
    if not installed:
        ok = False

    say(OK, "依赖安装阶段结束，开始自检")
    check_deps()
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="VocalSeparator 环境自检 / 安装助手")
    ap.add_argument("--install", action="store_true", help="创建 venv 并安装依赖")
    ap.add_argument("--check-models", action="store_true", help="只校验模型目录")
    args = ap.parse_args()

    print("=" * 60)
    print("  VocalSeparator 环境助手")
    print(f"  项目目录：{ROOT}")
    print("=" * 60)

    if args.install:
        return install()

    if args.check_models:
        return 0 if check_models() else 1

    print("\n-- 基础环境 --")
    check_python()
    check_venv()
    print("\n-- 本地资源 --")
    check_bin()
    check_models()
    print("\n-- 依赖 --")
    check_deps()

    print("\n" + "=" * 60)
    print("  提示：")
    print("    python scripts/setup.py --install   创建 venv 并装依赖")
    print("    详细部署说明见 docs/部署与打包.md")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
