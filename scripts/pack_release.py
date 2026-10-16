#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把本仓库打成「绿色部署包 / 一键安装包」，并把大文件放进 Release。

仓库本身只存代码与文档（<5 MB）。模型（约 7 GB）、FFmpeg（约 200 MB）、
自带 Python 运行时（约 90 MB）都走 GitHub Release 或用户自行下载 ——
GitHub 单文件上限 100 MB，直接提交必然失败。

用法：
    # 1) 生成文件清单 + SHA256（用于 Release 说明）
    python scripts/pack_release.py manifest

    # 2) 组装绿色部署包（把仓库代码 + 本地 models/bin/runtime 合到一起）
    python scripts/pack_release.py assemble --out D:\\VocalSeparator-green

    # 3) 生成 7z payload（给 installer_web.py 用）
    python scripts/pack_release.py payload --src D:\\VocalSeparator-green

    # 4) 一次性全做
    python scripts/pack_release.py all --out D:\\VocalSeparator-green
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist_send"

# 需要打进绿色包的仓库文件/目录（代码 + 文档 + 启动器资源）
CODE_ITEMS = [
    "app.py", "launcher.py", "installer_web.py", "uninstall.py",
    "catalog.json", "requirements.txt",
    "start.bat", "停止 VocalSeparator.bat", "一键安装.bat",
    "README.md", "LICENSE",
    "assets", "ui", "show", "docs", "scripts",
]

# 体积很大、不进 Git 但必须随包分发的本机资源
HEAVY_DIRS = ["models", "bin", "runtime"]

# 安装包（installer_web.py）期望的顶层结构
PAYLOAD_ITEMS = CODE_ITEMS + [
    "venv", "models", "bin", "runtime", "assets", "ui", "show",
    "VocalSeparator.exe",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} TB"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1) manifest —— 扫描需要分发的重型资源，产出名字 / 大小 / SHA256
# ---------------------------------------------------------------------------
def cmd_manifest(args) -> int:
    entries = []
    for d in HEAVY_DIRS:
        base = ROOT / d
        if not base.is_dir():
            log(f"跳过（不存在）：{d}")
            continue
        files = sorted(p for p in base.rglob("*") if p.is_file())
        log(f"扫描 {d}：{len(files)} 个文件 ...")
        for p in files:
            entries.append({
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": sha256(p),
            })
    total = sum(e["bytes"] for e in entries)
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(ROOT),
        "file_count": len(entries),
        "total_bytes": total,
        "entries": entries,
    }
    dest = ROOT / "release_manifest.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同时产出人类可读的校验清单（附在 Release 说明里）
    txt = ROOT / "RELEASE_SHA256SUMS.txt"
    lines = [f"{e['sha256']}  {e['path']}" for e in entries]
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log(f"共 {len(entries)} 个文件，{human(total)}")
    log(f"已写出：{dest}")
    log(f"已写出：{txt}")
    return 0


# ---------------------------------------------------------------------------
# 2) assemble —— 组装绿色部署包
# ---------------------------------------------------------------------------
def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def cmd_assemble(args) -> int:
    out = Path(args.out or (DIST / "VocalSeparator-green")).resolve()
    if out.exists() and not args.force:
        log(f"目标已存在：{out}（加 --force 覆盖）")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    for item in CODE_ITEMS:
        src = ROOT / item
        if not src.exists():
            log(f"  跳过（仓库里没有）：{item}")
            continue
        _copy(src, out / item)
        log(f"  代码/文档 -> {item}")

    for d in HEAVY_DIRS:
        src = ROOT / d
        if not src.is_dir():
            log(f"  跳过（本机没有）：{d}/")
            continue
        log(f"  复制 {d}/ ...（可能很慢）")
        _copy(src, out / d)

    # 绿色包里也放一份说明，方便拷给别人
    note = out / "【这是什么】.txt"
    note.write_text(
        "【这是什么】VocalSeparator 人声 / 伴奏分离工具（绿色部署包）\n"
        "\n"
        "用途：把一首歌拆成「人声」和「伴奏」两条音轨；也可去人声、提取伴奏。\n"
        "\n"
        "怎么用：\n"
        "  1. 双击「一键安装.bat」完成部署（建 venv、装依赖、建快捷方式）\n"
        "  2. 双击「start.bat」启动，界面地址 http://127.0.0.1:7861\n"
        "  3. 不想用了运行「停止 VocalSeparator.bat」关服务，或 python uninstall.py 卸载\n"
        "\n"
        "注意：models\\ 与 bin\\ 缺一不可，别单独拷单个文件。\n"
        "     文件夹名不要改，停止脚本靠它找进程。\n"
        "\n"
        "详细文档见 docs\\ 目录。\n",
        encoding="utf-8")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"绿色包已生成：{out}")
    log(f"体积：{human(size)}")
    return 0


# ---------------------------------------------------------------------------
# 3) payload —— 用 7z 打成安装程序需要的 payload.7z
# ---------------------------------------------------------------------------
def find_7z() -> str | None:
    for c in (ROOT / "bin" / "7z.exe", ROOT / "7z.exe",
              Path(r"C:\Program Files\7-Zip\7z.exe")):
        if c.exists():
            return str(c)
    return shutil.which("7z") or shutil.which("7za")


def cmd_payload(args) -> int:
    src = Path(args.src).resolve() if args.src else (DIST / "VocalSeparator-green")
    if not src.is_dir():
        log(f"源目录不存在：{src}（先跑 assemble）")
        return 1
    out = Path(args.out or (DIST / "payload.7z")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # 安装程序只认 payload*.7z 这个名字，用 7z 分卷按需切分
    if out.exists() and not args.force:
        log(f"已存在：{out}（加 --force 覆盖）")
        return 1
    if out.exists():
        out.unlink()

    exe = find_7z()
    if not exe:
        log("找不到 7z.exe / 7za —— 请先安装 7-Zip 或把 7z.exe 放进 bin/")
        return 1

    cmd = [exe, "a", "-t7z", str(out), ".\\*", "-mx=1", "-mmt=on"]
    if args.volume:
        cmd.append(f"-v{args.volume}")
    log("开始压缩（-mx=1 快速模式，模型本身已是压缩格式）...")
    log("$ " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(src))
    if rc != 0:
        log(f"7z 返回 {rc}")
        return rc

    made = sorted(out.parent.glob(out.name + "*"))
    for p in made:
        log(f"  {p.name}  {human(p.stat().st_size)}")
    log("提示：把 payload*.7z 和 VocalSeparator-Setup.exe 放在同一文件夹即可分发。")
    return 0


def cmd_all(args) -> int:
    if cmd_manifest(args) != 0:
        return 1
    if cmd_assemble(args) != 0:
        return 1
    args.src = str(Path(args.out or (DIST / "VocalSeparator-green")).resolve())
    args.out = str(DIST / "payload.7z")
    return cmd_payload(args)


def main() -> int:
    ap = argparse.ArgumentParser(description="VocalSeparator 打包 / Release 助手")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest", help="生成文件清单与 SHA256")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("assemble", help="组装绿色部署包")
    p.add_argument("--out", help="输出目录")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("payload", help="生成 payload.7z")
    p.add_argument("--src", help="绿色包目录")
    p.add_argument("--out", help="输出 7z 路径")
    p.add_argument("--volume", help="分卷大小，如 1900m；不填则单文件")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_payload)

    p = sub.add_parser("all", help="manifest + assemble + payload")
    p.add_argument("--out", help="绿色包输出目录")
    p.add_argument("--volume")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_all)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
