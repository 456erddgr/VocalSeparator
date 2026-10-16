#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CI：检查所有 Markdown 里的相对链接都指向存在的文件 / 目录。

只检查仓库内的相对路径；http(s)、mailto、纯锚点（#...）跳过。
同时报告：
  - 断链（指向不存在的路径）      -> 导致 CI 失败
  - 指向 images 之外的可疑引用      -> 仅提示

设计说明：之前用 bash `while read` + 管道写临时文件统计失败数，
子 shell 里变量丢失导致判断永远不成立；这里改成 Python 直接收集，
行为明确可测。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# [文本](目标)  —— 只取目标部分，支持标题与路径里的空格用 <> 包裹
LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<[^>]*>|[^)\s]+)")

SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "data:", "ftp://")


def normalize(target: str) -> str:
    t = target.strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1].strip()
    # 去掉锚点与查询串
    t = t.split("#", 1)[0].split("?", 1)[0]
    return t


def strip_code(text: str) -> str:
    """把代码块与行内 code span 挖空，避免把「示例路径」当成真链接。

    例如文档里写 `![图](../show/1.png)` 用反引号包起来讲解写法时，
    那只是一个字符串示例，不是真的引用，不该参与断链检查。
    """
    # 先去掉 ``` 围起来的代码块（含语言标注行）
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    # 再去掉 ~~~ 围栏
    text = re.sub(r"~~~.*?~~~", "", text, flags=re.S)
    # 最后去掉行内 code span（允许用多个反引号包裹）
    text = re.sub(r"`+[^`]*`+", "", text)
    return text


def main() -> int:
    broken: list[tuple[str, str, str]] = []
    checked = 0
    files = sorted(p for p in ROOT.rglob("*.md")
                   if ".git" not in p.parts and "node_modules" not in p.parts)

    if not files:
        print("没找到 Markdown 文件")
        return 1

    for md in files:
        text = strip_code(md.read_text(encoding="utf-8", errors="replace"))
        for raw in LINK_RE.findall(text):
            target = normalize(raw)
            if not target or target.startswith(SKIP_PREFIX):
                continue
            # 绝对路径不在仓库范围内，跳过
            if target.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", target):
                continue
            checked += 1
            # 链接可能是 URL 编码的（例如含中文空格）
            from urllib.parse import unquote
            rel = unquote(target).replace("\\", "/")
            dest = (md.parent / rel).resolve()
            if not dest.exists():
                try:
                    shown = md.relative_to(ROOT)
                except ValueError:
                    shown = md
                broken.append((str(shown), raw, str(dest)))

    print(f"扫描 {len(files)} 个 Markdown，检查 {checked} 个相对链接")

    if broken:
        print()
        print(f"发现 {len(broken)} 个断链：")
        for src, raw, dest in broken:
            print(f"  {src}: {raw}")
            print(f"      -> 解析为 {dest}（不存在）")
        return 1

    print("全部链接有效")
    return 0


if __name__ == "__main__":
    sys.exit(main())
