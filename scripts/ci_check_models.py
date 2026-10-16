#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CI：校验 app.py 里的模型声明都能在 catalog.json 中找到。

检查两处声明：
  - BUILTIN_MODELS     —— 界面「内置模型」下拉里的权重
  - DEREVERB_CHOICES   —— 去混响 / 去回声档位，值是模型文件名，
                          双模型串联时用 "+" 连接，要逐个拆开检查

失败即说明代码与模型目录登记不同步。退出码非 0。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
CATALOG = ROOT / "catalog.json"


def extract_list(source: str, name: str) -> list[str]:
    """从源码里抽出 `name = [ ... ]` 里的所有双引号字符串。"""
    m = re.search(rf"{name}\s*=\s*\[(.*?)\]", source, re.S)
    if not m:
        return []
    return re.findall(r'"([^"]+)"', m.group(1))


def extract_dict_values(source: str, name: str) -> list[str]:
    """从源码里抽出 `name = { ... }` 的所有「值」字符串。

    早期版本用逐行正则，结果漏掉了「键与值分处两行」的写法：

        "去回声 · De-Reverb-Echo V2（推荐）":
            "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",

    改用 ast 解析，彻底摆脱缩进/换行/注释带来的脆弱性。解析失败时
    回退到「正则抽 dict body 内全部字符串，偶数下标为键」的粗略策略。
    """
    import ast
    import warnings

    # app.py 里内嵌了 JavaScript（VS_JS 等字符串常量），其中的正则字面量
    # 例如 /\s+/g 会被 Python 当成无效转义序列告警。这不是代码缺陷，
    # 只是解析整个文件时的噪音，这里静音。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        # 首选：真正的语法树解析
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if name in targets and isinstance(node.value, ast.Dict):
                        return [v.value for v in node.value.values
                                if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        except SyntaxError:
            pass

    # 回退：只取冒号右侧的字符串
    m = re.search(rf"{name}\s*=\s*\{{(.*?)\n\}}", source, re.S)
    if not m:
        return []
    return re.findall(r':\s*"([^"]+)"', m.group(1))


def main() -> int:
    if not APP.is_file() or not CATALOG.is_file():
        print("缺少 app.py 或 catalog.json")
        return 1

    src = APP.read_text(encoding="utf-8", errors="replace")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    known = {it["file"] for it in catalog.get("all", [])}

    problems: list[str] = []

    # 1) 内置模型
    builtin = extract_list(src, "BUILTIN_MODELS")
    print(f"BUILTIN_MODELS: {len(builtin)} 个")
    if not builtin:
        problems.append("没能从 app.py 解析出 BUILTIN_MODELS（正则失效？）")
    for name in builtin:
        if name not in known:
            problems.append(f"BUILTIN_MODELS 里的 {name} 未登记在 catalog.json")

    # 2) 去混响 / 去回声档位（dict 的值，形如 "a.ckpt" 或 "a.ckpt+b.ckpt"）
    dereverb = extract_dict_values(src, "DEREVERB_CHOICES")
    files: list[str] = []
    for value in dereverb:
        for part in value.split("+"):
            part = part.strip()
            if part.endswith((".ckpt", ".pth", ".onnx", ".yaml", ".th")):
                files.append(part)
    print(f"DEREVERB_CHOICES: {len(dereverb)} 个档位 -> {len(files)} 个模型文件")
    if not dereverb:
        problems.append("没能从 app.py 解析出 DEREVERB_CHOICES（正则失效？）")
    for name in files:
        if name not in known:
            problems.append(f"DEREVERB_CHOICES 引用的 {name} 未登记在 catalog.json")

    # 3) catalog.json 自身的完整性
    bad = [it for it in catalog.get("all", []) if not it.get("file")]
    if bad:
        problems.append(f"catalog.json 里有 {len(bad)} 个条目缺 file 字段")
    if not known:
        problems.append("catalog.json 的 all 列表是空的")

    print(f"catalog.json all: {len(known)} 个模型")

    if problems:
        print()
        print("发现问题：")
        for p in problems:
            print("  - " + p)
        return 1

    print()
    print("全部一致：代码里声明的模型都已登记在 catalog.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
