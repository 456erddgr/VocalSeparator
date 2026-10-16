# 第三方组件与模型许可说明

本仓库的**代码**（`app.py`、`launcher.py`、`installer_web.py`、`scripts/` 等）以
[MIT](LICENSE) 发布 —— 详见根目录 `LICENSE`。

但本程序的分离能力并非自研：它是「**外壳 + 调度 + 后处理**」，真正的推理来自下列
第三方项目。它们各自的许可与著作权归原作者所有，本项目仅以依赖形式调用，**未修改其源码**。

---

## 一、运行时依赖

| 组件 | 许可 | 项目地址 |
| --- | --- | --- |
| audio-separator | MIT | <https://github.com/nomadkaraoke/python-audio-separator> |
| PyTorch / torchaudio / torchvision | BSD-3-Clause | <https://github.com/pytorch/pytorch> |
| Gradio | Apache-2.0 | <https://github.com/gradio-app/gradio> |
| pywebview | BSD-3-Clause | <https://github.com/r0x0r/pywebview> |
| FFmpeg | LGPL / GPL（取决于编译选项） | <https://ffmpeg.org/> |
| Demucs | MIT | <https://github.com/facebookresearch/demucs> |

> FFmpeg 以**可执行文件**形式随一键安装包分发（`bin\ffmpeg.exe`、`bin\ffprobe.exe`）。
> 该构建来自 <https://www.gyan.dev/ffmpeg/builds/> 的 essentials 版，其许可随构建而变，
> 再分发前请核对上游说明。

---

## 二、模型权重

`.ckpt` / `.pth` 模型权重**不包含在本仓库中**（体积原因，走 GitHub Release 单独分发）。
它们的著作权与许可取决于各自作者：

| 系列 | 作者 / 来源 |
| --- | --- |
| BS-Roformer / Mel-Band Roformer 系列 | UVR 社区，以及 unwa、aufr33、viperx、Gabox 等 |
| De-Reverb-Echo / DeReverb 系列 | Sucial、anvuew 等 |

**使用、再分发或商用这些权重前，请自行核对其原始发布页的许可条款。**

---

## 三、一句话总结

MIT 只覆盖**本仓库的代码**。它**不会**授予你任何第三方库或第三方模型权重的额外权利 ——
那些仍然由各自的原作者与原始许可决定。
