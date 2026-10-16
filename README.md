# VocalSeparator · 本地人声 / 伴奏分离工作站

> 把一首歌拆成「人声」「伴奏」等音轨，顺带把 RVC 翻唱最头疼的**混响 / 回声**也处理掉。
> **全部推理在本机 GPU 完成，音频不上传云端，模型全部本地内置，运行期完全离线。**

[![CI](https://github.com/456erddgr/VocalSeparator/actions/workflows/ci.yml/badge.svg)](https://github.com/456erddgr/VocalSeparator/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11%20x64-0078d4)](#环境要求)
[![Python](https://img.shields.io/badge/python-3.10%20~%203.12-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![单曲分离主界面](show/1.png)

---

## 目录

- [这是什么](#这是什么)
- [特性](#特性)
- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [怎么用](#怎么用)
- [效果档位怎么选](#效果档位怎么选)
- [人声去回声（RVC 翻唱前置）](#人声去回声rvc-翻唱前置)
- [参数建议](#参数建议)
- [一键安装包](#一键安装包)
- [目录结构](#目录结构)
- [文档](#文档)
- [常见问题](#常见问题)
- [已知限制](#已知限制)
- [致谢与许可](#致谢与许可)

---

## 这是什么

一个**本地运行**的人声 / 伴奏分离工具。核心能力由 [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
提供，这个项目在它之上做了三件它没有的事：

1. **一条完整的桌面产品链路** —— 拖进去就出结果，带实时进度、任务台账、暂停续跑、
   自动打包下载、下载后自动清理本机副本。不用碰命令行。
2. **离线化加固** —— 模型全部预置在 `models\`，并且**用代码堵死了联网下载**：
   一旦缺少权重，程序会明确报错告诉你缺哪个文件，而不是悄悄走网络。
3. **去回声 / 去混响的工程化收尾** —— 去混响模型普遍会把整轨电平压小、把 8kHz 以上
   空气感滤掉，直接拿去跑 RVC 会「又小又闷」。这里做了等响补偿 + 1/3 倍频程频谱倾斜
   匹配，把产物拉回和原曲人声一致。

---

## 特性

| 能力 | 说明 |
| --- | --- |
| **单曲分离** | 上传一首歌，选模型 / 预设，出人声、伴奏等主干，在线试听 + 下载 |
| **批量分离** | 一次丢多个文件或整个文件夹，连续处理并打包 zip |
| **多模型集成** | 3 个内置组合预设（无需下载），也可自由勾选 2 个以上模型配融合算法 |
| **去回声 / 去混响** | 两段式：先分离，再**只对「人声」轨**做去混响 / 去回声，含等响 + 频谱倾斜修正 |
| **真实进度** | hook `audio-separator` 内部 tqdm，给出真实分段进度与 ETA，进度单调不减 |
| **任务台账** | 后台线程 + JSON 落盘，刷新页面 / 切标签都不丢，能看到运行中与历史任务 |
| **暂停 / 继续** | 每个任务独立的等待闸门，可中途暂停再续跑 |
| **全局串行锁** | 同一时刻只有一个任务真正推理，避免连点两次并发抢显存 |
| **完全离线** | `HF_HUB_OFFLINE=1` + 代码级下载拦截，运行期零网络请求 |
| **模型库 / 环境自检** | 界面内查看全部模型 SDR 评分；GPU / torch / CUDA / ONNX / FFmpeg 一键自检 |

---

## 快速开始

### 方式 A：下载一键安装包（普通用户推荐）

1. 到 [**Releases**](https://github.com/456erddgr/VocalSeparator/releases) 下载最新版的
   `VocalSeparator-Setup.exe` **和全部 `payload.7z.00N` 分卷**。

   > ⚠️ **分卷必须下全**。因为 release 单个文件上限 2 GB，8 GB 的 payload 被切成
   > `payload.7z.001` … `payload.7z.00N` 若干卷，**少下载任意一卷都会解压失败**。
   > 浏览器批量下载时容易漏，建议下完先核对卷数与文件大小（Release 说明里有每卷字节数）。

2. 把 `VocalSeparator-Setup.exe` 和所有分卷**放在同一个文件夹**里（不要改名、不要解压分卷）。
3. 双击 `VocalSeparator-Setup.exe` → 点「开始安装」。
4. 装完桌面会出现 `VocalSeparator` 快捷方式，双击即用。

> 安装包内含自带 Python 运行时与全部模型权重，**装完不需要联网**，也不需要单独装 Python。
> 安装程序会自动发现分卷并把 `.001` 交给 7-Zip 解压；不需要你手动合并或改名。

### 方式 B：从源码运行（开发者 / 想自己改）

```powershell
git clone https://github.com/456erddgr/VocalSeparator.git
cd VocalSeparator

# 1) 自检：看看缺什么
python scripts\setup.py

# 2) 装依赖（会创建 venv，先装 cu128 版 torch，再装其余依赖）
python scripts\setup.py --install

# 3) 补齐不在仓库里的大文件：FFmpeg 与模型权重
python scripts\fetch_assets.py list     # 先看看要下什么
python scripts\fetch_assets.py ffmpeg   # 下载 FFmpeg 到 bin\

# 4) 启动
start.bat
```

Windows 用户也可以直接双击 **`一键安装.bat`**，逻辑与 `setup.py --install` 相同。

---

## 环境要求

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| 系统 | Windows 10 / 11 64 位 | 停止脚本用 `Get-NetTCPConnection`，仅 Windows 有效 |
| Python | 3.10 ~ 3.12（推荐 3.12） | 仅源码方式需要；安装包装的是自带运行时 |
| 显卡 | NVIDIA 显卡 + 较新驱动（推荐 CUDA 12.8 以上） | 没有显卡也能跑，回退 CPU，速度慢很多 |
| 显存 | 建议 8 GB 以上 | 作者测试机为 RTX 5080 Laptop 16 GB |
| 磁盘 | 源码约 5 MB + 模型约 7 GB + FFmpeg 约 200 MB + venv 约 5 GB | 留 15 GB 余量比较稳 |
| 网络 | **仅首次安装依赖需要** | 装完运行期完全离线 |
| 权限 | 普通用户即可，**不需要管理员权限** | 所有注册表项都写在 `HKCU` |

---

## 怎么用

启动后浏览器打开 `http://127.0.0.1:7861`（只监听回环地址，局域网访问不到）。

界面上有 5 个标签页：

| 标签页 | 用途 |
| --- | --- |
| **单曲分离** | 拖一首歌进去，选模型 / 预设，点开始分离 |
| **批量分离** | 一次丢多个文件或整个文件夹，排队连续处理 |
| **任务列表** | 运行中与历史任务，可暂停 / 继续 / 查看详情 / 导出 |
| **模型库** | 全部模型的 SDR 评分、主干定义、内置状态 |
| **环境** | GPU / torch / CUDA / ONNX Runtime / FFmpeg 自检，性能设置与实测对比 |

**典型流程**：

```
拖入音频 → 选分离模式（单模型 / 内置集成 / 自定义）
        → （可选）展开「人声去回声」，选模型
        → 点「开始分离」→ 看进度 → 下载人声 / 伴奏
```

![分离模式与模型选择](show/2.png)

![输出与参数](show/3.png)

---

## 效果档位怎么选

### 单模型

| 场景 | 推荐模型 | 说明 |
| --- | --- | --- |
| RVC 翻唱人声（默认） | **Revive V3e** | 内置里人声最干净的一档 |
| 想要更快 | Mel-Band gabox | 约快 2.5 倍，但更偏伴奏 |
| 人声 + 伴奏都要 | InstVoc Duality V2 | 两头兼顾 |
| 只要伴奏、不要人声 | Viperx | 伴奏专用 |

### 内置集成组合（推荐，成员全为本地模型）

多模型频谱融合，实测优于任意单模型：

| 预设 | 成员 | 融合算法 | 特点 |
| --- | --- | --- | --- |
| `builtin_vocal_revive`（默认） | Revive V3e + BS-Roformer 1297 | `avg_fft` | 人声极致干净 |
| `builtin_vocal_dual` | BS-Roformer 1297 + 1296 | `avg_fft` | 稳定干净，速度快于三合一 |
| `builtin_vocal_triple` | 1297 + 1296 + InstVoc Duality V2 | `avg_fft` | 细节更饱满，更慢 |

> 官方社区预设（`vocal_full` / `instrumental_full` / `karaoke` 等）依赖额外模型权重，
> 仓库与安装包内**未内置**这些权重，离线模式下不会出现在列表中；
> 把对应权重放进 `models\` 后会自动出现。

### 自定义集成

自由勾选 2 个以上内置模型，配 `avg_fft` / `min_fft` / `max_fft` / `uvr_max_spec`
等融合算法。各算法的差别见 [docs/原理与架构.md](docs/原理与架构.md)。

---

## 人声去回声（RVC 翻唱前置）

**强烈建议做 RVC 翻唱时开启。** 带混响 / 回声的人声会让 RVC 把残响也一起学成人声，
成品出现闷糊、回声叠加。

- 执行顺序是**先分离、再只对分离出的「人声」轨做去混响 / 去回声**（两段式），
  伴奏等其它主干完全不受影响。
- 产物里会多出一份 `歌名_人声去回声.wav`，**直接用这份做 RVC 输入**。
- 代价是与分离同量级的一遍推理（只处理人声轨），长音频请预留时间。

### 模型怎么选

| 模型 | 类型 | 说明 |
| --- | --- | --- |
| **De-Reverb-Echo V2**（推荐） | 去回声 | Sucial 系列，真实混响 + 回声场景实测最强 |
| Fused | 去回声 | 小混响 + 大混响通吃，官方推荐融合版 |
| 标准 anvuew（SDR 19.17） | **只去混响** | 对离散回声（延迟、回声尾巴）几乎无效 |
| 轻柔 anvuew（SDR 18.81） | **只去混响** | 混响轻或怕伤高频时用 |
| 双模型串联 V2 → Fused | 去回声 | 最干净，耗时约翻倍 |

> 关键结论（作者实测）：**anvuew 系列是「去混响」，不是「去回声」。**
> 面对真实带混响 + 回声的人声，它的 SDR 只有 15.1；而 Sucial 的 De-Reverb-Echo
> 系列在同一素材上是 24.3，且不掉电平、不掉高频。选错模型会白跑一遍。

### 自动等响 + 频谱倾斜匹配（默认开启，建议保留）

去混响模型会把人声整体压小（实测约 4 dB），并且可能滤掉 8 kHz 以上的空气感。
不做补偿，听感就是「又小又闷、和原曲对不上」。开启后：

- 产物与原曲人声**等响**（实测 RMS 比约 1.00）；
- 宽带频谱包络按 1/3 倍频程拉回与输入人声一致，修「高低频对不上」。

### 去回声遍数

默认 **1 遍**。实测 2 遍对回声的额外削减极小（-8.48 → -8.52 dB）却让耗时翻倍，
所以默认锁定 1 遍；只有残留回声仍然明显时才手动选 2 遍。

---

## 参数建议

以 16 GB 显存为例：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `overlap` | **4** | 分段重叠窗口倍数（2~64）。**对耗时影响最大：8 → 32 约 4 倍慢** |
| `batch_size` | 8 | 越大越快，用显存换速度，16 GB 可拉到 12~16 |
| `segment_size` | 0 | 0 = 跟随模型原生（Roformer 为 256），一般不动 |
| Demucs `shifts` | 4 | 多轨模型（鼓 / 贝斯 / 钢琴）精度，越高越慢 |
| 输出格式 | WAV | 无损首选；FLAC 省空间；MP3 便于分享 |

> **实测结论**：CPU 线程数、`cudnn.benchmark`、`batch_size` 这几项差异 **小于 15%**，
> 因为瓶颈在 GPU 推理本身。真正影响最大的是 `overlap`。建议保持默认 `overlap = 4`
> —— 它与 Roformer 的训练配置一致；想要更保守的边缘处理可调到 8，代价是线性变慢。

多轨拆分（人声 / 鼓 / 贝斯 / 其他 / 吉他 / 钢琴）走 Demucs v4：
`htdemucs_ft.yaml`（四轨）、`htdemucs_6s.yaml`（六轨），需把 `.yaml` 与同名 `.th`
一起放进 `models\`。

---

## 一键安装包

仓库**不含**模型与安装包本体（GitHub 单文件上限 100 MB），需要自己构建或直接从
[Releases](https://github.com/456erddgr/VocalSeparator/releases) 下载。

自己构建安装包：

```powershell
# 1) 组装绿色包（把仓库代码 + 本机 models/ bin/ runtime/ 合到一起）
python scripts\pack_release.py assemble --out D:\VocalSeparator-green

# 2) 生成安装程序用的 payload（本机 payload.7z 约 8 GB，耗时较长）
python scripts\pack_release.py payload --src D:\VocalSeparator-green

# 2b) 要发到 GitHub Release 就加分卷（单文件必须 ≤ 2 GB，否则会被拒绝）
python scripts\pack_release.py payload --src D:\VocalSeparator-green --volume 1900m

# 3) 生成校验清单（贴到 Release 说明里）
python scripts\pack_release.py manifest

# 或者一次全做
python scripts\pack_release.py all --out D:\VocalSeparator-green
```

> **为什么需要分卷**：GitHub Release 单个资产上限 2 GB，而完整 payload 约 8~13 GB。
> 用 `--volume 1900m` 会产出 `payload.7z.001` / `.002` / … 一串分卷，**每一个都是发布资产**。
> 用户必须下全所有卷并放在同一目录，`installer_web.py` 会把 `.001` 交给 7-Zip，
> 7-Zip 自动按序接续读取其余卷 —— 传 `.002` 或其他任意一卷都读不到数据。
> 分卷数随体积增长；公开的 v1.1.0 Release 为 7 卷（6×1900 MB + 1×1441 MB，共约 12.74 GB）。

分发给别人时，把 `installer_web.py`（打包成 `VocalSeparator-Setup.exe`）
与 `payload.7z` **或全部分卷**放在同一个文件夹即可。

安装程序会：解压 payload 到 `%LOCALAPPDATA%\VocalSeparator` → 把 `venv\pyvenv.cfg`
的 `home` / `executable` 指向包内自带的 `runtime\python311` → 自检依赖（import torch / gradio）
→ 建桌面与开始菜单快捷方式 → 写卸载注册表项（全在 `HKCU`，不需要管理员）。
作者机器上实测约 **150 ~ 160 秒**。

详细打包说明与离线安装变体见 [docs/部署与打包.md](docs/部署与打包.md)。

---

## 目录结构

```
VocalSeparator\
├─ app.py                   主程序：Gradio 界面 + 分离引擎 + 任务调度
├─ launcher.py              桌面启动器（打包为 VocalSeparator.exe）
├─ installer_web.py         一键安装程序（WebView2 界面）
├─ uninstall.py             卸载程序
├─ start.bat                启动脚本（浏览器模式）
├─ 停止 VocalSeparator.bat  关闭正在运行的服务
├─ 一键安装.bat              Windows 一键装依赖
├─ catalog.json             模型目录（169 条，界面读取）
├─ requirements.txt         依赖清单
├─ assets\                  图标资源（vs.ico / vs.png）
├─ ui\installer.html        安装程序界面
├─ show\                    界面截图
├─ docs\                    完整文档
├─ scripts\
│   ├─ setup.py             环境自检 / 依赖安装助手
│   ├─ fetch_assets.py      下载 FFmpeg 与模型权重
│   └─ pack_release.py      打包绿色包 / payload / 校验清单
├─ models\                  【不入库】本地模型权重，约 7 GB
├─ bin\                     【不入库】ffmpeg.exe / ffprobe.exe，约 200 MB
├─ runtime\                 【不入库】自带 Python 3.11 运行时
├─ venv\                    【不入库】虚拟环境，首次安装生成
└─ output\                  【不入库】分离结果
```

---

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/原理与架构.md](docs/原理与架构.md) | 分离管线、STFT 分块推理、融合算法、并发设计、去回声后处理数学 |
| [docs/使用手册.md](docs/使用手册.md) | 每个界面控件、档位选择、去回声工作流、任务管理 |
| [docs/部署与打包.md](docs/部署与打包.md) | 三种部署方式、模型与 FFmpeg 获取、打包成安装包、离线安装 |
| [docs/常见问题.md](docs/常见问题.md) | 装了打不开、CUDA 不可用、显存不足、声音变小变闷…… |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 开发环境、代码风格、怎么加模型 / 加去回声档位 |

---

## 常见问题

**Q：提示「模型未内置」怎么办？**
该权重不在 `models\` 目录中。本工具**完全离线、不会联网下载**（这是刻意用代码堵死的）。
把权重文件放进 `models\` 后重启即可，界面下拉只列已内置的模型。

**Q：分离很慢，显卡占用不高？**
先确认 `overlap` 是默认的 4。再去「环境」页点「实测对比」，看本机各项差异。
若显示 `CUDA 可用：False`，说明驱动过旧或装的是 CPU 版 torch。

**Q：去完之后人声又小又闷？**
确认「自动等响 + 频谱倾斜匹配」是开着的（默认开）。若选了 anvuew 系列去混响模型，
说明**模型类型选错了** —— 去回声请用 De-Reverb-Echo 系列。

**Q：输出文件夹在哪？**
包目录下的 `output\`。在页面上下载结果后，程序会自动把本机那份移入回收站，
避免双份占空间（不会硬删除，可还原）。

更多见 [docs/常见问题.md](docs/常见问题.md)。

---

## 已知限制

- **仅 Windows。** 停止脚本与启动器依赖 Windows 特有 API（`Get-NetTCPConnection`、
  `CREATE_NO_WINDOW`、WebView2）。`app.py` 本身跨平台，但未经测试。
- **界面语言为中文**，暂无国际化。
- 模型与 FFmpeg 不入库，首次搭建需要自行获取（见 `docs/部署与打包.md`）。
- `catalog.json` 登记 169 个模型，但仓库内置权重只覆盖其中 **8 个**（`app.py` 的
  `BUILTIN_MODELS`）；其余档位需要自行补齐权重才会出现在界面里。
- 作者实测数据均基于 **RTX 5080 Laptop 16 GB** 单机，你的机器数字会不同，
  请以「环境」页的实测对比为准。

---

## 致谢与许可

本项目的分离能力来自这些优秀开源项目，特此致谢：

- [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) —— 分离引擎
- [PyTorch](https://github.com/pytorch/pytorch) / [Gradio](https://github.com/gradio-app/gradio) / [pywebview](https://github.com/r0x0r/pywebview) / [FFmpeg](https://ffmpeg.org/)
- 模型权重作者：UVR 社区、unwa、aufr33、viperx、Gabox、Sucial、anvuew 等

本项目以 **MIT** 许可开源，详见 [LICENSE](LICENSE)。
**注意**：MIT 只覆盖本仓库的代码 —— 模型权重不包含在本仓库中，其许可取决于各权重作者，
使用与再分发前请自行核对。第三方组件与权重的完整许可清单见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
