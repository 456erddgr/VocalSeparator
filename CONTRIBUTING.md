# 贡献指南

感谢你愿意为 **VocalSeparator（人声分离器）** 做贡献。本文说明**怎么把环境跑起来**、**代码风格**、**三种最常见的扩展怎么做**，以及 **PR 检查清单**。

> 本项目面向**中文用户**、**Windows**、**完全离线**三个前提。改动时请优先保持这三点不被破坏。

---

## 目录

- [从哪里开始](#从哪里开始)
- [开发环境搭建](#开发环境搭建)
- [代码风格](#代码风格)
- [项目结构与模块职责](#项目结构与模块职责)
- [扩展一：往 catalog.json 加模型](#扩展一往-catalogjson-加模型)
- [扩展二：往 BUILTIN_MODELS 加一个内置模型](#扩展二往-builtin_models-加一个内置模型)
- [扩展三：新增一个去回声选项](#扩展三新增一个去回声选项)
- [其它常见扩展点](#其它常见扩展点)
- [硬性规则](#硬性规则)
- [测试与自检](#测试与自检)
- [PR 检查清单](#pr-检查清单)
- [提交 Issue](#提交-issue)

---

## 从哪里开始

适合新手的入手点，从易到难：

1. **文档纠错 / 补 FAQ** — `docs/` 下任意文件；
2. **给 `catalog.json` 补模型条目** — 纯 JSON，不影响代码；
3. **补模型标签** — 「推荐 / 速度 / 擅长轨」这类提示文案；
4. **模型库 / 环境页的界面文案与布局**；
5. **新的去回声后处理策略**。

改之前建议先提 Issue 说明方案，避免白做。

---

## 开发环境搭建

完整步骤见 [部署与打包](docs/部署与打包.md#方式三从源码部署开发者)，这里是要点：

```bat
git clone <repo-url>
cd VocalSeparator
python -m venv venv

REM 1) 必须先单独装 CUDA 版 torch（PyPI 上没有 +cu128 版本）
venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
venv\Scripts\python.exe -m pip install torch torchaudio torchvision ^
    --index-url https://download.pytorch.org/whl/cu128

REM 2) 再装其余依赖
venv\Scripts\python.exe -m pip install -r requirements.txt

REM 3) 自检
venv\Scripts\python.exe scripts\setup.py
venv\Scripts\python.exe scripts\setup.py --check-models

REM 4) 启动
venv\Scripts\python.exe app.py
```

还需要本地放好 `bin\ffmpeg.exe`、`bin\ffprobe.exe` 和至少一个 `models\*.ckpt`，否则跑不了真实任务。

> **不要提交 `venv/`、`models/`、`bin/`、`runtime/`、`dist_send/`** —— 它们在 `.gitignore` 里，且体积远超 GitHub 单文件上限。

---

## 代码风格

### 中文注释是本项目的规范

**保留中文注释**。这个项目的注释解释的是「为什么这么做」——尤其是那些踩过坑才写下的取舍（比如为什么去回声后处理**不做**窄带增强、为什么 `restore_air` 对新模型默认关闭）。这类注释比代码本身更有价值。

- 改代码时，**不要顺手把中文注释翻译成英文**，也不要把它们删成「代码自解释」；
- 新增逻辑时，如果做了非显然的取舍，**请写清楚原因**（「因为 X 会导致 Y，所以选 Z」），而不只是描述代码在做什么。

### 界面文案

- 全部用**简体中文**，风格与现有文案一致；
- 提示要**可执行**：说清楚「怎么办」，而不只是「出错了」。

### 一般约定

- **Python**：4 空格缩进，`snake_case` 函数/变量，常量用 `UPPER_SNAKE_CASE`；
- **不要引入新的重量级依赖**。`requirements.txt` 每多一项，安装包就大一截、离线部署就多一个坑。加依赖请在 PR 里说明必要性；
- **`requirements.txt` 必须保持纯 ASCII** —— pip 会用系统区域设置解码它，混入非 ASCII 字符在某些中文 Windows 上会直接报错（文件里有注释说明）；
- 路径拼接一律用 `pathlib` / `os.path.join`，注意 Windows 下中文路径。

---

## 项目结构与模块职责

| 文件 | 职责 |
| --- | --- |
| `app.py` | **主程序**：Gradio 界面 + 推理引擎 + 任务台账（约 5200 行） |
| `launcher.py` | 桌面启动器，打包为 `VocalSeparator.exe` |
| `installer_web.py` | WebView2 图形安装程序 |
| `uninstall.py` | 卸载程序 |
| `catalog.json` | 模型目录（169 个条目 / 8 个分类桶 / 9 个社区预设） |
| `scripts/setup.py` | 环境自检与安装助手 |
| `scripts/pack_release.py` | 打绿色包 / payload.7z / 校验清单 |
| `scripts/fetch_assets.py` | 下载 FFmpeg 与模型 |

`app.py` 里几个关键区域（改之前先读一遍）：

| 行号附近 | 内容 |
| --- | --- |
| `app.py:52-240` | 性能设置、离线守卫、tqdm hook |
| `app.py:320` | `TaskLedger` 任务台账 |
| `app.py:475` | `SeparationProgress` |
| `app.py:645-654` | `BUILTIN_MODELS` |
| `app.py:661-674` | `DEREVERB_CHOICES` |
| `app.py:890-1050` | `make_separator` / `load_models` / `friendly_rename` |
| `app.py:1091-1255` | `_spectral_tilt_match` / `_dereverb_postprocess` / `run_dereverb` |
| `app.py:1277-1478` | `_run_single_task` / `_run_batch_task` |
| `app.py:1632-1723` | 暂停 / 续跑闸门 |
| `app.py:2190+` | 轮询与会话级去重缓存 |
| `app.py:2590-4157` | Gradio 界面定义 |
| `app.py:4682-4810` | `progress_html` |

> 行号会随改动漂移，**以函数/常量名为准**。

---

## 扩展一：往 `catalog.json` 加模型

`catalog.json` 是**纯数据**，改它不需要动 Python。

### 结构

顶层大致是：

```jsonc
{
  "models": [ /* 169 个条目 */ ],
  "buckets": { /* 8 个分类桶 */ },
  "presets": { /* 9 个社区预设 */ }
}
```

> **待确认**：各字段的确切拼写请以文件现有条目为模板照抄 —— 加条目时**复制一个同类条目再改**，比照文档写更安全。

### 步骤

1. 打开 `catalog.json`，找一条**同类**的现有条目复制；
2. 改这些字段：
   - **文件名 / 权重文件名**：必须与放进 `models\` 的文件名**逐字符一致**（大小写、下划线、扩展名）；
   - **显示名**：界面上看到的名字，中文要自然；
   - **分类**：归到合适的 bucket；
   - **标签**：标注「推荐 / 速度 / 擅长轨」等——这是用户最主要的决策依据，请认真写；
   - **下载来源 / 主页**（若有该字段）：填**原始**来源，方便用户自己准备权重。
3. 在本地验证 JSON 合法：

   ```bat
   venv\Scripts\python.exe -c "import json;json.load(open('catalog.json',encoding='utf-8'));print('ok')"
   ```

4. 启动程序，在**模型库**页确认新条目出现、分类正确、文案没串行。

### 注意

- **加入目录 ≠ 权重在本地**。目录只是「告诉用户有这么个模型」，选了还是要用户自己放权重。请在条目文案里说清楚；
- **不要往仓库提交权重文件**（单文件常超 100 MB，必被 GitHub 拒绝）；
- 社区预设（`vocal_full` / `instrumental_full` / `karaoke` 等）依赖**非内置权重**，加预设时确认用户拿得到这些权重。

---

## 扩展二：往 `BUILTIN_MODELS` 加一个内置模型

「内置」= **能在单模型下拉里直接选，且预期随发布分发的模型**。

### 步骤

1. **把权重放进 `models\`**（本地验证用）。文件名保持官方原名，别改名——改名会让别人按官方名字放文件时对不上。

2. **加到 `BUILTIN_MODELS`**（`app.py:645-654`）：

   ```python
   BUILTIN_MODELS = {
       # ... 现有的 8 个
       "your_model_file.ckpt": "显示名 · 一句说明",
   }
   ```

   - key 是**文件名**，必须与 `models\` 里的完全一致；
   - value 是界面显示文案，沿用现有风格（用 `·` 分隔名称与说明）。

3. **同步 `catalog.json`**（见上一节），否则模型库页看不到它。

4. **验证**：
   - 下拉里出现该模型且能选中；
   - 选它跑一首曲子，输出正常；
   - **故意把权重改名**跑一次，确认报错是那句清楚的「本地缺少模型文件 XXX，请放入 models\ 后重试」，而不是 traceback。

5. **考虑是否要进集成预设**。只有在它单独跑得通、且与其他模型有互补性时才加。

### 附带说明

- 需要配套 `.yaml` 的模型（Mel-Band Roformer 有些是），**两个文件都要说明**，别只写 `.ckpt`；
- 多轨 Demucs 的 `.yaml` + `.th` **必须成对同名**；
- 新内置模型会**增加发布包体积**，PR 里请说明大概多大。

---

## 扩展三：新增一个去回声选项

去回声是**两段式**的：第一遍在人声上做，分离出干净的干声；必要时第二遍再处理。相关常量在 `app.py:661-674`。

### 步骤

1. **加选项** 到 `DEREVERB_CHOICES`：

   ```python
   DEREVERB_CHOICES = [
       # ... 现有 5 个
       ("你的选项显示名", "your_dereverb_model.ckpt"),
   ]
   ```

   顺序有讲究：**第一项是默认**，通常放推荐的那个。

2. **确认 `DEREVERB_PRIMARY_STEM` 是否正确**。它决定后处理作用在**人声**还是**伴奏**上。去回声几乎总是人声 → 保持人声。

3. **双模型串联**这种复合选项：`DEREVERB_CHOICES` 的 value 需要能表达「先 A 后 B」。**待确认**：具体怎么表达（列表？特殊分隔符？）请直接读 `run_dereverb`（`app.py:1225+`）确认，不要猜。

4. **验证清单**：
   - 选项在界面出现、默认项正确；
   - 选它跑一首有明显回声的曲子，输出**确实更干**；
   - 输出的**响度**没有异常（会被 RMS 匹配补偿，钳在 `[0.2, 12.0]`）；
   - 高频**没有变闷**（`restore_air` 的默认值是否适合这个模型）；
   - **没有把回声重新抬回来**（后处理的 `_spectral_tilt_match` 故意不做窄带增强，就是防这个）。

5. **在 PR 描述里写清楚**：
   - 模型来源与许可；
   - 它擅长**去混响**还是**去离散回声**（两者完全不同，anvuew 系列是前者）；
   - 大概多慢（比单模型多几倍时间）；
   - 你有没有对比过现有选项。

> **重要**：请勿把「去混响」模型放到推荐位当成「去回声」用。这是本项目已经被用户踩过的坑。

---

## 其它常见扩展点

- **后处理参数**：响度钳制范围、`_spectral_tilt_match` 的 `max_db`（默认 6.0）、`f_lo`（180 Hz）/ `f_hi`（16 kHz）——都在 `app.py:1091-1255`。**改这些会明显影响音色，请附 A/B 试听说明**。
- **性能参数默认值**：`overlap`、`batch_size`（默认 8）、`segment_size`（256）等。**注意 `overlap` 有一个默认值矛盾尚未收敛**：`app.py:4871` 与 `app.py:4978` 的代码是 `gr.Slider(2, 64, value=4, ...)`（实际默认 **4**），但 `app.py:18` 的改造说明写「overlap 默认值 32 -> 8」，仓库根 `README.md:39` 写「默认 32」，`SECONDS_PER_AUDIO_SECOND`（`app.py:472`）的注释自述是「overlap=4 + native fp16」。**改这个参数前请先把口径统一**，否则文档会互相打架。另外实测结论是：CPU 线程 / `cudnn.benchmark` / `batch_size` 差异 **< 15%**，而 `overlap` **主导成本**（8→32 约 4 倍慢）。改默认值要有数据支撑。
- **进度条**：`install_tqdm_hook()` 把 `audio_separator.separator.architectures.mdxc_separator` 里的 tqdm 换成子类，把 `(n, total)` 报到当前任务。要动进度显示，先读这一段。
- **界面**：`app.py:2590-4157`。改布局注意轮询组件的去重缓存（`_cached_upd` / `_cached_choices`）——直接改组件值而不走缓存，可能引入闪烁。

---

## 硬性规则

这几条是红线，PR 违反会被要求修改：

### 1. 运行期不得联网

- `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 必须保持；
- **`install_offline_guard()` 必须保留**：它把 `Separator.download_file_if_not_exists` 改成**抛异常**而不是下载（`app.py:184-215`）。删掉它 = 程序会在用户不知情时联网下载；
- 新增任何会发起网络请求的代码，都必须显式说明理由并在 PR 里讨论。

### 2. 不得提交大文件

`models/`、`bin/`、`runtime/`、`venv/`、`dist_send/` 一律不进 Git。

### 3. 端口与绑定

- 端口固定 **7861**，只绑 **127.0.0.1**；
- 文档里**不要写 7860**（旧部署说明里的 7860 是错的）。

### 4. 并发模型不要破坏

- `RUN_LOCK` 保证**同一时刻只有一个任务在推理**；
- 后提交的任务状态是 `排队`，文案「排队中 · 前一个任务完成后自动开始」；
- `demo.queue(default_concurrency_limit=4)` 是为了让轮询在推理期间仍然响应 —— 别调成 0/1；
- 任务跑在**后台线程**，状态落 `tasks.json`，必须**刷新页面不丢**。

### 5. 删除走回收站

清理本地副本必须经 `_to_recycle_bin`（`SHFileOperationW` + `FOF_ALLOWUNDO`），**不要改成 `os.remove`**。

### 6. 不要破坏增量进度

进度**单调不减**。暂停时允许停在当前值，但不允许倒退。

---

## 测试与自检

提交前至少跑一遍：

```bat
REM 环境自检
venv\Scripts\python.exe scripts\setup.py
venv\Scripts\python.exe scripts\setup.py --check-models

REM 语法检查（app.py 很大，语法错误不容易肉眼发现）
venv\Scripts\python.exe -m py_compile app.py launcher.py installer_web.py uninstall.py scripts\setup.py scripts\pack_release.py scripts\fetch_assets.py

REM JSON 合法性
venv\Scripts\python.exe -c "import json;json.load(open('catalog.json',encoding='utf-8'));print('catalog ok')"
```

**手工验证**（针对你的改动）：

| 改动类型 | 至少验证 |
| --- | --- |
| 新增模型 | 下拉可选；跑一首 60 秒片段成功；缺文件时报错清楚 |
| 新增去回声 | A/B 试听；响度正常；高频没变闷；时间可接受 |
| 改界面 | 刷新页面任务不丢；进度不跳变；三个 tab 都点一遍 |
| 改并发/台账 | 连续提交 3 个任务看到排队；暂停/续跑；刷新页面 |
| 改清理逻辑 | 下载一个文件，确认进回收站且可还原 |
| 改打包脚本 | `pack_release.py manifest` 能跑通 |

> 项目目前**没有自动化测试套件**。所以 PR 里请写清楚**你手工验证了什么、怎么验证的**。

---

## PR 检查清单

复制到你的 PR 描述里逐条勾选：

```markdown
## 改动说明
- 做了什么：
- 为什么：

## 检查清单
- [ ] 只改了与本次目的相关的文件（没有顺手大范围格式化）
- [ ] 保留并正确更新了中文注释
- [ ] 未在运行期引入任何网络请求；`install_offline_guard()` 未被削弱
- [ ] 未提交 `models/` `bin/` `runtime/` `venv/` `dist_send/` 等大文件
- [ ] `requirements.txt` 仍为纯 ASCII；未无谓新增依赖
- [ ] 未改端口（7861）与绑定地址（127.0.0.1）
- [ ] 未破坏 `RUN_LOCK` 串行语义与 `tasks.json` 台账
- [ ] 删除路径仍走 `_to_recycle_bin`（回收站）
- [ ] 进度仍单调不减
- [ ] 已跑 `py_compile` 与 `setup.py` 自检
- [ ] 已在 PR 中说明**手工验证步骤与结果**
- [ ] 相关文档已同步更新（`README.md` / `docs/`）

## 涉及模型/权重（如有）
- 模型名：
- 来源与许可：
- 文件大小：
- 是否需要用户自行准备权重：
```

### 文档改动额外要求

- `docs/` 里的技术论断必须**能在代码里找到依据**，引用时给**文件名 + 行号**（例：`app.py:1091`）；
- 拿不准的地方写「**已知限制 / 待确认**」，不要编；
- 不要**编造**没实测过的性能数字。目前唯一实测结论是：CPU 线程 / `cudnn.benchmark` / `batch_size` 差异 < 15%，`overlap` 8→32 约 4 倍慢；
- 截图统一放 `show/`，文档里用相对路径引用（如 `![单曲分离](../show/1.png)`）。

---

## 提交 Issue

**Bug 报告**请附：

1. `logs\` 里的相关日志（脱敏后）；
2. `python scripts\setup.py` 的输出；
3. `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 的输出；
4. 显卡型号 / 驱动版本 / 显存；
5. 模型或预设名、`overlap`、`batch_size`；
6. 复现步骤（能附 30~60 秒的音频片段最好，注意版权）。

**功能请求**请说明使用场景——「我要做什么」比「我要什么按钮」更有用。

---

## 相关文档

- [部署与打包](docs/部署与打包.md) — 环境、模型、打包发布
- [原理与架构](docs/原理与架构.md) — 内部实现细节
- [使用手册](docs/使用手册.md) — 用户视角的功能说明
- [常见问题](docs/常见问题.md) — 排查清单
