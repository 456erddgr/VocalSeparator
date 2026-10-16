# -*- coding: utf-8 -*-
"""
VocalSeparator · 本地音乐人声/伴奏分离工作站

本机环境：
    E:\\VocalSeparator\\
        app.py            <- 本文件
        start.bat         <- 双击启动
        venv\\             <- Python 3.11 虚拟环境（torch cu128 / audio-separator / gradio）
        models\\           <- 本地内置分离模型（离线可用，不走网络下载）
        output\\           <- 分离结果
        tasks.json        <- 任务台账（JSON / UTF-8 落盘）
        bin\\ffmpeg.exe   <- 音频编解码

设计取向：效果优先，不吝惜显存与时间。

本轮改造要点：
    1) overlap 默认值 32 -> 8（单曲页 / 批量页），滑杆标签标注 标准/精细/极致；
    2) 新增默认关闭的高级开关 use_native_fp16 / use_torch_compile（透传 Separator）；
    3) 分离阶段显示真实进度：hook audio_separator 内部 tqdm，把 (n / total) 映射到
       0.55-0.90 区间并给出 ETA，进度单调不减；取不到时按音频时长估算；
    4) 后台线程 + JSON 任务台账：提交即入队，前端每 2s 轮询任务列表，
       刷新页面 / 切换标签后依然能看到运行中与历史任务。

第二轮改造要点：
    5) 修轮询失效：全局只挂一个 gr.Timer（写在 Blocks 顶层、正常挂载），
       同一个 tick 同时驱动单曲 / 批量 / 任务列表三处 poll，
       页内进度条与顶部「运行中 N 个任务」实时刷新；
    6) 全局串行锁 RUN_LOCK：同一时刻只允许一个分离任务真正推理，
       后提交的任务状态显示「排队」，避免连点两次并发抢显存；
    7) 主题联动：页面顶部按钮切换深 / 浅色（localStorage 记忆），同步 Gradio 主题 class。

第三轮改造要点：
    8) 界面回归最初极简风格：删除全部 @keyframes 装饰动画 / backdrop-filter / 自定义卡片装饰，
       进度条改为 6px 细线单色填充；overlap=8、高级开关、真实进度、任务台账、全局 Timer、
       全局串行锁等能力全部保留。
"""

import json
import gc
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径与运行环境（必须在导入 torch / audio_separator 之前完成注入）
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
BIN_DIR = BASE_DIR / "bin"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
ICON_PNG = ASSETS_DIR / "vs.png"          # 浏览器标签页图标
ICON_ICO = ASSETS_DIR / "vs.ico"          # 桌面快捷方式 / 启动器图标
CATALOG_PATH = BASE_DIR / "catalog.json"
TASKS_PATH = BASE_DIR / "tasks.json"
TORCH_LIB = BASE_DIR / "venv" / "Lib" / "site-packages" / "torch" / "lib"

for _d in (MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ffmpeg 与 CUDA 运行库（torch 自带 cu12 系列 DLL，供 onnxruntime 使用）
os.environ["PATH"] = os.pathsep.join(
    [str(BIN_DIR), str(TORCH_LIB)] + [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
)
# 离线模式：模型全部本地内置，禁止任何联网下载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("HF_ENDPOINT", None)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 关闭 gradio 版本检查 / 匿名统计等联网行为，保证纯离线运行
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ---------------------------------------------------------------------------
# CPU / GPU 协同性能设置（perf.json 持久化；必须在 import torch 前注入线程环境变量）
#   cpu_threads      : CPU 线程数，0 = 交给 torch/OpenMP 自动决定
#   cudnn_benchmark  : GPU 卷积自动调优（分段长度固定时能省掉重复的算法选择开销）
# ---------------------------------------------------------------------------
PERF_PATH = BASE_DIR / "perf.json"
DEFAULT_PERF = {"cpu_threads": 0, "cudnn_benchmark": True, "total_threads_limit": 0}


def _load_perf_cfg():
    cfg = dict(DEFAULT_PERF)
    try:
        if PERF_PATH.exists():
            raw = json.loads(PERF_PATH.read_text(encoding="utf-8") or "{}")
            if isinstance(raw, dict):
                for k in cfg:
                    if k in raw and raw[k] is not None:
                        cfg[k] = int(raw[k]) if k != "cudnn_benchmark" else bool(raw[k])
    except Exception:  # noqa: BLE001
        pass
    return cfg


PERF_CFG = _load_perf_cfg()
_PERF_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
if int(PERF_CFG.get("cpu_threads") or 0) > 0:
    for _k in _PERF_ENV_KEYS:
        os.environ.setdefault(_k, str(int(PERF_CFG["cpu_threads"])))

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from audio_separator.separator import Separator  # noqa: E402

LOG = logging.getLogger("vocalsep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_DEFAULT_TORCH_THREADS = int(torch.get_num_threads())


def apply_perf_settings(cpu_threads=0, cudnn_benchmark=True, persist=True):
    """把 CPU 线程数 / cudnn 策略落到运行时（并对后续任务立即生效）。"""
    cpu_threads = int(cpu_threads or 0)
    if cpu_threads > 0:
        try:
            torch.set_num_threads(cpu_threads)
        except Exception as e:  # noqa: BLE001
            LOG.warning("设置 CPU 线程数失败：%s", e)
        for k in _PERF_ENV_KEYS:
            os.environ[k] = str(cpu_threads)
    else:
        try:
            torch.set_num_threads(_DEFAULT_TORCH_THREADS)
        except Exception:  # noqa: BLE001
            pass
    try:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    except Exception:  # noqa: BLE001
        pass
    PERF_CFG["cpu_threads"] = cpu_threads
    PERF_CFG["cudnn_benchmark"] = bool(cudnn_benchmark)
    if persist:
        try:
            PERF_PATH.write_text(json.dumps(PERF_CFG, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            LOG.warning("性能设置写入失败：%s", e)
    LOG.info("性能设置已应用：cpu_threads=%s（torch=%s）cudnn_benchmark=%s",
             cpu_threads, torch.get_num_threads(), PERF_CFG["cudnn_benchmark"])
    return PERF_CFG


apply_perf_settings(PERF_CFG.get("cpu_threads") or 0, PERF_CFG.get("cudnn_benchmark", True), persist=False)


# 离线守护：任何「本地不存在、需联网下载」的请求一律拦截并报错
def install_offline_guard():
    def _guard(self, url, path, *args, **kwargs):
        if os.path.exists(path):
            return
        raise RuntimeError(
            f"离线模式：本地缺少模型文件 {os.path.basename(str(path))}，已阻止联网下载。"
            "请先把该文件放入 models 目录。"
        )

    Separator.download_file_if_not_exists = _guard


install_offline_guard()


# ---------------------------------------------------------------------------
# 真实进度：hook audio_separator 内部使用的 tqdm
#   audio_separator.separator.architectures.mdxc_separator 里：
#       for start_idx in tqdm(chunk_starts):   # RoFormer 分块推理
#       for batch in tqdm(batches):            # MDXC 分批推理
#   把模块内的 tqdm 符号替换为子类，在 update() 中读取 self.n / self.total。
# ---------------------------------------------------------------------------
_progress_ctx = threading.local()
_TQDM_HOOK = {"installed": False, "reason": ""}


def _current_reporter():
    return getattr(_progress_ctx, "reporter", None)


def _set_reporter(reporter):
    _progress_ctx.reporter = reporter


def install_tqdm_hook():
    """把 mdxc_separator 模块内的 tqdm 替换为可回传进度的子类。"""
    if _TQDM_HOOK["installed"]:
        return True
    try:
        from tqdm import tqdm as _base_tqdm
        from audio_separator.separator.architectures import mdxc_separator as _mdxc
    except Exception as e:  # noqa: BLE001
        _TQDM_HOOK["reason"] = f"模块导入失败：{type(e).__name__}: {e}"
        LOG.warning("tqdm hook 安装失败：%s", _TQDM_HOOK["reason"])
        return False

    class _ProgressTqdm(_base_tqdm):
        """与 tqdm 行为一致，仅在 update() 时把 (n, total) 回传给当前任务的 reporter。"""

        def _vs_report(self):
            rep = _current_reporter()
            if rep is None:
                return
            try:
                rep.update_segment(self.n, self.total, token=id(self))
            except Exception:  # noqa: BLE001
                pass

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._vs_report()

        def update(self, n=1):
            r = super().update(n)
            self._vs_report()
            return r

        def close(self):
            try:
                self._vs_report()
            finally:
                return super().close()

    _mdxc.tqdm = _ProgressTqdm
    _TQDM_HOOK["installed"] = True
    LOG.info("已在 mdxc_separator 上安装 tqdm 进度 hook。")
    return True


install_tqdm_hook()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def probe_duration(path):
    """用自带 ffprobe 读取音频时长（秒）；失败返回 0.0。"""
    exe = BIN_DIR / "ffprobe.exe"
    if not exe.exists():
        return 0.0
    try:
        out = subprocess.run(
            [str(exe), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return max(0.0, float((out.stdout or "").strip().splitlines()[-1]))
    except Exception:  # noqa: BLE001
        return 0.0


def fmt_sec(v):
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if v < 0:
        v = 0.0
    if v < 60:
        return f"{v:.1f}s"
    m, s = divmod(int(v), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def fmt_pct(p):
    try:
        return f"{max(0.0, min(1.0, float(p))) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def fmt_ts(ts):
    if not ts:
        return "-"
    try:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:  # noqa: BLE001
        return "-"


# ---------------------------------------------------------------------------
# 任务台账（JSON 落盘 / 线程安全 / 进度单调不减）
# ---------------------------------------------------------------------------
STATUS_QUEUED = "排队"
STATUS_LOADING = "加载模型"
STATUS_RUNNING = "分离中"
STATUS_DONE = "完成"
STATUS_FAILED = "失败"
STATUS_PAUSED = "已暂停"
ACTIVE_STATES = (STATUS_QUEUED, STATUS_LOADING, STATUS_RUNNING)

# ---------------------------------------------------------------------------
# 运行时全局：暂停闸门 / 下载待清理清单 / 详情导出目录
# ---------------------------------------------------------------------------
EXPORT_DIR = OUTPUT_DIR / "任务详情导出"
PENDING_PATH = OUTPUT_DIR / ".vs_pending_cleanup.json"
_PAUSE_LOCK = threading.Lock()
_PAUSE_SET = set()
_GATE_LOCK = threading.Lock()
_GATES = {}
_PENDING_LOCK = threading.Lock()
_PENDING = set()
_CLEANED_TOTAL = 0

TASK_COLUMNS = ["任务ID", "模式", "源文件", "状态", "进度", "已用", "剩余(ETA)",
                "输出文件数", "本机副本", "输出目录"]


class TaskLedger:
    """后台任务台账：dict + Lock + 原子写 tasks.json（UTF-8）。"""

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._tasks = {}
        self._load()

    # ---- 落盘 ----
    def _load(self):
        data = []
        try:
            if self._path.exists():
                raw = self._path.read_text(encoding="utf-8", errors="replace").strip()
                if raw:
                    loaded = json.loads(raw)
                    if isinstance(loaded, list):
                        data = [t for t in loaded if isinstance(t, dict) and t.get("id")]
                    elif isinstance(loaded, dict) and isinstance(loaded.get("tasks"), list):
                        data = [t for t in loaded["tasks"] if isinstance(t, dict) and t.get("id")]
        except Exception as e:  # noqa: BLE001
            LOG.warning("任务台账读取失败，将重置为空：%s", e)
            data = []

        with self._lock:
            self._tasks = {t["id"]: t for t in data}
            # 上次进程残留下来的活跃任务 -> 标记为中断
            now = time.time()
            for t in self._tasks.values():
                # 暂停态同样依赖进程内的工作线程，重启后无法续跑，一并标记为中断
                if t.get("status") in ACTIVE_STATES or t.get("status") == STATUS_PAUSED:
                    t["status"] = STATUS_FAILED
                    t["paused"] = False
                    t["stage"] = "进程已退出，任务未完成"
                    t["error"] = t.get("error") or "服务重启导致任务中断"
                    t["finished_at"] = now
                    st = t.get("started_at")
                    t["elapsed"] = round(now - st, 1) if st else 0.0
        self._flush()

    def _flush(self):
        with self._lock:
            payload = list(self._tasks.values())
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("任务台账写入失败：%s", e)

    # ---- 对外 API ----
    def create(self, mode, source, output_dir, duration=None, params=None):
        now = time.time()
        with self._lock:
            tid = f"T{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
            self._tasks[tid] = {
                "id": tid,
                "mode": mode,
                "source": source,
                "status": STATUS_QUEUED,
                "stage": "已入队",
                "progress": 0.0,
                "eta": None,
                "seg_done": 0,
                "seg_total": 0,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "elapsed": 0.0,
                "load_sec": None,
                "sep_sec": None,
                "duration": round(float(duration), 2) if duration else None,
                "output_dir": str(output_dir),
                "zip": None,
                "files": [],
                "file_count": 0,
                "error": None,
                "params": params or {},
            }
        self._flush()
        return tid

    def update(self, tid, **fields):
        with self._lock:
            t = self._tasks.get(tid)
            if t is None:
                return None
            for k, v in fields.items():
                if k == "progress":
                    try:
                        v = max(float(t.get("progress") or 0.0), float(v))
                    except (TypeError, ValueError):
                        continue
                t[k] = v
            st = t.get("started_at")
            if st and t.get("status") in ACTIVE_STATES:
                t["elapsed"] = round(time.time() - st, 1)
            out = dict(t)
        self._flush()
        return out

    def finish(self, tid, status, **fields):
        now = time.time()
        with self._lock:
            t = self._tasks.get(tid)
            if t is None:
                return None
            st = t.get("started_at") or now
            t.update(fields)
            t["status"] = status
            t["finished_at"] = now
            t["elapsed"] = round(now - st, 1)
            if status == STATUS_DONE:
                t["progress"] = 1.0
                t["eta"] = 0
            out = dict(t)
        self._flush()
        return out

    def _live(self, t):
        d = dict(t)
        st = d.get("started_at")
        if st and d.get("status") in ACTIVE_STATES:
            d["elapsed"] = round(time.time() - st, 1)
        return d

    def get(self, tid):
        if not tid:
            return None
        with self._lock:
            t = self._tasks.get(tid)
            return self._live(t) if t else None

    def list_tasks(self):
        with self._lock:
            items = [self._live(t) for t in self._tasks.values()]
        items.sort(key=lambda x: (x.get("created_at") or 0), reverse=True)
        return items


LEDGER = TaskLedger(TASKS_PATH)


# ---------------------------------------------------------------------------
# 分离进度换算（0.06 - 0.94：加载一小段 + 分离主体 + 可选的人声去回声）
# ---------------------------------------------------------------------------
P_START = 0.02              # 入队/准备（加载模型前）
P_LOAD_DONE = 0.06          # 模型加载完成（加载只占很小一段，把主体留给分离）
P_SEP_SPAN = 0.88           # 分离阶段占 0.06 -> 0.94
P_SEP_MAX = P_LOAD_DONE + P_SEP_SPAN
P_DRE_LO = 0.56             # 开启人声去回声：分离（1/2）到 0.56，去回声（2/2）0.56 -> 0.94
SECONDS_PER_AUDIO_SECOND = 0.30  # 经验值：overlap=4 + native fp16、单模型下每 1s 音频约需 0.30s 计算


class SeparationProgress:
    """把内部 tqdm 的 (n / total) 映射到给定进度区间（默认 0.06-0.94），并计算 ETA。

    - 能取到 tqdm：按「第几个模型 round_index + 本轮完成比例」计算；
    - 取不到（如 Demucs / VR 架构）：回退为按音频时长的经验估算；
    - 进度单调不减，永不回退。
    """

    def __init__(self, ledger, tid, duration=0.0, n_models=1, overlap=8,
                 out_lo=P_LOAD_DONE, out_hi=P_SEP_MAX, stage=None):
        self.ledger = ledger
        self.tid = tid
        self.out_lo = float(out_lo)
        self.out_hi = float(out_hi)
        self.stage = stage
        self.n_models = max(1, int(n_models))
        self.overlap = max(1, int(overlap or 8))
        self.duration = max(0.0, float(duration or 0.0))
        self.est_total = max(
            20.0,
            self.duration * SECONDS_PER_AUDIO_SECOND * self.n_models * (self.overlap / 8.0),
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.start = time.time()
        self.token = None
        self.round_index = -1
        self.round_best = 0.0
        self.seg_done = 0
        self.seg_total = 0
        self.best_frac = 0.0
        self.last_seg_time = 0.0

    # ---- 由 tqdm hook 调用 ----
    def update_segment(self, n, total, token=None):
        now = time.time()
        with self._lock:
            if token is not None and token != self.token:
                self.token = token
                self.round_index = min(self.round_index + 1, self.n_models - 1)
                self.round_best = 0.0
            try:
                self.seg_done = int(n or 0)
            except (TypeError, ValueError):
                self.seg_done = 0
            try:
                self.seg_total = int(total or 0)
            except (TypeError, ValueError):
                self.seg_total = 0
            self.last_seg_time = now

    # ---- 计算当前进度 / ETA ----
    def snapshot(self):
        with self._lock:
            done, total = self.seg_done, self.seg_total
            round_index = max(0, self.round_index)
            has_tqdm = self.token is not None
        elapsed = max(0.0, time.time() - self.start)

        source = "estimate"
        frac = None
        if has_tqdm and total > 0:
            round_frac = min(1.0, done / total)
            frac = min(1.0, (round_index + round_frac) / self.n_models)
            source = "tqdm"
        if frac is None:
            frac = min(0.95, elapsed / self.est_total) if self.est_total > 0 else 0.0

        # 单调不减
        with self._lock:
            if frac > self.best_frac:
                self.best_frac = frac
            frac = self.best_frac
            round_best = self.round_best

        progress = min(P_SEP_MAX, P_LOAD_DONE + P_SEP_SPAN * frac)

        if source == "tqdm" and frac > 0.02:
            eta = max(0.0, elapsed * (1.0 - frac) / frac)
        else:
            eta = max(0.0, self.est_total - elapsed)

        return {
            "progress": round(progress, 4),
            "frac": round(frac, 6),
            "eta": round(eta, 1),
            "seg_done": done,
            "seg_total": total,
            "source": source,
        }

    def push(self):
        snap = self.snapshot()
        span = self.out_hi - self.out_lo
        prog = self.out_lo + span * snap["frac"]
        eta = snap["eta"]
        self.ledger.update(
            self.tid,
            status=STATUS_RUNNING,
            stage=self.stage or f"分离中（{snap['source']}）",
            progress=round(min(self.out_hi, prog), 4),
            eta=round(eta, 1),
            seg_done=snap["seg_done"],
            seg_total=snap["seg_total"],
        )

    def run_monitor(self):
        while not self._stop.wait(0.8):
            try:
                self.push()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.push()
        except Exception:  # noqa: BLE001
            pass

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# 模型目录
# ---------------------------------------------------------------------------
CATALOG = {"buckets": {}, "all": [], "presets": {}}
if CATALOG_PATH.exists():
    CATALOG = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

PRESETS = CATALOG.get("presets", {})
ALL_MODEL_FILES = {it["file"] for it in CATALOG.get("all", [])}

DEFAULT_MODEL = "bs_roformer_vocals_revive_v3e_unwa.ckpt"   # Revive V3e：目前本地最强的人声提取
ENSEMBLE_ALGOS = [
    "avg_fft",
    "min_fft",
    "max_fft",
    "avg_wave",
    "min_mag",
    "max_mag",
    "uvr_max_spec",
    "uvr_min_spec",
    "ensemble_wav",
]

# 文件名 -> 中文主干名
STEM_CN = {
    "vocals": "人声",
    "instrumental": "伴奏",
    "drums": "鼓",
    "bass": "贝斯",
    "other": "其他",
    "guitar": "吉他",
    "piano": "钢琴",
    "male": "男声",
    "female": "女声",
    "no reverb": "无混响",
    "noreverb": "无混响",
    "reverb": "混响",
    "noise": "噪声",
    "no noise": "已降噪",
    "echo": "回声",
    "no echo": "已去回声",
}

TIER_SINGLE = "单模型"
TIER_PRESET = "内置集成组合（效果更佳）"
TIER_CUSTOM = "自定义多模型集成"
MAX_STEMS = 8

# 本地内置模型：已随工程落盘于 models 目录，离线可用
BUILTIN_MODELS = [
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "bs_roformer_vocals_revive_v3e_unwa.ckpt",
    "melband_roformer_instvox_duality_v2.ckpt",
    "mel_band_roformer_instrumental_fv7z_gabox.ckpt",
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
]
BUILTIN_SET = set(BUILTIN_MODELS)

# 去回声 / 去混响模型：本地已内置，分离完成后可对「人声」轨做可选后处理。
# 关键结论（实测，见 README）：anvuew 系列是「去混响」，对离散回声（延迟 / 回声尾巴）
# 几乎无效（真实湿人声上 SDR 只有 15.1）；Sucial 的 De-Reverb-Echo 系列才是治回声的，
# 同一素材上 SDR 24.3，且不掉电平、不掉高频。
DEREVERB_CHOICES = {
    "去回声 · De-Reverb-Echo V2（推荐 · 真实混响+回声实测最强 SDR 13.48）":
        "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    "去回声 · Fused（小混响+大混响通吃，官方推荐融合版）":
        "dereverb_echo_mbr_fused.ckpt",
    "去混响 · 标准 anvuew（SDR 19.17 · 只去混响，去回声几乎无效）":
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "去混响 · 轻柔 anvuew（SDR 18.81 · 混响轻或怕伤高频时用）":
        "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    # ★ 双模型串联：先用 V2 压掉大部分混响+回声，再用 Fused 清残留尾巴。
    #   两个模型的失败模式不同，串起来比同一模型跑两遍干净得多（耗时约翻倍）。
    "去回声 · 双模型串联（V2 → Fused · 最干净 · 慢一倍）":
        "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt+dereverb_echo_mbr_fused.ckpt",
}
# 模型文件名（含 + 串联）→ 界面文案，仅用于日志/状态展示
DEREVERB_LABEL_FOR = {v: k for k, v in DEREVERB_CHOICES.items()}


def dereverb_friendly(model_file=None):
    """把内部模型串（可能是 ‘A+B’）翻译成好读的名字。"""
    raw = str(model_file or "")
    if raw in DEREVERB_LABEL_FOR:
        return DEREVERB_LABEL_FOR[raw]
    if "+" in raw:
        return " → ".join(Path(x).stem for x in raw.split("+") if x)
    return Path(raw).stem if raw else raw
DEFAULT_DEREVERB = "去回声 · De-Reverb-Echo V2（推荐 · 真实混响+回声实测最强 SDR 13.48）"

# 这些模型各自的「干净主干」标签（audio-separator 输出名里的括号标签）
DEREVERB_PRIMARY_STEM = {
    "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt": "dry",
    "dereverb_echo_mbr_fused.ckpt": "dry",
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": "noreverb",
    "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt": "noreverb",
}

# 去回声遍数：实测「2 遍」对回声的额外削减极小（-8.48 → -8.52dB）却让耗时翻倍，
# 所以默认锁定 1 遍；2 遍只留作极端情况下手动尝试。
DEREVERB_PASSES = {
    "1 遍 · 默认（推荐）": 1,
    "2 遍 · 更狠（实测对回声增益极小，耗时翻倍）": 2,
}
DEFAULT_DEREVERB_PASSES = "1 遍 · 默认（推荐）"


def dereverb_passes_value(label=None):
    """把界面上的遍数文案换算成整数（未知取值一律按 1 遍）。"""
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        return max(1, min(4, int(label)))
    try:
        return int(DEREVERB_PASSES.get(label, 1))
    except Exception:  # noqa: BLE001
        return 1


def dereverb_model_ready(label=None):
    """返回 (是否就绪, 模型文件名, 说明)。"""
    if not label:
        label = DEFAULT_DEREVERB
    if label not in DEREVERB_CHOICES:
        label = DEFAULT_DEREVERB
    fn = DEREVERB_CHOICES[label]
    missing = [x for x in fn.split("+") if x and not (MODEL_DIR / x).exists()]
    ok = not missing
    return ok, fn, ("" if ok else "去回声模型未内置：" + "、".join(missing))


def dereverb_primary_stem(model_file=None):
    """该去回声 / 去混响模型的「干净主干」标签（用于挑轨与 output_single_stem）。"""
    return DEREVERB_PRIMARY_STEM.get(model_file or "", "dry")

# 内置集成组合：成员全部为本地模型，无需任何下载
BUILTIN_PRESETS = {
    "builtin_vocal_revive": {
        "name": "内置 · 人声极致干净（Revive V3e + BS-Roformer 1297）",
        "models": [
            "bs_roformer_vocals_revive_v3e_unwa.ckpt",
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        ],
        "algo": "avg_fft",
    },
    "builtin_vocal_dual": {
        "name": "内置 · 人声双雄（BS-Roformer 1297 + 1296）",
        "models": [
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        ],
        "algo": "avg_fft",
    },
    "builtin_vocal_triple": {
        "name": "内置 · 人声三合一（1297 + 1296 + InstVoc V2）",
        "models": [
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            "melband_roformer_instvox_duality_v2.ckpt",
        ],
        "algo": "avg_fft",
    },
}


def _peak_of(filename):
    for it in CATALOG.get("all", []):
        if it["file"] == filename:
            return it.get("peak", -1)
    for items in CATALOG.get("buckets", {}).values():
        for it in items:
            if it["file"] == filename:
                return it.get("peak", -1)
    return -1


# 每个分离模型在界面上的「人话备注」：(排序权重, 显示名)。
# 人声 SDR 取自 catalog.json 里各模型自带的 vocals 轨评分；Revive V3e 无公开评分，
# 但实测人声最干净，因此固定在第一位并标注「推荐」。
# 45s 音频耗时为本机实测（batch=8 / overlap=4 / autocast，见 work/model_speed.json）。
MODEL_ROLE = {
    "bs_roformer_vocals_revive_v3e_unwa.ckpt":
        (0, "⭐ 推荐 · Revive V3e｜人声最干净"),
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt":
        (1, "🎤 人声 12.1 · ep368｜均衡"),
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt":
        (2, "🎤 人声 11.8 · ep317｜均衡"),
    "melband_roformer_instvox_duality_v2.ckpt":
        (3, "🎛 双输出 · InstVox Duality v2"),
    "mel_band_roformer_instrumental_fv7z_gabox.ckpt":
        (4, "⚡ 最快 · Mel-Band gabox（偏伴奏）"),
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt":
        (5, "🎙 卡拉OK · Viperx｜去人声做伴奏"),
}
# 去混响 / 去回声是「后处理专用」模型，不该混进分离模型下拉（它们只输出干净/带混响两轨）
NON_SEPARATION_MODELS = {
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
    "dereverb_echo_mbr_fused.ckpt",
}

MODEL_HELP = (
    "**怎么选**：要人声（RVC 翻唱）直接用默认的 ⭐ Revive V3e；"
    "想更快可以用 ⚡ Mel-Band gabox（约快 2.5 倍，但更偏伴奏）；"
    "要人声 + 伴奏同时都要，用 🎛 InstVox Duality v2；"
    "只要伴奏、不要人声，用 🎙 Viperx。"
    "去混响 / 去回声是后处理模型，请在「人声去回声」里选，不要在这里选。"
)


def build_model_choices():
    """只列出本地已内置的分离模型（离线可用），值为真实文件名，标签带推荐/速度提示。"""
    rows = []
    for f in BUILTIN_MODELS:
        if f in NON_SEPARATION_MODELS:
            continue
        if not (MODEL_DIR / f).exists():
            continue
        order, name = MODEL_ROLE.get(f, (90, f))
        rows.append((order, name, f))
    rows.sort(key=lambda r: (r[0], r[2]))
    return [(name, f) for _o, name, f in rows]


def _build_model_choices_legacy():
    """原逻辑：列出全部模型（保留备查）。"""
    seen, out = set(), []
    for bucket, items in CATALOG.get("buckets", {}).items():
        for it in items:
            f = it["file"]
            if f in seen:
                continue
            seen.add(f)
            peak = it.get("peak", -1)
            tag = f"  (SDR {peak:.1f})" if isinstance(peak, (int, float)) and peak > 0 else ""
            out.append((f"★ {bucket} · {f}{tag}", f))
    for it in CATALOG.get("all", []):
        f = it["file"]
        if f in seen:
            continue
        seen.add(f)
        peak = it.get("peak", -1)
        tag = f"  (SDR {peak:.1f})" if isinstance(peak, (int, float)) and peak > 0 else ""
        out.append((f"{f}{tag}", f))
    return out


def build_preset_choices():
    """内置集成组合（成员全部为本地模型）。"""
    out = []
    for key, meta in BUILTIN_PRESETS.items():
        if all((MODEL_DIR / m).exists() for m in meta["models"]):
            out.append((meta["name"], key))
    return out


def normalize_models(values):
    """兼容控件返回值是 value 还是 label。"""
    res = []
    for v in values or []:
        if v in ALL_MODEL_FILES:
            res.append(v)
            continue
        m = re.search(r"·\s*([^\s(]+)", v) or re.search(r"\]\s*([^\s(]+)", v)
        if m and m.group(1) in ALL_MODEL_FILES:
            res.append(m.group(1))
            continue
        tail = v.strip().split()[-1] if v.strip() else ""
        res.append(tail if tail in ALL_MODEL_FILES else v)
    return res


def model_downloaded(filename):
    if filename.endswith(".yaml"):  # demucs：权重为同名 .th
        candidates = [filename, filename.replace(".yaml", ".th")]
    else:
        candidates = [filename, filename + ".json"]
    return any((MODEL_DIR / c).exists() for c in candidates)


def models_needed(tier, single_model, preset_key, custom_models):
    if tier == TIER_PRESET and preset_key:
        return list(BUILTIN_PRESETS.get(preset_key, {}).get("models", []))
    if tier == TIER_CUSTOM:
        return normalize_models(custom_models)
    return [single_model]


# ---------------------------------------------------------------------------
# 分离核心
# ---------------------------------------------------------------------------
def _mdxc_params(seg, batch, overlap):
    """Roformer / MDXC 推理参数。

    overlap 是「重叠窗口倍数」（整数，库默认 8），越大边界拼接越平滑、越精细，
    但计算量近似成正比增长：overlap=32 相当于把同一段音频重复推理 32 倍。
    segment_size 传 0 表示跟随模型原生分段，传正数则强制覆盖。
    """
    params = {
        "batch_size": max(1, int(batch)),
        "overlap": max(1, int(overlap)),
    }
    seg_val = int(seg)
    if seg_val > 0:
        params["segment_size"] = seg_val
        params["override_model_segment_size"] = True
    return params


def make_separator(out_dir, fmt, sr, norm, amp, single_stem, batch, seg, overlap, shifts, dem_overlap,
                   tier, preset_key, algo, use_native_fp16=False, use_torch_compile=False):
    # use_autocast 与 use_native_fp16 互斥（audio-separator 会直接抛错）：
    # 默认仍然只启用 use_autocast=True，与改造前保持完全一致。
    use_native_fp16 = bool(use_native_fp16)
    kwargs = dict(
        log_level=logging.WARNING,
        model_file_dir=str(MODEL_DIR),
        output_dir=str(out_dir),
        output_format=fmt,
        sample_rate=int(sr),
        use_autocast=not use_native_fp16,
        use_native_fp16=use_native_fp16,
        use_torch_compile=bool(use_torch_compile),
        normalization_threshold=float(norm),
        amplification_threshold=float(amp),
        output_single_stem=(single_stem or None),
        mdxc_params=_mdxc_params(seg, batch, overlap),
        demucs_params={
            "segment_size": "Default",
            "shifts": int(shifts),
            "overlap": float(dem_overlap),
            "segments_enabled": True,
        },
    )
    if tier == TIER_PRESET and preset_key:
        kwargs["ensemble_algorithm"] = BUILTIN_PRESETS.get(preset_key, {}).get("algo", "avg_fft")
    elif tier == TIER_CUSTOM:
        kwargs["ensemble_algorithm"] = algo
    return Separator(**kwargs)


def load_models(sep, tier, preset_key, models, single_model):
    if tier == TIER_PRESET:
        meta = BUILTIN_PRESETS.get(preset_key)
        if not meta:
            raise gr.Error("请选择一个内置集成组合")
        sep.load_model(meta["models"])  # 全部本地模型
    elif tier == TIER_CUSTOM:
        files = normalize_models(models)
        if not files:
            raise gr.Error("请至少选择一个模型用于集成")
        sep.load_model(files)
    else:
        sep.load_model(single_model)


def resolve_outputs(names, out_dir):
    res = []
    for n in names or []:
        p = Path(n)
        if p.is_absolute() and p.exists():
            res.append(p)
            continue
        cand = Path(out_dir) / p.name
        if cand.exists():
            res.append(cand)
            continue
        hit = list(Path(out_dir).rglob(p.name))
        res.append(hit[0] if hit else cand)
    return [p for p in res if p.exists()]


def friendly_rename(files, song_stem):
    """把 (Vocals) 之类的英文主干名换成中文，便于辨认。"""
    out = []
    for p in files:
        m = re.search(r"\(([^()]+)\)", p.stem)
        cn = None
        if m:
            raw = m.group(1).strip().lower()
            cn = STEM_CN.get(raw) or STEM_CN.get(raw.split("*")[0].strip())
        if not cn:
            out.append(p)
            continue
        target = p.with_name(f"{song_stem}_{cn}{p.suffix}")
        if target.exists() and target != p:
            target = p.with_name(f"{song_stem}_{cn}_{int(time.time() % 1000)}{p.suffix}")
        try:
            p.rename(target)
            out.append(target)
        except OSError:
            out.append(p)
    return out


def _pick_vocals(files):
    """从分离结果里挑出「人声」轨。

    坑：audio-separator 的输出名是 `歌名_(主干)_模型名.wav`，而像
    bs_roformer_vocals_revive_v3e 这种模型名本身就带 "vocals"，
    若整名匹配就会把「其他 / 伴奏」轨误判成人声（去回声会去错轨，产物又小又怪）。
    所以这里优先只看括号里的「主干标签」，并显式排除伴奏类标签。
    """
    bad = ("instrumental", "other", "karaoke", "accompaniment", "inst_", "instrument")
    labeled, plain = [], []
    for f in files or []:
        p = Path(str(f))
        name = p.name
        low = name.lower()
        m = re.search(r"\(([^()]*)\)", low)
        label = m.group(1).strip() if m else None
        if label is not None:
            if any(k in label for k in bad):
                continue
            if "vocal" in label or "voice" in label or "dry" in label or "人声" in name:
                labeled.append(p)
        else:
            if any(k in low for k in bad):
                continue
            if "人声" in name or "vocal" in low or "voice" in low:
                plain.append(p)
    return (labeled or plain or [None])[0]


def _rename_unique(src, target):
    """把 src 改名到 target；目标已存在时加时间后缀避免覆盖。失败时返回原路径。"""
    src, target = Path(src), Path(target)
    if target.exists() and target.resolve() != src.resolve():
        target = target.with_name(f"{target.stem}_{int(time.time() % 1000)}{target.suffix}")
    try:
        src.rename(target)
        return target
    except OSError:
        return src


def zip_files(files, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    return zip_path


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}


def collect_audio_paths(files_in):
    paths = []
    for f in files_in or []:
        p = Path(getattr(f, "name", f))
        if p.is_dir():
            paths.extend([x for x in sorted(p.rglob("*")) if x.suffix.lower() in AUDIO_EXTS])
        elif p.suffix.lower() in AUDIO_EXTS:
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# 后台任务执行体（全局串行：同一时刻只允许一个分离任务真正占用 GPU）
# ---------------------------------------------------------------------------
RUN_LOCK = threading.Lock()


def run_lock_busy():
    """当前是否已有任务持锁在跑（提交时据此给出「已开始」或「已排队」文案）。"""
    return RUN_LOCK.locked()


def _wait_run_slot(tid, start_progress=0.05):
    """等待全局运行槽位；若前面已有任务在跑，先把状态标成「排队」。
    拿到槽位后若用户此前点过「暂停」，就在这里停住，等「继续」再开始加载模型。"""
    if RUN_LOCK.locked():
        LEDGER.update(tid, status=STATUS_QUEUED,
                      stage="排队中 · 前一个任务完成后自动开始", eta=None)
    RUN_LOCK.acquire()
    if _is_paused(tid):
        LEDGER.update(tid, status=STATUS_PAUSED, stage="已暂停 · 点「继续」后开始推理",
                      paused=True, pause_started_at=time.time(), eta=None)
        _pause_gate(tid)
    LEDGER.update(tid, status=STATUS_LOADING, stage="加载本地内置模型",
                  started_at=time.time(), progress=start_progress, eta=None,
                  paused=False, pause_started_at=None)


def _release_run_slot():
    """释放全局运行槽位（未持锁时静默忽略，避免异常路径二次释放报错）。"""
    try:
        RUN_LOCK.release()
    except RuntimeError:
        pass


def _spectral_tilt_match(sig, ref, sr, max_db=6.0, f_lo=180.0, f_hi=16000.0):
    """把 sig 的「整体频谱倾斜」朝 ref 靠拢，限幅 ±max_db。

    为什么需要它：去回声 / 去混响模型多少都会改变频谱包络（典型是掉低频下盘、
    掉 8kHz 以上空气感），于是用户听到「去完之后高低频跟原曲对不上」。
    这里按 1/3 倍频程算出 ref/sig 的宽带增益差，限幅后平滑铺回，只做「宽频带倾斜」
    修正 —— 不做窄带 / 共振峰提升，因此不会把刚刚压下去的回声能量重新抬起来。
    """
    import numpy as np
    n = int(min(len(sig), len(ref)))
    if n < 8192:
        return sig, 0.0
    win = np.hanning(n).astype("float32")[:, None]
    mS = np.abs(np.fft.rfft(sig[:n] * win, axis=0)).mean(axis=1) + 1e-12
    mR = np.abs(np.fft.rfft(ref[:n] * win, axis=0)).mean(axis=1) + 1e-12
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    hi = float(min(f_hi, sr * 0.49))
    if hi <= f_lo * 2.0:
        return sig, 0.0
    centers = f_lo * (2.0 ** (np.arange(0, 40) / 3.0))
    centers = centers[centers <= hi]
    if len(centers) < 3:
        return sig, 0.0
    band = []
    for c in centers:
        half = c * (2.0 ** (1 / 6.0)) - c * (2.0 ** (-1 / 6.0))
        m = (freqs >= c - half) & (freqs <= c + half)
        band.append(0.0 if not m.any() else 20.0 * np.log10(float(mR[m].mean()) / float(mS[m].mean())))
    band = np.clip(np.asarray(band, dtype="float64"), -max_db, max_db)
    fq = np.clip(freqs, f_lo, hi)
    corr = np.interp(np.log(fq), np.log(centers), band)
    ramp = np.clip((freqs - f_lo) / (f_lo * 0.6 + 1e-9), 0.0, 1.0)
    fade = np.clip((hi - freqs) / (hi * 0.12 + 1e-9), 0.0, 1.0)
    corr = corr * ramp * fade
    applied = float(np.abs(band).max())
    g = (10.0 ** (corr / 20.0)).astype("float32")
    y = np.fft.irfft(np.fft.rfft(sig, axis=0) * g[:, None], n=len(sig), axis=0)
    return y.astype("float32"), applied


def _dereverb_postprocess(path, ref_path, restore_air=False, air_cut=8000.0, match_tilt=True):
    """去回声收尾：把产物拉回与「输入人声」同等响度（必做），高频补回默认关闭。

    - 等响补偿：所有这类模型都可能轻微改变整体增益，拉回与分离人声同名电平，
      避免「去完之后音量跟原曲对不上」。
    - 频谱倾斜匹配（match_tilt，默认开）：按 1/3 倍频程把宽带频谱包络拉回与
      输入人声一致（限幅 ±4dB），修「去完之后高低频跟原曲对不上」。
    - 高频补回（restore_air，默认关）：老式 anvuew 去混响会吃掉 8kHz 以上空气感，
      需要补；但新去回声模型（De-Reverb-Echo V2 / Fused）实测完全不掉高频，
      再补就等于把「原人声里的回声高频」又灌回去 —— 越补回声越多。
    就地改写 path，返回 (增益, 是否补高频)。
    """
    import numpy as np
    import soundfile as sf
    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception:  # noqa: BLE001
        sosfiltfilt = None
    sig, sr = sf.read(str(path), dtype='float32', always_2d=True)
    ref, ref_sr = sf.read(str(ref_path), dtype='float32', always_2d=True)
    if int(ref_sr) != int(sr):
        raise RuntimeError("采样率不一致：%s != %s" % (ref_sr, sr))
    ch = min(sig.shape[1], ref.shape[1])
    sig, ref = sig[:, :ch], ref[:, :ch]
    n_ov = min(len(sig), len(ref))
    if len(ref) < len(sig):
        ref = np.pad(ref, ((0, len(sig) - len(ref)), (0, 0)))
    else:
        ref = ref[:len(sig)]

    def _rms(x):
        return float(np.sqrt((x.astype('float64') ** 2).mean()) + 1e-12)

    # 只用「重叠区」算增益：产物比原轨长/短时，用补零后的全长会把增益算偏
    # 上限放到 12 倍（约 +21.6dB）：这类去回声 / 去混响模型实测可能把整轨压低十几 dB，
    # 原来 5 倍的上限补不回来 —— 这正是「去完之后音量跟原曲对不上」的直接原因。
    gain = min(max(_rms(ref[:n_ov]) / _rms(sig[:n_ov]), 0.2), 12.0)
    out = sig * gain
    tilt = 0.0
    if match_tilt:
        try:
            out, tilt = _spectral_tilt_match(out, ref[:len(out)], sr)
        except Exception:  # noqa: BLE001
            LOG.exception("频谱倾斜匹配失败，跳过（不影响其余后处理）")
            tilt = 0.0
    air = False
    if restore_air and sosfiltfilt is not None:
        sos = butter(4, air_cut / (sr / 2.0), btype='lowpass', output='sos')
        out = sosfiltfilt(sos, out, axis=0).astype('float32') + \
            (ref - sosfiltfilt(sos, ref, axis=0)).astype('float32')
        air = True
    peak = float(np.abs(out).max())
    if peak > 0.999:
        out = out * (0.999 / peak)
    sf.write(str(path), out, sr, subtype='PCM_16')
    LOG.info("去回声后处理明细：等响 ×%.3f，频谱倾斜修正上限 %.2fdB，高频补回=%s", gain, tilt, air)
    return gain, air


def run_dereverb(tid, src_path, work_dir, model_file, duration=0.0, batch=8, overlap=4,
                 out_lo=P_DRE_LO, out_hi=P_SEP_MAX, stage="人声去回声",
                 passes=1, match_level=True, restore_air=False):
    """可选后处理链路：对（已分离出的）人声轨去回声 / 去混响（RVC 翻唱需要干净的无回声输入）。

    - 只导出该模型自带的主干标签（dry / noreverb），避免把「回声轨」当成结果交付；
    - passes > 1 时把上一遍的结果再喂进模型（实测对回声增益极小，默认 1 遍）；
    - 收尾做「等响补偿」；高频补回默认关闭（新去回声模型不掉高频，补了反而把回声带回来）。
    返回 (去回声后音频路径, 该阶段耗时秒数)。
    """
    import soundfile as sf

    ddir = Path(work_dir) / "人声去回声"
    ddir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = max(1, int(passes or 1))
    models = [m.strip() for m in str(model_file or "").split("+") if m.strip()]
    if not models:
        raise RuntimeError("未指定去回声模型")
    # 串联时把「每个模型 × 遍数」摊平成步骤，进度条按步骤数均分
    steps = []
    for _mf in models:
        for _k in range(n):
            steps.append(_mf)
    span = (float(out_hi) - float(out_lo)) / max(1, len(steps))
    # 必须用「输入人声」的采样率推理，否则收尾的等响补偿会因为采样率不一致而失败
    try:
        src_sr = int(sf.info(str(src_path)).samplerate)
    except Exception:  # noqa: BLE001
        src_sr = 44100
    cur, out = str(src_path), None
    for i, mf in enumerate(steps):
        primary = dereverb_primary_stem(mf)
        sep = Separator(log_level=logging.WARNING, model_file_dir=str(MODEL_DIR), output_dir=str(ddir),
                        output_format="WAV", sample_rate=src_sr, use_autocast=True,
                        output_single_stem=primary,
                        mdxc_params={"batch_size": max(1, int(batch or 8)), "overlap": max(1, int(overlap or 4))})
        sep.load_model(mf)
        rep = SeparationProgress(LEDGER, tid, duration=duration or 0.0, n_models=1, overlap=overlap or 4,
                                 out_lo=out_lo + span * i, out_hi=out_lo + span * (i + 1), stage=stage)
        monitor = threading.Thread(target=rep.run_monitor, daemon=True)
        _set_reporter(rep)
        monitor.start()
        try:
            names = sep.separate(str(cur))
        finally:
            _set_reporter(None)
            rep.stop()
            monitor.join(timeout=3)
        out = _pick_clean_stem(resolve_outputs(names, ddir), primary)
        if out is None:
            raise RuntimeError("去回声模型未产出可用文件")
        cur = str(out)
    if match_level or restore_air:
        try:
            gain, air = _dereverb_postprocess(out, src_path, restore_air=restore_air)
            LOG.info("[%s] 去回声后处理：等响 ×%.2f，高频补回=%s", tid, gain, air)
        except Exception:  # noqa: BLE001
            LOG.exception("[%s] 去回声后处理失败，保留原始产物", tid)
    sec = time.time() - t0
    LOG.info("[%s] 去回声完成（%.1fs / %s / %dHz / 步骤=%d）：%s", tid, sec,
             " → ".join(Path(m).stem for m in models), src_sr, len(steps), out)
    return out, sec


def _pick_clean_stem(files, primary="dry"):
    """从去回声产物里挑出「干净主干」那一轨。

    audio-separator 输出名形如 `歌名_(dry)_模型名.wav`：先按 `(primary)` 精确匹配，
    再退化成包含匹配，最后兜底取第一个文件（保证不会把「回声轨」交付出去）。
    """
    files = [Path(str(f)) for f in (files or [])]
    if not files:
        return None
    want = (primary or "dry").lower()
    exact, loose = [], []
    for f in files:
        low = f.name.lower()
        m = re.search(r"\(([^()]*)\)", low)
        tag = m.group(1).strip() if m else ""
        if tag == want:
            exact.append(f)
        elif want and want in low:
            loose.append(f)
    return (exact or loose or files)[0]


def _run_single_task(tid, src_path, song, job_dir, opt):
    """单曲分离（后台线程；需先拿到全局串行锁才真正开始推理）。

    开启「人声去回声」时走两段式：1/2 分离 → 2/2 只对分离出的人声轨去回声，
    最终连同「歌名_人声去回声.wav」一并交付（RVC 翻唱直接可用）。
    """
    _register_gate(tid)
    try:
        _wait_run_slot(tid, P_START)
        t0 = time.time()
        dre_sec, dre_out, dre_note = 0.0, None, ""
        want_dre = bool(opt.get("dereverb"))
        sep = make_separator(job_dir, opt["fmt"], opt["sr"], opt["norm"], opt["amp"], opt["single_stem"],
                             opt["batch"], opt["seg"], opt["overlap"], opt["shifts"], opt["dem_overlap"],
                             opt["tier"], opt["preset_key"], opt["algo"],
                             use_native_fp16=opt["use_native_fp16"],
                             use_torch_compile=opt["use_torch_compile"])
        load_models(sep, opt["tier"], opt["preset_key"], opt["custom_models"], opt["single_model"])
        load_sec = time.time() - t0
        LOG.info("[%s] 模型加载完成，用时 %.1fs", tid, load_sec)

        sep_hi = P_DRE_LO if want_dre else P_SEP_MAX
        LEDGER.update(tid, status=STATUS_RUNNING,
                      stage="分离中（1/2）" if want_dre else "分离中",
                      progress=P_LOAD_DONE, load_sec=round(load_sec, 2),
                      dereverb_model=opt.get("dereverb_model") if want_dre else None,
                      dereverb_sec=None, dereverb_out=None)

        reporter = SeparationProgress(LEDGER, tid, duration=opt.get("duration") or 0.0,
                                      n_models=max(1, len(opt.get("models") or [1])),
                                      overlap=opt.get("overlap") or 8,
                                      out_lo=P_LOAD_DONE, out_hi=sep_hi)
        monitor = threading.Thread(target=reporter.run_monitor, daemon=True)
        _set_reporter(reporter)
        monitor.start()
        t1 = time.time()
        try:
            names = sep.separate(str(src_path))
        finally:
            _set_reporter(None)
            reporter.stop()
            monitor.join(timeout=3)
        sep_sec = time.time() - t1

        _pause_gate(tid)  # 完成后若处于暂停态，先停住，不整理输出
        files_raw = resolve_outputs(names, job_dir)

        # ---- 可选后处理（2/2）：只对分离出的人声轨去混响 / 去回声 ----
        if want_dre:
            vocals = _pick_vocals(files_raw)
            if vocals is None:
                dre_note = "（未找到人声轨，已跳过去回声）"
                LOG.warning("[%s] 未找到人声轨，跳过去回声", tid)
            else:
                ok, dre_model, why = dereverb_model_ready(opt.get("dereverb_model"))
                if not ok:
                    raise RuntimeError(why)
                LEDGER.update(tid, stage="人声去回声（2/2）· 去除回声与混响",
                              progress=P_DRE_LO, eta=None)
                _path, dre_sec = run_dereverb(
                    tid, vocals, job_dir, dre_model,
                    duration=opt.get("duration") or 0.0,
                    batch=opt.get("batch") or 8, overlap=opt.get("overlap") or 4,
                    out_lo=P_DRE_LO, out_hi=P_SEP_MAX,
                    stage="人声去回声（2/2）· 去除回声与混响",
                    passes=opt.get("dereverb_passes") or 1,
                    match_level=opt.get("dereverb_match", True),
                    restore_air=opt.get("dereverb_air", False))
                dre_out = str(_rename_unique(_path, Path(_path).with_name(f"{song}_人声去回声{Path(_path).suffix}")))
                LOG.info("[%s] 人声去回声产物：%s（%.1fs）", tid, dre_out, dre_sec)

        LEDGER.update(tid, progress=0.96, stage="整理输出文件", eta=0)
        files = friendly_rename(files_raw, song)
        if dre_out:  # 去回声人声放最前，方便直接取用
            files = [Path(dre_out)] + list(files)
        zip_path = zip_files(files, Path(job_dir) / f"{song}_分离结果.zip")
        tail = f" / 人声去回声 {dre_sec:.1f}s" if dre_sec else dre_note
        LEDGER.finish(
            tid, STATUS_DONE, stage=f"完成（模型加载 {load_sec:.1f}s / 分离 {sep_sec:.1f}s{tail}）",
            load_sec=round(load_sec, 2), sep_sec=round(sep_sec, 2),
            dereverb_sec=(round(dre_sec, 2) if dre_sec else None),
            dereverb_out=(str(dre_out) if dre_out else None),
            files=[str(f) for f in files], file_count=len(files), zip=str(zip_path),
            paused=False, pause_started_at=None,
        )
        LOG.info("[%s] 分离完成：%d 个文件，分离用时 %.1fs", tid, len(files), sep_sec)

    except Exception as e:  # noqa: BLE001
        LOG.exception("[%s] 任务失败", tid)
        LEDGER.finish(tid, STATUS_FAILED, stage="失败", paused=False,
                      error=f"{type(e).__name__}: {e}")
    finally:
        _unregister_gate(tid)
        _release_run_slot()


def _run_batch_task(tid, paths, batch_dir, opt):
    """批量分离（后台线程，单任务内串行处理多个文件；需先拿到全局串行锁）。"""
    _register_gate(tid)
    try:
        paths = [Path(p) for p in paths]   # submit 传入的是字符串列表，统一成 Path
        _wait_run_slot(tid, 0.03)
        t0 = time.time()
        sep = make_separator(batch_dir, opt["fmt"], opt["sr"], opt["norm"], opt["amp"], "",
                             opt["batch"], opt["seg"], opt["overlap"], opt["shifts"], opt["dem_overlap"],
                             opt["tier"], opt["preset_key"], "avg_fft",
                             use_native_fp16=opt["use_native_fp16"],
                             use_torch_compile=opt["use_torch_compile"])
        load_models(sep, opt["tier"], opt["preset_key"], [], opt["single_model"])
        load_sec = time.time() - t0
        LEDGER.update(tid, load_sec=round(load_sec, 2), status=STATUS_RUNNING,
                      stage="分离中", progress=0.05)

        all_files, failed = [], []
        total = len(paths)
        want_dre = bool(opt.get("dereverb"))
        dre_total_sec, dre_last = 0.0, None
        for i, p in enumerate(paths):
            _pause_gate(tid)  # 暂停后：当前文件处理完就在这里停住，等「继续」
            # 每个文件在 0.05-0.90 区间内占一格；开启去回声时，前 55% 给分离、后 45% 给人声去回声
            lo = 0.05 + 0.85 * (i / total)
            hi = 0.05 + 0.85 * ((i + 1) / total)
            sep_hi = lo + 0.55 * (hi - lo) if want_dre else hi

            LEDGER.update(tid, stage=f"分离 {i + 1}/{total}：{p.name}", progress=round(lo, 4))
            rep = SeparationProgress(LEDGER, tid, duration=probe_duration(p), n_models=1,
                                     overlap=opt.get("overlap") or 8,
                                     out_lo=lo, out_hi=sep_hi,
                                     stage=f"分离 {i + 1}/{total}：{p.name}")
            monitor = threading.Thread(target=rep.run_monitor, daemon=True)
            _set_reporter(rep)
            monitor.start()
            names = None
            try:
                names = sep.separate(str(p))
            except Exception as e:  # noqa: BLE001
                LOG.exception("[%s] 文件失败：%s", tid, p.name)
                failed.append(f"{p.name}: {type(e).__name__}")
            finally:
                _set_reporter(None)
                rep.stop()
                monitor.join(timeout=3)

            if names is not None:
                files_raw = resolve_outputs(names, batch_dir)
                dre_add = []
                # ---- 可选后处理：分离完成后只对人声轨去回声 ----
                if want_dre:
                    vocals = _pick_vocals(files_raw)
                    if vocals is None:
                        LOG.warning("[%s] %s 未找到人声轨，跳过去回声", tid, p.name)
                    else:
                        try:
                            ok, dre_model, why = dereverb_model_ready(opt.get("dereverb_model"))
                            if not ok:
                                raise RuntimeError(why)
                            LEDGER.update(tid, stage=f"人声去回声 {i + 1}/{total}：{p.name}",
                                          progress=round(sep_hi, 4), eta=None)
                            dre_path, _dsec = run_dereverb(
                                tid, vocals, batch_dir, dre_model, duration=probe_duration(p),
                                batch=opt.get("batch") or 8, overlap=opt.get("overlap") or 4,
                                out_lo=sep_hi, out_hi=hi,
                                stage=f"人声去回声 {i + 1}/{total}：{p.name}",
                                passes=opt.get("dereverb_passes") or 1,
                                match_level=opt.get("dereverb_match", True),
                                restore_air=opt.get("dereverb_air", False))
                            dre_final = _rename_unique(
                                dre_path,
                                Path(dre_path).with_name(f"{p.stem}_人声去回声{Path(dre_path).suffix}"))
                            dre_add = [dre_final]
                            dre_total_sec += float(_dsec or 0.0)
                            dre_last = str(dre_final)
                        except Exception as e:  # noqa: BLE001
                            LOG.exception("[%s] 人声去回声失败：%s", tid, p.name)
                            failed.append(f"{p.name}: 去回声 {type(e).__name__}")
                all_files.extend(friendly_rename(files_raw, p.stem))
                all_files.extend(dre_add)
            LEDGER.update(tid, progress=round(hi, 4), eta=0)
        _pause_gate(tid)
        total_sec = time.time() - t0
        zip_path = zip_files(all_files, Path(batch_dir) / f"批量结果_{len(all_files)}个文件.zip")
        LEDGER.finish(
            tid, STATUS_DONE,
            stage=f"完成（成功 {total - len(failed)}/{total}）",
            load_sec=round(load_sec, 2), sep_sec=round(total_sec, 2),
            dereverb_sec=(round(dre_total_sec, 2) if dre_total_sec else None),
            dereverb_out=(str(dre_last) if dre_last else None),
            files=[str(f) for f in all_files], file_count=len(all_files), zip=str(zip_path),
            error=("失败：" + "; ".join(failed)) if failed else None,
            paused=False, pause_started_at=None,
        )
        LOG.info("[%s] 批量完成：%d 个文件", tid, len(all_files))
    except Exception as e:  # noqa: BLE001
        LOG.exception("[%s] 批量任务失败", tid)
        LEDGER.finish(tid, STATUS_FAILED, stage="失败", paused=False,
                      error=f"{type(e).__name__}: {e}")
    finally:
        _unregister_gate(tid)
        _release_run_slot()


# ---------------------------------------------------------------------------
# 提交入口（与前端请求生命周期解耦：入队 + 起线程即返回）
# ---------------------------------------------------------------------------
def submit_single(audio_path, tier, single_model, preset_key, custom_models, algo, fmt, sr,
                  batch, seg, overlap, shifts, dem_overlap, norm, amp, single_stem,
                  use_native_fp16, use_torch_compile, dereverb=False, dereverb_model=None,
                  dereverb_passes=DEFAULT_DEREVERB_PASSES, dereverb_match=True, dereverb_air=False):
    if not audio_path:
        raise gr.Error("请先上传一个音频文件")
    src = Path(audio_path)
    if not src.exists():
        raise gr.Error(f"输入文件不存在：{src}")

    need = models_needed(tier, single_model, preset_key, custom_models)
    missing = [m for m in need if not model_downloaded(m)]
    if missing:
        raise gr.Error("以下模型未内置到本地模型目录，离线模式下不可用：" + "、".join(missing))
    if dereverb:
        ok, _fn, why = dereverb_model_ready(dereverb_model)
        if not ok:
            raise gr.Error(why)

    song = src.stem
    job_dir = OUTPUT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{song}"
    job_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(src)

    opt = {
        "tier": tier, "single_model": single_model, "preset_key": preset_key,
        "custom_models": custom_models, "algo": algo, "fmt": fmt, "sr": int(sr),
        "batch": int(batch), "seg": int(seg), "overlap": int(overlap),
        "shifts": int(shifts), "dem_overlap": float(dem_overlap),
        "norm": float(norm), "amp": float(amp), "single_stem": single_stem,
        "use_native_fp16": bool(use_native_fp16), "use_torch_compile": bool(use_torch_compile),
        "dereverb": bool(dereverb),
        "dereverb_model": dereverb_model if dereverb else None,
        "dereverb_passes": dereverb_passes_value(dereverb_passes),
        "dereverb_match": bool(dereverb_match),
        "dereverb_air": bool(dereverb_air),
        "models": need, "duration": duration,
    }
    tid = LEDGER.create("单曲分离", src.name, job_dir, duration=duration,
                        params={k: v for k, v in opt.items() if k != "models"})
    queued = run_lock_busy()
    threading.Thread(target=_run_single_task, args=(tid, str(src), song, str(job_dir), opt),
                     daemon=True, name=f"sep-{tid}").start()

    status = (
        f"**{'已排队' if queued else '已开始分离'}** `{tid}`\n\n"
        f"- 源文件：{src.name}（时长 {fmt_sec(duration)}）\n"
        f"- overlap = {int(overlap)} · 高级开关 native_fp16={bool(use_native_fp16)} / "
        f"torch_compile={bool(use_torch_compile)}\n"
        + (f"- 后处理：分离完成后对「人声」轨去回声（{dereverb_model}"
           f" · {dereverb_passes_value(dereverb_passes)} 遍"
           f"{' · 自动等响' if dereverb_match else ''}"
           f"{' · 补回 8kHz 以上高频' if dereverb_air else ''}）\n" if dereverb else "")
        + f"- 输出目录：`{job_dir}`\n\n"
        + ("前面还有一个任务在跑，轮到本任务时自动开始，" if queued else "分离正在进行，")
        + "右侧进度条实时走动，完成后结果会自动出现在本页，也可切到「任务列表」查看全部任务。"
    )
    return ((tid, status, progress_html(LEDGER.get(tid)))
            + tuple(gr.update(visible=False) for _ in range(MAX_STEMS)) + (None, None))


def submit_batch(files_in, tier, single_model, preset_key, fmt, sr, batch, seg, overlap,
                 shifts, dem_overlap, norm, amp, use_native_fp16, use_torch_compile,
                 dereverb=False, dereverb_model=None,
                 dereverb_passes=DEFAULT_DEREVERB_PASSES, dereverb_match=True, dereverb_air=False):
    paths = collect_audio_paths(files_in)
    if not paths:
        raise gr.Error("请上传音频文件或包含音频的文件夹")

    need = models_needed(tier, single_model, preset_key, [])
    missing = [m for m in need if not model_downloaded(m)]
    if missing:
        raise gr.Error("以下模型未内置到本地模型目录，离线模式下不可用：" + "、".join(missing))
    if dereverb:
        ok, _fn, why = dereverb_model_ready(dereverb_model)
        if not ok:
            raise gr.Error(why)

    batch_dir = OUTPUT_DIR / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    opt = {
        "tier": tier, "single_model": single_model, "preset_key": preset_key,
        "fmt": fmt, "sr": int(sr), "batch": int(batch), "seg": int(seg),
        "overlap": int(overlap), "shifts": int(shifts), "dem_overlap": float(dem_overlap),
        "norm": float(norm), "amp": float(amp),
        "use_native_fp16": bool(use_native_fp16), "use_torch_compile": bool(use_torch_compile),
        "dereverb": bool(dereverb),
        "dereverb_model": dereverb_model if dereverb else None,
        "dereverb_passes": dereverb_passes_value(dereverb_passes),
        "dereverb_match": bool(dereverb_match),
        "dereverb_air": bool(dereverb_air),
    }
    tid = LEDGER.create("批量分离", f"{len(paths)} 个音频文件", batch_dir, params=opt)
    queued = run_lock_busy()
    threading.Thread(target=_run_batch_task, args=(tid, [str(p) for p in paths], str(batch_dir), opt),
                     daemon=True, name=f"batch-{tid}").start()

    status = (
        f"**{'已排队' if queued else '已开始批量分离'}** `{tid}`（共 {len(paths)} 个文件）\n\n"
        f"- overlap = {int(overlap)} · 高级开关 native_fp16={bool(use_native_fp16)} / "
        f"torch_compile={bool(use_torch_compile)}\n"
        + (f"- 后处理：分离完成后对「人声」轨去回声（{dereverb_model}"
           f" · {dereverb_passes_value(dereverb_passes)} 遍"
           f"{' · 自动等响' if dereverb_match else ''}"
           f"{' · 补回 8kHz 以上高频' if dereverb_air else ''}）\n" if dereverb else "")
        + f"- 输出目录：`{batch_dir}`\n\n"
        + ("前面还有一个任务在跑，轮到本任务时自动开始，" if queued else "正在按顺序处理这些文件，")
        + "右侧进度条实时走动，完成后结果会自动出现在本页。"
    )
    return tid, status, progress_html(LEDGER.get(tid)), None, None


# ---------------------------------------------------------------------------
# 前端轮询
# ---------------------------------------------------------------------------
_RENDERED = {}   # {(tid, kind): signature} 用于跨重渲染去重

def _task_status_text(t):
    lines = [f"**{t.get('status', '')}** · {t.get('stage', '')}"]
    lines.append(f"- 进度：{fmt_pct(t.get('progress'))}"
                 + (f" · 剩余约 {fmt_sec(t.get('eta'))}" if t.get("eta") else ""))
    st = t.get("seg_total") or 0
    if st:
        lines.append(f"- 分块：{t.get('seg_done', 0)}/{st}")
    lines.append(f"- 已用：{fmt_sec(t.get('elapsed'))}"
                 + (f" · 模型加载 {fmt_sec(t.get('load_sec'))}" if t.get("load_sec") else "")
                 + (f" · 分离 {fmt_sec(t.get('sep_sec'))}" if t.get("sep_sec") else ""))
    if t.get("output_dir"):
        lines.append(f"- 输出目录：`{t.get('output_dir')}`")
    if t.get("zip"):
        lines.append(f"- 打包：`{t.get('zip')}`")
    if t.get("error"):
        lines.append(f"- 错误：{t.get('error')}")
    if t.get("status") in ACTIVE_STATES:
        lines.append("\n_任务持续运行中，切换标签 / 刷新页面均不会中断，进度会自动刷新。_")
    if t.get("status") == STATUS_PAUSED:
        lines.append("\n_已暂停：点「继续」从当前进度接着跑，结果不会丢。_")
    return "\n".join(lines)


def cleanup_picked(tid):
    """任务列表页：清理选中任务的本机副本，并整表刷新。"""
    msg, _, _ = cleanup_local_outputs(tid)
    return (msg, task_rows(), task_summary_text(), gr.update(choices=task_choices()),
            progress_html(_focus_task()), activity_html())


# ---------------------------------------------------------------------------
# 暂停 / 继续（推理边界处的协作式暂停：不打断当前分段，进度不丢）
# ---------------------------------------------------------------------------
def _is_paused(tid):
    """任务是否处于暂停态（内存闸门优先，进程重启后以台账状态为准）。"""
    if not tid:
        return False
    with _PAUSE_LOCK:
        if tid in _PAUSE_SET:
            return True
    t = LEDGER.get(tid)
    return bool(t and t.get("status") == STATUS_PAUSED)


def _register_gate(tid):
    """为任务登记一个暂停闸门（工作线程启动时调用）。"""
    with _GATE_LOCK:
        _GATES.setdefault(tid, threading.Event())


def _unregister_gate(tid):
    """任务结束，回收闸门。"""
    with _GATE_LOCK:
        _GATES.pop(tid, None)
    with _PAUSE_LOCK:
        _PAUSE_SET.discard(tid)


def _pause_gate(tid):
    """暂停闸门：处于暂停态就在这里停住（分段落边界），「继续」后被唤醒。
    单次等待上限 1.5s 后复查标记，避免漏掉「先暂停再继续」的竞态。"""
    while _is_paused(tid):
        with _GATE_LOCK:
            ev = _GATES.get(tid)
        if ev is None:
            time.sleep(0.3)
            continue
        ev.wait(1.5)
        ev.clear()


def pause_task(tid):
    """暂停任务：置暂停标记，工作线程会在当前分段/文件收尾后停下。"""
    tid = str(tid or "").strip()
    t = LEDGER.get(tid) if tid else None
    if not t:
        t = _focus_task()
        tid = (t or {}).get("id") or ""
    if not t:
        return "当前没有可暂停的任务", progress_html(None), activity_html()
    if t.get("status") not in ACTIVE_STATES:
        return (f"任务 `{tid}` 当前状态为「{t.get('status')}」，无需暂停",
                progress_html(t), activity_html())
    with _PAUSE_LOCK:
        _PAUSE_SET.add(tid)
    LEDGER.update(tid, status=STATUS_PAUSED, paused=True, pause_started_at=time.time(),
                  stage="已暂停 · 点「继续」从当前进度接着跑", eta=None)
    LOG.info("[%s] 用户暂停任务", tid)
    return ("已暂停：正在跑的这一分段会做完后停下，点「继续」即可从当前进度接着跑（结果不会丢）。",
            progress_html(LEDGER.get(tid)), activity_html())


def resume_task(tid):
    """继续任务：清暂停标记并唤醒等待中的工作线程。"""
    tid = str(tid or "").strip()
    t = LEDGER.get(tid) if tid else None
    if not t:
        t = _focus_task()
        tid = (t or {}).get("id") or ""
    if not t:
        return "当前没有可继续的任务", progress_html(None), activity_html()
    was_paused = t.get("status") == STATUS_PAUSED
    with _GATE_LOCK:
        ev = _GATES.get(tid)
    with _PAUSE_LOCK:
        _PAUSE_SET.discard(tid)
    if was_paused and ev is None:
        # 暂停标记还在、但工作线程已不存在（服务重启过）——不能假装在跑
        LEDGER.update(tid, status=STATUS_FAILED, paused=False, pause_started_at=None,
                      stage="暂停期间服务重启，任务未能续跑", eta=None,
                      error=t.get("error") or "服务重启导致任务中断")
        LOG.info("[%s] 暂停任务所在进程已退出，无法继续", tid)
        return ("该任务的运行进程已退出（服务重启过），无法从暂停处继续，请重新提交。",
                progress_html(LEDGER.get(tid)), activity_html())
    if was_paused:
        LEDGER.update(tid, status=STATUS_RUNNING, paused=False, pause_started_at=None,
                      stage="已继续 · 接着跑", eta=None)
    if ev is not None:
        ev.set()
    LOG.info("[%s] 用户继续任务", tid)
    return (f"已继续：`{tid}` 从暂停处接着跑。",
            progress_html(LEDGER.get(tid)), activity_html())


# ---------------------------------------------------------------------------
# 任务详情导出（参数 / 模型 / 耗时 / 输出路径 / 日志）
# ---------------------------------------------------------------------------
def _log_file_path():
    """定位日志文件（从 logging 的 FileHandler 上取，不依赖硬编码路径）。"""
    for h in list(LOG.handlers) + list(logging.getLogger().handlers):
        fn = getattr(h, "baseFilename", None)
        if fn:
            return Path(fn)
    return None


def _log_tail(tid, n=200):
    """抓取日志中与该任务相关的最后 n 行。"""
    p = _log_file_path()
    if not p or not p.exists():
        return "（日志文件尚未生成）"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:  # noqa: BLE001
        return f"（日志读取失败：{e}）"
    hit = [ln for ln in lines if tid and tid in ln]
    if not hit:
        return "（未在日志中找到该任务的记录）"
    return "\n".join(hit[-n:])


def _fmt_or_dash(ts):
    return fmt_ts(ts) if ts else "—"


def _dereverb_line(t):
    """导出用：该任务是否做了人声去回声后处理。"""
    p = t.get("params") or {}
    if p.get("dereverb") or t.get("dereverb_model"):
        return f"已开启（{t.get('dereverb_model') or p.get('dereverb_model') or '默认模型'}）"
    return "未开启"


def _task_detail_md(t):
    """把单个任务的完整信息整理成 Markdown（用于导出）。"""
    p = t.get("params") or {}
    files = list(t.get("files") or [])
    lines = [
        f"# 任务详情 · {t.get('id')}",
        "",
        "## 基本信息",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| 任务 ID | {t.get('id')} |",
        f"| 模式 | {t.get('mode')} |",
        f"| 源文件 | {t.get('source')} |",
        f"| 状态 | {t.get('status')} |",
        f"| 阶段 | {t.get('stage')} |",
        f"| 进度 | {fmt_pct(t.get('progress'))} |",
        f"| 音频时长 | {fmt_sec(t.get('duration')) if t.get('duration') else '未知'} |",
        f"| 创建时间 | {_fmt_or_dash(t.get('created_at'))} |",
        f"| 开始时间 | {_fmt_or_dash(t.get('started_at'))} |",
        f"| 结束时间 | {_fmt_or_dash(t.get('finished_at'))} |",
        f"| 总耗时 | {fmt_sec(t.get('elapsed'))} |",
        f"| 模型加载耗时 | {fmt_sec(t.get('load_sec')) if t.get('load_sec') is not None else '—'} |",
        f"| 推理耗时 | {fmt_sec(t.get('sep_sec')) if t.get('sep_sec') is not None else '—'} |",
        f"| 人声去回声 | {_dereverb_line(t)} |",
        f"| 去回声耗时 | {fmt_sec(t.get('dereverb_sec')) if t.get('dereverb_sec') is not None else '—'} |",
        f"| 去回声产物 | {t.get('dereverb_out') or '—'} |",
        f"| 分块进度 | {t.get('seg_done', 0)} / {t.get('seg_total', 0)} |",
        f"| 输出目录 | {t.get('output_dir')} |",
        f"| 本机副本 | {'已清理' if t.get('local_cleaned') else '保留中'} |",
        f"| 打包文件 | {t.get('zip') or '（无）'} |",
        "",
        "## 分离参数",
        "",
        "| 参数 | 值 |",
        "| --- | --- |",
    ]
    if p:
        for k, v in p.items():
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("| （无记录） |  |")
    lines += ["", "## 输出文件", ""]
    if files:
        for i, f in enumerate(files, 1):
            ok = "存在" if Path(str(f)).exists() else "已清理/缺失"
            lines.append(f"{i}. `{f}`（{ok}）")
    else:
        lines.append("（暂无输出文件）")
    lines += ["", "## 运行环境与性能设置", "",
              f"- CPU 线程（torch）：{torch.get_num_threads()} / 逻辑核心 {os.cpu_count()}",
              f"- cudnn.benchmark：{bool(getattr(torch.backends.cudnn, 'benchmark', False))}",
              f"- 线程环境变量：OMP={os.environ.get('OMP_NUM_THREADS', '（未设置）')}、"
              f"MKL={os.environ.get('MKL_NUM_THREADS', '（未设置）')}",
              f"- GPU：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到 CUDA'}",
              f"- 性能配置来源：{PERF_PATH}", ""]
    lines += ["", "## 运行日志（与该任务相关的最后若干行）", "", "```text",
              _log_tail(t.get("id") or ""), "```", ""]
    if t.get("error"):
        lines += ["## 错误信息", "", "```text", str(t.get("error")), "```", ""]
    return "\n".join(lines)


def export_task_detail(tid):
    """导出单个任务的详情到 Markdown 文件（结果产物目录）。"""
    tid = str(tid or "").strip()
    t = LEDGER.get(tid) if tid else None
    if not t:
        t = _focus_task()
        tid = (t or {}).get("id") or ""
    if not t:
        return "暂无可导出的任务详情（请先提交任务或选择一个任务）", gr.update()
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch for ch in tid if (ch.isalnum() or ch in "-_")) or "task"
        out = EXPORT_DIR / f"{safe}_任务详情.md"
        out.write_text(_task_detail_md(t), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        LOG.exception("导出任务详情失败")
        return f"导出失败：{type(e).__name__}: {e}", gr.update()
    LOG.info("[%s] 任务详情已导出：%s", tid, out)
    return f"已导出任务详情：`{out}`（右侧文件卡片可下载）", gr.update(value=str(out), visible=True)


def export_all_tasks():
    """导出全部任务详情（单文件 Markdown + 打包 zip）。"""
    tasks = LEDGER.list_tasks()
    if not tasks:
        return "暂无任务记录，无可导出内容", gr.update()
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        md = EXPORT_DIR / f"全部任务详情_{stamp}.md"
        body = [f"# VocalSeparator · 全部任务详情（导出时间 {_fmt_or_dash(time.time())}）",
                "", f"共 {len(tasks)} 个任务。", ""]
        for t in tasks:
            body += [_task_detail_md(t), "\n---\n"]
        md.write_text("\n".join(body), encoding="utf-8")
        z = zip_files([md], EXPORT_DIR / f"任务详情导出_{stamp}.zip")
    except Exception as e:  # noqa: BLE001
        LOG.exception("导出全部任务详情失败")
        return f"导出失败：{type(e).__name__}: {e}", gr.update()
    LOG.info("全部任务详情已导出：%s（%d 个任务）", md, len(tasks))
    return (f"已导出 {len(tasks)} 个任务的详情：`{md}`；打包：`{z}`",
            gr.update(value=str(z), visible=True))


# ---------------------------------------------------------------------------
# 下载后清理：浏览器下载完 → 服务端删掉输出目录里的这份本地副本
# ---------------------------------------------------------------------------
def _to_recycle_bin(fpath):
    """把文件移入回收站（Windows 原生 Shell API，无第三方依赖）。
    移入回收站意味着「误删可找回」，比直接永久删除更安全；失败返回 False。"""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND),
                        ("wFunc", wintypes.UINT),
                        ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR),
                        ("fFlags", wintypes.WORD),
                        ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p),
                        ("lpszProgressTitle", wintypes.LPCWSTR)]

        FO_DELETE = 3
        FOF_SILENT = 0x0004
        FOF_NOCONFIRMATION = 0x0010
        FOF_ALLOWUNDO = 0x0040
        FOF_NOERRORUI = 0x0400

        try:  # 工作线程可能尚未初始化 COM，失败也不影响后续逻辑
            ctypes.windll.ole32.CoInitializeEx(None, 0x2)
        except Exception:  # noqa: BLE001
            pass

        op = _SHFILEOPSTRUCTW(
            None, FO_DELETE, str(fpath) + "\0\0", None,
            FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI,
            False, None, None)
        rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return rc == 0 and not op.fAnyOperationsAborted
    except Exception:  # noqa: BLE001
        return False


def _remove_with_retry(fpath, tries=4, delay=0.8, recycle=True):
    """删除单个本地文件：优先移入回收站（可恢复），失败再永久删除；
    被浏览器 / 播放器占用时短暂重试。"""
    p = Path(str(fpath))
    for _ in range(max(1, tries)):
        try:
            if not p.exists():
                return False
        except OSError:
            return False
        if recycle and _to_recycle_bin(p):
            return True
        try:
            p.unlink()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(delay)
    return False


def _pending_save(items):
    try:
        PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        PENDING_PATH.write_text(json.dumps(sorted(items), ensure_ascii=False, indent=1),
                                encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        LOG.warning("待清理清单写入失败：%s", e)


def _pending_load():
    try:
        if PENDING_PATH.exists():
            raw = (PENDING_PATH.read_text(encoding="utf-8") or "").strip()
            if raw:
                data = json.loads(raw)
                if isinstance(data, list):
                    return set(str(x) for x in data)
    except Exception as e:  # noqa: BLE001
        LOG.warning("待清理清单读取失败：%s", e)
    return set()


def _mark_task_cleaned(path):
    """某份本地副本被删掉后，标记对应任务（用于任务列表展示）。"""
    try:
        s = str(path)
        for t in LEDGER.list_tasks()[:80]:
            if s == str(t.get("zip") or "") or s in [str(x) for x in (t.get("files") or [])]:
                if not t.get("local_cleaned"):
                    LEDGER.update(t["id"], local_cleaned=True)
    except Exception:  # noqa: BLE001
        pass


def vs_mark_downloaded(raw_path):
    """浏览器端下载完成后的回调入口：登记并清理输出目录里的这份副本。
    只允许操作输出目录内的文件（防止路径穿越）。"""
    rel = (raw_path or "").replace("\\", "/").strip()
    if not rel:
        return "bad"
    try:
        root = OUTPUT_DIR.resolve()
        cand = Path(rel)
        target = cand.resolve() if cand.is_absolute() else (root / rel).resolve()
        target.relative_to(root)
    except Exception:  # noqa: BLE001
        LOG.warning("拒绝清理输出目录之外的文件：%s", raw_path)
        return "denied"
    if not target.is_file():
        return "ok"
    with _PENDING_LOCK:
        _PENDING.add(str(target))
        _pending_save(_PENDING)
    gone = _remove_with_retry(target, tries=3, delay=0.6)
    if gone:
        with _PENDING_LOCK:
            _PENDING.discard(str(target))
            _pending_save(_PENDING)
        _mark_task_cleaned(target)
        global _CLEANED_TOTAL
        _CLEANED_TOTAL += 1
    LOG.info("浏览器已下载 %s → 本机副本%s", target.name,
             "已删除" if gone else "被占用，稍后重试")
    return "ok"


def _sweep_pending():
    """重试清理「已下载但当时被占用」的本地副本。"""
    with _PENDING_LOCK:
        items = sorted(_PENDING)
    if not items:
        return 0
    done = 0
    for s in items:
        p = Path(s)
        try:
            if not p.exists():
                with _PENDING_LOCK:
                    _PENDING.discard(s)
                continue
            if _remove_with_retry(p, tries=1, delay=0.3):
                with _PENDING_LOCK:
                    _PENDING.discard(s)
                _mark_task_cleaned(p)
                global _CLEANED_TOTAL
                _CLEANED_TOTAL += 1
                done += 1
        except Exception:  # noqa: BLE001
            continue
    if done:
        with _PENDING_LOCK:
            _pending_save(_PENDING)
    return done


def cleanup_local_outputs(tid):
    """手动清理：删掉本机输出目录里该任务留下的成品与打包（浏览器下载的那份不受影响）。"""
    tid = str(tid or "").strip()
    if not tid:
        # 没有明确任务时绝不猜测目标，避免误删最近一次任务的成果
        return ("请先选择要清理的任务：在「任务列表」里选一行后点「清理本机副本」，"
                "或先提交一次分离任务。", gr.update(), gr.update())
    t = LEDGER.get(tid)
    if not t:
        return f"未找到任务 `{tid}`，请刷新后重试。", gr.update(), gr.update()
    targets = []
    if t.get("zip"):
        targets.append(str(t.get("zip")))
    targets += [str(f) for f in (t.get("files") or [])]
    out_dir = str(t.get("output_dir") or "")
    if out_dir:
        targets.append(str(Path(out_dir) / "输出列表.txt"))
    removed, missed = [], []
    for s in targets:
        p = Path(s)
        if not s or not p.is_file():
            continue
        if _remove_with_retry(p, tries=1, delay=0.2):
            removed.append(p.name)
        else:
            missed.append(p.name)
    LEDGER.update(tid, local_cleaned=True)
    if not removed and not missed:
        return (f"任务 `{tid}` 的本机副本已不存在（可能此前已清理或尚未产出）。",
                gr.update(), gr.update())
    msg = f"已删除本机副本 {len(removed)} 个" + (f"，{len(missed)} 个被占用未删：{'、'.join(missed)}" if missed else "")
    if removed:
        msg += f"：{'、'.join(removed[:8])}" + ("…" if len(removed) > 8 else "")
    msg += "\n\n浏览器里已下载的那份不受影响；输出文件夹现在只剩浏览器没有下载过的内容。"
    LOG.info("[%s] 手动清理本机副本：%s", tid, removed)
    return msg, gr.update(value=None, visible=False), gr.update(value=None, visible=False)


# ---------------------------------------------------------------------------
# 前端轮询
# ---------------------------------------------------------------------------
def _stems_update(files):
    ups = []
    for i in range(MAX_STEMS):
        if i < len(files):
            ups.append(gr.update(value=files[i], visible=True, label=Path(files[i]).stem))
        else:
            ups.append(gr.update(visible=False))
    return ups


def _poll_stems(tid, t, force_hide=False, sid=None):
    """主干试听区：只在任务完成且未清理时展示（避免每 2s 无谓重渲染打断试听）。"""
    files = list(t.get("files") or []) if t else []
    show = bool(files) and t.get("status") == STATUS_DONE and not t.get("local_cleaned") \
        and not force_hide
    sig = (t.get("status"), bool(show), tuple(files) if show else ())
    # 注意：读写必须用同一个复合键，且缓存按会话隔离（否则新页面拿不到试听区）
    d = _sess(sid)
    if d.get(("stems", tid)) == sig:
        return [gr.update() for _ in range(MAX_STEMS)]
    d[("stems", tid)] = sig
    if not show:
        return [gr.update(visible=False) for _ in range(MAX_STEMS)]
    return _stems_update(files)


def _poll_files(tid, t, force_hide=False, sid=None):
    files = list(t.get("files") or []) if t else []
    show = bool(files) and t.get("status") == STATUS_DONE and not t.get("local_cleaned") \
        and not force_hide
    zip_show = bool(t and t.get("zip")) and t.get("status") == STATUS_DONE \
        and not t.get("local_cleaned") and not force_hide
    sig = (bool(show), bool(zip_show), tuple(files) if show else (), (t or {}).get("zip"))
    d = _sess(sid)
    if d.get(("files", tid)) == sig:
        return gr.update(), gr.update()
    d[("files", tid)] = sig
    return (gr.update(value=files if show else None, visible=show),
            gr.update(value=(t.get("zip") if zip_show else None), visible=zip_show))


def poll_single(tid):
    """单曲页轮询：进度条 / 状态文本 / 试听与输出（返回顺序固定，勿改）。"""
    t = LEDGER.get(tid) if tid else _focus_task()
    if not t:
        return (tuple(gr.update(visible=False) for _ in range(MAX_STEMS))
                + (gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                   "等待开始…", progress_html(None), activity_html()))
    stems_upd = _poll_stems(tid or t.get("id"), t)
    files_upd, zip_upd = _poll_files(tid or t.get("id"), t)
    return (tuple(stems_upd) + (files_upd, zip_upd, _task_status_text(t),
                                progress_html(t), activity_html()))


def poll_batch(tid):
    """批量页轮询（返回顺序固定，勿改）。"""
    t = LEDGER.get(tid) if tid else _focus_task()
    if not t:
        return (gr.update(value=None, visible=False), gr.update(value=None, visible=False),
                "等待开始…", progress_html(None), activity_html())
    files_upd, zip_upd = _poll_files(tid or t.get("id"), t)
    return (files_upd, zip_upd, _task_status_text(t), progress_html(t), activity_html())


def task_rows():
    rows = []
    for t in LEDGER.list_tasks()[:200]:
        cleaned = bool(t.get("local_cleaned"))
        rows.append([
            t.get("id") or "",
            t.get("mode") or "",
            Path(str(t.get("source") or "")).name,
            t.get("status") or "",
            fmt_pct(t.get("progress")),
            fmt_sec(t.get("elapsed")) + ("（暂停中）" if t.get("status") == STATUS_PAUSED else ""),
            fmt_sec(t.get("eta")) if t.get("eta") is not None else "",
            str(t.get("file_count") or 0),
            "已清理" if cleaned else "保留中",
            t.get("output_dir") or "",
        ])
    return rows


def task_choices():
    out = []
    for t in LEDGER.list_tasks()[:200]:
        label = f"{t.get('id')} · {t.get('mode')} · {t.get('status')} · {Path(str(t.get('source') or '')).name}"
        out.append((label, t.get("id")))
    return out


def task_summary_text():
    tasks = LEDGER.list_tasks()
    if not tasks:
        return "暂无任务记录。提交分离任务后，这里会持续记录进度、耗时与输出位置。"
    active = [t for t in tasks if t.get("status") in ACTIVE_STATES]
    paused = [t for t in tasks if t.get("status") == STATUS_PAUSED]
    done = [t for t in tasks if t.get("status") == STATUS_DONE]
    failed = [t for t in tasks if t.get("status") == STATUS_FAILED]
    parts = [f"共 **{len(tasks)}** 个任务",
             f"运行中 {len(active)}",
             f"暂停 {len(paused)}",
             f"完成 {len(done)}",
             f"失败 {len(failed)}"]
    if _CLEANED_TOTAL:
        parts.append(f"已自动清理本机副本 {_CLEANED_TOTAL} 个")
    return " · ".join(parts) + f"\n\n输出目录：`{OUTPUT_DIR}`"


def poll_task_list():
    _sweep_pending()
    return [task_rows(), task_summary_text(), gr.update(choices=task_choices()),
            progress_html(_focus_task()), activity_html()]


# ---------------------------------------------------------------------------
# 统一轮询（关键修复）：同一个 Timer 只挂一条事件链，一次 tick 覆盖三页
# ---------------------------------------------------------------------------
_CACHE = {}
_SESS = {}          # {sid: {key: sig}}：按页面会话隔离的去重桶
_SESS_MAX = 128     # 会话上限，超出整体清空（去重缓存丢失只会多下发一帧，无副作用）


def _sess(sid):
    """取（必要时新建）某个页面会话专属的去重桶。

    Gradio 每次打开 / 刷新页面都会新建会话，而组件初值只在会话建立那一刻下发；
    若沿用进程级全局缓存去重，后来的会话会被误判成「已经下发过」而收不到更新，
    表现就是进度卡停在「空闲」、试听区 / 输出区空白。所以按会话隔离。
    """
    d = _SESS.get(sid)
    if d is None:
        if len(_SESS) >= _SESS_MAX:
            _SESS.clear()
            _RENDERED.clear()
        d = _SESS[sid] = {}
    return d


def _new_sid():
    """给每个页面会话发一个随机标记（demo.load 时下发一次）。"""
    return uuid.uuid4().hex[:12]


def _cached_upd(key, sig, value, sid=None):
    """内容未变化时返回 no-op 更新（缓存按会话隔离）。

    避免每 2s 重建大组件（进度卡 / 表格 / 下拉框）：重建会清掉滚动位置与选中态，
    用户看到的就是「页面被重置」。
    """
    d = _sess(sid)
    if d.get(key) == sig:
        return gr.update()
    d[key] = sig
    return gr.update(value=value)


def _cached_choices(key, sig, choices, sid=None):
    """下拉框专用：只更新选项，不动当前选中值。"""
    d = _sess(sid)
    if d.get(key) == sig:
        return gr.update()
    d[key] = sig
    return gr.update(choices=choices)


def _quant(v, step):
    """把易变的数值量化成桶（仅用于去重签名，不影响真实显示值）。"""
    try:
        return None if v is None else int(round(float(v) / float(step)))
    except (TypeError, ValueError):
        return None


def _idle_outputs(page, stems_n, sid=None):
    """空闲态（没有任何任务）占位更新：整套只下发一次，之后全部 no-op。

    原实现每 2s 都真实下发 8 个 stems 隐藏 + 2 个 File 清空 + 状态文本，
    这些「其实没变化」的更新正是空闲时 9 秒 330 次 DOM 变更的主因。
    """
    key = ("idle", page)
    d = _sess(sid)
    if d.get(key) == "sent":
        return (tuple(gr.update() for _ in range(stems_n)),
                gr.update(), gr.update(), gr.update())
    d[key] = "sent"
    return (tuple(gr.update(visible=False) for _ in range(stems_n)),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="等待开始…"))


def _arm_page_cache(page, tid, sid=None):
    """任务身份变化（出现 / 切换 / 结束）时，才重新武装该页的占位与状态缓存。

    注意：这里必须按「任务身份是否变化」判断，不能每 tick 无条件清缓存。
    旧实现每 2s pop 一次 status/idle 缓存，导致状态卡文本即使一字未改，
    也要每 2s 真实下发一次（实测每轮仍有多条真实重赋值）。
    另：缓存按会话隔离（见 _sess），否则新开 / 刷新的页面会被误判为
    「已下发过」而拿不到首帧内容。
    """
    key = ("seen", page)
    d = _sess(sid)
    if d.get(key) == tid:
        return
    d[key] = tid
    d.pop(("idle", page), None)
    d.pop(("status", page), None)
    d.pop(("prog", page), None)


def _task_table_sig():
    """任务表去重签名：进度按 10%、已用 / 剩余按 5s 量化。

    表格里显示的仍是真实数值，只是不再每 2s 重建整张表
    （重建会清掉滚动位置与选中行，看起来就是「页面在抖」）。
    """
    out = []
    for t in LEDGER.list_tasks()[:200]:
        out.append((t.get("id"), t.get("mode"), t.get("source"), t.get("status"),
                    _quant(t.get("progress"), 0.1), _quant(t.get("elapsed"), 5.0),
                    _quant(t.get("eta"), 5.0), t.get("file_count"),
                    bool(t.get("local_cleaned")), t.get("output_dir")))
    return tuple(out)


def poll_all(single_tid, batch_tid, sid=None):
    """统一轮询（返回顺序必须与 _poll_outputs 严格一致，勿改）。

    sid = 页面会话标记：所有去重缓存按会话隔离（见 _sess()），
    否则新打开 / 刷新的页面会被上一轮会话留下的缓存挡掉首帧内容
    （表现：进度卡停在「空闲」、试听区 / 输出区空白）。
    """
    _sweep_pending()

    # ---------- 单曲页 ----------
    t_s = LEDGER.get(single_tid) if single_tid else _focus_task()
    _arm_page_cache("single", t_s.get("id") if t_s else None, sid)
    if t_s:
        s_key = single_tid or t_s.get("id")
        stems_upd = tuple(_poll_stems(s_key, t_s, sid=sid))
        s_files, s_zip = _poll_files(s_key, t_s, sid=sid)
        s_status = _task_status_text(t_s)
        s_status_upd = _cached_upd(("status", "single"), s_status, s_status, sid)
        html_s = progress_html(t_s)
    else:
        stems_upd, s_files, s_zip, s_status_upd = _idle_outputs("single", MAX_STEMS, sid)
        html_s = progress_html(None)
    s_prog = _cached_upd(("prog", "single"), html_s, html_s, sid)

    # ---------- 批量页 ----------
    t_b = LEDGER.get(batch_tid) if batch_tid else _focus_task()
    _arm_page_cache("batch", t_b.get("id") if t_b else None, sid)
    if t_b:
        b_key = batch_tid or t_b.get("id")
        b_files_upd, b_zip_upd = _poll_files(b_key, t_b, sid=sid)
        b_status_txt = _task_status_text(t_b)
        b_status_upd = _cached_upd(("status", "batch"), b_status_txt, b_status_txt, sid)
        html_b = progress_html(t_b)
    else:
        # 批量页没有主干试听组件，stems_n=0 时 _idle_outputs 的第一个元素是空元组 ()，
        # 必须丢弃它（原写法把 () 解包给了 b_files_upd，File 组件收到 tuple 就报
        # "expected str, bytes or os.PathLike object, not tuple"）。
        _, b_files_upd, b_zip_upd, b_status_upd = _idle_outputs("batch", 0, sid)
        html_b = progress_html(None)
    b_prog_upd = _cached_upd(("prog", "batch"), html_b, html_b, sid)

    # ---------- 任务列表页 ----------
    rows = task_rows()
    summary = task_summary_text()
    choices = list(task_choices())
    html_tl = progress_html(_focus_task())
    tl_table_upd = _cached_upd(("tl", "table"), _task_table_sig(), rows, sid)
    tl_summary_upd = _cached_upd(("tl", "summary"), summary, summary, sid)
    tl_pick_upd = _cached_choices(("tl", "pick"), tuple(choices), choices, sid)
    tl_prog_upd = _cached_upd(("tl", "prog"), html_tl, html_tl, sid)

    # ---------- 顶部全局活动指示 ----------
    html_ga = activity_html()
    ga_upd = _cached_upd(("global", "activity"), html_ga, html_ga, sid)

    return (stems_upd + (s_files, s_zip, s_status_upd, s_prog)
            + (b_files_upd, b_zip_upd, b_status_upd, b_prog_upd)
            + (tl_table_upd, tl_summary_upd, tl_pick_upd, tl_prog_upd, ga_upd))


def open_task_dir(tid):
    t = LEDGER.get(str(tid or "").strip())
    if not t:
        return "请先在下拉框选择一个任务；或直接点「打开输出目录」看全部结果。"
    return open_dir(t.get("output_dir") or OUTPUT_DIR)


# ---------------------------------------------------------------------------
# 环境信息 / 模型库
# ---------------------------------------------------------------------------
def env_report():
    lines = []
    try:
        lines.append(f"- GPU：{torch.cuda.get_device_name(0)}（显存 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB）")
        lines.append(f"- CUDA 可用：{torch.cuda.is_available()} · torch {torch.__version__} · CUDA {torch.version.cuda}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- GPU 信息读取失败：{e}")
    try:
        import onnxruntime as ort

        lines.append(f"- onnxruntime {ort.__version__} · providers {ort.get_available_providers()}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- onnxruntime 不可用：{e}")
    ffmpeg = BIN_DIR / "ffmpeg.exe"
    if ffmpeg.exists():
        try:
            out = subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, timeout=15)
            lines.append(f"- FFmpeg：{out.stdout.splitlines()[0][:80]}")
        except Exception:  # noqa: BLE001
            lines.append("- FFmpeg：已就位（版本读取失败）")
    else:
        lines.append("- FFmpeg：缺失")
    lines.append(f"- 模型目录：`{MODEL_DIR}`（{len(list(MODEL_DIR.glob('*')))} 个文件）")
    lines.append(f"- 输出目录：`{OUTPUT_DIR}`")
    ok = _TQDM_HOOK["installed"]
    lines.append(f"- 真实进度 hook：{'已启用（真实分块进度 + ETA）' if ok else '未启用，回退为时长估算'}"
                 + ("" if ok else f"（{_TQDM_HOOK['reason']}）"))
    builtin_cnt = sum(1 for m in BUILTIN_MODELS if (MODEL_DIR / m).exists())
    lines.append(f"- 运行模式：离线（本地内置模型 {builtin_cnt}/{len(BUILTIN_MODELS)} 个，禁止联网下载）")
    lines.append(f"- 性能设置：CPU 线程 {PERF_CFG.get('cpu_threads') or 0}（0=自动，当前 torch "
                 f"{torch.get_num_threads()}）· cudnn.benchmark={bool(PERF_CFG.get('cudnn_benchmark'))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 性能：CPU / GPU 协同设置 + 本机真实实测（不写没测过的结论）
# ---------------------------------------------------------------------------
BENCH_DIR = OUTPUT_DIR / "_perf_bench"


def perf_state_md():
    lines = [
        f"- CPU 逻辑核心：{os.cpu_count()} · torch 线程数：{torch.get_num_threads()}",
        f"- 线程环境变量：OMP={os.environ.get('OMP_NUM_THREADS', '（未设置）')} · "
        f"MKL={os.environ.get('MKL_NUM_THREADS', '（未设置）')}",
        f"- cudnn.benchmark：{bool(getattr(torch.backends.cudnn, 'benchmark', False))}",
        f"- 配置文件：`{PERF_PATH}`（改完立即生效，重启后仍保留）",
    ]
    try:
        import onnxruntime as ort
        lines.append(f"- onnxruntime 执行提供器：{', '.join(ort.get_available_providers())}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- onnxruntime：不可用（{type(e).__name__}）")
    return "\n".join(lines)


def apply_perf_ui(cpu_threads, cudnn_benchmark):
    """保存并立即应用 CPU 线程 / cudnn 设置。"""
    try:
        apply_perf_settings(cpu_threads, cudnn_benchmark, persist=True)
    except Exception as e:  # noqa: BLE001
        return f"应用失败：{type(e).__name__}: {e}"
    return ("已应用并写入 `perf.json`（重启后继续生效）：\n\n" + perf_state_md()
            + "\n\n_说明：这些设置只影响调度策略，不改变分离质量。_")


def _bench_clip(seconds):
    """从 samples 里的样带切一段固定长度测试片段（同一片段用于所有对比组）。"""
    import soundfile as sf

    src = BASE_DIR / "samples" / "test_mix.wav"
    if not src.exists():
        cands = sorted((BASE_DIR / "samples").glob("*.wav")) if (BASE_DIR / "samples").exists() else []
        if not cands:
            raise RuntimeError("缺少 samples/test_mix.wav，无法做实测对比")
        src = cands[0]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / f"clip{int(seconds)}.wav"
    if out.exists():
        return out, float(seconds)
    import numpy as np

    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    need = int(seconds * sr)
    if len(data) < need:
        import math
        data = np.tile(data, (math.ceil(need / max(1, len(data))), 1))
    sf.write(str(out), data[:need], sr)
    return out, float(seconds)


def _bench_run(tag, clip, dur, cpu_threads, cudnn_benchmark, batch, overlap, model_file, base_state):
    """跑一组配置并返回 (秒, 相对倍率)。线程数 / cudnn 设置在这里临时切换。"""
    out_dir = BENCH_DIR / f"out_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_perf_settings(cpu_threads, cudnn_benchmark, persist=False)
    sep = Separator(
        log_level=logging.ERROR,
        model_file_dir=str(MODEL_DIR), output_dir=str(out_dir), output_format="wav",
        use_autocast=True, use_native_fp16=False,
        mdxc_params=_mdxc_params(0, int(batch), int(overlap)),
    )
    t0 = time.time()
    sep.load_model(model_filename=model_file)
    load = time.time() - t0
    t1 = time.time()
    sep.separate(str(clip))
    sec = time.time() - t1
    del sep
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return load, sec, sec / max(0.001, dur)


def bench_perf_ui(cpu_threads, cudnn_benchmark):
    """真实实测：同一段音频、同一模型，对比几组 CPU / GPU 协同设置的实际耗时。

    会临时占用 GPU（与分离任务互斥），测完把用户设置恢复回去；不写没有实测支撑的结论。"""
    if not RUN_LOCK.acquire(blocking=False):
        return ("已有分离任务在运行：实测会和它抢 GPU 与显存，此时的数据不可信。"
                "请等任务结束后再点「实测对比」。")
    model_file = DEFAULT_MODEL if (MODEL_DIR / DEFAULT_MODEL).exists() else None
    if not model_file:
        RUN_LOCK.release()
        return f"本机没找到默认模型 {DEFAULT_MODEL}，无法实测。"
    rows, notes = [], []
    try:
        clip, dur = _bench_clip(60)
        auto_threads = 0
        n_cpu = os.cpu_count() or 8
        configs = [
            ("当前设置", int(cpu_threads or 0), bool(cudnn_benchmark), 8, 8),
            (f"CPU 线程 = 4", 4, bool(cudnn_benchmark), 8, 8),
            (f"CPU 线程 = {max(2, n_cpu // 2)}（半数核心）", max(2, n_cpu // 2), bool(cudnn_benchmark), 8, 8),
            ("关闭 cudnn.benchmark", int(cpu_threads or 0), False, 8, 8),
            ("batch_size = 16", int(cpu_threads or 0), bool(cudnn_benchmark), 16, 8),
            ("overlap = 32（对照：冗余推理）", int(cpu_threads or 0), bool(cudnn_benchmark), 8, 32),
        ]
        seen, uniq = set(), []
        for c in configs:
            key = (c[1], c[2], c[3], c[4])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        base = None
        total = len(uniq) + 1
        for i, (tag, th, cb, bs, ov) in enumerate(uniq, 1):
            LOG.info("[实测 %d/%d] %s：threads=%s cudnn=%s batch=%s overlap=%s",
                     i, total, tag, th, cb, bs, ov)
            load, sec, per = _bench_run(tag, clip, dur, th, cb, bs, ov, model_file, None)
            if base is None:
                base = per
            rows.append((tag, th, cb, bs, ov, load, sec, per, (base / per) if per else 0))
    except Exception as e:  # noqa: BLE001
        LOG.exception("性能实测失败")
        return f"实测失败：{type(e).__name__}: {e}"
    finally:
        try:
            apply_perf_settings(cpu_threads or 0, bool(cudnn_benchmark), persist=False)
        except Exception:  # noqa: BLE001
            pass
        RUN_LOCK.release()

    # 结论：完全基于上面测出来的数字，不做没测过的推断
    per_list = [(r[0], r[7]) for r in rows]
    best = min(per_list, key=lambda x: x[1])
    worst = max(per_list, key=lambda x: x[1])
    hard = [r for r in rows if r[4] == 32]
    base_row = next((r for r in rows if r[4] == 8), None)
    notes.append(f"- 最快：**{best[0]}**（{best[1]:.3f} 秒/秒音频）"
                 f"；最慢：**{worst[0]}**（{worst[1]:.3f} 秒/秒音频）。")
    if hard and base_row and base_row[7]:
        notes.append(f"- overlap 是本机最大的一项速度变量：overlap=32 实测 {hard[0][7]:.3f} 秒/秒音频，"
                     f"overlap=8 实测 {base_row[7]:.3f} 秒/秒音频，相差约 "
                     f"{hard[0][7] / base_row[7]:.1f} 倍——overlap 越大，同一段音频被重复推理的份数越多。")
    spread = (worst[1] / best[1]) if best[1] else 0
    if spread < 1.15:
        notes.append("- 线程数 / cudnn.benchmark / batch_size 这几项在本机差异 < 15%，属于同一档次："
                     "瓶颈在 GPU 推理本身，改这些只能小幅波动，不是「CPU 一起加速」的关键。")
    else:
        notes.append(f"- 线程 / cudnn / batch 各组的最大差异约 {spread:.2f} 倍，"
                     "上表就是本机实测值，请按自己机器的数字选择。")
    lines = ["| 配置 | CPU 线程 | cudnn.benchmark | batch | overlap | 加载(s) | 分离(s) | 秒/秒音频 | 相对速度 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tag, th, cb, bs, ov, load, sec, per, sp in rows:
        lines.append(f"| {tag} | {th or '自动'} | {'开' if cb else '关'} | {bs} | {ov} | "
                     f"{load:.1f} | {sec:.1f} | {per:.3f} | {sp:.2f}× |")
    header = (f"实测片段：`{clip.name}`（{dur:.0f} 秒）· 模型：`{model_file}` · "
              f"同片段连续多次运行，测完已恢复你的设置。")
    return header + "\n\n" + "\n".join(lines) + "\n\n" + "\n".join(notes)


def model_table(keyword=""):
    rows = []
    kw = (keyword or "").strip().lower()
    for it in CATALOG.get("all", []):
        f = it["file"]
        if kw and kw not in f.lower() and kw not in it.get("name", "").lower():
            continue
        peak = it.get("peak", -1)
        rows.append([
            f,
            it.get("name", ""),
            it.get("type", ""),
            round(peak, 2) if isinstance(peak, (int, float)) and peak > 0 else None,
            ", ".join(it.get("stems", [])) or "未标注",
            "已内置（可用）" if model_downloaded(f) else "未内置（离线不可用）",
        ])
    rows.sort(key=lambda r: (r[5] != "已内置（可用）", r[3] is None, -(r[3] or 0)))
    return rows


def open_dir(path):
    target = Path(path or "")
    if not target.exists():
        return f"目录不存在：`{target}`"
    subprocess.Popen(["explorer", str(target)])
    return f"已在资源管理器中打开：`{target}`"


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
CSS = """
/* ===== VocalSeparator · 界面样式（极简 / 无装饰特效 / 仅运动曲线） ===== */
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }

/* ---------- 顶部极简工具条（替代原大标题区） ---------- */
.vs-toolbar {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 2px 2px 12px; margin-bottom: 2px;
  border-bottom: 1px solid var(--border-color-primary, #e5e7eb);
}
.vs-toolbar .vs-status {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 13px; line-height: 1.2; color: var(--body-text-color-subdued, #6b7280);
}
.vs-toolbar .vs-status strong { color: var(--body-text-color, #111827); font-weight: 600; }
.vs-status-dot {
  width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; flex: none;
}
.vs-toolbar .vs-status[data-state="running"] .vs-status-dot { background: #6366f1; }
.vs-toolbar .vs-status[data-state="paused"] .vs-status-dot { background: #f59e0b; }
.vs-toolbar .vs-status[data-state="done"] .vs-status-dot { background: #10b981; }
.vs-toolbar .vs-spacer { flex: 1 1 auto; }
.vs-icon-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 10px; border-radius: 9px; cursor: pointer;
  font-size: 12.5px; color: var(--body-text-color-subdued, #6b7280);
  background: var(--background-fill-primary, #fff);
  border: 1px solid var(--border-color-primary, #e5e7eb);
  transition: transform .22s cubic-bezier(.34, 1.56, .64, 1), border-color .22s ease,
              color .22s ease, box-shadow .22s ease;
}
.vs-icon-btn:hover {
  color: var(--body-text-color, #111827); border-color: rgba(99, 102, 241, .45);
  transform: translateY(-1px); box-shadow: 0 4px 12px rgba(15, 23, 42, .07);
}
.vs-icon-btn:active { transform: translateY(0) scale(.97); }
.vs-icon-btn svg { width: 15px; height: 15px; }

/* ---------- 进度卡片 ---------- */
.vs-prog {
  position: relative; border-radius: 16px; padding: 16px 18px 14px;
  background: var(--background-fill-secondary, #f8fafc);
  border: 1px solid var(--border-color-primary, #e5e7eb);
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
  transition: border-color .35s ease, box-shadow .35s ease;
}
.vs-prog[data-state="done"] { border-color: rgba(16, 185, 129, .38); box-shadow: 0 6px 20px rgba(16, 185, 129, .10); }
.vs-prog[data-state="failed"] { border-color: rgba(244, 63, 94, .38); }
.vs-prog[data-state="paused"] { border-color: rgba(245, 158, 11, .48); box-shadow: 0 6px 20px rgba(245, 158, 11, .10); }
.vs-prog[data-state="idle"] { opacity: .92; }

.vs-prog-top { display: flex; align-items: flex-end; justify-content: space-between; gap: 14px; }
.vs-prog-stage {
  display: flex; align-items: center; gap: 8px; min-width: 0; flex: 1 1 auto;
  font-size: 14.5px; font-weight: 600; color: var(--body-text-color, #111827);
}
.vs-stage-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.vs-prog-pct {
  display: inline-flex; align-items: baseline; gap: 2px; flex: none;
  font-variant-numeric: tabular-nums; letter-spacing: -.02em; line-height: 1;
}
.vs-prog-pct .vs-num {
  font-size: 30px; font-weight: 700;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.vs-prog-pct .vs-unit { font-size: 14px; font-weight: 600; color: var(--body-text-color-subdued, #6b7280); }
.vs-prog-pct.vs-pct-idle .vs-dash { font-size: 26px; font-weight: 700; color: var(--body-text-color-subdued, #9ca3af); }

/* 轨道：拖尾 + 弹性填充（宽度由 JS 逐帧驱动，故不设 transition） */
.vs-track {
  position: relative; height: 12px; margin-top: 13px; border-radius: 999px; overflow: hidden;
  background: color-mix(in srgb, var(--border-color-primary, #e5e7eb) 55%, transparent);
}
.vs-ghost {
  position: absolute; left: 0; top: 0; bottom: 0; width: 0;
  border-radius: 999px; background: rgba(99, 102, 241, .16);
}
.vs-fill {
  position: absolute; left: 0; top: 0; bottom: 0; width: 0;
  border-radius: 999px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .28);
}
.vs-prog[data-state="done"] .vs-fill { background: linear-gradient(90deg, #10b981, #059669); }
.vs-prog[data-state="failed"] .vs-fill { background: linear-gradient(90deg, #f43f5e, #e11d48); }
.vs-prog[data-state="paused"] .vs-fill { background: linear-gradient(90deg, #f59e0b, #d97706); }

/* 分块刻度（每个推理分块点亮一格） */
.vs-segs { display: flex; gap: 3px; align-items: stretch; height: 9px; margin-top: 11px; overflow: hidden; }
.vs-segs > i {
  flex: 1 1 0; border-radius: 2px;
  background: var(--border-color-primary, #e5e7eb); opacity: .55; transform: scaleY(.62);
  transition: transform .3s cubic-bezier(.34, 1.56, .64, 1), opacity .3s ease, background .3s ease;
}
.vs-segs > i.vs-seg-on {
  background: linear-gradient(180deg, #818cf8, #6366f1); opacity: 1; transform: scaleY(1.18);
}
/* 拿不到真实分块数时的「刻度尺」变体：更细更淡，避免被误读成真实分块数 */
.vs-segs[data-kind="scale"] { height: 7px; margin-top: 9px; }
.vs-segs[data-kind="scale"] > i { opacity: .34; transform: scaleY(.44); }
.vs-segs[data-kind="scale"] > i.vs-seg-on { opacity: .74; transform: scaleY(.84); }
.vs-prog[data-state="idle"] .vs-segs > i { opacity: .28; }

/* 信息 chips */
.vs-prog-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; align-items: center; }
.vs-chip {
  display: inline-flex; align-items: center; gap: 5px; padding: 3px 9px; border-radius: 999px;
  font-size: 12px; line-height: 1.55; font-variant-numeric: tabular-nums;
  color: var(--body-text-color-subdued, #6b7280);
  background: var(--background-fill-primary, #fff);
  border: 1px solid var(--border-color-primary, #e5e7eb);
}
.vs-chip.vs-chip-accent { color: #4f46e5; border-color: rgba(99, 102, 241, .35); background: rgba(99, 102, 241, .07); }
.vs-badge-done, .vs-badge-failed, .vs-badge-paused {
  display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 600;
}
.vs-badge-done { color: #047857; background: rgba(16, 185, 129, .12); border: 1px solid rgba(16, 185, 129, .3); }
.vs-badge-failed { color: #be123c; background: rgba(244, 63, 94, .1); border: 1px solid rgba(244, 63, 94, .28); }
.vs-badge-paused { color: #b45309; background: rgba(245, 158, 11, .14); border: 1px solid rgba(245, 158, 11, .34); }

/* 运行中/空闲 圆点 + 三点加载动画（加载指示，非装饰特效） */
.vs-live-dot, .vs-idle-dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.vs-live-dot { background: #6366f1; }
.vs-idle-dot { background: #cbd5e1; }
.vs-dots { display: inline-flex; align-items: center; gap: 3px; color: #6366f1; }
.vs-dots > i {
  width: 4.5px; height: 4.5px; border-radius: 50%; background: currentColor; opacity: .35;
  animation: vsDotWave 1.05s ease-in-out infinite;
}
.vs-dots > i:nth-child(2) { animation-delay: .15s; }
.vs-dots > i:nth-child(3) { animation-delay: .3s; }
.vs-dots[data-off="1"] { display: none; }
@keyframes vsDotWave {
  0%, 100% { transform: translateY(0) scale(1); opacity: .32; }
  40% { transform: translateY(-3.5px) scale(1.12); opacity: 1; }
}

/* ---------- 全局活动指示（任意标签页可见，单行） ---------- */
.vs-float {
  display: flex; align-items: center; gap: 9px; flex-wrap: wrap;
  font-size: 13px; color: var(--body-text-color-subdued, #6b7280);
  padding: 7px 12px; border-radius: 10px;
  background: var(--background-fill-secondary, #f8fafc);
  border: 1px solid var(--border-color-primary, #e5e7eb);
}
.vs-float[data-state="idle"] { display: none; }
.vs-mini-track {
  position: relative; display: inline-block; width: 96px; height: 6px; border-radius: 999px; overflow: hidden;
  background: color-mix(in srgb, var(--border-color-primary, #e5e7eb) 60%, transparent);
}
.vs-mini-fill {
  position: absolute; left: 0; top: 0; bottom: 0; width: 0; border-radius: 999px;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
}
.vs-float .vs-num { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--body-text-color, #374151); }
.vs-float .vs-unit { margin-left: -5px; }

/* ---------- 一次性动画（弹性入场 / 完成强调） ---------- */
/* 防回归（2026-09-29 修「折叠面板展开后整块变白」）：
   Gradio 的隐藏占位层（.hide / data-testid="status-tracker"）是「绝对定位 + 白底 + z-index:30」，
   一旦被挂上带 fill:both 的入场动画，它的 opacity 就会从 0 动画到 1 并永久停在 1，
   于是整块折叠面板（连标题行）会被这层白布盖住 → 表现为点开后全空白。
   这里兜底：隐藏层永远不吃一次性动画。 */
.hide.vs-in, .hide.vs-pop, .hide.vs-fade,
[data-testid="status-tracker"].vs-in, [data-testid="status-tracker"].vs-pop,
[data-testid="status-tracker"].vs-fade {
  animation: none !important;
}
.vs-in { animation: vsIn .34s cubic-bezier(.22, 1, .36, 1) both; }
@keyframes vsIn {
  from { opacity: 0; transform: translateY(9px) scale(.985); }
  to { opacity: 1; transform: none; }
}
.vs-pop { animation: vsPop .58s cubic-bezier(.34, 1.56, .64, 1) both; }
@keyframes vsPop {
  0% { transform: scale(1); }
  32% { transform: scale(1.055); }
  66% { transform: scale(.988); }
  100% { transform: scale(1); }
}

/* 下载清理提示 */
.vs-toast {
  position: fixed; right: 22px; bottom: 22px; z-index: 60;
  padding: 9px 14px; border-radius: 10px; font-size: 13px;
  color: #065f46; background: rgba(16, 185, 129, .14);
  border: 1px solid rgba(16, 185, 129, .36);
  box-shadow: 0 10px 26px rgba(15, 23, 42, .12);
  animation: vsToastIn .3s cubic-bezier(.22, 1, .36, 1) both;
}
@keyframes vsToastIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

/* 标签页面板：仅透明度淡入（不动 transform，避免整块面板重排卡顿） */
.vs-fade { animation: vsFade .2s ease-out both; }
@keyframes vsFade { from { opacity: .55; } to { opacity: 1; } }

/* 按钮 / 折叠面板的轻量交互反馈（.vs-hover-lift 由前端脚本挂到真实存在的元素上） */
.vs-hover-lift {
  transition: transform .2s cubic-bezier(.34, 1.56, .64, 1), box-shadow .2s ease,
              background-color .2s ease;
}
button.vs-hover-lift:hover { transform: translateY(-1.5px); box-shadow: 0 6px 16px rgba(15, 23, 42, .10); }
button.vs-hover-lift:active { transform: translateY(0) scale(.985); }
.label-wrap.vs-hover-lift { border-radius: 10px; }
.label-wrap.vs-hover-lift:hover { background: var(--background-fill-secondary, #f8fafc); }

/* ---------- 统一观感：圆角 / 间距 / 层次（一次改全站，不堆特效） ---------- */
.gradio-container {
  --block-radius: 14px;
  --block-label-radius: 12px;
  --input-radius: 12px;
  --button-small-radius: 10px;
  --button-large-radius: 12px;
  --layout-gap: 10px;
  --form-gap-width: 10px;
  --block-border-width: 1px;
  --block-shadow: none;
  --section-header-text-size: 15px;
  --body-text-size: 14px;
  padding: 18px 16px 42px;
}
/* 页面顶部一层极淡主色渐变，建立视觉层次（不引入任何动画） */
gradio-app { background-image: linear-gradient(180deg, rgba(99, 102, 241, .05), rgba(99, 102, 241, 0) 340px); }
.gradio-container .prose { font-size: 13.5px; line-height: 1.68; }
.gradio-container .prose h2 { font-size: 17px; margin: 18px 0 8px; letter-spacing: -.01em; }
.gradio-container .prose h3 { font-size: 15px; margin: 15px 0 6px; }
.gradio-container .prose code {
  padding: 1px 5px; border-radius: 6px; font-size: 12.5px;
  background: rgba(99, 102, 241, .08); border: 1px solid rgba(99, 102, 241, .14);
}
/* 标签页导航：Gradio 6 的真实结构是 .tabs > .tab-wrapper > .tab-container > button[role=tab]，
   旧版 .tab-nav 选择器在此版本已失效（这正是标签栏与全站风格割裂的原因）。 */
.gradio-container .tabs > .tab-wrapper {
  border-bottom: 1px solid var(--border-color-primary, #e5e7eb) !important;
  padding: 0 !important;
}
.gradio-container .tabs > .tab-wrapper > .tab-container {
  gap: 4px !important; padding: 6px 2px 0 !important; flex-wrap: wrap !important;
  background: transparent !important; border: none !important;
}
.gradio-container .tabs .tab-container button[role="tab"] {
  border: none !important; outline: none !important;
  border-radius: 10px 10px 0 0 !important;
  padding: 7px 13px !important; margin: 0 !important;
  font-size: 13.5px !important; font-weight: 600 !important; letter-spacing: 0 !important;
  color: var(--body-text-color-subdued, #6b7280) !important;
  background: transparent !important;
  box-shadow: inset 0 -2px 0 0 transparent !important;
  transition: color .18s ease, background .18s ease, box-shadow .18s ease;
}
.gradio-container .tabs .tab-container button[role="tab"]:hover {
  color: var(--body-text-color, #111827) !important;
  background: var(--background-fill-secondary, #f8fafc) !important;
}
.gradio-container .tabs .tab-container button[role="tab"].selected {
  color: var(--body-text-color, #111827) !important;
  background: rgba(99, 102, 241, .09) !important;
  box-shadow: inset 0 -2px 0 0 var(--color-accent, #6366f1) !important;
}
.tabitem { padding: 14px 2px 2px !important; }
/* 主操作按钮：主色实心 + 悬浮微抬升 */
button.vs-cta { border: none !important; color: #fff !important;
  background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
  box-shadow: 0 6px 16px rgba(99, 102, 241, .26) !important;
  transition: transform .2s cubic-bezier(.34, 1.56, .64, 1), box-shadow .2s ease !important; }
button.vs-cta:hover { transform: translateY(-1.5px); box-shadow: 0 10px 22px rgba(99, 102, 241, .32) !important; }
button.vs-cta:active { transform: translateY(0) scale(.99); }
/* 表格：更紧凑、表头对比更清楚 */
.gradio-container table { font-size: 12.5px !important; }
.gradio-container thead th { background: var(--background-fill-secondary, #f8fafc) !important;
  font-weight: 600 !important; }
/* 折叠面板标题 */
.gradio-container .label-wrap > span { font-weight: 600 !important; }

/* 窄屏适配 */
@media (max-width: 720px) {
  .vs-prog-top { flex-direction: column; align-items: flex-start; gap: 6px; }
  .vs-prog-pct .vs-num { font-size: 26px; }
}

/* 尊重系统「减少动态效果」 */
@media (prefers-reduced-motion: reduce) {
  .vs-in, .vs-pop, .vs-fade { animation: none !important; }
  .vs-dots > i { animation: none !important; }
  .vs-segs > i, .vs-icon-btn, .vs-hover-lift, button.vs-cta { transition: none !important; }
}

/* =========================================================================
   液态玻璃主题 · Liquid Glass（Apple 风格）—— GLASS_V3
   按参考图的 7 层分解落地：背景模糊 → 形状(SDF 圆角) → 染色 → 折射 → 高光 → 色散 → 接触阴影

   三条硬约束（务必保留）：
   1) .block 自身绝不能带 backdrop-filter / filter / transform —— 三者任一都会让
      .block 成为 position:fixed 后代的「包含块」，Gradio 下拉弹层会整体错位到屏外
      （历史 bug）。所以：磨砂挂 ::before，折射/高光挂 ::after，hover 一律不用 transform。
   2) 折射 = JS 为每个 .block 生成 SDF 法线位移图 + SVG feDisplacementMap，结果写进
      CSS 变量 --vs-rf，由 ::after 的 backdrop-filter 引用（Chromium 实测可用）。
      变量缺失时退化成 none，纯 CSS 依然成立。
   3) 视觉克制：高透明底 + 一层磨砂 + 1px 白色高光边。不做流光 / 呼吸 / 循环变色。
   ========================================================================= */

/* ---------- 主题变量 ---------- */
html.vs-glass {
  --vs-glass-blur: 8px;
  --vs-glass-sat: 185%;
  --vs-glass-bright: 1.04;
  --vs-panel-tint: rgba(255,255,255,.28);
  --vs-panel-veil: linear-gradient(135deg, rgba(255,255,255,.34) 0%, rgba(255,255,255,.13) 38%, rgba(255,255,255,.06) 68%, rgba(255,255,255,.22) 100%);
  --vs-panel-inner:
      inset 1px 1px 0 rgba(255,255,255,.72),
      inset -1px -1px 1px rgba(255,255,255,.30),
      inset 0 0 26px rgba(255,255,255,.10),
      inset 0 0 0 1px rgba(255,255,255,.24),
      inset 0 -18px 32px -22px rgba(12,18,38,.20);
  --vs-panel-gloss: linear-gradient(180deg, rgba(255,255,255,.10) 0%, rgba(255,255,255,.01) 24%, rgba(255,255,255,0) 58%, rgba(10,16,34,.05) 100%);
  --vs-panel-line: rgba(255,255,255,.50);
  --vs-panel-line-hi: rgba(255,255,255,.88);
  --vs-panel-shadow: 0 1px 2px rgba(12,18,38,.06), 0 10px 22px -14px rgba(12,18,38,.20), 0 28px 56px -26px rgba(12,18,38,.40);
  --vs-field: rgba(255,255,255,.26);
  --vs-field-line: rgba(255,255,255,.52);
  --vs-field-line-hi: rgba(99,102,241,.55);
  --vs-field-focus: rgba(255,255,255,.44);
  --vs-chip: rgba(255,255,255,.10);
  --vs-glass-tint: var(--vs-panel-tint);
  --vs-glass-tint-hi: rgba(255,255,255,.72);
}
html.vs-glass.dark {
  --vs-glass-sat: 165%;
  --vs-glass-bright: 1.20;
  --vs-panel-tint: rgba(255,255,255,.11);
  --vs-panel-veil: linear-gradient(135deg, rgba(255,255,255,.20) 0%, rgba(255,255,255,.07) 38%, rgba(255,255,255,.03) 68%, rgba(255,255,255,.13) 100%);
  --vs-panel-inner:
      inset 1px 1px 0 rgba(255,255,255,.28),
      inset -1px -1px 1px rgba(255,255,255,.12),
      inset 0 0 26px rgba(255,255,255,.06),
      inset 0 0 0 1px rgba(255,255,255,.14),
      inset 0 -18px 32px -22px rgba(0,0,0,.35);
  --vs-panel-gloss: linear-gradient(180deg, rgba(255,255,255,.08) 0%, rgba(255,255,255,.01) 24%, rgba(255,255,255,0) 58%, rgba(0,0,0,.14) 100%);
  --vs-panel-line: rgba(255,255,255,.13);
  --vs-panel-line-hi: rgba(255,255,255,.30);
  --vs-panel-shadow: 0 1px 2px rgba(0,0,0,.42), 0 12px 26px -16px rgba(0,0,0,.55), 0 34px 64px -28px rgba(0,0,0,.90);
  --vs-field: rgba(255,255,255,.06);
  --vs-field-line: rgba(255,255,255,.13);
  --vs-field-line-hi: rgba(129,140,248,.60);
  --vs-field-focus: rgba(255,255,255,.12);
  --vs-chip: rgba(255,255,255,.05);
  --vs-glass-tint: var(--vs-panel-tint);
  --vs-glass-tint-hi: rgba(40,44,54,.72);
}

/* ---------- 背景预设（浅色） ---------- */
html.vs-glass[data-bg="aurora"] {
  --vs-g1: rgba(99,102,241,.44); --vs-g2: rgba(236,72,153,.30);
  --vs-g3: rgba(34,211,238,.36); --vs-g4: rgba(167,139,250,.34);
  --vs-base1: #f7f8fc; --vs-base2: #f2f5fd; --vs-base3: #f3f8fd;
}
html.vs-glass[data-bg="sunset"] {
  --vs-g1: rgba(251,146,60,.44); --vs-g2: rgba(244,63,94,.34);
  --vs-g3: rgba(250,204,21,.30); --vs-g4: rgba(168,85,247,.30);
  --vs-base1: #fdf9f6; --vs-base2: #fdf5f2; --vs-base3: #fbf5fb;
}
html.vs-glass[data-bg="ocean"] {
  --vs-g1: rgba(56,189,248,.42); --vs-g2: rgba(20,184,166,.36);
  --vs-g3: rgba(59,130,246,.32); --vs-g4: rgba(14,165,233,.30);
  --vs-base1: #f4fbfe; --vs-base2: #f1faf8; --vs-base3: #f2f7fd;
}
html.vs-glass[data-bg="mint"] {
  --vs-g1: rgba(52,211,153,.42); --vs-g2: rgba(34,197,94,.30);
  --vs-g3: rgba(45,212,191,.32); --vs-g4: rgba(163,230,53,.26);
  --vs-base1: #f5fdf9; --vs-base2: #f2fdf5; --vs-base3: #f3fdfc;
}
html.vs-glass[data-bg="peach"] {
  --vs-g1: rgba(251,113,133,.36); --vs-g2: rgba(249,168,212,.36);
  --vs-g3: rgba(253,186,116,.34); --vs-g4: rgba(196,181,253,.30);
  --vs-base1: #fdf8f8; --vs-base2: #fdf7fb; --vs-base3: #fdfaf5;
}
html.vs-glass[data-bg="night"] {
  --vs-g1: rgba(99,102,241,.42); --vs-g2: rgba(56,189,248,.32);
  --vs-g3: rgba(139,92,246,.36); --vs-g4: rgba(236,72,153,.26);
  --vs-base1: #f7f8fe; --vs-base2: #f2f6fe; --vs-base3: #f8f5fe;
}
html.vs-glass[data-bg="mono"] {
  --vs-g1: rgba(148,163,184,.34); --vs-g2: rgba(203,213,225,.44);
  --vs-g3: rgba(100,116,139,.22); --vs-g4: rgba(226,232,240,.50);
  --vs-base1: #f8f9fb; --vs-base2: #f3f5f8; --vs-base3: #f7f8fa;
}
html.vs-glass[data-bg="paper"] {
  --vs-g1: rgba(148,163,184,.16); --vs-g2: rgba(203,213,225,.22);
  --vs-g3: rgba(148,163,184,.15); --vs-g4: rgba(226,232,240,.26);
  --vs-base1: #fdfdfe; --vs-base2: #ffffff; --vs-base3: #fbfcfd;
}
/* ---------- 背景预设（深色） ---------- */
html.vs-glass.dark[data-bg="aurora"] {
  --vs-g1: rgba(99,90,255,.48); --vs-g2: rgba(236,72,153,.32);
  --vs-g3: rgba(34,211,238,.38); --vs-g4: rgba(139,92,246,.42);
  --vs-base1: #080a11; --vs-base2: #0b0f1c; --vs-base3: #060d15;
}
html.vs-glass.dark[data-bg="sunset"] {
  --vs-g1: rgba(234,88,12,.42); --vs-g2: rgba(190,18,60,.38);
  --vs-g3: rgba(202,138,4,.28); --vs-g4: rgba(126,34,206,.40);
  --vs-base1: #120a07; --vs-base2: #170b0c; --vs-base3: #0f060a;
}
html.vs-glass.dark[data-bg="ocean"] {
  --vs-g1: rgba(2,132,199,.46); --vs-g2: rgba(13,148,136,.38);
  --vs-g3: rgba(37,99,235,.38); --vs-g4: rgba(8,145,178,.38);
  --vs-base1: #040b11; --vs-base2: #041114; --vs-base3: #050a12;
}
html.vs-glass.dark[data-bg="mint"] {
  --vs-g1: rgba(5,150,105,.44); --vs-g2: rgba(22,163,74,.30);
  --vs-g3: rgba(13,148,136,.38); --vs-g4: rgba(77,124,15,.30);
  --vs-base1: #04100b; --vs-base2: #06120c; --vs-base3: #04100c;
}
html.vs-glass.dark[data-bg="peach"] {
  --vs-g1: rgba(190,24,93,.38); --vs-g2: rgba(219,39,119,.30);
  --vs-g3: rgba(217,119,6,.30); --vs-g4: rgba(124,58,237,.34);
  --vs-base1: #130810; --vs-base2: #140911; --vs-base3: #130a06;
}
html.vs-glass.dark[data-bg="night"] {
  --vs-g1: rgba(67,56,202,.46); --vs-g2: rgba(3,105,161,.38);
  --vs-g3: rgba(91,33,182,.40); --vs-g4: rgba(157,23,77,.32);
  --vs-base1: #06080f; --vs-base2: #070a17; --vs-base3: #0a0815;
}
html.vs-glass.dark[data-bg="mono"] {
  --vs-g1: rgba(71,85,105,.40); --vs-g2: rgba(51,65,85,.38);
  --vs-g3: rgba(100,116,139,.26); --vs-g4: rgba(30,41,59,.48);
  --vs-base1: #090b0e; --vs-base2: #0c0f14; --vs-base3: #080a0d;
}
html.vs-glass.dark[data-bg="paper"] {
  --vs-g1: rgba(148,163,184,.12); --vs-g2: rgba(71,85,105,.22);
  --vs-g3: rgba(100,116,139,.12); --vs-g4: rgba(30,41,59,.32);
  --vs-base1: #0b0c0f; --vs-base2: #0f1014; --vs-base3: #0b0c10;
}

/* ---------- 背景层（固定铺满视口；玻璃靠它才有东西可以透过来看） ---------- */
html.vs-glass { background: var(--vs-base1, #eef1fb); }
html.vs-glass.dark { background: var(--vs-base1, #080a11); }
html.vs-glass body { background: transparent !important; }
html.vs-glass body::before {
  content: "";
  position: fixed;
  inset: -12vmax;
  z-index: -3;
  pointer-events: none;
  background-image:
    var(--vs-bg-custom, none),
    radial-gradient(66% 60% at 5% 3%,    var(--vs-g1, rgba(99,102,241,.44)), transparent 72%),
    radial-gradient(62% 56% at 98% 0%,   var(--vs-g2, rgba(236,72,153,.30)), transparent 70%),
    radial-gradient(70% 64% at 86% 100%, var(--vs-g3, rgba(34,211,238,.36)), transparent 72%),
    radial-gradient(60% 56% at 0% 100%,  var(--vs-g4, rgba(167,139,250,.34)), transparent 70%),
    radial-gradient(72% 62% at 52% 46%,  var(--vs-g2, rgba(236,72,153,.18)), transparent 80%),
    linear-gradient(158deg, var(--vs-base1, #f7f8fc) 0%, var(--vs-base2, #f2f5fd) 46%, var(--vs-base3, #f3f8fd) 100%),
    linear-gradient(rgba(0,0,0,var(--vs-bg-dim, 0)) 0 100%);
  background-size: cover, 165% 165%, 165% 165%, 165% 165%, 165% 165%, 175% 175%, 100% 100%, 100% 100%;
  background-position: 50% 50%, 0% 0%, 100% 0%, 86% 100%, 0% 100%, 52% 46%, 0 0, 0 0;
  background-repeat: no-repeat;
  filter: blur(var(--vs-bg-blur, 0px)) saturate(1.10);
}
html.vs-glass gradio-app,
html.vs-glass .gradio-container,
html.vs-glass .gradio-container > .main,
html.vs-glass .gradio-container .main.fillable,
html.vs-glass .app { background: transparent !important; }

/* ---------- 玻璃面板 ----------
   层 1+3  背景模糊 + 染色      → ::before
   层 2    形状（圆角）         → border-radius（决定高光与色散走向）
   层 4    折射                 → ::after 的 backdrop-filter: var(--vs-rf)（SDF 位移图）
   层 5    高光（随鼠标移动）   → ::after 的 radial-gradient at var(--vs-lx) var(--vs-ly)
   层 6    色散                 → ::before 内阴影叠红/蓝各 1px
   层 7    接触阴影             → box-shadow 双层（贴地硬阴影 + 大范围软阴影）
   注意：--vs-lx / --vs-ly 必须在这里给普通声明兜底。只靠 @property 注册不行，
   Gradio 的 CSS 管线会把 @property 规则丢掉，变量一空整条 background 就作废。 */
html.vs-glass .gradio-container .block {
  position: relative;
  --vs-lx: 26%;
  --vs-ly: -10%;
  background: transparent !important;
  border: none !important;
  border-radius: 24px !important;
  box-shadow: var(--vs-panel-shadow) !important;
  transition: box-shadow .34s cubic-bezier(.32,.72,0,1);
}
/* JS 会给「内部本来就不需要滚动」的块加 .vs-clip，让内容不再戳出圆角。
   绝不能无条件 overflow:hidden：会把需要滚动的面板（日志、表格）锁死。 */
html.vs-glass .gradio-container .block.vs-clip { overflow: hidden !important; }

html.vs-glass .gradio-container .block::before {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  border-radius: inherit;
  pointer-events: none;
  background: var(--vs-panel-gloss),
    radial-gradient(58% 48% at 0% 0%, rgba(255,255,255,.18) 0%, rgba(255,255,255,0) 66%),
    radial-gradient(52% 44% at 100% 100%, rgba(255,255,255,.10) 0%, rgba(255,255,255,0) 68%),
    var(--vs-panel-veil);
  -webkit-backdrop-filter: blur(var(--vs-glass-blur)) saturate(var(--vs-glass-sat)) brightness(var(--vs-glass-bright));
  backdrop-filter: blur(var(--vs-glass-blur)) saturate(var(--vs-glass-sat)) brightness(var(--vs-glass-bright));
  box-shadow: var(--vs-panel-inner);
}
html.vs-glass .gradio-container .block::after {
  content: "";
  position: absolute;
  inset: 0;
  z-index: -1;
  box-sizing: border-box;
  padding: 2.2px;
  border-radius: inherit;
  pointer-events: none;
  opacity: .95;
  background:
    radial-gradient(48% 38% at var(--vs-lx, 26%) var(--vs-ly, -10%),
        rgba(255,255,255,1) 0%, rgba(255,255,255,.50) 30%, rgba(255,255,255,0) 78%),
    radial-gradient(44% 38% at 94% 112%,
        rgba(255,255,255,.70) 0%, rgba(255,255,255,.16) 40%, rgba(255,255,255,0) 80%),
    linear-gradient(198deg, rgba(152,206,255,.40) 0%, rgba(152,206,255,0) 34%),
    linear-gradient(22deg, rgba(255,152,192,.30) 0%, rgba(255,152,192,0) 38%),
    linear-gradient(158deg,
        rgba(255,255,255,.66) 0%, rgba(255,255,255,.22) 14%,
        rgba(255,255,255,.05) 44%, rgba(255,255,255,.03) 68%, rgba(255,255,255,.52) 100%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  mask-composite: exclude;
}
html.vs-glass.dark .gradio-container .block::after {
  background:
    radial-gradient(46% 36% at var(--vs-lx, 26%) var(--vs-ly, -10%),
        rgba(255,255,255,.86) 0%, rgba(255,255,255,.34) 34%, rgba(255,255,255,0) 76%),
    radial-gradient(42% 36% at 94% 112%,
        rgba(255,255,255,.46) 0%, rgba(255,255,255,.10) 40%, rgba(255,255,255,0) 78%),
    linear-gradient(198deg, rgba(152,206,255,.30) 0%, rgba(152,206,255,0) 34%),
    linear-gradient(22deg, rgba(255,152,192,.22) 0%, rgba(255,152,192,0) 38%),
    linear-gradient(158deg,
        rgba(255,255,255,.40) 0%, rgba(255,255,255,.14) 14%,
        rgba(255,255,255,.04) 44%, rgba(255,255,255,.02) 68%, rgba(255,255,255,.30) 100%);
}
html.vs-glass .gradio-container .block:hover {
  box-shadow: 0 2px 4px rgba(12,18,38,.08), 0 14px 28px -16px rgba(12,18,38,.24), 0 36px 70px -26px rgba(12,18,38,.46) !important;
}
html.vs-glass.dark .gradio-container .block:hover {
  box-shadow: 0 2px 4px rgba(0,0,0,.5), 0 16px 32px -18px rgba(0,0,0,.6), 0 40px 76px -28px rgba(0,0,0,.95) !important;
}
/* 嵌套面板不再重复磨砂，避免叠加发灰 */
html.vs-glass .gradio-container .block .block {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}
html.vs-glass .gradio-container .block .block::before,
html.vs-glass .gradio-container .block .block::after { display: none !important; }

/* ---------- 交互反馈：苹果式按压与过渡（按钮内没有 fixed 后代，用 transform 是安全的） ---------- */
html.vs-glass .gradio-container button,
html.vs-glass .gradio-container .label-wrap,
html.vs-glass .gradio-container .vs-bg-sw,
html.vs-glass .gradio-container input[type=range] {
  transition: background-color .18s ease, border-color .18s ease,
              box-shadow .24s cubic-bezier(.32,.72,0,1), transform .16s cubic-bezier(.32,.72,0,1);
}
html.vs-glass .gradio-container button:active { transform: scale(.976); }
html.vs-glass .gradio-container .vs-bg-sw:active { transform: scale(.955); }
html.vs-glass .gradio-container .vs-bg-sw:hover { transform: translateY(-1px); }
html.vs-glass .gradio-container .block:hover::after { opacity: 1; }

/* ---------- 空面板不画玻璃：Gradio 会生成一些完全没内容的占位 .block，
   画上玻璃就变成一排无意义的灰色「大物块」。这里由 JS 打标记后直接不画。 ---------- */
html.vs-glass .gradio-container .block.vs-empty {
  box-shadow: none !important;
  background: transparent !important;
}
html.vs-glass .gradio-container .block.vs-empty::before,
html.vs-glass .gradio-container .block.vs-empty::after { display: none !important; }

/* ---------- 苹果式 Q 弹（弹簧缓动）动效 ----------
   统一用 cubic-bezier(.34,1.56,.64,1)：会轻微过冲再回弹，就是苹果那种「Q弹」。 */
@keyframes vs-boing {
  0%   { opacity: 0; transform: scale(.88); }
  55%  { opacity: 1; transform: scale(1.035); }
  100% { opacity: 1; transform: scale(1); }
}
@keyframes vs-boing-up {
  0%   { opacity: 0; transform: translateY(10px) scale(.965); }
  60%  { opacity: 1; transform: translateY(-3px) scale(1.01); }
  100% { opacity: 1; transform: translateY(0) scale(1); }
}
html.vs-glass .gradio-container button,
html.vs-glass .gradio-container .label-wrap,
html.vs-glass .gradio-container .vs-bg-sw,
html.vs-glass .gradio-container input[type=range] {
  transition: background-color .18s ease, border-color .18s ease,
              box-shadow .3s cubic-bezier(.34,1.56,.64,1),
              transform .34s cubic-bezier(.34,1.56,.64,1) !important;
}
html.vs-glass .gradio-container button:active { transform: scale(.945); }
html.vs-glass .gradio-container .label-wrap:active { transform: scale(.985); }
html.vs-glass .gradio-container .vs-bg-sw:hover { transform: translateY(-2px); }
html.vs-glass .gradio-container .vs-bg-sw:active { transform: scale(.92); }
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container button.selected {
  animation: vs-boing .34s cubic-bezier(.34,1.56,.64,1) both;
}
html.vs-glass .gradio-container .label-wrap,
html.vs-glass .gradio-container button.vs-cta { animation: vs-boing .42s cubic-bezier(.34,1.56,.64,1) both; }
html.vs-glass .gradio-container .block:hover::after { opacity: 1; }
html.vs-glass .gradio-container .block::after { transition: opacity .3s cubic-bezier(.34,1.56,.64,1); }
.vs-bg-panel[data-open="1"] { animation: vs-boing-up .36s cubic-bezier(.34,1.56,.64,1); }
.vs-sw::after { transition: transform .34s cubic-bezier(.34,1.56,.64,1); }
html.vs-glass .gradio-container .options,
html.vs-glass .gradio-container ul.options,
html.vs-glass .gradio-container [role="listbox"] {
  animation: vs-boing .32s cubic-bezier(.34,1.56,.64,1) both !important;
}

/* ---------- 空占位面板彻底不画（含子元素背景），避免出现无意义的灰色大物块 ---------- */
html.vs-glass .gradio-container .block.vs-empty,
html.vs-glass .gradio-container .block.vs-empty > *,
html.vs-glass .gradio-container .block.vs-empty > * > *,
html.vs-glass .gradio-container .block.vs-empty > * > * > * {
  background: transparent !important;
  border-color: transparent !important;
  box-shadow: none !important;
}

/* ---------- 文字完整性：正文 / 标签绝不允许被裁切 ---------- */
html.vs-glass .gradio-container .prose,
html.vs-glass .gradio-container .prose > p,
html.vs-glass .gradio-container .prose > ul,
html.vs-glass .gradio-container .prose > ol,
html.vs-glass .gradio-container .prose > blockquote,
html.vs-glass .gradio-container .prose > h1,
html.vs-glass .gradio-container .prose > h2,
html.vs-glass .gradio-container .prose > h3,
html.vs-glass .gradio-container .prose > h4,
html.vs-glass .gradio-container .prose > table,
html.vs-glass .gradio-container .md,
html.vs-glass .gradio-container .md > p {
  overflow: visible !important;
  max-height: none !important;
  height: auto !important;
  -webkit-line-clamp: unset !important;
  white-space: normal !important;
  text-overflow: clip !important;
  word-break: break-word;
  overflow-wrap: anywhere;
}
html.vs-glass .gradio-container .prose pre,
html.vs-glass .gradio-container .prose pre code {
  overflow: auto !important;
  white-space: pre !important;
  max-height: none !important;
}
html.vs-glass .gradio-container .label-wrap,
html.vs-glass .gradio-container .label-wrap > span,
html.vs-glass .gradio-container .label-wrap > .label-text,
html.vs-glass .gradio-container .block > .label-wrap {
  overflow: visible !important;
  max-height: none !important;
  white-space: normal !important;
  text-overflow: clip !important;
  overflow-wrap: anywhere;
}
html.vs-glass .gradio-container .label-wrap > span { flex: 1 1 auto; min-width: 0; }
html.vs-glass .gradio-container .label-wrap { padding-right: 26px !important; box-sizing: border-box; }
/* Gradio 的 radio / checkbox 组：允许换行，圆点不再被挤到面板外。
   坑（已踩）：绝不能笼统地给 .wrap 加 flex-wrap —— .wrap 也被 Gradio 当作列容器用，
   在超宽屏（约 2000px 以上）会把 .column 撑到 5000px+，整页布局直接塌掉。
   所以这里只对真正的选项卡组（role=radiogroup / .checkbox-group）生效。 */
html.vs-glass .gradio-container [role="radiogroup"],
html.vs-glass .gradio-container .checkbox-group > .wrap,
html.vs-glass .gradio-container .checkbox-group .wrap {
  flex-wrap: wrap !important;
  min-width: 0 !important;
}
html.vs-glass .gradio-container [role="radiogroup"] > label,
html.vs-glass .gradio-container .checkbox-group .wrap > label {
  min-width: 0 !important;
  max-width: 100% !important;
}

/* ---------- 折叠面板 / 表单 / 输入 ---------- */
html.vs-glass .gradio-container .gr-accordion,
html.vs-glass .gradio-container .form { background: transparent !important; }
html.vs-glass .gradio-container .label-wrap {
  background: var(--vs-chip) !important;
  border-radius: 13px !important;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.20);
  transition: background-color .2s ease, box-shadow .24s cubic-bezier(.32,.72,0,1);
}
html.vs-glass.dark .gradio-container .label-wrap { box-shadow: inset 0 0 0 1px rgba(255,255,255,.10); }
html.vs-glass .gradio-container .label-wrap:hover { background: rgba(255,255,255,.20) !important; }
html.vs-glass.dark .gradio-container .label-wrap:hover { background: rgba(255,255,255,.09) !important; }

html.vs-glass .gradio-container input:not([type=checkbox]):not([type=radio]):not([type=color]):not([type=range]),
html.vs-glass .gradio-container textarea {
  background: var(--vs-field) !important;
  border: 1px solid var(--vs-field-line) !important;
  border-radius: 12px !important;
  transition: background-color .2s ease, border-color .2s ease;
}
html.vs-glass .gradio-container input:not([type=checkbox]):not([type=radio]):not([type=color]):not([type=range]):focus,
html.vs-glass .gradio-container textarea:focus {
  background: var(--vs-field-focus) !important;
  border-color: var(--vs-field-line-hi) !important;
}

/* ---------- 下拉弹层：高透玻璃 + 苹果式弹出动效 ----------
   .options 是 position:fixed，任何祖先的 overflow 都裁不到它；
   反之祖先一旦带 backdrop-filter / filter / transform 就会被当成包含块而错位。 */
@keyframes vs-pop-in {
  from { opacity: 0; transform: scale(.94) translateY(-6px); }
  to   { opacity: 1; transform: scale(1) translateY(0); }
}
html.vs-glass .gradio-container .options,
html.vs-glass .gradio-container ul.options,
html.vs-glass .gradio-container [role="listbox"] {
  background: rgba(255,255,255,.52) !important;
  color: #0b1220 !important;
  border: 1px solid rgba(255,255,255,.72) !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.9), inset 0 0 0 1px rgba(255,255,255,.35),
              0 1px 1px rgba(15,23,42,.05), 0 12px 28px -10px rgba(15,23,42,.22),
              0 30px 64px -24px rgba(15,23,42,.45) !important;
  -webkit-backdrop-filter: blur(28px) saturate(190%) brightness(1.06) !important;
  backdrop-filter: blur(28px) saturate(190%) brightness(1.06) !important;
  padding: 6px !important;
  transform-origin: top center;
  animation: vs-pop-in .20s cubic-bezier(.32,.72,0,1) both;
  will-change: transform, opacity;
}
html.vs-glass.dark .gradio-container .options,
html.vs-glass.dark .gradio-container ul.options,
html.vs-glass.dark .gradio-container [role="listbox"] {
  background: rgba(18,21,30,.60) !important;
  color: #eef2f9 !important;
  border-color: rgba(255,255,255,.14) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.14), inset 0 0 0 1px rgba(255,255,255,.06),
              0 1px 1px rgba(0,0,0,.4), 0 30px 64px -24px rgba(0,0,0,.85) !important;
}
html.vs-glass .gradio-container .options .item,
html.vs-glass .gradio-container ul.options > li {
  border-radius: 12px !important;
  transition: background-color .16s ease;
}
html.vs-glass .gradio-container .options .item,
html.vs-glass .gradio-container .options .item *,
html.vs-glass .gradio-container ul.options > li,
html.vs-glass .gradio-container ul.options > li * {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  line-height: 1.4 !important;
}
html.vs-glass .gradio-container .options .item:hover,
html.vs-glass .gradio-container ul.options > li:hover { background: rgba(255,255,255,.55) !important; }
html.vs-glass.dark .gradio-container .options .item:hover,
html.vs-glass.dark .gradio-container ul.options > li:hover { background: rgba(255,255,255,.10) !important; }
html.vs-glass .gradio-container .options .item.selected,
html.vs-glass .gradio-container ul.options > li.selected { background: rgba(99,102,241,.20) !important; }

/* ---------- 标签页：玻璃胶囊 ---------- */
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container {
  background: rgba(255,255,255,.30) !important;
  border: 1px solid var(--vs-panel-line) !important;
  border-radius: 16px !important;
  padding: 4px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.75), 0 6px 20px -12px rgba(15,23,42,.24) !important;
}
html.vs-glass.dark .gradio-container .tabs > .tab-wrapper > .tab-container {
  background: rgba(20,24,32,.36) !important;
  border-color: rgba(255,255,255,.10) !important;
}
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container button.selected {
  background: rgba(255,255,255,.80) !important;
  border-radius: 12px !important;
  box-shadow: 0 2px 8px rgba(15,23,42,.12), inset 0 1px 0 rgba(255,255,255,.9) !important;
}
html.vs-glass.dark .gradio-container .tabs > .tab-wrapper > .tab-container button.selected {
  background: rgba(255,255,255,.14) !important;
}
/* ---------- 主按钮 ---------- */
html.vs-glass button.vs-cta {
  background: linear-gradient(180deg, #6d70f6 0%, #5a5df0 100%) !important;
  border: 1px solid rgba(255,255,255,.34) !important;
  border-radius: 14px !important;
  box-shadow: 0 8px 22px -10px rgba(79,70,229,.55), inset 0 1px 0 rgba(255,255,255,.42) !important;
  transition: filter .2s ease, box-shadow .2s ease;
}
html.vs-glass button.vs-cta:hover { filter: brightness(1.07); }
/* ---------- 进度条 ---------- */
html.vs-glass .vs-prog {
  background: var(--vs-panel-tint) !important;
  border-radius: 18px !important;
}
html.vs-glass .vs-prog .vs-fill { position: relative; overflow: hidden; }
html.vs-glass .vs-prog .vs-fill::after {
  content: "";
  position: absolute; inset: 0;
  background: linear-gradient(180deg, rgba(255,255,255,.34), rgba(255,255,255,0) 62%);
  pointer-events: none;
}
/* ---------- 顶部工具条玻璃化 ---------- */
html.vs-glass .vs-toolbar {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
html.vs-glass.dark .vs-toolbar { background: transparent !important; }
.vs-icon-btn[data-on="1"] {
  color: #4338ca !important;
  border-color: rgba(99,102,241,.55) !important;
  background: rgba(99,102,241,.12) !important;
}
html.dark .vs-icon-btn[data-on="1"] {
  color: #c7d2fe !important;
  background: rgba(129,140,248,.18) !important;
}

/* =========================================================================
   外观面板（液态玻璃开关 + 背景）
   ========================================================================= */
.vs-bg-panel {
  position: fixed;
  top: 64px;
  right: 16px;
  z-index: 2147483000;
  width: min(340px, calc(100vw - 32px));
  box-sizing: border-box;
  padding: 15px 15px 13px;
  border-radius: 22px;
  overflow: hidden;
  background: rgba(255,255,255,.40);
  -webkit-backdrop-filter: blur(26px) saturate(190%) brightness(1.06);
  backdrop-filter: blur(26px) saturate(190%) brightness(1.06);
  border: 1px solid rgba(255,255,255,.58);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.9), inset 0 0 0 1px rgba(255,255,255,.30),
              0 30px 68px -26px rgba(15,23,42,.48), 0 2px 8px rgba(15,23,42,.08);
  color: #0b1220;
  font-size: 13px;
  line-height: 1.5;
  display: none;
}
html.dark .vs-bg-panel {
  background: rgba(16,19,28,.50);
  border-color: rgba(255,255,255,.14);
  color: #e8ecf4;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.16), inset 0 0 0 1px rgba(255,255,255,.06),
              0 32px 74px -26px rgba(0,0,0,.88), 0 2px 8px rgba(0,0,0,.5);
}
.vs-bg-panel[data-open="1"] { display: block; animation: vs-pop .2s cubic-bezier(.22,1,.36,1); }
.vs-bg-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.vs-bg-head b { font-size: 13.5px; font-weight: 650; letter-spacing: .2px; }
.vs-bg-head .vs-bg-x {
  margin-left: auto; cursor: pointer; border: 0; background: transparent;
  color: inherit; opacity: .5; font-size: 15px; line-height: 1; padding: 3px 7px; border-radius: 9px;
}
.vs-bg-head .vs-bg-x:hover { opacity: 1; background: rgba(127,127,127,.16); }
.vs-bg-sec { border-top: 1px solid rgba(127,127,127,.16); padding-top: 11px; margin-top: 11px; }
.vs-bg-sec:first-of-type { border-top: 0; padding-top: 0; margin-top: 0; }
.vs-bg-cap { font-size: 10.5px; letter-spacing: .8px; text-transform: uppercase; opacity: .45; margin-bottom: 8px; }
.vs-bg-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 7px; margin-bottom: 9px; }
.vs-bg-sw {
  height: 38px; border-radius: 11px; cursor: pointer; padding: 0;
  border: 1px solid rgba(255,255,255,.6);
  background-size: cover; background-position: center;
  transition: transform .16s ease, box-shadow .16s ease;
}
.vs-bg-sw:hover { box-shadow: 0 4px 12px rgba(15,23,42,.14); }
.vs-bg-sw[data-active="1"] {
  border-color: rgba(99,102,241,.95);
  box-shadow: 0 0 0 2px rgba(99,102,241,.5);
}

.vs-bg-sw[data-bg="aurora"] { background-image: radial-gradient(70% 70% at 18% 20%, #818cf8, transparent 72%), radial-gradient(70% 70% at 82% 26%, #ec4899, transparent 72%), radial-gradient(80% 80% at 70% 92%, #22d3ee, transparent 72%), linear-gradient(135deg, #f8faff, #ecf8ff); }
.vs-bg-sw[data-bg="sunset"] { background-image: radial-gradient(70% 70% at 20% 22%, #fb923c, transparent 72%), radial-gradient(70% 70% at 80% 26%, #f43f5e, transparent 72%), radial-gradient(80% 80% at 68% 90%, #facc15, transparent 72%), linear-gradient(135deg, #fff8f3, #fdf4ff); }
.vs-bg-sw[data-bg="ocean"] { background-image: radial-gradient(70% 70% at 20% 20%, #38bdf8, transparent 72%), radial-gradient(70% 70% at 80% 28%, #14b8a6, transparent 72%), radial-gradient(80% 80% at 70% 92%, #3b82f6, transparent 72%), linear-gradient(135deg, #f2fbff, #eff6ff); }
.vs-bg-sw[data-bg="mint"] { background-image: radial-gradient(70% 70% at 20% 22%, #34d399, transparent 72%), radial-gradient(70% 70% at 80% 26%, #22c55e, transparent 72%), radial-gradient(80% 80% at 70% 90%, #2dd4bf, transparent 72%), linear-gradient(135deg, #f3fdf7, #f2fdfb); }
.vs-bg-sw[data-bg="peach"] { background-image: radial-gradient(70% 70% at 20% 22%, #fb7185, transparent 72%), radial-gradient(70% 70% at 80% 26%, #f9a8d4, transparent 72%), radial-gradient(80% 80% at 70% 90%, #fdba74, transparent 72%), linear-gradient(135deg, #fff6f7, #fff9f3); }
.vs-bg-sw[data-bg="night"] { background-image: radial-gradient(70% 70% at 20% 22%, #6366f1, transparent 72%), radial-gradient(70% 70% at 80% 26%, #38bdf8, transparent 72%), radial-gradient(80% 80% at 70% 90%, #8b5cf6, transparent 72%), linear-gradient(135deg, #f6f7ff, #f7f3ff); }
.vs-bg-sw[data-bg="mono"] { background-image: radial-gradient(70% 70% at 20% 22%, #94a3b8, transparent 72%), radial-gradient(70% 70% at 80% 26%, #cbd5e1, transparent 72%), radial-gradient(80% 80% at 70% 90%, #64748b, transparent 72%), linear-gradient(135deg, #f8f9fb, #f6f7f9); }
.vs-bg-sw[data-bg="paper"] { background-image: radial-gradient(80% 80% at 25% 20%, #ffffff, transparent 72%), radial-gradient(80% 80% at 78% 30%, #e2e8f0, transparent 72%), linear-gradient(135deg, #fcfcfe, #fafbfd); }
.vs-bg-sw[data-bg="custom"] { background-image: conic-gradient(from 210deg, #6366f1, #ec4899, #f59e0b, #22d3ee, #6366f1); }
.vs-bg-row { display: flex; align-items: center; gap: 8px; margin: 7px 0; min-width: 0; }
.vs-bg-row > label { flex: 0 0 52px; opacity: .7; font-size: 12px; white-space: nowrap; }
.vs-bg-row input[type=range] { flex: 1 1 auto; min-width: 0; accent-color: #6366f1; }
.vs-bg-val { flex: 0 0 46px; text-align: right; font-variant-numeric: tabular-nums; opacity: .6; font-size: 11.5px; white-space: nowrap; }
.vs-bg-color { flex: 0 0 auto !important; width: 40px !important; height: 28px; padding: 0; border: 1px solid rgba(255,255,255,.75); border-radius: 9px; background: transparent; cursor: pointer; }

.vs-bg-btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  flex: 1 1 0; min-width: 0; height: 32px; padding: 0 10px; cursor: pointer;
  border-radius: 11px; font-size: 12.5px; color: inherit;
  border: 1px solid rgba(255,255,255,.72);
  background: rgba(255,255,255,.42);
  transition: background-color .18s ease;
}
.vs-bg-btn:hover { background: rgba(255,255,255,.7); }
html.dark .vs-bg-btn { background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.16); }
html.dark .vs-bg-btn:hover { background: rgba(255,255,255,.15); }
.vs-sw {
  position: relative; flex: 0 0 auto; width: 44px; height: 26px; padding: 0;
  border: 0; border-radius: 13px; cursor: pointer;
  background: rgba(120,120,128,.30);
  transition: background-color .22s ease;
}
.vs-sw::after {
  content: ""; position: absolute; top: 2px; left: 2px; width: 22px; height: 22px;
  border-radius: 50%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.28);
  transition: transform .22s cubic-bezier(.22,1,.36,1);
}
.vs-sw[data-on="1"] { background: #34c759; }
.vs-sw[data-on="1"]::after { transform: translateX(18px); }
.vs-bg-note { opacity: .48; font-size: 11px; margin-top: 10px; line-height: 1.45; overflow-wrap: anywhere; }
/* ---------- 滚动条：原生滚动条太丑，换成细玻璃条（顺带去掉 ▲▼ 箭头方块） ---------- */
html.vs-glass .gradio-container,
html.vs-glass .gradio-container * {
  scrollbar-width: thin;
  scrollbar-color: rgba(124,136,170,.55) transparent;
}
html.vs-glass.dark .gradio-container,
html.vs-glass.dark .gradio-container * {
  scrollbar-color: rgba(200,210,238,.42) transparent;
}
html.vs-glass .gradio-container ::-webkit-scrollbar { width: 10px; height: 10px; }
html.vs-glass .gradio-container ::-webkit-scrollbar-track { background: transparent; }
html.vs-glass .gradio-container ::-webkit-scrollbar-thumb {
  background: rgba(124,136,170,.50);
  background-clip: content-box;
  border: 3px solid transparent;
  border-radius: 10px;
}
html.vs-glass .gradio-container ::-webkit-scrollbar-thumb:hover {
  background: rgba(124,136,170,.72);
  background-clip: content-box;
  border: 3px solid transparent;
}
html.vs-glass.dark .gradio-container ::-webkit-scrollbar-thumb {
  background: rgba(200,210,238,.34);
  background-clip: content-box;
  border: 3px solid transparent;
}
html.vs-glass .gradio-container ::-webkit-scrollbar-button { display: none; width: 0; height: 0; }
html.vs-glass .gradio-container ::-webkit-scrollbar-corner { background: transparent; }

/* ---------- 标签页：苹果式分段控件 ----------
   Gradio 自带的 .tab-container::after 是一条深色实心条（#374151），
   在玻璃上会变成一块突兀的方块，这里彻底接管标签栏的绘制。 */
html.vs-glass .gradio-container .tabs > .tab-wrapper {
  background: transparent !important;
  border: 0 !important;
}
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container {
  background: rgba(255,255,255,.30) !important;
  border: 1px solid var(--vs-panel-line) !important;
  border-radius: 16px !important;
  padding: 4px !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
  scrollbar-width: none;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.75), 0 6px 20px -12px rgba(15,23,42,.24) !important;
}
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container::-webkit-scrollbar { height: 0 !important; width: 0 !important; }
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container::after { display: none !important; }
html.vs-glass.dark .gradio-container .tabs > .tab-wrapper > .tab-container {
  background: rgba(20,24,32,.36) !important;
  border-color: rgba(255,255,255,.10) !important;
}
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container button {
  background: transparent !important;
  border: 0 !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  max-width: 100%;
  white-space: nowrap;
}
html.vs-glass .gradio-container .tabs > .tab-wrapper > .tab-container button.selected {
  background: rgba(255,255,255,.80) !important;
  box-shadow: 0 2px 8px rgba(15,23,42,.12), inset 0 1px 0 rgba(255,255,255,.9) !important;
}
html.vs-glass.dark .gradio-container .tabs > .tab-wrapper > .tab-container button.selected {
  background: rgba(255,255,255,.14) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.18) !important;
}

@media (prefers-reduced-motion: reduce) {
  .vs-bg-panel[data-open="1"],
  html.vs-glass .gradio-container .options,
  html.vs-glass .gradio-container ul.options,
  html.vs-glass .gradio-container [role="listbox"] { animation: none !important; }
}
"""

VS_HEAD = """
<script>
(function () {
  var KEY = 'vocal-separator-theme';
  function readStored() {
    try { var v = localStorage.getItem(KEY); if (v === 'dark' || v === 'light') return v; } catch (e) {}
    return null;
  }
  function preferred() {
    var s = readStored();
    if (s) return s;
    try { if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) return 'light'; } catch (e) {}
    return 'dark';
  }
  function apply(mode) {
    window.vsTheme = mode;
    var dark = (mode === 'dark');
    var roots = [document.documentElement];
    if (document.body) { roots.push(document.body); }
    var app = document.querySelector('gradio-app');
    if (app) { roots.push(app); }
    for (var i = 0; i < roots.length; i++) {
      try { roots[i].classList.toggle('dark', dark); } catch (e) {}
    }
    document.documentElement.setAttribute('data-theme', mode);
    try { localStorage.setItem(KEY, mode); } catch (e) {}
    try { if (window.vsBgApply) { window.vsBgApply(); } } catch (e) {}
  }
  window.vsApplyTheme = apply;
  window.vsToggleTheme = function () { apply((window.vsTheme === 'dark') ? 'light' : 'dark'); };
  apply(preferred());
  document.addEventListener('DOMContentLoaded', function () { apply(preferred()); });
  document.addEventListener('gradio-loaded', function () { apply(window.vsTheme); });
  window.addEventListener('load', function () { apply(window.vsTheme); });
})();
</script>
<script>
(function () {
  var K_ON     = 'vocal-separator-glass';
  var K_PRESET = 'vocal-separator-bg';
  var K_COLOR  = 'vocal-separator-bg-color';
  var K_BLUR   = 'vocal-separator-bg-blur';
  var K_DIM    = 'vocal-separator-bg-dim';
  var K_IMAGE  = 'vocal-separator-bg-image';
  var K_GBLUR  = 'vocal-separator-glass-blur';
  var PRESETS  = ['aurora', 'sunset', 'ocean', 'mint', 'peach', 'night', 'mono', 'paper'];
  var NAMES    = { aurora: '极光', sunset: '暮色', ocean: '海洋', mint: '薄荷',
                   peach: '蜜桃', night: '星空', mono: '素灰', paper: '素白' };
  var VARS     = ['--vs-g1', '--vs-g2', '--vs-g3', '--vs-g4', '--vs-base1', '--vs-base2', '--vs-base3'];

  function lsGet(k, d) { try { var v = localStorage.getItem(k); return (v === null || v === '') ? d : v; } catch (e) { return d; } }
  function lsSet(k, v) { try { if (v === '' || v === null) { localStorage.removeItem(k); } else { localStorage.setItem(k, v); } } catch (e) {} }

  function roots() {
    var r = [document.documentElement];
    if (document.body) { r.push(document.body); }
    var a = document.querySelector('gradio-app');
    if (a) { r.push(a); }
    return r;
  }
  function setVar(name, val) {
    var rs = roots();
    for (var i = 0; i < rs.length; i++) { try { rs[i].style.setProperty(name, val); } catch (e) {} }
  }
  function delVar(name) {
    var rs = roots();
    for (var i = 0; i < rs.length; i++) { try { rs[i].style.removeProperty(name); } catch (e) {} }
  }
  function setAttr(name, val) {
    var rs = roots();
    for (var i = 0; i < rs.length; i++) { try { rs[i].setAttribute(name, val); } catch (e) {} }
  }
  function clamp(v, lo, hi) { v = parseFloat(v); if (!isFinite(v)) { v = lo; } return v < lo ? lo : (v > hi ? hi : v); }
  function hexToHsl(hex) {
    var s = String(hex || '').replace('#', '');
    if (s.length === 3) { s = s.charAt(0) + s.charAt(0) + s.charAt(1) + s.charAt(1) + s.charAt(2) + s.charAt(2); }
    if (s.length !== 6) { return [232, 68, 60]; }
    var r = parseInt(s.substr(0, 2), 16) / 255;
    var g = parseInt(s.substr(2, 2), 16) / 255;
    var b = parseInt(s.substr(4, 2), 16) / 255;
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    var l = (mx + mn) / 2, h = 0, sat = 0, d = mx - mn;
    if (d > 0.0001) {
      sat = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
      if (mx === r) { h = (g - b) / d + (g < b ? 6 : 0); }
      else if (mx === g) { h = (b - r) / d + 2; }
      else { h = (r - g) / d + 4; }
      h = h * 60;
    }
    return [Math.round(h), Math.round(sat * 100), Math.round(l * 100)];
  }
  function hsl(h, s, l) { return 'hsl(' + h + ',' + s + '%,' + l + '%)'; }
  function hsla(h, s, l, a) { return 'hsla(' + h + ',' + s + '%,' + l + '%,' + a + ')'; }
  function isDark() { try { return document.documentElement.classList.contains('dark'); } catch (e) { return false; } }

  /* ---------------- 液态玻璃开关 ---------------- */
  window.vsGlassState = function () {
    return { on: (lsGet(K_ON, 'on') !== 'off'), blur: clamp(lsGet(K_GBLUR, '16'), 0, 26) };
  };
  window.vsApplyGlass = function (mode) {
    var on = (mode === 'on');
    window.vsGlass = mode;
    var rs = roots();
    for (var i = 0; i < rs.length; i++) { try { rs[i].classList.toggle('vs-glass', on); } catch (e) {} }
    try { document.documentElement.setAttribute('data-glass', mode); } catch (e) {}
    lsSet(K_ON, mode);
    var gs = window.vsGlassState();
    setVar('--vs-glass-blur', gs.blur + 'px');
    var sw = document.querySelector('.vs-bg-panel .vs-sw[data-role="glass"]');
    if (sw) { sw.setAttribute('data-on', on ? '1' : '0'); }
    if (window.vsBgApply) { window.vsBgApply(); }
  };
  window.vsToggleGlass = function () { window.vsApplyGlass(window.vsGlass === 'on' ? 'off' : 'on'); };

  /* ---------------- 背景状态 ---------------- */
  window.vsBgState = function () {
    return {
      preset: lsGet(K_PRESET, 'aurora'),
      color:  lsGet(K_COLOR, '#6366f1'),
      blur:   clamp(lsGet(K_BLUR, '0'), 0, 40),
      dim:    clamp(lsGet(K_DIM, '0'), 0, 70),
      image:  lsGet(K_IMAGE, '')
    };
  };
  window.vsBgApply = function () {
    var st = window.vsBgState();
    var dark = isDark();
    setAttr('data-bg', st.preset);
    setAttr('data-bgimg', st.image ? '1' : '0');
    if (st.image) { setVar('--vs-bg-custom', 'url("' + st.image + '")'); } else { setVar('--vs-bg-custom', 'none'); }
    if (st.preset === 'custom') {
      var hl = hexToHsl(st.color), h = hl[0], s0 = hl[1];
      setVar('--vs-g1', hsla(h, Math.min(64, s0 + 12), dark ? 40 : 66, dark ? 0.50 : 0.42));
      setVar('--vs-g2', hsla((h + 46) % 360, Math.min(66, s0 + 16), dark ? 36 : 70, dark ? 0.36 : 0.32));
      setVar('--vs-g3', hsla((h + 314) % 360, Math.min(66, s0 + 14), dark ? 36 : 72, dark ? 0.38 : 0.34));
      setVar('--vs-g4', hsla((h + 22) % 360, Math.min(62, s0 + 8), dark ? 32 : 74, dark ? 0.34 : 0.30));
      if (dark) {
        setVar('--vs-base1', hsl(h, 22, 5)); setVar('--vs-base2', hsl(h, 20, 8)); setVar('--vs-base3', hsl(h, 24, 6));
      } else {
        setVar('--vs-base1', hsl(h, 40, 98)); setVar('--vs-base2', hsl(h, 38, 97)); setVar('--vs-base3', hsl(h, 42, 98));
      }
    } else {
      for (var i = 0; i < VARS.length; i++) { delVar(VARS[i]); }
    }
    setVar('--vs-bg-blur', st.blur + 'px');
    setVar('--vs-bg-dim', String(st.dim / 100));
    var gs = window.vsGlassState();
    setVar('--vs-glass-blur', gs.blur + 'px');
    if (window.vsSyncGlassClass) { try { window.vsSyncGlassClass(); } catch (e) {} }
    syncPanel();
  };

  /* ---------------- 外观面板 ---------------- */
  var panel = null;

  function swatchesHtml() {
    var out = '';
    for (var i = 0; i < PRESETS.length; i++) {
      var p = PRESETS[i];
      out += '<button type="button" class="vs-bg-sw" data-bg="' + p + '" title="' + NAMES[p] + '"></button>';
    }
    out += '<button type="button" class="vs-bg-sw" data-bg="custom" title="用自定义主色生成"></button>';
    return out;
  }

  function buildPanel() {
    var el = document.createElement('div');
    el.className = 'vs-bg-panel';
    el.setAttribute('data-open', '0');
    el.innerHTML = ''
      + '<div class="vs-bg-head"><b>外观</b><button class="vs-bg-x" type="button" title="关闭">&#10005;</button></div>'
      + '<div class="vs-bg-sec">'
      +   '<div class="vs-bg-cap">液态玻璃</div>'
      +   '<div class="vs-bg-row"><label>开关</label>'
      +     '<button type="button" class="vs-sw" data-role="glass" data-on="1" title="开 / 关 液态玻璃面板"></button>'
      +     '<span class="vs-bg-val" data-role="glassval"></span></div>'
      +   '<div class="vs-bg-row"><label>磨砂</label>'
      +     '<input type="range" min="0" max="60" step="1" data-role="gblur">'
      +     '<span class="vs-bg-val" data-role="gblurval">30px</span></div>'
      + '</div>'
      + '<div class="vs-bg-sec">'
      +   '<div class="vs-bg-cap">背景</div>'
      +   '<div class="vs-bg-grid">' + swatchesHtml() + '</div>'
      +   '<div class="vs-bg-row"><label>主色</label>'
      +     '<input class="vs-bg-color" type="color" value="#6366f1">'
      +     '<span class="vs-bg-val" data-role="colorval">#6366f1</span></div>'
      +   '<div class="vs-bg-row"><label>图片</label>'
      +     '<label class="vs-bg-btn" for="vsBgFile">选择图片</label>'
      +     '<button type="button" class="vs-bg-btn" data-role="clearimg">清除</button></div>'
      +   '<input id="vsBgFile" type="file" accept="image/*" style="display:none">'
      +   '<div class="vs-bg-row"><label>虚化</label>'
      +     '<input type="range" min="0" max="40" step="1" data-role="blur">'
      +     '<span class="vs-bg-val" data-role="blurval">0px</span></div>'
      +   '<div class="vs-bg-row"><label>明暗</label>'
      +     '<input type="range" min="0" max="70" step="1" data-role="dim">'
      +     '<span class="vs-bg-val" data-role="dimval">0%</span></div>'
      +   '<div class="vs-bg-row"><button type="button" class="vs-bg-btn" data-role="reset">恢复默认背景</button></div>'
      + '</div>'
      + '<div class="vs-bg-note">选择任意色块或用主色生成；图片会自动压缩后存在本机浏览器。偏好只保存在本机，不影响音频处理。</div>';
    document.body.appendChild(el);

    var colorIn = el.querySelector('.vs-bg-color');
    var blurIn  = el.querySelector('[data-role="blur"]');
    var dimIn   = el.querySelector('[data-role="dim"]');
    var gblurIn = el.querySelector('[data-role="gblur"]');

    el.addEventListener('click', function (ev) {
      var t = ev.target;
      if (!t || !t.closest) { return; }
      var sw = t.closest('.vs-bg-sw');
      if (sw) { lsSet(K_PRESET, sw.getAttribute('data-bg')); window.vsBgApply(); return; }
      if (t.closest('.vs-bg-x')) { window.vsBgTogglePanel(false); return; }
      if (t.closest('[data-role="glass"]')) { window.vsToggleGlass(); return; }
      if (t.closest('[data-role="reset"]')) {
        lsSet(K_PRESET, ''); lsSet(K_COLOR, ''); lsSet(K_BLUR, ''); lsSet(K_DIM, ''); lsSet(K_IMAGE, '');
        window.vsBgApply(); return;
      }
      if (t.closest('[data-role="clearimg"]')) { lsSet(K_IMAGE, ''); window.vsBgApply(); return; }
    });
    colorIn.addEventListener('input', function () {
      lsSet(K_COLOR, colorIn.value);
      if (lsGet(K_PRESET, 'aurora') !== 'custom') { lsSet(K_PRESET, 'custom'); }
      window.vsBgApply();
    });
    blurIn.addEventListener('input', function () { lsSet(K_BLUR, String(clamp(blurIn.value, 0, 40))); window.vsBgApply(); });
    dimIn.addEventListener('input', function () { lsSet(K_DIM, String(clamp(dimIn.value, 0, 70))); window.vsBgApply(); });
    gblurIn.addEventListener('input', function () {
      lsSet(K_GBLUR, String(clamp(gblurIn.value, 0, 60)));
      setVar('--vs-glass-blur', clamp(gblurIn.value, 0, 60) + 'px');
      syncPanel();
    });
    el.querySelector('#vsBgFile').addEventListener('change', function (ev) {
      var f = ev.target.files && ev.target.files[0];
      if (f) { loadImage(f); }
      try { ev.target.value = ''; } catch (e) {}
    });
    return el;
  }

  function syncPanel() {
    if (!panel) { return; }
    var st = window.vsBgState();
    var gs = window.vsGlassState();
    var sws = panel.querySelectorAll('.vs-bg-sw');
    for (var i = 0; i < sws.length; i++) {
      sws[i].setAttribute('data-active', sws[i].getAttribute('data-bg') === st.preset ? '1' : '0');
    }
    var c = panel.querySelector('.vs-bg-color');
    if (c && st.color) { try { c.value = st.color; } catch (e) {} }
    var node;
    node = panel.querySelector('[data-role="colorval"]'); if (node) { node.textContent = st.color || '-'; }
    node = panel.querySelector('[data-role="blur"]');  if (node) { node.value = String(st.blur); }
    node = panel.querySelector('[data-role="blurval"]'); if (node) { node.textContent = st.blur + 'px'; }
    node = panel.querySelector('[data-role="dim"]');   if (node) { node.value = String(st.dim); }
    node = panel.querySelector('[data-role="dimval"]'); if (node) { node.textContent = st.dim + '%'; }
    node = panel.querySelector('[data-role="gblur"]'); if (node) { node.value = String(gs.blur); }
    node = panel.querySelector('[data-role="gblurval"]'); if (node) { node.textContent = gs.blur + 'px'; }
    node = panel.querySelector('[data-role="glassval"]'); if (node) { node.textContent = gs.on ? '已开启' : '已关闭'; }
    node = panel.querySelector('.vs-sw[data-role="glass"]'); if (node) { node.setAttribute('data-on', gs.on ? '1' : '0'); }
  }

  function loadImage(file) {
    var fr = new FileReader();
    fr.onload = function () {
      var img = new Image();
      img.onload = function () {
        var max = 2400;
        var sc = Math.min(1, max / Math.max(img.width || 1, img.height || 1));
        var cv = document.createElement('canvas');
        cv.width = Math.max(1, Math.round((img.width || max) * sc));
        cv.height = Math.max(1, Math.round((img.height || max) * sc));
        try { cv.getContext('2d').drawImage(img, 0, 0, cv.width, cv.height); } catch (e) { return; }
        var q = 0.86, out = cv.toDataURL('image/jpeg', q);
        while (out.length > 3400000 && q > 0.45) { q = q - 0.14; out = cv.toDataURL('image/jpeg', q); }
        lsSet(K_IMAGE, out);
        window.vsBgApply();
      };
      img.src = fr.result;
    };
    fr.readAsDataURL(file);
  }

  window.vsEnsureBgPanel = function () {
    if (!document.body) { return; }
    if (!panel || !document.body.contains(panel)) { panel = buildPanel(); }
  };
  window.vsBgTogglePanel = function (force) {
    window.vsEnsureBgPanel();
    if (!panel) { return; }
    var open = (typeof force === 'boolean') ? force : (panel.getAttribute('data-open') !== '1');
    panel.setAttribute('data-open', open ? '1' : '0');
    var lbs = document.querySelectorAll('.vs-look-btn');
    for (var li = 0; li < lbs.length; li++) { try { lbs[li].setAttribute('data-on', open ? '1' : '0'); } catch (e) {} }
    if (open) { syncPanel(); }
  };
  window.vsToggleBgPanel = function () { window.vsBgTogglePanel(); };
  window.vsToggleLook = function () { window.vsBgTogglePanel(); };

  /* 关键：把「液态玻璃开关」的真实状态落到 DOM 上。
     之前只有点击开关时才会调用 vsApplyGlass，页面刷新后 html 上没有 vs-glass 类，
     于是面板里显示「已开启」但整页其实没有任何玻璃效果 —— 就是用户看到的「打开了却没生效」。 */
  window.vsSyncGlassClass = function () {
    var on = window.vsGlassState().on;
    var rs = roots();
    for (var i = 0; i < rs.length; i++) {
      try { rs[i].classList.toggle('vs-glass', on); } catch (e) {}
    }
    try { document.documentElement.setAttribute('data-glass', on ? 'on' : 'off'); } catch (e) {}
    window.vsGlass = on ? 'on' : 'off';
  };

  /* ---------------- 液态玻璃折射：SDF 法线位移图 + feDisplacementMap ----------------
     参考图的原理：玻璃中心法线朝向屏幕外（几乎不偏移），越靠边法线越倾斜（偏移越大），
     于是「中心基本不变、边缘被拉伸压缩」。这里用 canvas 逐像素算圆角矩形的 SDF，
     把梯度编码成一张 RGB 法线图，交给 SVG 的 feDisplacementMap 去偏移背景采样坐标。
     结果写进 CSS 变量 --vs-rf，由 .block::after 的 backdrop-filter 引用。
     Chromium 实测可用；变量缺失时退化成 none，纯 CSS 照样成立。 */
  var LENS_MAX = 0;           // 折射已停用：backdrop-filter 里混 SVG 位移会在细节背景上露出方角接缝
  var LENS_MIN_AREA = 42000;  // 太小的块不做折射
  var lensCache = Object.create(null);
  var lensCacheKeys = [];
  var lensSvg = null;
  var lensTimer = null;
  var lensRaf = 0;
  var lensApplied = (window.WeakMap ? new WeakMap() : null);

  function lensRoot() {
    if (lensSvg && document.body && document.body.contains(lensSvg)) { return lensSvg; }
    var NS = 'http://www.w3.org/2000/svg';
    lensSvg = document.createElementNS(NS, 'svg');
    lensSvg.setAttribute('width', '0');
    lensSvg.setAttribute('height', '0');
    lensSvg.setAttribute('aria-hidden', 'true');
    lensSvg.style.cssText = 'position:fixed;left:0;top:0;width:0;height:0;overflow:hidden;pointer-events:none';
    document.body.appendChild(lensSvg);
    return lensSvg;
  }

  function lensMap(w, h) {
    var key = w + 'x' + h;
    if (lensCache[key]) { return lensCache[key]; }
    var R = Math.min(26, Math.max(8, Math.min(w, h) * 0.16));
    var D = Math.min(36, Math.max(10, Math.min(w, h) * 0.15));
    var cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    var ctx = cv.getContext('2d');
    var im = ctx.createImageData(w, h);
    var d = im.data;
    var cx = w / 2, cy = h / 2;
    var ax = Math.max(1, w / 2 - R), ay = Math.max(1, h / 2 - R);
    for (var y = 0; y < h; y++) {
      var py = y - cy;
      for (var x = 0; x < w; x++) {
        var px = x - cx;
        var qx = Math.abs(px) - ax, qy = Math.abs(py) - ay;
        var mx = qx > 0 ? qx : 0, my = qy > 0 ? qy : 0;
        var s = Math.sqrt(mx * mx + my * my) + Math.min(Math.max(qx, qy), 0) - R;
        // 只在「边界向内 D」的环带里给偏移，且边界处与带内缘处都收敛到 0（正弦窗），
        // 并且方向改为向内采样 —— 采样点永远不会越出面板，不会把面板外的背景拉进来，
        // 因此不会再出现那条硬接缝。中心区域 e=0，完全不偏移（对应参考图「中心变化小」）。
        var e = 0;
        if (s < 0) {
          var u = -s / D;
          if (u < 1) { e = Math.sin(Math.PI * u); }
        }
        var nx = px / ax, ny = py / ay;
        var len = Math.sqrt(nx * nx + ny * ny) || 1;
        var i = (y * w + x) * 4;
        d[i]     = Math.round(128 - (nx / len) * e * 118);
        d[i + 1] = Math.round(128 - (ny / len) * e * 118);
        d[i + 2] = 128;
        d[i + 3] = 255;
      }
    }
    ctx.putImageData(im, 0, 0);
    var url = cv.toDataURL();
    lensCache[key] = url;
    lensCacheKeys.push(key);
    if (lensCacheKeys.length > 60) {
      var old = lensCacheKeys.shift();
      if (old !== key) { delete lensCache[old]; }
    }
    return url;
  }

  function lensClear(el) {
    try {
      if (el.style.getPropertyValue('--vs-rf')) {
        el.style.removeProperty('--vs-rf');
        if (lensApplied) { lensApplied.delete(el); }
      }
    } catch (e) {}
  }

  function lensAttach(el, idx) {
    var w = Math.max(2, Math.round(el.offsetWidth));
    var h = Math.max(2, Math.round(el.offsetHeight));
    var NS = 'http://www.w3.org/2000/svg';
    var id = 'vsRf' + idx;
    var svg = lensRoot();
    var f = svg.querySelector('#' + id);
    if (!f) {
      f = document.createElementNS(NS, 'filter');
      f.setAttribute('id', id);
      f.setAttribute('x', '-14%'); f.setAttribute('y', '-14%');
      f.setAttribute('width', '128%'); f.setAttribute('height', '128%');
      f.setAttribute('color-interpolation-filters', 'sRGB');
      var imgEl = document.createElementNS(NS, 'feImage');
      imgEl.setAttribute('result', 'm');
      imgEl.setAttribute('preserveAspectRatio', 'none');
      var disEl = document.createElementNS(NS, 'feDisplacementMap');
      disEl.setAttribute('in', 'SourceGraphic');
      disEl.setAttribute('in2', 'm');
      disEl.setAttribute('xChannelSelector', 'R');
      disEl.setAttribute('yChannelSelector', 'G');
      f.appendChild(imgEl);
      f.appendChild(disEl);
      svg.appendChild(f);
    }
    var img = f.firstChild;
    var dis = f.lastChild;
    var key = w + 'x' + h;
    if (img.getAttribute('data-k') !== key) {
      img.setAttribute('href', lensMap(w, h));
      img.setAttribute('data-k', key);
    }
    var scale = Math.max(6, Math.min(30, Math.round(Math.min(w, h) * 0.09)));
    if (dis.getAttribute('scale') !== String(scale)) { dis.setAttribute('scale', String(scale)); }
    var prev = lensApplied ? lensApplied.get(el) : null;
    if (prev !== id) {
      el.style.setProperty('--vs-rf', 'url(#' + id + ')');
      if (lensApplied) { lensApplied.set(el, id); }
    }
  }

  window.vsBuildLenses = function () {
    if (!document.body) { return; }
    var blocks = document.querySelectorAll('.gradio-container .block');
    var list = [];
    for (var i = 0; i < blocks.length; i++) {
      var el = blocks[i];
      var w = el.offsetWidth, h = el.offsetHeight;
      if (w < 96 || h < 64 || w * h < LENS_MIN_AREA) { lensClear(el); continue; }
      list.push({ el: el, a: w * h });
    }
    list.sort(function (x, y) { return y.a - x.a; });
    for (var j = 0; j < list.length; j++) {
      if (j < LENS_MAX) { lensAttach(list[j].el, j); } else { lensClear(list[j].el); }
    }
  };

  /* 只给「本来就不需要滚动」的块加 overflow:hidden，堵住戳出圆角的内容。
     需要滚动的面板（日志 / 表格）保持原样，功能优先于外观。 */
  window.vsApplyClip = function () {
    var blocks = document.querySelectorAll('.gradio-container .block');
    for (var i = 0; i < blocks.length; i++) {
      var el = blocks[i];
      var ok = (el.scrollHeight <= el.clientHeight + 2) && (el.scrollWidth <= el.clientWidth + 2);
      if (ok) { el.classList.add('vs-clip'); } else { el.classList.remove('vs-clip'); }
    }
  };

  /* 没有可见内容的占位面板：不画玻璃，避免出现一排无意义的灰色大物块。 */
  window.vsMarkEmpty = function () {
    var blocks = document.querySelectorAll('.gradio-container .block');
    for (var i = 0; i < blocks.length; i++) {
      var el = blocks[i];
      var empty = false;
      try {
        var txt = '';
        var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
        var node;
        while ((node = walker.nextNode())) {
          var pe = node.parentElement;
          if (!pe) { continue; }
          var pcs = getComputedStyle(pe);
          if (pcs.display === 'none' || pcs.visibility === 'hidden' || Number(pcs.opacity) === 0) { continue; }
          txt += node.nodeValue;
        }
        txt = txt.replace(/\s+/g, '');
        var hasMedia = !!el.querySelector('canvas, audio, video, img, input, textarea, select, button, .vs-prog, table, [role="listbox"], svg');
        empty = (txt.length === 0) && !hasMedia;
      } catch (e) {}
      if (empty) { el.classList.add('vs-empty'); } else { el.classList.remove('vs-empty'); }
    }
  };

  function vsScheduleLook() {
    if (lensTimer) { return; }
    lensTimer = setTimeout(function () {
      lensTimer = null;
      try { window.vsMarkEmpty(); window.vsApplyClip(); window.vsBuildLenses(); } catch (e) {}
    }, 240);
  }

  /* 高光跟着指针走：苹果的高光是「实时光源 + 法线」实时算出来的，
     这里用指针位置驱动 --vs-lx / --vs-ly，立刻反映到 ::after 的镜面高光上。 */
  /* 光源位置做插值缓动：指针一动，高光就平滑地滑过去，而不是瞬移。
     这就是参考图里「用光源和法线实时计算高光」的网页版 —— 光源是活的。 */
  var lookEl = null;
  var lookTX = 26, lookTY = -10;
  var lookCX = 26, lookCY = -10;
  var lookLoop = 0;
  var lookPending = 0;

  function lookStep() {
    lookLoop = 0;
    if (!lookEl) { return; }
    lookCX += (lookTX - lookCX) * 0.22;
    lookCY += (lookTY - lookCY) * 0.22;
    try {
      lookEl.style.setProperty('--vs-lx', (Math.round(lookCX * 10) / 10) + '%');
      lookEl.style.setProperty('--vs-ly', (Math.round(lookCY * 10) / 10) + '%');
    } catch (e) {}
    if (Math.abs(lookTX - lookCX) > 0.2 || Math.abs(lookTY - lookCY) > 0.2) {
      lookLoop = requestAnimationFrame(lookStep);
    }
  }

  document.addEventListener('pointermove', function (ev) {
    if (lookPending) { return; }
    lookPending = 1;
    requestAnimationFrame(function () {
      lookPending = 0;
      var t = ev.target;
      var el = (t && t.closest) ? t.closest('.gradio-container .block') : null;
      if (!el) { return; }
      var r = el.getBoundingClientRect();
      if (!r.width || !r.height) { return; }
      if (el !== lookEl) { lookEl = el; lookCX = 26; lookCY = -10; }
      lookTX = (ev.clientX - r.left) / r.width * 100;
      lookTY = (ev.clientY - r.top) / r.height * 100;
      if (!lookLoop) { lookLoop = requestAnimationFrame(lookStep); }
    });
  }, { passive: true });

  /* 指针离开整页时把光源收回左上角 */
  document.addEventListener('pointerleave', function () {
    lookTX = 26; lookTY = -10;
    if (!lookLoop && lookEl) { lookLoop = requestAnimationFrame(lookStep); }
  }, { passive: true });

  document.addEventListener('click', vsScheduleLook, true);
  window.addEventListener('resize', vsScheduleLook);
  if (window.ResizeObserver) {
    try { new ResizeObserver(vsScheduleLook).observe(document.documentElement); } catch (e) {}
  }

  function vsInit() {
    window.vsSyncGlassClass();
    window.vsEnsureBgPanel();
    window.vsBgApply();
    try { window.vsMarkEmpty(); } catch (e) {}
    vsScheduleLook();
  }
  document.addEventListener('DOMContentLoaded', vsInit);
  document.addEventListener('gradio-loaded', vsInit);
  window.addEventListener('load', vsInit);
  vsInit();
  // Gradio 重渲染可能整段换掉 gradio-app；定时把玻璃类与背景重新对齐（幂等、开销极小）
  setInterval(function () { window.vsSyncGlassClass(); window.vsEnsureBgPanel(); vsScheduleLook(); }, 1500);
})();
</script>
"""

# ---------------------------------------------------------------------------
# 弹性交互动画引擎（注入到页面 <script>）
#   1) 进度条宽度与百分比数字：按 easeOutBack 弹性缓动逐帧推进，跨 Gradio 重渲染保持连续
#   2) 任务完成：状态标记 / 填充条一次弹性放大（只播一次，不循环）
#   3) 标签页切换、折叠面板展开、任务列表新增行：重放一次弹性入场
# 仅使用运动曲线（缓动 + 回弹），不含流光 / 扫光 / 呼吸 / 声波等装饰特效。
# ---------------------------------------------------------------------------
VS_JS = """
<script>
(function () {
  if (window.__vsEngine) { return; }
  window.__vsEngine = true;
  try {
    /* ============ 0. 工具 ============ */
    var RM = false;
    try { RM = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches); } catch (e) {}
    function nv(v) { var n = parseFloat(v); return isFinite(n) ? n : 0; }
    function easeOutCubic(k) { return 1 - Math.pow(1 - k, 3); }
    function easeOutQuint(k) { return 1 - Math.pow(1 - k, 5); }
    function easeOutBack(k) { var c1 = 1.70158, c3 = c1 + 1, x = k - 1; return 1 + c3 * x * x * x + c1 * x * x; }
    function clamp01(k) { return k < 0 ? 0 : (k > 1 ? 1 : k); }

    /* ============ 1. 值跟踪（弹性 / 可打断 / 跨重渲染连续） ============ */
    var VAL = {};      // key -> {v, from, to, t0, dur, kind}
    var STATE = {};    // key -> 上一帧的状态，用于一次性强调
    var NODES = [];    // 正在驱动的 DOM 节点
    var BUSY = false;

    var SEL = '.vs-prog[data-vskey] .vs-fill, .vs-prog[data-vskey] .vs-ghost,'
            + ' .vs-prog[data-vskey] .vs-num, .vs-float[data-vskey] .vs-mini-fill,'
            + ' .vs-float[data-vskey] .vs-num';

    function kindOf(el) {
      if (el.classList.contains('vs-num')) { return 'num'; }
      if (el.classList.contains('vs-ghost')) { return 'ghost'; }
      if (el.classList.contains('vs-mini-fill')) { return 'mini'; }
      return 'fill';
    }
    function durOf(kind) {
      if (RM) { return 0; }
      if (kind === 'num') { return 620; }
      if (kind === 'ghost') { return 980; }
      if (kind === 'mini') { return 720; }
      return 700;
    }
    function easeOf(kind) {
      if (kind === 'num') { return easeOutCubic; }
      if (kind === 'ghost') { return easeOutQuint; }
      return easeOutBack;
    }
    function keyOf(el, root) {
      var rk = (root && root.getAttribute('data-vskey')) || 'k';
      return rk + '::' + kindOf(el);
    }
    function put(el, st, now) {
      var k = st.dur <= 0 ? 1 : clamp01((now - st.t0) / st.dur);
      var v = st.from + (st.to - st.from) * easeOf(st.kind)(k);
      if (k >= 1) { v = st.to; }
      st.v = v;
      /* 只在数值真的变化时写 DOM：减少无谓重排，也避免自触发 MutationObserver 形成回环 */
      if (st.kind === 'num') {
        var dec = parseInt(el.getAttribute('data-dec') || '1', 10);
        var txt = v.toFixed(dec);
        if (el.textContent !== txt) { el.textContent = txt; }
      } else {
        var w = Math.max(0, v).toFixed(2) + '%';
        if (el.style.width !== w) { el.style.width = w; }
      }
      return k < 1;
    }
    function track(el, now) {
      var root = el.closest ? el.closest('.vs-prog, .vs-float') : null;
      var kind = kindOf(el);
      var key = keyOf(el, root);
      var to = nv(el.getAttribute('data-p'));
      var st = VAL[key];
      if (!st) {
        st = VAL[key] = { v: to, from: to, to: to, t0: now - 1e6, dur: 0, kind: kind };
        el._vsKey = key;
      }
      if (st.to !== to) {              // 目标变化：从“当前显示值”起步 → 天然支持被打断后平滑接续
        st.from = st.v; st.to = to; st.t0 = now; st.dur = durOf(kind);
      }
      el._vsSt = st;
      if (NODES.indexOf(el) < 0) { NODES.push(el); }
      return st;
    }

    /* 一次性强调：完成时轻弹一下（只播一次） */
    function pulse(list) {
      for (var i = 0; i < list.length; i++) {
        var n = list[i];
        if (skippable(n)) { continue; }
        n.classList.remove('vs-pop');
        void n.offsetWidth;
        playOnce(n, 'vs-pop', 800);
      }
    }
    function settleState() {
      var roots = document.querySelectorAll('.vs-prog[data-vskey]');
      for (var i = 0; i < roots.length; i++) {
        var r = roots[i];
        var key = r.getAttribute('data-vskey') || 'k';
        var state = r.getAttribute('data-state') || '';
        var prev = STATE[key];
        if (prev === undefined) { STATE[key] = state; continue; }
        if (prev === state) { continue; }
        STATE[key] = state;
        if (state === 'done') { pulse(r.querySelectorAll('.vs-fill, .vs-badge-done, .vs-live-dot')); }
      }
    }

    /* ============ 2. 分块刻度（每格点亮带错峰弹性） ============ */
    function drawSegs() {
      var boxes = document.querySelectorAll('.vs-segs');
      for (var i = 0; i < boxes.length; i++) {
        var box = boxes[i];
        var n = parseInt(box.getAttribute('data-segs') || '0', 10);
        var done = parseInt(box.getAttribute('data-done') || '0', 10);
        if (!n || n < 1 || n > 300) { continue; }
        if (box.getAttribute('data-n') !== String(n)) {
          var html = '';
          for (var j = 0; j < n; j++) { html += '<i></i>'; }
          box.innerHTML = html;
          box.setAttribute('data-n', String(n));
          box.setAttribute('data-last', '0');
          for (var z = 0; z < Math.min(n, done); z++) {
            if (box.children[z]) { box.children[z].classList.add('vs-seg-on'); }
          }
          box.setAttribute('data-last', String(done));
          continue;
        }
        var last = parseInt(box.getAttribute('data-last') || '0', 10);
        if (done === last) { continue; }
        var kids = box.children;
        for (var m = 0; m < kids.length; m++) {
          var on = m < done;
          var isOn = kids[m].classList.contains('vs-seg-on');
          if (on && !isOn) {
            (function (node, d) {
              setTimeout(function () { node.classList.add('vs-seg-on'); },
                RM ? 0 : Math.min(7, Math.max(0, d)) * 55);
            })(kids[m], m - last);
          } else if (!on && isOn) {
            kids[m].classList.remove('vs-seg-on');
          }
        }
        box.setAttribute('data-last', String(done));
      }
    }

    /* ============ 3. 主循环 ============ */
    function collect(now) {
      var list = document.querySelectorAll(SEL);
      var pending = false;
      for (var i = 0; i < list.length; i++) {
        var st = track(list[i], now);
        put(list[i], st, now);          // 首次出现立即落到目标值，避免从 0 起跳
        if (st.dur > 0 && (now - st.t0) < st.dur) { pending = true; }
      }
      return pending;
    }
    function frame() {
      var now = performance.now(), busy = false;
      for (var i = NODES.length - 1; i >= 0; i--) {
        var el = NODES[i];
        if (!el || !el.isConnected || !el._vsSt) { NODES.splice(i, 1); continue; }
        if (put(el, el._vsSt, now)) { busy = true; }
      }
      settleState();
      drawSegs();
      BUSY = busy;
      if (busy) { requestAnimationFrame(frame); }
    }
    function kick() {
      var now = performance.now();
      var pending = collect(now);
      if (!BUSY && pending) { BUSY = true; requestAnimationFrame(frame); }
      else if (!BUSY) { settleState(); drawSegs(); }
    }
    setInterval(kick, 120);
    document.addEventListener('DOMContentLoaded', kick);
    window.addEventListener('load', kick);
    kick();

    /* ===== 2.1 交互反馈：把 .vs-hover-lift 挂到真实存在的按钮 / 折叠面板上。
       CSS 里早就定义了 .vs-hover-lift，但没有任何元素用过它（选择器命中数 0），
       所以按钮与折叠面板一直缺悬浮反馈。这里按 Gradio 6 真实结构补挂。 ===== */
    function liftScan() {
      try {
        var list = document.querySelectorAll(
          '.gradio-container button:not([role="tab"]), .gradio-container .label-wrap');
        for (var i = 0; i < list.length; i++) {
          var n = list[i];
          if (!n.classList || n.classList.contains('vs-hover-lift')) { continue; }
          if ((n.textContent || '').trim() === '') { continue; }   /* 纯图标按钮跳过 */
          if (n.closest && (n.closest('.tab-container') || n.closest('.vs-prog')
                            || n.closest('.vs-toolbar'))) { continue; }
          n.classList.add('vs-hover-lift');
        }
      } catch (e) {}
    }
    liftScan();
    setInterval(liftScan, 1200);

    /* ===== 3.1 关键修复（两代）：Gradio 轮询更新进度卡有两种方式——
       (a) 重建节点；(b) DOM diff 就地更新（morphdom 式：复用节点，只改
       data-p 属性、style 与文本；2026-09-29 用 MutationObserver 取证确认）。
       两种方式都必须赶在浏览器绘制前把内容接续成动画状态，否则 (b) 会
       先闪一帧「裸目标值」再被旧动画值回拖（视觉上进度数字往回跳）。
       守卫升级：新节点 或 data-p 未被消费，都立刻 kick。 ===== */
    var SEEN = (typeof WeakSet === 'function') ? new WeakSet() : null;
    function scanStale() {
      var list = document.querySelectorAll(SEL), stale = false;
      for (var i = 0; i < list.length; i++) {
        var el = list[i];
        if (SEEN && !SEEN.has(el)) { SEEN.add(el); }
        var dp = el.getAttribute('data-p');
        if (el.__vsDp !== dp) { el.__vsDp = dp; stale = true; }
      }
      return stale;
    }
    if (typeof MutationObserver === 'function') {
      try {
        var mo = new MutationObserver(function () {
          try {
            if (!scanStale()) { return; }
            kick();
          } catch (e) {}
        });
        mo.observe(document.documentElement, {
          childList: true, subtree: true,
          characterData: true,
          attributes: true, attributeFilter: ['data-p']
        });
      } catch (e) {}
    }

    /* ============ 4. 入场动画（标签页 / 折叠面板 / 列表新增行） ============ */
    /* 幽灵层守卫：Gradio 的 .hide 占位层与 status-tracker 是「绝对定位 + 白底 + z-index:30」，
       给它们挂 fill:both 的入场动画会被永久定格在 opacity:1，直接盖住整个折叠面板（2026-09-29 修复）。 */
    function skippable(n) {
      if (!n || !n.classList) { return true; }
      if (n.classList.contains('hide')) { return true; }
      if (n.getAttribute && n.getAttribute('data-testid') === 'status-tracker') { return true; }
      return false;
    }
    /* 动画类播完即摘：避免 fill:both 的类永久残留（残留会让下次展开没有动画，也埋下遮挡隐患） */
    function playOnce(n, cls, ms) {
      var key = '_vsT_' + cls;
      if (n[key]) { clearTimeout(n[key]); }
      n.classList.add(cls);
      n[key] = setTimeout(function () {
        n[key] = null;
        try { n.classList.remove(cls); } catch (e) {}
      }, ms);
    }
    function replay(list) {
      for (var i = 0; i < list.length; i++) {
        var n = list[i];
        if (skippable(n)) { continue; }
        n.classList.remove('vs-in');
        void n.offsetWidth;
        playOnce(n, 'vs-in', 620);
      }
    }
    /* 标签页整块面板只做「纯透明度」淡入：不动 transform、不触发布局，最大程度避免卡顿 */
    function fade(list) {
      for (var i = 0; i < list.length; i++) {
        var n = list[i];
        if (skippable(n)) { continue; }
        n.classList.remove('vs-fade');
        void n.offsetWidth;
        playOnce(n, 'vs-fade', 420);
      }
    }
    document.addEventListener('click', function (ev) {
      var t = ev.target;
      if (!t || !t.closest) { return; }
      if (t.closest('.tab-nav button, button[role="tab"]')) {
        setTimeout(function () {
          var panels = document.querySelectorAll('.tabitem, [role="tabpanel"]');
          for (var i = 0; i < panels.length; i++) {
            var p = panels[i];
            if (p.hidden || p.getAttribute('aria-hidden') === 'true') { continue; }
            if (p.offsetParent && p.offsetHeight > 0) { fade([p]); }
          }
        }, 40);
        return;
      }
      var wrap = t.closest('.label-wrap');
      if (wrap) {
        var box = wrap.parentElement, before = box ? box.offsetHeight : 0;
        setTimeout(function () {
          if (!box || box.offsetHeight <= before + 4) { return; }
          var kids = box.querySelectorAll(':scope > div:not(.hide):not([data-testid="status-tracker"])');
          replay(kids.length ? kids : [box]);
        }, 60);
      }
    }, true);

    /* 任务表格：仅在行数增加时做一次错峰入场，避免定时刷新闪烁 */
    var lastRows = -1;
    setInterval(function () {
      try {
        var tb = document.querySelector('#vs-task-table tbody');
        if (!tb) { return; }
        var rows = tb.querySelectorAll('tr');
        if (rows.length === lastRows) { return; }
        var grew = rows.length > lastRows;
        lastRows = rows.length;
        if (!grew) { return; }
        for (var i = 0; i < rows.length; i++) {
          rows[i].style.animationDelay = Math.min(i, 10) * 28 + 'ms';
          replay([rows[i]]);
        }
      } catch (e) {}
    }, 700);

    /* ============ 5. 浏览器下载完成 → 通知服务端删掉本机副本 ============ */
    var toastTimer = null;
    function toast(text) {
      try {
        var el = document.querySelector('.vs-toast');
        if (!el) {
          el = document.createElement('div');
          el.className = 'vs-toast';
          document.body.appendChild(el);
        }
        el.textContent = text;
        el.style.animation = 'none';
        void el.offsetWidth;
        el.style.animation = '';
        if (toastTimer) { clearTimeout(toastTimer); }
        toastTimer = setTimeout(function () { try { el.remove(); } catch (e) {} }, 2600);
      } catch (e) {}
    }
    function notifyDownload(path) {
      if (!path) { return; }
      try {
        fetch('/vs_del?f=' + encodeURIComponent(path),
              { method: 'POST', credentials: 'same-origin' })
          .then(function (r) { return r.text(); })
          .then(function (t) { if (t === 'ok') { toast('已清理输出目录中的本机副本'); } })
          .catch(function () {});
      } catch (e) {}
    }
    document.addEventListener('click', function (ev) {
      var t = ev.target;
      if (!t || !t.closest) { return; }
      var a = t.closest('a[download], a[href*="file="]');
      if (!a) { return; }
      var href = a.getAttribute('href') || '';
      var i = href.indexOf('file=');
      if (i < 0) { return; }
      var raw = href.slice(i + 5);
      var amp = raw.indexOf('&');
      if (amp >= 0) { raw = raw.slice(0, amp); }
      try { raw = decodeURIComponent(raw); } catch (e) {}
      setTimeout(function () { notifyDownload(raw); }, 1500);
    }, true);
  } catch (e) {}
})();
</script>

<script>
/* ===========================================================================
   原生窗口下载桥（vsDownloadBridge）
   在 pywebview / WebView2 里，点击 Gradio 的下载链接不会有任何反应（浏览器里正常）。
   这里在捕获阶段拦下这类点击，改叫 Python 把本机文件复制到「下载」目录，
   并在右下角弹一条提示（可点「打开所在文件夹」）。
   普通浏览器里 window.pywebview 不存在 -> 完全不拦截，走原生下载。
   =========================================================================== */
(function () {
  if (window.vsDownloadBridge) { return; }
  window.vsDownloadBridge = 1;

  function api() { return window.pywebview && window.pywebview.api; }

  function toast(msg, path) {
    var box = document.getElementById('vs-dl-toast');
    if (!box) {
      box = document.createElement('div');
      box.id = 'vs-dl-toast';
      box.style.cssText = 'position:fixed;right:18px;bottom:18px;z-index:2147483600;max-width:460px;' +
        'padding:12px 14px;border-radius:14px;font-size:13px;line-height:1.5;color:#fff;' +
        'background:rgba(22,24,34,.92);border:1px solid rgba(255,255,255,.22);' +
        'box-shadow:0 18px 44px -18px rgba(0,0,0,.8);backdrop-filter:blur(14px);' +
        'transition:opacity .3s,transform .3s;opacity:0;transform:translateY(8px)';
      document.body.appendChild(box);
    }
    box.innerHTML = '';
    var t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = 'word-break:break-all';
    box.appendChild(t);
    if (path) {
      var b = document.createElement('button');
      b.textContent = '打开所在文件夹';
      b.style.cssText = 'margin-top:8px;padding:5px 10px;border-radius:9px;cursor:pointer;' +
        'border:1px solid rgba(255,255,255,.28);background:rgba(255,255,255,.12);color:#fff;font-size:12px';
      b.onclick = function () { var a = api(); if (a && a.reveal) { a.reveal(path); } };
      box.appendChild(b);
    }
    requestAnimationFrame(function () { box.style.opacity = '1'; box.style.transform = 'none'; });
    clearTimeout(box._tm);
    box._tm = setTimeout(function () {
      box.style.opacity = '0'; box.style.transform = 'translateY(8px)';
    }, 9000);
  }

  document.addEventListener('click', function (ev) {
    var a = api();
    if (!a || !a.save_file) { return; }          // 普通浏览器：不拦
    var el = ev.target;
    while (el && el !== document.body) {
      if (el.tagName === 'A' || el.tagName === 'BUTTON') { break; }
      el = el.parentElement;
    }
    if (!el || el === document.body) { return; }
    var href = el.getAttribute('href') || el.getAttribute('data-href') || '';
    var isFile = href.indexOf('gradio_api/file=') >= 0 || el.hasAttribute('download');
    if (!isFile) { return; }
    ev.preventDefault();
    ev.stopPropagation();
    var name = el.getAttribute('download') || '';
    try {
      a.save_file(href, name).then(function (r) {
        if (r && r.ok) { toast('已保存到：' + r.path, r.path); }
        else { toast('保存失败：' + ((r && r.err) || '未知错误')); }
      }).catch(function (e) { toast('保存失败：' + e); });
    } catch (e) { toast('保存失败：' + e); }
  }, true);
})();
</script>
"""


# 顶部极简工具条（不再放大图标 / 大标题 / 副标题；纯内联 SVG，不依赖图标字体或 CDN）
_ICO_SUN = ('<svg class="vs-ico-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4.1"/>'
            '<path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.3 5.3l1.6 1.6'
            'M17.1 17.1l1.6 1.6M18.7 5.3l-1.6 1.6M6.9 17.1l-1.6 1.6"/></svg>')
_ICO_MOON = ('<svg class="vs-ico-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="1.8" stroke-linecap="round"><path d="M20.4 14.7A8.6 8.6 0 0 1 9.3 3.6'
             'a8.8 8.8 0 1 0 11.1 11.1Z"/></svg>')
_ICO_GLASS = ('<svg class="vs-ico-glass" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
              '<path d="M5.2 4.6h13.6l-1.5 15.8H6.7L5.2 4.6Z"/><path d="M6.6 9.3h10.8"/>'
              '<path d="M12 12.2c1.9 0 3 1.1 3 2.6s-1.4 2.4-3 2.4-2.6-1.1-2.6-2.4 1-2.6 2.6-2.6Z"/></svg>')

_ICO_IMG = ('<svg class="vs-ico-img" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="3.2" y="4.6" width="17.6" height="14.8" rx="3.4"/>'
            '<circle cx="9.1" cy="10" r="1.7"/>'
            '<path d="M4.3 17.6l4.7-4.4 3.2 2.9 3-2.6 4.5 4.1"/></svg>')

TOOLBAR_HTML = """
<style>
  html.dark .vs-ico-moon { display: none; }
  html:not(.dark) .vs-ico-sun { display: none; }
</style>
<div class="vs-toolbar">
  <span class="vs-status">
    <span class="vs-status-dot"></span>
    <span>本地离线运行 · 模型全部内置 · 音频不外传</span>
  </span>
  <span class="vs-spacer"></span>
  <button class="vs-icon-btn vs-look-btn" type="button"
          title="外观：液态玻璃开关 / 磨砂强度 / 背景与配色"
          onclick="window.vsToggleLook &amp;&amp; window.vsToggleLook()">
    __GLASS__<span>外观</span>
  </button>
  <button class="vs-icon-btn" type="button" title="切换浅色 / 深色主题"
          onclick="window.vsToggleTheme &amp;&amp; window.vsToggleTheme()">
    __ICONS__<span>主题</span>
  </button>
</div>
""".replace("__ICONS__", _ICO_SUN + _ICO_MOON).replace("__GLASS__", _ICO_GLASS).replace("__BG__", _ICO_IMG)


def _esc(v):
    return (str(v if v is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


_PROG_STATE = {STATUS_DONE: "done", STATUS_FAILED: "failed"}


def _prog_state(task):
    if not task:
        return "idle"
    return _PROG_STATE.get(task.get("status"), "running")


def _prog_pct(task):
    try:
        return max(0.0, min(1.0, float((task or {}).get("progress") or 0.0))) * 100.0
    except (TypeError, ValueError):
        return 0.0


SCALE_SEGS = 20   # 拿不到真实分块数时的刻度尺格数（只表示整体进度，不代表真实分块）


def _segs_html(task=None, pct=0.0):
    """分块刻度行。

    - 有真实分块数（seg_total）时：一格 = 一个推理分块，点亮数量 = seg_done；
    - 拿不到真实分块数（空闲 / 尚未上报）时：退化成 20 格「刻度尺」，
      data-kind="scale"，点亮比例跟随整体进度（样式上更细更淡，避免被误读成分块数）。
    无论哪种情况都必须渲染出 .vs-segs，否则进度条下面只剩一根空轨道。
    """
    seg_total, seg_done = 0, 0
    if task:
        try:
            seg_total = int(task.get("seg_total") or 0)
            seg_done = int(task.get("seg_done") or 0)
        except (TypeError, ValueError):
            seg_total, seg_done = 0, 0
    if 1 <= seg_total <= 300:
        done = max(0, min(seg_done, seg_total))
        return ('<div class="vs-segs" data-kind="chunk" data-segs="{}" data-done="{}"></div>'
                .format(seg_total, done))
    n = SCALE_SEGS
    done = int(round(max(0.0, min(100.0, pct)) / 100.0 * n))
    return ('<div class="vs-segs" data-kind="scale" data-segs="{}" data-done="{}"></div>'
            .format(n, done))


def progress_html(task=None):
    """进度卡片：轨道 + 拖尾 + 弹性填充 + 大号百分比 + 阶段 + 分块刻度 + ETA。
    宽度与数字均由前端逐帧驱动（data-p 为目标值），跨 2s 定时重渲染保持连续。"""
    if not task:
        return (
            '<div class="vs-prog" data-state="idle" data-vskey="idle">'
            '<div class="vs-prog-top">'
            '<div class="vs-prog-stage"><span class="vs-idle-dot"></span>'
            '<span class="vs-stage-text">空闲 · 暂无运行中的任务</span></div>'
            '<div class="vs-prog-pct vs-pct-idle"><span class="vs-dash">—</span></div>'
            '</div>'
            '<div class="vs-track">'
            '<span class="vs-ghost" data-p="0" style="width:0%"></span>'
            '<span class="vs-fill" data-p="0" style="width:0%"></span>'
            '</div>'
            + _segs_html(None) +
            '<div class="vs-prog-meta">'
            '<span class="vs-chip">1 · 上传音频</span>'
            '<span class="vs-chip">2 · 选模型/预设</span>'
            '<span class="vs-chip">3 · 开始分离</span>'
            '<span class="vs-chip">完成后可试听 / 下载 / 导出详情</span>'
            '</div></div>'
        )

    status = task.get("status") or ""
    if status == STATUS_DONE:
        state = "done"
    elif status == STATUS_FAILED:
        state = "failed"
    elif status == STATUS_PAUSED:
        state = "paused"
    else:
        state = "running"

    try:
        pct = round(max(0.0, min(1.0, float(task.get("progress") or 0.0))) * 100.0, 1)
    except (TypeError, ValueError):
        pct = 0.0
    dec = 0 if pct >= 10 else 1

    tid = str(task.get("id") or "focus")
    stage = _esc(task.get("stage") or status or "")
    src = Path(str(task.get("source") or "")).name

    head_dot = '<span class="vs-live-dot"></span>' if state == "running" else '<span class="vs-idle-dot"></span>'
    dots = ('<span class="vs-dots"><i></i><i></i><i></i></span>' if state == "running" else "")

    chips = [f'<span class="vs-chip">{_esc(task.get("mode") or "单曲分离")}</span>']
    if src:
        chips.append(f'<span class="vs-chip" title="{_esc(src)}">{_esc(src)}</span>')
    if task.get("duration"):
        chips.append(f'<span class="vs-chip">时长 {fmt_sec(task.get("duration"))}</span>')
    chips.append(f'<span class="vs-chip">已用 {fmt_sec(task.get("elapsed"))}</span>')
    if task.get("eta") is not None and state in ("running", "paused"):
        chips.append(f'<span class="vs-chip vs-chip-accent">剩余 {fmt_sec(task.get("eta"))}</span>')
    if task.get("seg_total"):
        chips.append(f'<span class="vs-chip">分块 {task.get("seg_done") or 0}/{task.get("seg_total")}</span>')
    if state == "paused":
        chips.append('<span class="vs-chip vs-chip-accent">已暂停 · 点「继续」接着跑</span>')
    if state == "done":
        if task.get("load_sec") is not None or task.get("sep_sec") is not None:
            chips.append('<span class="vs-chip">加载 {} · 推理 {}</span>'.format(
                fmt_sec(task.get("load_sec") or 0), fmt_sec(task.get("sep_sec") or 0)))
        chips.append(f'<span class="vs-chip">输出 {task.get("file_count") or 0} 个文件</span>')
        chips.append('<span class="vs-chip">浏览器下载后会自动清理本机副本</span>')

    badge = ""
    if state == "done":
        badge = '<span class="vs-badge-done">已完成</span>'
    elif state == "failed":
        badge = '<span class="vs-badge-failed">失败</span>'
    elif state == "paused":
        badge = '<span class="vs-badge-paused">已暂停</span>'

    segs = _segs_html(task, pct)

    tail = ""
    if state == "done":
        out = _esc(task.get("output_dir") or "")
        cleaned = "（本机副本已清理）" if task.get("local_cleaned") else ""
        tail = (f'<div class="vs-tail">输出目录：<code>{out}</code>{cleaned}</div>')
    elif state == "failed" and task.get("error"):
        tail = f'<div class="vs-tail vs-tail-err">{_esc(task.get("error"))}</div>'

    return (
        f'<div class="vs-prog" data-state="{state}" data-vskey="{_esc(tid)}">'
        '<div class="vs-prog-top">'
        f'<div class="vs-prog-stage">{head_dot}{dots}'
        f'<span class="vs-stage-text" title="{stage}">{stage}</span></div>'
        f'<div class="vs-prog-pct"><span class="vs-num" data-dec="{dec}" data-p="{pct:.1f}">{pct:.{dec}f}</span>'
        '<span class="vs-unit">%</span></div>'
        '</div>'
        '<div class="vs-track">'
        f'<span class="vs-ghost" data-p="{pct:.1f}" style="width:{pct:.1f}%"></span>'
        f'<span class="vs-fill" data-p="{pct:.1f}" style="width:{pct:.1f}%"></span>'
        '</div>'
        f'{segs}'
        f'<div class="vs-prog-meta">{badge}{"".join(chips)}</div>'
        f'{tail}'
        '</div>'
    )


def activity_html():
    """全局活动指示（顶部一行，任意标签页都能看到正在跑的任务）。"""
    tasks = [t for t in LEDGER.list_tasks()
             if t.get("status") in ACTIVE_STATES or t.get("status") == STATUS_PAUSED]
    if not tasks:
        return '<div class="vs-float" data-state="idle"></div>'
    t = tasks[0]
    try:
        pct = round(max(0.0, min(1.0, float(t.get("progress") or 0.0))) * 100.0, 1)
    except (TypeError, ValueError):
        pct = 0.0
    stage = _esc(t.get("stage") or t.get("status") or "")
    more = f'<span class="vs-chip">另有 {len(tasks) - 1} 个任务在队列中</span>' if len(tasks) > 1 else ""
    return (
        '<div class="vs-float" data-state="running" data-vskey="global">'
        '<span class="vs-live-dot"></span>'
        f'<span>{_esc(t.get("mode") or "任务")} · {_esc(t.get("status") or "")}</span>'
        f'<span class="vs-mini-track"><span class="vs-mini-fill" data-p="{pct:.1f}" '
        f'style="width:{pct:.1f}%"></span></span>'
        f'<span class="vs-num" data-dec="0" data-p="{pct:.1f}">{pct:.0f}</span><span class="vs-unit">%</span>'
        f'<span class="vs-chip">{stage}</span>{more}'
        '</div>'
    )


def _focus_task():
    """当前应聚焦展示的任务：优先运行中最久的一个，否则最近的一条。
"""
    tasks = LEDGER.list_tasks()
    if not tasks:
        return None
    active = [t for t in tasks if t.get("status") in ACTIVE_STATES]
    if active:
        active.sort(key=lambda t: t.get("created_at") or 0)
        return active[0]
    return tasks[0]

OVERLAP_LABEL = "重叠倍率 overlap（4 默认·与 Roformer 训练配置一致 / 8 更保守 / 16 精细，越高越慢成正比）"
FP16_LABEL = "高级：native fp16 推理（默认开启：实测输出与 fp32 逐样本一致，约快 6%~20%）"
COMPILE_LABEL = "高级：torch.compile 编译加速（默认关闭；首次推理有编译预热耗时）"

with gr.Blocks(title="VocalSeparator · 本地人声分离工作站") as demo:
    gr.HTML(TOOLBAR_HTML)
    global_activity = gr.HTML(activity_html())
    gr.Markdown(
        "任务提交后持续运行：切换标签或刷新页面都不会中断，页内进度条与「任务列表」都会实时刷新。"
    )

    # 全局轮询计时器：必须写在 Blocks 顶层并正常挂载（render=False 会让组件不上屏、tick 永不触发）
    tick_timer = gr.Timer(2.0)
    # 页面会话标记：每个页面（含每次刷新）由 demo.load 发一个新值，
    # 轮询的去重缓存按它隔离，保证新页面第一帧就能拿到真实内容。
    page_sid = gr.State("")

    with gr.Tabs():
        # ================= 单曲分离 =================
        with gr.Tab("单曲分离"):
            with gr.Row():
                with gr.Column(scale=4):
                    audio_in = gr.Audio(label="输入音频（支持 wav / mp3 / flac / m4a 等）",
                                        sources=["upload"], type="filepath")
                    tier = gr.Radio([TIER_SINGLE, TIER_PRESET, TIER_CUSTOM], value=TIER_SINGLE,
                                    label="分离模式")
                    single_model = gr.Dropdown(
                        choices=build_model_choices(),
                        value=DEFAULT_MODEL if DEFAULT_MODEL in ALL_MODEL_FILES else None,
                        label="单模型（仅列本地内置模型，可搜索；标签已标注推荐 / 速度 / 擅长轨）",
                        info=MODEL_HELP,
                        filterable=True,
                    )
                    preset_dd = gr.Dropdown(choices=build_preset_choices(),
                                            value="builtin_vocal_revive",
                                            label="内置集成组合（成员全为本地模型）", visible=False, filterable=True)
                    custom_models = gr.Dropdown(choices=build_model_choices(), multiselect=True,
                                                label="集成模型（选 2 个以上，仅列本地内置）",
                                                info=MODEL_HELP, visible=False, filterable=True)
                    algo_dd = gr.Dropdown(choices=ENSEMBLE_ALGOS, value="avg_fft",
                                          label="集成算法", visible=False)

                    with gr.Accordion("输出与参数（默认已是效果优先）", open=False):
                        fmt = gr.Radio(["WAV", "FLAC", "MP3"], value="WAV", label="输出格式")
                        sr = gr.Radio([44100, 48000], value=44100, label="采样率")
                        single_stem = gr.Textbox(value="", label="只输出指定主干（留空=全部）",
                                                 placeholder="如 Vocals / Instrumental")
                        batch = gr.Slider(1, 16, value=8, step=1, label="批大小 batch_size（显存换速度）")
                        seg = gr.Slider(0, 1024, value=0, step=64, label="分段长度 segment_size（0 = 跟随模型原生）")
                        overlap = gr.Slider(2, 64, value=4, step=1, label=OVERLAP_LABEL)
                        shifts = gr.Slider(1, 8, value=4, step=1, label="Demucs shifts（多轨模型，越高越精细）")
                        dem_overlap = gr.Slider(0.0, 0.9, value=0.25, step=0.05, label="Demucs overlap")
                        norm = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="归一化阈值")
                        amp = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="增益阈值")

                    with gr.Accordion("人声去回声（可选 · RVC 翻唱前置，分离完成后自动执行）", open=False):
                        dereverb_ck = gr.Checkbox(
                            value=False,
                            label="分离完成后，对「人声」轨额外去回声与去混响（两段式：1/2 分离 → 2/2 人声去回声）")
                        dereverb_model_dd = gr.Dropdown(
                            choices=list(DEREVERB_CHOICES.keys()), value=DEFAULT_DEREVERB,
                            label="去回声 / 去混响模型", filterable=True)
                        dereverb_passes_dd = gr.Dropdown(
                            choices=list(DEREVERB_PASSES.keys()), value=DEFAULT_DEREVERB_PASSES,
                            label="去回声遍数（2 遍残留更少，耗时翻倍）")
                        dereverb_match_ck = gr.Checkbox(
                            value=True,
                            label="自动等响（推荐：把去回声产物拉回与分离人声同等音量）")
                        dereverb_air_ck = gr.Checkbox(
                            value=False,
                            label="额外：补回 8kHz 以上空气感（只有老式 anvuew 去混响需要；会带回一部分回声）")
                        gr.Markdown(
                            "RVC 翻唱请开启：带混响/回声的人声会让翻唱模型把残响也学成人声，"
                            "出现闷糊、回声叠加等错误。开启后先正常分离，再只对分离出的「人声」轨做一遍"
                            "去回声推理，多交付一份 `歌名_人声去回声.wav`，直接用这份做 RVC 输入即可。"
                            "去回声是与分离同量级的额外一遍推理（只处理人声轨），长音频请预留时间。"
                            "\n\n**默认模型已换成专用去回声模型**：旧版的 anvuew 是「去混响」，"
                            "对回声（延迟尾巴）几乎无效；现在默认用 De-Reverb-Echo V2 —— 同一段真实湿人声上"
                            "实测 SDR 由 15.1 提升到 24.3，且不掉音量、不掉高频。"
                            "\n\n**为什么「补回高频」默认不勾**：新模型本来就不掉 8kHz 以上，"
                            "硬补等于把原人声里的回声高频再灌回去，越补回声越多。")

                    with gr.Accordion("高级推理选项（默认关闭，改动可能影响结果一致性）", open=False):
                        use_native_fp16 = gr.Checkbox(value=True, label=FP16_LABEL)
                        use_torch_compile = gr.Checkbox(value=False, label=COMPILE_LABEL)

                    run_btn = gr.Button("开始分离", variant="primary", size="lg",
                                        elem_classes=["vs-cta"])
                    single_tid = gr.State("")

                with gr.Column(scale=5):
                    single_prog = gr.HTML(progress_html())
                    status_md = gr.Markdown("等待开始…")
                    stems = [gr.Audio(label=f"主干 {i + 1}", visible=False, type="filepath")
                             for i in range(MAX_STEMS)]
                    files_out = gr.Files(label="全部输出文件（可下载/试听）")
                    zip_out = gr.File(label="打包下载")
                    with gr.Row():
                        btn_pause = gr.Button("暂停", scale=1)
                        btn_resume = gr.Button("继续", scale=1)
                        btn_clean = gr.Button("清理本机副本", scale=1)
                    ctl_msg = gr.Markdown("")

            def _tier_change(t):
                return (
                    gr.update(visible=t == TIER_SINGLE),
                    gr.update(visible=t == TIER_PRESET),
                    gr.update(visible=t == TIER_CUSTOM),
                    gr.update(visible=t == TIER_CUSTOM),
                )

            tier.change(_tier_change, inputs=tier,
                        outputs=[single_model, preset_dd, custom_models, algo_dd],
                        show_progress="hidden")

            run_btn.click(
                submit_single,
                inputs=[audio_in, tier, single_model, preset_dd, custom_models, algo_dd, fmt, sr,
                        batch, seg, overlap, shifts, dem_overlap, norm, amp, single_stem,
                        use_native_fp16, use_torch_compile, dereverb_ck, dereverb_model_dd,
                        dereverb_passes_dd, dereverb_match_ck, dereverb_air_ck],
                outputs=[single_tid, status_md, single_prog] + stems + [files_out, zip_out],
                show_progress="hidden",
            )

            btn_pause.click(pause_task, inputs=[single_tid],
                            outputs=[ctl_msg, single_prog, global_activity],
                            show_progress="hidden")
            btn_resume.click(resume_task, inputs=[single_tid],
                             outputs=[ctl_msg, single_prog, global_activity],
                             show_progress="hidden")
            btn_clean.click(cleanup_local_outputs, inputs=[single_tid],
                            outputs=[ctl_msg, files_out, zip_out],
                            show_progress="hidden")

            # 本页轮询改由页面底部唯一的统一 tick 驱动（见 poll_all），不再单独挂 timer

        # ================= 批量分离 =================
        with gr.Tab("批量分离"):
            gr.Markdown("上传多个文件或整个文件夹，用同一套本地内置模型连续处理，结果自动打包。"
                        "任务提交后持续运行，本页进度条与「任务列表」都会实时刷新。")
            with gr.Row():
                with gr.Column(scale=4):
                    batch_in = gr.Files(label="音频文件 / 文件夹（可多选）", file_count="multiple")
                    b_tier = gr.Radio([TIER_SINGLE, TIER_PRESET], value=TIER_SINGLE, label="分离模式")
                    b_model = gr.Dropdown(info=MODEL_HELP, choices=build_model_choices(),
                                          value=DEFAULT_MODEL if DEFAULT_MODEL in ALL_MODEL_FILES else None,
                                          label="单模型", filterable=True)
                    b_preset = gr.Dropdown(choices=build_preset_choices(),
                                           value="builtin_vocal_revive",
                                           label="内置集成组合", visible=False, filterable=True)
                    with gr.Accordion("输出与参数", open=False):
                        b_fmt = gr.Radio(["WAV", "FLAC", "MP3"], value="WAV", label="输出格式")
                        b_sr = gr.Radio([44100, 48000], value=44100, label="采样率")
                        b_batch = gr.Slider(1, 16, value=8, step=1, label="批大小 batch_size")
                        b_seg = gr.Slider(0, 1024, value=0, step=64, label="分段长度 segment_size（0 = 跟随模型原生）")
                        b_overlap = gr.Slider(2, 64, value=4, step=1, label=OVERLAP_LABEL)
                        b_shifts = gr.Slider(1, 8, value=4, step=1, label="Demucs shifts（越高越精细）")
                        b_dem_ov = gr.Slider(0.0, 0.9, value=0.25, step=0.05, label="Demucs overlap")
                        b_norm = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="归一化阈值")
                        b_amp = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="增益阈值")
                    with gr.Accordion("人声去回声（可选 · RVC 翻唱前置，分离完成后自动执行）", open=False):
                        b_dereverb = gr.Checkbox(value=False,
                                                 label="每个文件分离完成后，对「人声」轨额外去回声与去混响")
                        b_dereverb_model = gr.Dropdown(choices=list(DEREVERB_CHOICES.keys()),
                                                       value=DEFAULT_DEREVERB,
                                                       label="去回声 / 去混响模型", filterable=True)
                        b_dereverb_passes = gr.Dropdown(choices=list(DEREVERB_PASSES.keys()),
                                                        value=DEFAULT_DEREVERB_PASSES,
                                                        label="去回声遍数（2 遍残留更少，耗时翻倍）")
                        b_dereverb_match = gr.Checkbox(
                            value=True,
                            label="自动等响（推荐：去完与原人声同等音量）")
                        b_dereverb_air = gr.Checkbox(
                            value=False,
                            label="额外：补回 8kHz 以上空气感（会带回一部分回声，默认关）")
                    with gr.Accordion("高级推理选项（默认关闭，改动可能影响结果一致性）", open=False):
                        b_fp16 = gr.Checkbox(value=True, label=FP16_LABEL)
                        b_compile = gr.Checkbox(value=False, label=COMPILE_LABEL)
                    b_btn = gr.Button("批量开始", variant="primary", size="lg",
                                      elem_classes=["vs-cta"])
                    batch_tid = gr.State("")
                with gr.Column(scale=5):
                    b_prog = gr.HTML(progress_html())
                    b_status = gr.Markdown("等待开始…")
                    b_files = gr.Files(label="全部输出文件")
                    b_zip = gr.File(label="打包下载")
                    with gr.Row():
                        b_pause = gr.Button("暂停", scale=1)
                        b_resume = gr.Button("继续", scale=1)
                        b_clean = gr.Button("清理本机副本", scale=1)
                    b_ctl_msg = gr.Markdown("")

            b_tier.change(lambda t: (gr.update(visible=t == TIER_SINGLE), gr.update(visible=t == TIER_PRESET)),
                          inputs=b_tier, outputs=[b_model, b_preset],
                          show_progress="hidden")

            b_btn.click(
                submit_batch,
                inputs=[batch_in, b_tier, b_model, b_preset, b_fmt, b_sr, b_batch, b_seg,
                        b_overlap, b_shifts, b_dem_ov, b_norm, b_amp, b_fp16, b_compile,
                        b_dereverb, b_dereverb_model, b_dereverb_passes, b_dereverb_match,
                        b_dereverb_air],
                outputs=[batch_tid, b_status, b_prog, b_files, b_zip],
                show_progress="hidden",
            )

            b_pause.click(pause_task, inputs=[batch_tid],
                          outputs=[b_ctl_msg, b_prog, global_activity],
                          show_progress="hidden")
            b_resume.click(resume_task, inputs=[batch_tid],
                           outputs=[b_ctl_msg, b_prog, global_activity],
                           show_progress="hidden")
            b_clean.click(cleanup_local_outputs, inputs=[batch_tid],
                          outputs=[b_ctl_msg, b_files, b_zip],
                          show_progress="hidden")

            # 本页轮询同样由页面底部唯一的统一 tick 驱动（见 poll_all）

        # ================= 任务列表 =================
        with gr.Tab("任务列表"):
            gr.Markdown("任务台账（JSON 落盘）：分离提交后持续运行，刷新页面或切换标签后依然能看到"
                        "运行中与历史任务。")
            tl_summary = gr.Markdown(task_summary_text())
            tl_prog = gr.HTML(progress_html(_focus_task()))
            with gr.Row():
                tl_refresh = gr.Button("立即刷新", scale=1)
                tl_pick = gr.Dropdown(choices=task_choices(), label="选择任务（用于打开输出目录）",
                                      scale=3, filterable=True)
                tl_open = gr.Button("打开输出目录", scale=1)
            # wrap=False：列宽按内容撑开（表格内部横向滚动），
            # 避免「模式」列被挤到 4 个字竖排换行（如「单曲分 / 离」）。
            tl_table = gr.Dataframe(headers=TASK_COLUMNS, value=task_rows(),
                                    interactive=False, wrap=False, elem_id="vs-task-table")
            tl_msg = gr.Markdown("")
            with gr.Row():
                tl_pause = gr.Button("暂停选中任务", scale=1)
                tl_resume = gr.Button("继续选中任务", scale=1)
                tl_clean = gr.Button("清理本机副本", scale=1)
            with gr.Row():
                tl_export = gr.Button("导出选中任务详情", variant="primary", scale=1)
                tl_export_all = gr.Button("导出全部任务详情", scale=1)
            export_file = gr.File(label="导出的详情文件（点击下载）", visible=False)

            _tl_outputs = [tl_table, tl_summary, tl_pick, tl_prog, global_activity]
            tl_refresh.click(poll_task_list, outputs=_tl_outputs, show_progress="hidden")
            tl_open.click(open_task_dir, inputs=tl_pick, outputs=tl_msg, show_progress="hidden")
            tl_pause.click(pause_task, inputs=tl_pick,
                           outputs=[tl_msg, tl_prog, global_activity],
                           show_progress="hidden")
            tl_resume.click(resume_task, inputs=tl_pick,
                            outputs=[tl_msg, tl_prog, global_activity],
                            show_progress="hidden")
            tl_clean.click(cleanup_picked, inputs=tl_pick,
                           outputs=[tl_msg, tl_table, tl_summary, tl_pick, tl_prog,
                                    global_activity],
                           show_progress="hidden")
            tl_export.click(export_task_detail, inputs=tl_pick,
                            outputs=[tl_msg, export_file], show_progress="hidden")
            tl_export_all.click(export_all_tasks, outputs=[tl_msg, export_file],
                                show_progress="hidden")

        # ================= 模型库 =================
        with gr.Tab("模型库"):
            gr.Markdown("共收录全部分离模型；离线模式下仅「已内置」的模型可用。SDR 越高代表分离越干净，`未内置` 的模型需自行放入 models 目录，本工具不会联网下载。")
            with gr.Row():
                kw = gr.Textbox(label="筛选（文件名或名称关键词）", scale=3)
                refresh = gr.Button("刷新", scale=1)
            # wrap=False：同上，「架构」列按内容撑开，不再被右侧裁掉（如「MDXC 1...」）。
            table = gr.Dataframe(headers=["文件", "名称", "架构", "最高SDR", "主干", "状态"],
                                 value=model_table(), interactive=False, wrap=False,
                                 elem_id="vs-model-table")
            kw.submit(model_table, inputs=kw, outputs=table, show_progress="hidden")
            refresh.click(model_table, inputs=kw, outputs=table, show_progress="hidden")

        # ================= 环境 =================
        with gr.Tab("环境"):
            env_md = gr.Markdown(env_report())
            with gr.Row():
                btn_refresh_env = gr.Button("刷新检测")
                btn_out = gr.Button("打开输出目录")
                btn_models = gr.Button("打开模型目录")
            act_md = gr.Markdown("")
            btn_refresh_env.click(env_report, outputs=env_md, show_progress="hidden")
            btn_out.click(lambda: open_dir(OUTPUT_DIR), outputs=act_md, show_progress="hidden")
            btn_models.click(lambda: open_dir(MODEL_DIR), outputs=act_md, show_progress="hidden")

            gr.Markdown("---")
            with gr.Accordion("性能：CPU / GPU 协同设置与实测", open=False):
                gr.Markdown(
                    "GPU 负责卷积推理、CPU 负责读盘 / 重采样 / 拼装，两者本来就在并行。"
                    "下面这些开关是「调度层」的调节项，不改分离质量；不同机器差异很大，"
                    "请用本机「实测对比」的数字做决定，不要照抄别人的结论。")
                with gr.Row():
                    perf_threads = gr.Slider(0, max(2, os.cpu_count() or 8), step=1,
                                             value=int(PERF_CFG.get("cpu_threads") or 0),
                                             label="CPU 线程数（0 = 自动 / 交给 torch）")
                    perf_cudnn = gr.Checkbox(value=bool(PERF_CFG.get("cudnn_benchmark")),
                                             label="cudnn.benchmark（固定分段长度下自动选最优卷积算法）")
                with gr.Row():
                    btn_perf_apply = gr.Button("保存并应用", variant="primary")
                    btn_perf_bench = gr.Button("实测对比（约 3~6 分钟，会占用 GPU）")
                perf_md = gr.Markdown(perf_state_md())
                perf_out = gr.Markdown("")
                btn_perf_apply.click(apply_perf_ui, inputs=[perf_threads, perf_cudnn],
                                     outputs=[perf_md], show_progress="hidden")
                btn_perf_bench.click(bench_perf_ui, inputs=[perf_threads, perf_cudnn],
                                     outputs=[perf_out], show_progress="full")
                gr.Markdown(
                    "**参考结论（本机 RTX 5080 Laptop · bs_roformer · 60 秒样本实测）**\n\n"
                    "- 真正有效的只有 **`overlap`（重叠倍率）**：8 → 16 → 32 耗时近似成倍递增"
                    "（24.3s / 47.5s / 98.0s），因为 RoFormer 会按倍率对同一段重复推理。建议保持 8。\n"
                    "- **CPU 线程数**（4 / 24）、**cudnn.benchmark**、**batch 8→16** 三组差异都 < 5%，"
                    "在测量抖动范围内——GPU 推理时 CPU 不是瓶颈，这类开关只能「不拖后腿」，不会加速。\n"
                    "- **torch.compile** 实测无收益（二次运行 25.1s vs 原生 24.3s），还要多付编译开销，默认关闭。\n"
                    "- 结论：本工具默认配置已接近该模型在本机的最快档；"
                    "「让 CPU 一起参与推理」在 RoFormer 这类 GPU 批处理架构上不会更快，反而增加数据搬运。")

    gr.Markdown(
        f"输出目录：`{OUTPUT_DIR}` · 模型目录：`{MODEL_DIR}` · 模型本地内置、全程离线计算，音频不会上传。"
    )

    # ================= 唯一的统一轮询 tick（关键修复） =================
    # 原实现：同一个 gr.Timer 分别 tick 到 poll_single / poll_batch / poll_task_list，
    # 每 2 秒并发下发 3 条事件链、互相覆盖同一批输出（global_activity 被重复写 3 次），
    # 队列串行执行时事件还会堆积 → 前端表现为卡顿、控件被反复重建（看起来像“页面重置”）。
    # 现在只保留一条事件链，按固定顺序一次性返回三页所需输出。
    _poll_outputs = (list(stems) + [files_out, zip_out, status_md, single_prog]
                     + [b_files, b_zip, b_status, b_prog] + list(_tl_outputs))
    assert len(_poll_outputs) == 21, ("poll_all 返回长度与 outputs 不一致", len(_poll_outputs))
    tick_timer.tick(poll_all, inputs=[single_tid, batch_tid, page_sid],
                    outputs=_poll_outputs, show_progress="hidden")
    # 页面每次加载（含刷新）分配一个新的会话标记，用于隔离轮询去重缓存
    demo.load(_new_sid, outputs=[page_sid], show_progress="hidden")


# ---------------------------------------------------------------------------
# 浏览器下载后自动清理本机副本（前端下载完成后回调）
# ---------------------------------------------------------------------------
def _register_download_cleanup_route():
    try:
        @demo.app.post("/vs_del")
        def _vs_del(f: str = ""):
            return vs_mark_downloaded(f)
        LOG.info("已登记下载清理接口 /vs_del")
    except Exception:  # noqa: BLE001
        LOG.exception("下载清理接口登记失败（不影响其它功能）")


_register_download_cleanup_route()


if __name__ == "__main__":
    # default_concurrency_limit 提为 4：轮询与「暂停 / 继续 / 刷新」等轻量事件不再被长任务堵住，
    # 避免点了没反应（分离任务本身仍是单线程串行，互不影响）。
    # VS_NO_BROWSER=1 时由外部启动器负责「服务就绪后再开浏览器」，避免重复开两个标签页。
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=not os.environ.get("VS_NO_BROWSER"),
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR), str(MODEL_DIR)],
        theme=gr.themes.Soft(),
        css=CSS,
        head=VS_HEAD + VS_JS,
        favicon_path=str(ICON_PNG),
    )
