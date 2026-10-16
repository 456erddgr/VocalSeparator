# -*- coding: utf-8 -*-
"""VocalSeparator 桌面启动器（打包为 VocalSeparator.exe，图标 assets/vs.ico）。

双击图标 = 启动本地分离服务 -> 等服务真正就绪 -> 用一个「原生应用窗口」打开界面。
不弹黑框、不进浏览器标签页、不需要用户敲任何命令。

逻辑要点：
1. 先探测 http://127.0.0.1:7861 是否已经在跑：在跑就直接开窗口（不会重复启动服务）。
2. 没在跑则用 venv 里的 python 启动 app.py（无黑框），并显示一个启动画面。
3. 轮询直到服务返回 200（首次加载较慢，最多等 5 分钟），随后关闭启动画面并开窗口。
4. 开窗口优先用 Edge / Chrome 的「应用模式」(--app=)，外观就是一个独立程序窗口，
   没有地址栏和标签页；两个浏览器都没有时回退到系统默认浏览器。
5. 启动失败弹 Windows 提示框并写明日志路径，不静默失败。
"""
from __future__ import annotations

import ctypes
import datetime
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 7861
URL = f"http://{HOST}:{PORT}/"
APP_NAME = "VocalSeparator"
WAIT_SECONDS = 300

CREATE_NO_WINDOW = 0x08000000


def _resolve_app_dir() -> Path:
    """定位应用目录：优先启动器所在目录，其次环境变量 / 常见安装位置。"""
    cands = []
    if getattr(sys, "frozen", False):
        cands.append(Path(sys.executable).resolve().parent)
    cands.append(Path(__file__).resolve().parent)
    env = os.environ.get("VS_APP_DIR")
    if env:
        cands.append(Path(env))
    cands.append(Path(r"E:\VocalSeparator"))
    for c in cands:
        if (c / "app.py").is_file():
            return c
    return cands[0]


APP_DIR = _resolve_app_dir()
LOG_DIR = APP_DIR / "logs"
PYTHON = APP_DIR / "venv" / "Scripts" / "python.exe"
LOG_FILE = LOG_DIR / "launcher.log"
# 独立的浏览器配置目录：窗口尺寸/位置独立记忆，也不干扰用户自己的浏览器
PROFILE_DIR = Path(os.environ.get("LOCALAPPDATA", str(APP_DIR))) / "VocalSeparator" / "webview"


def log(msg: str) -> None:
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def alert(msg: str, *, error: bool = False) -> None:
    log(("错误：" if error else "") + msg)
    try:
        ctypes.windll.user32.MessageBoxW(None, msg, APP_NAME, 0x10 if error else 0x40)
    except Exception:  # noqa: BLE001
        print(msg)


def server_up(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 400
    except Exception:  # noqa: BLE001
        return False


def start_server() -> None:
    if not PYTHON.is_file():
        alert(f"没有找到运行环境：\n{PYTHON}\n\n请先双击目录里的「一键安装.bat」完成安装。", error=True)
        sys.exit(1)
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VS_NO_BROWSER": "1",  # 由本启动器在服务就绪后统一开窗口
    })
    env["PATH"] = os.pathsep.join(
        [str(APP_DIR / "bin"), str(APP_DIR / "venv" / "Lib" / "site-packages" / "torch" / "lib"),
         env.get("PATH", "")])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(LOG_DIR / "server.out.log", "a", encoding="utf-8", errors="replace")
    out.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} 由启动器拉起 =====\n")
    subprocess.Popen(
        [str(PYTHON), str(APP_DIR / "app.py")],
        cwd=str(APP_DIR),
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    log(f"已拉起 app.py（{PYTHON}）")


# ---------------------------------------------------------------------------
# 启动画面（Tkinter 自带，不引入额外依赖）
# ---------------------------------------------------------------------------
def make_splash():
    """返回 (root, set_text) 或 (None, None)：创建失败时静默降级。"""
    try:
        import tkinter as tk
    except Exception:  # noqa: BLE001
        return None, None
    try:
        root = tk.Tk()
        root.title(APP_NAME)
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        w, h = 460, 150
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")
        root.configure(bg="#141821")
        tk.Label(root, text=APP_NAME, bg="#141821", fg="#f2f5ff",
                 font=("Microsoft YaHei UI", 20, "bold")).pack(pady=(26, 4))
        msg = tk.Label(root, text="正在启动本地分离服务…", bg="#141821", fg="#9fb0d0",
                       font=("Microsoft YaHei UI", 10))
        msg.pack()
        tip = tk.Label(root, text="首次启动需要加载环境，约 20~40 秒，请稍候",
                       bg="#141821", fg="#5f6d85", font=("Microsoft YaHei UI", 9))
        tip.pack(pady=(10, 0))
        root.update()

        def set_text(text: str) -> None:
            try:
                msg.config(text=text)
                root.update()
            except Exception:  # noqa: BLE001
                pass

        return root, set_text
    except Exception:  # noqa: BLE001
        return None, None


# ---------------------------------------------------------------------------
# 用「应用模式」开窗口，外观等同原生程序
# ---------------------------------------------------------------------------
def browser_app_cmd() -> list | None:
    """找一个支持 --app= 的 Chromium 浏览器（Chrome 优先，其次 Edge）。"""
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    cands = [
        Path(pf) / "Google/Chrome/Application/chrome.exe",
        Path(pf86) / "Google/Chrome/Application/chrome.exe",
        Path(local) / "Google/Chrome/Application/chrome.exe",
        Path(pf86) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for c in cands:
        if c.is_file():
            return [str(c), f"--app={URL}", "--window-size=1500,1000",
                    f"--user-data-dir={PROFILE_DIR}", "--no-first-run",
                    "--no-default-browser-check", "--disable-features=Translate"]
    return None



# ---------------------------------------------------------------------------
# 原生窗口的「保存文件」接口
#   问题：WebView2 默认不处理下载，Gradio 的下载链接在原生窗口里点了没反应
#         （浏览器里正常）。而分离出来的文件本来就在本机，直接把源文件复制到
#         下载目录最快也最稳，不用走网络。
# ---------------------------------------------------------------------------
def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf, i = path.stem, path.suffix, 1
    while True:
        c = path.with_name("%s (%d)%s" % (stem, i, suf))
        if not c.exists():
            return c
        i += 1


class JsApi:
    def save_file(self, href, name=None):
        """把 /gradio_api/file=<本地路径> 复制到 下载 目录。"""
        try:
            from urllib.parse import unquote, urlparse
            h = str(href or "")
            p = None
            if "gradio_api/file=" in h:
                p = h.split("gradio_api/file=", 1)[1]
            elif h.startswith("file:"):
                p = urlparse(h).path
            if not p:
                return {"ok": False, "err": "不认识的下载地址"}
            p = unquote(p.split("?")[0].split("#")[0])
            src = Path(p)
            if not src.is_file():
                return {"ok": False, "err": "文件不存在：" + str(src)}
            dst = _unique(Path(os.path.expanduser("~")) / "Downloads" / (name or src.name))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            log("saved -> " + str(dst))
            return {"ok": True, "path": str(dst)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "err": str(e)[:140]}

    def reveal(self, path):
        """在资源管理器里定位文件。"""
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
            return {"ok": True}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "err": str(e)[:140]}

def native_window() -> bool:
    """真正的原生窗口：pywebview + WebView2（Edge Chromium 内核）。

    这是首选方式 —— 窗口就是程序本身：没有地址栏、没有标签页、没有浏览器菜单，
    任务栏里是独立应用图标。只有它不可用（缺 WebView2 运行时等）才回退。
    函数会阻塞到窗口关闭为止（和普通桌面程序一致）。
    """
    try:
        import webview  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        log(f"pywebview 不可用，改用其它方式：{e}")
        return False
    try:
        store = Path(APP_DIR) / "webview_data"
        store.mkdir(parents=True, exist_ok=True)
        webview.create_window(APP_NAME, URL, width=1500, height=1000, min_size=(980, 640),
                              js_api=JsApi())
        log("已用原生窗口打开（pywebview / WebView2）")
        webview.start(gui="edgechromium", private_mode=False, storage_path=str(store))
        log("窗口已关闭")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"原生窗口打开失败，改用应用模式：{e}")
        return False


def open_window() -> None:
    if native_window():
        return
    cmd = browser_app_cmd()
    try:
        if cmd:
            # 已经在运行同配置的窗口时，再点一次只是把新窗口带出来，不会重复起服务
            subprocess.Popen(cmd, creationflags=CREATE_NO_WINDOW)
            log("已用应用模式打开程序窗口")
            return
    except Exception as e:  # noqa: BLE001
        log(f"应用模式打开失败：{e}")
    log("回退到系统默认浏览器")
    webbrowser.open(URL)


def main() -> int:
    log(f"启动器启动；应用目录={APP_DIR}")
    open_browser = not os.environ.get("VS_NO_BROWSER")
    if server_up():
        log("检测到服务已在运行，直接打开程序窗口")
        if open_browser:
            open_window()
        return 0

    splash, set_text = (None, None)
    if open_browser:
        splash, set_text = make_splash()

    start_server()
    deadline = time.time() + WAIT_SECONDS
    ready = False
    while time.time() < deadline:
        if server_up():
            ready = True
            break
        if set_text:
            set_text("正在启动本地分离服务… 已等待 %d 秒" % int(time.time() - (deadline - WAIT_SECONDS)))
        time.sleep(1.0)

    if splash is not None:
        try:
            splash.destroy()
        except Exception:  # noqa: BLE001
            pass

    if not ready:
        alert("服务启动超时（5 分钟）。\n\n请查看日志：\n"
              f"{LOG_DIR / 'server.out.log'}\n\n常见原因：显存被其它程序占用、模型文件缺失。", error=True)
        return 1

    log("服务已就绪，打开程序窗口")
    if open_browser:
        open_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())