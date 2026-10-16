# -*- coding: utf-8 -*-
"""VocalSeparator 安装程序 · WebView2 版（从零重构）

为什么换栈：Tkinter 那版无论怎么调都躲不开三个问题 ——
  · 透明色分层窗口导致鼠标命中异常，× 和取消点了没反应
  · 每帧 CPU 合成一张大图，主循环被堵死
  · 高 DPI 下二次放大，字糊
WebView2 里这些都不存在：按钮是真正的 DOM 元素，动画走 GPU 合成，圆角/毛玻璃是 CSS。
"""
import base64
import codecs
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import winreg
from pathlib import Path

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import webview

HERE = Path(__file__).resolve().parent
if getattr(sys, "frozen", False):
    HERE = Path(sys.executable).resolve().parent
AUTOTEST = bool(os.environ.get("VS_AUTOTEST"))
TARGET = Path(os.environ.get("VS_INSTALL_DIR") or
              ((Path(os.environ.get("LOCALAPPDATA") or Path.home())) / "VocalSeparator"))
VER = "1.1.0"
PAYLOAD_NAME = "payload.7z"


def log(m):
    try:
        with open(HERE / "installer_web.log", "a", encoding="utf-8") as f:
            f.write("%s %s\n" % (time.strftime("%H:%M:%S"), m))
    except Exception:
        pass


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def make_lnk(path, target, workdir, desc):
    return ps("$s=New-Object -ComObject WScript.Shell; $l=$s.CreateShortcut('%s'); "
              "$l.TargetPath='%s'; $l.WorkingDirectory='%s'; $l.Description='%s'; $l.Save()"
              % (path, target, workdir, desc)).returncode == 0


def find_payload():
    """定位 payload。

    两种分发形态都要支持：
      1) 单文件  payload.7z                （本地直接分发）
      2) 分卷    payload.7z.001 / .002 ... （GitHub Release 分发，单文件上限 2GB）

    分卷时必须把**第一个卷**交给 7z，7z 会自动按 .001/.002 顺序接续读取
    同目录下的其余卷；传其它卷会读不到数据。
    """
    # 单文件优先
    for c in (HERE / PAYLOAD_NAME, Path.cwd() / PAYLOAD_NAME):
        if c.exists():
            return c
    # 再找分卷：只认 .001 作为入口
    for base in (HERE, Path.cwd()):
        first = base / (PAYLOAD_NAME + ".001")
        if first.exists():
            return first
    # 最后兜底：任意 payload*.7z*（例如用户自行改名），但排出 .002+ 这类非首卷
    cands = [p for p in sorted(HERE.glob("payload*.7z*"))
             if not re.search(r"\.7z\.\d+$", p.name) or p.name.endswith(".7z.001")]
    return cands[0] if cands else None


def describe_payload(p) -> str:
    """给人看的 payload 描述，说明是单文件还是分卷、共几卷。"""
    if p is None:
        return "未找到"
    if re.search(r"\.7z\.\d+$", p.name):
        vols = sorted(p.parent.glob(Path(p.name).name[:-3] + "*"))
        vols = [v for v in vols if re.search(r"\.7z\.\d+$", v.name)]
        total = sum(v.stat().st_size for v in vols)
        return "%s（分卷，共 %d 卷 / %.2f GB）" % (p.name, len(vols), total / 1073741824.0)
    return "%s（单文件 / %.2f GB）" % (p.name, p.stat().st_size / 1073741824.0)


def sevenzip():
    for b in (Path(getattr(sys, "_MEIPASS", HERE)), HERE):
        for n in ("7z.exe", "7za.exe"):
            if (b / n).exists():
                return b / n
    p = Path(r"C:\Program Files\7-Zip\7z.exe")
    return p if p.exists() else None


def _res(rel):
    for b in (Path(getattr(sys, "_MEIPASS", HERE)), HERE, HERE / "ui"):
        f = b / rel
        if f.exists():
            return f
    return None


def prepare_ui():
    """把界面 + 截图解到一个临时目录，用【本地文件】方式加载。

    教训：之前把 8 张 1000x714 的截图全部内联成 data URI，HTML 涨到 4.5MB，
    pywebview 把这么大的字符串交给 WebView2 时直接炸了（窗口一片黑、只剩报错文本）。
    改成落地文件 + file:// 加载，既稳又快。
    """
    base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or ".")
    dst = base / "VocalSeparatorUI"
    if dst.exists():
        try:
            shutil.rmtree(dst)
        except Exception:
            pass
    (dst / "show").mkdir(parents=True, exist_ok=True)
    f = _res("installer.html")
    if f:
        shutil.copy2(f, dst / "installer.html")
    n = 0
    for i in range(1, 13):
        g = _res("show/%d.png" % i)
        if not g:
            break
        shutil.copy2(g, dst / "show" / ("%d.png" % i))
        n += 1
    log("prepare_ui: html=%s shots=%d" % (dst / "installer.html", n))
    return dst / "installer.html"


class Api:
    def __init__(self):
        self._win = None
        self._p = 0.0
        self._msg = "准备中…"
        self._done = False
        self._err = None

    def state(self):
        """界面每 200ms 来拉一次。绝不反向调用 JS —— 之前就是在后台线程里
        疯狂 evaluate_js 把 WebView2 的 UI 线程淹没，导致窗口无响应、× 点不动。"""
        return {"p": self._p, "m": self._msg, "done": self._done, "err": self._err}

    def close_win(self):
        log("close_win called")
        try:
            self._win.destroy()
        except Exception as e:  # noqa: BLE001
            log("destroy fail %r" % (e,))

    def begin(self):
        """界面点「开始安装」才动手 —— 之前一加载就自动装，用户还没来得及操作就被重 I/O 拖死。"""
        if getattr(self, "_started", False):
            return
        self._started = True
        self.push(0.01, "正在准备…")
        threading.Thread(target=self._install, daemon=True).start()

    def launch_app(self):
        p = TARGET / "VocalSeparator.exe"
        if p.exists():
            try:
                subprocess.Popen([str(p)], cwd=str(TARGET))
            except Exception:
                pass

    def uninstall(self):
        threading.Thread(target=self._uni, daemon=True).start()

    def _uni(self):
        try:
            self.push(0.4, "正在卸载…")
            do_uninstall()
            self.push(1.0, "已卸载：快捷方式、注册表项、安装目录都已删除")
        except Exception as e:  # noqa: BLE001
            self.push(0.4, "卸载出错：" + str(e)[:60])

    def push(self, p, msg=None):
        """只改内存里的状态，不碰界面 —— 界面自己来拉。"""
        if p is not None:
            self._p = max(0.0, min(1.0, float(p)))
        if msg:
            self._msg = msg

    def start_install(self):
        threading.Thread(target=self._install, daemon=True).start()

    def _install(self):
        t0 = time.time()
        try:
            log("autotest begin -> TARGET=%s" % TARGET)
            self.push(0.02, "正在释放组件…")
            payload = find_payload()
            log("payload=%s" % describe_payload(payload))
            if payload is None:
                raise RuntimeError(
                    "没找到 %s 或 %s.001 —— 请把安装程序与 payload 文件"
                    "放在同一个文件夹（分卷分发时所有 .001/.002/... 都要下全）"
                    % (PAYLOAD_NAME, PAYLOAD_NAME))
            extract(payload, lambda f: self.push(0.02 + 0.82 * f, "正在释放组件… %d%%" % int(f * 100)))
            log("extract done %.1fs" % (time.time() - t0))
            self.push(0.86, "正在绑定自带 Python…")
            fix_venv()
            log("venv fixed -> %s" % (TARGET / "runtime" / "python311"))
            self.push(0.92, "正在自检依赖…")
            log("deps: " + check_deps())
            log("deps done %.1fs" % (time.time() - t0))
            if AUTOTEST:
                log("autotest: 跳过快捷方式 / 注册表")
            else:
                self.push(0.96, "正在创建快捷方式…")
                shortcuts()
                self.push(0.99, "正在写入注册表…")
                register()
            self.push(1.0, "安装完成 ✓")
            self._done = True
            log("ALL DONE in %.1fs" % (time.time() - t0))
            if AUTOTEST:
                threading.Timer(4.0, self.close_win).start()
        except Exception as e:  # noqa: BLE001
            log("install fail %r" % (e,))
            self._err = str(e)[:70]
            self.push(0.0, "出错：" + str(e)[:70])


def extract(payload, on_pct):
    """解压并上报进度。

    BUG（已修）：原来用 proc.stdout.read(4096) 读进度 —— 管道上的 read(n) 会
    阻塞到「读满 n 个字符或 EOF」。7z 的进度是 \r 覆盖式的、总量很小，
    所以那一行会卡住很久不返回，回调一次都执行不到，进度条于是永远不动。
    现在改为二进制 + os.read(fd, 65536)：有多少读多少，立刻返回。
    """
    sz = sevenzip()
    if sz is None:
        raise RuntimeError("缺少 7z.exe")
    TARGET.mkdir(parents=True, exist_ok=True)
    flags = 0x08000000 | 0x00004000        # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
    proc = subprocess.Popen([str(sz), "x", str(payload), "-o" + str(TARGET), "-y",
                             "-bsp1", "-bse1", "-bso0", "-mmt=2"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=flags)
    fd = proc.stdout.fileno()
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    last = -1
    logged = False
    while True:
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        if not logged:
            logged = True
            log("7z sample: %r" % data[:140])
        buf += dec.decode(data)
        k = buf.rfind("%")
        if k > 0:
            j = k - 1
            while j >= 0 and (buf[j].isdigit() or buf[j] == " "):
                j -= 1
            try:
                pct = int(buf[j + 1:k].strip())
                if pct != last:
                    last = pct
                    on_pct(pct / 100.0)
            except Exception:
                pass
            buf = buf[k + 1:]
        if len(buf) > 8192:
            buf = buf[-1024:]
    if proc.wait() != 0:
        raise RuntimeError("解压失败（7z 返回 %s）" % proc.returncode)


def fix_venv():
    cfg = TARGET / "venv" / "pyvenv.cfg"
    if not cfg.exists():
        return
    py = TARGET / "runtime" / "python311"
    keep = []
    for ln in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        k = ln.split("=", 1)[0].strip().lower()
        if k == "home":
            ln = "home = " + str(py)
        elif k == "executable":
            ln = "executable = " + str(py / "python.exe")
        elif k == "command":
            continue
        keep.append(ln)
    cfg.write_text("\n".join(keep) + "\n", encoding="utf-8")


def check_deps():
    r = subprocess.run([str(TARGET / "venv" / "Scripts" / "python.exe"), "-c",
                        "import torch,gradio;print('torch',torch.__version__)"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(TARGET))
    return (r.stdout or "").strip() or (r.stderr or "").strip()[-160:]


def shortcuts():
    exe = str(TARGET / "VocalSeparator.exe")
    make_lnk(str(Path(os.path.expanduser("~")) / "Desktop" / "VocalSeparator.lnk"), exe, str(TARGET), "VocalSeparator")
    sm = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    sm.mkdir(parents=True, exist_ok=True)
    make_lnk(str(sm / "VocalSeparator.lnk"), exe, str(TARGET), "VocalSeparator")
    make_lnk(str(sm / "卸载 VocalSeparator.lnk"),
             str(TARGET / "runtime" / "python311" / "python.exe"), str(TARGET), "卸载 VocalSeparator")


def register():
    exe = str(TARGET / "VocalSeparator.exe")
    ico = TARGET / "assets" / "vs.ico"
    un = '"%s" "%s"' % (TARGET / "runtime" / "python311" / "python.exe", TARGET / "uninstall.py")
    try:
        total = sum(p.stat().st_size for p in TARGET.rglob("*") if p.is_file())
    except Exception:
        total = 0
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                          r"Software\Microsoft\Windows\CurrentVersion\Uninstall\VocalSeparator") as h:
        for n, v in (("DisplayName", "VocalSeparator 人声分离工作站"), ("DisplayVersion", VER),
                     ("Publisher", "VocalSeparator"), ("InstallLocation", str(TARGET)),
                     ("DisplayIcon", str(ico) if ico.exists() else exe),
                     ("UninstallString", un), ("QuietUninstallString", un + " --quiet")):
            winreg.SetValueEx(h, n, 0, winreg.REG_SZ, v)
        for n in ("NoModify", "NoRepair"):
            winreg.SetValueEx(h, n, 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(h, "EstimatedSize", 0, winreg.REG_DWORD, int(total / 1024) or 1)
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                          r"Software\Microsoft\Windows\CurrentVersion\App Paths\VocalSeparator.exe") as h:
        winreg.SetValueEx(h, "", 0, winreg.REG_SZ, exe)
        winreg.SetValueEx(h, "Path", 0, winreg.REG_SZ, str(TARGET))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\VocalSeparator") as h:
        winreg.SetValueEx(h, "InstallDir", 0, winreg.REG_SZ, str(TARGET))
        winreg.SetValueEx(h, "Version", 0, winreg.REG_SZ, VER)


def do_uninstall():
    ps("$s=New-Object -ComObject WScript.Shell; "
       "$d=[Environment]::GetFolderPath('Desktop'); "
       "Remove-Item ($d+'\\VocalSeparator.lnk') -ErrorAction SilentlyContinue; "
       "$m=Join-Path $env:APPDATA 'Microsoft\\Windows\\Start Menu\\Programs'; "
       "Remove-Item (Join-Path $m 'VocalSeparator.lnk') -ErrorAction SilentlyContinue; "
       "Remove-Item (Join-Path $m '卸载 VocalSeparator.lnk') -ErrorAction SilentlyContinue")
    for k in (r"Software\Microsoft\Windows\CurrentVersion\Uninstall\VocalSeparator",
              r"Software\Microsoft\Windows\CurrentVersion\App Paths\VocalSeparator.exe",
              r"Software\VocalSeparator"):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, k)
        except Exception:
            pass
    if TARGET.exists():
        shutil.rmtree(TARGET, ignore_errors=True)


def main():
    api = Api()
    # frameless=False：保留系统标题栏，它的 × 由 Windows 负责，永远可用（终极兜底）
    ui_file = prepare_ui()
    win = webview.create_window("VocalSeparator 安装程序  v" + VER, url=str(ui_file), js_api=api,
                                width=1120, height=700, min_size=(1020, 660),
                                frameless=False, background_color="#06070d")
    api._win = win
    if os.environ.get("VS_QUICKTEST"):
        def _q():
            time.sleep(8.0)
            log("quicktest: 调用 close_win")
            api.close_win()
        threading.Thread(target=_q, daemon=True).start()
    if AUTOTEST:
        # 测试模式：不依赖任何人点击，等待 WebView 就绪后自动开始
        def _go():
            time.sleep(6.0)
            log("autotest: auto begin")
            api.begin()
        threading.Thread(target=_go, daemon=True).start()
    webview.start(gui="edgechromium", private_mode=False,
                  storage_path=str(HERE / "webview_data"))
    log("window closed")


if __name__ == "__main__":
    main()
