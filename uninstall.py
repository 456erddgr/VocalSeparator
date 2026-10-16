# -*- coding: utf-8 -*-
"""VocalSeparator 卸载程序（由「应用和功能 / 开始菜单」调用）。"""
import os
import subprocess
import sys
import winreg
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = [
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall\VocalSeparator",
    r"Software\Microsoft\Windows\CurrentVersion\App Paths\VocalSeparator.exe",
    r"Software\VocalSeparator",
]


def main():
    print("正在卸载 VocalSeparator …")
    print("安装目录：" + str(HERE))
    ps = ("$s=New-Object -ComObject WScript.Shell; "
          "$d=[Environment]::GetFolderPath('Desktop'); "
          "Remove-Item ($d+'\\VocalSeparator.lnk') -ErrorAction SilentlyContinue; "
          "$m=Join-Path $env:APPDATA 'Microsoft\\Windows\\Start Menu\\Programs'; "
          "Remove-Item (Join-Path $m 'VocalSeparator.lnk') -ErrorAction SilentlyContinue; "
          "Remove-Item (Join-Path $m '卸载 VocalSeparator.lnk') -ErrorAction SilentlyContinue")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
        print("[OK] 已移除快捷方式")
    except Exception as e:  # noqa: BLE001
        print("[!] 移除快捷方式失败：" + str(e))
    for k in KEYS:
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, k)
            print("[OK] 已删除注册表项 " + k)
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            print("[!] 删除 " + k + " 失败：" + str(e))
    print("")
    print("正在删除安装目录（本窗口稍后自动关闭）…")
    subprocess.Popen('cmd /c "timeout /t 2 >nul & rmdir /s /q ""%s"""' % str(HERE), shell=True)


if __name__ == "__main__":
    main()
    if "--quiet" not in sys.argv:
        try:
            input("按回车关闭…")
        except Exception:
            pass
