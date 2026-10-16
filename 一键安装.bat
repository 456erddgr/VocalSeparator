@echo off
chcp 65001 >nul
title VocalSeparator 一键安装
cd /d "%~dp0"

echo ============================================================
echo   VocalSeparator 人声分离器 - 一键安装
echo ============================================================
echo.
echo   本脚本会在当前目录内创建 venv 并安装全部依赖。
echo   全程不需要管理员权限，不会污染系统 Python。
echo.
echo   前置要求：
echo     - Windows 10 / 11 64 位
echo     - 已安装 Python 3.10 ~ 3.12（推荐 3.12），且在 PATH 中
echo     - 首次安装需要联网（torch / gradio 等依赖）
echo     - 目标目录建议不含中文与空格
echo.
echo   另外：models\ 与 bin\ 里的模型和 FFmpeg 不在仓库内，
echo   请先按 docs\部署与打包.md 放好，否则无法分离。
echo.
pause

echo.
echo [1/5] 检查 Python ...
set "PY="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; assert sys.version_info[:2] in ((3,10),(3,11),(3,12))" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo [错误] 没找到 Python 3.10 ~ 3.12。
  echo        请到 https://www.python.org/downloads/ 安装 3.12，
  echo        安装时务必勾选 "Add python.exe to PATH"，然后重新运行本脚本。
  echo.
  pause
  exit /b 1
)
echo       使用：%PY%

echo.
echo [2/5] 创建虚拟环境 venv ...
if exist "venv\Scripts\python.exe" (
  echo       venv 已存在，跳过创建。
) else (
  %PY% -m venv venv
  if errorlevel 1 (
    echo [错误] 创建 venv 失败。
    pause
    exit /b 1
  )
)

echo.
echo [3/5] 安装 PyTorch CUDA 12.8 版（体积最大，请耐心等）...
set "VS_PIP=venv\Scripts\python.exe -m pip"
%VS_PIP% install --upgrade pip setuptools wheel
%VS_PIP% install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo.
  echo [警告] CUDA 版 torch 安装失败（可能是没显卡或网络问题）。
  echo        正在回退安装 CPU 版 torch —— 程序仍能用，但分离会慢很多。
  echo.
  %VS_PIP% install torch torchaudio torchvision
)

echo.
echo [4/5] 安装其余依赖（清华镜像）...
%VS_PIP% install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
  echo.
  echo [警告] 清华镜像安装失败，改用阿里镜像重试 ...
  %VS_PIP% install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
)

echo.
echo [5/5] 环境自检 ...
venv\Scripts\python.exe -c "import sys,torch,gradio;print('  Python  ',sys.version.split()[0]);print('  torch   ',torch.__version__);print('  cuda    ',torch.cuda.is_available());print('  gradio  ',gradio.__version__)"
if errorlevel 1 (
  echo.
  echo [错误] 依赖自检失败，请把上面的报错复制出来排查。
  pause
  exit /b 1
)

if not exist "bin\ffmpeg.exe" (
  echo.
  echo [提示] 还没放 bin\ffmpeg.exe —— 没有它无法解码音频。
  echo        下载地址见 docs\部署与打包.md
) else (
  echo   ffmpeg   OK
)

if not exist "models\*.ckpt" (
  echo.
  echo [提示] models\ 里还没有模型权重 —— 没有它无法分离。
  echo        下载地址见 docs\部署与打包.md
)

echo.
echo ============================================================
echo   安装完成 ✓
echo.
echo   启动方式：双击 start.bat（浏览器界面 http://127.0.0.1:7861）
echo   桌面窗口：python launcher.py（带启动画面与原生窗口）
echo ============================================================
echo.
pause
