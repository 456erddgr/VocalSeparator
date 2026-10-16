# 校验 app.py 里 BUILTIN_MODELS / DEREVERB_CHOICES 声明的模型文件
# 都能在 catalog.json 里找到登记项。
#
# 失败意味着：程序界面会列出一个 catalog 里没有的档位，
# 或者有人改了 app.py 却忘了同步 catalog.json。
$ErrorActionPreference = 'Stop'

python scripts/ci_check_models.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
