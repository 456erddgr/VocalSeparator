@echo off
chcp 65001 >nul
title 停止 VocalSeparator
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$c = Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue;" ^
  "if (-not $c) { Write-Host '当前没有在 7861 端口运行的服务。'; exit 0 }" ^
  "$pids = $c | Select-Object -ExpandProperty OwningProcess -Unique;" ^
  "foreach ($pid in $pids) {" ^
  "  $p = Get-Process -Id $pid -ErrorAction SilentlyContinue;" ^
  "  if (-not $p) { continue }" ^
  "  if ($p.ProcessName -like 'python*') {" ^
  "    $path = $p.Path;" ^
  "    if ($path -like '*VocalSeparator*') { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue; Write-Host ('已停止 VocalSeparator 服务（进程 ' + $pid + '）') }" ^
  "    else { Write-Host ('跳过：进程 ' + $pid + ' 不是本项目环境（' + $path + '）') }" ^
  "  } else { Write-Host ('跳过：进程 ' + $pid + ' 不是 python（' + $p.ProcessName + '）') }" ^
  "}"
echo.
pause >nul
