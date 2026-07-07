# run_inference.ps1
# Khoi dong toan bo he thong inference: Ollama (UOS LLM) -> Backend (FastAPI) -> Frontend (React)
# Moi service chay trong 1 cua so PowerShell rieng; dong cua so do de dung service tuong ung.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_inference.ps1
#   (hoac double-click run_inference.bat)
#
# Tuy chon:
#   -ClauseSplitMode none|rulebase|uos   (mac dinh: uos)
#   -SkipOllama                          (bo qua buoc khoi dong/kiem tra Ollama)

param(
    [string]$ClauseSplitMode = "uos",
    [switch]$SkipOllama
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[!] Khong tim thay .venv\Scripts\python.exe - dung python tu PATH thay the." -ForegroundColor Yellow
    $VenvPython = "python"
}

# ─── 1. Ollama (UOS LLM) ───────────────────────────────────────────────────────
# Luu y: Ollama tren Windows la app nen (system tray) tu khoi dong server ngay khi
# co lenh CLI dau tien duoc goi ("ollama list", "ollama pull", ...) - khac voi Linux
# noi phai tu chay "ollama serve" o foreground. Vi vay script chi can goi CLI, KHONG
# tu spawn "ollama serve" rieng (tranh xung dot port neu app da tu khoi dong).
if (-not $SkipOllama) {
    Write-Host "=== 1. Ollama (UOS LLM) ===" -ForegroundColor Cyan

    function Test-OllamaUp {
        try {
            Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null
            return $true
        } catch {
            return $false
        }
    }

    if (Test-OllamaUp) {
        Write-Host "Ollama da chay san."
    } else {
        Write-Host "Ollama chua san sang, dang khoi dong (qua 'ollama list')..."
        try { & ollama list 2>$null | Out-Null } catch {}

        $waited = 0
        while (-not (Test-OllamaUp) -and $waited -lt 20) {
            Start-Sleep -Seconds 1
            $waited += 1
        }
        if (Test-OllamaUp) {
            Write-Host "Ollama da san sang."
        } else {
            Write-Host "[!] Ollama van chua phan hoi sau ${waited}s. Kiem tra lai app Ollama, hoac chay thu cong 'ollama serve'." -ForegroundColor Yellow
            Write-Host "    UOS se tu fallback ve cau goc neu khong ket noi duoc Ollama (khong crash)." -ForegroundColor Yellow
        }
    }

    Write-Host "Kiem tra model qwen3:8b..."
    try {
        $models = & ollama list 2>$null
        if ($models -notmatch "qwen3:8b") {
            Write-Host "Chua co qwen3:8b, dang pull (co the mat vai phut, ~5GB)..."
            & ollama pull qwen3:8b
        } else {
            Write-Host "qwen3:8b da san sang."
        }
    } catch {
        Write-Host "[!] Khong the kiem tra/pull model qua CLI 'ollama'. Hay chay thu cong: ollama pull qwen3:8b" -ForegroundColor Yellow
    }
} else {
    Write-Host "=== 1. Ollama: bo qua (-SkipOllama) ===" -ForegroundColor DarkGray
}

# ─── 2. Backend (FastAPI) ──────────────────────────────────────────────────────
Write-Host "=== 2. Backend (FastAPI, port 5000) ===" -ForegroundColor Cyan

$backendCmd = "cd '$RepoRoot'; " +
    "`$env:CLAUSE_SPLIT_MODE = '$ClauseSplitMode'; " +
    "& '$VenvPython' -m uvicorn server.app:app --host 0.0.0.0 --port 5000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Normal

# ─── 3. Frontend (React + Vite) ────────────────────────────────────────────────
Write-Host "=== 3. Frontend (React + Vite, port 5173) ===" -ForegroundColor Cyan

$frontendDir = Join-Path $RepoRoot "frontend"
if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Write-Host "Khong thay frontend/node_modules, chay npm install truoc..."
    Push-Location $frontendDir
    npm install
    Pop-Location
}

$frontendCmd = "cd '$frontendDir'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -WindowStyle Normal

# ─── Tong ket ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Da khoi dong xong. Moi service chay o 1 cua so PowerShell rieng:" -ForegroundColor Green
Write-Host "  - Ollama   : http://localhost:11434"
Write-Host "  - Backend  : http://localhost:5000  (CLAUSE_SPLIT_MODE=$ClauseSplitMode)"
Write-Host "  - Frontend : http://localhost:5173"
Write-Host ""
Write-Host "Mo trinh duyet: http://localhost:5173"
Write-Host "Dong cua so tuong ung de dung tung service."
