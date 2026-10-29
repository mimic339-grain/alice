# Starts every backend service this project needs, each in its own window.
#
#   1. VOICEVOX        - TTS engine           (http://127.0.0.1:50021)
#   2. Main LLM        - Qwen3-8B on GPU      (http://127.0.0.1:5000)
#   3. Reflex LLM      - Qwen3-0.6B on CPU    (http://127.0.0.1:5001)  [off by default]
#   4. Vision LLM      - Qwen3-VL-2B on CPU   (http://127.0.0.1:5002)
#
# Then run the app itself with:  .\venv\Scripts\python.exe main.py

$ErrorActionPreference = "Stop"

# RealtimeSTT records in child processes. If main.py did not exit cleanly, those children stay
# alive and hold the microphone open -- the next run then opens a stream that returns a flat,
# silent signal, prints "STT Ready", and never transcribes anything. It looks exactly like dead
# hardware, so clear them out before starting rather than debugging the same ghost twice.
#
# Matched on the command line, NOT on the image name: a blanket kill of python.exe also takes
# down the LLM server and anything else the user happens to be running.
$stale = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'RealtimeSTT|multiprocessing' }
if ($stale) {
    Write-Host "Clearing $($stale.Count) leftover STT process(es) holding the microphone..." -ForegroundColor Yellow
    foreach ($p in $stale) {
        Write-Host "  PID $($p.ProcessId)" -ForegroundColor DarkGray
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

$textgen  = "$env:USERPROFILE\source\repos\textgen\textgen-4.9"
$voicevox = "$env:LOCALAPPDATA\Programs\VOICEVOX\VOICEVOX.exe"

function Test-Port($port) {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $c
}

# --- VOICEVOX ---
if (Test-Port 50021) {
    Write-Host "VOICEVOX already running on 50021" -ForegroundColor DarkGray
} else {
    Write-Host "Starting VOICEVOX..." -ForegroundColor Cyan
    Start-Process -FilePath $voicevox
}

# --- Main LLM (GPU) ---
if (Test-Port 5000) {
    Write-Host "Main LLM already running on 5000" -ForegroundColor DarkGray
} else {
    Write-Host "Starting main LLM (Qwen3-8B, GPU)..." -ForegroundColor Cyan
    Start-Process -FilePath "$textgen\app\portable_env\python.exe" -WorkingDirectory "$textgen\app" -ArgumentList "server.py --portable --api --nowebui --model Qwen3-8B-Q4_K_M.gguf --gpu-layers -1 --api-port 5000"
}

# --- Reflex LLM (CPU) ---
# Off by default: constants.REFLEX_USE_LLM is False because the canned phrase list beat the
# 0.6B model on latency, variety and reliability. Set $StartReflexLLM to $true to A/B it again.
$StartReflexLLM = $false
if ($StartReflexLLM) {
    if (Test-Port 5001) {
        Write-Host "Reflex LLM already running on 5001" -ForegroundColor DarkGray
    } else {
        Write-Host "Starting reflex LLM (Qwen3-0.6B, CPU)..." -ForegroundColor Cyan
        Start-Process -FilePath "$textgen\app\portable_env\python.exe" -WorkingDirectory "$textgen\app" -ArgumentList "server.py --portable --api --nowebui --model Qwen3-0.6B-Q8_0.gguf --gpu-layers 0 --api-port 5001"
    }
}

# --- Vision LLM (CPU) ---
# On CPU because the 6GB card is already full with the chat model. A look takes a few seconds
# this way, which is fine: the AI only looks when it decides to, not every turn.
# On the 24GB machine, drop --gpu-layers 0 to put it on the GPU and make looks near-instant.
$StartVisionLLM = $true
if ($StartVisionLLM) {
    if (Test-Port 5002) {
        Write-Host "Vision LLM already running on 5002" -ForegroundColor DarkGray
    } else {
        Write-Host "Starting vision LLM (Qwen3-VL-2B, CPU)..." -ForegroundColor Cyan
        Start-Process -FilePath "$textgen\app\portable_env\python.exe" -WorkingDirectory "$textgen\app" -ArgumentList "server.py --portable --api --nowebui --model Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --gpu-layers 0 --api-port 5002"
    }
}

Write-Host ""
Write-Host "Waiting for services to come up..." -ForegroundColor Cyan
$ports = @(50021, 5000)
if ($StartReflexLLM) { $ports += 5001 }
if ($StartVisionLLM) { $ports += 5002 }
foreach ($p in $ports) {
    $waited = 0
    while (-not (Test-Port $p) -and $waited -lt 120) {
        Start-Sleep -Seconds 2
        $waited += 2
    }
    if (Test-Port $p) {
        Write-Host "  port $p ready" -ForegroundColor Green
    } else {
        Write-Host "  port $p DID NOT COME UP" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Now run:  .\venv\Scripts\python.exe main.py" -ForegroundColor Yellow
