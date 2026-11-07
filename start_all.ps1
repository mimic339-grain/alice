# Starts every backend service this project needs, each in its own window.
#
#   1. AivisSpeech     - TTS engine           (http://127.0.0.1:10101)
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

$textgen = "$env:USERPROFILE\source\repos\textgen\textgen-4.9"
$aivis   = "$env:LOCALAPPDATA\Programs\AivisSpeech-Engine\Windows-x64\run.exe"

function Test-Port($port) {
    # Ground truth is "can something connect to it", not "does a listener enumerate".
    #
    # Get-NetTCPConnection -State Listen does not see the AivisSpeech engine. Its loopback
    # listening sockets come back with an EMPTY State field, so the -State Listen filter drops
    # them and the port reads as closed while the engine happily answers GET /version with
    # 200 "1.2.0". netstat -ano shows the same thing: ESTABLISHED rows for 10101, no LISTENING
    # row. The LLM servers on 5000/5002 do report Listen normally, so this looks like a
    # per-process quirk rather than anything configurable.
    #
    # That false negative is not harmless. It made the stale-process cleanup below kill an
    # engine that had been serving for three hours, and then made the wait loop print
    # "DID NOT COME UP" for the healthy replacement. A connect attempt cannot be fooled this
    # way, and it is what the callers actually care about.
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connect = $client.ConnectAsync("127.0.0.1", $port)
        if ($connect.Wait(500) -and $client.Connected) { return $true }
        return $false
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

# --- AivisSpeech ---
#
# The engine's executable is called run.exe, and so is VOICEVOX's, so it is matched on the
# full command line. Never match on the image name: killing every run.exe would take out any
# other engine the user has open.
#
# A previous engine that did not shut down cleanly keeps the port bound without serving, and
# the new one then exits immediately with nothing on screen to say why. If nothing is
# listening on 10101 but an engine process is alive, it is that case -- clear it first. This
# is the same failure that cost an afternoon with VOICEVOX.
if (-not (Test-Port 10101)) {
    $orphans = Get-CimInstance Win32_Process -Filter "Name = 'run.exe'" |
        Where-Object { $_.CommandLine -match 'AivisSpeech' }
    if ($orphans) {
        Write-Host "Clearing $($orphans.Count) stale AivisSpeech engine process(es)..." -ForegroundColor Yellow
        foreach ($p in $orphans) {
            Write-Host "  PID $($p.ProcessId)" -ForegroundColor DarkGray
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Seconds 2
    }
}

if (Test-Port 10101) {
    Write-Host "AivisSpeech already running on 10101" -ForegroundColor DarkGray
} elseif (-not (Test-Path $aivis)) {
    Write-Host "AivisSpeech Engine not found at $aivis" -ForegroundColor Red
    Write-Host "  Get it from https://github.com/Aivis-Project/AivisSpeech-Engine/releases" -ForegroundColor DarkGray
    Write-Host "  and put voice models (.aivmx) in $env:APPDATA\AivisSpeech-Engine\Models" -ForegroundColor DarkGray
} else {
    Write-Host "Starting AivisSpeech Engine..." -ForegroundColor Cyan
    Start-Process -FilePath $aivis -ArgumentList "--host 127.0.0.1 --port 10101" `
                  -WorkingDirectory (Split-Path $aivis)
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
$ports = @(10101, 5000)
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
