param(
    [switch]$Piper
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Llama = Join-Path $Root "llama"
$Models = Join-Path $Root "models"
$Venv = Join-Path $Root ".venv"
$Requirements = Join-Path $Root "requirements.txt"

$VoskModel = "vosk-model-small-ru-0.22"
$VoskUrls = @(
    "https://huggingface.co/rhasspy/vosk-models/resolve/main/ru/$VoskModel.zip",
    "https://alphacephei.com/vosk/models/$VoskModel.zip"
)

$LlmFile = "LFM3-1.7B-Q4_K_M.gguf"
if ($env:LLM_FILE) { $LlmFile = $env:LLM_FILE }
$LlmUrl = "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/$LlmFile"
if ($env:LLM_URL) { $LlmUrl = $env:LLM_URL }

$PiperVoice = "ru_RU-irina-medium.onnx"
$PiperUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Не найдена команда '$Name'. Установите её и повторите: $Name"
    }
}

function Download($Url, $Path) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Force $parent | Out-Null }
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -fL -C - --retry 3 --connect-timeout 20 -o $Path $Url
        if ($LASTEXITCODE -eq 0) { return }
        Write-Host "curl не сработал, качаю через PowerShell..."
    }
    Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
}

Require-Command python
Require-Command git
Require-Command cmake

$Arch = switch ($env:PROCESSOR_ARCHITECTURE) {
    "AMD64" { "x86_64" }
    "ARM64" { "arm64" }
    default { throw "Неподдерживаемая архитектура: $env:PROCESSOR_ARCHITECTURE" }
}

Write-Host "Архитектура: $Arch"

if (-not (Test-Path $Venv)) {
    Write-Host "Создаю виртуальное окружение .venv"
    & python -m venv "$Venv"
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Не удалось создать .venv. Проверьте установку Python."
}

if (-not (Test-Path $Requirements)) {
    throw "Не найден $Requirements"
}

Write-Host "Ставлю Python-зависимости..."
& $VenvPython -m pip install --quiet --upgrade pip
& $VenvPython -m pip install --quiet -r $Requirements

$Config = Join-Path $Root "config.json"
if (-not (Test-Path $Config) -and (Test-Path (Join-Path $Root "config.example.json"))) {
    Copy-Item (Join-Path $Root "config.example.json") $Config
    Write-Host "Создан config.json"
}

$ToolsDir = Join-Path $Root "tools"
New-Item -ItemType Directory -Force $ToolsDir | Out-Null
$ToolsFile = Join-Path $ToolsDir "tools.json"
if (-not (Test-Path $ToolsFile) -and (Test-Path (Join-Path $Root "tools.example.json"))) {
    Copy-Item (Join-Path $Root "tools.example.json") $ToolsFile
    Write-Host "Создан tools/tools.json"
}

$VoskPath = Join-Path $Models $VoskModel
if (-not (Test-Path $VoskPath)) {
    New-Item -ItemType Directory -Force $Models | Out-Null
    $zip = Join-Path $Models "$VoskModel.zip"
    $ok = $false
    foreach ($url in $VoskUrls) {
        try {
            Write-Host "Качаю модель Vosk: $url"
            Download $url $zip
            Expand-Archive -Path $zip -DestinationPath $Models -Force
            Remove-Item $zip -Force
            if (Test-Path $VoskPath) { $ok = $true; break }
        } catch {
            Write-Host "Не вышло: $($_.Exception.Message)"
            if (Test-Path $zip) { Remove-Item $zip -Force }
        }
    }
    if (-not $ok) {
        throw "Не удалось скачать $VoskModel. Скачайте вручную: https://alphacephei.com/vosk/models"
    }
}

$Source = Join-Path $Llama "source\llama.cpp"
$Build = Join-Path $Llama "build"
$Output = Join-Path $Llama $Arch
New-Item -ItemType Directory -Force (Join-Path $Llama "source") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Llama "gguf") | Out-Null

if (-not (Test-Path (Join-Path $Source ".git"))) {
    Write-Host "Клонирую llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$Source"
} else {
    git -C $Source pull --ff-only
}

if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }

Write-Host "Собираю llama.cpp под $Arch (это долго)..."
$CmakeArgs = @(
    "-S", $Source,
    "-B", $Build,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$Output",
    "-DCMAKE_INSTALL_BINDIR=bin",
    "-DCMAKE_INSTALL_LIBDIR=lib",
    "-DGGML_NATIVE=ON",
    "-DGGML_OPENMP=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=ON"
)
if ($Arch -eq "arm64") { $CmakeArgs += @("-A", "ARM64") }
cmake @CmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure не удался" }

cmake --build "$Build" --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Сборка llama.cpp не удалась" }

cmake --install "$Build" --config Release
if ($LASTEXITCODE -ne 0) { throw "Установка llama.cpp не удалась" }

Remove-Item -Recurse -Force $Build

$Server = Join-Path $Output "bin\llama-server.exe"
if (-not (Test-Path $Server)) {
    $found = Get-ChildItem -Path $Output -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if ($found) { $Server = $found.FullName }
}
if (-not (Test-Path $Server)) {
    throw "llama-server.exe не найден после сборки"
}

$Server = (Resolve-Path $Server).Path
$LlamaFull = (Resolve-Path $Llama).Path
$Relative = $Server.Substring($LlamaFull.Length + 1).Replace("\", "/")

$ArchJson = @{
    arch      = $Arch
    platform  = "Windows"
    processor = $env:PROCESSOR_ARCHITECTURE
    llama_path = $Arch
    binary    = $Relative
} | ConvertTo-Json

$ArchFile = Join-Path $Llama "arch.json"
[System.IO.File]::WriteAllText($ArchFile, $ArchJson, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "arch.json: $ArchFile"

$LlmPath = Join-Path $Llama "gguf\$LlmFile"
if (-not (Test-Path $LlmPath)) {
    $part = "$LlmPath.part"
    Write-Host "Качаю LLM $LlmFile (~1 ГБ, с докачкой при обрыве)"
    Download $LlmUrl $part

    $stream = [System.IO.File]::OpenRead($part)
    try {
        $buffer = New-Object byte[] 4
        [void]$stream.Read($buffer, 0, 4)
    } finally {
        $stream.Close()
    }
    $magic = [System.Text.Encoding]::ASCII.GetString($buffer)
    if ($magic -ne "GGUF") {
        Remove-Item $part -Force
        throw "Скачанный файл не похож на GGUF. Проверьте $LlmUrl"
    }
    Move-Item $part $LlmPath -Force
}

if ($Piper) {
    Write-Host "Озвучка: piper-tts и русский голос"
    & $VenvPython -m pip install --quiet piper-tts
    $PiperDir = Join-Path $Models "piper"
    New-Item -ItemType Directory -Force $PiperDir | Out-Null
    foreach ($file in @($PiperVoice, "$PiperVoice.json")) {
        $target = Join-Path $PiperDir $file
        if (-not (Test-Path $target)) {
            Download "$PiperUrl/$file" $target
        }
    }
}

Write-Host ""
Write-Host "================================"
Write-Host "СБОРКА ЗАВЕРШЕНА"
Write-Host "Архитектура: $Arch"
Write-Host "llama-server: $Server"
Write-Host "Vosk-модель: $VoskPath"
Write-Host "LLM: $LlmPath"
if ($Piper) {
    $PiperTarget = Join-Path $Models "piper\$PiperVoice"
    Write-Host "Голос: $PiperTarget"
} else {
    Write-Host "Голос: не установлен (build.ps1 -Piper)"
}
Write-Host ""
Write-Host "Дальше:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python main.py"
Write-Host "================================"
