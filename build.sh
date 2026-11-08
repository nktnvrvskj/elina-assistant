#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="$ROOT/llama"
MODELS="$ROOT/models"
VENV="$ROOT/.venv"
REQUIREMENTS="$ROOT/requirements.txt"

PYTHON_BIN="${PYTHON:-python3}"
VOSK_MODEL="vosk-model-small-ru-0.22"
VOSK_URLS=(
    "https://huggingface.co/rhasspy/vosk-models/resolve/main/ru/$VOSK_MODEL.zip"
    "https://alphacephei.com/vosk/models/$VOSK_MODEL.zip"
)

LLM_FILE="${LLM_FILE:-LFM3-1.7B-Q4_K_M.gguf}"
LLM_URL="${LLM_URL:-https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/$LLM_FILE}"

download() {
    # download URL FILE
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 --connect-timeout 20 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 --timeout=20 -O "$2" "$1"
    else
        echo "Не найдена команда: curl (или wget)"
        return 1
    fi
}

download_resume() {
    # download_resume URL FILE — докачивает с места обрыва
    if command -v curl >/dev/null 2>&1; then
        curl -fL -C - --retry 5 --retry-delay 2 --connect-timeout 20 -o "$2" "$1"
    else
        wget -c -q --tries=5 --timeout=20 -O "$2" "$1"
    fi
}

# =============================================================================
#  Архитектура
# =============================================================================
case "$(uname -m)" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    aarch64|arm64)
        ARCH="arm64"
        ;;
    armv7l|armv7*)
        ARCH="armv7"
        ;;
    *)
        echo "Неподдерживаемая архитектура: $(uname -m)"
        exit 1
        ;;
esac

SOURCE="$LLAMA/source/llama.cpp"
BUILD="$LLAMA/build/$ARCH"
OUTPUT="$LLAMA/$ARCH"

echo "Архитектура: $ARCH"

# =============================================================================
#  Проверка зависимостей
# =============================================================================
for cmd in git cmake c++ unzip "$PYTHON_BIN"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Не найдена команда: $cmd"
        exit 1
    fi
done

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo "Не найдена команда: curl (или wget)"
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "Нужен Python 3.9+, найден: $("$PYTHON_BIN" -V 2>&1)"
    exit 1
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "Не найден $REQUIREMENTS"
    exit 1
fi

# =============================================================================
#  Python-окружение и зависимости (vosk, sounddevice)
# =============================================================================
echo
echo "Python: $("$PYTHON_BIN" -V 2>&1)"

if [[ ! -d "$VENV" ]]; then
    echo "Создаю виртуальное окружение .venv"
    if ! "$PYTHON_BIN" -m venv "$VENV"; then
        echo
        echo "Не удалось создать venv. Установите пакет:"
        echo "  Debian/Ubuntu/Raspberry Pi OS: sudo apt install python3-venv"
        exit 1
    fi
fi

STAMP="$VENV/.deps-ok"
if [[ ! -f "$STAMP" || "$REQUIREMENTS" -nt "$STAMP" ]]; then
    echo "Ставлю Python-зависимости…"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip || true
    "$VENV/bin/python" -m pip install --quiet -r "$REQUIREMENTS"
    touch "$STAMP"
else
    echo "Python-зависимости уже установлены"
fi

# =============================================================================
#  Модель распознавания речи (Vosk, ~45 МБ)
# =============================================================================
echo
if [[ -d "$MODELS/$VOSK_MODEL" ]]; then
    echo "Модель Vosk уже на месте: $MODELS/$VOSK_MODEL"
else
    mkdir -p "$MODELS"
    ARCHIVE="$MODELS/$VOSK_MODEL.zip"
    OK=0
    for url in "${VOSK_URLS[@]}"; do
        echo "Качаю модель Vosk: ${url%%/ru/*}"
        if download "$url" "$ARCHIVE"; then
            if unzip -q "$ARCHIVE" -d "$MODELS"; then
                rm -f "$ARCHIVE"
                OK=1
                break
            fi
        fi
        rm -f "$ARCHIVE"
    done
    if [[ "$OK" -ne 1 || ! -d "$MODELS/$VOSK_MODEL" ]]; then
        echo
        echo "Не удалось скачать модель $VOSK_MODEL."
        echo "Скачайте вручную: https://alphacephei.com/vosk/models"
        echo "и распакуйте в $MODELS/"
        exit 1
    fi
    echo "Модель Vosk установлена"
fi

# =============================================================================
#  Сборка llama.cpp
# =============================================================================
echo
mkdir -p "$LLAMA/source" "$LLAMA/gguf"

if [[ ! -d "$SOURCE/.git" ]]; then
    git clone --depth 1 \
        https://github.com/ggml-org/llama.cpp.git \
        "$SOURCE"
else
    git -C "$SOURCE" pull --ff-only
fi

rm -rf "$BUILD"

cmake -S "$SOURCE" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$OUTPUT" \
    -DCMAKE_INSTALL_BINDIR=bin \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON

cmake --build "$BUILD" \
    --config Release \
    --parallel "$(nproc)"

cmake --install "$BUILD"
rm -rf "$BUILD"

cat > "$LLAMA/arch.json" <<EOF
{
    "arch": "$ARCH",
    "platform": "$(uname -s)",
    "processor": "$(uname -m)",
    "llama_path": "$ARCH",
    "binary": "$ARCH/bin/llama-server"
}
EOF

# =============================================================================
#  Модель LLM (LFM3-1.7B, Q4_K_M, ~1 ГБ)
# =============================================================================
echo
if [[ -f "$LLAMA/gguf/$LLM_FILE" ]]; then
    echo "Модель LLM уже на месте: $LLAMA/gguf/$LLM_FILE"
else
    mkdir -p "$LLAMA/gguf"
    PART="$LLAMA/gguf/$LLM_FILE.part"
    echo "Качаю LLM: $LLM_FILE (~1 ГБ, с докачкой при обрыве)"
    if download_resume "$LLM_URL" "$PART" && [[ "$(head -c 4 "$PART" 2>/dev/null)" == "GGUF" ]]; then
        mv "$PART" "$LLAMA/gguf/$LLM_FILE"
        echo "Модель LLM установлена"
    else
        rm -f "$PART"
        echo
        echo "Не удалось скачать модель LLM."
        echo "Скачайте вручную: $LLM_URL"
        echo "и положите в $LLAMA/gguf/"
        exit 1
    fi
fi

echo
echo "================================"
echo "СБОРКА ЗАВЕРШЕНА"
echo "Архитектура: $ARCH"
echo "JSON: $LLAMA/arch.json"
echo "Результат: $OUTPUT"
echo "Vosk-модель: $MODELS/$VOSK_MODEL"
echo "LLM: $LLAMA/gguf/$LLM_FILE"
echo "Python: $VENV"
echo
echo "Дальше:"
echo "  source .venv/bin/activate"
echo "  python test.py"
echo "================================"
