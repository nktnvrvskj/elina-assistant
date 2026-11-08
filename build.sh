#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="$ROOT/llama"

# Определение архитектуры
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

# Проверка зависимостей
for cmd in git cmake c++; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Не найдена команда: $cmd"
        exit 1
    fi
done

mkdir -p "$LLAMA/source"

# Скачивание исходников
if [[ ! -d "$SOURCE/.git" ]]; then
    git clone --depth 1 \
        https://github.com/ggml-org/llama.cpp.git \
        "$SOURCE"
else
    git -C "$SOURCE" pull --ff-only
fi

# Настройка сборки
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

# Компиляция
cmake --build "$BUILD" \
    --config Release \
    --parallel "$(nproc)"

# Установка
cmake --install "$BUILD"
rm -rf "$BUILD"
# Создание JSON для Python-скрипта
cat > "$LLAMA/arch.json" <<EOF
{
    "arch": "$ARCH",
    "platform": "$(uname -s)",
    "processor": "$(uname -m)",
    "llama_path": "$ARCH",
    "binary": "$ARCH/bin/llama-server"
}
EOF

echo
echo "================================"
echo "СБОРКА ЗАВЕРШЕНА"
echo "Архитектура: $ARCH"
echo "JSON: $LLAMA/arch.json"
echo "Результат: $OUTPUT"
echo "================================"
