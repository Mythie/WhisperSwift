#!/bin/bash
#
# Build whisper.xcframework for macOS only
# Based on whisper.cpp/build-xcframework.sh
# Modified to work without full Xcode (using Makefiles)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_DIR="$ROOT_DIR/whisper.cpp"

# Options
MACOS_MIN_OS_VERSION=13.3

BUILD_SHARED_LIBS=OFF
WHISPER_BUILD_EXAMPLES=OFF
WHISPER_BUILD_TESTS=OFF
WHISPER_BUILD_SERVER=OFF
GGML_METAL=ON
GGML_METAL_EMBED_LIBRARY=ON
GGML_BLAS_DEFAULT=ON
GGML_METAL_USE_BF16=ON
GGML_OPENMP=OFF

COMMON_C_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"
COMMON_CXX_FLAGS="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"

check_required_tool() {
    local tool=$1
    local install_message=$2

    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not found."
        echo "$install_message"
        exit 1
    fi
}

echo "Checking for required tools..."
check_required_tool "cmake" "Please install CMake 3.28.0 or later (brew install cmake)"
check_required_tool "libtool" "Please install libtool (should be available with Xcode CLT)"

cd "$WHISPER_DIR"

# Clean up previous builds
echo "Cleaning previous builds..."
rm -rf build-apple
rm -rf build-macos
rm -rf build-macos-arm64
rm -rf build-macos-x86_64

# Determine host architecture
HOST_ARCH=$(uname -m)
echo "Host architecture: $HOST_ARCH"

# Build function for a specific architecture
build_arch() {
    local arch=$1
    local build_dir="build-macos-${arch}"
    
    echo ""
    echo "========================================="
    echo "Building for macOS ${arch}..."
    echo "========================================="
    
    # Set arch-specific flags
    local arch_flags="-arch ${arch}"
    if [ "$arch" = "x86_64" ]; then
        arch_flags+=" -target x86_64-apple-macos${MACOS_MIN_OS_VERSION}"
    else
        arch_flags+=" -target arm64-apple-macos${MACOS_MIN_OS_VERSION}"
    fi
    
    cmake -B ${build_dir} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_MIN_OS_VERSION} \
        -DCMAKE_OSX_ARCHITECTURES="${arch}" \
        -DCMAKE_C_FLAGS="${COMMON_C_FLAGS} ${arch_flags}" \
        -DCMAKE_CXX_FLAGS="${COMMON_CXX_FLAGS} ${arch_flags}" \
        -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
        -DWHISPER_BUILD_EXAMPLES=${WHISPER_BUILD_EXAMPLES} \
        -DWHISPER_BUILD_TESTS=${WHISPER_BUILD_TESTS} \
        -DWHISPER_BUILD_SERVER=${WHISPER_BUILD_SERVER} \
        -DGGML_METAL_EMBED_LIBRARY=${GGML_METAL_EMBED_LIBRARY} \
        -DGGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT} \
        -DGGML_METAL=${GGML_METAL} \
        -DGGML_METAL_USE_BF16=${GGML_METAL_USE_BF16} \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=${GGML_OPENMP} \
        -DWHISPER_COREML=ON \
        -DWHISPER_COREML_ALLOW_FALLBACK=ON \
        -S .
    
    cmake --build ${build_dir} --config Release -j$(sysctl -n hw.ncpu)
}

# Build for both architectures
build_arch "arm64"
build_arch "x86_64"

echo ""
echo "========================================="
echo "Creating universal libraries..."
echo "========================================="

# Create output directory
mkdir -p build-macos/libs

# Function to create universal binary
create_universal() {
    local lib_name=$1
    local arm64_path=$2
    local x86_64_path=$3
    local output_path=$4
    
    if [ -f "$arm64_path" ] && [ -f "$x86_64_path" ]; then
        echo "Creating universal binary for ${lib_name}..."
        lipo -create "$arm64_path" "$x86_64_path" -output "$output_path"
    elif [ -f "$arm64_path" ]; then
        echo "Warning: Only arm64 available for ${lib_name}, copying..."
        cp "$arm64_path" "$output_path"
    elif [ -f "$x86_64_path" ]; then
        echo "Warning: Only x86_64 available for ${lib_name}, copying..."
        cp "$x86_64_path" "$output_path"
    else
        echo "Error: No library found for ${lib_name}"
        return 1
    fi
}

# Create universal libraries
create_universal "libwhisper" \
    "build-macos-arm64/src/libwhisper.a" \
    "build-macos-x86_64/src/libwhisper.a" \
    "build-macos/libs/libwhisper.a"

create_universal "libggml" \
    "build-macos-arm64/ggml/src/libggml.a" \
    "build-macos-x86_64/ggml/src/libggml.a" \
    "build-macos/libs/libggml.a"

create_universal "libggml-base" \
    "build-macos-arm64/ggml/src/libggml-base.a" \
    "build-macos-x86_64/ggml/src/libggml-base.a" \
    "build-macos/libs/libggml-base.a"

create_universal "libggml-cpu" \
    "build-macos-arm64/ggml/src/libggml-cpu.a" \
    "build-macos-x86_64/ggml/src/libggml-cpu.a" \
    "build-macos/libs/libggml-cpu.a"

create_universal "libggml-metal" \
    "build-macos-arm64/ggml/src/ggml-metal/libggml-metal.a" \
    "build-macos-x86_64/ggml/src/ggml-metal/libggml-metal.a" \
    "build-macos/libs/libggml-metal.a"

create_universal "libggml-blas" \
    "build-macos-arm64/ggml/src/ggml-blas/libggml-blas.a" \
    "build-macos-x86_64/ggml/src/ggml-blas/libggml-blas.a" \
    "build-macos/libs/libggml-blas.a"

# CoreML library (may only exist for one arch)
if [ -f "build-macos-arm64/src/libwhisper.coreml.a" ] || [ -f "build-macos-x86_64/src/libwhisper.coreml.a" ]; then
    create_universal "libwhisper.coreml" \
        "build-macos-arm64/src/libwhisper.coreml.a" \
        "build-macos-x86_64/src/libwhisper.coreml.a" \
        "build-macos/libs/libwhisper.coreml.a" || true
fi

echo ""
echo "========================================="
echo "Setting up framework structure..."
echo "========================================="

FRAMEWORK_DIR="build-macos/framework/whisper.framework"
mkdir -p "${FRAMEWORK_DIR}/Versions/A/Headers"
mkdir -p "${FRAMEWORK_DIR}/Versions/A/Modules"
mkdir -p "${FRAMEWORK_DIR}/Versions/A/Resources"

# Create symbolic links
ln -sf A "${FRAMEWORK_DIR}/Versions/Current"
ln -sf Versions/Current/Headers "${FRAMEWORK_DIR}/Headers"
ln -sf Versions/Current/Modules "${FRAMEWORK_DIR}/Modules"
ln -sf Versions/Current/Resources "${FRAMEWORK_DIR}/Resources"
ln -sf Versions/Current/whisper "${FRAMEWORK_DIR}/whisper"

# Copy headers
cp include/whisper.h           "${FRAMEWORK_DIR}/Versions/A/Headers/"
cp ggml/include/ggml.h         "${FRAMEWORK_DIR}/Versions/A/Headers/"
cp ggml/include/ggml-alloc.h   "${FRAMEWORK_DIR}/Versions/A/Headers/"
cp ggml/include/ggml-backend.h "${FRAMEWORK_DIR}/Versions/A/Headers/"
cp ggml/include/ggml-metal.h   "${FRAMEWORK_DIR}/Versions/A/Headers/"
cp ggml/include/ggml-cpu.h     "${FRAMEWORK_DIR}/Versions/A/Headers/"
cp ggml/include/ggml-blas.h    "${FRAMEWORK_DIR}/Versions/A/Headers/"
cp ggml/include/gguf.h         "${FRAMEWORK_DIR}/Versions/A/Headers/"

# Create module map
cat > "${FRAMEWORK_DIR}/Versions/A/Modules/module.modulemap" << EOF
framework module whisper {
    header "whisper.h"
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "ggml-metal.h"
    header "ggml-cpu.h"
    header "ggml-blas.h"
    header "gguf.h"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF

# Create Info.plist
cat > "${FRAMEWORK_DIR}/Versions/A/Resources/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>whisper</string>
    <key>CFBundleIdentifier</key>
    <string>org.ggml.whisper</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>whisper</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${MACOS_MIN_OS_VERSION}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
    <key>DTPlatformName</key>
    <string>macosx</string>
    <key>DTSDKName</key>
    <string>macosx${MACOS_MIN_OS_VERSION}</string>
</dict>
</plist>
EOF

echo ""
echo "========================================="
echo "Creating dynamic library..."
echo "========================================="

# Combine all static libraries
LIBS_TO_COMBINE=(
    "build-macos/libs/libwhisper.a"
    "build-macos/libs/libggml.a"
    "build-macos/libs/libggml-base.a"
    "build-macos/libs/libggml-cpu.a"
    "build-macos/libs/libggml-metal.a"
    "build-macos/libs/libggml-blas.a"
)

# Add CoreML if available
if [ -f "build-macos/libs/libwhisper.coreml.a" ]; then
    LIBS_TO_COMBINE+=("build-macos/libs/libwhisper.coreml.a")
fi

mkdir -p build-macos/temp
libtool -static -o "build-macos/temp/combined.a" "${LIBS_TO_COMBINE[@]}" 2>/dev/null

# Create universal dynamic library
clang++ -dynamiclib \
    -arch arm64 -arch x86_64 \
    -mmacosx-version-min=${MACOS_MIN_OS_VERSION} \
    -Wl,-force_load,"build-macos/temp/combined.a" \
    -framework Foundation \
    -framework Metal \
    -framework Accelerate \
    -framework CoreML \
    -install_name "@rpath/whisper.framework/Versions/Current/whisper" \
    -o "${FRAMEWORK_DIR}/Versions/A/whisper"

# Verify the binary
echo "Verifying universal binary..."
lipo -info "${FRAMEWORK_DIR}/Versions/A/whisper"

# Clean up temp
rm -rf build-macos/temp

echo ""
echo "========================================="
echo "Creating XCFramework..."
echo "========================================="

mkdir -p build-apple

# Create xcframework (for macOS only, we don't need xcodebuild)
# We'll create it manually since we only have one platform

XCFRAMEWORK_DIR="build-apple/whisper.xcframework"
rm -rf "$XCFRAMEWORK_DIR"
mkdir -p "$XCFRAMEWORK_DIR/macos-arm64_x86_64"

# Copy framework
cp -R "${FRAMEWORK_DIR}" "$XCFRAMEWORK_DIR/macos-arm64_x86_64/"

# Create Info.plist for xcframework
cat > "$XCFRAMEWORK_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>AvailableLibraries</key>
    <array>
        <dict>
            <key>BinaryPath</key>
            <string>whisper.framework/Versions/A/whisper</string>
            <key>LibraryIdentifier</key>
            <string>macos-arm64_x86_64</string>
            <key>LibraryPath</key>
            <string>whisper.framework</string>
            <key>SupportedArchitectures</key>
            <array>
                <string>arm64</string>
                <string>x86_64</string>
            </array>
            <key>SupportedPlatform</key>
            <string>macos</string>
        </dict>
    </array>
    <key>CFBundlePackageType</key>
    <string>XFWK</string>
    <key>XCFrameworkFormatVersion</key>
    <string>1.0</string>
</dict>
</plist>
EOF

echo ""
echo "========================================="
echo "Build complete!"
echo "========================================="
echo "XCFramework location: $WHISPER_DIR/build-apple/whisper.xcframework"
echo ""
echo "Framework contents:"
ls -la "$XCFRAMEWORK_DIR/macos-arm64_x86_64/whisper.framework/"
echo ""
echo "Binary info:"
file "$XCFRAMEWORK_DIR/macos-arm64_x86_64/whisper.framework/whisper"
