#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p build output

# Configure and build (Release by default)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Try common executable names
EXE=""
for cand in optical_flow opt-flow-parallel xcorr_parallel; do
  if [[ -x "build/$cand" ]]; then
    EXE="build/$cand"
    break
  fi
done

if [[ -z "$EXE" ]]; then
  echo "Error: built executable not found in ./build."
  echo "Look for the target name in CMakeLists.txt."
  exit 1
fi

# Usage/help
usage() {
  cat <<EOF
Usage: $0 [OPTIONS] [IMAGE1 IMAGE2]

Builds the project (Release) and runs the built executable on two images.

If IMAGE1 and IMAGE2 are provided they will be used. If only one is
provided the script will prompt for the second. If none are provided the
script will prompt interactively for both paths.

Options:
  -h, --help    Show this help message and exit

Examples:
  $0 ./data/rm1.jpg ./data/rm2.jpg      # run with two images
  $0                                     # interactive prompt
EOF
}

# Parse simple options
if [[ ${#@} -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
fi

# Determine images from args or interactive prompt
IMG1=""
IMG2=""
if [[ ${#@} -ge 2 ]]; then
  IMG1="$1"
  IMG2="$2"
elif [[ ${#@} -eq 1 ]]; then
  IMG1="$1"
  read -r -p "Enter path for second image: " IMG2
else
  read -r -p "Enter path for first image: " IMG1
  read -r -p "Enter path for second image: " IMG2
fi

# Trim surrounding quotes/spaces (simple)
IMG1=$(echo "$IMG1" | sed -e 's/^\s*"\?//;s/"\?\s*$//')
IMG2=$(echo "$IMG2" | sed -e 's/^\s*"\?//;s/"\?\s*$//')

# Validate image files
if [[ ! -f "$IMG1" ]]; then
  echo "Error: image1 not found: $IMG1"
  exit 2
fi
if [[ ! -f "$IMG2" ]]; then
  echo "Error: image2 not found: $IMG2"
  exit 2
fi

echo "Running: $EXE $IMG1 $IMG2"
"$EXE" "$IMG1" "$IMG2"