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

# Run with sample inputs
"$EXE" ./data/rm1.jpg ./data/rm2.jpg