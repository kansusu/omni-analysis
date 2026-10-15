#!/usr/bin/env bash
# Extract a standalone 16kHz mono WAV for each video chunk, so audio can be
# passed to Qwen3-Omni as a separate input instead of via the buggy
# use_audio_in_video interleaved-merge path.
# Usage: ./extract_audio.sh [chunk_dir]
set -euo pipefail

CHUNK_DIR="${1:-chunks}"

for f in "$CHUNK_DIR"/*.mp4; do
  wav="${f%.mp4}.wav"
  ffmpeg -y -i "$f" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$wav"
done

echo "Extracted audio for all chunks in $CHUNK_DIR"
