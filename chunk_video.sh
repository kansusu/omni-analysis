#!/usr/bin/env bash
# Split a video into overlapping chunks for Qwen3-Omni processing.
# Usage: ./chunk_video.sh input.mp4 [outdir]
set -euo pipefail

INPUT="$1"
OUTDIR="${2:-chunks}"
CHUNK_LEN=80
STRIDE=75

mkdir -p "$OUTDIR"

DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
DURATION_INT=${DURATION%.*}

i=0
start=0
while [ "$start" -lt "$DURATION_INT" ]; do
  outfile=$(printf "%s/chunk_%03d_start%05d.mp4" "$OUTDIR" "$i" "$start")
  # Stream copy (no re-encode) since source is already H.264/AAC in MP4 and
  # this ffmpeg build doesn't have libx264 compiled in. Cuts snap to the
  # nearest keyframe rather than being frame-exact, which the 5s overlap
  # above already accounts for.
  ffmpeg -y -ss "$start" -i "$INPUT" -t "$CHUNK_LEN" \
    -c copy \
    "$outfile"
  i=$((i+1))
  start=$((start+STRIDE))
done

echo "Created $i chunks in $OUTDIR"
