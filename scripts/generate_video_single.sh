#!/bin/bash
set -e

# Save positional args
DIR1="$1"
shift 1

# Parse keyword arguments
for arg in "$@"; do
  case $arg in
    --height=*) HEIGHT="${arg#*=}" ;;
    --framerate=*) FRAMERATE="${arg#*=}" ;;
    --output=*) OUT="${arg#*=}" ;;
    --ext=*) EXT="${arg#*=}" ;;
    --label=*) LABEL="${arg#*=}" ;;
    --labelsize=*) LABELSIZE="${arg#*=}" ;;
  esac
done

# Validate required inputs
if [ -z "$DIR1" ] || \
   [ -z "$HEIGHT" ] || [ -z "$FRAMERATE" ] || [ -z "$OUT" ] || [ -z "$EXT" ]; then
  echo "Usage: $0 dir1 --height=HEIGHT --framerate=FPS --output=FILE --ext=ext --label=TXT [--labelsize=N]"
  exit 1
fi

LABELSIZE="${LABELSIZE:-24}"

ffmpeg \
  -framerate "$FRAMERATE" -pattern_type glob -i "$DIR1/*.${EXT}" \
  -filter_complex "\
    crop='if(gte(in_w\,in_h)\,in_h\,in_w):if(gte(in_w\,in_h)\,in_h\,in_w):(in_w-if(gte(in_w\,in_h)\,in_h\,in_w))/2:(in_h-if(gte(in_w\,in_h)\,in_h\,in_w))/2',scale=${HEIGHT}:${HEIGHT},drawtext=text='${LABEL}':x=10:y=10:fontsize=${LABELSIZE}:fontcolor=black" \
  -shortest \
  -pix_fmt yuv420p \
  -y "$OUT"
