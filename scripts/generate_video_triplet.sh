#!/bin/bash
set -e

# Save positional args
DIR1="$1"
DIR2="$2"
DIR3="$3"
shift 3

# Parse keyword arguments
for arg in "$@"; do
  case $arg in
    --height=*) HEIGHT="${arg#*=}" ;;
    --framerate=*) FRAMERATE="${arg#*=}" ;;
    --output=*) OUT="${arg#*=}" ;;
    --exts=*) EXTS="${arg#*=}" ;;
    --label1=*) LABEL1="${arg#*=}" ;;
    --label2=*) LABEL2="${arg#*=}" ;;
    --label3=*) LABEL3="${arg#*=}" ;;
    --labelsize=*) LABELSIZE="${arg#*=}" ;;
  esac
done

# Validate required inputs
if [ -z "$DIR1" ] || [ -z "$DIR2" ] || [ -z "$DIR3" ] || \
   [ -z "$HEIGHT" ] || [ -z "$FRAMERATE" ] || [ -z "$OUT" ] || [ -z "$EXTS" ]; then
  echo "Usage: $0 dir1 dir2 dir3 --height=HEIGHT --framerate=FPS --output=FILE --exts=ext1,ext2,ext3 [--label1=TXT] [--label2=TXT] [--label3=TXT] [--labelsize=N]"
  exit 1
fi

IFS=',' read -r EXT1 EXT2 EXT3 <<< "$EXTS"
LABELSIZE="${LABELSIZE:-24}"

ffmpeg \
  -framerate "$FRAMERATE" -pattern_type glob -i "$DIR1/*.${EXT1}" \
  -framerate "$FRAMERATE" -pattern_type glob -i "$DIR2/*.${EXT2}" \
  -framerate "$FRAMERATE" -pattern_type glob -i "$DIR3/*.${EXT3}" \
  -filter_complex "\
    [0:v]crop='if(gte(in_w\,in_h)\,in_h\,in_w):if(gte(in_w\,in_h)\,in_h\,in_w):(in_w-if(gte(in_w\,in_h)\,in_h\,in_w))/2:(in_h-if(gte(in_w\,in_h)\,in_h\,in_w))/2',scale=${HEIGHT}:${HEIGHT},drawtext=text='${LABEL1}':x=10:y=10:fontsize=${LABELSIZE}:fontcolor=black[v0]; \
    [1:v]crop='if(gte(in_w\,in_h)\,in_h\,in_w):if(gte(in_w\,in_h)\,in_h\,in_w):(in_w-if(gte(in_w\,in_h)\,in_h\,in_w))/2:(in_h-if(gte(in_w\,in_h)\,in_h\,in_w))/2',scale=${HEIGHT}:${HEIGHT},drawtext=text='${LABEL2}':x=10:y=10:fontsize=${LABELSIZE}:fontcolor=black[v1]; \
    [2:v]crop='if(gte(in_w\,in_h)\,in_h\,in_w):if(gte(in_w\,in_h)\,in_h\,in_w):(in_w-if(gte(in_w\,in_h)\,in_h\,in_w))/2:(in_h-if(gte(in_w\,in_h)\,in_h\,in_w))/2',scale=${HEIGHT}:${HEIGHT},drawtext=text='${LABEL3}':x=10:y=10:fontsize=${LABELSIZE}:fontcolor=black[v2]; \
    [v0][v1][v2]hstack=3" \
  -shortest \
  -pix_fmt yuv420p \
  -y "$OUT"
