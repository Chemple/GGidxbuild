#!/bin/bash

for rounds in 11 12; do
  echo "Running with $rounds rounds..."
  if MATCH_ROUNDS=$rounds SAVE_PATH_BASE="/home/yuxiang/GPUproject/indices/webvid-2.5M/webvid-gpu-r" /home/yuxiang/Alaya/GGidxbuild/build/tests/test_webend > logs/run_webvid_r${rounds}.log 2>&1; then
    echo "Run completed successfully for $rounds rounds"
  else
    echo "ERROR: Run failed for $rounds rounds with exit code $?"
  fi
  sleep 5
done
