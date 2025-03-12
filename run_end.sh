#!/bin/bash

# 运行不同轮数的匹配
for rounds in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "Running with $rounds rounds..."
  if MATCH_ROUNDS=$rounds SAVE_PATH_BASE="/home/yuxiang/GPUproject/indices/laion-1M/laion-gpu-r" /home/yuxiang/Alaya/GGidxbuild/build/tests/test_laionend > logs/run_laion_r${rounds}.log 2>&1; then
    echo "Run completed successfully for $rounds rounds"
  else
    echo "ERROR: Run failed for $rounds rounds with exit code $?"
  fi
  sleep 5
done

for rounds in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "Running with $rounds rounds..."
  if MATCH_ROUNDS=$rounds SAVE_PATH_BASE="/home/yuxiang/GPUproject/indices/webvid-2.5M/webvid-gpu-r" /home/yuxiang/Alaya/GGidxbuild/build/tests/test_webend > logs/run_webvid_r${rounds}.log 2>&1; then
    echo "Run completed successfully for $rounds rounds"
  else
    echo "ERROR: Run failed for $rounds rounds with exit code $?"
  fi
  sleep 5
done

for rounds in 0 1 2 3 4 5 6 7 8 9 10; do
  echo "Running with $rounds rounds..."
  if MATCH_ROUNDS=$rounds SAVE_PATH_BASE="/home/yuxiang/GPUproject/indices/t2i-10M/t2i-gpu-r" /home/yuxiang/Alaya/GGidxbuild/build/tests/test_t2iend > logs/run_t2i_r${rounds}.log 2>&1; then
    echo "Run completed successfully for $rounds rounds"
  else
    echo "ERROR: Run failed for $rounds rounds with exit code $?"
  fi
  sleep 5
done

