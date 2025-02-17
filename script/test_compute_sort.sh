#!/bin/bash

# 设置可执行文件路径
executable="../build/tests/test_compute_sort_neighbor"

# 用于统计返回值的数组
declare -A return_codes

# 执行100次
for i in {1..100}
do
    # 执行程序并获取返回值
    $executable
    return_value=$?

    # 统计返回值的次数
    ((return_codes[$return_value]++))
done

# 输出每个返回值出现的次数
echo "Return code statistics:"
for return_value in "${!return_codes[@]}"
do
    echo "Return code $return_value: ${return_codes[$return_value]} times"
done
