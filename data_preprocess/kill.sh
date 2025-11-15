start_pid=2215541
end_pid=2218950

# 发送信号给指定范围的进程
for pid in $(seq $start_pid $end_pid); do
    kill -9 $pid
done