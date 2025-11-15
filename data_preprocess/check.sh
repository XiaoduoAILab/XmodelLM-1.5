folder_path="/data1/redpajama-v2-process/data_lsh_within"

for file in "$folder_path"/*.txt; do
    echo "File: $file"
    head -1 "$file"
    tail -3 "$file"
    echo "------------------------"
done