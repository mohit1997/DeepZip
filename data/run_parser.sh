#bin/bash
data_dir="files_to_be_compressed"
processed_dir="processed_files"


for f in $data_dir/*.txt
do
    echo "filename: "$f
    s=${f##*/}
    basename=${s%.*}
    echo $basename
    
    output_file="$processed_dir/$basename.npy"
    param_file="$processed_dir/$basename.param.json"

    python parse_new.py -input $f -output $output_file -param_file $param_file
    echo "- - - - - "
done



