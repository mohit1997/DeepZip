#/bin/bash
fasta_dir="fasta_files"
data_dir="files_to_be_compressed"


for f in $fasta_dir/*.fa
do
    echo "filename: "$f
    s=${f##*/}
    basename=${s%.*}
    echo $basename
    
    output_file="$data_dir/$basename.txt"
    sed '/>/d' $f | tr -d '\n' | tr '[:lower:]' '[:upper:]' > $output_file
    echo "- - - - - "
done



