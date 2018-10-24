#/bin/bash
model_dir="models_$1"
compressed_dir="compressed_$1"
#data_dir="../Parser/chromosomes_N"
data_dir="test"
export CUDA_VISIBLE_DEVICES=$2

mkdir -p $model_dir
mkdir -p $compressed_dir

for f in $data_dir/*.npy 
do
    echo $f
    s=${f##*/}
    basename=${s%.*}
    echo $basename
    model_name="$model_dir/$basename.hdf5"
    echo $model_name
    script_name=$1.py
    echo $script_name
    python $script_name -data $f -epochs 2 -gpu $2 -model $model_name
    
    output_prefix="$compressed_dir/$basename/compressed"
    python ../encoder_decoder/compressor.py -data $f -model $model_name -output $output_prefix  
    echo "- - - - - "
done



