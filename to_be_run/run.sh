#/bin/bash
model_dir="models_$1"
data_dir="../Parser/chromosomes_N"
export CUDA_VISIBLE_DEVICES=$2

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
    python $script_name -data $f -epochs 10 -gpu $2 -model $model_name
    echo "- - - - - "
done



