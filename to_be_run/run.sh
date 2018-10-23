#/bin/bash
model_dir=""
data_dir=""
scripts_dir=""

for f in $data_dir/*.npy do
    echo $f
    s=${f##*/}
    basename=${s%.*}
    echo $basename
    model_name="$model_dir/$basename.hdf5"
done



