#/bin/bash
model_dir="../data/trained_models"
compressed_dir="../data/compressed"
data_dir="../data/processed_files"
logs_dir="../data/logs_data"
original_dir="../data/files_to_be_compressed"
export CUDA_VISIBLE_DEVICES=$2

pip install tqdm
mkdir -p $model_dir
mkdir -p $compressed_dir

for f in $data_dir/*.npy 
do
    echo $f
    s=${f##*/}
    basename=${s%.*}
    echo $basename

    # Storing the model
    mkdir -p "$model_dir/$basename"
    model_file="$model_dir/$basename/$1.hdf5"
    log_file="$logs_dir/$basename/$1.log.csv"
    output_dir="$compressed_dir/$basename"
    recon_file_name="$output_dir/$1.reconstructed.txt"
    params_file="$data_dir/$basename.param.json"
    echo $params_file
    output_prefix="$output_dir/$1.compressed"
    mkdir -p "$output_dir"
    cmp $recon_file_name "$original_dir/$basename.txt"
    status=$?
    
    echo $status
    
    if cmp $recon_file_name "$original_dir/$basename.txt"; then
        continue 
    else
        echo "continuing"
    fi

    mkdir -p "$model_dir/$basename"
    mkdir -p "$logs_dir/$basename" 
    echo "Starting training ..." | tee -a $log_file
    
    python trainer.py -model_name $1 -d $f -gpu $2 -name $model_file -log_file $log_file 
    
    
    # Perform Compression
    echo "Starting Compression ..." | tee -a $log_file
    /usr/bin/time -v python compressor.py -data $f -data_params $params_file -model $model_file -model_name $1 -output $output_prefix -batch_size 1000 2>&1 | tee -a $log_file 
    /usr/bin/time -v python decompressor.py -output $recon_file_name -model $model_file -model_name $1 -input_file_prefix $output_prefix -batch_size 1000 2>&1 | tee -a $log_file
    cmp $recon_file_name "$original_dir/$basename.txt" >> $log_file
    #echo "- - - - - "
done



