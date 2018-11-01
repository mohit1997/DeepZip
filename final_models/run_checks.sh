#/bin/bash
model_dir="../data/trained_models"
compressed_dir="../data/compressed"
data_dir="../data/processed_files"
logs_dir="../data/logs_data"
original_dir="../data/files_to_be_compressed"
final_log="../data/final_log.csv"

echo "filename,file_length,gzip,model,model-size,compressed-size,total-size,status" > $final_log

for f in $data_dir/*.npy 
do
    echo $f
    s=${f##*/}
    basename=${s%.*}
    echo $basename
    original_file="$original_dir/$basename.txt"
    gzip -9 -f <$original_file > "test.gz"
    gzip_file_size=$(stat --printf="%s" "test.gz")
    
    
    for m in "biLSTM_16bit" "biGRU" "biGRU_16bit" "LSTM_multi" "FC" "FC_4layer" "LSTM_multi_16bit" "GRU_multi_16bit" "FC_16bit" "FC_4layer_16bit"
    do
         
        model_file="$model_dir/$basename/$m.hdf5"
        output_dir="$compressed_dir/$basename"
        recon_file_name="$output_dir/$m.reconstructed.txt"
        params_file="$data_dir/$basename.param.json"
        original_file="$original_dir/$basename.txt"

        original_file_size=$(stat --printf="%s" $original_file)
        echo $original_file_size

        #################
        compressed_file_size="0"
        echo $compressed_file_size

        if [ -f $recon_file_name ]; then
            compressed_file_size=$(stat --printf="%s" $output_dir/$m.compressed.combined) 
        fi
        echo $compressed_file_size

        ##################
        model_file_size="0"
        echo $model_file_size
        echo $model_file

        if [ -f $model_file ]; then
            model_file_size=$(stat --printf="%s" $model_file)  
        fi
        echo $model_file_size
        
        ###################
        cmp $recon_file_name "$original_dir/$basename.txt"
        status=$?

        ###################
        total_file_size=$(($model_file_size + $compressed_file_size))
        if [ -f $recon_file_name ]; then
            echo "$basename,$original_file_size,$gzip_file_size,$m,$model_file_size,$compressed_file_size,$total_file_size,$status" >> $final_log
        fi


    done
done



