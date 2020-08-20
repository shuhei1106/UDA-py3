python original_data_format.py \
    --sup_data='data/original/v2/raw_data.csv' \
    --unsup_data='data/original/unsup_32m/all_repo_32m.csv' \
    --save_dir='data/dataset/v2' \
    --sup_cont_col='text' \
    --label_col='label' \
    --sup_id_col='facility code,accession number' \
    --smethod_arg='train_ratio' \
    --unsup_cont_col='FINDING,DIAGNOSIS' \
    --unsup_id_col='FACILITY_CODE,ACCESSION_NUMBER'