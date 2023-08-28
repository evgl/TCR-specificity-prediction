# TCR-specificity-prediction
Multi-scale Convolutional Attention Networks for TCR and epitope classification

This project is done for educational and research purpose only.
# Example usages

## Finetune an existing model
To load a model after pretraining and finetune it on another dataset, use the `semifrozen_finetuning.py` script. Model will freeze epitope input channel first and the final dense layers last. 

```console
python3 scripts/semifrozen_finetuning.py \
name_of_training_data_files.csv \
name_of_testing_data_files.csv \
path_to_tcr_file.csv \
path_to_epitope_file.smi \
path_to_pretrained_model \
path_to_store_model \
training_name \
path_to_parameter_file \
model_type
```

## Run trained model on data
 The model is pretrained on BindingDB and finetuned using the semifrozen setting, on full TCR sequences and with SMILES encoding of epitopes.

```console
python3 scripts/model_eval.py \
name_of_test_data_file.csv \
path_to_tcr_file.csv \
path_to_epitope_file.smi \
path_to_trained_model_folder \
model_type \
save_name
```

## Data Handling
To generate full sequences of TCRs from CDR3 sequence and V and J segment names, the `cdr3_to_full_seq.py` script can be used. The script relies on the user having downloaded a fasta files containing the Names of V and J segments with their respecive sequences called `V_segment_sequences.fasta` and `J_segment_sequences.fasta`. These can be downloaded from IMGT.org. Header names must be provided to the script to adapt to different format of the input file.

```console
python3 scripts/cdr3_to_full_seq.py \
directoy_with_VJ_segment_fasta_files \
path_to_file_with_input_sequences.csv \
v_seq_header \
j_seq_header \
cdr3_header \
path_to_output_file.csv
```