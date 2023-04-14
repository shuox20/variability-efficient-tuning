model='roberta-large'
task='sst2'
max_seq_length=128
batch_size=8
lr=1e-3
epochs=0
checkpoint=''
layer_index=2 #layer to compute metric
seed=20
warmup_steps=10000
chosen_token='AVG' #choose from AVG, BOS, EOS
checkpoint_path=''
label=''

path=../log/${model}/glue_nc
dir=$path/${model}-task=$task-layer=$layer_index-max_seq_length=$max_seq_length-chosen_token=$chosen_token
python3 ../nc/run/nc_compute.py --model_name_or_path $model --task_name $task --max_length $max_seq_length --per_device_train_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --output_dir $dir --last_layer $layer_index --seed $seed --checkpointing_steps '' --num_warmup_steps $warmup_steps --chosen_token $chosen_token 
