
model='roberta-large'
task='sst2'
max_seq_length=128
batch_size=8
lr=3e-4
epochs=20
checkpoint=''
head_layer=23
last_finetune_layer=23
num_finetune_layers=24
seed=20
warmup_steps=0
chosen_token='BOS'
dir_name='finetune'
named_epochs=20


path=../log/$dir_name/$task
dir=$path/${model}-head_layer=$head_layer-last_finetune_layer=$last_finetune_layer-num_finetune_layers=$num_finetune_layers-lr=$lr-max_seq_length=$max_seq_length-batch_size=$batch_size-epochs=$named_epochs-seed=$seed-warmup_steps=$warmup_steps-chosen_token=$chosen_token

python3 ../nc/run/finetune.py --model_name_or_path $model --task_name $task --max_length $max_seq_length --per_device_train_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --output_dir $dir --num_finetune_layers $num_finetune_layers --last_finetune_layer $last_finetune_layer --seed $seed --checkpointing_steps 'epoch' --num_warmup_steps $warmup_steps --chosen_token $chosen_token --resume_from_checkpoint 'past_epoch' --save_best --head_layer $head_layer

