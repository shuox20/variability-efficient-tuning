
model='roberta-large'
task='sst2'
max_seq_length=128
batch_size=8
lr=3e-4
epochs=20
checkpoint=''
head_layer=23
last_finetune_layer=17
num_finetune_layers=2
seed=20
warmup_steps=0
chosen_token='BOS'

attn_mode="adapter"
attn_option="sequential"
attn_bn=256

ffn_mode="adapter"
ffn_option="sequential"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="bert"
ffn_adapter_scalar="1"
ffn_bn=256 # ffn bottleneck dim

dir_name='ef_finetune'

path=../log
dir=$path/${model}-head_layer=$head_layer-last_finetune_layer=$last_finetune_layer-num_finetune_layers=$num_finetune_layers-lr=$lr-max_len=$max_seq_length-batch=$batch_size-epochs=$epochs-seed=$seed

python3 ../run/adapter_tune.py --model_name_or_path $model --task_name $task --max_length $max_seq_length --per_device_train_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --output_dir $dir --num_finetune_layers $num_finetune_layers --last_finetune_layer $last_finetune_layer --seed $seed --checkpointing_steps 'epoch' --num_warmup_steps $warmup_steps --resume_from_checkpoint 'past_epoch' --attn_mode $attn_mode --attn_option $attn_option --attn_bn $attn_bn --ffn_mode $ffn_mode --ffn_option $ffn_option --ffn_adapter_layernorm_option $ffn_adapter_layernorm_option --ffn_adapter_init_option $ffn_adapter_init_option --ffn_adapter_scalar $ffn_adapter_scalar --ffn_bn $ffn_bn --head_layer $head_layer
