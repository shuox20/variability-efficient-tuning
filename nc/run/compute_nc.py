# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
import shutil
import torch

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import Repository
#from GPUtil import showUtilization as gpu_usage
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version
#from src.transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_false",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=None, 
        help="Change the number of hidden layers",
    )
    parser.add_argument(
        "--chosen_token",
        type=str,
        default='AVG',
        help="Which token to output to classifier",
    )
    parser.add_argument(
        "--save_hidden_output",
        action="store_false",
        help="Whether to save hidden states locally",
    )
    parser.add_argument(
        "--delete_hidden_output",
        action="store_false",
        help="Whether to delete hidden states after computing",
    )
    parser.add_argument(
        "--finetune_path",
        type=str,
        default='',
        help='path of finetune model',
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if os.path.exists(os.path.join(args.output_dir, 'nc_results.json')):
        return 0
    
    # Set output dir for log file
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, f'log.txt'), mode='a')
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d) : %(message)s'
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(accelerator.state)
    logger.info(args)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name, cache_dir="../cache/")
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    #config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name, output_hidden_states=True, cache_dir="../cache/")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir="../cache/")
    if args.num_hidden_layers is not None:
        config.num_hidden_layers=args.num_hidden_layers
    logger.info(config)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir="../cache/"
    )

    # load pretrained fine-tune model
    if args.finetune_path!='':
        finetune_model = torch.load(f'{args.finetune_path}/pytorch_model.bin', map_location='cpu')
        part_finetune_model = {k:v for k,v in finetune_model.items() if k in model.state_dict()}
        model.load_state_dict(part_finetune_model)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    dataset_category = {}
    for i in train_dataset:
        if i['labels'] not in dataset_category:
            dataset_category[i['labels']]=[i]
        else:
            dataset_category[i['labels']].append(i)
    logger.info('dataset seperate into', dataset_category.keys())
    del train_dataset
    torch.cuda.empty_cache()
    #eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

        # Log a few random samples from the training set:
    '''
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    '''

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    '''
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    '''
    dataloader_category = {}
    for k,v in dataset_category.items():
        logger.info(f'category {k} has {len(v)} samples.')
        dataloader_category[k]=DataLoader(v, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    logger.info(f'dataloader seperate into {dataloader_category.keys()}')
    
    
    #eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    '''
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # only train classification head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    '''

    logger.info("***** Running training *****")
    '''
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    '''
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            logger.info(f" Resumed from checkpoint:{args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            '''
            model_new = torch.load(f'{args.resume_from_checkpoint}/pytorch_model.bin',map_location='cpu')
            part_model = {k:v for k,v in model_new.items() if k in model.state_dict()}
            model.load_state_dict(part_model)
            '''
            #print(model_new)
            resume_step = None
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        '''
        if "epoch" in path:
            args.num_train_epochs -= int(path.replace("epoch_", ""))
        else:
            resume_step = int(path.replace("step_", ""))
            args.num_train_epochs -= resume_step // len(train_dataloader)
            resume_step = (args.num_train_epochs * len(train_dataloader)) - resume_step
        '''

    best_metric = -1
    best_epoch = -1
    model = accelerator.prepare(model)
    for k in dataloader_category:
        dataloader_category[k] = accelerator.prepare(dataloader_category[k])

    model.eval()
    
    vector_category = {}
    for k,v in dataloader_category.items():
        if os.path.exists(f'{args.output_dir}/category={k}_{args.num_hidden_layers}_{args.chosen_token}.pt'):
            #print(f'result from {k} is already logged')
            #logger.info(f'result from {k} is already logged')
            print(f'resume result in {k} from {args.output_dir}/category={k}_{args.num_hidden_layers}_{args.chosen_token}.pt')
            logger.info(f'resume result in {k} from {args.output_dir}/category={k}_{args.num_hidden_layers}_{args.chosen_token}.pt')
            vector_category[k]=torch.load(f'{args.output_dir}/category={k}_{args.num_hidden_layers}_{args.chosen_token}.pt', map_location=None)
            continue
        i=0
        for batch in v:
            with torch.no_grad():
                state = model(**batch).hidden_states[args.num_hidden_layers]
            input_ids = batch['input_ids']
            
            if args.chosen_token=='BOS':
                token = state[:,0,:].to('cpu')
            if args.chosen_token=='EOS':
                assert torch.sum(input_ids==2)==input_ids.shape[0], "some sentences have more than one EOS tokens. EOS method is not applied to this task."
                token = state[input_ids==2,:].to('cpu')
            if args.chosen_token=='AVG':
                non_special_token = (input_ids>2).unsqueeze(-1).expand(input_ids.shape[0], input_ids.shape[1], state.shape[-1])
                token = (torch.sum(state*non_special_token, 1)/torch.sum(non_special_token, 1)).to('cpu')
                del non_special_token
     
            if i==0:
                vector_category[k] = token.to('cpu')
                i=1
            else:
                vector_category[k] = torch.concat([vector_category[k], token],0).to('cpu')
            del token
            del state
            del input_ids
            torch.cuda.empty_cache()
        if args.save_hidden_output:
            torch.save(vector_category[k], f'{args.output_dir}/category={k}_{args.num_hidden_layers}_{args.chosen_token}.pt')
            logger.info(f'save vectors in category {k}')


    def compute_nc(vectors):
        m_all = torch.concat([torch.mean(vector, 0, keepdims=True).transpose(0,1) for vector in vectors.values()],1)
        weights = torch.tensor([vector.shape[0] for vector in vectors.values()])
        B = torch.cov(m_all, correction=0, fweights = weights)
        W = torch.zeros(B.shape)
        for v in vectors.values():
            W = W + torch.cov(v.transpose(0,1), correction=0)*v.shape[0]
        W = W/torch.sum(weights)

        nc = 1/len(vectors)*torch.trace(torch.matmul(W, torch.linalg.pinv(B)))
        return nc

    def nc_new(vectors): 
        m_all = torch.concat([torch.mean(vector, 0, keepdims=True).transpose(0,1) for vector in vectors.values()], 1)
        B_2 = torch.cov(m_all, correction=0)
        m_global = torch.mean(torch.concat([v for v in vectors.values()], 0), 0, keepdims=True).transpose(0,1)
        B_1 = torch.mm(m_all - m_global, (m_all - m_global).transpose(0,1))/len(vectors)
        weights = torch.tensor([vector.shape[0] for vector in vectors.values()])
        W_2 = torch.zeros(B_2.shape)
        W_1 = torch.zeros(B_2.shape)
        for v in vectors.values():
            W_2 = W_2 + torch.cov(v.transpose(0,1), correction=0)
            W_1 = W_1 + torch.cov(v.transpose(0,1), correction=0)*v.shape[0]
        W_2 = W_2/len(vectors)
        W_1 = W_1/torch.sum(weights)
        B_3 = torch.cov(m_all, correction=0, fweights = weights)

        nc_balanced = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_1)))
        nc_imbalanced = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_2)))
        nc_3 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_2)))
        nc_4 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_3)))
        nc_5 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_1)))
        nc_6 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_3)))
        #old_nc == nc_4
        return([nc_balanced,nc_imbalanced, nc_3, nc_4, nc_5, nc_6])

    def compute_nc_new(vectors):
        m_all = torch.concat([torch.mean(vector, 0, keepdims=True).transpose(0,1) for vector in vectors.values()], 1)
        B_2 = torch.cov(m_all, correction=0)
        W_2 = torch.zeros(B_2.shape)
        for v in vectors.values():
            W_2 = W_2 + torch.cov(v.transpose(0,1), correction=0)
        W_2 = W_2/len(vectors)
        weights = torch.tensor([vector.shape[0] for vector in vectors.values()])

    def nc_new1(vectors): 
        m_all = torch.concat([torch.mean(vector, 0, keepdims=True).transpose(0,1) for vector in vectors.values()], 1)
        B_2 = torch.cov(m_all, correction=0)
        m_global = torch.mean(torch.concat([v for v in vectors.values()], 0), 0, keepdims=True).transpose(0,1)
        B_1 = torch.mm(m_all - m_global, (m_all - m_global).transpose(0,1))/len(vectors)
        weights = torch.tensor([vector.shape[0] for vector in vectors.values()])
        W_2 = torch.zeros(B_2.shape)
        W_1 = torch.zeros(B_2.shape)
        for v in vectors.values():
            W_2 = W_2 + torch.cov(v.transpose(0,1), correction=0)
            W_1 = W_1 + torch.cov(v.transpose(0,1), correction=0)*v.shape[0]
        W_2 = W_2/len(vectors)
        W_1 = W_1/torch.sum(weights)
        B_3 = torch.cov(m_all, correction=0, fweights = weights)
        B_tmp3 = torch.zeros(B_2.shape)
        B_tmp4 = torch.zeros(B_2.shape)
        for i in range(len(vectors)):
            B_tmp4 = B_tmp4 + torch.mm((m_all[:,i].reshape(-1,1) - m_global), (m_all[:,i].reshape(-1,1) - m_global).transpose(0,1))
        B_tmp4 = B_tmp4/len(vectors)
        nc_balanced = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_1)))
        nc_imbalanced = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_2)))
        nc_3 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_2)))
        nc_4 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_3))) # paper version
        nc_5 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_1)))
        nc_6 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_3)))
        nc_7 = 1/len(vectors)*torch.trace(torch.matmul(W_1, torch.linalg.pinv(B_tmp4)))
        nc_8 = 1/len(vectors)*torch.trace(torch.matmul(W_2, torch.linalg.pinv(B_tmp4)))

        #old_nc == nc_4
        return([nc_balanced,nc_imbalanced, nc_3, nc_4, nc_5, nc_6, nc_7, nc_8])



    nc=nc_new1(vector_category)
    #logger.info(f'layer={args.num_hidden_layers}-token={args.chosen_token}-nc={compute_nc(vector_category)}')
    logger.info(f'layer={args.num_hidden_layers}-token={args.chosen_token}-nc_donoho={nc[0]}-nc_imbalanced={nc[1]}-nc_3={nc[2]}-nc_4={nc[3]}-nc_5={nc[4]}-nc_6={nc[5]}-nc_7={nc[6]}-nc_8={nc[7]}')
    with open(os.path.join(args.output_dir, "nc_results.json"), "w") as f:
        json.dump({"nc_donoho": nc[0].item(), "nc_imbalanced": nc[1].item(), "nc_3": nc[2].item(), "nc_4": nc[3].item(), "nc_5": nc[4].item(), "nc_6": nc[5].item(), "nc_7": nc[6].item(), "nc_8": nc[7].item()}, f)
        '''
        json.dump({"nc_donoho": nc[0].item()}, f)
        json.dump({"nc_imbalanced": nc[1].item()}, f)
        json.dump({"nc_3": nc[2].item()}, f)
        json.dump({"nc_4": nc[3].item()}, f)
        json.dump({"nc_5": nc[4].item()}, f)
        json.dump({"nc_6": nc[5].item()}, f)
        json.dump({"nc_7": nc[6].item()}, f)
        json.dump({"nc_8": nc[7].item()}, f)
        '''
    if args.delete_hidden_output:
        for k in dataloader_category:
            os.remove(f'{args.output_dir}/category={k}_{args.num_hidden_layers}_{args.chosen_token}.pt')
    '''
    nc=compute_nc(vector_category)
    #logger.info(f'layer={args.num_hidden_layers}-token={args.chosen_token}-nc={compute_nc(vector_category)}')
    logger.info(f'layer={args.num_hidden_layers}-token={args.chosen_token}-nc={nc}')
    with open(os.path.join(args.output_dir, "nc_results.json"), "w") as f:
        json.dump({"nc": nc.item()}, f)
    if args.delete_hidden_output:
        for k in dataloader_category:
            os.remove(f'{args.output_dir}/category={k}_{args.num_hidden_layers}_{args.chosen_token}.pt')
    '''
    
    return 0


    for epoch in range(args.num_train_epochs):
        model.train()
        '''
        if args.with_tracking:
            total_loss = 0
        '''
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == 0 and step < resume_step:
                continue
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            '''
            if args.with_tracking:
                total_loss += loss.detach().float()
            '''
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            '''
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            '''

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        #logger.info(f"epoch {epoch}: {eval_metric}")

        #accelerator.save_state(os.path.join(args.output_dir, 'last_model'))
        #print(eval_metric)
        if eval_metric['accuracy'] > best_metric:
            #accelerator.save_state(os.path.join(args.output_dir, 'best_model'))
            best_metric = eval_metric['accuracy']
            best_epoch = epoch
        logger.info(f"epoch {epoch}: train_loss={total_loss}|best_epoch={best_epoch}|accuracy={eval_metric['accuracy']}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                },
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                old_dir = os.path.join(args.output_dir, f"epoch_{epoch-1}")
            accelerator.save_state(output_dir)
            if epoch>=1:
                shutil.rmtree(old_dir)

    #temporarily not saving models
    '''
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
    '''

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)


if __name__ == "__main__":
    main()
