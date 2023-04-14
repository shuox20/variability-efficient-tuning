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
from nc.utils.nc_function import *

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
        "--last_layer",
        type=int,
        default=None, 
        help="index of output layer",
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
    if args.last_layer is not None:
        config.num_hidden_layers=args.last_layer+1
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

    #divide train dataset by their labels
    dataset_category = {}
    for sample in train_dataset:
        if sample['labels'] not in dataset_category:
            dataset_category[sample['labels']]=[sample]
        else:
            dataset_category[sample['labels']].append(sample)
    logger.info('dataset seperate into', dataset_category.keys())
    del train_dataset
    torch.cuda.empty_cache()

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

    dataloader_category = {}
    for k,v in dataset_category.items():
        logger.info(f'category {k} has {len(v)} samples.')
        dataloader_category[k]=DataLoader(v, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    logger.info(f'dataloader seperate into {dataloader_category.keys()}')
    
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
    model = accelerator.prepare(model)
    for k in dataloader_category:
        dataloader_category[k] = accelerator.prepare(dataloader_category[k])

    model.eval()
    
    vector_category = {}
    for k,v in dataloader_category.items():
        if os.path.exists(f'{args.output_dir}/category={k}_{args.last_layer}_{args.chosen_token}.pt'):
            print(f'resume result in {k} from {args.output_dir}/category={k}_{args.last_layer}_{args.chosen_token}.pt')
            logger.info(f'resume result in {k} from {args.output_dir}/category={k}_{args.last_layer}_{args.chosen_token}.pt')
            vector_category[k]=torch.load(f'{args.output_dir}/category={k}_{args.last_layer}_{args.chosen_token}.pt', map_location=None)
            continue
        i=0
        for batch in v:
            with torch.no_grad():
                state = model(**batch).hidden_states[args.last_layer+1]
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
            torch.save(vector_category[k], f'{args.output_dir}/category={k}_{args.last_layer}_{args.chosen_token}.pt')
            logger.info(f'save vectors in category {k}')

    nc=compute_nc(vector_category)
    logger.info(f'layer={args.last_layer}-token={args.chosen_token}-nc={nc}')
    with open(os.path.join(args.output_dir, "nc_results.json"), "w") as f:
        json.dump({"nc": nc.item()}, f)
    if args.delete_hidden_output:
        for k in dataloader_category:
            os.remove(f'{args.output_dir}/category={k}_{args.last_layer}_{args.chosen_token}.pt')
    
    return 0


if __name__ == "__main__":
    main()
