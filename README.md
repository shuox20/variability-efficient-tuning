# variability-efficient-tuning

This is the implementation of the methods described in our paper "[Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning](https://arxiv.org/abs/2210.10041)".

## Reference
If you use this code as part of any published research, please acknowledge the following paper:

```
@inproceedings{xie2022nc,
  abbr = {EMNLP},
  bibtex_show = {true},
  author = {Xie, Shuo and Qiu, Jiahao and Pasad, Ankita and Du, Li and Qu, Qing and Mei, Hongyuan},
  title = {Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning},
  booktitle = {Findings of EMNLP},
  year = {2022},
  arxiv = {2210.10041}
}
```

## Instructions
Here are the instructions to use the code base.

### Dependencies and Installation
This code is written in Python 3.7.

Run the command line below to install the package (add `-e` option if you need an editable installation):
```
pip install .
```
It will automatically install the following important dependencies: 
* [PyTorch 1.11.0](https://pytorch.org/) that handles auto-differentiation.
* [Transformers 4.19.0](https://huggingface.co/docs/transformers/index) that handles pretrained language models.

### Compute metrics
Use the following code to compute variability ratio metric. Note that the index of layers starts from 0. The top layer of a 24-layer model is indexed by 23. 
```
bash scripts/nc_compute.sh
```

### Tuning
There are scripts to finetune and adapter-tune PLMs. The place of classification head is controlled by `--head_layer`. The layers to tuned are controlled by `--num_finetune_layers` and `--last_finetune_layer`.

The code for the adapter-tuning method is borrowed from [this repo](https://github.com/jxhe/unify-parameter-efficient-tuning). It's possible to try other parameter efficient tuning methods following their instructions. 
```
bash scripts/finetune.sh
bash scripts/ef_finetune.sh
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
