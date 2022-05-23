# GraphFusionDocRE
Code for Fusing Graph Neural Networks with Pre-trained Language Models for Document-Level Relation Extraction. It is based on [SSAN](https://github.com/BenfengXu/SSAN) and [ATLOP](https://github.com/wzhouad/ATLOP).

## Requirements
* Python (tested on 3.7.4)
* CUDA (tested on 11.2)
* [PyTorch](http://pytorch.org/) (tested on 1.7.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 3.4.0)
* numpy (tested on 1.19.4)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The CDR and GDA datasets can be obtained following the instructions in [edge-oriented graph](https://github.com/fenchri/edge-oriented-graph). The expected structure of files is:
```
ATLOP
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |-- cdr
 |    |    |-- train_filter.data
 |    |    |-- dev_filter.data
 |    |    |-- test_filter.data
 |    |-- gda
 |    |    |-- train.data
 |    |    |-- dev.data
 |    |    |-- test.data
 |-- meta
 |    |-- rel2id.json
```

## Training and Evaluation
### DocRED
Train the BERT model on DocRED with the following command:

```bash
>> sh scripts/run_bert.sh  # for BERT
>> sh scripts/run_roberta.sh  # for RoBERTa
```

The training loss and evaluation results on the dev set are synced to the wandb dashboard.

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.

### CDR and GDA
Train CDA and GDA model with the following command:
```bash
>> sh scripts/run_cdr.sh  # for CDR
>> sh scripts/run_gda.sh  # for GDA
```

The training loss and evaluation results on the dev and test set are synced to the wandb dashboard.

## Saving and Evaluating Models
You can save the model by setting the `--save_path` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks.
