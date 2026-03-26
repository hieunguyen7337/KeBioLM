# KeBioLM

Improving Biomedical Pretrained Language Models with Knowledge.  
Accepted by BioNLP 2021.  
[Paper](https://arxiv.org/abs/2104.10344)

## Overview
![KeBioLM architecture](pic/kebiolm.png)

KeBioLM is a biomedical pretrained language model with knowledge-aware pretraining.  
This repo currently contains:

- the original KeBioLM model code
- relation extraction and NER fine-tuning code
- prepared binary evaluation datasets for Unified PPI and Phos
- a single PBS-ready eval script: `eval.sh`

## Environment
This codebase was originally tested with:

- Python 3.7
- PyTorch 1.7.0
- Transformers 3.4.0

The current cluster runs in a conda environment selected by `eval.sh`.

## Model Files
The base checkpoint is expected under `model/`, including:

- `pytorch_model.bin`
- `config.json`
- `vocab.txt`
- `entity.jsonl`

Pretrained model link:

- [Google Drive checkpoint](https://drive.google.com/file/d/1kMbTsc9rPpBc-6ezEHjMbQLljW3SUWG9/edit)

Important: the base checkpoint in `model/` is not a task-specific relation extraction classifier.  
If you evaluate directly from `model/`, the classifier head is newly initialized, so results are only sanity-check baselines.

## Prepared Eval Datasets
Two binary relation-extraction datasets are already prepared in KeBioLM TSV format:

- `data/Unified_PPI_binary`
- `data/Phos_binary`

Each folder contains:

- `train.tsv`
- `dev.tsv`
- `test.tsv`

Each row uses the 3-column KeBioLM RE format:

```tsv
sample_id<TAB>sentence_with_@E1$_and_@E2$_markers<TAB>label
```

Binary labels used in these prepared datasets:

- Unified PPI: `Association`, `None`
- Phos: `Association`, `None`

## How To Run Eval
The repository uses a single PBS script, [`eval.sh`](F:\document\QUT_research_assistant_file\Dr_Bashar_file\KeBioLM\eval.sh), for evaluation.

Default behavior:

- dataset: `data/Phos_binary`
- output: `evaluation_result/Phos_binary`
- task name: `phos_binary`
- eval only
- no debug
- cached features reused unless requested otherwise

Submit the default Phos binary eval with:

```bash
qsub eval.sh
```

Useful overrides:

Run Unified PPI instead:

```bash
qsub -v DATA_DIR=data/Unified_PPI_binary,OUTPUT_DIR=evaluation_result/Unified_PPI_binary,TASK_NAME=unified_ppi eval.sh
```

Enable the inline debug block:

```bash
qsub -v RUN_DEBUG=1 eval.sh
```

Run prediction as well:

```bash
qsub -v DO_PREDICT=1 eval.sh
```

Rebuild cached features:

```bash
qsub -v OVERWRITE_CACHE=1 eval.sh
```

Use a different checkpoint:

```bash
qsub -v MODEL_PATH=/path/to/checkpoint eval.sh
```

Main configurable variables in `eval.sh`:

- `MODEL_PATH`
- `DATA_DIR`
- `OUTPUT_DIR`
- `TASK_NAME`
- `MAX_SEQ_LENGTH`
- `EVAL_BATCH_SIZE`
- `EVAL_ACCUMULATION_STEPS`
- `RUN_DEBUG`
- `DO_EVAL`
- `DO_PREDICT`
- `OVERWRITE_CACHE`
- `CONDA_ENV_NAME`

## Eval Outputs
Evaluation outputs are written under `evaluation_result/<dataset_name>/`.

Typical files:

- `eval_results.txt`
- `test_results.txt` when `DO_PREDICT=1`
- cached feature files under the dataset directory

## Current Eval Results
These are baseline eval results produced from the prepared binary datasets.

| Dataset | Result file | Samples | Accuracy | Micro F1 | Macro F1 | Weighted F1 |
| :-----: | :---------: | :-----: | :------: | :------: | :------: | :---------: |
| Unified PPI binary | `evaluation_result/Unified_PPI_binary/eval_results.txt` | 28604 | 0.5333 | 0.5333 | 0.4968 | 0.5402 |
| Phos binary | `evaluation_result/Phos_binary/eval_results.txt` | 1391 | 0.3767 | 0.3767 | 0.3247 | 0.4771 |

Confusion counts:

| Dataset | TP | FP | TN | FN |
| :-----: | :-: | :-: | :-: | :-: |
| Unified PPI binary | 11477 | 5946 | 3777 | 7404 |
| Phos binary | 69 | 806 | 455 | 61 |

Per-class F1 from `eval_results.txt`:

| Dataset | Class 0 F1 | Class 1 F1 |
| :-----: | :--------: | :--------: |
| Unified PPI binary | 0.3614 | 0.6323 |
| Phos binary | 0.5121 | 0.1373 |

## Notes
- `relation_extraction.run` expects `train.tsv`, `dev.tsv`, and `test.tsv` to exist even for eval-only runs.
- The KeBioLM RE loader identifies entity positions from the marker tokens in the sentence text.
- For large datasets, `eval_accumulation_steps` and smaller `per_device_eval_batch_size` help avoid GPU memory issues.

## Citation
```bibtex
@inproceedings{yuan-etal-2021-improving,
    title = "Improving Biomedical Pretrained Language Models with Knowledge",
    author = "Yuan, Zheng  and
      Liu, Yijia  and
      Tan, Chuanqi  and
      Huang, Songfang  and
      Huang, Fei",
    booktitle = "Proceedings of the 20th Workshop on Biomedical Language Processing",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.bionlp-1.20",
    doi = "10.18653/v1/2021.bionlp-1.20",
    pages = "180--190"
}
```
