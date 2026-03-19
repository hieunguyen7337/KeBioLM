import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from .relation_extraction_classification import BertForRelationExtraction 
from .utils import RelationExtractionDataset, Split, REProcessor

import sys
sys.path.append('..')
from modeling_kebio import KebioForRelationExtraction
from configuration_kebio import KebioConfig
from entity_indexer import EntityIndexer

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        task = data_args.task_name.lower()
        ignore_first = True
        if task not in ["gad", "hoc", "jnlpba"]:
            ignore_first = False
        processor = REProcessor(ignore_first)
        train_text = processor._read_tsv(os.path.join(data_args.data_dir, "train.tsv"), ignore_first)
        label_list = processor._find_labels(train_text)
        marker_list = processor._find_marker(train_text)
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = KebioConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = KebioForRelationExtraction.from_pretrained(
        model_args.model_name_or_path, config=config
    )

    # Get datasets
    train_dataset = (
        RelationExtractionDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        RelationExtractionDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    tokenizer.add_tokens(marker_list)
    model.resize_token_embeddings(len(tokenizer))

    from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    
    def compute_metrics_universal(p) -> Dict:
        out: Dict = {}
    
        # ---- predictions -> class ids ----
        raw_pred = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        pred = np.asarray(raw_pred)
        labels = np.asarray(p.label_ids)
    
        # Debug: shapes
        out["pred_shape"] = tuple(pred.shape)
        out["labels_shape"] = tuple(labels.shape)
    
        # Handle typical shapes:
        # - (N,) or (N,1): binary single-logit/score -> threshold at 0 if logits (adjust if using probs)
        # - (N,2..C): multiclass logits/probs -> argmax
        if pred.ndim == 1:
            y_pred = (pred > 0).astype(int)
            out["pred_interpretation"] = "binary_single_score_threshold_0"
        elif pred.ndim == 2 and pred.shape[1] == 1:
            y_pred = (pred[:, 0] > 0).astype(int)
            out["pred_interpretation"] = "binary_single_score_column_threshold_0"
        else:
            y_pred = np.argmax(pred, axis=1)
            out["pred_interpretation"] = "multiclass_argmax"
    
        out["y_pred_unique"] = [int(x) for x in np.unique(y_pred)]
        out["labels_unique"] = [int(x) for x in np.unique(labels)]
    
        # ---- basic info ----
        unique_labels = np.unique(np.concatenate([labels, y_pred]))
        cm_labels = np.sort(unique_labels)
        n_classes = cm_labels.size
        out["observed_classes"] = [int(x) for x in unique_labels]
        out["cm_labels"] = [int(x) for x in cm_labels]
        out["n_classes"] = int(n_classes)
    
        # ---- overall metrics ----
        acc = accuracy_score(labels, y_pred)
        prec_micro = precision_score(labels, y_pred, average="micro",   zero_division=0)
        rec_micro  = recall_score(labels,  y_pred, average="micro",     zero_division=0)
        f1_micro   = f1_score(labels,      y_pred, average="micro",     zero_division=0)
    
        prec_macro = precision_score(labels, y_pred, average="macro",   zero_division=0)
        rec_macro  = recall_score(labels,    y_pred, average="macro",   zero_division=0)
        f1_macro   = f1_score(labels,        y_pred, average="macro",   zero_division=0)
    
        prec_w = precision_score(labels, y_pred, average="weighted", zero_division=0)
        rec_w  = recall_score(labels,    y_pred, average="weighted", zero_division=0)
        f1_w   = f1_score(labels,        y_pred, average="weighted", zero_division=0)
    
        out.update({
            "accuracy": acc,
            "precision_micro": prec_micro,
            "recall_micro": rec_micro,
            "f1_micro": f1_micro,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_w,
            "recall_weighted": rec_w,
            "f1_weighted": f1_w,
            "micro_minus_accuracy": float(f1_micro - acc)
        })
    
        # ---- confusion matrix (stable order) ----
        cm = confusion_matrix(labels, y_pred, labels=cm_labels)
        for i, li in enumerate(cm_labels):
            for j, lj in enumerate(cm_labels):
                out[f"cm_{li}_{lj}"] = int(cm[i, j])  # row=true li, col=pred lj
    
        # True supports (row sums) and predicted counts (col sums)
        row_sums = cm.sum(axis=1)  # actual class supports
        col_sums = cm.sum(axis=0)  # predicted counts per class
        for i, li in enumerate(cm_labels):
            out[f"support_true_{li}"] = int(row_sums[i])
        for j, lj in enumerate(cm_labels):
            out[f"count_pred_{lj}"] = int(col_sums[j])
    
        # Classes that were never predicted (useful for collapse detection)
        never_predicted = [int(cm_labels[j]) for j in range(len(cm_labels)) if col_sums[j] == 0]
        out["never_predicted_classes"] = never_predicted
    
        # Majority baseline accuracy
        # majority class = argmax of true supports
        maj_idx = int(np.argmax(row_sums))
        maj_class = int(cm_labels[maj_idx])
        maj_acc = float(row_sums[maj_idx] / row_sums.sum()) if row_sums.sum() > 0 else 0.0
        out["majority_class"] = maj_class
        out["majority_baseline_accuracy"] = maj_acc
        out["beats_majority_baseline"] = bool(acc >= maj_acc)
    
        # helper to pretty-name classes if available
        def name_for(cid: int) -> str:
            pretty = None
            try:
                pretty = p.model.config.id2label.get(int(cid))  # HF models
            except Exception:
                pass
            return str(pretty) if pretty is not None else str(cid)
    
        # echo HF label mapping if present (for debugging)
        try:
            id2label = getattr(p.model.config, "id2label", None)
            label2id = getattr(p.model.config, "label2id", None)
            if isinstance(id2label, dict):
                out["config_id2label"] = {str(k): str(v) for k, v in id2label.items()}
            if isinstance(label2id, dict):
                out["config_label2id"] = {str(k): int(v) for k, v in label2id.items()}
        except Exception:
            pass
    
        if n_classes == 2:
            pos_label = int(cm_labels.max())  # default: larger label is positive
    
            # find neg label
            if cm_labels[0] != pos_label:
                neg_label = int(cm_labels[0])
            else:
                neg_label = int(cm_labels[1])
    
            # indices in cm_labels
            pos_idx = int(np.where(cm_labels == pos_label)[0][0])
            neg_idx = int(np.where(cm_labels == neg_label)[0][0])
    
            # cm layout: rows = true, cols = pred
            tp = int(cm[pos_idx, pos_idx])
            tn = int(cm[neg_idx, neg_idx])
            fp = int(cm[neg_idx, pos_idx])
            fn = int(cm[pos_idx, neg_idx])
    
            out.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})
    
            # per-class metrics even for binary
            prec_c = precision_score(labels, y_pred, average=None, labels=cm_labels, zero_division=0)
            rec_c  = recall_score(labels,    y_pred, average=None, labels=cm_labels, zero_division=0)
            f1_c   = f1_score(labels,        y_pred, average=None, labels=cm_labels, zero_division=0)
            for i, cid in enumerate(cm_labels):
                cls = name_for(int(cid))
                out[f"precision_{cls}"] = float(prec_c[i])
                out[f"recall_{cls}"]    = float(rec_c[i])
                out[f"f1_{cls}"]        = float(f1_c[i])
    
        else:
            # One-vs-rest counts per class
            tp_vec = np.diag(cm)
            fp_vec = cm.sum(axis=0) - tp_vec
            fn_vec = cm.sum(axis=1) - tp_vec
            tn_vec = cm.sum() - (tp_vec + fp_vec + fn_vec)
    
            out.update({
                "tp_micro_sum": int(tp_vec.sum()),
                "fp_micro_sum": int(fp_vec.sum()),
                "fn_micro_sum": int(fn_vec.sum()),
                "tn_micro_sum": int(tn_vec.sum()),
            })
    
            # per-class breakdown
            for idx, cid in enumerate(cm_labels):
                cls = name_for(int(cid))
                out[f"tp_{cls}"] = int(tp_vec[idx])
                out[f"fp_{cls}"] = int(fp_vec[idx])
                out[f"fn_{cls}"] = int(fn_vec[idx])
                out[f"tn_{cls}"] = int(tn_vec[idx])
    
            # per-class precision/recall/f1
            prec_c = precision_score(labels, y_pred, average=None, labels=cm_labels, zero_division=0)
            rec_c  = recall_score(labels,    y_pred, average=None, labels=cm_labels, zero_division=0)
            f1_c   = f1_score(labels,        y_pred, average=None, labels=cm_labels, zero_division=0)
            for i, cid in enumerate(cm_labels):
                cls = name_for(int(cid))
                out[f"precision_{cls}"] = float(prec_c[i])
                out[f"recall_{cls}"]    = float(rec_c[i])
                out[f"f1_{cls}"]        = float(f1_c[i])
    
        return out


    # Example Trainer wiring (unchanged)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_universal,
    )


    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    if training_args.do_predict:
        test_dataset = RelationExtractionDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        pred = np.argmax(predictions[0], axis=1) 
        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                    
        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        id2label = {idx:label for idx, label in enumerate(label_list)}
        if trainer.is_world_master():
            with open(output_test_predictions_file, "w") as writer:
                for pre in pred:
                    writer.write(f'{id2label[pre]}\n')

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

