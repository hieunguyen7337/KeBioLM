#!/bin/bash -l

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/Unified_PPI_binary}"
MODEL_PATH="${MODEL_PATH:-model}"
TASK_NAME="${TASK_NAME:-unified_ppi}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"

echo '================================================'
echo 'Debug: Unified_PPI_binary dataset'
echo '================================================'
echo "DATA_DIR = ${DATA_DIR}"
echo "MODEL_PATH = ${MODEL_PATH}"
echo "TASK_NAME = ${TASK_NAME}"
echo "MAX_SEQ_LENGTH = ${MAX_SEQ_LENGTH}"

python - <<'EOF'
import csv
import os
from collections import Counter
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
for split in ["train", "dev", "test"]:
    path = data_dir / f"{split}.tsv"
    label_counter = Counter()
    rows = 0
    bad_cols = 0
    bad_markers = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            rows += 1
            if len(row) != 3:
                bad_cols += 1
                continue
            text = row[1]
            label_counter[row[2]] += 1
            if text.count("@E1$") != 1 or text.count("@E2$") != 1:
                bad_markers += 1

    print(path.as_posix())
    print("  rows:", rows)
    print("  labels:", dict(label_counter))
    print("  bad_cols:", bad_cols)
    print("  bad_markers:", bad_markers)

print("model_path_exists:", Path(os.environ["MODEL_PATH"]).exists())
EOF

python - <<'EOF'
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
specials_before = len(tokenizer)
tokenizer.add_tokens(["@E1$", "@E2$"])
specials_after = len(tokenizer)
print("tokenizer_len_before:", specials_before)
print("tokenizer_len_after:", specials_after)
print("added_tokens:", specials_after - specials_before)
print("encode(@E1$):", tokenizer.encode("@E1$", add_special_tokens=False))
print("encode(@E2$):", tokenizer.encode("@E2$", add_special_tokens=False))
EOF

python - <<'EOF'
import os
from transformers import AutoTokenizer
from relation_extraction.utils import RelationExtractionDataset, Split

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"])
dataset = RelationExtractionDataset(
    data_dir=os.environ["DATA_DIR"],
    tokenizer=tokenizer,
    task=os.environ["TASK_NAME"],
    max_seq_length=int(os.environ["MAX_SEQ_LENGTH"]),
    overwrite_cache=True,
    mode=Split.dev,
)
print("dev_dataset_len:", len(dataset))
first = dataset[0]
print("first_example_label:", first.label)
print("first_example_first_entity_position:", first.first_entity_position)
print("first_example_second_entity_position:", first.second_entity_position)
EOF
