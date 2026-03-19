import argparse
import csv
import re
from collections import Counter
from pathlib import Path


SRC_OPEN_RE = re.compile(r"@[^\s@/$]*Src\$")
SRC_CLOSE_RE = re.compile(r"@/[^\s@/$]*Src\$|@[^\s@/$]*Src/\$")
TGT_OPEN_RE = re.compile(r"@[^\s@/$]*Tgt\$")
TGT_CLOSE_RE = re.compile(r"@/[^\s@/$]*Tgt\$|@[^\s@/$]*Tgt/\$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LitCoin-style TSV into 3-column KeBioLM RE TSV."
    )
    parser.add_argument("--input", required=True, help="Input TSV path.")
    parser.add_argument("--output", required=True, help="Output TSV path.")
    parser.add_argument(
        "--id-col",
        type=int,
        default=0,
        help="0-based column index containing the sample id. Default: 0",
    )
    parser.add_argument(
        "--text-col",
        type=int,
        default=7,
        help="0-based column index containing the marked text. Default: 7",
    )
    parser.add_argument(
        "--label-col",
        type=int,
        default=-1,
        help="0-based column index containing the label. Default: -1",
    )
    parser.add_argument(
        "--label-mode",
        choices=["keep", "binary"],
        default="keep",
        help="Keep original labels, or collapse to binary None/Association.",
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first row of the input file.",
    )
    parser.add_argument(
        "--negative-label",
        default="None",
        help="Negative label used in --label-mode binary. Default: None",
    )
    parser.add_argument(
        "--positive-label",
        default="Association",
        help="Positive label used in --label-mode binary. Default: Association",
    )
    return parser.parse_args()


def maybe_strip_prompt(text: str) -> str:
    if "[SEP]" in text:
        return text.split("[SEP]", 1)[1].strip()
    return text.strip()


def convert_entity_markers(text: str) -> str:
    text = maybe_strip_prompt(text)

    # Support either LitCoin-style tags or JSONL-style [E1]/[E2] tags.
    text = text.replace("[E1]", "@E1$")
    text = text.replace("[/E1]", "")
    text = text.replace("[E2]", "@E2$")
    text = text.replace("[/E2]", "")

    text = SRC_OPEN_RE.sub("@E1$", text)
    text = SRC_CLOSE_RE.sub("", text)
    text = TGT_OPEN_RE.sub("@E2$", text)
    text = TGT_CLOSE_RE.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert_label(label: str, mode: str, negative_label: str, positive_label: str) -> str:
    label = label.strip()
    if mode == "keep":
        return label

    lowered = label.lower()
    negative_values = {
        "none",
        "false",
        "negative",
        "ddi-false",
        negative_label.lower(),
    }
    if lowered in negative_values:
        return negative_label
    return positive_label


def validate_text(text: str) -> bool:
    return text.count("@E1$") == 1 and text.count("@E2$") == 1


def convert_file(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    label_counter = Counter()
    skipped_examples = []

    with input_path.open("r", encoding="utf-8", newline="") as fin, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")

        for row_idx, row in enumerate(reader):
            if row_idx == 0 and args.skip_header:
                continue
            if not row:
                continue

            total += 1

            try:
                sample_id = row[args.id_col].strip()
                raw_text = row[args.text_col]
                raw_label = row[args.label_col]
            except IndexError:
                skipped += 1
                skipped_examples.append((row_idx + 1, "missing_columns"))
                continue

            text = convert_entity_markers(raw_text)
            label = convert_label(
                raw_label,
                mode=args.label_mode,
                negative_label=args.negative_label,
                positive_label=args.positive_label,
            )

            if not validate_text(text):
                skipped += 1
                skipped_examples.append(
                    (
                        row_idx + 1,
                        f"marker_count_e1={text.count('@E1$')},marker_count_e2={text.count('@E2$')}",
                    )
                )
                continue

            writer.writerow([sample_id, text, label])
            written += 1
            label_counter[label] += 1

    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"rows_seen={total}")
    print(f"rows_written={written}")
    print(f"rows_skipped={skipped}")
    print(f"label_distribution={dict(label_counter)}")
    if skipped_examples:
        print("first_skipped_examples=")
        for line_no, reason in skipped_examples[:10]:
            print(f"  line={line_no} reason={reason}")


if __name__ == "__main__":
    convert_file(parse_args())
