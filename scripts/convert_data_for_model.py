#!/usr/bin/env python3
"""Convert simpleRL parquet data from Qwen ChatML format to model-specific formats.

The original simplerl-8k-{easy,hard} datasets have Qwen ChatML tokens baked into
the prompt content (<|im_start|>, <|im_end|>). This script extracts the raw
question from extra_info and wraps it in the target model's prompt template.

Usage:
    python scripts/convert_data_for_model.py --model llama --data-dir data/simplerl-8k-easy
    python scripts/convert_data_for_model.py --model deepseek-math --data-dir data/simplerl-8k-hard
"""

import argparse
import os

import pandas as pd

PROMPT_TEMPLATES = {
    # SimpleRL-Zoo simple prompt for weak instruction-followers (Figure 10).
    # No \boxed{} requirement — §3.1 shows format reward kills LLaMA training.
    "llama": (
        "Question:\n"
        "{question}\n"
        "Answer:\n"
        "Let's think step by step.\n"
    ),
    # SimpleRL-Zoo complex prompt for strong instruction-followers (Figure 10).
    # Clean prompt WITHOUT ChatML tokens — veRL's tokenizer applies chat template separately.
    "qwen": (
        "{question}\n"
        "Please reason step by step, "
        "and put your final answer within \\boxed{{}}."
    ),
    # For instruction-tuned LLaMA — can follow \boxed{} instructions.
    # Same content as qwen template; VeRL applies model-specific chat template automatically.
    # Ref: FastGRPO uses \boxed{} for LLaMA-Instruct training.
    "llama-instruct": (
        "{question}\n"
        "Please reason step by step, "
        "and put your final answer within \\boxed{{}}."
    ),
    "deepseek-math": (
        "User: {question}\n"
        "Please reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\n"
        "Assistant:"
    ),
}

# data_source to use per model (controls reward function dispatch)
MODEL_DATA_SOURCE = {
    "llama": "math_llama",
    "llama-instruct": "math_dapo",
    "qwen": "math_dapo",
    "deepseek-math": "math_dapo",
}


def convert_row(row, template: str, data_source: str = None):
    """Convert a single row, extracting raw question from extra_info."""
    extra = row["extra_info"]
    if isinstance(extra, str):
        import json
        extra = json.loads(extra)

    question = extra.get("question", "")
    if not question:
        return None

    content = template.format(question=question)
    new_prompt = [{"role": "user", "content": content}]

    # Preserve all original columns, replace prompt and optionally data_source
    result = row.to_dict()
    result["prompt"] = new_prompt
    if data_source is not None:
        result["data_source"] = data_source
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert simpleRL data for different models")
    parser.add_argument("--model", required=True, choices=list(PROMPT_TEMPLATES.keys()),
                        help="Target model family (llama, qwen, deepseek-math)")
    parser.add_argument("--data-dir", required=True,
                        help="Path to data directory (e.g., data/simplerl-8k-easy)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: {data-dir}-{model})")
    args = parser.parse_args()

    template = PROMPT_TEMPLATES[args.model]
    data_source = MODEL_DATA_SOURCE.get(args.model)
    output_dir = args.output_dir or f"{args.data_dir}-{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "test"]:
        input_path = os.path.join(args.data_dir, f"{split}.parquet")
        if not os.path.exists(input_path):
            print(f"Skipping {input_path} (not found)")
            continue

        df = pd.read_parquet(input_path)
        rows = []
        skipped = 0
        for _, row in df.iterrows():
            converted = convert_row(row, template, data_source=data_source)
            if converted is None:
                skipped += 1
                continue
            rows.append(converted)

        out_df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, f"{split}.parquet")
        out_df.to_parquet(output_path, index=False)
        print(f"{split}: {len(out_df)} samples written to {output_path}"
              f" (skipped {skipped})")

    print(f"\nDone. Output: {output_dir}")
    # Show sample
    sample_df = pd.read_parquet(os.path.join(output_dir, "train.parquet"))
    print(f"\nSample prompt:\n{sample_df.iloc[0]['prompt'][0]['content'][:300]}")


if __name__ == "__main__":
    main()
