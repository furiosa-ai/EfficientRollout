"""
Prepare SimpleRL-Zoo-Data for verl 0.7.0 GRPO/DAPO training.

Downloads from https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data
and ensures compatibility with verl's reward function dispatching.

Difficulty buckets (each ~8K examples):
  easy   : GSM8K + MATH level 1   (simplelr_qwen_gsm8k_level1)
  medium : MATH level 1-4         (simplelr_qwen_level1to4)
  hard   : MATH level 3-5         (simplelr_qwen_level3to5)

Reference: SimpleRL-Zoo (arXiv:2503.18892)

Usage:
  python scripts/prepare_simplerl_zoo.py --level hard
  python scripts/prepare_simplerl_zoo.py --level easy --save_dir data/simplerl-8k-easy
"""

import argparse
import os
import json

import pandas as pd

LEVEL_TO_HF_DIR = {
    "easy": "simplelr_qwen_gsm8k_level1",
    "medium": "simplelr_qwen_level1to4",
    "hard": "simplelr_qwen_level3to5",
}

HF_BASE_URL = "https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main"

# verl 0.7.0 default_compute_score dispatches on data_source.
# SimpleRL-Zoo qwen-format data uses \boxed{} answers → math_dapo handles this correctly.
VERL_DATA_SOURCE = "math_dapo"


def download_file(url: str, dest: str) -> None:
    """Download a file if it doesn't exist, using urllib (no wget dependency)."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    from urllib.request import urlretrieve

    print(f"  Downloading {url}")
    urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def ensure_verl_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """Patch parquet columns for verl 0.7.0 compatibility.

    verl expects: data_source, prompt, reward_model (with style + ground_truth), extra_info
    """
    required_cols = {"prompt", "reward_model"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    if "data_source" not in df.columns:
        print(f"  Adding data_source='{VERL_DATA_SOURCE}' (was missing)")
        df["data_source"] = VERL_DATA_SOURCE
    else:
        original_sources = df["data_source"].unique()
        needs_patch = not all(
            src in (
                "openai/gsm8k",
                "DigitalLearningGmbH/MATH-lighteval",
                "HuggingFaceH4/MATH-500",
                "math_dapo", "math", "math_dapo_reasoning",
            ) or str(src).startswith("aime")
            for src in original_sources
        )
        if needs_patch:
            print(f"  Patching data_source: {list(original_sources)} → '{VERL_DATA_SOURCE}'")
            df["data_source"] = VERL_DATA_SOURCE
        else:
            print(f"  data_source already compatible: {list(original_sources)}")

    if "ability" not in df.columns:
        df["ability"] = "math"

    if "extra_info" not in df.columns:
        df["extra_info"] = [{"split": "train", "index": i} for i in range(len(df))]

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare SimpleRL-Zoo data for verl training")
    parser.add_argument(
        "--level",
        required=True,
        choices=["easy", "medium", "hard"],
        help="Difficulty level: easy (GSM8K+MATH lv1), medium (MATH lv1-4), hard (MATH lv3-5)",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Output directory (default: data/simplerl-8k-{level})",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.save_dir is None:
        save_dir = os.path.join(project_root, "data", f"simplerl-8k-{args.level}")
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    hf_dir = LEVEL_TO_HF_DIR[args.level]
    train_url = f"{HF_BASE_URL}/{hf_dir}/train.parquet?download=true"
    test_url = f"{HF_BASE_URL}/{hf_dir}/test.parquet?download=true"

    raw_train = os.path.join(save_dir, "raw_train.parquet")
    raw_test = os.path.join(save_dir, "raw_test.parquet")

    print(f"=== Preparing SimpleRL-Zoo [{args.level}] ({hf_dir}) ===")
    print()

    print("Step 1: Download from HuggingFace")
    download_file(train_url, raw_train)
    download_file(test_url, raw_test)
    print()

    print("Step 2: Verify and patch for verl 0.7.0 compatibility")
    for split, raw_path in [("train", raw_train), ("test", raw_test)]:
        print(f"  --- {split} ---")
        df = pd.read_parquet(raw_path)
        print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
        df = ensure_verl_compatible(df)

        out_path = os.path.join(save_dir, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")

        if split == "train":
            example = df.iloc[0].to_dict()
            example_path = os.path.join(save_dir, "train_example.json")
            with open(example_path, "w") as f:
                json.dump(example, f, indent=2, default=str, ensure_ascii=False)
            print(f"  Example: {example_path}")
    print()

    # Clean up raw files if patching was done
    for raw in [raw_train, raw_test]:
        final = raw.replace("raw_", "")
        if os.path.exists(raw) and os.path.exists(final) and raw != final:
            os.remove(raw)

    print("=== Done ===")
    print(f"Training data: {os.path.join(save_dir, 'train.parquet')}")
    print(f"Test data:     {os.path.join(save_dir, 'test.parquet')}")
    print(f"Example:       {os.path.join(save_dir, 'train_example.json')}")


if __name__ == "__main__":
    main()
