# Reward function for base models without instruction-following ability (e.g., LLaMA-3.1-8B).
#
# Key differences from math_dapo:
#   1. Flexible answer extraction: cascading fallback chain
#      (boxed → "the answer is" → "final answer is" → last number)
#   2. Reward: +1 correct, 0 incorrect — following SimpleRL-Zoo §B.2
#   3. No format penalty — following SimpleRL-Zoo §3.1 finding that format
#      reward kills exploration for weak instruction-followers
#
# Reference: SimpleRL-Zoo (Zeng et al., 2025) — https://github.com/hkust-nlp/simpleRL-reason

import re
from typing import Optional


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{} in text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx + len("\\boxed{")
    depth = 1
    content = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
            content.append(text[i])
        elif text[i] == "}":
            depth -= 1
            if depth > 0:
                content.append(text[i])
        else:
            content.append(text[i])
        i += 1

    return "".join(content) if depth == 0 else None


def extract_answer_flexible(text: str) -> Optional[str]:
    """Cascading answer extraction for base models.

    Tries multiple patterns in order of specificity, falling back to the
    last number in the text.  Ported from SimpleRL-Zoo's
    qwen_math_eval_toolkit/parser.py:extract_answer().
    """
    # 1. Minerva format: "final answer is $...$. I hope"
    if "final answer is $" in text and "$. I hope" in text:
        tmp = text.split("final answer is $", 1)[1]
        return tmp.split("$. I hope", 1)[0].strip()

    # 2. LaTeX \boxed{}
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed

    # 3. "the answer is" (case-insensitive, word-boundary match)
    m_answer = re.search(r"(?i)\bthe answer is\b", text)
    if m_answer:
        tail = text[m_answer.end():].strip()
        # Extract the first token/expression: a number (possibly decimal/negative/fraction),
        # a $..$ LaTeX expression, or a short word answer
        m = re.match(r"([\$]?-?\d[\d,./]*\d*[\$]?|[\$][^$]+[\$]|\S+)", tail)
        return m.group(1).strip().strip("$") if m else tail

    # 4. "final answer is" (case-insensitive, word-boundary match)
    m_final = re.search(r"(?i)\bfinal answer is\b", text)
    if m_final:
        tail = text[m_final.end():].strip()
        m = re.match(r"([\$]?-?\d[\d,./]*\d*[\$]?|[\$][^$]+[\$]|\S+)", tail)
        return m.group(1).strip().strip("$") if m else tail

    # 5. "Answer:" pattern (Minerva-style)
    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", text)
    if match:
        return match[-1].strip()

    # 6. Last number fallback — critical for base models
    #    Only scan the last 1000 chars to reduce false positives from reasoning steps
    scan_text = text[-1000:] if len(text) > 1000 else text
    numbers = re.findall(r"-?\d*\.?\d+", scan_text.replace(",", ""))
    if numbers:
        return numbers[-1]

    return None


def _clean_prediction(pred: str) -> str:
    """Post-process extracted prediction string."""
    if not pred:
        return ""
    # Remove leading/trailing noise
    pred = re.sub(r"\n\s*", "", pred)
    if pred and pred[0] == ":":
        pred = pred[1:]
    if pred and pred[-1] == ".":
        pred = pred[:-1]
    if pred and pred[-1] == "/":
        pred = pred[:-1]
    # Strip whitespace and common wrappers
    pred = pred.strip().strip("$").strip()
    return pred


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Handles commas in numbers, common LaTeX wrappers, and whitespace.
    """
    answer = answer.strip()
    # Remove LaTeX wrappers
    answer = re.sub(r"\\text\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\\textbf\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\\overline\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\\boxed\{(.*)\}", r"\1", answer)
    # Remove dollar signs
    answer = answer.replace("$", "")
    # Normalize commas in numbers
    if answer.replace(",", "").replace(".", "").replace("-", "").isdigit():
        answer = answer.replace(",", "")
    # Normalize fractions: \frac{a}{b} → a/b for simple cases
    answer = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", answer)
    # Remove common units/suffixes
    for unit in ["square", "ways", "integers", "dollars", "mph", "inches",
                 "hours", "km", "units", "feet", "minutes", "degrees",
                 "cm", "meters", "pounds"]:
        answer = answer.replace(unit, "")
    answer = answer.strip()
    return answer


def _parse_numeric(s: str) -> Optional[float]:
    """Try to parse a string as a number, handling simple fractions like '1/2'."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    # Handle simple fractions: a/b
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    return None


def _numeric_equal(a: str, b: str) -> bool:
    """Check if two strings represent the same number."""
    fa = _parse_numeric(a)
    fb = _parse_numeric(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-6
    return False


def compute_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> dict:
    """Compute reward for base-model math solutions.

    Returns:
        dict with keys: score (float), acc (bool), pred (str|None)
    """
    # Extract prediction using flexible cascading strategy
    pred_raw = extract_answer_flexible(solution_str)
    pred = _clean_prediction(pred_raw) if pred_raw else ""

    # Normalize both for comparison
    pred_norm = normalize_answer(pred) if pred else ""
    gt_norm = normalize_answer(ground_truth)

    # Check correctness: exact match after normalization, or numeric equality
    correct = False
    if pred_norm and gt_norm:
        correct = (pred_norm == gt_norm) or _numeric_equal(pred_norm, gt_norm)

    # Reward: +1 correct, 0 incorrect (SimpleRL-Zoo §B.2, no penalty for exploration)
    reward = 1.0 if correct else 0.0

    return {
        "score": reward,
        "acc": correct,
        "pred": pred,
    }
