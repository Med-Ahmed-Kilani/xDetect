"""
Load HC3 (English subset only) directly from HuggingFace.

The HF datasets library handles caching in ~/.cache/huggingface/datasets/,
so no local raw file is written.
"""
import logging

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

EXPECTED_HUMAN_MIN = 20000
EXPECTED_MACHINE_MIN = 24000


def _flatten(dataset) -> pd.DataFrame:
    """Flatten HC3 Q&A rows into individual (text, label, generator) rows."""
    rows = []
    for record in dataset:
        question = record.get("question", "")
        for answer in record.get("human_answers", []):
            if answer and answer.strip():
                rows.append({"text": answer.strip(), "label": 0,
                             "generator": "human", "question": question})
        for answer in record.get("chatgpt_answers", []):
            if answer and answer.strip():
                rows.append({"text": answer.strip(), "label": 1,
                             "generator": "ChatGPT", "question": question})
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    """
    Return the HC3 English subset as a flattened DataFrame.

    Columns: text, label (0=human/1=machine), generator, question.
    HuggingFace caches the download automatically.
    """
    logger.info("Loading HC3 (Hello-SimpleAI/HC3, config=all) from HuggingFace …")
    ds = load_dataset("Hello-SimpleAI/HC3", "all")
    df = _flatten(ds["train"])

    n_human = (df["label"] == 0).sum()
    n_machine = (df["label"] == 1).sum()
    logger.info("HC3 EN: %d human rows, %d machine rows", n_human, n_machine)

    if n_human < EXPECTED_HUMAN_MIN:
        raise ValueError(f"HC3 human rows={n_human} < expected minimum {EXPECTED_HUMAN_MIN}")
    if n_machine < EXPECTED_MACHINE_MIN:
        raise ValueError(f"HC3 machine rows={n_machine} < expected minimum {EXPECTED_MACHINE_MIN}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load()
    print(df.head())
    print(f"Total rows: {len(df)}")
