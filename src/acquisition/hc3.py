"""
Load HC3 (English subset only) directly from HuggingFace parquet files.

Bypasses the `datasets` library entirely — HC3 uses a custom loading script
that is incompatible with datasets>=3.0, and datasets<3.0 has a dill/Python
3.14 incompatibility. Instead, we fetch parquet URLs from the HF datasets
server API and load them with pandas directly.

Validation thresholds come from configs/datasets.yaml.
"""
import logging

import pandas as pd
import requests

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

_HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"


def _get_parquet_urls(dataset_id: str, config: str, split: str) -> list[str]:
    """Fetch parquet file URLs for a given dataset/config/split from the HF server."""
    url = f"{_HF_DATASETS_SERVER}/parquet?dataset={dataset_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("parquet_files", [])
    urls = [f["url"] for f in files if f["config"] == config and f["split"] == split]
    if not urls:
        raise ValueError(
            f"No parquet files found for {dataset_id} config={config} split={split}. "
            f"Available: {[(f['config'], f['split']) for f in files]}"
        )
    return urls


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten HC3 Q&A rows into individual (text, label, generator) rows."""
    rows = []
    for _, record in df.iterrows():
        question = record.get("question", "")
        human_answers   = record.get("human_answers")
        chatgpt_answers = record.get("chatgpt_answers")
        # Answers may be stored as numpy arrays in parquet — convert to list first
        if hasattr(human_answers, "tolist"):
            human_answers = human_answers.tolist()
        if hasattr(chatgpt_answers, "tolist"):
            chatgpt_answers = chatgpt_answers.tolist()
        for answer in (human_answers or []):
            if answer and str(answer).strip():
                rows.append({"text": str(answer).strip(), "label": 0,
                             "generator": "human", "question": question})
        for answer in (chatgpt_answers or []):
            if answer and str(answer).strip():
                rows.append({"text": str(answer).strip(), "label": 1,
                             "generator": "ChatGPT", "question": question})
    return pd.DataFrame(rows)


def _validate(df: pd.DataFrame, expected_human_min: int, expected_machine_min: int) -> None:
    n_human   = (df["label"] == 0).sum()
    n_machine = (df["label"] == 1).sum()
    logger.info("HC3 EN: %d human rows, %d machine rows", n_human, n_machine)
    if n_human < expected_human_min:
        raise ValueError(
            f"HC3 human rows={n_human} < expected minimum {expected_human_min} "
            f"(set in configs/datasets.yaml hc3.expected_human_min)"
        )
    if n_machine < expected_machine_min:
        raise ValueError(
            f"HC3 machine rows={n_machine} < expected minimum {expected_machine_min} "
            f"(set in configs/datasets.yaml hc3.expected_machine_min)"
        )


def load(force: bool = False) -> pd.DataFrame:
    """
    Return the HC3 English subset as a flattened DataFrame.

    Columns: text, label (0=human / 1=machine), generator, question.
    Results are cached locally at data/raw/hc3/hc3_en.parquet. Subsequent
    calls return the cached file without hitting the network unless force=True.
    Validation thresholds are read from configs/datasets.yaml.
    """
    cfg = load_config("datasets")["hc3"]
    expected_human_min   = cfg["expected_human_min"]
    expected_machine_min = cfg["expected_machine_min"]

    cache_path = resolve_path("data/raw/hc3/hc3_en.parquet")

    if cache_path.exists() and not force:
        logger.info("Loading HC3 from local cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        _validate(df, expected_human_min, expected_machine_min)
        return df

    logger.info("Fetching HC3 parquet URLs from HuggingFace datasets server …")
    urls = _get_parquet_urls(cfg["hf_id"], cfg["config"], split="train")
    logger.info("Found %d parquet file(s). Downloading …", len(urls))

    chunks = []
    for url in urls:
        logger.info("  Loading %s …", url)
        chunks.append(pd.read_parquet(url))
    raw_df = pd.concat(chunks, ignore_index=True)

    logger.info("Flattening Q&A rows …")
    df = _flatten(raw_df)

    _validate(df, expected_human_min, expected_machine_min)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("HC3 cached to %s", cache_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load()
    print(df.head())
    print(f"Total rows: {len(df)}")
