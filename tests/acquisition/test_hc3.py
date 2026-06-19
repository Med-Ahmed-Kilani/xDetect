"""
Tests for src/acquisition/hc3.py — focusing on the local caching behaviour.
No real network calls are made; requests.get and pd.read_parquet are mocked.
Row-count validation is patched out in caching tests — it is the responsibility
of _validate() which is tested separately.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_raw_df() -> pd.DataFrame:
    """Minimal raw HC3-shaped DataFrame (one Q&A row, both answer types)."""
    return pd.DataFrame([{
        "question": "What is AI?",
        "human_answers": ["A field of computer science."],
        "chatgpt_answers": ["Artificial intelligence is the simulation of human intelligence."],
    }])


def _mock_requests_get(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "parquet_files": [
            {"config": "all", "split": "train",
             "url": "https://fake-hf.co/hc3/train/0000.parquet"},
        ]
    }
    return resp


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestHc3Caching:
    """
    All tests patch _validate() to a no-op — row-count correctness is a
    separate concern tested in TestHc3Validate below.
    """

    def test_second_call_skips_network(self, tmp_path):
        """Two load() calls must result in exactly one network fetch."""
        cache = tmp_path / "hc3_en.parquet"

        from src.acquisition.hc3 import load

        with patch("src.acquisition.hc3.resolve_path", return_value=cache), \
             patch("src.acquisition.hc3.requests.get", side_effect=_mock_requests_get) as mock_get, \
             patch("src.acquisition.hc3.pd.read_parquet", return_value=_fake_raw_df()), \
             patch("src.acquisition.hc3._validate"):

            # First call — cache miss, must hit network
            load(force=False)
            assert mock_get.call_count == 1

            # Second call — cache hit, must NOT hit network
            load(force=False)
            assert mock_get.call_count == 1  # still 1, not 2

    def test_force_refetches(self, tmp_path):
        """force=True must bypass the cache and hit the network even if cache exists."""
        cache = tmp_path / "hc3_en.parquet"
        # Pre-populate the cache so it exists on disk
        _fake_raw_df().to_parquet(cache, index=False)

        from src.acquisition.hc3 import load

        with patch("src.acquisition.hc3.resolve_path", return_value=cache), \
             patch("src.acquisition.hc3.requests.get", side_effect=_mock_requests_get) as mock_get, \
             patch("src.acquisition.hc3.pd.read_parquet", return_value=_fake_raw_df()), \
             patch("src.acquisition.hc3._validate"):

            load(force=True)
            assert mock_get.call_count == 1  # fetched despite cache existing

    def test_cache_file_written_after_fetch(self, tmp_path):
        """After the first load(), the cache parquet file must exist on disk."""
        cache = tmp_path / "hc3_en.parquet"
        assert not cache.exists()

        from src.acquisition.hc3 import load

        with patch("src.acquisition.hc3.resolve_path", return_value=cache), \
             patch("src.acquisition.hc3.requests.get", side_effect=_mock_requests_get), \
             patch("src.acquisition.hc3.pd.read_parquet", return_value=_fake_raw_df()), \
             patch("src.acquisition.hc3._validate"):

            load(force=False)

        assert cache.exists()

    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        """load() must return a DataFrame with text, label, generator, question."""
        cache = tmp_path / "hc3_en.parquet"

        from src.acquisition.hc3 import load

        with patch("src.acquisition.hc3.resolve_path", return_value=cache), \
             patch("src.acquisition.hc3.requests.get", side_effect=_mock_requests_get), \
             patch("src.acquisition.hc3.pd.read_parquet", return_value=_fake_raw_df()), \
             patch("src.acquisition.hc3._validate"):

            df = load(force=False)

        assert set(["text", "label", "generator", "question"]).issubset(df.columns)
        assert set(df["label"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

class TestHc3Validate:
    def _call(self, n_human: int, n_machine: int):
        from src.acquisition.hc3 import _validate
        rows = (
            [{"text": "h", "label": 0, "generator": "human", "question": "q"}] * n_human
            + [{"text": "m", "label": 1, "generator": "ChatGPT", "question": "q"}] * n_machine
        )
        _validate(pd.DataFrame(rows), expected_human_min=2, expected_machine_min=2)

    def test_passes_when_counts_met(self):
        self._call(n_human=3, n_machine=3)  # should not raise

    def test_raises_on_too_few_human(self):
        with pytest.raises(ValueError, match="human rows"):
            self._call(n_human=1, n_machine=3)

    def test_raises_on_too_few_machine(self):
        with pytest.raises(ValueError, match="machine rows"):
            self._call(n_human=3, n_machine=1)
