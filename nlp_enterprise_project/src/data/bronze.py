"""Bronze layer: raw data ingestion from trusted sources.

In the bronze layer we simply download and persist the dataset in its raw form,
plus record provenance metadata.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict
from datasets import load_dataset, Dataset


class BronzeDataLoader:
    """Load raw data from Hugging Face and store it without modification."""

    def __init__(self, dataset_name: str, bronze_dir: str):
        self.dataset_name = dataset_name
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

    def load(self, config_subset: str = "default") -> "Dataset":
        """Load dataset from Hugging Face and return the DatasetDict."""
        raw = load_dataset(self.dataset_name)
        return raw

    def save(self, dataset, split: str = "all") -> Path:
        """Persist raw dataset to disk as JSON Lines."""
        record = {}
        for split_name in dataset.keys():
            out_path = self.bronze_dir / f"{split_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in dataset[split_name]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            record[split_name] = str(out_path)

        # provenance file
        provenance = {
            "dataset": self.dataset_name,
            "splits": list(dataset.keys()),
            "num_rows": {k: len(v) for k, v in dataset.items()},
            "files": record,
            "source": "Hugging Face",
        }
        prov_path = self.bronze_dir / "provenance.json"
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
        return self.bronze_dir

    def provenance(self) -> Dict[str, str]:
        with open(self.bronze_dir / "provenance.json", "r", encoding="utf-8") as f:
            return json.load(f)
