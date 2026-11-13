#!/usr/bin/env python
"""Run both enterprise pipelines sequentially."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scripts = [
    ("Project 1: Word2Vec + BiLSTM", "run_project1.py"),
    ("Project 2: DistilBERT",        "run_project2.py"),
]

print("=" * 80)
print("ENTERPRISE NLP PROJECT: Running both end-to-end pipelines")
print("=" * 80)

for title, script in scripts:
    print(f"\n[{title}]")
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\nFAILED: {script}")
        sys.exit(result.returncode)

print("\n" + "=" * 80)
print("BOTH PIPELINES COMPLETED SUCCESSFULLY")
print("Run APIs:")
print("  uvicorn src.api.word2vec_lstm_api:app --port 8001")
print("  uvicorn src.api.transformer_api:app --port 8002")
print("Run UI:")
print("  streamlit run src/ui/demo.py")
print("=" * 80)
