#!/usr/bin/env python
"""
Production-level end-to-end pipeline for Project 1:
Word2Vec + BiLSTM on BBC News with Bronze/Silver/Gold data layers.

Usage:
  cd nlp_enterprise_project
  python run_project1.py
"""
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

# Ensure imports from src/ work when script is run directly
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils.config import Config
from utils.logger import ExperimentLogger
from utils.common import set_seed, get_device, make_dirs
from data.bronze import BronzeDataLoader
from data.silver import SilverPreprocessor
from features.gold import GoldBuilder
from models.word2vec_lstm import Word2VecLSTMTrainer


# Ensure imported numpy/torch are clearly used
_ = np, torch


def main():
    cfg = Config(str(ROOT / "configs" / "word2vec_lstm.yaml"))
    cfg_dict = cfg.to_dict()
    set_seed(cfg_dict["seed"])

    log_dir = ROOT / cfg_dict["logging"]["log_dir"]
    metrics_dir = ROOT / cfg_dict["logging"]["metrics_dir"]
    figures_dir = ROOT / cfg_dict["logging"]["figures_dir"]
    artifacts_dir = ROOT / cfg_dict["logging"]["artifacts_dir"]
    make_dirs([log_dir, metrics_dir, figures_dir, artifacts_dir,
               ROOT / cfg_dict["data"]["bronze_dir"],
               ROOT / cfg_dict["data"]["silver_dir"],
               ROOT / cfg_dict["data"]["gold_dir"]])

    logger = ExperimentLogger(str(log_dir), cfg_dict["experiment_name"], seed=cfg_dict["seed"])
    logger.log("=" * 80)
    logger.log("PROJECT 1: Word2Vec + BiLSTM | Bronze/Silver/Gold Pipeline")
    logger.log("=" * 80)
    logger.log_params({
        "dataset": cfg_dict["data"]["dataset_name"],
        "max_words": cfg_dict["preprocessing"]["max_num_words"],
        "max_seq_len": cfg_dict["preprocessing"]["max_seq_len"],
        "embedding_dim": cfg_dict["preprocessing"]["embedding_dim"],
        "lstm_hidden": cfg_dict["model"]["hidden_dim"],
    })

    device = get_device()

    # ------------------------------------------------------------------
    # Bronze layer: ingest raw data
    # ------------------------------------------------------------------
    logger.log("\n[BRONZE] Loading raw dataset...")
    bronze_dir = ROOT / cfg_dict["data"]["bronze_dir"]
    loader = BronzeDataLoader(cfg_dict["data"]["dataset_name"], str(bronze_dir))
    raw_dataset = loader.load()
    loader.save(raw_dataset)
    logger.log(f"Bronze dataset size: train={len(raw_dataset['train'])}, test={len(raw_dataset['test'])}")

    # ------------------------------------------------------------------
    # Silver layer: clean and standardize
    # ------------------------------------------------------------------
    logger.log("\n[SILVER] Preprocessing text...")
    preprocessor = SilverPreprocessor(
        stopwords_source=cfg_dict["preprocessing"]["stopwords_source"]
    )
    silver_dir = ROOT / cfg_dict["data"]["silver_dir"]
    df = preprocessor.transform_dataset(
        raw_dataset,
        text_column=cfg_dict["data"]["text_column"],
        label_column=cfg_dict["data"]["label_column"],
        output_dir=str(silver_dir),
    )
    logger.log(f"Silver records: {len(df)}")
    logger.log_param("silver_vocab_size", int(df["tokens"].explode().nunique()))

    # ------------------------------------------------------------------
    # Gold layer: sequences and splits
    # ------------------------------------------------------------------
    logger.log("\n[GOLD] Building sequences and embedding matrix...")
    gold_dir = ROOT / cfg_dict["data"]["gold_dir"]
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold = GoldBuilder(
        max_num_words=cfg_dict["preprocessing"]["max_num_words"],
        max_seq_len=cfg_dict["preprocessing"]["max_seq_len"],
        embedding_dim=cfg_dict["preprocessing"]["embedding_dim"],
    )
    X_train, X_val, X_test, y_train, y_val, y_test = gold.transform(
        df, label_column="label", text_column="text_clean", seed=cfg_dict["seed"]
    )
    logger.log(f"Data splits -> train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logger.log(f"Number of classes: {len(gold.label_encoder.classes_)} -> {gold.label_encoder.classes_}")

    # ------------------------------------------------------------------
    # Train Word2Vec
    # ------------------------------------------------------------------
    logger.log("\n[EMBEDDINGS] Training Word2Vec...")
    trainer = Word2VecLSTMTrainer(cfg_dict, logger, device)
    w2v_model = trainer.train_word2vec(df["tokens"])
    w2v_path = artifacts_dir / "word2vec_bbc.model"
    w2v_model.save(str(w2v_path))
    logger.log_artifact(str(w2v_path), "word2vec_model")

    embedding_matrix = gold.build_embedding_matrix(str(w2v_path))
    logger.log(f"Embedding matrix shape: {embedding_matrix.shape}")
    np.save(gold_dir / "embedding_matrix.npy", embedding_matrix)
    gold.save(str(gold_dir))

    # ------------------------------------------------------------------
    # Build and train BiLSTM
    # ------------------------------------------------------------------
    logger.log("\n[MODEL] Building BiLSTM...")
    trainer.build_model(len(gold.word_to_idx), embedding_matrix, len(gold.label_encoder.classes_))
    logger.log(str(trainer.model))

    logger.log("\n[TRAINING] Training BiLSTM...")
    history = trainer.train(X_train, y_train, X_val, y_val)
    trainer.plot_history(history, str(figures_dir))
    logger.log_artifact(str(figures_dir / "training_history.png"), "training_history")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    logger.log("\n[EVALUATION] Evaluating on test set...")
    metrics = trainer.evaluate(
        X_test, y_test, gold.label_encoder.classes_, str(metrics_dir)
    )
    logger.log(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    logger.log(f"Test F1-macro:  {metrics['f1_macro']:.4f}")
    logger.log_artifact(str(metrics_dir / "test_metrics.json"), "test_metrics")
    logger.log_artifact(str(metrics_dir / "confusion_matrix.png"), "confusion_matrix")

    # ------------------------------------------------------------------
    # Persist final model + artifacts
    # ------------------------------------------------------------------
    logger.log("\n[DEPLOY] Saving final artifacts...")
    torch_path = artifacts_dir / "bilstm_bbc_classifier.pth"
    torch.save(trainer.model.state_dict(), str(torch_path))
    logger.log_artifact(str(torch_path), "pytorch_model_state_dict")

    # Inference-ready metadata
    meta = {
        "vocab": gold.word_to_idx,
        "max_seq_len": cfg_dict["preprocessing"]["max_seq_len"],
        "embedding_dim": cfg_dict["preprocessing"]["embedding_dim"],
        "label_classes": gold.label_encoder.classes_.tolist(),
        "metrics": metrics,
    }
    with open(artifacts_dir / "deployment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.log_artifact(str(artifacts_dir / "deployment_metadata.json"), "deployment_metadata")

    logger.save_summary()
    logger.log("\n" + "=" * 80)
    logger.log("PROJECT 1 COMPLETE")
    logger.log(f"  Final Accuracy : {metrics['accuracy']:.4f}")
    logger.log(f"  Final F1-macro : {metrics['f1_macro']:.4f}")
    logger.log(f"  Artifacts saved in: {artifacts_dir}")
    logger.log("=" * 80)


if __name__ == "__main__":
    main()
