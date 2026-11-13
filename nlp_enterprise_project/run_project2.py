#!/usr/bin/env python
"""
Production-level end-to-end pipeline for Project 2:
DistilBERT fine-tuning on AG News with Bronze/Silver/Gold data layers.

Usage:
  cd nlp_enterprise_project
  python run_project2.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from utils.config import Config
from utils.logger import ExperimentLogger
from utils.common import set_seed, get_device, make_dirs
from data.bronze import BronzeDataLoader
from data.silver import SilverPreprocessor
from features.gold import GoldBuilder
from models.transformer_bert import TransformerTrainer


def main():
    cfg = Config(str(ROOT / "configs" / "transformer.yaml"))
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
    logger.log("PROJECT 2: DistilBERT Fine-Tuning | Bronze/Silver/Gold Pipeline")
    logger.log("=" * 80)
    logger.log_params({
        "dataset": cfg_dict["data"]["dataset_name"],
        "model": cfg_dict["model"]["model_name"],
        "max_length": cfg_dict["tokenizer"]["max_length"],
        "batch_size": cfg_dict["training"]["batch_size"],
        "epochs": cfg_dict["training"]["num_epochs"],
        "learning_rate": cfg_dict["training"]["learning_rate"],
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
        stopwords_source=cfg_dict.get("preprocessing", {}).get("stopwords_source", "nltk")
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
    # Gold layer: Hugging Face dataset with cleaned text
    # ------------------------------------------------------------------
    logger.log("\n[GOLD] Preparing tokenized dataset...")
    # Add cleaned text to raw dataset for consistency
    text_map = {(row["text_raw"], row["label"]): row["text_clean"] for _, row in df.iterrows()}
    # Use raw HF dataset with text_clean replaced for tokenization
    def add_clean_text(examples, split):
        examples["text"] = examples[cfg_dict["data"]["text_column"]]
        return examples
    dataset = raw_dataset

    # ------------------------------------------------------------------
    # Tokenize and load model
    # ------------------------------------------------------------------
    trainer = TransformerTrainer(cfg_dict, logger, device)
    logger.log("\n[TOKENIZER] Loading tokenizer...")
    trainer.load_tokenizer()

    logger.log("\n[MODEL] Loading DistilBERT...")
    trainer.load_model()
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    logger.log(f"Total parameters:     {total:,}")
    logger.log(f"Trainable parameters: {trainable:,}")

    logger.log("\n[TOKENIZE] Tokenizing dataset...")
    tokenized_datasets = dataset.map(trainer.tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "label_text"])
    tokenized_datasets.set_format("torch")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.log("\n[TRAINING] Fine-tuning DistilBERT...")
    trainer.train(dataset, tokenized_datasets)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    logger.log("\n[EVALUATION] Evaluating on test set...")
    metrics, y_true, y_pred, preds = trainer.evaluate(tokenized_datasets, str(metrics_dir))
    logger.log(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    logger.log(f"Test F1-macro:  {metrics['f1_macro']:.4f}")
    logger.log_artifact(str(metrics_dir / "test_metrics.json"), "test_metrics")
    logger.log_artifact(str(metrics_dir / "confusion_matrix.png"), "confusion_matrix")

    # ------------------------------------------------------------------
    # Sample predictions & interpretability
    # ------------------------------------------------------------------
    logger.log("\n[INSIGHTS] Saving sample predictions...")
    rows = trainer.plot_sample_predictions(dataset, tokenized_datasets,
                                           type("PredictionOutput", (), {
                                               "label_ids": y_true,
                                               "predictions": preds,
                                           })(), str(metrics_dir), n=5)
    logger.log("Sample predictions:")
    for row in rows:
        logger.log(f"  True={row['true']:10s} Pred={row['predicted']:10s} Conf={row['confidence']}")

    logger.log("\n[INTERPRETABILITY] Generating attention heatmap...")
    first_text = dataset["test"]["text"][0]
    heatmap_path = trainer.attention_visualization(first_text, str(figures_dir))
    if heatmap_path:
        logger.log_artifact(heatmap_path, "attention_heatmap")

    logger.save_summary()
    logger.log("\n" + "=" * 80)
    logger.log("PROJECT 2 COMPLETE")
    logger.log(f"  Final Accuracy : {metrics['accuracy']:.4f}")
    logger.log(f"  Final F1-macro : {metrics['f1_macro']:.4f}")
    logger.log(f"  Artifacts saved in: {artifacts_dir}")
    logger.log("=" * 80)


if __name__ == "__main__":
    main()
