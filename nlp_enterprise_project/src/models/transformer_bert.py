"""DistilBERT (Transformer) fine-tuning module."""
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


class TransformerTrainer:
    """Configurable trainer for Hugging Face Transformer classifiers."""

    def __init__(self, config: Dict[str, Any], logger, device: torch.device):
        self.cfg = config
        self.logger = logger
        self.device = device
        self.id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["tokenizer"]["model_name"])

    def tokenize_function(self, examples):
        tcfg = self.cfg["tokenizer"]
        return self.tokenizer(
            examples["text"],
            truncation=tcfg["truncation"],
            padding=tcfg["padding"],
            max_length=tcfg["max_length"],
        )

    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg["model"]["model_name"],
            num_labels=self.cfg["model"]["num_labels"],
            id2label=self.id2label,
            label2id=self.label2id,
            attn_implementation=self.cfg["tokenizer"]["attn_implementation"],
        ).to(self.device)

    def train(self, dataset, tokenized_datasets) -> "Trainer":
        """Fine-tune the model and return the Trainer."""
        tcfg = self.cfg["training"]
        log_dir = Path(self.cfg["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(self.cfg["logging"]["artifacts_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1_macro": f1_score(labels, predictions, average="macro"),
            }

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=tcfg["batch_size"],
            per_device_eval_batch_size=tcfg["eval_batch_size"],
            num_train_epochs=tcfg["num_epochs"],
            learning_rate=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
            warmup_ratio=tcfg.get("warmup_ratio", 0.0),
            eval_strategy=tcfg["eval_strategy"],
            save_strategy=tcfg["save_strategy"],
            logging_strategy="epoch",
            logging_steps=tcfg.get("logging_steps", 50),
            load_best_model_at_end=tcfg["load_best_model_at_end"],
            metric_for_best_model=tcfg["metric_for_best_model"],
            report_to="none",
            seed=self.cfg["seed"],
            fp16=torch.cuda.is_available() and tcfg.get("fp16", False),
            gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=tcfg["early_stopping_patience"])],
        )

        t_start = time.time()
        self.logger.log("Starting Transformer fine-tuning...")
        trainer.train()
        train_time = time.time() - t_start
        self.logger.log_metric("train_time_seconds", train_time)

        trainer.save_model(str(output_dir / "best"))
        self.tokenizer.save_pretrained(str(output_dir / "best"))
        self.trainer = trainer
        return trainer

    def evaluate(self, tokenized_datasets, output_dir: str) -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        eval_results = self.trainer.evaluate()
        clean_eval = {}
        for k, v in eval_results.items():
            try:
                clean_eval[k] = float(v)
            except Exception:
                clean_eval[k] = str(v)
        self.logger.log_metrics(clean_eval)

        predictions_output = self.trainer.predict(tokenized_datasets["test"])
        y_true = predictions_output.label_ids
        y_pred = np.argmax(predictions_output.predictions, axis=-1)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        report = classification_report(y_true, y_pred,
                                       target_names=list(self.id2label.values()),
                                       digits=self.cfg["evaluation"]["report_digits"],
                                       output_dict=True)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(self.id2label.values()),
                    yticklabels=list(self.id2label.values()))
        plt.title("Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(out / "confusion_matrix.png")
        plt.close()

        metrics = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "classification_report": report,
            "raw_eval_results": clean_eval,
        }
        with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics, y_true, y_pred, predictions_output.predictions

    def plot_sample_predictions(self, dataset, tokenized_datasets,
                                predictions_output, output_dir: str, n: int = 5):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        y_true = predictions_output.label_ids
        y_pred = np.argmax(predictions_output.predictions, axis=-1)
        texts = dataset["test"]["text"]
        labels = dataset["test"]["label_text"]
        rows = []
        for i in range(min(n, len(texts))):
            confidence = float(torch.softmax(torch.tensor(predictions_output.predictions[i]), dim=0).max())
            rows.append({
                "text": texts[i][:120],
                "true": labels[i],
                "predicted": self.id2label[y_pred[i]],
                "confidence": round(confidence, 4),
            })
        with open(out / "sample_predictions.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return rows

    def attention_visualization(self, text: str, output_dir: str):
        """Generate an attention heatmap for a sample text if supported."""
        try:
            import seaborn as sns
            self.model.eval()
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=self.cfg["tokenizer"]["max_length"])
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            attentions = torch.stack(outputs.attentions)
            avg_attention = attentions.mean(dim=[0, 2])[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            seq_len = len(tokens)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                avg_attention[:seq_len, :seq_len],
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="viridis",
            )
            plt.title("Average Attention (all layers/heads)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            plt.savefig(out / "attention_heatmap.png")
            plt.close()
            return str(out / "attention_heatmap.png")
        except Exception as e:
            self.logger.log(f"Attention heatmap generation failed: {e}")
            return None
