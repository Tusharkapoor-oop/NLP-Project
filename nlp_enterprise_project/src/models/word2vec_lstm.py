"""Word2Vec + BiLSTM training and evaluation module."""
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec


class TextClassifier(nn.Module):
    """PyTorch BiLSTM text classifier with frozen pretrained embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_classes: int, embedding_matrix: np.ndarray,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # h_n shape: (num_layers*2, batch, hidden)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        hidden = torch.cat((h_forward, h_backward), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class Word2VecLSTMTrainer:
    """End-to-end trainer for Word2Vec + BiLSTM."""

    def __init__(self, config: Any, logger, device: torch.device):
        self.cfg = config
        self.logger = logger
        self.device = device

    def train_word2vec(self, tokens_series) -> Word2Vec:
        """Train or load a corpus Word2Vec model."""
        model = Word2Vec(
            sentences=tokens_series.tolist(),
            vector_size=self.cfg["word2vec"]["vector_size"],
            window=self.cfg["word2vec"]["window"],
            min_count=self.cfg["word2vec"]["min_count"],
            sg=self.cfg["word2vec"]["sg"],
            workers=self.cfg["word2vec"]["workers"],
            seed=self.cfg["seed"],
            epochs=self.cfg["word2vec"]["epochs"],
        )
        return model

    def build_model(self, vocab_size: int, embedding_matrix: np.ndarray, num_classes: int):
        self.model = TextClassifier(
            vocab_size=vocab_size,
            embed_dim=self.cfg["preprocessing"]["embedding_dim"],
            hidden_dim=self.cfg["model"]["hidden_dim"],
            num_classes=num_classes,
            embedding_matrix=embedding_matrix,
            num_layers=self.cfg["model"]["num_layers"],
            dropout=self.cfg["model"]["dropout"],
        ).to(self.device)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        batch_size = self.cfg["training"]["batch_size"]
        train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg["training"]["learning_rate"],
            weight_decay=self.cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=self.cfg["training"]["min_lr"]
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience = self.cfg["training"]["patience"]
        epochs_no_improve = 0

        t_start = time.time()
        for epoch in range(self.cfg["training"]["epochs"]):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            val_loss, val_acc = self._evaluate_epoch(val_loader, criterion)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            self.logger.log_metric("train_loss", train_loss, step=epoch)
            self.logger.log_metric("val_loss", val_loss, step=epoch)
            self.logger.log_metric("val_accuracy", val_acc, step=epoch)
            self.logger.log(
                f"Epoch {epoch + 1:02d}/{self.cfg['training']['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    self.logger.log("Early stopping triggered")
                    break

        self.train_time = time.time() - t_start
        self.logger.log_metric("train_time_seconds", self.train_time)
        self.model.load_state_dict(self.best_state)
        return history

    def _evaluate_epoch(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                total_loss += loss.item() * xb.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return total_loss / total, correct / total

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 label_classes, output_dir: str) -> Dict[str, Any]:
        batch_size = self.cfg["training"]["batch_size"]
        test_ds = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        self.model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                outputs = self.model(xb)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        acc = accuracy_score(all_true, all_preds)
        f1_macro = f1_score(all_true, all_preds, average="macro")
        report = classification_report(all_true, all_preds, target_names=label_classes,
                                       digits=self.cfg["evaluation"]["report_digits"], output_dict=True)

        # confusion matrix
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(all_true, all_preds)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_classes, yticklabels=label_classes)
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
        }
        with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    def plot_history(self, history: Dict[str, list], output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        epochs_range = range(1, len(history["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs_range, history["train_loss"], label="Train loss")
        axes[0].plot(epochs_range, history["val_loss"], label="Val loss")
        axes[0].set_title("Loss over epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].plot(epochs_range, history["val_acc"], label="Val accuracy")
        axes[1].set_title("Accuracy over epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out / "training_history.png")
        plt.close()
