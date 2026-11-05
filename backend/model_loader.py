from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
except Exception:  # pragma: no cover - optional dependency may be absent
    DistilBertForSequenceClassification = None
    DistilBertTokenizer = None

try:
    from torch_geometric.nn import GCNConv
except Exception:  # pragma: no cover - optional dependency may be absent
    GCNConv = None


class ContentClassifier(nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class PropagationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("torch_geometric is required for propagation inference")
        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        embedding = x.mean(dim=0)
        output = self.lin(embedding)
        return output, embedding


class HybridFusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(832, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ModelRegistry:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_root = Path(__file__).resolve().parents[1]
        self.backend_dir = Path(__file__).resolve().parent
        self.content_model = None
        self.content_tokenizer = None
        self.propagation_model = None
        self.hybrid_model = None
        self.content_error = None
        self.propagation_error = None
        self.hybrid_error = None

    def load_all_models(self) -> Dict[str, Any]:
        self.content_model, self.content_tokenizer, self.content_error = self._load_content_model()
        self.propagation_model, self.propagation_error = self._load_propagation_model()
        self.hybrid_model, self.hybrid_error = self._load_hybrid_model()
        return self.model_status()

    def model_status(self) -> Dict[str, Any]:
        content_ready = self.content_model is not None and self.content_error is None
        propagation_ready = self.propagation_model is not None and self.propagation_error is None
        hybrid_ready = self.hybrid_model is not None and self.hybrid_error is None and content_ready and propagation_ready
        hybrid_error = self.hybrid_error
        if hybrid_error is None and not hybrid_ready:
            hybrid_error = "hybrid inference requires the content, propagation, and fusion models to be available"
        return {
            "models": {
                "content": content_ready,
                "propagation": propagation_ready,
                "hybrid": hybrid_ready,
            },
            "errors": {
                name: error
                for name, error in (
                    ("content", self.content_error),
                    ("propagation", self.propagation_error),
                    ("hybrid", hybrid_error),
                )
                if error is not None
            },
            "device": str(self.device),
        }

    def _resolve_model_path(self, *names: str) -> Optional[Path]:
        search_roots = [self.repo_root, self.backend_dir, self.repo_root / "models", self.backend_dir / "models"]
        for root in search_roots:
            for name in names:
                candidate = root / name
                if candidate.exists():
                    return candidate
        return None

    def _load_content_model(self) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
        if DistilBertForSequenceClassification is None or DistilBertTokenizer is None:
            return None, None, "transformers is not installed"

        model_path = self._resolve_model_path(
            "content_analysis_model.pth",
            "content_model.pth",
            "distilbert_model.pth",
            "pheme_distilbert_model",
        )
        if model_path is None:
            return None, None, "content model checkpoint was not found; expected content_analysis_model.pth, content_model.pth, or pheme_distilbert_model/"

        try:
            tokenizer_path = str(model_path) if model_path.is_dir() else "distilbert-base-uncased"
            tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        except Exception as exc:  # pragma: no cover - network may be unavailable
            return None, None, f"could not download DistilBERT tokenizer: {exc}"

        try:
            if model_path.suffix == ".pth":
                base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
                state = torch.load(model_path, map_location=self.device)
                if isinstance(state, dict) and any(k.startswith("model") or k.startswith("distilbert") for k in state.keys()):
                    base_model.load_state_dict(state)
                else:
                    base_model.load_state_dict(state)
            else:
                base_model = DistilBertForSequenceClassification.from_pretrained(str(model_path), num_labels=2)
            base_model = base_model.to(self.device)
            base_model.eval()
            return base_model, tokenizer, None
        except Exception as exc:
            return None, None, f"content model could not be loaded: {exc}"

    def _load_propagation_model(self) -> Tuple[Optional[Any], Optional[str]]:
        model_path = self._resolve_model_path("propagation_model.pth", "gnn_propagation_model.pth")
        if model_path is None:
            return None, "propagation model checkpoint was not found; expected one of: propagation_model.pth, gnn_propagation_model.pth"
        try:
            model = PropagationModel().to(self.device)
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            return model, None
        except Exception as exc:
            return None, f"propagation model could not be loaded: {exc}"

    def _load_hybrid_model(self) -> Tuple[Optional[Any], Optional[str]]:
        model_path = self._resolve_model_path("hybrid_model.pth", "hybrid_fusion_model.pth")
        if model_path is None:
            return None, "hybrid model checkpoint was not found; expected one of: hybrid_model.pth, hybrid_fusion_model.pth"
        try:
            model = HybridFusionModel().to(self.device)
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            return model, None
        except Exception as exc:
            return None, f"hybrid model could not be loaded: {exc}"

    def predict_content(self, text: str) -> Dict[str, Any]:
        if self.content_model is None or self.content_tokenizer is None:
            raise RuntimeError(self.content_error or "content model is unavailable")
        inputs = self.content_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            logits = self.content_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probabilities = torch.softmax(logits[0], dim=0).cpu().tolist()
        confidence = max(probabilities)
        label_idx = int(probabilities.index(max(probabilities)))
        label = "Real" if label_idx == 0 else "Fake"
        return {
            "label": label,
            "confidence": round(float(confidence) * 100, 2),
            "probabilities": {
                "Real": round(float(probabilities[0]), 4),
                "Fake": round(float(probabilities[1]), 4),
            },
        }

    def _build_graph_tensor(self, payload: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = payload.get("nodes") or []
        edges = payload.get("edges") or []
        node_features = payload.get("node_features")

        if node_features is None:
            features = []
            for index, _ in enumerate(nodes):
                features.append([float(index), float(len(nodes) - index)])
            if not features and nodes:
                features = [[1.0, 1.0] for _ in nodes]
            if not features:
                features = [[1.0, 0.0]]
        else:
            features = [list(item) for item in node_features]
            if len(features) < len(nodes):
                features.extend([[0.0, 0.0]] * (len(nodes) - len(features)))

        if len(features) < 1:
            features = [[1.0, 0.0]]

        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        if x.shape[1] < 2:
            x = torch.cat([x, torch.zeros((x.shape[0], 2 - x.shape[1]), dtype=torch.float32, device=self.device)], dim=1)
        if x.shape[1] > 2:
            x = x[:, :2]

        edge_index: list[list[int]] = []
        for edge in edges:
            if isinstance(edge, dict):
                source = edge.get("source", edge.get("from", edge.get("src")))
                target = edge.get("target", edge.get("to", edge.get("dst")))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                source, target = edge[0], edge[1]
            else:
                continue
            if source is None or target is None:
                continue
            edge_index.append([int(source), int(target)])

        if not edge_index:
            if len(nodes) > 1:
                edge_index = [[0, 1]] if len(nodes) >= 2 else [[0, 0]]
            else:
                edge_index = [[0, 0]]

        edge_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        return x, edge_tensor

    def predict_propagation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.propagation_model is None:
            raise RuntimeError(self.propagation_error or "propagation model is unavailable")
        x, edge_index = self._build_graph_tensor(payload)
        with torch.no_grad():
            logits, _ = self.propagation_model(x, edge_index)
        probabilities = torch.softmax(logits, dim=0).cpu().tolist()
        confidence = max(probabilities)
        label_idx = int(probabilities.index(max(probabilities)))
        label = "Real" if label_idx == 0 else "Fake"
        return {
            "label": label,
            "confidence": round(float(confidence) * 100, 2),
            "probabilities": {
                "Real": round(float(probabilities[0]), 4),
                "Fake": round(float(probabilities[1]), 4),
            },
        }

    def _content_embedding(self, text: str) -> torch.Tensor:
        if self.content_model is None or self.content_tokenizer is None:
            raise RuntimeError(self.content_error or "content model is unavailable")
        tokens = self.content_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.content_model.distilbert(
                input_ids=tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device),
            ).last_hidden_state
        return hidden.mean(dim=1).squeeze(0).cpu()

    def predict_hybrid(self, text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.hybrid_model is None:
            raise RuntimeError(self.hybrid_error or "hybrid model is unavailable")
        content_result = self.predict_content(text)
        propagation_result = self.predict_propagation(payload)
        content_embedding = self._content_embedding(text)
        x, edge_index = self._build_graph_tensor(payload)
        with torch.no_grad():
            _, propagation_embedding = self.propagation_model(x, edge_index)
        combined = torch.cat([content_embedding.to(self.device), propagation_embedding.to(self.device)], dim=0).unsqueeze(0)
        logits = self.hybrid_model(combined)
        probabilities = torch.softmax(logits[0], dim=0).cpu().tolist()
        confidence = max(probabilities)
        label_idx = int(probabilities.index(max(probabilities)))
        label = "Real" if label_idx == 0 else "Fake"
        return {
            "label": label,
            "confidence": round(float(confidence) * 100, 2),
            "probabilities": {
                "Real": round(float(probabilities[0]), 4),
                "Fake": round(float(probabilities[1]), 4),
            },
            "component_scores": {
                "content": content_result,
                "propagation": propagation_result,
            },
        }


registry = ModelRegistry()
