"""Fine-tuned-transformer reward model: pairwise-trained DistilBERT cross-encoder.

Unlike the lr_* family (a linear probe over frozen sentence-transformer embeddings),
this lets a model attend jointly over (sql_context, question, sql) in one forward
pass, rather than embedding each piece separately and combining via dot product.

Encoding: a single DistilBERT sequence built manually, since DistilBertTokenizer's
native API only supports 2-segment (text, text_pair) inputs —
    [CLS] sql_context [SEP] question [SEP] sql [SEP]
(DistilBERT has no token_type_ids parameter at all — no segment embeddings — so the
3-segment structure is communicated purely via [SEP] positions + attention_mask.)
sql_context is joined with "\n" (matching the convention already used by
sql_agent.py / lr_model_context.py) and is the only segment ever truncated, from the
end, to fit max_length — question and sql are never truncated except as a last-resort
if they alone exceed the budget (should be rare; see distilbert_preflight.py).

Training objective mirrors LRRewardModel.fit's exact anti-positional-bias pattern: one
random.Random(seed), consumed sequentially per sql_bad candidate, decides which side
of the pair is "good" (~50/50) so the head can't shortcut on position. Unlike
GBMRewardModel (pointwise, because differencing *features* phi(A)-phi(B) into one
nonlinear classifier couples A and B into a joint, not-necessarily-transitive
comparator), this stays genuinely pairwise: score_A and score_B are each an
independent forward pass over a single (context, question, sql) input, and only the
*loss* combines the two resulting scalars —
    loss = BCEWithLogitsLoss(score_A - score_B, label)
— which preserves a well-defined, always-transitive per-candidate score() (sorting
real numbers is always transitive), so this is safe despite DistilBERT being
nonlinear; GBM's problem was about differencing inputs before a joint classifier, not
about nonlinearity per se.
"""
from __future__ import annotations

import random
import sys
import time
import warnings
from pathlib import Path
from typing import NamedTuple, Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from walt.rm.data.base import Example
from walt.rm.model.base import BaseRewardModel, group_split

DEFAULT_MODEL_NAME = "distilbert-base-cased"


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class _Pair(NamedTuple):
    question: str
    sql_context: tuple[str, ...]
    sql_a: str
    sql_b: str
    label: float
    reason: str


def build_pairs(examples: list[Example], seed: int) -> list[_Pair]:
    """Same anti-positional-bias pattern as LRRewardModel.fit: one RNG, created once,
    consumed sequentially per (example, sql_bad) in iteration order."""
    rng = random.Random(seed)
    pairs: list[_Pair] = []
    for ex in examples:
        for bad in ex.sql_bad:
            if rng.random() < 0.5:
                sql_a, sql_b, label = ex.sql_good, bad.sql, 1.0
            else:
                sql_a, sql_b, label = bad.sql, ex.sql_good, 0.0
            pairs.append(_Pair(ex.question, ex.sql_context, sql_a, sql_b, label, bad.reason))
    return pairs


def encode_ids(tokenizer, context_text: str, question: str, sql: str, max_length: int | None) -> list[int]:
    """Builds [CLS] context [SEP] question [SEP] sql [SEP] token ids. If max_length is
    given, truncates context_text's tokens from the end to fit (question/sql are only
    truncated, from the end, as a last-resort if they alone exceed the budget). If
    max_length is None, returns the full untruncated sequence (used by
    distilbert_preflight.py to measure what truncation would need to do)."""
    context_ids = tokenizer.encode(context_text, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    sql_ids = tokenizer.encode(sql, add_special_tokens=False)
    if max_length is not None:
        budget = max_length - 4 - len(question_ids) - len(sql_ids)  # 4 = [CLS] + 3x[SEP]
        if budget < 0:
            sql_ids = sql_ids[: max(len(sql_ids) + budget, 0)]  # emergency last-resort truncation
            context_ids = []
        else:
            context_ids = context_ids[:budget]
    return [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id] + question_ids + [
        tokenizer.sep_token_id
    ] + sql_ids + [tokenizer.sep_token_id]


def pad_batch(id_lists: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in id_lists)
    input_ids = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids.to(device), attention_mask.to(device)


class _ScoringNet(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0, :]
        return self.head(cls).squeeze(-1)


class DistilBertRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        seed: int = 42,
        device: str | None = None,
        max_length: int = 512,
        learning_rate: float = 2e-5,
        num_epochs: int = 15,
        batch_size: int = 8,
        grad_accum_steps: int = 2,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        val_fraction: float = 0.1,
        early_stop_patience: int = 3,
        early_stop_metric: str = "pairwise_accuracy",  # "pairwise_accuracy" (higher better) | "loss" (lower better)
        eval_batch_size: int = 32,
    ):
        if early_stop_metric not in ("pairwise_accuracy", "loss"):
            raise ValueError(f"early_stop_metric must be 'pairwise_accuracy' or 'loss', got {early_stop_metric!r}")
        self.model_name = model_name
        self.seed = seed
        self.device = resolve_device(device)
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.early_stop_patience = early_stop_patience
        self.early_stop_metric = early_stop_metric
        self.eval_batch_size = eval_batch_size

        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.net = _ScoringNet(model_name).to(self.device)
        self._loss_fn = nn.BCEWithLogitsLoss()
        self._fitted = False
        self._score_cache: dict[tuple[str, str, str], float] = {}
        self.fit_info: dict = {}

    def warm_cache(self, examples: list[Example]) -> None:
        """No-op: train.py calls this unconditionally for every model (an LR-specific
        embedding pre-warm optimization); DistilBERT has no separable embedding step to
        pre-warm, so this exists purely for interface compatibility."""

    def _encode_ids(self, context_text: str, question: str, sql: str) -> list[int]:
        return encode_ids(self.tokenizer, context_text, question, sql, self.max_length)

    def score(self, question: str, sql: str, sql_context: tuple[str, ...] = ()) -> float:
        if not self._fitted:
            raise RuntimeError("DistilBertRewardModel.score() called before fit()/load()")
        key = (question, "\n".join(sql_context), sql)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        self.net.eval()
        with torch.no_grad():
            ids = self._encode_ids("\n".join(sql_context), question, sql)
            input_ids, attention_mask = pad_batch([ids], self.tokenizer.pad_token_id, self.device)
            logit = self.net(input_ids, attention_mask)
        val = float(logit.item())
        self._score_cache[key] = val
        return val

    def _pretokenize(self, pairs: list[_Pair]) -> list[tuple[list[int], list[int], float]]:
        out = []
        for p in pairs:
            ctx_text = "\n".join(p.sql_context)
            ids_a = self._encode_ids(ctx_text, p.question, p.sql_a)
            ids_b = self._encode_ids(ctx_text, p.question, p.sql_b)
            out.append((ids_a, ids_b, p.label))
        return out

    def _forward_pairs(self, batch: list[tuple[list[int], list[int], float]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids_a = [b[0] for b in batch]
        ids_b = [b[1] for b in batch]
        labels = torch.tensor([b[2] for b in batch], dtype=torch.float, device=self.device)
        input_ids, attention_mask = pad_batch(ids_a + ids_b, self.tokenizer.pad_token_id, self.device)
        logits = self.net(input_ids, attention_mask)
        n = len(batch)
        return logits[:n], logits[n:], labels

    def _eval_pairs(self, pretokenized: list[tuple[list[int], list[int], float]]) -> tuple[float, float]:
        self.net.eval()
        total_loss, n_correct, n = 0.0, 0, len(pretokenized)
        with torch.no_grad():
            for start in range(0, n, self.eval_batch_size):
                chunk = pretokenized[start : start + self.eval_batch_size]
                logits_a, logits_b, labels = self._forward_pairs(chunk)
                loss = self._loss_fn(logits_a - logits_b, labels)
                total_loss += loss.item() * len(chunk)
                pred_a_wins = logits_a > logits_b
                actual_a_wins = labels == 1
                n_correct += int((pred_a_wins == actual_a_wins).sum().item())
        return total_loss / n, n_correct / n

    def fit(self, train_examples: list[Example]) -> None:
        fit_start = time.perf_counter()
        self._score_cache = {}

        internal_train, internal_val = group_split(train_examples, test_size=self.val_fraction, seed=self.seed + 1)
        train_pairs = build_pairs(internal_train, seed=self.seed)
        val_pairs = build_pairs(internal_val, seed=self.seed)
        train_pre = self._pretokenize(train_pairs)
        val_pre = self._pretokenize(val_pairs)

        n_pos = sum(1 for p in train_pairs if p.label == 1.0)
        micro_batches_per_epoch = (len(train_pre) + self.batch_size - 1) // self.batch_size
        optimizer_steps_per_epoch = max(1, (micro_batches_per_epoch + self.grad_accum_steps - 1) // self.grad_accum_steps)
        total_steps = optimizer_steps_per_epoch * self.num_epochs
        warmup_steps = int(self.warmup_ratio * total_steps)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        higher_better = self.early_stop_metric == "pairwise_accuracy"
        best_metric = float("-inf") if higher_better else float("inf")
        best_state = None
        best_epoch = -1
        patience_counter = 0
        early_stopped = False
        train_loss_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch_seconds_list = [], [], [], []
        mps_fallback_ops: set[str] = set()
        epoch_rng = random.Random(self.seed + 1000)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for epoch in range(1, self.num_epochs + 1):
                epoch_start = time.perf_counter()
                self.net.train()
                order = list(range(len(train_pre)))
                epoch_rng.shuffle(order)
                running_loss, n_seen, step = 0.0, 0, 0
                optimizer.zero_grad()
                for step, start in enumerate(range(0, len(order), self.batch_size), start=1):
                    micro_idx = order[start : start + self.batch_size]
                    batch = [train_pre[i] for i in micro_idx]
                    logits_a, logits_b, labels = self._forward_pairs(batch)
                    loss = self._loss_fn(logits_a - logits_b, labels)
                    (loss / self.grad_accum_steps).backward()
                    running_loss += loss.item() * len(batch)
                    n_seen += len(batch)
                    if step % self.grad_accum_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                if step % self.grad_accum_steps != 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss = running_loss / n_seen
                val_loss, val_acc = self._eval_pairs(val_pre)
                epoch_seconds = time.perf_counter() - epoch_start
                train_loss_per_epoch.append(round(train_loss, 5))
                val_loss_per_epoch.append(round(val_loss, 5))
                val_acc_per_epoch.append(round(val_acc, 5))
                epoch_seconds_list.append(round(epoch_seconds, 3))
                print(
                    f"  epoch {epoch}/{self.num_epochs}: train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_pairwise_accuracy={val_acc:.4f} ({epoch_seconds:.1f}s)"
                )

                current = val_acc if higher_better else val_loss
                improved = current > best_metric if higher_better else current < best_metric
                if improved:
                    best_metric, best_epoch, patience_counter = current, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        early_stopped = True
                        break
            for w in caught:
                msg = str(w.message)
                if "not currently supported on the MPS backend" in msg:
                    mps_fallback_ops.add(msg.split("'")[1] if "'" in msg else msg)

        assert best_state is not None
        self.net.load_state_dict(best_state)
        self.net.to(self.device)
        self._fitted = True

        raw_rss = _peak_rss_bytes()
        peak_rss_mb = raw_rss / (1024 * 1024) if sys.platform == "darwin" else raw_rss / 1024
        peak_mps_mb = None
        if self.device.type == "mps":
            try:
                peak_mps_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
            except Exception:
                peak_mps_mb = None

        self.fit_info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "seed": self.seed,
            "max_length": self.max_length,
            "n_train_examples": len(internal_train),
            "n_val_examples": len(internal_val),
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "label_balance": {"a_is_good": n_pos, "b_is_good": len(train_pairs) - n_pos},
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "effective_batch_size_pairs": self.batch_size * self.grad_accum_steps,
            "warmup_ratio": self.warmup_ratio,
            "num_epochs_cap": self.num_epochs,
            "num_epochs_run": len(train_loss_per_epoch),
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "early_stop_metric": self.early_stop_metric,
            "early_stop_patience": self.early_stop_patience,
            "train_loss_per_epoch": train_loss_per_epoch,
            "val_loss_per_epoch": val_loss_per_epoch,
            "val_pairwise_accuracy_per_epoch": val_acc_per_epoch,
            "epoch_seconds": epoch_seconds_list,
            "mps_fallback_ops": sorted(mps_fallback_ops),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "peak_mps_allocated_mb": round(peak_mps_mb, 1) if peak_mps_mb is not None else None,
            "total_train_seconds": round(time.perf_counter() - fit_start, 3),
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
            "model_name": self.model_name,
            "seed": self.seed,
            "max_length": self.max_length,
            "hyperparams": {
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "warmup_ratio": self.warmup_ratio,
                "weight_decay": self.weight_decay,
                "val_fraction": self.val_fraction,
                "early_stop_patience": self.early_stop_patience,
                "early_stop_metric": self.early_stop_metric,
                "eval_batch_size": self.eval_batch_size,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DistilBertRewardModel":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            model_name=payload["model_name"],
            seed=payload["seed"],
            device=device,
            max_length=payload["max_length"],
            **payload["hyperparams"],
        )
        model.net.load_state_dict(payload["state_dict"])
        model.net.to(model.device)
        model._fitted = True
        return model


def _peak_rss_bytes() -> int:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
