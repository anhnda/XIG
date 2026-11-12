"""
HuggingFace glue for the token experiments (transformers + datasets required).

Suggested fine-tuned checkpoints (pass with --model):
    SST-2   distilbert-base-uncased-finetuned-sst-2-english
            textattack/bert-base-uncased-SST-2
            textattack/roberta-base-SST-2
    AG News textattack/bert-base-uncased-ag-news
    IMDB    textattack/bert-base-uncased-imdb
Check that the label order of the checkpoint matches the dataset if you use
gold labels; the experiments only use the predicted class.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from pathattr.tokens import TokenInput

DATASETS = {
    #  name      (path, config, split, text field)
    "sst2":    ("glue", "sst2", "validation", "sentence"),
    "ag_news": ("ag_news", None, "test", "text"),
    "imdb":    ("imdb", None, "test", "text"),
}


def load_texts(name: str, n: int, seed: int = 0) -> list[tuple[int, str]]:
    from datasets import load_dataset
    path, cfg, split, field = DATASETS[name]
    ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
    idx = list(range(len(ds)))
    if n and n < len(idx):
        idx = sorted(random.Random(seed).sample(idx, n))
    return [(i, ds[i][field]) for i in idx]


def load_model(name: str, device: torch.device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if tok.mask_token_id is None:
        raise ValueError(f"{name}: tokenizer has no mask token")
    return tok, model


class EmbTarget(nn.Module):
    """embeddings (B, L, h) → logit (or softmax prob) of one class, shape (B,).
    attention_mask = 1 everywhere: [MASK] tokens are attended, never padded."""

    def __init__(self, model: nn.Module, target: int, prob: bool = False):
        super().__init__()
        self.model, self.target, self.prob = model, int(target), prob

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        am = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
        logits = self.model(inputs_embeds=emb, attention_mask=am).logits
        if self.prob:
            return torch.softmax(logits, dim=-1)[:, self.target]
        return logits[:, self.target]


@torch.no_grad()
def prepare(tok, model, text: str, max_len: int, device: torch.device) -> tuple[TokenInput, int, float]:
    """Tokenise, build E / E_b ([MASK] at every non-special position), predict."""
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    special = tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True)
    pos = torch.tensor([i for i, s in enumerate(special) if s == 0], dtype=torch.long)
    ids_b = ids.clone()
    ids_b[0, pos.to(device)] = tok.mask_token_id
    emb = model.get_input_embeddings()
    E, Eb = emb(ids).detach(), emb(ids_b).detach()
    am = torch.ones_like(ids)
    probs = torch.softmax(model(input_ids=ids, attention_mask=am).logits, dim=-1)[0]
    cls = int(probs.argmax())
    return TokenInput(E=E, Eb=Eb, pos=pos), cls, float(probs[cls])
