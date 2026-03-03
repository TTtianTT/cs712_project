#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import pickle
import zipfile
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Basic IO
# -----------------------------
def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_sentence(line: str) -> List[str]:
    return line.split()


def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)


# -----------------------------
# Relative helpers (optional)
# -----------------------------
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def is_hex_token(tok: str) -> bool:
    return bool(_HEX_RE.match(tok))


def token_to_int(tok: str) -> Optional[int]:
    if tok in ("MASK", "<PAD>", "<UNK>"):
        return None
    try:
        if tok.isdigit():
            return int(tok, 10)
        if is_hex_token(tok):
            return int(tok, 16)
        return None
    except Exception:
        return None


def infer_k_bits_from_tokens(tokens: List[str]) -> int:
    hex_lens = []
    bit_lens = []
    for t in tokens:
        if t in ("MASK", "<PAD>", "<UNK>"):
            continue
        v = token_to_int(t)
        if v is None:
            continue
        if is_hex_token(t):
            hex_lens.append(len(t))
        bit_lens.append(max(1, v.bit_length()))
    if hex_lens:
        from collections import Counter as C
        L = C(hex_lens).most_common(1)[0][0]
        return max(1, 4 * L)
    if bit_lens:
        from collections import Counter as C
        bl = C(bit_lens).most_common(1)[0][0]
        return max(1, ((bl + 7) // 8) * 8)
    return 32


def hamming_distance_kbits(a: int, b: int, k: int) -> int:
    x = (a ^ b)
    if k < 1024:
        x &= (1 << k) - 1
    return x.bit_count()


# -----------------------------
# N-gram model
# -----------------------------
BOS2 = "<BOS2>"
BOS1 = "<BOS1>"
EOS1 = "<EOS1>"
EOS2 = "<EOS2>"
SPECIAL_MASK = "MASK"


@dataclass
class NgramModel:
    eligible_set: set
    alpha: float
    # maps: key -> Counter(center_token)
    ctx4: Dict[Tuple[str, str, str, str], Counter]
    ctx3L: Dict[Tuple[str, str, str], Counter]   # (l2,l1,r1)
    ctx3R: Dict[Tuple[str, str, str], Counter]   # (l1,r1,r2)
    ctx2: Dict[Tuple[str, str], Counter]         # (l1,r1)
    left1: Dict[str, Counter]                    # l1
    right1: Dict[str, Counter]                   # r1
    unigram: Counter
    uni_total: int
    V: int
    k_bits: int


def _add_count(map_: Dict[Any, Counter], key: Any, center: str):
    c = map_.get(key)
    if c is None:
        c = Counter()
        map_[key] = c
    c[center] += 1


def _counter_total(c: Counter) -> int:
    return sum(c.values())


def build_ngram_model(
    train_path: str,
    eligible_path: str,
    max_len: int = 512,
    alpha: float = 0.1,
) -> NgramModel:
    elig = []
    for ln in read_lines(eligible_path):
        elig.extend(ln.split())
    eligible_set = set(elig)
    V = len(eligible_set)

    ctx4: Dict[Tuple[str, str, str, str], Counter] = {}
    ctx3L: Dict[Tuple[str, str, str], Counter] = {}
    ctx3R: Dict[Tuple[str, str, str], Counter] = {}
    ctx2: Dict[Tuple[str, str], Counter] = {}
    left1: Dict[str, Counter] = {}
    right1: Dict[str, Counter] = {}
    unigram = Counter()

    for ln in read_lines(train_path):
        toks = parse_sentence(ln)[:max_len]
        if not toks:
            continue
        ext = [BOS2, BOS1] + toks + [EOS1, EOS2]
        for i, center in enumerate(toks):
            if center not in eligible_set:
                continue
            j = i + 2
            l2, l1t, r1t, r2 = ext[j - 2], ext[j - 1], ext[j + 1], ext[j + 2]

            _add_count(ctx4, (l2, l1t, r1t, r2), center)
            _add_count(ctx3L, (l2, l1t, r1t), center)
            _add_count(ctx3R, (l1t, r1t, r2), center)
            _add_count(ctx2, (l1t, r1t), center)
            _add_count(left1, l1t, center)
            _add_count(right1, r1t, center)
            unigram[center] += 1

    uni_total = sum(unigram.values())
    k_bits = infer_k_bits_from_tokens(list(eligible_set))
    return NgramModel(
        eligible_set=eligible_set,
        alpha=alpha,
        ctx4=ctx4,
        ctx3L=ctx3L,
        ctx3R=ctx3R,
        ctx2=ctx2,
        left1=left1,
        right1=right1,
        unigram=unigram,
        uni_total=uni_total,
        V=V,
        k_bits=k_bits,
    )


def save_ngram(model: NgramModel, out_path: str):
    ensure_dir(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_ngram(path: str) -> NgramModel:
    with open(path, "rb") as f:
        return pickle.load(f)


def ngram_logp(
    model: NgramModel,
    token: str,
    counter: Optional[Counter],
    total: Optional[int],
    use_unigram: bool = False,
) -> float:
    if token not in model.eligible_set:
        return float("-inf")
    alpha = model.alpha
    V = model.V

    if use_unigram or counter is None or total is None or total <= 0:
        c = model.unigram.get(token, 0)
        denom = model.uni_total + alpha * V
        return math.log(c + alpha) - math.log(denom)

    c = counter.get(token, 0)
    denom = total + alpha * V
    return math.log(c + alpha) - math.log(denom)


def get_best_counter_with_backoff(
    model: NgramModel,
    tokens_with_mask: List[str],
    mask_pos: int,
) -> Tuple[Optional[Counter], Optional[int], str]:
    ext = [BOS2, BOS1] + tokens_with_mask + [EOS1, EOS2]
    j = mask_pos + 2
    l2, l1t, r1t, r2 = ext[j - 2], ext[j - 1], ext[j + 1], ext[j + 2]

    c = model.ctx4.get((l2, l1t, r1t, r2))
    if c:
        return c, _counter_total(c), "ctx4"
    c = model.ctx3L.get((l2, l1t, r1t))
    if c:
        return c, _counter_total(c), "ctx3L"
    c = model.ctx3R.get((l1t, r1t, r2))
    if c:
        return c, _counter_total(c), "ctx3R"
    c = model.ctx2.get((l1t, r1t))
    if c:
        return c, _counter_total(c), "ctx2"
    c = model.left1.get(l1t)
    if c:
        return c, _counter_total(c), "left1"
    c = model.right1.get(r1t)
    if c:
        return c, _counter_total(c), "right1"
    return None, None, "unigram"


def ngram_predict_one(model: NgramModel, toks: List[str]) -> str:
    try:
        mp = toks.index(SPECIAL_MASK)
    except ValueError:
        return "<UNK>"
    counter, _, level = get_best_counter_with_backoff(model, toks, mp)
    if counter is None:
        return model.unigram.most_common(1)[0][0] if model.unigram else "<UNK>"
    return counter.most_common(1)[0][0]


def ngram_topk(model: NgramModel, toks: List[str], k: int = 500) -> Tuple[List[Tuple[str, float]], str]:
    mp = toks.index(SPECIAL_MASK)
    counter, total, level = get_best_counter_with_backoff(model, toks, mp)

    # choose candidate counter
    use_uni = False
    cand = counter
    tot = total
    if cand is None:
        cand = model.unigram
        tot = model.uni_total
        use_uni = True

    items: List[Tuple[str, float]] = []
    for tok, _cnt in cand.most_common(k):
        lp = ngram_logp(model, tok, counter=cand if not use_uni else None, total=tot if not use_uni else None, use_unigram=use_uni)
        items.append((tok, lp))
    return items, level


def eval_ngram_on_labeled_tsv(model: NgramModel, labeled_tsv: str, relative: bool = False) -> Dict[str, float]:
    total = 0
    correct = 0
    rel_sum = 0.0

    for ln in read_lines(labeled_tsv):
        if "\t" not in ln:
            continue
        sent, label = ln.split("\t")[0], ln.split("\t")[-1]
        toks = sent.split()
        pred = ngram_predict_one(model, toks)
        total += 1
        if pred == label:
            correct += 1

        if relative:
            pi = token_to_int(pred)
            ti = token_to_int(label)
            if pi is None or ti is None:
                continue
            hd = hamming_distance_kbits(pi, ti, model.k_bits)
            rel = 1.0 - (hd / float(model.k_bits))
            if rel < 0:
                rel = 0.0
            rel_sum += rel

    out = {"abs_acc": (correct / total if total else 0.0)}
    if relative:
        out["rel_acc"] = (rel_sum / total if total else 0.0)
    return out


# -----------------------------
# Neural model (for hybrid)
# -----------------------------
class Vocab:
    def __init__(self, tokens: List[str]):
        uniq = []
        seen = set()
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        self.id2tok = uniq
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    def __len__(self):
        return len(self.id2tok)

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.tok2id.get("<UNK>", 1)
        return [self.tok2id.get(t, unk) for t in toks]

    @property
    def pad_id(self):
        return self.tok2id.get("<PAD>", 0)

    @property
    def unk_id(self):
        return self.tok2id.get("<UNK>", 1)

    @property
    def mask_id(self):
        return self.tok2id.get("MASK", 2)


class MaskPredictor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, dropout: float, max_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        key_padding_mask = (attention_mask == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.ln(h)
        return self.lm_head(h)


class MaskedInputDataset(Dataset):
    """accept plain masked lines OR 'masked<TAB>label' (label ignored)"""
    def __init__(self, path: str, vocab: Vocab, max_len: int = 512):
        self.vocab = vocab
        self.max_len = max_len
        self.samples: List[List[str]] = []
        for ln in read_lines(path):
            sent = ln.split("\t")[0]  # drop label if exists
            toks = sent.split()[:max_len]
            self.samples.append(toks)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        toks = self.samples[idx]
        ids = self.vocab.encode(toks)
        try:
            mp = toks.index("MASK")
        except ValueError:
            mp = -1
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "mask_pos": mp, "toks": toks}


class LabeledMaskedDataset(Dataset):
    """'masked<TAB>label'"""
    def __init__(self, labeled_tsv: str, vocab: Vocab, max_len: int = 512):
        self.vocab = vocab
        self.max_len = max_len
        self.samples: List[Tuple[List[str], str]] = []
        for ln in read_lines(labeled_tsv):
            if "\t" not in ln:
                continue
            sent, label = ln.split("\t")[0], ln.split("\t")[-1]
            toks = sent.split()[:max_len]
            self.samples.append((toks, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        toks, label = self.samples[idx]
        ids = self.vocab.encode(toks)
        try:
            mp = toks.index("MASK")
        except ValueError:
            mp = -1
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "mask_pos": mp, "toks": toks, "label": label}


def collate_masked(batch):
    maxlen = max(x["input_ids"].shape[0] for x in batch)
    input_ids, attn, mask_pos, toks = [], [], [], []
    labels = []
    has_label = ("label" in batch[0])
    for x in batch:
        L = x["input_ids"].shape[0]
        pad = maxlen - L
        input_ids.append(F.pad(x["input_ids"], (0, pad), value=0))
        attn.append(torch.cat([torch.ones(L, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        mask_pos.append(x["mask_pos"])
        toks.append(x["toks"])
        if has_label:
            labels.append(x["label"])
    out = {
        "input_ids": torch.stack(input_ids, 0),
        "attention_mask": torch.stack(attn, 0),
        "mask_pos": torch.tensor(mask_pos, dtype=torch.long),
        "toks": toks,
    }
    if has_label:
        out["labels"] = labels
    return out


def load_neural(meta_path: str, ckpt_path: str, device: str):
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    vocab = Vocab(meta["vocab"])
    model = MaskPredictor(
        vocab_size=len(vocab),
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        n_layers=meta["n_layers"],
        dropout=meta["dropout"],
        max_len=meta["max_len"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    eligible_set = set(meta["eligible_tokens"])
    eligible_ids = torch.tensor(
        sorted({vocab.tok2id[t] for t in eligible_set if t in vocab.tok2id}),
        dtype=torch.long,
        device=device,
    )
    # id2elig map
    id2elig = torch.full((len(vocab),), -1, dtype=torch.long, device=device)
    id2elig[eligible_ids] = torch.arange(len(eligible_ids), device=device, dtype=torch.long)
    return meta, vocab, model, eligible_set, eligible_ids, id2elig


# -----------------------------
# Hybrid: score = nn_logprob(nn_temp) + lambda * ngram_logp(backoff)
# -----------------------------
@torch.no_grad()
def hybrid_infer_one(
    *,
    ngram: NgramModel,
    toks: List[str],
    vocab: Vocab,
    scores_elig_at_mp: torch.Tensor,  # [V_elig] logits (not softmaxed)
    eligible_ids: torch.Tensor,       # [V_elig] full vocab ids
    id2elig: torch.Tensor,            # [V_full] -> elig idx or -1
    lambda_ng: float,
    nn_temp: float,
    k_nn: int,
    k_ng: int,
) -> str:
    try:
        mp = toks.index(SPECIAL_MASK)
    except ValueError:
        return vocab.id2tok[vocab.unk_id]

    # neural logprobs (temperature calibrated)
    if nn_temp <= 0:
        nn_temp = 1.0
    nn_logp_vec = F.log_softmax(scores_elig_at_mp / nn_temp, dim=-1)  # [V_elig]

    # candidate set from neural topk + ngram topk
    cand_set = set()

    # neural topK
    k_take = min(k_nn, nn_logp_vec.numel())
    topv, topi = torch.topk(nn_logp_vec, k_take)
    for si in topi.tolist():
        full_id = int(eligible_ids[si].item())
        tok = vocab.id2tok[full_id]
        if tok in ngram.eligible_set:
            cand_set.add(tok)

    # ngram topK
    ng_items, _level = ngram_topk(ngram, toks, k=k_ng)
    for tok, _lp in ng_items:
        if tok in ngram.eligible_set:
            cand_set.add(tok)

    if not cand_set:
        return ngram.unigram.most_common(1)[0][0] if ngram.unigram else vocab.id2tok[vocab.unk_id]

    # best backoff counter for this instance
    counter, total, level = get_best_counter_with_backoff(ngram, toks, mp)
    use_uni = (level == "unigram")

    best_tok = None
    best_score = -1e30

    for tok in cand_set:
        vid = vocab.tok2id.get(tok, None)
        if vid is None:
            continue
        elig_idx = int(id2elig[vid].item())
        if elig_idx < 0:
            continue
        nn_lp = float(nn_logp_vec[elig_idx].item())

        ng_lp = ngram_logp(
            ngram,
            tok,
            counter=None if use_uni else counter,
            total=None if use_uni else total,
            use_unigram=use_uni,
        )

        sc = nn_lp + lambda_ng * ng_lp
        if sc > best_score:
            best_score = sc
            best_tok = tok

    return best_tok if best_tok is not None else vocab.id2tok[vocab.unk_id]


@torch.no_grad()
def hybrid_predict_file(
    *,
    ngram: NgramModel,
    meta_path: str,
    ckpt_path: str,
    input_path: str,
    out_pred: str,
    out_zip: Optional[str],
    batch_size: int,
    lambda_ng: float,
    nn_temp: float,
    k_nn: int,
    k_ng: int,
    max_len_override: int,
    cpu: bool,
):
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    meta, vocab, model, eligible_set_neural, eligible_ids, id2elig = load_neural(meta_path, ckpt_path, device)

    max_len = int(meta["max_len"])
    if max_len_override and max_len_override > 0:
        max_len = max_len_override

    ds = MaskedInputDataset(input_path, vocab=vocab, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_masked)

    preds: List[str] = []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask_pos = batch["mask_pos"].to(device)
        toks_list = batch["toks"]

        logits = model(input_ids, attn)  # [B,L,V_full]
        # eligible slice
        logits_elig = logits.index_select(-1, eligible_ids)  # [B,L,V_elig]

        B = logits.shape[0]
        for i in range(B):
            mp = int(mask_pos[i].item())
            if mp < 0:
                preds.append(vocab.id2tok[vocab.unk_id])
                continue
            pred = hybrid_infer_one(
                ngram=ngram,
                toks=toks_list[i],
                vocab=vocab,
                scores_elig_at_mp=logits_elig[i, mp],
                eligible_ids=eligible_ids,
                id2elig=id2elig,
                lambda_ng=lambda_ng,
                nn_temp=nn_temp,
                k_nn=k_nn,
                k_ng=k_ng,
            )
            preds.append(pred)

    ensure_dir(out_pred)
    with open(out_pred, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")

    if out_zip:
        ensure_dir(out_zip)
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_pred, arcname=os.path.basename(out_pred))

    print(f"[hybrid_predict] wrote: {out_pred}")
    if out_zip:
        print(f"[hybrid_predict] zipped: {out_zip}")


@torch.no_grad()
def hybrid_eval_file(
    *,
    ngram: NgramModel,
    meta_path: str,
    ckpt_path: str,
    labeled_tsv: str,
    batch_size: int,
    lambda_ng: float,
    nn_temp: float,
    k_nn: int,
    k_ng: int,
    max_len_override: int,
    cpu: bool,
    relative: bool,
):
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    meta, vocab, model, eligible_set_neural, eligible_ids, id2elig = load_neural(meta_path, ckpt_path, device)

    max_len = int(meta["max_len"])
    if max_len_override and max_len_override > 0:
        max_len = max_len_override

    ds = LabeledMaskedDataset(labeled_tsv, vocab=vocab, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_masked)

    total = 0
    correct = 0
    rel_sum = 0.0
    k_bits = ngram.k_bits

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask_pos = batch["mask_pos"].to(device)
        toks_list = batch["toks"]
        labels = batch["labels"]

        logits = model(input_ids, attn)
        logits_elig = logits.index_select(-1, eligible_ids)

        B = logits.shape[0]
        for i in range(B):
            mp = int(mask_pos[i].item())
            if mp < 0:
                continue
            pred = hybrid_infer_one(
                ngram=ngram,
                toks=toks_list[i],
                vocab=vocab,
                scores_elig_at_mp=logits_elig[i, mp],
                eligible_ids=eligible_ids,
                id2elig=id2elig,
                lambda_ng=lambda_ng,
                nn_temp=nn_temp,
                k_nn=k_nn,
                k_ng=k_ng,
            )
            truth = labels[i]
            total += 1
            if pred == truth:
                correct += 1

            if relative:
                pi = token_to_int(pred)
                ti = token_to_int(truth)
                if pi is None or ti is None:
                    continue
                hd = hamming_distance_kbits(pi, ti, k_bits)
                rel = 1.0 - (hd / float(k_bits))
                if rel < 0:
                    rel = 0.0
                rel_sum += rel

    abs_acc = correct / total if total else 0.0
    if relative:
        rel_acc = rel_sum / total if total else 0.0
        print(f"[hybrid_eval] abs_acc={abs_acc:.4f} rel_acc={rel_acc:.4f} (lambda={lambda_ng}, nn_temp={nn_temp})")
    else:
        print(f"[hybrid_eval] abs_acc={abs_acc:.4f} (lambda={lambda_ng}, nn_temp={nn_temp})")


# -----------------------------
# Pure ngram predict
# -----------------------------
def ngram_predict_file(model: NgramModel, input_path: str, out_pred: str, out_zip: Optional[str]):
    preds = []
    for ln in read_lines(input_path):
        sent = ln.split("\t")[0]
        toks = sent.split()
        preds.append(ngram_predict_one(model, toks))

    ensure_dir(out_pred)
    with open(out_pred, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")

    if out_zip:
        ensure_dir(out_zip)
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_pred, arcname=os.path.basename(out_pred))

    print(f"[ngram_predict] wrote: {out_pred}")
    if out_zip:
        print(f"[ngram_predict] zipped: {out_zip}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_b = sub.add_parser("build")
    ap_b.add_argument("--train", required=True)
    ap_b.add_argument("--eligible", required=True)
    ap_b.add_argument("--out", required=True)
    ap_b.add_argument("--max_len", type=int, default=512)
    ap_b.add_argument("--alpha", type=float, default=0.1)

    ap_e = sub.add_parser("eval")
    ap_e.add_argument("--model", required=True)
    ap_e.add_argument("--labeled_tsv", required=True)
    ap_e.add_argument("--relative", action="store_true")

    ap_p = sub.add_parser("predict")
    ap_p.add_argument("--model", required=True)
    ap_p.add_argument("--input", required=True)
    ap_p.add_argument("--out_pred", required=True)
    ap_p.add_argument("--out_zip", default="")

    ap_hp = sub.add_parser("hybrid_predict")
    ap_hp.add_argument("--ngram_model", required=True)
    ap_hp.add_argument("--meta", required=True)
    ap_hp.add_argument("--ckpt", required=True)
    ap_hp.add_argument("--input", required=True)
    ap_hp.add_argument("--out_pred", required=True)
    ap_hp.add_argument("--out_zip", default="")
    ap_hp.add_argument("--batch_size", type=int, default=256)
    ap_hp.add_argument("--lambda_ng", type=float, default=1.0)
    ap_hp.add_argument("--nn_temp", type=float, default=1.0)
    ap_hp.add_argument("--k_nn", type=int, default=300)
    ap_hp.add_argument("--k_ng", type=int, default=500)
    ap_hp.add_argument("--max_len", type=int, default=0)
    ap_hp.add_argument("--cpu", action="store_true")

    ap_he = sub.add_parser("hybrid_eval")
    ap_he.add_argument("--ngram_model", required=True)
    ap_he.add_argument("--meta", required=True)
    ap_he.add_argument("--ckpt", required=True)
    ap_he.add_argument("--labeled_tsv", required=True)
    ap_he.add_argument("--batch_size", type=int, default=256)
    ap_he.add_argument("--lambda_ng", type=float, default=1.0)
    ap_he.add_argument("--nn_temp", type=float, default=1.0)
    ap_he.add_argument("--k_nn", type=int, default=300)
    ap_he.add_argument("--k_ng", type=int, default=500)
    ap_he.add_argument("--max_len", type=int, default=0)
    ap_he.add_argument("--cpu", action="store_true")
    ap_he.add_argument("--relative", action="store_true")

    args = ap.parse_args()

    if args.cmd == "build":
        m = build_ngram_model(args.train, args.eligible, max_len=args.max_len, alpha=args.alpha)
        save_ngram(m, args.out)
        print(f"[build] saved ngram model to {args.out}")
        print(f"[build] V_eligible={m.V} uni_total={m.uni_total} k_bits={m.k_bits}")
        print(f"[build] ctx4={len(m.ctx4)} ctx3L={len(m.ctx3L)} ctx3R={len(m.ctx3R)} ctx2={len(m.ctx2)} left1={len(m.left1)} right1={len(m.right1)}")

    elif args.cmd == "eval":
        m = load_ngram(args.model)
        res = eval_ngram_on_labeled_tsv(m, args.labeled_tsv, relative=args.relative)
        if args.relative:
            print(f"[eval_ngram] abs_acc={res['abs_acc']:.4f} rel_acc={res['rel_acc']:.4f}")
        else:
            print(f"[eval_ngram] abs_acc={res['abs_acc']:.4f}")

    elif args.cmd == "predict":
        m = load_ngram(args.model)
        out_zip = args.out_zip if args.out_zip else None
        ngram_predict_file(m, args.input, args.out_pred, out_zip)

    elif args.cmd == "hybrid_predict":
        m = load_ngram(args.ngram_model)
        out_zip = args.out_zip if args.out_zip else None
        hybrid_predict_file(
            ngram=m,
            meta_path=args.meta,
            ckpt_path=args.ckpt,
            input_path=args.input,
            out_pred=args.out_pred,
            out_zip=out_zip,
            batch_size=args.batch_size,
            lambda_ng=args.lambda_ng,
            nn_temp=args.nn_temp,
            k_nn=args.k_nn,
            k_ng=args.k_ng,
            max_len_override=args.max_len,
            cpu=args.cpu,
        )

    else:
        m = load_ngram(args.ngram_model)
        hybrid_eval_file(
            ngram=m,
            meta_path=args.meta,
            ckpt_path=args.ckpt,
            labeled_tsv=args.labeled_tsv,
            batch_size=args.batch_size,
            lambda_ng=args.lambda_ng,
            nn_temp=args.nn_temp,
            k_nn=args.k_nn,
            k_ng=args.k_ng,
            max_len_override=args.max_len,
            cpu=args.cpu,
            relative=args.relative,
        )


if __name__ == "__main__":
    main()
