# ngram_hybrid_v2.py

`ngram_hybrid_v2.py` is the final inference utility used to combine a stronger count-based baseline with the neural masked-token model.

It does four things:

1. Build an improved n-gram model from the train split.
2. Evaluate the pure n-gram baseline on labeled dev data.
3. Evaluate the hybrid model on labeled dev data so `lambda_ng` and `nn_temp` can be tuned.
4. Generate final validation predictions and an optional zip submission.

## What the script is doing

### 1. Stronger n-gram backoff

The script builds a count-based model over eligible tokens only. For each masked position, it stores counts under these backoff contexts:

- `ctx4`: `(l2, l1, r1, r2)`
- `ctx3L`: `(l2, l1, r1)`
- `ctx3R`: `(l1, r1, r2)`
- `ctx2`: `(l1, r1)`
- `left1`: `l1`
- `right1`: `r1`
- `unigram`

At inference time it backs off in exactly that order. The extra `ctx3L` and `ctx3R` layers are the main difference from the earlier version and usually improve hit rate compared with jumping directly from `ctx4` to `ctx2`.

All n-gram probabilities use add-`alpha` smoothing over the eligible vocabulary.

### 2. Temperature-calibrated neural branch

For the neural model, the script loads:

- `meta.json`
- `ckpt_best.pt` or another checkpoint

It takes the logits at the `MASK` position, restricts them to the eligible vocabulary, and converts them to calibrated log-probabilities with:

```text
log_softmax(logits / nn_temp)
```

`nn_temp > 1` makes the neural distribution flatter, which often makes hybrid fusion more stable.

### 3. Hybrid scoring

The final hybrid score is:

```text
score(token) = nn_logprob(token; nn_temp) + lambda_ng * ngram_logprob(token)
```

- `lambda_ng = 0` means pure neural inference.
- Larger `lambda_ng` makes the result closer to pure n-gram.

To keep inference efficient, the script only scores candidates from:

- neural top-`k_nn`
- n-gram top-`k_ng`

It takes the union of those candidates and returns the highest-scoring token.

### 4. Direct dev evaluation

The key practical improvement is `hybrid_eval`: it evaluates the hybrid model directly on `dev_labeled.tsv`, so you can tune the fusion on labeled dev instead of guessing from validation outputs.

It reports:

- `abs_acc`
- optional `rel_acc`

## Commands

### Build

Build the stronger n-gram model:

```bash
python3 ngram_hybrid_v2.py build \
  --train outputs_maskdiff_2stage/auto_dev/train_split.txt \
  --eligible data/vocab_eligible.txt \
  --out outputs_maskdiff_2stage/ngram_v2.pkl \
  --max_len 512 \
  --alpha 0.1
```

### Eval

Evaluate the pure n-gram baseline on labeled dev:

```bash
python3 ngram_hybrid_v2.py eval \
  --model outputs_maskdiff_2stage/ngram_v2.pkl \
  --labeled_tsv outputs_maskdiff_2stage/auto_dev/dev_labeled.tsv
```

Add `--relative` if you also want relative accuracy.

### Hybrid Eval

Evaluate the hybrid model on labeled dev:

```bash
python3 ngram_hybrid_v2.py hybrid_eval \
  --ngram_model outputs_maskdiff_2stage/ngram_v2.pkl \
  --meta outputs_maskdiff_2stage/meta.json \
  --ckpt outputs_maskdiff_2stage/ckpt_best.pt \
  --labeled_tsv outputs_maskdiff_2stage/auto_dev/dev_labeled.tsv \
  --lambda_ng 1.5 \
  --nn_temp 1.2 \
  --k_nn 400 \
  --k_ng 800
```

Typical dev sweep:

```bash
for lam in 0 0.5 1 1.5 2 3 4; do
  for temp in 0.8 1.0 1.2 1.5 2.0; do
    python3 ngram_hybrid_v2.py hybrid_eval \
      --ngram_model outputs_maskdiff_2stage/ngram_v2.pkl \
      --meta outputs_maskdiff_2stage/meta.json \
      --ckpt outputs_maskdiff_2stage/ckpt_best.pt \
      --labeled_tsv outputs_maskdiff_2stage/auto_dev/dev_labeled.tsv \
      --lambda_ng "$lam" \
      --nn_temp "$temp" \
      --k_nn 400 \
      --k_ng 800
  done
done
```

### Hybrid Predict

Use the best dev setting to generate the final submission:

```bash
python3 ngram_hybrid_v2.py hybrid_predict \
  --ngram_model outputs_maskdiff_2stage/ngram_v2.pkl \
  --meta outputs_maskdiff_2stage/meta.json \
  --ckpt outputs_maskdiff_2stage/ckpt_best.pt \
  --input data/validation.txt \
  --out_pred submission/pred.txt \
  --out_zip submission/pred.zip \
  --lambda_ng <best_lambda> \
  --nn_temp <best_temp> \
  --k_nn 400 \
  --k_ng 800
```

## Recommended workflow

1. Run `build` to rebuild the stronger n-gram model with `ctx3` backoff.
2. Run `eval` to check the pure n-gram baseline on dev.
3. Run `hybrid_eval` to sweep `lambda_ng` and `nn_temp` on `dev_labeled.tsv`.
4. Run `hybrid_predict` with the best dev setting to produce `pred.txt` and `pred.zip`.

## Inputs and outputs

Main inputs:

- training split text for n-gram counts
- eligible vocabulary
- labeled dev set
- neural `meta.json`
- neural checkpoint
- validation text

Main outputs:

- `ngram_v2.pkl`
- dev accuracy printed to stdout
- `submission/pred.txt`
- `submission/pred.zip`
