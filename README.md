# LSH Masked Token Prediction

This codebase implements:

- `Baseline 1`: unigram prior over eligible tokens
- `Baseline 2`: windowed co-occurrence scorer with add-k smoothing
- `Main model`: tiny Transformer MLM with eligible-only softmax
- `Bit-aware enhancement`: optional 32-bit auxiliary head
- `Fusion inference`: neural log-probabilities plus co-occurrence scores and optional bit scores
- `Best dev setup`: `n-gram v2` backoff model plus temperature-calibrated neural hybrid
- `DDP training`: `torchrun` on `cuda:0,1,2,3`

## Files

- `data_utils.py`: file IO, token validation, deterministic train/dev split, bit/Hamming utilities, datasets
- `baselines.py`: unigram and co-occurrence baselines
- `models.py`: Transformer encoder MLM, bit head, and scoring helpers
- `train.py`: train/dev split, baseline evaluation, dense masked-label training, DDP, checkpointing
- `predict.py`: baseline, transformer, or fusion inference plus `pred.txt` and `pred.zip`
- `ngram_hybrid.py`: first hybrid implementation with ctx4 -> ctx2 backoff
- `ngram_hybrid_v2.py`: improved n-gram backoff, hybrid dev evaluation, and submission generation

## Data assumptions

- Tokens are always 8-char lowercase hex strings, so `k = 32` bits.
- Validation lines contain exactly one literal `MASK`.
- Predictions are always constrained to the eligible vocabulary in `vocab_eligible.txt`.

## Method

The strongest setup in this repo is a two-part hybrid rather than a larger neural model alone.

1. `Neural scorer`
   A Transformer masked-language model is trained with eligible-only prediction. The strongest training setup so far is the deterministic sentence-level split plus `all_single` masking, which gives one masked example per eligible position.
2. `n-gram v2 scorer`
   A separate count-based model is built on the same train split and stores counts for the masked token under progressively weaker contexts:
   - `ctx4`: `(l2, l1, r1, r2)`
   - `ctx3L`: `(l2, l1, r1)`
   - `ctx3R`: `(l1, r1, r2)`
   - `ctx2`: `(l1, r1)`
   - `left1`: `l1`
   - `right1`: `r1`
   - `unigram`
   Backoff follows that exact order. This extra `ctx3` layer improves hit rate compared with dropping directly from `ctx4` to `ctx2`. Each level uses add-`alpha` smoothing over the eligible vocabulary.
3. `Hybrid scoring`
   At inference time, the neural branch is calibrated with `log_softmax(logits / nn_temp)`. The final score for a candidate token is:

```text
score(token) = nn_logprob(token; nn_temp) + lambda_ng * ngram_logprob(token)
```

   Candidate tokens are restricted to the union of neural top-`k_nn` and n-gram top-`k_ng`, then the best-scoring token is selected.
4. `Dev tuning`
   `ngram_hybrid_v2.py hybrid_eval` evaluates directly on `dev_labeled.tsv`, so `lambda_ng`, `nn_temp`, and optionally `alpha` can be tuned on the labeled dev split before generating the validation submission.

## Masking strategies

Training supports:

- `single`: one eligible position per sentence, one example per sentence
- `all_single`: one example per eligible position, still single-mask per example
- `ratio`: one multi-mask example per sentence, masking `max(1, round(mask_ratio * num_eligible_positions))` eligible positions

Relevant flags:

- `--mask_strategy {single,all_single,ratio}`
- `--mask_ratio 0.1` or `--mask_ratio 0.3` when `--mask_strategy=ratio`

The train/dev split stays sentence-level and deterministic.
Model selection defaults to `--selection_metric harmonic_mean`, which matches the final leaderboard objective.

## Train

Recommended `all_single` run without the bit head:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --data_dir /mnt/data/zailongtian/workspace/cs712_project/data \
  --output_dir outputs/exp_all_single_nb0 \
  --ddp 1 \
  --num_gpus 4 \
  --batch_size_per_gpu 64 \
  --eval_batch_size 256 \
  --mask_strategy all_single \
  --selection_metric harmonic_mean \
  --use_bit_head 0 \
  --epochs 12 \
  --patience 4 \
  --lr 1e-4
```

Requested comparison runs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --data_dir /mnt/data/zailongtian/workspace/cs712_project/data \
  --output_dir outputs/exp_ratio01_nb0 \
  --ddp 1 \
  --num_gpus 4 \
  --batch_size_per_gpu 128 \
  --eval_batch_size 256 \
  --mask_strategy ratio \
  --mask_ratio 0.1 \
  --selection_metric harmonic_mean \
  --use_bit_head 0 \
  --epochs 12 \
  --patience 4 \
  --lr 1e-4
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --data_dir /mnt/data/zailongtian/workspace/cs712_project/data \
  --output_dir outputs/exp_ratio03_nb0 \
  --ddp 1 \
  --num_gpus 4 \
  --batch_size_per_gpu 128 \
  --eval_batch_size 256 \
  --mask_strategy ratio \
  --mask_ratio 0.3 \
  --selection_metric harmonic_mean \
  --use_bit_head 0 \
  --epochs 12 \
  --patience 4 \
  --lr 1e-4
```

Optional bit-head follow-up:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
  --data_dir /mnt/data/zailongtian/workspace/cs712_project/data \
  --output_dir outputs/exp_all_single_bit \
  --ddp 1 \
  --num_gpus 4 \
  --batch_size_per_gpu 64 \
  --eval_batch_size 256 \
  --mask_strategy all_single \
  --selection_metric harmonic_mean \
  --use_bit_head 1 \
  --alpha 0.1 \
  --beta 0.0 \
  --epochs 12 \
  --patience 4 \
  --lr 1e-4
```

Key outputs under `--output_dir`:

- `baseline_metrics.json`
- `train_history.jsonl`
- `final_summary.json`
- `best.ckpt.pt` or the file named by `--save_name`

## Predict

Co-occurrence baseline:

```bash
python predict.py \
  --data_dir /mnt/data/zailongtian/workspace/cs712_project/data \
  --predictor cooccurrence \
  --window_size 4 \
  --cooc_add_k 0.25 \
  --out_name outputs/preds_cooccurrence/pred.txt
```

Transformer checkpoint only:

```bash
python predict.py \
  --data_dir /mnt/data/zailongtian/workspace/cs712_project/data \
  --predictor transformer \
  --ckpt_path outputs/exp_all_single_nb0/best.ckpt.pt \
  --out_name outputs/preds_transformer/pred.txt
```

Fusion predictor:

```bash
python predict.py \
  --data_dir /mnt/data/zailongtian/workspace/cs712_project/data \
  --predictor fusion \
  --ckpt_path outputs/exp_all_single_nb0/best.ckpt.pt \
  --gamma 0.1 \
  --beta 0.0 \
  --out_name /mnt/data/zailongtian/workspace/cs712_project/data/pred.txt
```

`predict.py` writes:

- `pred.txt`: one eligible token per validation line
- `pred.zip`: contains only `pred.txt`

The `--out_name` path must end with `pred.txt`.

## Best Hybrid Workflow

Rebuild the stronger n-gram with `ctx3` backoff:

```bash
python3 ngram_hybrid_v2.py build \
  --train outputs_maskdiff_2stage/auto_dev/train_split.txt \
  --eligible data/vocab_eligible.txt \
  --out outputs_maskdiff_2stage/ngram_v2.pkl \
  --max_len 512 \
  --alpha 0.1
```

Check the pure n-gram baseline on labeled dev:

```bash
python3 ngram_hybrid_v2.py eval \
  --model outputs_maskdiff_2stage/ngram_v2.pkl \
  --labeled_tsv outputs_maskdiff_2stage/auto_dev/dev_labeled.tsv
```

Sweep hybrid weights on labeled dev:

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

Use the best dev setting to write the final submission:

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

`lambda_ng=0` is pure neural inference. Larger `lambda_ng` moves the prediction closer to pure n-gram. Values `nn_temp > 1` flatten the neural distribution and often make the fusion more stable.

## Dev metrics

`train.py` creates a deterministic sentence-level train/dev split from `train.txt`.

- Absolute accuracy: exact token match
- Relative accuracy: `1 - HD(pred, truth) / 32`
- Early stopping uses dev absolute accuracy
- Relative accuracy and harmonic mean are also logged

## Notes

- The classifier head predicts only over eligible tokens, not the full train vocabulary.
- Dense labels use `ignore_index=-100` for unmasked positions.
- `ratio` masking trains on all masked positions in the sentence simultaneously.
- Validation inference still predicts exactly one token because validation has exactly one `MASK` per line.
- If `--max_len` is smaller than a sentence, cropping keeps the masked region in view.
