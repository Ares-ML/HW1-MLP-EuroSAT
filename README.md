# EuroSAT NumPy MLP — HW1

A from-scratch (NumPy-only) three-layer fully-connected classifier for the EuroSAT RGB land-cover dataset. **No PyTorch / TensorFlow / JAX** — all forward, backward, and optimization logic is implemented manually and validated against centered finite differences (worst relative error ≤ 7×10⁻⁷ across ReLU/Sigmoid/Tanh × label-smoothing combinations).

- **Repository**: <https://github.com/Ares-ML/HW1-MLP-EuroSAT>
- **Trained checkpoints**: https://pan.baidu.com/s/1-pBVqo4zNioc2-GREOLYjA?pwd=7wzt 
- **Headline results** (no data-split contamination):
  - 3-seed ensemble + TTA: **test_acc = 0.7930** (balanced 0.7910, macro-F1 0.7904, Cohen κ 0.7696, top-3 acc 0.9663)
  - Best single model (seed 43) + TTA: **test_acc = 0.7948**
- All course-required artefacts live under [runs/v3c_ensemble_final_eval/](runs/v3c_ensemble_final_eval/).

---

## Project layout

```
DL/
├── EuroSAT_RGB/                # dataset (10 class folders of 64×64 jpgs)
├── src/                        # library — 10 NumPy-only modules
│   ├── layers.py                 # Linear / ReLU / Sigmoid / Tanh / Dropout / softmax+CE+smoothing
│   ├── model.py                  # MLP class — forward/backward/state_dict/load
│   ├── data.py                   # JPEG load + 80/10/10 stratified split + z-score + D4 augmentation
│   ├── optim.py                  # SGD+momentum + Cosine / Step LR scheduler
│   ├── trainer.py                # training loop + best-by-val checkpointing + early stopping
│   ├── evaluate.py               # accuracy / confusion matrix / P-R-F1 / kappa / TTA inference
│   ├── visualize.py              # loss curves / W1 plots / error galleries / neuron preference plots
│   ├── analysis.py               # per-class fc1 activation stats + neuron→class preference scoring
│   ├── ensemble.py               # multi-checkpoint logit averaging
│   └── utils.py                  # set_seed + numerical gradient checker
├── scripts/                    # entry points
│   ├── sanity_check.py           # initial-loss / gradient-check / overfit-20 sanity tests
│   ├── train.py                  # train one MLP, with all hp surfaces exposed via CLI
│   ├── test.py                   # full evaluation of one checkpoint (incl. TTA)
│   ├── search.py                 # log-uniform random hp search + auto-retrain best
│   └── ensemble.py               # multi-seed logit averaging evaluation
├── runs/                       # one output dir per experiment (see .gitignore for what's tracked)
├── cache/                      # decoded uint8 dataset cache (~316 MB, git-ignored)
├── report.md                   # the experiment report (pdf for LMS submission)
├── SUBMISSION_CHECKLIST.md
├── requirements.txt
├── .gitignore
└── README.md                   # this file
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the EuroSAT RGB dataset so that paths look like
`EuroSAT_RGB/<Class>/<Class>_N.jpg` (the layout that the official
[`phelber/EuroSAT`](https://github.com/phelber/EuroSAT) zip unpacks to).

The first run of any script decodes all 27 000 jpgs and writes a 316 MB cache
to `cache/eurosat_raw.npz`; subsequent runs load this cache directly.

---

## Sanity check first

```bash
python -m scripts.sanity_check
```

Verifies in ~30 s:
1. Initial loss ≈ log(K) = 2.302 with random init
2. Centered finite-difference gradient check (float64) — worst relative error < 1e-4
3. Overfit a 20-sample subset to ~zero loss / 100% acc (model capacity OK)

---

## Reproducing the headline results (v3c)

```bash
# 1. Train 3 seeds — ALL three share the seed=42 data split (--data-seed 42)
#    but use different init seeds for parameter init / dropout / augmentation.
#    ~30 min per seed on a modest GPU; ~90 min total.
for SEED in 42 43 44; do
    python -m scripts.train \
        --hidden1 384 --hidden2 96 --dropout 0.1 --batch-size 128 \
        --lr 1.235e-2 --lr-min 1e-5 \
        --weight-decay 8.57e-4 --label-smoothing 0.1 \
        --epochs 200 --patience 25 \
        --seed $SEED --data-seed 42 \
        --out-dir runs/v3c_seed${SEED}
done

# 2. Best single-model evaluation with TTA (seed 43 reaches 0.7948).
python -m scripts.test \
    --checkpoint runs/v3c_seed43/best.pkl \
    --tta --top-per-class 1 --neuron-samples 4 \
    --out-dir runs/v3c_seed43/eval_tta

# 3. 3-seed ensemble + TTA evaluation (reaches 0.7930).
python -m scripts.ensemble \
    --checkpoints runs/v3c_seed42/best.pkl runs/v3c_seed43/best.pkl runs/v3c_seed44/best.pkl \
    --tta --top-per-class 1 --neuron-samples 4 \
    --out-dir runs/v3c_ensemble_final_eval
```

`--data-seed 42` is **required** for honest ensemble evaluation: it forces the same train/val/test split for every seed, so the ensemble inference set is held-out from all three members. Without it, each seed produces its own independent split and the ensemble inference set ends up overlapping with the training data of seeds 43 and 44 — see `report.md` §7.3 for the full diagnostic of an earlier run that hit this trap.

### Outputs

`runs/v3c_ensemble_final_eval/` — the canonical artefact directory referenced by `report.md`:

| File | Course requirement |
|---|---|
| `summary.json` | overall metrics (acc, balanced, macro-F1, kappa, top-3) + per-member acc + top-5 confusion pairs |
| `confusion_matrix.png` | row-normalized confusion heatmap |
| `w1_filters.png` | top-64 first-layer filter templates (signed-normalized RGB) |
| `w1_filters_per_channel.png` | top-16 filters split into R/G/B grayscale strips |
| `error_gallery.png` | top-16 highest-confidence misclassifications |
| `error_pairs/errors_*.png` | one PNG per top-K confusion pair (default K=4) |
| `neuron_class_preferences.png` | each row = one fc1 neuron preferring a class, with W1 template + class samples side by side |
| `neuron_activation_heatmap.png` | per-class mean activation of class-preferring neurons |
| `neuron_analysis.json` | numerical preference scores cited in `report.md` §8.2 |

Each `runs/v3c_seed*/` additionally contains `curves.png` (the **loss + accuracy curves required by the assignment**, course requirement ①) and `history.json` (per-epoch raw numbers).

---

## Hyperparameter search

```bash
python -m scripts.search --out-dir runs/v3_search
```

Default search space (after three rounds of refinement; see `report.md` §5):
- `lr` ∈ log-uniform[4e-3, 1.5e-2]
- `weight_decay` ∈ log-uniform[5e-4, 5e-3]
- `hidden1` ∈ {192, 256, 384}, `hidden2` ∈ {96, 128, 192}
- `batch_size` ∈ {64, 128}, `dropout` ∈ {0.05, 0.1, 0.15, 0.2}
- 25 trials × 30 epochs per trial; final retrain with the best config for 120 epochs.

---

## Training script options

```text
--hidden1, --hidden2     hidden layer sizes (default 384, 96)
--activation             relu | sigmoid | tanh                 (default relu)
--dropout                inverted dropout p                    (default 0.2)
--epochs                 number of training epochs             (default 50)
--batch-size             mini-batch size                       (default 128)
--lr / --lr-min          cosine schedule endpoints             (default 1e-2 → 5e-5)
--momentum               classical momentum                    (default 0.9)
--weight-decay           L2 strength on weights only           (default 1e-3)
--label-smoothing        ε in soft cross-entropy               (default 0.1)
--scheduler              cosine | step | none                  (default cosine)
--patience               early-stopping patience               (default 10)
--seed                   model init / shuffle seed             (default 42)
--data-seed              data-split seed; defaults to --seed   (use this to share splits across seeds)
--augment / --no-augment toggle D4 augmentation                (default on)
```

---

## Reproducibility

A fixed seed (default 42) controls (a) the stratified split (via `--data-seed` if set, otherwise via `--seed`), (b) parameter initialization, (c) per-epoch shuffling, and (d) the dropout mask. Re-running any script with the same seeds reproduces results bit-for-bit on a single machine (modulo float32 BLAS non-determinism on multi-threaded matmul; in practice differences are < 0.1pt).

The gradient checker in `scripts.sanity_check` runs in float64 and reports worst relative error per parameter array. A fresh checkout passes at < 1e-6 across all activation × label-smoothing combinations.

---

## Submission deliverables

- This repository (public on GitHub) — link at the top of `report.md`
- Trained checkpoints (3 × `best.pkl` + the `runs/v3c_ensemble_final_eval/` directory) — uploaded to Google Drive, link at the top of `report.md`
- Experiment report (`report.md`, exported to PDF for the LMS)

Last update: 2026-04-30
