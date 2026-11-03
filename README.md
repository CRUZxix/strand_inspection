# Strand Inspection System
### Fully local computer-vision inspection of strand components in a packet

An image-based system that inspects a packet holding **10 components in a 5×2 grid**, counts the strands in each component, and decides whether the packet passes.

- Each component can have 0–5 strands. **The expected count is 4** (count 4 = GOOD, any other count = DEFECT).
- The pipeline runs fully **locally**: image processing plus a small CNN (ResNet18). No external vision API is used.
- An earlier prototype (`app.py`, Streamlit + OpenAI vision) is still in the repo but is **not** part of the local pipeline.

---

## Current status

| Phase | Description | Status |
|---|---|---|
| 1 | Automatic localisation and cropping of the 10 components | Done |
| 2 | Preprocessing + packet-level train / CV / test splits | Done |
| 3 | PyTorch dataset loader (`src/dataset.py`) | Done |
| 4 | ResNet18 training and evaluation (`scripts/03_train.py`) | Done: 5-fold CV comparison run |
| 5 | Inference with visual result (`scripts/04_predict.py`) | Working; see limitations |
| 6 | Local Streamlit app, no API key (`app_local.py`) | Working |
| – | Final model trained on all CV folds + one-time frozen-test evaluation | **Not done yet** |
| – | UNCERTAIN threshold tuning, per-belt localiser, error analysis | Planned |

## Dataset

67 packets, 670 labelled component crops (`data_build/dataset.csv`; labels are clean: count 4 ⇔ GOOD).

| Strand count | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Crops | 10 | 10 | 30 | 45 | 500 | 75 |

**Splits are by packet, never by crop** (neighbouring crops overlap, so crop-level splitting leaks). The split is frozen in `data_build/splits.json`:

- **Test:** 11 packets / 110 crops (`split == "test"`). Used only once for the final evaluation, never for model selection.
- **CV:** 56 packets in 5 folds (`cv_fold` 0–4), used for training and validation.

Two preprocessing variants are generated for every crop (576×320):

- `enh`: contrast-stretched, brightness-normalised RGB.
- `cs3`: CLAHE lightness + Sobel-Feldman edges + yellow-belt mask as 3 channels.

## Pipeline

1. **Localise** (`src/localize.py`): a yellow/orange HSV mask finds the belt block, then a regular overlapping 5×2 grid is laid over it. Boxes are cropped to a fixed aspect ratio.
2. **Preprocess** (`src/preprocess.py`): `enh` / `cs3` variants.
3. **Classify**: ResNet18 (ImageNet-pretrained, 6 outputs for counts 0–5) with class-weighted cross-entropy and light augmentation (horizontal flip, small rotation/shift/scale, brightness/contrast jitter).
4. **Decide**: count 4 = GOOD, other counts = DEFECT, low-confidence = UNCERTAIN. The packet **passes only if all 10 components are GOOD**.

## Scripts

| File | Purpose |
|---|---|
| `scripts/01_make_crops.py` | Phase 1: crop every packet photo into 10 components and attach labels |
| `scripts/02_preprocess_split.py` | Phase 2: clean labels, packet-level splits, `enh`/`cs3` preprocessing |
| `scripts/03_train.py` | Phase 4: train and evaluate ResNet18 |
| `scripts/04_predict.py` | Inspect a new photo and save an annotated result image |
| `app_local.py` | Local Streamlit app using the trained model (no API key) |
| `src/infer.py` | Shared inference + result drawing (`inspect_packet`, `render`) |
| `scripts/check_low_counts.py` | Helper for reviewing low-count crops |
| `src/localize.py`, `src/preprocess.py` | Localisation and preprocessing |
| `src/dataset.py` | PyTorch dataset, split helper, class weights |

### Train

Run from the repo root (the `.venv` needs `torch` and `torchvision`):

```powershell
# one validation fold
python scripts/03_train.py --variant enh --val-fold 0 --epochs 15

# full 5-fold comparison of both variants
foreach ($v in "enh","cs3") { foreach ($f in 0..4) { python -u scripts/03_train.py --variant $v --val-fold $f --epochs 15 } }

# quick smoke test
python scripts/03_train.py --variant enh --smoke

# final model: train on all CV folds, evaluate the frozen test set ONCE
python scripts/03_train.py --variant enh --final --epochs <N>
```

The best epoch is chosen by macro-F1 on the validation fold only. Results, history and the model are saved to `runs/phase4/<variant>_val<fold>/` (`metrics.json`, `model.pt`). Metrics: accuracy, macro-F1, per-class recall, confusion matrix, defect recall, false acceptance rate (defect passed as GOOD) and false rejection rate. Training is on CPU.

### Inspect a new packet photo

```powershell
python scripts/04_predict.py "path\to\photo.jpg" --open
```

Saves `runs/predictions/<name>_result.png`: the photo with boxes and predicted counts, plus a 5×2 result matrix with crop thumbnails (green GOOD, red DEFECT, amber UNCERTAIN) and a PACKET PASS / FAIL banner. By default the 5 `enh` fold models are averaged. Pass `--models` to choose specific `model.pt` files and `--conf` to change the UNCERTAIN threshold.

### Local app (no API key)

```powershell
python -m streamlit run app_local.py
```

Upload one or more packet photos in the browser to get the annotated photo, the 5×2 result matrix, a PASS / FAIL verdict and a download button. It runs the trained model entirely offline through `src/infer.py` (`inspect_packet(image, nets)`), which is also what `04_predict.py` uses. The older `app.py` (OpenAI) is unchanged.

If a yellow table or background is detected behind the packet, the localiser switches to a table-rejecting mask automatically.

## Results so far (5-fold CV, validation folds, mean)

| Metric | `enh` | `cs3` |
|---|---:|---:|
| Accuracy | 0.936 | 0.928 |
| Macro-F1 | 0.726 | 0.673 |
| Defect recall | 0.922 | 0.915 |
| False acceptance rate | 0.078 | 0.085 |
| False rejection rate | 0.045 | 0.039 |

- `enh` is slightly ahead, but the difference is about the size of the fold-to-fold spread.
- Counts 2–5 reach recall of about 0.88–0.97 (`enh`). Counts 0 and 1 are weak (about 0.30 and 0.50) and have very few examples, so those numbers are noisy.
- About 8% of defects are still accepted as GOOD, which is too high for an inspection system. Several true count-0 crops are predicted as 4, which needs checking (localisation or label problems).
- These are CV validation numbers only. The frozen test set has not been used yet.

## Known limitations

- **Localisation is approximate.** A regular grid is laid over the belt block rather than detecting each belt, so crops can be off-centre or include neighbouring belts, especially when the two columns are offset. Some training crops have localisation imperfections.
- **Photos that look unlike the training set** (different angle, lighting, background) may localise badly. Check that the boxes sit on the components before trusting the counts.
- **The UNCERTAIN threshold (0.6) is a placeholder** and has not been tuned.
- The fold models used by `04_predict.py` are not the final model.

## Next steps

1. Error analysis of misclassified crops (especially count 0 vs 4) and localisation failures.
2. A per-belt localiser (connected components in the mask, sorted into 5×2).
3. Tune the UNCERTAIN threshold on validation folds to reduce the false acceptance rate.
4. Train the final model on all CV folds and evaluate on the frozen test set once.

## Project structure

```text
strand-inspection/
├── images/            Original packet photos
├── data_build/        dataset.csv, splits.json, crops.csv, reports (committed);
│                      crops/, processed/{enh,cs3}/, contact/ (generated, NOT committed)
├── src/               localize.py, preprocess.py, dataset.py, infer.py
├── scripts/           01_make_crops.py, 02_preprocess_split.py, 03_train.py, 04_predict.py
├── runs/              training outputs and predictions (NOT committed, incl. model.pt files)
├── labels.csv, packets.csv
├── app_local.py       Local Streamlit app (trained model, no API key)
├── app.py             Earlier Streamlit + OpenAI prototype (not part of the local pipeline)
└── README.md
```

### What is and isn't in the repository

The repository holds the code, the original photos, the labels, `dataset.csv` and the frozen `splits.json`. Generated and large files are not committed, so after cloning:

```powershell
python scripts/01_make_crops.py          # regenerates data_build/crops
python scripts/02_preprocess_split.py    # regenerates data_build/processed (do NOT regenerate splits casually)
python scripts/03_train.py --variant enh --val-fold 0   # trains and writes runs/phase4/... (model.pt)
```

Trained models (`runs/**/model.pt`) are not in the repository, so `app_local.py` and `04_predict.py` need training first.

## Setup

```bash
git clone https://github.com/CRUZxix/strand_inspection.git
cd strand_inspection
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision pandas numpy pillow opencv-python streamlit
```

(`streamlit` is needed for `app_local.py`. `openai` is only needed for the legacy `app.py`; that prototype needs an `OPENAI_API_KEY` in `.streamlit/secrets.toml`. Never commit keys.)

## Repository

GitHub: https://github.com/CRUZxix/strand_inspection

*This project is under active development. Performance figures are from cross-validation and are subject to further testing.*
