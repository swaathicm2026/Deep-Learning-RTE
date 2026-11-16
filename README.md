# RTE Challenge: fine-tuning DeBERTa-v3-base (CS6315E)

Binary textual entailment (RTE): given a premise and a hypothesis, predict `entailment` or `not_entailment`.
All models are fully fine-tuned from public pretrained checkpoints (no LoRA/adapters, no NLI-fine-tuned checkpoints, no external data).

Author: Swaathi | M260560CS

## 1. What you need

- A Google account and [Google Colab](https://colab.research.google.com/).
- A **GPU runtime**: Runtime -> Change runtime type -> **T4 GPU** -> Save.
  (On CPU the notebook runs but is far too slow.)
- The data files `train.jsonl` and `val.jsonl`. Each line is one JSON object with the fields
  `premise`, `hypothesis` and `label` (`entailment` or `not_entailment`).
- The notebook (`.ipynb`) in this repository.

No packages need installing on Colab; it already has `torch`, `transformers`, `pandas`, `scikit-learn` and `matplotlib`.
Environment used for the reported numbers: Tesla T4, PyTorch 2.14.1+cu130, transformers [version].
If the tokenizer fails to load, run `!pip install sentencepiece` in a cell, then restart the session.

## 2. Upload the files

1. Open the notebook in Colab (File -> Upload notebook).
2. Open the **Files** panel (folder icon on the left).
3. Drag `train.jsonl` and `val.jsonl` into `/content/` (the top-level folder, next to `sample_data`).
   The notebook reads exactly `/content/train.jsonl` and `/content/val.jsonl`.

Files in `/content/` disappear if Colab disconnects or the runtime is deleted, so re-upload them if the notebook cannot find them.

## 3. Run one experiment

1. **Runtime -> Restart session** (so no model from an earlier run is left in memory).
2. Edit the settings, then **Runtime -> Run all**, once, from the top:

| Setting | Where | Meaning |
|---|---|---|
| `SEED` | first code cell | random seed (1, 2 or 42) |
| `MODEL_NAME` | model-loading cell | e.g. `microsoft/deberta-v3-base` |
| `LR` | training cell | learning rate (1e-5, 2e-5 or 3e-5) |
| `EPOCHS` | training cell | 5 for base models, 10 for DeBERTa-v3-large |
| `BATCH_SIZE`, `ACCUM_STEPS` | data cell | 4 and 4 (effective batch 16) |
| `USE_FP16` | training cell | `True`: fp16 autocast with GradScaler (weights stay fp32) |

3. Wait for the training cell to finish. It prints one line per epoch, then the final results.

**Important:** always use *Restart session -> Run all*. Re-running only the training cell continues training from
weights that are already fine-tuned, which gives invalid, non-reproducible numbers. A healthy run starts with an epoch-1
loss of about 0.65 to 0.70; a loss near 0.02 means the model was not fresh.

For `deberta-v3-large`, keep `model.gradient_checkpointing_enable()` (uncomment it) and use `BATCH_SIZE = 4`,
`ACCUM_STEPS = 4`, `LR = 1e-5`, `EPOCHS = 10`. For all base-size models leave it commented out.

## 4. What the notebook does

1. Loads `train.jsonl` and `val.jsonl`, maps labels to 0 (`entailment`) and 1 (`not_entailment`).
2. Splits the training data 90/10 with a fixed split (`random_state=42`, stratified): 2,241 examples to train,
   249 as a hold-out used **only** to choose the best epoch.
3. Tokenizes each pair as `[CLS] premise [SEP] hypothesis [SEP]`, max length 128, dynamic padding.
4. Fine-tunes with AdamW (weight decay 0.01, none on biases/LayerNorm), 10% linear warm-up then linear decay,
   gradient clipping at 1.0, effective batch size 16.
5. After each epoch, evaluates on the hold-out and keeps the best epoch's weights.
6. Loads the best weights and evaluates **once** on the official dev set (`val.jsonl`).
7. Saves dev predictions, counts parameters, and appends one row to the results log.

## 5. Outputs

All outputs go to `/content/results/`:

| File | Content |
|---|---|
| `runs.csv` | one row per run: model, seed, LR, epochs, effective batch, total and trainable parameters, dev accuracy |
| `preds/<model>_seed<SEED>.csv` | dev predictions (`id`, `prediction`, `label`) |
| `final_table.csv`, `acc_vs_params.png` | summary table and accuracy-vs-parameters plot (from the last cells) |

The prediction file name does not include the learning rate, so a later run with the same seed **overwrites** it.
Download the file right after each run and rename it, e.g. `deberta-v3-base_seed2_lr2e-5.csv`
(Files panel -> right-click -> Download). Download `runs.csv` too, because `/content/` is wiped on disconnect.

## 6. Reproducing the submitted models

Model: `microsoft/deberta-v3-base`, 184,423,682 parameters (all trainable), 5 epochs, effective batch 16.

| Run | Seed | LR | Dev accuracy |
|---|---|---|---|
| 1 | 1 | 1e-5 | 83.03 |
| 2 | 2 | 2e-5 | 83.39 |
| 3 | 2 | 3e-5 | 85.56 |

Run the notebook three times (Restart session -> Run all each time) with these settings.
GPU training is not bit-for-bit deterministic, so expect dev accuracy to differ by a point or two across
sessions or library versions (the dev set has only 277 examples; one example is 0.36 points).

## 7. Other models in the report

Same notebook, change only `MODEL_NAME` (and `LR`, `EPOCHS`, `SEED` as below):

| Model | LR | Epochs | Seeds |
|---|---|---|---|
| `google/electra-base-discriminator`, `FacebookAI/roberta-base`, `albert/albert-base-v2`, `xlnet/xlnet-base-cased`, `distilbert/distilbert-base-uncased`, `google-bert/bert-base-uncased` | 2e-5 | 5 | 42 |
| `microsoft/deberta-v3-base` | 1e-5, 2e-5, 3e-5 | 5 | 42, 1, 2 |
| `microsoft/deberta-v3-large` | 1e-5, 2e-5, 3e-5 | 10 | 42 (and 1 for LR 3e-5) |

## 8. Build the summary table and plot

After all runs are logged in `runs.csv`, run the last two cells of the notebook (summary table and plot).
Delete any duplicate or smoke-test rows from `runs.csv` first; the table averages everything in the file.

## 9. Deliverables in this repository

- `README.md`: this file
- the training notebook (`.ipynb`)
- `model_card.pdf`, `RTE_report.pdf`
- prediction files for the three submitted runs
