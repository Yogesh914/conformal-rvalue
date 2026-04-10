# `vit_setup`

This directory contains the paper's vision-model experiments, where posterior variability is approximated with weighted-bootstrap training of lightweight adapter heads attached to a frozen image classifier.

## Main Scripts

- `main.py`: train one adapter checkpoint bundle.
- `adaptors_eval.py`: evaluate all saved adapter checkpoints and export calibration/test arrays.
- `inference.py`: run the single-image diagnostic workflow.
- `config.py`: path and training configuration.
- `dataset.py`: ImageNet loading and weighted-bootstrap sampling.
- `model.py`: frozen backbone plus trainable adapter head definition.
- `trainer.py`: adapter training loop.

## Required Data

The code expects ImageNet saved with `datasets.load_from_disk(...)`.

Set:

- `IMAGENET_DATASET_DIR`: local Hugging Face dataset path for ImageNet.

Optional path variables from `config.py`:

- `CONFORMAL_RVALUE_DATA_DIR`
- `HF_DATASETS_CACHE`
- `VIT_ADAPTER_SAVE_DIR`
- `VIT_EVAL_OUTPUT_DIR`
- `VIT_INFERENCE_OUTPUT_PATH`

If unset, repo-local paths under `vit_setup/data/` are used where possible.

## Quick Start

Train one saved adapter bundle:

```bash
python vit_setup/main.py 0
```

Evaluate saved checkpoints on the 40k/10k split carved out of the ImageNet validation set:

```bash
python vit_setup/adaptors_eval.py
```

Run the single-image diagnostic:

```bash
python vit_setup/inference.py
```

## Outputs

Training writes:

- `Adaptor_<id>.pt` files into `VIT_ADAPTER_SAVE_DIR`

Evaluation writes arrays into `VIT_EVAL_OUTPUT_DIR`:

- `val_true_logits.npy`
- `val_true_probs.npy`
- `test_logits.npy`
- `test_true_logits.npy`
- `test_true_probs.npy`
- `test_probs.npy`

The diagnostic script writes a pickle to `VIT_INFERENCE_OUTPUT_PATH`.

## Notes

- `Config.NUM_MODELS` controls the number of parallel bootstrap-weighted heads trained inside one adapter bundle.
- Each saved checkpoint contains the adapter parameters and the bootstrap weights used for that training run.
- The current backbone selection is defined in `model.py`; adjust that file if you want to switch between the commented model variants.
