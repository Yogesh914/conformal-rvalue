# `demo`

This directory contains a small notebook-first demo for the repository's `r-value` conformal workflow on CIFAR-100 CLIP outputs. Unlike the main experiment folders, it ships a compact set of precomputed artifacts so you can inspect the method without rerunning the full pipelines.

## Main Files

- `demo.ipynb`: quickest walkthrough; loads bundled scores and targets, then compares `r-value` and standard LAC conformal sets on individual examples.
- `set_size_eval.ipynb`: repeated-split evaluation of empirical coverage and average prediction-set size using the same bundled score tensors.
- `clip_inference.ipynb`: regenerates CLIP probabilities from CIFAR-100 images plus rephrased label prompts and writes fresh arrays under `data/`.
- `rephrase.py`: generates 30 paraphrases per CIFAR-100 class label with the OpenAI API.

## Bundled Data

The checked-in `data/` directory includes the artifacts used by the demo notebooks:

- `clip_probs_matrix_cifar100_2k.npy`: CLIP probability tensor for a 2k-example CIFAR-100 subset.
- `clip_targets_cifar100_2k.npy`: integer labels aligned with the bundled score tensor.
- `label_rephrased_dict_cifar100.json`: original and paraphrased class-label prompt text used by the demo.

## Quick Start

Run the notebooks from inside `demo/` so their relative `./data/...` paths resolve correctly:

```bash
cd demo
jupyter notebook
```

Suggested order:

1. Open `demo.ipynb` for the smallest end-to-end illustration.
2. Open `set_size_eval.ipynb` for aggregate set-size and coverage experiments.
3. Use `clip_inference.ipynb` only if you want to regenerate CLIP scores instead of using the bundled arrays.

## Dependencies

The demo uses the same research-style environment as the rest of the repo. In practice you will want:

- `jupyter`
- `numpy`
- `torch`
- `transformers`
- `datasets`
- `openai`

## Notes

- `clip_inference.ipynb` expects a local dataset saved at `./cifar_100_rephrased_labels` and reads prompt variants from `./data/label_rephrased_dict_cifar100.json`.
- `rephrase.py` is research code rather than a polished CLI; it initializes the OpenAI client in-code and may need small credential/path cleanup before rerunning it in a fresh environment.
- For the maintained end-to-end VLM pipeline, see [`../clip_setup/README.md`](../clip_setup/README.md).
