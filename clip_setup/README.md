# `clip_setup`

This directory contains the paper's vision-language experiments, where uncertainty is induced by paraphrasing class-label prompts and re-scoring images against those prompt variants.

## Main Scripts

- `rephrase_labels.py`: generate paraphrased label prompts with the OpenAI API.
- `run_inference.py`: run zero-shot inference with CLIP / SigLIP / OpenCLIP-style models and save per-variant outputs.
- `settings.py`: dataset and model registry.
- `comparison.ipynb`: notebook for downstream comparison and analysis.

## Supported Datasets

- `cifar10`
- `cifar100`
- `traffic`
- `satellite`
- `imagenet_a`
- `imagenet_r`

## Supported Models

- `clip_b16`
- `clip_b32`
- `siglip2_base`
- `mobileclip2_b`

## Quick Start

Generate paraphrased prompts:

```bash
python clip_setup/rephrase_labels.py --dataset cifar10
```

Run zero-shot evaluation:

```bash
python clip_setup/run_inference.py --dataset cifar10 --model clip_b16
```

Use a local dataset instead of downloading from Hugging Face:

```bash
python clip_setup/run_inference.py \
  --dataset cifar10 \
  --model clip_b16 \
  --dataset-path /path/to/local/dataset
```

## Outputs

`run_inference.py` writes files under `data/` by default:

- `*_accuracy_results.json`
- `*_probs_matrix.npy`
- `*_targets.npy`
- `*_class_accuracies.npy`

`rephrase_labels.py` writes paraphrased prompt JSON files under `data/` by default.

## Notes

- If no paraphrased prompt file is found, `run_inference.py` falls back to the default prompt only.
- `rephrase_labels.py` requires `OPENAI_API_KEY`.
- The prompt variability used by the paper comes from these paraphrased label sets, so the default-prompt-only path is mainly for sanity checks.
