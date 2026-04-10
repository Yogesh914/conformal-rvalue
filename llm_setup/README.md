# `llm_setup`

This directory contains the paper's closed-ended LLM experiments, where multiple-choice questions are paraphrased to induce prompt variability and the resulting score distributions are analyzed with conformal prediction and `r-value`.

## Main Files

- `rephrase.py`: generate rephrased MMLU-style question variants and save them to disk.
- `run_split.py`: score answer options with a local Hugging Face causal LM and save arrays for downstream analysis.
- `run_rvalue.sh`: example launcher for split-based runs.
- `conformal_llm_scores.py`: older single-subject workflow.
- `rvalue.ipynb`: main notebook for downstream analysis.
- `rvalue_gpqa.ipynb`: related GPQA-style notebook.
- `prompt_questions.py`: canned prompt variants used by the scoring scripts.

## Quick Start

Generate rephrased datasets:

```bash
export OPENAI_API_KEY=...
python llm_setup/rephrase.py
```

Run one evaluation split with a local model:

```bash
cd llm_setup
python run_split.py --split_number 0 --total_splits 3 --model_path /path/to/model --output_dir ./output
```

## Outputs

`run_split.py` writes arrays such as:

- `*_escors.npy`
- `*_logits.npy`
- `*_targets.npy`

These outputs are intended for notebook-based downstream conformal / `r-value` analysis.

## Notes

- `rephrase.py` requires `OPENAI_API_KEY`.
- `rephrase.py` uses `OPENAI_REPHRASE_MODEL` if set, otherwise defaults to `gpt-4o-2024-08-06`.
- The code is still research-oriented: task lists, prompt lists, GPU layout, and some dataset paths are partially hard-coded.
- `run_split.py` currently assumes datasets were already saved to disk and uses relative `../data/...` paths.
