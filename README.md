# Empirical Bayes Conformal Prediction for Vision and Language Models

Code accompanying the paper *Empirical Bayes Conformal Prediction for Vision and Language Models*.

The paper studies `r-value` conformal prediction as an uncertainty-aware alternative to standard conformal prediction (`CP`) and average-then-conformal baselines (`CPavg`). The key idea is to rank classes or responses using both score level and score variability, then calibrate those rankings with conformal prediction to obtain valid coverage sets.

The repository has three experiment tracks:

- [`clip_setup/README.md`](./clip_setup/README.md): CLIP / SigLIP / OpenCLIP vision-language experiments using paraphrased label prompts.
- [`vit_setup/README.md`](./vit_setup/README.md): frozen image-classifier experiments using weighted-bootstrap adapter ensembles.
- [`llm_setup/README.md`](./llm_setup/README.md): closed-ended LLM experiments using paraphrased multiple-choice prompts.

At a high level, the paper studies `r-value` conformal prediction as an uncertainty-aware alternative to standard conformal prediction (`CP`) and average-then-conformal baselines (`CPavg`).

## Repo Layout

- `clip_setup/`: VLM prompt rephrasing and zero-shot evaluation.
- `vit_setup/`: vision-model posterior approximation with lightweight adapters.
- `llm_setup/`: LLM rephrasing, scoring, and notebook-based analysis.

## Common Dependencies

This is a research-script repository rather than a packaged library. The main dependencies used across setups are:

- `torch`
- `transformers`
- `datasets`
- `numpy`
- `tqdm`
- `Pillow`
- `openai`
- `open_clip_torch`
- `jupyter`

OpenAI-backed paraphrasing scripts require `OPENAI_API_KEY`. A root-level `.env.example` is included as a reference for commonly used environment variables.

## Citation

Until the final public bibliographic record is available, cite the manuscript by title and use the bundled PDF as the reference artifact.
