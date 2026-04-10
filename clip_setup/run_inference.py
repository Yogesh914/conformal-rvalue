from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

from settings import (
    DEFAULT_NUM_REPHRASES,
    DEFAULT_OUTPUT_DIR,
    DatasetConfig,
    ModelConfig,
    available_dataset_names,
    available_model_names,
    get_dataset_config,
    get_model_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP-style inference with prompt rephrasing ensembles.")
    parser.add_argument("--dataset", required=True, choices=available_dataset_names())
    parser.add_argument("--model", required=True, choices=available_model_names())
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--rephrased-labels", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-rephrases", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device):
    device_type = device.type if device.type in {"cpu", "cuda"} else "cpu"
    return torch.amp.autocast(device_type=device_type, enabled=device.type == "cuda")


def ensure_rgb_image(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    return Image.fromarray(np.asarray(image_value)).convert("RGB")


def sanitize_rephrases(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def load_rephrased_prompts(path: Path | None) -> tuple[dict[str, list[str]], int | None]:
    if path is None or not path.exists():
        return {}, None

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("prompts"), dict):
        prompts = {key: sanitize_rephrases(value) for key, value in payload["prompts"].items()}
        num_rephrases = payload.get("num_rephrases")
        return prompts, num_rephrases if isinstance(num_rephrases, int) else None

    if isinstance(payload, dict):
        prompts = {key: sanitize_rephrases(value) for key, value in payload.items()}
        return prompts, None

    raise ValueError(f"Unsupported JSON structure in {path}")


def get_rephrases_for_class(prompt_map: dict[str, list[str]], class_name: str, display_name: str) -> list[str]:
    candidate_keys = (
        class_name,
        display_name,
        class_name.replace("_", " "),
        class_name.replace(" ", "_"),
        display_name.replace(" ", "_"),
    )
    for key in candidate_keys:
        if key in prompt_map:
            return sanitize_rephrases(prompt_map[key])
    return []


def build_prompt_variants(
    dataset_config: DatasetConfig,
    prompt_map: dict[str, list[str]],
    num_rephrases: int,
) -> list[list[str]]:
    prompts_per_class: list[list[str]] = []

    for class_name in dataset_config.class_names:
        base_prompt = dataset_config.base_prompt(class_name)
        display_name = dataset_config.display_name(class_name)
        rephrased = get_rephrases_for_class(prompt_map, class_name, display_name)[:num_rephrases]
        prompts = [base_prompt] + rephrased
        while len(prompts) < num_rephrases + 1:
            prompts.append(base_prompt)
        prompts_per_class.append(prompts)

    return [list(variant) for variant in zip(*prompts_per_class)]


def load_eval_dataset(
    dataset_config: DatasetConfig,
    dataset_path: Path | None,
    limit: int | None,
    shuffle: bool,
    seed: int,
):
    dataset = (
        load_from_disk(str(dataset_path))
        if dataset_path is not None
        else load_dataset(dataset_config.hf_name, split=dataset_config.split)
    )

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return dataset


@dataclass
class EvalArtifacts:
    probs_matrix: np.ndarray
    targets: np.ndarray
    variant_accuracies: list[float]
    default_accuracy: float
    ensemble_accuracy: float
    class_accuracies: np.ndarray


class HFClipRunner:
    def __init__(self, model_config: ModelConfig, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_config.model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_config.processor_name or model_config.model_name)

    def prepare_variant(self, text_prompts: list[str]) -> list[str]:
        return text_prompts

    def predict_probs(self, prepared_variant: list[str], batch_images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(text=prepared_variant, images=batch_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            with autocast_context(self.device):
                outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        return probs.cpu().numpy().astype(np.float32, copy=False)


class HFSiglipRunner:
    def __init__(self, model_config: ModelConfig, device: torch.device):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        model_kwargs: dict[str, Any] = {}
        if model_config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = model_config.attn_implementation
        self.model = AutoModel.from_pretrained(model_config.model_name, **model_kwargs).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_config.processor_name or model_config.model_name)

    def prepare_variant(self, text_prompts: list[str]) -> list[str]:
        return text_prompts

    def predict_probs(self, prepared_variant: list[str], batch_images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(
            text=prepared_variant,
            images=batch_images,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            with autocast_context(self.device):
                outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits_per_image)
        return probs.cpu().numpy().astype(np.float32, copy=False)


class OpenClipRunner:
    def __init__(self, model_config: ModelConfig, device: torch.device):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_config.model_name,
            pretrained=model_config.pretrained,
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_config.model_name)

    def prepare_variant(self, text_prompts: list[str]) -> torch.Tensor:
        text_tokens = self.tokenizer(text_prompts).to(self.device)
        with torch.no_grad():
            with autocast_context(self.device):
                text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def predict_probs(self, prepared_variant: torch.Tensor, batch_images: list[Image.Image]) -> np.ndarray:
        image_tensor = torch.stack([self.preprocess(image) for image in batch_images]).to(self.device)
        with torch.no_grad():
            with autocast_context(self.device):
                image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ prepared_variant.T
            probs = logits.softmax(dim=-1)
        return probs.cpu().numpy().astype(np.float32, copy=False)


def build_runner(model_config: ModelConfig, device: torch.device):
    if model_config.family == "hf_clip":
        return HFClipRunner(model_config, device)
    if model_config.family == "hf_siglip":
        return HFSiglipRunner(model_config, device)
    if model_config.family == "open_clip":
        return OpenClipRunner(model_config, device)
    raise ValueError(f"Unsupported model family: {model_config.family}")


def compute_class_accuracies(probs_matrix: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    predicted = probs_matrix.argmax(axis=2)
    num_variants = probs_matrix.shape[1]
    class_accuracies = np.zeros((num_variants, num_classes), dtype=np.float32)

    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if not np.any(class_mask):
            continue
        class_accuracies[:, class_idx] = (predicted[class_mask] == class_idx).mean(axis=0)

    return class_accuracies


def evaluate(
    dataset,
    dataset_config: DatasetConfig,
    runner,
    prompt_variants: list[list[str]],
    batch_size: int,
) -> EvalArtifacts:
    num_samples = len(dataset)
    num_variants = len(prompt_variants)
    num_classes = dataset_config.num_classes

    probs_matrix = np.zeros((num_samples, num_variants, num_classes), dtype=np.float32)
    targets = np.zeros((num_samples,), dtype=np.int64)
    per_variant_correct = np.zeros((num_variants,), dtype=np.int64)

    for variant_idx, text_prompts in enumerate(prompt_variants):
        print(f"Processing variant {variant_idx + 1}/{num_variants}")
        prepared_variant = runner.prepare_variant(text_prompts)

        for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"Variant {variant_idx + 1}"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = range(batch_start, batch_end)

            batch_images: list[Image.Image] = []
            batch_targets = np.zeros((batch_end - batch_start,), dtype=np.int64)

            for offset, sample_idx in enumerate(batch_indices):
                example = dataset[sample_idx]
                batch_images.append(ensure_rgb_image(example[dataset_config.image_field]))
                batch_targets[offset] = int(example[dataset_config.label_field])

            probs = runner.predict_probs(prepared_variant, batch_images)
            if probs.shape != (batch_end - batch_start, num_classes):
                raise ValueError(
                    f"Unexpected probability shape {probs.shape}; expected {(batch_end - batch_start, num_classes)}"
                )

            probs_matrix[batch_start:batch_end, variant_idx] = probs
            if variant_idx == 0:
                targets[batch_start:batch_end] = batch_targets

            preds = probs.argmax(axis=1)
            per_variant_correct[variant_idx] += int((preds == batch_targets).sum())

    ensemble_probs = probs_matrix.mean(axis=1)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_accuracy = float((ensemble_preds == targets).mean())
    variant_accuracies = (per_variant_correct / num_samples).astype(np.float64).tolist()
    default_accuracy = float(variant_accuracies[0])
    class_accuracies = compute_class_accuracies(probs_matrix, targets, num_classes)

    return EvalArtifacts(
        probs_matrix=probs_matrix,
        targets=targets,
        variant_accuracies=variant_accuracies,
        default_accuracy=default_accuracy,
        ensemble_accuracy=ensemble_accuracy,
        class_accuracies=class_accuracies,
    )


def save_outputs(
    dataset_name: str,
    model_name: str,
    output_dir: Path,
    prompt_variants: list[list[str]],
    rephrased_labels_path: Path | None,
    artifacts: EvalArtifacts,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{model_name}_{dataset_name}"

    best_variant = int(np.argmax(artifacts.variant_accuracies))
    worst_variant = int(np.argmin(artifacts.variant_accuracies))

    accuracy_payload = {
        "dataset": dataset_name,
        "model": model_name,
        "num_samples": int(len(artifacts.targets)),
        "num_variants": int(len(prompt_variants)),
        "default_accuracy": artifacts.default_accuracy,
        "ensemble_accuracy": artifacts.ensemble_accuracy,
        "variant_accuracies": [float(value) for value in artifacts.variant_accuracies],
        "best_variant": best_variant,
        "worst_variant": worst_variant,
        "best_variant_example_prompt": prompt_variants[best_variant][0],
        "worst_variant_example_prompt": prompt_variants[worst_variant][0],
        "rephrased_labels_path": str(rephrased_labels_path) if rephrased_labels_path is not None else None,
    }

    with (output_dir / f"{prefix.name}_accuracy_results.json").open("w", encoding="utf-8") as handle:
        json.dump(accuracy_payload, handle, indent=2, ensure_ascii=True)

    np.save(output_dir / f"{prefix.name}_probs_matrix.npy", artifacts.probs_matrix)
    np.save(output_dir / f"{prefix.name}_targets.npy", artifacts.targets)
    np.save(output_dir / f"{prefix.name}_class_accuracies.npy", artifacts.class_accuracies)


def main() -> None:
    args = parse_args()
    dataset_config = get_dataset_config(args.dataset)
    model_config = get_model_config(args.model)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.rephrased_labels is not None:
        if not args.rephrased_labels.exists():
            raise FileNotFoundError(f"Rephrased label file not found: {args.rephrased_labels}")
        rephrased_labels_path = args.rephrased_labels
    else:
        rephrased_labels_path = dataset_config.default_rephrased_path(DEFAULT_OUTPUT_DIR)
        if not rephrased_labels_path.exists():
            rephrased_labels_path = None
            print("No rephrased label file found. Falling back to default prompts only.")

    prompt_map, prompt_file_num_rephrases = load_rephrased_prompts(rephrased_labels_path)
    num_rephrases = (
        args.num_rephrases
        if args.num_rephrases is not None
        else prompt_file_num_rephrases
        if prompt_file_num_rephrases is not None
        else DEFAULT_NUM_REPHRASES
    )

    dataset = load_eval_dataset(
        dataset_config=dataset_config,
        dataset_path=args.dataset_path,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} examples for dataset '{args.dataset}'")

    prompt_variants = build_prompt_variants(dataset_config, prompt_map, num_rephrases)
    runner = build_runner(model_config, device)
    artifacts = evaluate(
        dataset=dataset,
        dataset_config=dataset_config,
        runner=runner,
        prompt_variants=prompt_variants,
        batch_size=args.batch_size,
    )

    save_outputs(
        dataset_name=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        prompt_variants=prompt_variants,
        rephrased_labels_path=rephrased_labels_path,
        artifacts=artifacts,
    )

    best_variant = int(np.argmax(artifacts.variant_accuracies))
    worst_variant = int(np.argmin(artifacts.variant_accuracies))
    print(f"Default prompt accuracy: {artifacts.default_accuracy:.4f}")
    print(f"Ensemble prompt accuracy: {artifacts.ensemble_accuracy:.4f}")
    print(f"Best variant accuracy: {artifacts.variant_accuracies[best_variant]:.4f} (variant {best_variant})")
    print(f"Worst variant accuracy: {artifacts.variant_accuracies[worst_variant]:.4f} (variant {worst_variant})")


if __name__ == "__main__":
    main()
