from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from settings import (
    DEFAULT_NUM_REPHRASES,
    DEFAULT_OUTPUT_DIR,
    available_dataset_names,
    get_dataset_config,
)

SYSTEM_PROMPT = """You are a creative AI tasked with generating rephrasings of short image descriptions.
Your task is to:
1. Read the description carefully.
2. Generate {num_rephrases} diverse and natural-sounding rephrasings of the description.
3. Keep the core meaning the same.
4. Return a JSON object with a key 'rephrased_descriptions' containing a list of strings."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rephrased class prompts for a configured dataset.")
    parser.add_argument("--dataset", required=True, choices=available_dataset_names())
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--num-rephrases", type=int, default=DEFAULT_NUM_REPHRASES)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--class-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    from openai import OpenAI

    return OpenAI()


def load_existing_output(path: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    if not path.exists():
        return {}, {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and isinstance(payload.get("prompts"), dict):
        metadata = {key: value for key, value in payload.items() if key != "prompts"}
        prompts = payload["prompts"]
        return metadata, prompts

    if isinstance(payload, dict):
        return {}, payload

    raise ValueError(f"Unsupported JSON structure in {path}")


def sanitize_rephrases(values: Any, expected_count: int) -> list[str]:
    if not isinstance(values, list):
        return []

    cleaned = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if len(cleaned) != expected_count:
        return []

    return cleaned


def request_rephrases(
    client,
    prompt: str,
    openai_model: str,
    num_rephrases: int,
    retries: int,
) -> list[str]:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(num_rephrases=num_rephrases),
                    },
                    {
                        "role": "user",
                        "content": f"Original description:\n{prompt}",
                    },
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            payload = json.loads(content)
            cleaned = sanitize_rephrases(payload.get("rephrased_descriptions"), num_rephrases)
            if cleaned:
                return cleaned

            raise ValueError(f"Expected {num_rephrases} valid rephrases.")
        except Exception as exc:  # pragma: no cover - network/runtime behavior
            last_error = exc
            print(f"[attempt {attempt}/{retries}] failed for prompt {prompt!r}: {exc}")
            if attempt < retries:
                time.sleep(min(2**attempt, 8))

    if last_error is not None:
        print(f"Giving up on prompt {prompt!r}: {last_error}")

    return []


def write_output(
    path: Path,
    dataset_name: str,
    openai_model: str,
    num_rephrases: int,
    prompts: dict[str, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "openai_model": openai_model,
        "num_rephrases": num_rephrases,
        "prompts": prompts,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    dataset_config = get_dataset_config(args.dataset)
    output_path = args.output or dataset_config.default_rephrased_path(DEFAULT_OUTPUT_DIR)

    _, prompts = load_existing_output(output_path)
    if args.overwrite:
        prompts = {}

    client = build_client()

    class_names = list(dataset_config.class_names)
    if args.class_limit is not None:
        class_names = class_names[: args.class_limit]

    for class_name in tqdm(class_names, desc=f"Rephrasing {args.dataset} labels"):
        existing = sanitize_rephrases(prompts.get(class_name), args.num_rephrases)
        if existing and not args.overwrite:
            continue

        base_prompt = dataset_config.base_prompt(class_name)
        prompts[class_name] = request_rephrases(
            client=client,
            prompt=base_prompt,
            openai_model=args.openai_model,
            num_rephrases=args.num_rephrases,
            retries=args.retries,
        )
        write_output(
            path=output_path,
            dataset_name=args.dataset,
            openai_model=args.openai_model,
            num_rephrases=args.num_rephrases,
            prompts=prompts,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_output(
        path=output_path,
        dataset_name=args.dataset,
        openai_model=args.openai_model,
        num_rephrases=args.num_rephrases,
        prompts=prompts,
    )
    print(f"Saved rephrased prompts to {output_path}")


if __name__ == "__main__":
    main()
