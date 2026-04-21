#!/usr/bin/env python3
"""
Constrained SFT / soft-SFT for Qwen2.5-3B (same trainer stack as ai-safety-align/finetune.py).

Uses the local `finetuning_buckets/` package in this directory.

  cd /data/b22cs089/FML_RAG/finetune
  python constrained_finetune_qwen.py --output-dir ./runs/qwen25-c --dataset-name sql_create_context

  # Optional HuggingFace TrainingArguments / TRL ModelConfig after `--`:
  python constrained_finetune_qwen.py --output-dir ./runs/o -- --logging_steps 5 --bf16 true
"""

from __future__ import annotations

import argparse
import os
import sys

_FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
if _FINETUNE_DIR not in sys.path:
    sys.path.insert(0, _FINETUNE_DIR)

import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from finetuning_buckets.datasets.utils import get_finetuning_data
from finetuning_buckets.models import get_model
from finetuning_buckets.trainer.trainer import ConstrainedSFTTrainer

try:
    from datasets import set_caching_enabled

    set_caching_enabled(False)
except ImportError:
    from datasets import disable_caching

    disable_caching()


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


@dataclass
class ScriptArguments:
    dataset_name: str = field(default="sql_create_context")
    model_family: str = field(default="qwen25")
    max_seq_length: int = field(default=512)
    sft_type: str = field(default="soft_sft")
    beta: float = field(default=0.1)
    bias_factor: float = field(default=20.0)
    first_token_bias_factor: float = field(default=5.0)
    bias_length: int = field(default=5)
    use_warmup: bool = field(default=False)
    use_anchor: bool = field(default=False)
    anchor_batch_size_per_device: int = field(default=16)
    safety_augmentation: bool = field(default=False)


def parse_hf_args(output_dir: str, rest: list[str]) -> tuple[TrainingArguments, ModelConfig]:
    base = ["--output_dir", output_dir, "--overwrite_output_dir", "--report_to", "none"]
    args_list = base + (rest if rest else ["--logging_steps", "10", "--save_steps", "500"])
    hp = HfArgumentParser((TrainingArguments, ModelConfig))
    return hp.parse_args_into_dataclasses(args_list=args_list, look_from_args_file=False)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Constrained fine-tune Qwen2.5-3B (ConstrainedSFTTrainer).",
        epilog="Pass extra TrainingArguments / ModelConfig after `--`.",
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", default="sql_create_context")
    p.add_argument("--model-family", default="qwen25", choices=["qwen25", "qwen2.5"])
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--sft-type", choices=["sft", "soft_sft"], default="soft_sft")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per-device-bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--bias-factor", type=float, default=20.0)
    p.add_argument("--first-token-bias-factor", type=float, default=5.0)
    p.add_argument("--bias-length", type=int, default=5)
    p.add_argument("--no-4bit", action="store_true")
    known, rest = p.parse_known_args()
    if "--" in rest:
        i = rest.index("--")
        rest = rest[i + 1 :]

    script = ScriptArguments(
        dataset_name=known.dataset_name,
        model_family=known.model_family,
        max_seq_length=known.max_seq_length,
        sft_type=known.sft_type,
        beta=known.beta,
        bias_factor=known.bias_factor,
        first_token_bias_factor=known.first_token_bias_factor,
        bias_length=known.bias_length,
    )

    training_args, model_config = parse_hf_args(known.output_dir, rest)
    training_args.num_train_epochs = known.epochs
    training_args.per_device_train_batch_size = known.per_device_bs
    training_args.gradient_accumulation_steps = known.grad_accum
    training_args.learning_rate = known.lr
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    if training_args.report_to is None:
        training_args.report_to = "none"
    if script.use_warmup:
        training_args.warmup_steps = 10
    training_args.adam_beta1 = 0.9 if script.safety_augmentation else 0.5

    model_config.model_name_or_path = known.model_id
    model_config.trust_remote_code = True
    model_config.load_in_4bit = not known.no_4bit
    if model_config.attn_implementation is None and torch.cuda.is_available():
        model_config.attn_implementation = "sdpa"

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, str(model_config.torch_dtype))
    )
    quantization_config = get_quantization_config(model_config)
    trust_remote_code = model_config.trust_remote_code if model_config.trust_remote_code is not None else True
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model, tokenizer = get_model.get_model(
        model_config.model_name_or_path, model_kwargs, model_family=script.model_family
    )
    disable_dropout(model)
    ref_model = model if script.sft_type == "soft_sft" else None

    dataset = get_finetuning_data.get_dataset(
        script.dataset_name,
        split="train",
        string_format=script.model_family,
        safety_augmentation=script.safety_augmentation,
    )
    if not script.safety_augmentation:
        data_collator = get_finetuning_data.get_data_collator(
            tokenizer,
            dataset_name=script.dataset_name,
            model_family=script.model_family,
        )
    else:
        if script.model_family == "llama2":
            from finetuning_buckets.models.model_families.llama2 import (
                AugmentedSafetyDataCollator as Llama2AugmentedSafetyDataCollator,
            )

            data_collator = Llama2AugmentedSafetyDataCollator(tokenizer=tokenizer)
        else:
            raise ValueError("safety_augmentation only supported for llama2 in this codebase")

    anchor_dataset = anchor_data_collator = None
    if script.use_anchor:
        anchor_dataset = get_finetuning_data.get_dataset(
            "alpaca_instruction", split="train", string_format=script.model_family
        )
        anchor_data_collator = get_finetuning_data.get_data_collator(
            tokenizer, dataset_name="alpaca_instruction", model_family=script.model_family
        )

    if script.sft_type == "sft":
        if script.use_anchor:
            trainer = ConstrainedSFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                anchor_dataset=anchor_dataset,
                max_seq_length=script.max_seq_length,
                packing=False,
                dataset_text_field="text",
                data_collator=data_collator,
                use_soft_sft=False,
                use_anchor=True,
                anchor_batch_size_per_device=script.anchor_batch_size_per_device,
                anchor_data_collator=anchor_data_collator,
                safety_augmentation=script.safety_augmentation,
            )
        else:
            trainer = ConstrainedSFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=dataset,
                max_seq_length=script.max_seq_length,
                packing=False,
                dataset_text_field="text",
                data_collator=data_collator,
                use_soft_sft=False,
                use_anchor=False,
                safety_augmentation=script.safety_augmentation,
            )
    else:
        trainer = ConstrainedSFTTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            max_seq_length=script.max_seq_length,
            packing=False,
            dataset_text_field="text",
            data_collator=data_collator,
            beta=script.beta,
            bias_factor=script.bias_factor,
            bias_length=script.bias_length,
            first_token_bias_factor=script.first_token_bias_factor,
            use_soft_sft=True,
        )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Saved → {training_args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
