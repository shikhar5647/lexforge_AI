#!/usr/bin/env python3
"""
QLoRA SFT for Qwen2.5-3B on CUAD-QA (or any HF QA dataset with {question, context, answers}).

Same CLI shell as ``constrained_finetune_qwen.py`` / ``adaptive_reg.py`` — thin wrapper
over the ``lexforge_finetune.ipynb`` notebook: 4-bit NF4 base + LoRA adapters on the
attention + MLP linears, trained with ``trl.SFTTrainer`` / ``SFTConfig``.

  cd /data/b22cs089/FML_RAG/finetune
  python sft.py --output-dir ./runs/lexforge-qwen25-cuad

  # QLoRA off, full-precision LoRA:
  python sft.py --output-dir ./runs/lexforge-fp --no-4bit

  # Push the final adapter to the Hub:
  python sft.py --output-dir ./runs/lexforge-qwen25-cuad \\
      --hf-repo-id your-username/lexforge-qwen25-3b-cuad --push-to-hub

  # Extra HF/TRL arguments (any SFTConfig field) can be passed after ``--``:
  python sft.py --output-dir ./runs/o -- --warmup_ratio 0.05 --weight_decay 0.0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from trl import SFTConfig, SFTTrainer

try:
    from datasets import set_caching_enabled

    set_caching_enabled(False)
except ImportError:
    from datasets import disable_caching

    disable_caching()


SYSTEM_PROMPT = (
    "You are LexForge AI, a legal-domain assistant specialized in contract analysis. "
    "Given a contract excerpt and a question about a specific clause or provision, "
    "answer precisely using only information from the excerpt. "
    "If the contract does not contain the information, respond exactly: "
    "'Not specified in the provided contract.'"
)

NO_ANSWER = "Not specified in the provided contract."


def build_formatter(tokenizer: Any):
    """Factory: returns a ``.map`` function that turns a QA row into ``{"text": ...}``."""

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        question = example["question"]
        context = example["context"]
        answers_field = example.get("answers", {}) or {}
        spans = answers_field.get("text", []) if isinstance(answers_field, dict) else []
        answer = spans[0].strip() if spans and spans[0].strip() else NO_ANSWER

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Contract excerpt:\n{context}\n\nQuestion: {question}",
            },
            {"role": "assistant", "content": answer},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    return format_example


def _load_dataset_safely(dataset_id: str):
    """
    Try a normal load; if the repo ships a legacy loading script (rejected by
    ``datasets >= 4``), fall back to the auto-converted Parquet mirror that HF
    hosts under ``refs/convert/parquet``.
    """
    try:
        return load_dataset(dataset_id)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        print(
            f"[sft.py] '{dataset_id}' is script-based; "
            "retrying with refs/convert/parquet.",
            file=sys.stderr,
        )
        return load_dataset(dataset_id, revision="refs/convert/parquet")


def load_and_format(
    tokenizer: Any,
    dataset_id: str,
    *,
    eval_samples: int,
    seed: int,
) -> tuple[Any, Any]:
    raw = _load_dataset_safely(dataset_id)
    fmt = build_formatter(tokenizer)

    train_cols = raw["train"].column_names
    train = raw["train"].map(fmt, remove_columns=train_cols, desc="Formatting train")

    eval_split = "test" if "test" in raw else ("validation" if "validation" in raw else None)
    if eval_split is None:
        return train, None

    eval_cols = raw[eval_split].column_names
    evalds = raw[eval_split].map(fmt, remove_columns=eval_cols, desc="Formatting eval")
    if eval_samples > 0:
        evalds = evalds.shuffle(seed=seed).select(range(min(eval_samples, len(evalds))))
    return train, evalds


def parse_extra_sft_config(output_dir: str, rest: list[str]) -> SFTConfig:
    """Allow arbitrary SFTConfig fields after ``--`` (matches sibling scripts' style)."""
    base = ["--output_dir", output_dir, "--overwrite_output_dir", "--report_to", "none"]
    args_list = base + (rest if rest else [])
    hp = HfArgumentParser(SFTConfig)
    (cfg,) = hp.parse_args_into_dataclasses(args=args_list)
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(
        description="QLoRA SFT for Qwen2.5-3B on CUAD-QA (notebook parity).",
        epilog="Pass extra SFTConfig fields after `--`.",
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-id", default="theatticusproject/cuad-qa")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per-device-bs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-4bit", action="store_true", help="Disable QLoRA 4-bit loading")
    p.add_argument("--no-flash-attn", action="store_true", help="Force sdpa instead of flash_attention_2")
    p.add_argument("--no-eval", action="store_true", help="Skip eval split entirely")
    p.add_argument("--hf-repo-id", default=None, help="Hub repo id (e.g. user/lexforge-cuad)")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--report-to", default="none", help="HF reporter (none, mlflow, wandb, ...)")
    p.add_argument("--run-name", default="lexforge-qwen25-cuad-qlora")

    known, rest = p.parse_known_args()
    if "--" in rest:
        i = rest.index("--")
        rest = rest[i + 1 :]

    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    use_fp16 = (not use_bf16) and torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    def _flash_attn_available() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except Exception:
            return False

    if known.no_flash_attn or not use_bf16 or not _flash_attn_available():
        attn_impl = "sdpa"
        if use_bf16 and not known.no_flash_attn:
            print("flash_attn not installed → falling back to sdpa.", file=sys.stderr)
    else:
        attn_impl = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(known.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if not known.no_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        known.model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=known.lora_r,
        lora_alpha=known.lora_alpha,
        lora_dropout=known.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_and_format(
        tokenizer,
        known.dataset_id,
        eval_samples=0 if known.no_eval else known.eval_samples,
        seed=known.seed,
    )
    if known.no_eval:
        eval_dataset = None

    sft_config = parse_extra_sft_config(known.output_dir, rest)
    sft_config.num_train_epochs = known.epochs
    sft_config.per_device_train_batch_size = known.per_device_bs
    sft_config.per_device_eval_batch_size = known.per_device_bs
    sft_config.gradient_accumulation_steps = known.grad_accum
    sft_config.gradient_checkpointing = True
    sft_config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    sft_config.learning_rate = known.lr
    sft_config.lr_scheduler_type = "cosine"
    sft_config.warmup_ratio = known.warmup_ratio
    sft_config.weight_decay = known.weight_decay
    sft_config.max_grad_norm = 1.0
    sft_config.optim = "paged_adamw_8bit" if bnb_config is not None else "adamw_torch"
    sft_config.bf16 = use_bf16
    sft_config.fp16 = use_fp16
    sft_config.tf32 = use_bf16
    sft_config.logging_strategy = "steps"
    sft_config.logging_steps = known.logging_steps
    sft_config.save_strategy = "steps"
    sft_config.save_steps = known.save_steps
    sft_config.save_total_limit = known.save_total_limit
    if eval_dataset is not None:
        sft_config.eval_strategy = "steps"
        sft_config.eval_steps = known.eval_steps
        sft_config.load_best_model_at_end = True
        sft_config.metric_for_best_model = "eval_loss"
        sft_config.greater_is_better = False
    else:
        sft_config.eval_strategy = "no"
        sft_config.load_best_model_at_end = False
    sft_config.report_to = known.report_to
    sft_config.run_name = known.run_name
    if hasattr(sft_config, "max_length"):
        sft_config.max_length = known.max_seq_length
    if hasattr(sft_config, "max_seq_length"):
        sft_config.max_seq_length = known.max_seq_length
    sft_config.packing = False
    sft_config.dataset_text_field = "text"
    sft_config.seed = known.seed
    sft_config.dataloader_num_workers = 4
    sft_config.remove_unused_columns = True
    sft_config.push_to_hub = known.push_to_hub and bool(known.hf_repo_id)
    sft_config.hub_model_id = known.hf_repo_id
    sft_config.hub_strategy = "end"

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("final_eval", metrics)
        trainer.save_metrics("final_eval", metrics)
        print(metrics, file=sys.stderr)

    if sft_config.push_to_hub:
        trainer.push_to_hub()
        print(f"Pushed adapter → {known.hf_repo_id}", file=sys.stderr)

    print(f"Saved → {sft_config.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
