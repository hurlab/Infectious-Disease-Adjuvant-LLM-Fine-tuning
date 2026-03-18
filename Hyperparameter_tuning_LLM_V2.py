#!/usr/bin/env python
# coding: utf-8

# =========================================================
# SAFE MULTIPROCESSING (REQUIRED FOR MULTI-GPU 70B)
# =========================================================
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

# =========================================================
# IMPORTS
# =========================================================
import os
import json
import csv
import argparse
import shutil
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# =========================================================
# ENV + REPRODUCIBILITY (MATCH CODE 1)
# =========================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

seed = 42
set_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True

# =========================================================
# ARGUMENTS (ONE RUN PER PROCESS — SAME AS YOUR BASH LOOP)
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument(
    "--base_model",
    required=True,
    help="Path or HF identifier for the base model"
)

args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT = OUT_DIR.parent
run_id = OUT_DIR.name
BASE_MODEL = args.base_model


# =========================================================
# SWEEP SETUP
# =========================================================
summary_csv = OUT_ROOT / "sweep_summary.csv"
fieldnames = [
    "run_id",
    "lr", "epochs", "batch_size", "grad_accum", "warmup_ratio",
    "lora_r", "lora_alpha", "lora_dropout",
    "best_eval_loss",
    "test_eval_loss",
]


# =========================================================
# PATHS + CONSTANTS (MATCH CODE 1)
# =========================================================
#LOCAL_MODEL_DIR = "/home/hasin.rehana/models/Llama-3.3-70B-Instruct"

DATA_JSONL = "Dataset/VIOLIN_12-10-2025/final/llm_adjuvant_evidence_corpus_finetune.jsonl"
SPLITS_JSON = "Dataset/VIOLIN_12-10-2025/final/splits_fixed.json"

MAX_LENGTH = 1024

SYS_PROMPT = "You are a biomedical information extraction assistant."
PROMPT_INSTRUCTION = (
    "Extract infectious-disease adjuvants from the text and provide evidence snippets.\n"
    "Return ONLY valid JSON in this format:\n"
    "[{\"adjuvant\": \"<string>\", \"evidence\": \"<string>\"}, ...]\n"
    "Do not include any extra keys or explanation."
)


# =========================================================
# HELPERS (IDENTICAL TO CODE 1)
# =========================================================
def create_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # fp16 ONLY (same as Code 1)
    )

def find_linear_leaf_names(model):
    return sorted({
        name.split(".")[-1]
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    })

def build_fields(example):
    return {
        "user_prompt": f"{PROMPT_INSTRUCTION}\n\n{example['input']}",
        "answer": json.dumps(
            example["output"]["adjuvant_evidence"],
            ensure_ascii=False,
        ),
    }

def preprocess_batch(batch, tokenizer):
    input_ids, attention_mask, labels = [], [], []
    for user_prompt, answer in zip(batch["user_prompt"], batch["answer"]):
        prefix_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": ""},
            ],
            tokenize=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": answer},
                #{"role": "assistant", "content": answer.strip()}

            ],
            tokenize=True,
        )
        lbl = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
        input_ids.append(full_ids[:MAX_LENGTH])
        attention_mask.append([1] * min(len(full_ids), MAX_LENGTH))
        labels.append(lbl[:MAX_LENGTH])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def preprocess_dataset(ds, tokenizer, shuffle=False):
    ds = ds.map(build_fields, batched=False)
    ds = ds.map(
        lambda b: preprocess_batch(b, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
    )

    if shuffle:
        ds = ds.shuffle(seed=seed)

    return ds


def collate_with_labels(tokenizer, features):
    batch = tokenizer.pad(
        {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        },
        return_tensors="pt",
    )
    max_len = batch["input_ids"].shape[1]
    labels = [
        f["labels"] + [-100] * (max_len - len(f["labels"]))
        for f in features
    ]
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch

# =========================================================
# GENERATION-BASED TEST EVAL
# =========================================================
def decode_predictions(model, tokenizer, test_raw, out_jsonl, out_pretty):
    model.eval()
    preds = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in test_raw:
            prompt = f"{PROMPT_INSTRUCTION}\n\n{rec['input']}"
            chat = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                chat, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
            ).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
            gen = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            row = {"pmid": rec["pmid"], "gold": rec["output"], "prediction": gen}
            preds.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_pretty, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)


# =========================================================
# LOSS CSV + CURVES (SAME AS CODE 1)
# =========================================================
def save_loss_artifacts(log_history, out_dir):
    out_dir = Path(out_dir)
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for e in log_history:
        step = e.get("step")
        if step is None:
            continue
        if "loss" in e:
            train_steps.append(step)
            train_loss.append(e["loss"])
        if "eval_loss" in e:
            eval_steps.append(step)
            eval_loss.append(e["eval_loss"])

    with open(out_dir / "loss_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "eval_loss"])
        for s in sorted(set(train_steps + eval_steps)):
            w.writerow([
                s,
                dict(zip(train_steps, train_loss)).get(s, ""),
                dict(zip(eval_steps, eval_loss)).get(s, ""),
            ])

    if train_steps:
        plt.figure()
        plt.plot(train_steps, train_loss)
        plt.tight_layout()
        plt.savefig(out_dir / "train_loss_curve.png", dpi=150)
        plt.close()

    if eval_steps:
        plt.figure()
        plt.plot(eval_steps, eval_loss)
        plt.tight_layout()
        plt.savefig(out_dir / "eval_loss_curve.png", dpi=150)
        plt.close()

# =========================================================
# LOAD DATA (IDENTICAL TO CODE 1)
# =========================================================
with open(SPLITS_JSON) as f:
    splits = json.load(f)

dataset = load_dataset("json", data_files={"data": DATA_JSONL}, split="data")

"""train_all = dataset.filter(lambda x: x["pmid"] in set(splits["train_pmids"]))
test_raw = dataset.filter(lambda x: x["pmid"] in set(splits["test_pmids"]))

train_all = train_all.shuffle(seed=seed)
val_size = max(1, int(0.1 * len(train_all)))
val_raw = train_all.select(range(val_size))
train_raw = train_all.select(range(val_size, len(train_all)))"""

train_raw = dataset.filter(lambda x: x["pmid"] in set(splits["train_pmids"]))
val_raw   = dataset.filter(lambda x: x["pmid"] in set(splits["val_pmids"]))
test_raw  = dataset.filter(lambda x: x["pmid"] in set(splits["test_pmids"]))

# =========================================================
# MODEL LOAD (MULTI-GPU, SAME SEMANTICS)
# =========================================================
# =========================================================
# MODEL LOAD (Gemma / LLaMA / Mistral – WORKING BASELINE)
# =========================================================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=create_bnb_config(),
    device_map="auto",
    local_files_only=True,
)


model.config.use_cache = False

# IMPORTANT: ENABLE checkpointing again
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    use_fast=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

target_modules = [
    m for m in ["q_proj", "k_proj", "v_proj", "o_proj"]
    if m in find_linear_leaf_names(model)
]

model = get_peft_model(
    model,
    LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

# =========================================================
# TRAINER (NUMBERS IDENTICAL TO CODE 1)
# =========================================================
trainer = Trainer(
    model=model,
    train_dataset=preprocess_dataset(train_raw, tokenizer, shuffle=True), #preprocess_dataset(train_raw, tokenizer),
    eval_dataset=preprocess_dataset(val_raw, tokenizer, shuffle=False), #preprocess_dataset(val_raw, tokenizer),
    args=TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        report_to=None,
        per_device_eval_batch_size=1, ## to vaoid OOM
    ),
    data_collator=lambda f: collate_with_labels(tokenizer, f),
)

trainer.train()

best_eval = trainer.state.best_metric

# =========================================================
# SAVE OUTPUTS (SAME AS CODE 1)
# =========================================================
save_loss_artifacts(trainer.state.log_history, OUT_DIR)
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)


# =========================================================
# GENERATION-BASED VALIDATION OUTPUT (NEW)
# =========================================================
decode_predictions(
    trainer.model,            # best model (same as test)
    tokenizer,
    val_raw,                  # validation split
    OUT_DIR / "val_predictions.jsonl",
    OUT_DIR / "val_predictions_pretty.json",
)

test_metrics = trainer.evaluate(preprocess_dataset(test_raw, tokenizer, shuffle=False))

decode_predictions(
    trainer.model,  # IMPORTANT: best model
    tokenizer,
    test_raw,
    OUT_DIR / "test_predictions.jsonl",
    OUT_DIR / "test_predictions_pretty.json",
)

for p in OUT_DIR.glob("checkpoint-*"):
    shutil.rmtree(p)

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames)
    if f.tell() == 0:
        writer.writeheader()
    writer.writerow({
        **cfg,
        "run_id": run_id,
        "best_eval_loss": best_eval,
        "test_eval_loss": test_metrics["eval_loss"],
    })

    
torch.cuda.empty_cache()
torch.cuda.ipc_collect()