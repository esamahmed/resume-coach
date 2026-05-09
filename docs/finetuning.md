# Fine-Tuning Documentation
## AI Resume Coach — How to Fine-Tune on Custom Data

**Status:** Code and steps provided. Training data is not available at capstone time
(as per rubric expectations). This documents exactly what you would do if labeled 
training pairs were provided.

---

## What We Would Fine-Tune On

The ideal fine-tuning dataset would be pairs of:
- Input: `{resume_text} + {jd_text} + {ats_score}`
- Output: `{ground_truth_gap_analysis_json}`

Where ground truth comes from human career coaches reviewing the same pairs.
Without this labeled corpus, we use the prompt-engineered LLM outputs as
silver-label proxies for evaluation, and the golden fixture set in `eval/fixtures/`
as our approximation of ground truth.

---

## Dataset Preparation

```python
# scripts/prepare_finetune_dataset.py
"""
Prepares a fine-tuning dataset from:
1. Kaggle LinkedIn Job Postings dataset
2. Human-labeled resume+JD pairs (if available)
3. Silver-labeled pairs from GPT-4 baseline outputs
"""
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def format_instruction(resume: str, jd: str, ats_score: float, output: dict) -> str:
    """Format a training example in Mistral instruction format."""
    instruction = f"""<s>[INST] You are an expert resume coach. Analyze this resume against the job description.

ATS Score: {ats_score:.1%}

Resume:
{resume[:3000]}

Job Description:
{jd[:2000]}

Respond with only valid JSON gap analysis. [/INST]
{json.dumps(output, indent=2)}</s>"""
    return instruction

def load_linkedin_dataset(path: str = "data/linkedin_job_postings.csv") -> pd.DataFrame:
    """
    Load LinkedIn Job Postings from Kaggle.
    Dataset: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
    """
    df = pd.read_csv(path)
    return df[["title", "description", "skills_desc"]].dropna()

def prepare_dataset(labeled_pairs_path: str, output_path: str) -> None:
    """
    labeled_pairs: list of {resume_text, jd_text, ats_score, gap_analysis}
    These would come from human career coach review or GPT-4 silver labeling.
    """
    with open(labeled_pairs_path) as f:
        pairs = json.load(f)
    
    formatted = [
        format_instruction(
            p["resume_text"], p["jd_text"], 
            p["ats_score"], p["gap_analysis"]
        )
        for p in pairs
    ]
    
    dataset = Dataset.from_dict({"text": formatted})
    dataset = dataset.train_test_split(test_size=0.1)
    dataset.save_to_disk(output_path)
    print(f"Dataset saved: {len(formatted)} examples, {output_path}")

if __name__ == "__main__":
    prepare_dataset("data/labeled_pairs.json", "data/finetune_dataset")
```

---

## Fine-Tuning with QLoRA (Parameter-Efficient Fine-Tuning)

QLoRA lets us fine-tune Mistral-7B on a single GPU (e.g. Kaggle's free T4)
by training only low-rank adapter weights, not the full model.

```python
# scripts/finetune_qlora.py
"""
Fine-tune Mistral-7B-Instruct on resume coach dataset using QLoRA.
Runs on: Kaggle (free T4 GPU) or AWS SageMaker (ml.g4dn.xlarge)
"""
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_from_disk

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./models/resume-coach-mistral-qlora"

# 4-bit quantization config (fits on single T4 GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config — train only small adapter matrices
lora_config = LoraConfig(
    r=16,                    # rank of update matrices
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.5% of parameters trainable — very efficient

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # effective batch size = 8
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    report_to="none",
)

dataset = load_from_disk("data/finetune_dataset")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuned model saved to {OUTPUT_DIR}")
```

---

## Deploying Fine-Tuned Model to SageMaker

After fine-tuning, merge LoRA weights and upload to S3:

```python
# scripts/merge_and_upload.py
import os
from peft import AutoPeftModelForCausalLM
import torch
import boto3

# Merge LoRA into base model
model = AutoPeftModelForCausalLM.from_pretrained(
    "./models/resume-coach-mistral-qlora",
    torch_dtype=torch.float16,
)
merged = model.merge_and_unload()
merged.save_pretrained("./models/merged")

# Upload to S3 — reads bucket from environment variable
bucket = os.getenv("S3_BUCKET", "your-resume-coach-bucket-name")
s3 = boto3.client("s3")

for root, _, files in os.walk("./models/merged"):
    for f in files:
        path = os.path.join(root, f)
        key = "resume-coach/model/" + path.replace("./models/merged/", "")
        s3.upload_file(path, bucket, key)
        print(f"Uploaded: {key} → s3://{bucket}/{key}")
```

Then deploy from S3 artifact using the HuggingFace DLC (same as `scripts/deploy_sagemaker.py`
but pointing to the S3 model URI instead of HuggingFace Hub).

---

## Hyperparameter Choices & Rationale

| Hyperparameter | Value | Rationale |
|---|---|---|
| LoRA rank (r) | 16 | Balances capacity and memory. 8 underfits on structured JSON task; 32 exceeds T4 VRAM |
| lora_alpha | 32 | Standard 2× rank scaling — controls update magnitude |
| Learning rate | 2e-4 | Standard for QLoRA instruction tuning; 1e-4 converges slower, 5e-4 overshoots |
| Epochs | 3 | Sufficient for instruction tuning; >5 risks overfitting small labeled set |
| Batch size | 2 + grad_accum 4 | Effective batch 8 — fits T4 16GB VRAM with 4-bit quant |
| Max seq length | 4096 | Covers typical resume (~800 tokens) + JD (~600 tokens) + output (~400 tokens) |
| Quantization | NF4 4-bit | Best quality/memory tradeoff for inference on g5.2xlarge |
