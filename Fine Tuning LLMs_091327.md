---
---

# The Complete Guide to Fine-Tuning Large Language Models: From Theory to Production

**A Deep Technical Dive into LoRA, QLoRA, and Full Fine-Tuning with Modern Open-Source Models**

---

## Table of Contents

1. [Introduction to LLM Fine-Tuning](#introduction-to-llm-fine-tuning)
2. [Why Fine-Tune? Use Cases and Benefits](#why-fine-tune-use-cases-and-benefits)
3. [Understanding Fine-Tuning Approaches](#understanding-fine-tuning-approaches)
4. [Technical Deep-Dive: Full Fine-Tuning](#technical-deep-dive-full-fine-tuning)
5. [Technical Deep-Dive: LoRA and Variants](#technical-deep-dive-lora-and-variants)
6. [Technical Deep-Dive: QLoRA](#technical-deep-dive-qlora)
7. [Data Preparation Pipeline](#data-preparation-pipeline)
8. [Implementation: Full Fine-Tuning](#implementation-full-fine-tuning)
9. [Implementation: LoRA Fine-Tuning](#implementation-lora-fine-tuning)
10. [Implementation: QLoRA Fine-Tuning](#implementation-qlora-fine-tuning)
11. [Evaluation and Metrics](#evaluation-and-metrics)
12. [Best Practices and Optimization Tips](#best-practices-and-optimization-tips)
13. [Comparison of Approaches](#comparison-of-approaches)
14. [Conclusion](#conclusion)

---

## Introduction to LLM Fine-Tuning

Large Language Models (LLMs) have revolutionized natural language processing, demonstrating remarkable capabilities across diverse tasks. However, pre-trained models, while powerful, often require adaptation to perform optimally on domain-specific tasks. This is where **fine-tuning** comes into play—the process of continuing the training of a pre-trained model on a smaller, task-specific dataset.

The challenge with modern LLMs lies in their scale. Models like Llama 4, Qwen 3, DeepSeek-V3.2, and Gemma 3 contain billions of parameters, making traditional fine-tuning computationally prohibitive for most practitioners. This has led to the development of parameter-efficient fine-tuning (PEFT) methods that achieve comparable results while training only a fraction of the model's parameters.

![LLM Fine-Tuning Lifecycle: Pre-training, Fine-tuning, and Deployment phases](diagrams/01_llm_lifecycle.png)

---

## Why Fine-Tune? Use Cases and Benefits

### Primary Use Cases

1. **Domain Adaptation**: Adapting a general-purpose model to specialized domains like legal, medical, or financial text.

2. **Task-Specific Optimization**: Improving performance on specific tasks such as code generation, summarization, or question answering.

3. **Style and Tone Alignment**: Training models to match specific writing styles, brand voices, or communication patterns.

4. **Knowledge Injection**: Incorporating proprietary or recent knowledge not present in the pre-training data.

5. **Safety and Alignment**: Fine-tuning for responsible AI behavior, reducing harmful outputs, and improving instruction-following.

### Benefits Over Prompt Engineering

| Aspect | Prompt Engineering | Fine-Tuning |
|--------|-------------------|-------------|
| Performance | Good | Excellent |
| Consistency | Variable | High |
| Latency | Higher (longer prompts) | Lower |
| Cost per inference | Higher | Lower |
| Customization depth | Limited | Deep |
| Knowledge incorporation | Constrained | Extensive |

![Fine-Tuning Decision Framework: Benefits and Considerations](diagrams/02_finetuning_benefits.png)

---

## Understanding Fine-Tuning Approaches

Modern LLM fine-tuning encompasses three primary approaches, each with distinct trade-offs between computational efficiency, memory requirements, and model performance.

### Overview of Approaches

![Comparison of Fine-Tuning Approaches: Full, LoRA, and QLoRA](diagrams/03_approaches_overview.png)

### Parameter Comparison

For a 70B parameter model:

| Approach | Trainable Params | Memory (FP16) | Memory (QLoRA) | Training Speed |
|----------|-----------------|---------------|----------------|----------------|
| Full Fine-Tuning | 70B (100%) | ~280 GB | N/A | Slowest |
| LoRA (r=64) | ~100M (0.14%) | ~160 GB | ~48 GB | Fast |
| QLoRA (r=64, 4-bit) | ~100M (0.14%) | N/A | ~24 GB | Moderate |

---

## Technical Deep-Dive: Full Fine-Tuning

Traditional fine-tuning updates all parameters of the neural network. During backpropagation, gradients flow through the entire network, and all weights are adjusted based on the task-specific loss.

### Architecture and Gradient Flow

![Full Fine-Tuning Architecture and Gradient Flow](diagrams/04_full_finetuning_arch.png)

### Mathematical Formulation

For a weight matrix $W \in \mathbb{R}^{d \times d}$, full fine-tuning updates:

$$W_{t+1} = W_t - \alpha \frac{\partial \mathcal{L}}{\partial W_t}$$

Where:
- $\alpha$ is the learning rate
- $\mathcal{L}$ is the loss function
- $\frac{\partial \mathcal{L}}{\partial W_t}$ is the gradient of the loss with respect to weights

### When to Use Full Fine-Tuning

- **Sufficient compute resources** available (multiple high-end GPUs)
- **Significant domain shift** from pre-training data
- **Maximum performance** is critical
- **Large, high-quality dataset** available (>100K examples)

---

## Technical Deep-Dive: LoRA and Variants

### LoRA (Low-Rank Adaptation)

LoRA introduces a revolutionary approach: instead of updating the full weight matrix $W$, it decomposes the weight update into two low-rank matrices $A$ and $B$.

![LoRA Architecture: Low-Rank Adaptation with trainable A and B matrices](diagrams/05_lora_architecture.png)

**Key Insight**: The rank $r$ is typically 8-64, much smaller than $d$ (which can be 4096-8192 in modern LLMs). This reduces trainable parameters from $d^2$ to $2 \times d \times r$.

### Mathematical Foundation

The forward pass with LoRA:

$$h = Wx + \frac{\alpha}{r}BAx$$

Where:
- $W \in \mathbb{R}^{d \times d}$ is the frozen pre-trained weight
- $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$ are low-rank matrices
- $\alpha$ is a scaling factor
- $r$ is the rank (hyperparameter)

### LoRA Variants

#### LoRA-FA (Frozen-A)

LoRA-FA reduces activation memory by freezing matrix $A$ after random initialization, training only matrix $B$.

![LoRA-FA Architecture: Frozen-A variant for reduced activation memory](diagrams/06_lora_fa.png)

#### VeRA (Vector-based Random Adaptation)

VeRA takes efficiency further by sharing frozen random matrices across all layers and only training small scaling vectors.

![VeRA Architecture: Vector-based Random Adaptation with shared matrices](diagrams/07_vera.png)

#### Delta-LoRA

Delta-LoRA updates the base weight matrix $W$ using the difference between consecutive LoRA updates:

$$W_{t+1} = W_t + c(A_{t+1}B_{t+1} - A_tB_t)$$

![Delta-LoRA: Weight update mechanism using consecutive LoRA differences](diagrams/08_delta_lora.png)

#### LoRA+

LoRA+ optimizes convergence by using different learning rates for matrices $A$ and $B$:

![LoRA+ Learning Rate Strategy: Different rates for A and B matrices](diagrams/09_lora_plus.png)

**Research Finding**: Setting $\lambda = 16$ (i.e., 16× higher learning rate for $B$) often yields better convergence and final performance.

---

## Technical Deep-Dive: QLoRA

QLoRA combines quantization with LoRA to enable fine-tuning of massive models on consumer hardware.

### Key Innovations

1. **4-bit NormalFloat (NF4)**: An information-theoretically optimal quantization for normally distributed weights.

2. **Double Quantization**: Quantizes the quantization constants to further reduce memory.

3. **Paged Optimizers**: Uses NVIDIA unified memory to handle memory spikes during gradient checkpointing.

![QLoRA Architecture: 4-bit quantized base with full precision LoRA adapters](diagrams/10_qlora_architecture.png)

### Memory Breakdown

![Memory Distribution for 70B Model with QLoRA](diagrams/11_memory_distribution.png)

---

## Data Preparation Pipeline

Effective fine-tuning requires careful data preparation. Here's a production-ready pipeline:

![Data Preparation Pipeline for LLM Fine-Tuning](diagrams/12_data_pipeline.png)

### Complete Data Preparation Code

```python
#!/usr/bin/env python3
"""
Production-ready data preparation pipeline for LLM fine-tuning.
Compatible with Llama 4, Qwen 3, DeepSeek-V3.2, and Gemma 3.

Requirements:
    pip install datasets transformers torch pandas numpy tqdm
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data preparation pipeline."""
    model_name: str = "meta-llama/Llama-4-8B"
    max_seq_length: int = 2048
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    min_length: int = 10
    max_length: int = 4096
    deduplicate: bool = True
    quality_filter: bool = True
    seed: int = 42
    num_proc: int = 4


class DataPreparationPipeline:
    """End-to-end data preparation for LLM fine-tuning."""
    
    # Chat templates for different model families
    CHAT_TEMPLATES = {
        "llama": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
        "qwen": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>",
        "deepseek": "<|begin▁of▁sentence|>{system}\n\nUser: {user}\n\nAssistant: {assistant}<|end▁of▁sentence|>",
        "gemma": "<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>",
    }
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = self._load_tokenizer()
        self.model_family = self._detect_model_family()
        
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer with proper configuration."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return tokenizer
    
    def _detect_model_family(self) -> str:
        """Detect model family from model name."""
        model_lower = self.config.model_name.lower()
        if "llama" in model_lower:
            return "llama"
        elif "qwen" in model_lower:
            return "qwen"
        elif "deepseek" in model_lower:
            return "deepseek"
        elif "gemma" in model_lower:
            return "gemma"
        else:
            logger.warning(f"Unknown model family, defaulting to llama template")
            return "llama"
    
    def load_data(
        self, 
        source: str | Path | pd.DataFrame,
        text_column: str = "text",
        instruction_column: Optional[str] = None,
        response_column: Optional[str] = None,
    ) -> Dataset:
        """
        Load data from various sources.
        
        Args:
            source: Path to file, HuggingFace dataset name, or DataFrame
            text_column: Column containing text (for single-text format)
            instruction_column: Column with instructions (for instruction format)
            response_column: Column with responses (for instruction format)
        """
        if isinstance(source, pd.DataFrame):
            dataset = Dataset.from_pandas(source)
        elif isinstance(source, (str, Path)):
            source_str = str(source)
            if source_str.endswith('.json'):
                dataset = Dataset.from_json(source_str)
            elif source_str.endswith('.jsonl'):
                dataset = Dataset.from_json(source_str, field=None)
            elif source_str.endswith('.csv'):
                dataset = Dataset.from_csv(source_str)
            elif source_str.endswith('.parquet'):
                dataset = Dataset.from_parquet(source_str)
            else:
                # Assume HuggingFace dataset
                dataset = load_dataset(source_str, split="train")
        else:
            raise ValueError(f"Unsupported data source type: {type(source)}")
        
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove null bytes and other control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def deduplicate(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        """Remove duplicate entries based on content hash."""
        if not self.config.deduplicate:
            return dataset
        
        seen_hashes = set()
        indices_to_keep = []
        
        for idx, example in enumerate(tqdm(dataset, desc="Deduplicating")):
            text = example.get(text_column, "")
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                indices_to_keep.append(idx)
        
        original_len = len(dataset)
        dataset = dataset.select(indices_to_keep)
        removed = original_len - len(dataset)
        logger.info(f"Removed {removed} duplicates ({removed/original_len*100:.1f}%)")
        
        return dataset
    
    def quality_filter(self, dataset: Dataset, text_column: str = "text") -> Dataset:
        """Apply quality filters to the dataset."""
        if not self.config.quality_filter:
            return dataset
        
        def is_quality(example):
            text = example.get(text_column, "")
            
            # Length check
            if len(text) < self.config.min_length:
                return False
            if len(text) > self.config.max_length:
                return False
            
            # Basic quality heuristics
            alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
            if alpha_ratio < 0.5:  # At least 50% alphabetic characters
                return False
            
            # Check for excessive repetition
            words = text.lower().split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:  # Too repetitive
                    return False
            
            return True
        
        original_len = len(dataset)
        dataset = dataset.filter(is_quality, num_proc=self.config.num_proc)
        removed = original_len - len(dataset)
        logger.info(f"Quality filter removed {removed} examples ({removed/original_len*100:.1f}%)")
        
        return dataset
    
    def format_instruction(
        self,
        instruction: str,
        response: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Format instruction-response pair using model-specific template."""
        template = self.CHAT_TEMPLATES[self.model_family]
        
        return template.format(
            system=system_prompt,
            user=instruction,
            assistant=response,
        )
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
    ) -> Dataset:
        """Tokenize dataset for training."""
        
        def tokenize_function(examples):
            texts = examples[text_column]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors=None,
            )
            
            # For causal LM, labels are same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        
        return dataset
    
    def create_splits(self, dataset: Dataset) -> DatasetDict:
        """Split dataset into train, validation, and test sets."""
        # Shuffle first
        dataset = dataset.shuffle(seed=self.config.seed)
        
        # Calculate split sizes
        total = len(dataset)
        train_size = int(total * self.config.train_split)
        val_size = int(total * self.config.val_split)
        
        # Create splits
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total))
        
        splits = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        })
        
        logger.info(f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        return splits
    
    def process_instruction_dataset(
        self,
        dataset: Dataset,
        instruction_col: str = "instruction",
        response_col: str = "response",
        system_col: Optional[str] = None,
    ) -> Dataset:
        """Process an instruction-following dataset."""
        
        def format_example(example):
            instruction = self.clean_text(example[instruction_col])
            response = self.clean_text(example[response_col])
            system = example.get(system_col, "You are a helpful assistant.") if system_col else "You are a helpful assistant."
            
            formatted = self.format_instruction(instruction, response, system)
            return {"text": formatted}
        
        dataset = dataset.map(format_example, num_proc=self.config.num_proc, desc="Formatting")
        return dataset
    
    def run_pipeline(
        self,
        source: str | Path | pd.DataFrame,
        output_dir: str | Path = "./processed_data",
        instruction_col: Optional[str] = None,
        response_col: Optional[str] = None,
        text_col: str = "text",
    ) -> DatasetDict:
        """
        Run the complete data preparation pipeline.
        
        Args:
            source: Data source (path, HF dataset name, or DataFrame)
            output_dir: Directory to save processed data
            instruction_col: Column with instructions (for instruction format)
            response_col: Column with responses (for instruction format)
            text_col: Column with text (for pre-formatted data)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        dataset = self.load_data(source)
        
        # Step 2: Format instructions (if applicable)
        if instruction_col and response_col:
            logger.info("Step 2: Formatting instruction-response pairs...")
            dataset = self.process_instruction_dataset(
                dataset, instruction_col, response_col
            )
            text_col = "text"
        
        # Step 3: Clean text
        logger.info("Step 3: Cleaning text...")
        dataset = dataset.map(
            lambda x: {text_col: self.clean_text(x[text_col])},
            num_proc=self.config.num_proc,
            desc="Cleaning",
        )
        
        # Step 4: Deduplicate
        logger.info("Step 4: Deduplicating...")
        dataset = self.deduplicate(dataset, text_col)
        
        # Step 5: Quality filter
        logger.info("Step 5: Applying quality filters...")
        dataset = self.quality_filter(dataset, text_col)
        
        # Step 6: Tokenize
        logger.info("Step 6: Tokenizing...")
        dataset = self.tokenize_dataset(dataset, text_col)
        
        # Step 7: Create splits
        logger.info("Step 7: Creating train/val/test splits...")
        splits = self.create_splits(dataset)
        
        # Step 8: Save
        logger.info("Step 8: Saving processed data...")
        splits.save_to_disk(str(output_dir))
        
        # Save metadata
        metadata = {
            "model_name": self.config.model_name,
            "model_family": self.model_family,
            "max_seq_length": self.config.max_seq_length,
            "train_size": len(splits["train"]),
            "val_size": len(splits["validation"]),
            "test_size": len(splits["test"]),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Pipeline complete! Data saved to {output_dir}")
        
        return splits


def main():
    """Example usage of the data preparation pipeline."""
    
    # Configuration for Llama 4
    config = DataConfig(
        model_name="meta-llama/Llama-4-8B",
        max_seq_length=2048,
        train_split=0.9,
        val_split=0.05,
        test_split=0.05,
    )
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline(config)
    
    # Example: Process the Alpaca dataset
    splits = pipeline.run_pipeline(
        source="tatsu-lab/alpaca",
        output_dir="./processed_alpaca",
        instruction_col="instruction",
        response_col="output",
    )
    
    print(f"\nProcessed dataset statistics:")
    print(f"  Train: {len(splits['train']):,} examples")
    print(f"  Validation: {len(splits['validation']):,} examples")
    print(f"  Test: {len(splits['test']):,} examples")


if __name__ == "__main__":
    main()
```

---

## Implementation: Full Fine-Tuning

Full fine-tuning requires significant compute resources but offers the highest potential performance.

![Training Pipeline Flow: Setup, Training Loop, and Monitoring](diagrams/13_training_pipeline.png)

### Complete Full Fine-Tuning Code

```python
#!/usr/bin/env python3
"""
Full Fine-Tuning Pipeline for Large Language Models.
Supports: Llama 4, Qwen 3, DeepSeek-V3.2, Gemma 3

Requirements:
    pip install torch transformers datasets accelerate wandb tqdm
    pip install flash-attn --no-build-isolation  # Optional but recommended
"""

import os
import json
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FullFineTuningConfig:
    """Configuration for full fine-tuning."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-4-8B"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    
    # Optimization settings
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "fp16", "bf16", or "no"
    
    # Data settings
    data_dir: str = "./processed_data"
    max_seq_length: int = 2048
    
    # Output settings
    output_dir: str = "./full_finetuned_model"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Experiment tracking
    project_name: str = "llm-full-finetuning"
    run_name: Optional[str] = None
    use_wandb: bool = True
    
    # Hardware
    seed: int = 42


class FullFineTuner:
    """Production-ready full fine-tuning trainer."""
    
    def __init__(self, config: FullFineTuningConfig):
        self.config = config
        self.setup_accelerator()
        self.setup_seed()
        
    def setup_accelerator(self):
        """Initialize accelerator for distributed training."""
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            kwargs_handlers=[ddp_kwargs],
        )
        
        if self.accelerator.is_main_process:
            logger.info(f"Running on {self.accelerator.num_processes} processes")
            logger.info(f"Mixed precision: {self.config.mixed_precision}")
    
    def setup_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Determine torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": None,  # Let accelerator handle device placement
        }
        
        # Enable flash attention if available
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        # Enable gradient checkpointing to save memory
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model, self.tokenizer
    
    def load_data(self):
        """Load preprocessed datasets."""
        logger.info(f"Loading data from {self.config.data_dir}")
        
        dataset = load_from_disk(self.config.data_dir)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            dataset["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True,
        )
        
        self.eval_dataloader = DataLoader(
            dataset["validation"],
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=4,
            pin_memory=True,
        )
        
        logger.info(f"Train batches: {len(self.train_dataloader)}")
        logger.info(f"Eval batches: {len(self.eval_dataloader)}")
        
        return self.train_dataloader, self.eval_dataloader
    
    def setup_optimizer_and_scheduler(self):
        """Configure optimizer and learning rate scheduler."""
        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        self.total_training_steps = num_update_steps_per_epoch * self.config.num_epochs
        self.warmup_steps = int(self.total_training_steps * self.config.warmup_ratio)
        
        # Setup optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        
        logger.info(f"Total training steps: {self.total_training_steps}")
        logger.info(f"Warmup steps: {self.warmup_steps}")
        
        return self.optimizer, self.scheduler
    
    def setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            return
        
        if self.accelerator.is_main_process:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config),
            )
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Gather losses across processes
                gathered_loss = self.accelerator.gather(loss.repeat(self.config.batch_size))
                total_loss += gathered_loss.sum().item()
                total_tokens += batch["input_ids"].numel() * self.accelerator.num_processes
        
        avg_loss = total_loss / len(self.eval_dataloader)
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
        
        self.model.train()
        return {"eval_loss": avg_loss, "eval_perplexity": perplexity}
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Unwrap model and save
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        torch.save({
            "step": step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }, output_dir / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {output_dir}")
    
    def train(self):
        """Main training loop."""
        # Setup
        self.load_model_and_tokenizer()
        self.load_data()
        self.setup_optimizer_and_scheduler()
        self.setup_wandb()
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.scheduler
            )
        
        # Training loop
        global_step = 0
        best_eval_loss = float("inf")
        
        logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                
                if self.accelerator.sync_gradients:
                    global_step += 1
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        lr = self.scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })
                        
                        if self.config.use_wandb and WANDB_AVAILABLE and self.accelerator.is_main_process:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch + step / len(self.train_dataloader),
                            }, step=global_step)
                    
                    # Evaluation
                    if global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        if self.accelerator.is_main_process:
                            logger.info(f"Step {global_step}: {eval_metrics}")
                            
                            if self.config.use_wandb and WANDB_AVAILABLE:
                                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=global_step)
                            
                            if eval_metrics["eval_loss"] < best_eval_loss:
                                best_eval_loss = eval_metrics["eval_loss"]
                                self.save_checkpoint(global_step)
                    
                    # Regular checkpointing
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step)
        
        # Final save
        self.save_checkpoint(global_step)
        
        if self.config.use_wandb and WANDB_AVAILABLE and self.accelerator.is_main_process:
            wandb.finish()
        
        logger.info("Training complete!")
        return global_step


def main():
    """Run full fine-tuning."""
    
    config = FullFineTuningConfig(
        model_name="meta-llama/Llama-4-8B",
        learning_rate=2e-5,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=8,
        data_dir="./processed_data",
        output_dir="./full_finetuned_model",
    )
    
    trainer = FullFineTuner(config)
    trainer.train()


if __name__ == "__main__":
    main()
```

---

## Implementation: LoRA Fine-Tuning

LoRA dramatically reduces memory requirements while maintaining near full fine-tuning performance.

![LoRA Fine-Tuning Pipeline: Model setup through post-training](diagrams/14_lora_training_setup.png)

### Complete LoRA Fine-Tuning Code

```python
#!/usr/bin/env python3
"""
LoRA Fine-Tuning Pipeline for Large Language Models.
Supports: Llama 4, Qwen 3, DeepSeek-V3.2, Gemma 3

Requirements:
    pip install torch transformers datasets peft accelerate wandb tqdm bitsandbytes
"""

import os
import math
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from datasets import load_from_disk
from tqdm.auto import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-4-8B"
    torch_dtype: str = "bfloat16"
    use_flash_attention: bool = True
    trust_remote_code: bool = True
    
    # LoRA hyperparameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])
    use_rslora: bool = True  # Rank-stabilized LoRA
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # LoRA+ settings (different LR for A and B matrices)
    use_lora_plus: bool = True
    lora_plus_lambda: float = 16.0  # B learning rate multiplier
    
    # Data settings
    data_dir: str = "./processed_data"
    max_seq_length: int = 2048
    
    # Output settings
    output_dir: str = "./lora_finetuned_model"
    save_steps: int = 200
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Experiment tracking
    project_name: str = "llm-lora-finetuning"
    run_name: Optional[str] = None
    use_wandb: bool = True
    
    seed: int = 42


class LoRAFineTuner:
    """Production-ready LoRA fine-tuning trainer."""
    
    # Target modules for different model architectures
    TARGET_MODULES_MAP = {
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.model_family = self._detect_model_family()
        
        # Update target modules based on model family
        if not config.target_modules:
            config.target_modules = self.TARGET_MODULES_MAP.get(
                self.model_family, 
                self.TARGET_MODULES_MAP["llama"]
            )
    
    def _detect_model_family(self) -> str:
        """Detect model family from model name."""
        model_lower = self.config.model_name.lower()
        for family in ["llama", "qwen", "deepseek", "gemma"]:
            if family in model_lower:
                return family
        return "llama"
    
    def load_model_and_tokenizer(self):
        """Load base model and apply LoRA."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": "auto",
        }
        
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            modules_to_save=self.config.modules_to_save,
            bias="none",
            use_rslora=self.config.use_rslora,
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model, self.tokenizer
    
    def get_optimizer_grouped_parameters(self):
        """Get optimizer parameters with LoRA+ learning rate scheduling."""
        if not self.config.use_lora_plus:
            return None  # Use default optimizer
        
        # LoRA+ assigns higher learning rate to B matrices
        lora_a_params = []
        lora_b_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "lora_A" in name:
                lora_a_params.append(param)
            elif "lora_B" in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": lora_a_params,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": lora_b_params,
                "lr": self.config.learning_rate * self.config.lora_plus_lambda,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": other_params,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
        ]
        
        logger.info(f"LoRA+ enabled: A matrices LR = {self.config.learning_rate:.2e}, "
                   f"B matrices LR = {self.config.learning_rate * self.config.lora_plus_lambda:.2e}")
        
        return optimizer_grouped_parameters
    
    def load_data(self):
        """Load preprocessed datasets."""
        logger.info(f"Loading data from {self.config.data_dir}")
        self.dataset = load_from_disk(self.config.data_dir)
        return self.dataset
    
    def create_trainer(self):
        """Create HuggingFace Trainer with custom optimizer."""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=self.config.torch_dtype == "bfloat16",
            fp16=self.config.torch_dtype == "float16",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to="wandb" if self.config.use_wandb and WANDB_AVAILABLE else "none",
            run_name=self.config.run_name,
            seed=self.config.seed,
            remove_unused_columns=False,
        )
        
        # Custom optimizer for LoRA+
        optimizers = (None, None)  # Default
        if self.config.use_lora_plus:
            from torch.optim import AdamW
            optimizer_grouped_parameters = self.get_optimizer_grouped_parameters()
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95), eps=1e-8)
            optimizers = (optimizer, None)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            optimizers=optimizers,
        )
        
        return self.trainer
    
    def train(self):
        """Run the complete training pipeline."""
        # Setup
        self.load_model_and_tokenizer()
        self.load_data()
        self.create_trainer()
        
        # Initialize wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config),
            )
        
        # Train
        logger.info("Starting LoRA training...")
        train_result = self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", eval_metrics)
        self.trainer.save_metrics("eval", eval_metrics)
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        logger.info("Training complete!")
        return metrics
    
    def merge_and_save(self, output_path: str):
        """Merge LoRA weights with base model and save."""
        logger.info("Merging LoRA weights with base model...")
        
        # Merge weights
        merged_model = self.model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")


def main():
    """Run LoRA fine-tuning."""
    
    config = LoRAConfig(
        model_name="meta-llama/Llama-4-8B",
        lora_r=64,
        lora_alpha=128,
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=8,
        gradient_accumulation_steps=4,
        data_dir="./processed_data",
        output_dir="./lora_finetuned_model",
        use_lora_plus=True,
    )
    
    trainer = LoRAFineTuner(config)
    trainer.train()
    
    # Optionally merge and save
    trainer.merge_and_save("./merged_model")


if __name__ == "__main__":
    main()
```

---

## Implementation: QLoRA Fine-Tuning

QLoRA enables fine-tuning of the largest models on consumer hardware through 4-bit quantization.

![QLoRA Fine-Tuning Pipeline: Quantization, LoRA setup, and memory management](diagrams/15_qlora_setup.png)

### Complete QLoRA Fine-Tuning Code

```python
#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Pipeline for Large Language Models.
Enables fine-tuning of massive models on consumer GPUs.

Supports: Llama 4, Qwen 3, DeepSeek-V3.2, Gemma 3

Requirements:
    pip install torch transformers datasets peft accelerate wandb tqdm
    pip install bitsandbytes  # Required for 4-bit quantization
"""

import os
import math
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from datasets import load_from_disk
from tqdm.auto import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-4-8B"
    trust_remote_code: bool = True
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"  # nf4 or fp4
    bnb_4bit_use_double_quant: bool = True  # Double quantization for extra memory savings
    
    # LoRA hyperparameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 0.3  # Lower for QLoRA stability
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimizer settings (for QLoRA, use paged optimizers)
    optim: str = "paged_adamw_8bit"  # Memory-efficient optimizer
    
    # Data settings
    data_dir: str = "./processed_data"
    max_seq_length: int = 2048
    
    # Output settings
    output_dir: str = "./qlora_finetuned_model"
    save_steps: int = 200
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Experiment tracking
    project_name: str = "llm-qlora-finetuning"
    run_name: Optional[str] = None
    use_wandb: bool = True
    
    seed: int = 42


class QLoRAFineTuner:
    """Production-ready QLoRA fine-tuning trainer."""
    
    def __init__(self, config: QLoRAConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings."""
        if self.config.load_in_4bit:
            try:
                import bitsandbytes
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for 4-bit quantization. "
                    "Install with: pip install bitsandbytes"
                )
    
    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytes quantization configuration."""
        compute_dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        compute_dtype = compute_dtype_map.get(
            self.config.bnb_4bit_compute_dtype, 
            torch.bfloat16
        )
        
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )
    
    def load_model_and_tokenizer(self):
        """Load quantized model and apply LoRA."""
        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Quantization: 4-bit {self.config.bnb_4bit_quant_type}")
        
        # Quantization config
        bnb_config = self._get_quantization_config()
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            modules_to_save=self.config.modules_to_save,
            bias="none",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print memory usage and trainable parameters
        self._print_model_info()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model, self.tokenizer
    
    def _print_model_info(self):
        """Print model information and memory usage."""
        self.model.print_trainable_parameters()
        
        # Estimate memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    def load_data(self):
        """Load preprocessed datasets."""
        logger.info(f"Loading data from {self.config.data_dir}")
        self.dataset = load_from_disk(self.config.data_dir)
        return self.dataset
    
    def create_trainer(self):
        """Create HuggingFace Trainer optimized for QLoRA."""
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Training arguments optimized for QLoRA
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,  # Paged optimizer for memory efficiency
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,  # Use BF16 for compute
            tf32=True,  # Enable TF32 on Ampere+ GPUs
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="wandb" if self.config.use_wandb and WANDB_AVAILABLE else "none",
            run_name=self.config.run_name,
            seed=self.config.seed,
            remove_unused_columns=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
        )
        
        return self.trainer
    
    def train(self):
        """Run the complete QLoRA training pipeline."""
        # Setup
        self.load_model_and_tokenizer()
        self.load_data()
        self.create_trainer()
        
        # Initialize wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=vars(self.config),
            )
        
        # Train
        logger.info("Starting QLoRA training...")
        train_result = self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", eval_metrics)
        self.trainer.save_metrics("eval", eval_metrics)
        
        # Log final memory usage
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Peak GPU memory usage: {max_memory:.2f} GB")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        logger.info("Training complete!")
        return metrics
    
    def merge_and_save(self, output_path: str, safe_serialization: bool = True):
        """
        Merge LoRA weights with dequantized base model.
        Note: This requires enough memory to hold the full model in FP16.
        """
        logger.info("Merging QLoRA weights with base model...")
        logger.warning("This requires loading the full model in FP16. Ensure sufficient memory.")
        
        # Load base model in FP16
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Load and merge LoRA weights
        from peft import PeftModel
        merged_model = PeftModel.from_pretrained(base_model, self.config.output_dir)
        merged_model = merged_model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(
            output_path, 
            safe_serialization=safe_serialization,
        )
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")


def estimate_memory_requirements(model_name: str, batch_size: int = 4, seq_length: int = 2048):
    """
    Estimate GPU memory requirements for QLoRA training.
    
    Returns estimated memory in GB.
    """
    # Rough estimates based on model size
    model_params = {
        "7B": 7e9,
        "8B": 8e9,
        "13B": 13e9,
        "30B": 30e9,
        "65B": 65e9,
        "70B": 70e9,
    }
    
    # Extract size from model name
    size = None
    for key in model_params:
        if key.lower() in model_name.lower():
            size = model_params[key]
            break
    
    if size is None:
        logger.warning("Could not estimate model size, assuming 7B parameters")
        size = 7e9
    
    # Memory components for QLoRA
    # 4-bit weights: params * 0.5 bytes
    quantized_weights = size * 0.5 / 1024**3
    
    # LoRA adapters (FP16): ~0.1% of params * 2 bytes
    lora_weights = size * 0.001 * 2 / 1024**3
    
    # Optimizer states (8-bit paged): ~2 bytes per LoRA param
    optimizer_states = size * 0.001 * 2 / 1024**3
    
    # Activations (rough estimate)
    activations = batch_size * seq_length * 4096 * 4 / 1024**3  # Assume 4096 hidden dim
    
    total = quantized_weights + lora_weights + optimizer_states + activations
    
    logger.info(f"""
    Estimated GPU Memory for QLoRA:
    - Quantized weights: {quantized_weights:.2f} GB
    - LoRA adapters: {lora_weights:.2f} GB
    - Optimizer states: {optimizer_states:.2f} GB
    - Activations: {activations:.2f} GB
    - Total: {total:.2f} GB
    """)
    
    return total


def main():
    """Run QLoRA fine-tuning."""
    
    # Estimate memory requirements first
    estimate_memory_requirements("meta-llama/Llama-4-8B", batch_size=4)
    
    config = QLoRAConfig(
        model_name="meta-llama/Llama-4-8B",
        lora_r=64,
        lora_alpha=128,
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=8,
        data_dir="./processed_data",
        output_dir="./qlora_finetuned_model",
    )
    
    trainer = QLoRAFineTuner(config)
    trainer.train()


if __name__ == "__main__":
    main()
```

---

## Evaluation and Metrics

Proper evaluation is critical for understanding model performance and preventing overfitting.

![Evaluation Framework: Metrics, Benchmarks, and Analysis Tools](diagrams/16_evaluation_metrics.png)

### Complete Evaluation Code

```python
#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for Fine-Tuned LLMs.
Supports multiple metrics, benchmarks, and analysis tools.

Requirements:
    pip install torch transformers datasets evaluate nltk rouge-score sacrebleu
    pip install lm-eval  # For standard benchmarks
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Model settings
    model_path: str = "./finetuned_model"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = True
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Evaluation settings
    batch_size: int = 8
    num_samples: Optional[int] = None  # None = use all
    
    # Metrics to compute
    compute_perplexity: bool = True
    compute_rouge: bool = True
    compute_bleu: bool = True
    
    # Output
    output_dir: str = "./evaluation_results"


class LLMEvaluator:
    """Comprehensive evaluation suite for fine-tuned LLMs."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._load_model()
        self._load_metrics()
    
    def _load_model(self):
        """Load the fine-tuned model."""
        logger.info(f"Loading model from {self.config.model_path}")
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _load_metrics(self):
        """Load evaluation metrics."""
        self.metrics = {}
        
        if self.config.compute_rouge:
            self.metrics["rouge"] = evaluate.load("rouge")
        
        if self.config.compute_bleu:
            self.metrics["bleu"] = evaluate.load("sacrebleu")
    
    @torch.no_grad()
    def compute_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """Compute perplexity on a list of texts."""
        logger.info("Computing perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        for text in tqdm(texts, desc="Perplexity"):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].numel()
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            "perplexity": float(perplexity),
            "avg_loss": float(avg_loss),
            "total_tokens": total_tokens,
        }
    
    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate responses for a list of prompts."""
        logger.info(f"Generating responses for {len(prompts)} prompts...")
        
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        responses = []
        
        for prompt in tqdm(prompts, desc="Generating"):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - self.config.max_new_tokens,
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)
        
        return responses
    
    def compute_rouge_scores(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
        logger.info("Computing ROUGE scores...")
        
        results = self.metrics["rouge"].compute(
            predictions=predictions,
            references=references,
        )
        
        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
            "rougeLsum": results["rougeLsum"],
        }
    
    def compute_bleu_score(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """Compute BLEU score."""
        logger.info("Computing BLEU score...")
        
        results = self.metrics["bleu"].compute(
            predictions=predictions,
            references=references,
        )
        
        return {
            "bleu": results["score"],
            "precisions": results["precisions"],
        }
    
    def evaluate_instruction_following(
        self,
        dataset: Dataset,
        instruction_col: str = "instruction",
        response_col: str = "response",
    ) -> Dict[str, Any]:
        """Evaluate instruction-following capability."""
        logger.info("Evaluating instruction following...")
        
        # Limit samples if specified
        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
        
        # Extract prompts and references
        prompts = dataset[instruction_col]
        references = dataset[response_col]
        
        # Generate responses
        predictions = self.generate_responses(prompts)
        
        results = {}
        
        # ROUGE scores
        if self.config.compute_rouge:
            results["rouge"] = self.compute_rouge_scores(predictions, references)
        
        # BLEU score
        if self.config.compute_bleu:
            # BLEU expects list of reference lists
            ref_lists = [[ref] for ref in references]
            results["bleu"] = self.compute_bleu_score(predictions, ref_lists)
        
        # Save sample outputs
        results["samples"] = [
            {
                "instruction": p,
                "reference": r,
                "prediction": pred,
            }
            for p, r, pred in zip(prompts[:10], references[:10], predictions[:10])
        ]
        
        return results
    
    def run_lm_eval_harness(
        self,
        tasks: List[str] = ["hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        num_fewshot: int = 0,
    ) -> Dict[str, Any]:
        """
        Run evaluation using lm-evaluation-harness.
        
        Requires: pip install lm-eval
        """
        logger.info(f"Running lm-eval-harness on tasks: {tasks}")
        
        try:
            from lm_eval import evaluator, tasks as lm_tasks
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            logger.error("lm-eval not installed. Run: pip install lm-eval")
            return {"error": "lm-eval not installed"}
        
        # Create LM object
        lm = HFLM(
            pretrained=self.config.model_path,
            dtype=self.config.torch_dtype,
            batch_size=self.config.batch_size,
        )
        
        # Run evaluation
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
        )
        
        return results
    
    def analyze_errors(
        self,
        prompts: List[str],
        predictions: List[str],
        references: List[str],
        categorize_fn: Optional[Callable[[str, str, str], str]] = None,
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        logger.info("Analyzing errors...")
        
        errors_by_category = defaultdict(list)
        
        for prompt, pred, ref in zip(prompts, predictions, references):
            # Simple error detection: check if prediction differs significantly
            if pred.strip().lower() != ref.strip().lower():
                if categorize_fn:
                    category = categorize_fn(prompt, pred, ref)
                else:
                    # Default categorization by length difference
                    len_diff = len(pred) - len(ref)
                    if len_diff > 100:
                        category = "too_long"
                    elif len_diff < -100:
                        category = "too_short"
                    else:
                        category = "content_mismatch"
                
                errors_by_category[category].append({
                    "prompt": prompt[:200],
                    "prediction": pred[:200],
                    "reference": ref[:200],
                })
        
        analysis = {
            "total_errors": sum(len(v) for v in errors_by_category.values()),
            "errors_by_category": {k: len(v) for k, v in errors_by_category.items()},
            "sample_errors": {k: v[:3] for k, v in errors_by_category.items()},
        }
        
        return analysis
    
    def run_full_evaluation(
        self,
        test_dataset: Optional[Dataset] = None,
        test_texts: Optional[List[str]] = None,
        instruction_col: str = "instruction",
        response_col: str = "response",
        run_benchmarks: bool = False,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        results = {}
        
        # Perplexity evaluation
        if test_texts and self.config.compute_perplexity:
            results["perplexity"] = self.compute_perplexity(test_texts)
        
        # Instruction following evaluation
        if test_dataset:
            results["instruction_following"] = self.evaluate_instruction_following(
                test_dataset, instruction_col, response_col
            )
        
        # Standard benchmarks (optional)
        if run_benchmarks:
            results["benchmarks"] = self.run_lm_eval_harness()
        
        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir / 'evaluation_results.json'}")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if "perplexity" in results:
            print(f"\nPerplexity: {results['perplexity']['perplexity']:.2f}")
        
        if "instruction_following" in results:
            if "rouge" in results["instruction_following"]:
                rouge = results["instruction_following"]["rouge"]
                print(f"\nROUGE Scores:")
                print(f"  ROUGE-1: {rouge['rouge1']:.4f}")
                print(f"  ROUGE-2: {rouge['rouge2']:.4f}")
                print(f"  ROUGE-L: {rouge['rougeL']:.4f}")
            
            if "bleu" in results["instruction_following"]:
                print(f"\nBLEU Score: {results['instruction_following']['bleu']['bleu']:.2f}")
        
        if "benchmarks" in results and "results" in results["benchmarks"]:
            print("\nBenchmark Results:")
            for task, scores in results["benchmarks"]["results"].items():
                if "acc" in scores:
                    print(f"  {task}: {scores['acc']:.4f}")
        
        print("\n" + "="*60)


def main():
    """Run evaluation on a fine-tuned model."""
    
    config = EvaluationConfig(
        model_path="./finetuned_model",
        output_dir="./evaluation_results",
        batch_size=8,
        num_samples=100,
    )
    
    evaluator = LLMEvaluator(config)
    
    # Load test dataset (example: Alpaca)
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
    test_texts = [d["output"] for d in dataset]
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        test_dataset=dataset,
        test_texts=test_texts,
        instruction_col="instruction",
        response_col="output",
        run_benchmarks=False,  # Set to True to run standard benchmarks
    )
    
    # Print summary
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
```

---

## Best Practices and Optimization Tips

### Data Quality

![Memory Requirements Comparison for 70B Model Fine-Tuning](diagrams/17_memory_comparison.png)

### Hyperparameter Selection Guide

| Parameter | Full FT | LoRA | QLoRA | Notes |
|-----------|---------|------|-------|-------|
| Learning Rate | 1e-5 - 5e-5 | 1e-4 - 3e-4 | 1e-4 - 3e-4 | QLoRA can use same as LoRA |
| Batch Size | 32-128 | 16-64 | 4-16 | Limited by memory |
| Epochs | 1-3 | 1-3 | 2-4 | QLoRA may need more |
| Warmup Ratio | 0.03-0.1 | 0.03-0.1 | 0.03-0.1 | Standard across all |
| Max Grad Norm | 1.0 | 1.0 | 0.3 | Lower for QLoRA stability |
| Weight Decay | 0.01-0.1 | 0.01 | 0.01 | Lower for LoRA methods |
| LoRA r | N/A | 32-128 | 64-256 | Higher r = more capacity |
| LoRA α | N/A | 2×r | 2×r | Common heuristic |

### Memory Optimization Strategies

![Fine-Tuning Method Selection Guide](diagrams/18_decision_flowchart.png)

### Common Pitfalls and Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Overfitting | Val loss increases | Early stopping, more data, regularization |
| Catastrophic forgetting | Base capabilities degrade | Lower LR, LoRA, replay buffer |
| Gradient explosion | NaN losses | Lower LR, gradient clipping |
| Mode collapse | Repetitive outputs | Temperature, nucleus sampling |
| Slow convergence | Loss plateaus early | Higher LR, lr scheduling |

---

## Comparison of Approaches

### Decision Framework

![Memory Optimization Strategies for LLM Training](diagrams/19_memory_optimization.png)

### Comprehensive Comparison

| Aspect | Full Fine-Tuning | LoRA | QLoRA |
|--------|-----------------|------|-------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Memory Efficiency** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Training Speed** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Inference Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ (merged) | ⭐⭐⭐ |
| **Ease of Use** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Hardware Requirement** | Multiple A100s | Single A100 | Consumer GPU |

### Cost Comparison (70B Model, 10K Examples)

| Approach | Hardware | Training Time | Est. Cloud Cost |
|----------|----------|---------------|-----------------|
| Full FT | 8× A100 80GB | ~24 hours | ~$800-1200 |
| LoRA | 2× A100 80GB | ~12 hours | ~$150-250 |
| QLoRA | 1× A100 40GB | ~18 hours | ~$100-150 |
| QLoRA | 1× RTX 4090 | ~48 hours | Local hardware |

---

## Conclusion

Fine-tuning Large Language Models has evolved from an exclusively enterprise endeavor to something achievable on consumer hardware, thanks to innovations like LoRA and QLoRA. This guide has covered:

1. **The fundamentals** of why and when to fine-tune LLMs
2. **Three primary approaches**: Full fine-tuning, LoRA, and QLoRA
3. **Advanced techniques**: LoRA variants including LoRA-FA, VeRA, Delta-LoRA, and LoRA+
4. **Production-ready code** for data preparation, training, and evaluation
5. **Best practices** for achieving optimal results

### Key Takeaways

- **Start with QLoRA** if you have limited GPU memory—it's remarkably effective
- **Data quality trumps quantity**—focus on high-quality, diverse training examples
- **Use LoRA+** for potentially better convergence without additional complexity
- **Monitor validation metrics** carefully to prevent overfitting
- **Merge adapters** for deployment to eliminate inference overhead

### Next Steps

1. **Experiment** with different LoRA ranks and target modules
2. **Try advanced variants** like DoRA or AdaLoRA for specific use cases
3. **Implement continuous training** pipelines for ongoing improvement
4. **Explore RLHF** for alignment and preference optimization

The field continues to evolve rapidly, with new techniques emerging regularly. Stay updated with the latest research, and don't hesitate to experiment—the best configuration often depends on your specific use case and data.

---

## References

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
2. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs
3. Zhang, Q., et al. (2023). LoRA-FA: Memory-efficient Low-rank Adaptation
4. Kopiczko, D., et al. (2024). VeRA: Vector-based Random Matrix Adaptation
5. Zi, B., et al. (2024). Delta-LoRA: Fine-Tuning High-Rank Parameters
6. Hayou, S., et al. (2024). LoRA+: Efficient Low Rank Adaptation with Optimal Learning

---

*Last updated: February 2026*
