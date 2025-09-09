
# 📖 Differentiable Fuzzy Mixture of Adapters (DF-MoA)

## Overview

**Differentiable Fuzzy Mixture of Adapters (DF-MoA)** is a new parameter-efficient fine-tuning (PEFT) method that extends **LoRA** and **Mixture-of-Experts (MoE)** with **differentiable adaptation**.

Instead of fixing:

* LoRA rank (`r`),
* entropy regularization strength,
* or adapter selection strategy …

👉 **DF-MoA learns them end-to-end**, driven by gradient descent.

This results in a system that can **dynamically adjust capacity, diversity, and specialization** during fine-tuning without manual hyperparameter tuning.

---

## ✨ Key Innovations

1. **Differentiable LoRA Rank**

   * Traditional LoRA requires a fixed rank `r`.
   * DF-MoA introduces **soft rank masks** (`sigmoid`-gated dimensions).
   * Effective rank is learned, with sparsity regularization encouraging minimal capacity.

2. **Trainable Entropy Weight**

   * Standard fuzzy gating uses a fixed entropy regularizer.
   * DF-MoA replaces it with a **trainable parameter**.
   * Model learns how much diversity to enforce in routing.

3. **Fuzzy Gating Network**

   * Soft mixture of multiple LoRA adapters (like MoE).
   * Gating distribution is fully differentiable.
   * Adapters specialize automatically based on data and entropy pressure.

4. **End-to-End Differentiability**

   * All knobs (entropy strength, rank usage, gate weights) are optimized by SGD.
   * Removes need for manual heuristics (e.g., “double rank on plateau”).

---

## 🧠 Theory

DF-MoA builds on the hypothesis that:

> *Neural networks should adapt their parameter-efficient capacity and specialization continuously, not through fixed hyperparameters.*

By integrating:

* **LoRA (efficient adaptation)**
* **MoE (routing + specialization)**
* **Differentiable architecture search (soft rank masks)**
* **Meta-learning (trainable regularization)**

DF-MoA creates a framework for **self-tuning fine-tuning**.

---

## 🚀 Usage

### Install dependencies

```bash
pip install torch transformers datasets peft bitsandbytes
```

### Training Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from df_moa import DifferentiableFuzzyMoAForCausalLM

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained("gpt2", quantization_config=quant_config, device_map="auto")

model = DifferentiableFuzzyMoAForCausalLM(base, num_adapters=3, r_max=64, lora_alpha=16).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("The future of AI is", return_tensors="pt").to("cuda")

outputs = model(**inputs, labels=inputs["input_ids"])
print("Loss:", outputs.loss.item())
print("Effective ranks:", model.effective_ranks())
```

---

## 📊 Features

* ✅ **Adaptive LoRA Rank** — capacity grows/shrinks via gradient descent.
* ✅ **Trainable Entropy Weight** — no need to hand-tune gate regularization.
* ✅ **Mixture of Adapters** — task-specific specialization emerges automatically.
* ✅ **Quantization-ready** — works with 4-bit (`bitsandbytes`).

---

## 🔬 Research Directions

* How does DF-MoA compare to static LoRA on low-resource tasks?
* Do different tasks encourage different effective ranks?
* Does trainable entropy improve adapter diversity in practice?
* Can DF-MoA serve as a lightweight alternative to full Mixture-of-Experts?

---

## 📜 Citation (conceptual)

```
@misc{dfmoa2025,
  title={Differentiable Fuzzy Mixture of Adapters: End-to-End Adaptive Parameter-Efficient Fine-Tuning},
  author={Your Name},
  year={2025},
  howpublished={GitHub},
}
```

---

⚡ With DF-MoA, your model doesn’t just adapt **weights**, it adapts its **capacity and routing strategy** as well.
