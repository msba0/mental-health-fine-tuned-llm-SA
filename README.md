# Mental-Health Fine-Tuned LLM (SA)

This project fine-tunes a small Large Language Model — Qwen2.5-0.5B-Instruct — to generate empathetic, counselling-style responses for mental-health related questions.


### The fine-tuning uses:

  - CounselChat dataset (public counselling Q&A)

  - LoRA parameter-efficient fine-tuning (only ~1.7% of model parameters trained)

  - CPU-only training, ensuring the notebook runs on any machine

  - Before/After comparison between the baseline model and the fine-tuned model




### To run the demo:
**1. Clone the repository**
```
git clone https://github.com/msba0/mental-health-fine-tuned-llm-SA.git
```
```
cd mental-health-fine-tuned-llm-SA
```

**2. Create the environment**
```
conda env create -f environment.yml
```
```
conda activate psyche-r1
```


