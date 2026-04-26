# Prompt Sensitivity in Generative AI

**Authors:** Purav Khurana, Utsav Singh, Birinder Singh  
**Institution:** Dept. of CSE, Chitkara University, Punjab, India  
**Paper:** *Prompt Sensitivity in Generative AI: How Minor Input Variations Drive Output Instability in Large Language Models*  
**Conference:** AIC2026 (Submitted)

---

## Overview

This repository contains the full experimental code for our research on **prompt fragility** in Large Language Models (LLMs). We test how three types of prompt changes affect outputs from GPT-4, Claude 3 Sonnet, and Gemini 2.5 Pro across 30 baseline prompts.

### Three Perturbation Types Tested
| Type | Description | Avg. Similarity Score |
|------|-------------|----------------------|
| Syntactic Restructuring | Active-to-passive, clause inversion | 0.61 |
| Punctuation Modification | Period → Question mark, etc. | 0.68 |
| Synonym Substitution | WordNet synonym replacement | 0.84 |

### Key Finding
Syntactic restructuring is the most destabilising perturbation type. A five-point prompt engineering framework (G1–G5) achieves **91% output stability** without any model retraining.

---

## Project Structure

```
prompt_sensitivity/
│
├── main.py               # Main experiment script
├── requirements.txt      # Python dependencies
├── README.md             # This file
│
└── results/              # Auto-generated after running
    ├── experiment_results.csv
    ├── fig1_avg_similarity.png
    ├── fig4_domain_sensitivity.png
    ├── fig5_stability_comparison.png
    └── fig6_variance_reduction.png
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/prompt-sensitivity.git
cd prompt-sensitivity
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set API keys (for live experiments)
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

### 4. Run the experiment
```bash
# Mock mode (no API keys needed - for testing)
python main.py

# Live mode (requires API keys - set use_mock=False in main.py)
python main.py
```

---

## Prompt Engineering Framework (G1–G5)

| Guideline | Target | Failure Mode Addressed | Variance Reduction |
|-----------|--------|----------------------|-------------------|
| G1: Syntactic Consistency | Syntactic Restructuring | Scope drift, factual deviation | ~26% |
| G2: Punctuation Standardisation | Punctuation Modification | Instruction misinterpretation | ~16% |
| G3: Lexical Register | Synonym Substitution | Tonal inconsistency | ~8% |
| G4: Scope Delimiters | Syntactic, Punctuation | Response scope expansion | ~20% |
| G5: Output Validation | All types | Residual divergence | ~6% |

---

## Experimental Parameters

- **Baseline Prompts:** 30 (10 per domain: Factual QA, Instructional, Creative)
- **LLMs Evaluated:** GPT-4, Claude 3 Sonnet, Gemini 2.5 Pro
- **Temperature:** 0.7 (fixed)
- **Trials per pair:** 5 (averaged for stability)
- **Similarity Model:** Sentence-BERT (all-MiniLM-L6-v2)
- **Similarity Threshold:** 0.75

---

## Results

Full results are saved to `results/experiment_results.csv` after running.
