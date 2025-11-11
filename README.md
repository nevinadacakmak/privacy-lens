# PrivacyLens

PrivacyLens is a prototype framework to **evaluate memorization and privacy risks in large language models (LLMs)**.  
It demonstrates how fine-tuned models can memorize sensitive identifiers and explores how **differential privacy (DP)** mitigates leakage while balancing utility.

- Measuring memorization with **canary insertion tests** and **membership inference**.
- Applying **differentially private stochastic gradient descent (DP-SGD)** using [Opacus](https://opacus.ai/).
- Reporting trade-offs between model utility (accuracy, perplexity) and privacy guarantees (leakage reduction, ε).

## Features

- **Classification task (DistilBERT)**

  - Fine-tunes DistilBERT on text classification datasets containing synthetic identifiers.
  - Integrates **Opacus** for DP-SGD training.
  - Reports trade-offs between accuracy, privacy budgets (ε), and leakage scores.

- **Language modeling task (GPT-2)**

  - Fine-tunes GPT-2 on toy data with _canary strings_ (unique identifiers).
  - Tests for memorization by prompting the model.
  - Demonstrates privacy leakage in generative LLMs.

- **Evaluation metrics**
  - Accuracy / Perplexity for utility.
  - ε (privacy budget) for differential privacy guarantees.
  - Canary extraction rate for memorization risk.

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/nevinadacakmak/privacy-lens.git
cd privacy-lens
pip install -r requirements.txt
```

---

## Repository Structure

- `notebooks/` – Reproducible experiments and analysis.
- `scripts/` – Modular training and evaluation code.
- `data/` – Scripts or links for dataset preparation.
- `results/` – Experiment logs and plots.

## Usage

### 1. DistilBERT with Differential Privacy

Train a classifier on synthetic data with/without DP:

```bash
python train_distilbert_dp.py --dp --epsilon 8 --delta 1e-5
```

Arguments:

- `--dp`: enable differential privacy (default: off)
- `--epsilon`: target privacy budget
- `--delta`: privacy delta
- `--epochs`: number of epochs
- `--batch_size`: training batch size

Results are logged as accuracy, ε, and leakage scores.

---

### GPT-2 Leakage Demo

Fine-tune GPT-2 on toy data with a _canary identifier_:

```bash
python gpt2_leakage.py
```

If GPT-2 memorized the canary, you may see outputs like:

```
Leakage test outputs:
My secret code is NevinAda123
```

---

## Example Results

| Model      | DP Enabled | ε   | Accuracy | Leakage Observed |
| ---------- | ---------- | --- | -------- | ---------------- |
| DistilBERT | No         | ∞   | 91%      | High             |
| DistilBERT | Yes        | 8   | 87%      | Low              |
| GPT-2      | No         | ∞   | (N/A)    | Canary recovered |

---

## Project Status

PrivacyLens is **work-in-progress**. Current scope:

- DistilBERT with Opacus for DP-SGD
- GPT-2 canary memorization test
- Extended metrics and benchmarks (work in progress)
