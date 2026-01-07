# Lip-DARTS: Lipschitz-Regularized Path-Strength Refactoring for NAS

This repository provides the official PyTorch implementation for the paper:  
**"Lipschitz-Regularized Path-Strength Refactoring for Differentiable Architecture Search"** (Anonymous Submission).

##  Overview
Lip-DARTS introduces a decoupled search strategy to resolve the instability and "performance collapse" in DARTS. The framework consists of three sequential stages: Supernet Pre-training with Lipschitz regularization, Fast Architecture Search via Simulated Annealing (SA), and Architecture Evaluation.

##  Requirements
The following dependencies are required to run the code. You can install them via:
```bash
pip install -r requirements.txt
```

Key packages from our environment:

torch==2.1.0

torchvision==0.16.0

numpy==1.23.5

## Experimental Pipeline

1. Supernet Weight Pre-training
First, train the Lipschitz-regularized supernet on CIFAR-10 to obtain stable weights.

```bash
python train_search.py  --cutout --fixed_alphas
```

2. Decoupled Architecture Search (Fast Search)
After pre-training, use the Path-Gradient Integral and SA to find optimal genotypes. This step is extremely efficient (0.04 GPU days). We recommend running this 5 times to obtain 5 candidate genotypes (geno1~5).
```bash
python fast_search.py --weights [PRETRAINED_WEIGHTS].pt  
```

3. Architecture Evaluation (Retraining)
We first evaluate the 5 candidates on CIFAR-10 to identify the best-performing genotype.
```bash
python train.py --arch [candidate genotypes] --cutout --auxiliary 
```
Then, we retrain this optimal genotype across multiple datasets to report the accuracy range (mean ± std).

A. CIFAR-10 Retraining (5 independent runs)
```bash
python train.py --arch [BEST_GENOTYPE] --cutout --auxiliary 
```

B. CIFAR-100 Transferability (5 independent runs)
```bash
python train_cifar100.py --arch [BEST_GENOTYPE] --cutout --auxiliary 
```

C. ImageNet-10 Transferability 
```bash
python train_imagenet.py --arch [BEST_GENOTYPE] 
```

## Ablation Study
To verify the impact of bi-level coupling on performance collapse, run the supernet training without fixed alphas (reintroducing the architecture parameters $\alpha$):

```bash
python train_search.py  --cutout 
```

## Core Results
Efficiency: Marginal search cost is only 0.04 GPU days.

Stability: Zero "performance collapse" (no skip-connection explosion) observed across all runs.

