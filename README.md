# ARTEMIS on NEWS Dataset

This repository contains the adaptation and evaluation of the ARTEMIS architecture on the semi-synthetic NEWS dataset.

## Project Objective

The goal of this project is to:

- Replace the original IHDP loader with the NEWS dataset
- Maintain the original ARTEMIS training pipeline
- Extend the outcome head to a multi-treatment compatible structure
- Compare ARTEMIS with TARNet and MITNet under the same experimental setup

## Dataset

- N = 5000 samples  
- d = 3477 high-dimensional topic features  
- Binary treatment (T ∈ {0,1})  
- Semi-synthetic dataset with available potential outcomes (μ₀, μ₁)

## Experimental Setup

- 80/20 stratified split  
- random_state = 42  
- 5 different seeds  
- Metrics:
  - PEHE
  - ATE error
  - Paired t-test
  - Wilcoxon signed-rank test

## Results Summary

ARTEMIS shows a statistically significant improvement in PEHE compared to TARNet and MITNet, while differences in ATE are not statistically significant.

This confirms that contrastive learning and Mutual Information regularization improve the estimation of heterogeneous treatment effects.

## Repository Structure

- code/ → Model implementations and training scripts  
- results/ → CSV and JSON result files  
- figures/ → Plots used in the report  
- report/ → Final PDF report
