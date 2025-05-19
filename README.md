# Semantic Overlap in Psychological Scales

## Project Overview

This repository contains the scripts used for my master's thesis where I investigated how semantic overlap within psychological scales influences people's response patterns to said scales. The central question explored is whether the empirical item-item relationships in a scale are properties of the measured construct or are also influenced by the scale's semantic structure.

### Research Context

Psychological scales measure latent constructs through a series of related items. Traditional psychometric theory assumes that correlations between item responses arise from the underlying construct. However, emerging research suggests that semantic similarity between items may artificially inflate these correlations - people may respond consistently to similar-sounding items regardless of the underlying construct.

### Research Design

This study employed a novel approach to examine the relationship between semantic similarity and response patterns:

1. Five psychological scales were selected for analysis:
   - Fear of Negative Evaluation (FNE)
   - Fear of Missing Out (FOMO)
   - Mind-Wandering: Deliberate & Spontaneous (MW: D&S)
   - Conscientiousness (C / NEO-PI-C)
   - Positive Emotions (PE / Barchard-PE)

2. For each scale, three versions were tested:
   - Original version (OG)
   - Strictly paraphrased version (SP; paraphrased by GPT-4)
   - Loosely paraphrased versions (LP; paraphrased by GPT-4)

3. The Universal Sentence Encoder (Google's GUSE model) was used to quantify semantic similarity between items.

4. In a longitudinal study across three time points, I collected participant responses to all three versions of each scale.

5. The relationship between inter-item semantic similarity and response correlations was analyzed using Bayesian methods.

## Key Findings

The study found:

1. A positive association between items' semantic relationships and their empirical correlations.
2. Successfully increased semantic distance between scale items through paraphrasing.
3. No evidence that decreasing semantic overlap weakened item response patterns, contrary to expectations.

These findings have implications for scale development and the understanding of what psychological scales actually measure.

## Main Files

- `main.py`: Core script for processing GPT-4 paraphrased items, computing semantic similarity, and finding optimal item combinations with minimal semantic overlap
- `semantic_similarity.py`: Functions for computing and visualizing semantic similarity matrices between scale items
- `plots.py`: Scripts for generating visualization plots of semantic similarity matrices and distributions
- `data_analysis.r`: R script for Bayesian analysis of the relationship between semantic similarity and empirical item correlations
- `data_cleaning.r`: R script for preprocessing participant response data

## Methodology Details

### Semantic Similarity Computation

The study used Universal Sentence Encoder embeddings to quantify semantic similarity between items. Cosine similarity between item embeddings provided a metric of semantic overlap.

### Paraphrasing Approach

Items were paraphrased at two levels:
1. Strict paraphrasing: Preserving semantic cohesion while changing wording
2. Loose paraphrasing: More substantial rewording which lowered semantic cohesion

### Analytical Approach

The analysis employed Bayesian regression models to examine the relationship between semantic similarity matrices (SSMs) and inter-item correlation matrices (IICMs) across original and paraphrased versions of each scale.

## Usage

To reproduce the study:

1. Generate paraphrased versions of the scales using GPT-4
2. Place outputs in the appropriate directory structure
3. Run `main.py` to compute semantic similarities
4. Run `plots.py` to generate visualization plots
5. Run `data_analysis.r` to perform Bayesian analysis of results

## Author

Konrad Mikalauskas
