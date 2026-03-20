# GPU Infrastructure Analysis

This document summarizes the GPU-rental landscape considered in March 2026 for Neural Tablebases.

## Project Context

Current practical workload:

- MLP-heavy training
- datasets in the MB-to-GB range rather than multi-terabyte scale
- active emphasis on KPvKP canonical V5 training and iterative evaluation

This means the project currently benefits more from cheap iteration than from extreme multi-GPU scale.

## Strategic Recommendation

### Best fit for the current phase

- Consumer/high-end single-GPU rentals remain the best cost-performance choice.
- RTX 4090-class hardware is still the most attractive option for typical experiments.
- RunPod and Vast.ai remain the most plausible low-cost platforms for this style of workload.

### When to move up-market

Move toward providers such as OVH, GCP, or larger managed platforms when:

- dataset storage becomes the bottleneck
- training jobs become long enough that operational stability matters more than raw price
- 5-piece and 6-piece experiments start demanding more robust data locality and orchestration

## Provider Snapshot

Prices below are the rough March 2026 references captured during analysis and should be treated as directional, not fixed.

| Provider | Typical GPU | Approx. Price/h | Best Use |
|----------|-------------|-----------------|----------|
| RunPod | RTX 4090 | `$0.35-$0.50` | best general value for current experiments |
| Vast.ai | mixed marketplace | `<$0.30` | minimum cost, lower reliability |
| OVH AI Notebook | L4 / A100 | `0.83 EUR / 3.00 EUR` | stronger storage and EU-friendly hosting |
| Google Cloud Spot | L4 | `$0.15-$0.25` | strong cloud option if spot capacity is available |
| AWS Spot | A10G | `$0.40-$0.60` | good ecosystem, weaker on egress costs |
| Lambda Labs | H100-class options | `~$2.30` | premium high-end experiments |

## Practical Guidance For This Repo

### If the goal is cheap iteration

Use:

- local DirectML on AMD when convenient
- RunPod or Vast.ai when you need stronger NVIDIA throughput

### If the goal is more stable medium-scale work

Use:

- OVH if EU locality and storage simplicity matter
- GCP Spot if you can tolerate preemption and keep data in-region

### If the goal is maximum convenience

Use:

- a managed cloud setup with storage close to compute
- but only when the project actually grows beyond the current single-model cadence

## Main Risk

The main cloud cost trap is not GPU hourly price. It is data movement and storage architecture.

For this project, bad data locality can erase the savings from cheap GPU time.

## Free Or Nearly Free Options

- Google Colab for quick prototypes
- Kaggle Notebooks for limited weekly free GPU time
- SageMaker Studio Lab for lightweight no-card experimentation
- free cloud credits from GCP/Azure/AWS programs for short validation bursts

These are useful for prototyping, but they are not the main recommendation for repeatable project workflows.

## Decision Summary

| If you want... | Choose... |
|----------------|-----------|
| lowest cost | Vast.ai |
| easiest cheap iteration | RunPod |
| better storage story | OVH |
| cloud spot efficiency | GCP L4 Spot |
| premium top-end compute | Lambda Labs |

---

Analysis date: March 16, 2026
Documentation refresh: March 20, 2026
