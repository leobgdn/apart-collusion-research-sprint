# Judge validation

Labelled rows: 30 (LLM pre-labels, each reviewed by the author).

- **plan_insight**: n=20, human positives=6, calibrated threshold=0.60, balanced accuracy=0.71, Cohen's kappa=0.31, AUROC=0.76
- **message**: n=10, human positives=10, calibrated threshold=0.60, balanced accuracy=nan, Cohen's kappa=nan, AUROC=nan

Note: the threshold is chosen on these same labels (in-sample), so agreement is optimistic; reported as such.