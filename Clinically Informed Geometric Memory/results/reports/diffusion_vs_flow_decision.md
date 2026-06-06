# Diffusion vs Flow Decision

## Implemented in this run

- Main posterior evidence uses existing conditional diffusion/posterior benchmark outputs and deterministic/probabilistic baselines.
- A true directed endpoint path `u=G(g)` to `x=T(g,b)` was not trained in this run.
- No stochastic interpolant loss `x_t=(1-t)u+t*x+gamma(t)*noise` was trained.
- No entropy-regularized bridge objective was implemented.

## Terminology recommendation

Use `clinically informed conditional diffusion posterior sampler` for implemented diffusion results. Do not use `flow`, `bridge`, `stochastic interpolant`, or `Schrodinger bridge` in the title or main claim unless the optional endpoint-path branch is trained and evaluated.

Flow/stochastic interpolant can be added later as a Methods ablation only if it improves mechanics error, boundary preservation, calibration, OOD detection, or inference cost without weakening the existing posterior evidence.
