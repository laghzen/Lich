# iPC TileLang — Stage 1.10

## Fix
Corrected `scripts/smoke.py` so `build_weight_update(M,K,B,...)` receives the upper state `X_upper[K,B]`, matching the iPC weight update `ΔW = α E @ f(X_upper)^T`.

The previous smoke test accidentally passed the post-inference lower state `x[M,B]`. That passed only for the accidental square case `M=K=64` and failed for `M=10, K=64` with the expected ABI error.

The smoke test now explicitly validates the non-square `M=10, K=64, B=128` weight-update tail path for recomputed activation. The regular saved-activation test also now uses `relu(x_up)` as required by the weight-update equation.
