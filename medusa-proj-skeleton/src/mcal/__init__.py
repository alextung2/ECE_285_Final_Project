"""mcal: Conditional Acceptance Calibration for (Dynamic) Medusa Decoding.

Core modules (numpy only, run anywhere):
    traces      - verification-trace format + synthetic generator
    features    - node features available *before* verification
    estimators  - path scorers: Medusa accuracy product, MEDUSA+ prob product,
                  count-table conditional, logistic conditional, direct classifier
    tree        - budgeted best-first tree selection (ancestor-closed)
    evaluate    - trace-simulated accepted tokens, Brier score, calibration

GPU modules (need torch + Medusa, imported lazily):
    model       - load frozen Vicuna-7B v1.3 + Medusa-1 heads
    collect     - turn greedy generations into verification traces
    decode      - online decoding with a dynamic tree (timing / tokens per sec)
"""

__version__ = "0.1.0"
