# PeCyclesPerMac patch: dataflow tests

Run on 2026-11-23 with `python project/dataflow_test.py`. Workload: GEMM test (M=256, N=128, K=256) on a 32x32 array, default `configs/scale.cfg` otherwise.

**Overall: all tests passed.**

| Dataflow | PeCyclesPerMac (k) | Total cycles | Compute cycles | Expected compute cycles | Result |
| --- | --- | --- | --- | --- | --- |
| ws | 1 | 15,498 | 11,199 | 11,199 (unmodified SCALE-Sim) | PASS |
| ws | 4 | 49,098 | 44,799 | 44,799 (= 4 x 11,199 + 3) | PASS |
| ws | 8 | 93,898 | 89,599 | 89,599 (= 8 x 11,199 + 7) | PASS |
| os | 1 | 14,473 | 10,175 | 10,175 (unmodified SCALE-Sim) | PASS |
| os | 4 | 45,001 | 40,703 | 40,703 (= 4 x 10,175 + 3) | PASS |
| os | 8 | 85,705 | 81,407 | 81,407 (= 8 x 10,175 + 7) | PASS |
| is | 1 | 18,506 | 14,207 | 14,207 (unmodified SCALE-Sim) | PASS |
| is | 4 | 61,130 | 56,831 | 56,831 (= 4 x 14,207 + 3) | PASS |
| is | 8 | 117,962 | 113,663 | 113,663 (= 8 x 14,207 + 7) | PASS |

What the two checks show:

1. **k = 1 matches unmodified SCALE-Sim exactly** on every dataflow, so the patch changes nothing when it is off. Reference numbers were measured on an unpatched scale-sim-v3 checkout.
2. **k > 1 scales compute cycles by exactly k**, plus k - 1 cycles because the last request of the stretched demand matrix is held for k - 1 extra cycles. This holds for weight-, output- and input-stationary dataflows.

Limitation: the model stretches every phase of the computation, including weight loading and output drain, by k. A real bit-serial PE might load weights at full speed, so these cycle counts are an upper bound.
