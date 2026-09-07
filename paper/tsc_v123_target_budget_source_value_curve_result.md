# V123 Target-Budget Source-Value Curve Result

## Status

V123 completed under immutable snapshot `8281502f843b7886a4fe` in 1,438.77 s.
The local and remote `result.json` SHA-256 is
`cc59ab48768901a0e73d91cde878b0a764b6a8e16f8a929df4a2406efd80d944`.
The 209,690,586-byte prediction artifact remains on the server and was not
copied to the local workspace.

V123 is development evidence on the fixed 22-seed Jinan selector. It is not a
fresh-city confirmation.

## Results

Positive values below are worse than exact PhasePressure. The source
contribution is `source-augmented - architecture-matched target-only`, so a
negative value means that source-city information helped.

| Target groups | Target-only vs PhasePressure | Source-augmented vs PhasePressure | Source contribution | Paired 95% CI | Nested source choice |
|---:|---:|---:|---:|---:|---|
| 0 | n/a | +0.055% | n/a | n/a | Atlanta, weight 1.0 |
| 25 | +7.576% | +4.477% | -3.172% | [-3.607%, -2.768%] | Ingolstadt, weight 1.0 |
| 50 | +3.620% | +2.483% | -1.148% | [-1.574%, -0.721%] | Hangzhou, weight 0.75/1.0 |
| 100 | +3.817% | +2.534% | -1.309% | [-1.653%, -0.964%] | uniform sources, weight 1.0 |
| 250 | +1.589% | +1.563% | -0.043% | [-0.296%, +0.220%] | Atlanta, weight 0.25--0.75 |
| 500 | +1.944% | +1.122% | -0.853% | [-1.160%, -0.550%] | Atlanta, weight 0.75 |
| 1000 | +2.275% | +0.775% | -1.491% | [-2.041%, -1.004%] | Atlanta, weight 1.0 |

The paired source contribution is significant at B25, B50, B100, B500 and
B1000. At B25 all 22 held-out selector seeds improved relative to the matched
target-only model. This confirms that the positive V98 effect was not merely a
memory or logging error: source information can improve an
architecture-matched target model.

## Decision

The preregistered deployment-authorizing gate nevertheless fails because no
source-augmented policy beats PhasePressure on the selector estimand. The
result field `source_signal_present=false` combines two requirements and must
not be paraphrased as "source information has no value." The accurate result
is:

1. **Relative source value:** supported at five of six positive target budgets.
2. **Absolute deployability:** rejected; the residual action selector remains
   worse than PhasePressure.

The next development stage therefore keeps the causal source models and
replaces raw argmin intervention with a cross-fitted conservative gate. A fresh
city can be opened only after that guarded policy passes an absolute
PhasePressure gate.
