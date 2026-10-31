# Frozen V93: same-cohort MaxPressure supplement

The existing MaxPressure rule is evaluated on all 56 frozen V93 seeds and four
original scenarios (224 one-hour runs). Models, networks, executor and signal
timing are unchanged; no retraining, label collection or tuning occurs.

`protocol.json` fixes the cohort and analysis before outcomes. Two scheduler
submissions timed out before enqueue and their failure receipts are preserved.
`direct_launch.py` uses the original V93 direct-node mechanism; its receipt
records the six shards on three nodes, eight workers per shard.

Only compact `_shards` JSON is collected. Tripinfo XML and full raw results
remain on the server. `report.py` appends MP to the existing complete comparison
without replacing original records or selecting favorable seeds. Its extended
exports keep the original intervals and add the declared MP bootstrap and
expanded 15-comparison Holm correction.
