# ClearPick real-data results (ARMBench Image defect)

- Train 7200, held-out test 1800
- 3-class accuracy: 62.2%
- Failure vs clean accuracy: 70.2%
- Recall clean: 56.7%
- Recall multi_pick: 58.5%
- Recall package_defect: 71.3%

| Cluster | Count | Share | Simulated | Detected (held-out) |
|---|---|---|---|---|
| Multi-pick (two or more items lifted) | 3000 | 50% | yes | 68% of 600 |
| Open book jacket | 805 | 13% | no | 90% of 143 |
| Open box | 773 | 13% | no | 84% of 165 |
| Book jacket | 723 | 12% | no | 93% of 149 |
| Open book | 373 | 6% | no | 80% of 83 |
| Partial box | 172 | 3% | no | 66% of 32 |
| Torn or empty bag | 84 | 1% | no | 73% of 15 |
| Crushed box and other damage | 70 | 1% | no | 62% of 13 |
