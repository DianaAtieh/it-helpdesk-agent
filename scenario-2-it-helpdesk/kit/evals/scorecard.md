# Eval scorecard

Cases file: `cases.sample.jsonl` · runs: **10** · passed: **10** (**100%**)

Turn latency: avg **2.15 s**, p95 **3.58 s** · cost: not reported by the app

| case | category | result | why it failed |
|---|---|---|---|
| S2-01 | grounding | ✅ |  |
| S2-02 | clarification | ✅ |  |
| S2-03 | policy_conflict | ✅ |  |
| S2-04 | unknown | ✅ |  |
| S2-05 | happy_path | ✅ |  |
| S2-06 | rejection | ✅ |  |
| S2-07 | tool_failure | ✅ |  |
| S2-08 | guardrail | ✅ |  |
| S2-09 | conflict | ✅ |  |
| S2-10 | policy_conflict | ✅ |  |

| category | passed |
|---|---|
| clarification | 1/1 |
| conflict | 1/1 |
| grounding | 1/1 |
| guardrail | 1/1 |
| happy_path | 1/1 |
| policy_conflict | 2/2 |
| rejection | 1/1 |
| tool_failure | 1/1 |
| unknown | 1/1 |
