# Evaluation report

Written 13 November 2026. Real AI model (`openai/gpt-oss-120b` on Groq, with `openai/gpt-oss-20b` as backup). Cases are in `evals/cases.json`; purchase answers are worked out from the policy rules in code. Rerun with `scripts/run_eval.py`.

| Measure | First prompts | Before evaluation fixes | Current |
|---|---|---|---|
| Cases fully correct | 19/22 | 32/40 | 37/40 |
| Purchases fully correct | 13/16 | 12/16 | 14/16 |
| Questions fully correct | 6/6 | 16/18 | 18/18 |
| Guardrail cases correct | 0/0 | 4/6 | 5/6 |
| Right status or decision | 19/22 | 32/38 | 35/38 |
| Right vendor | 7/8 | 9/10 | 10/10 |
| Figures match the rules | 6/7 | 8/9 | 9/9 |
| Right approvers | 7/8 | 9/10 | 10/10 |
| Answer contains the fact | 6/6 | 15/16 | 16/16 |
| Right document cited | 6/6 | 14/16 | 16/16 |
| Declined when not covered | - | 2/2 | 2/2 |
| Automatic checks passed | 11/14 | 19/26 | 23/26 |
| Sent to a person (needs review) | 2 | 5 | 3 |
| Median seconds per AI case | 5.4 | 11.2 | 5.1 |
| Slowest case, seconds | 75.4 | 87.3 | 91.1 |
| Average tokens per AI case | 6,762 | 6,676 | 6,339 |
| Total tokens | 148,773 | 247,027 | 234,543 |

## First prompts: cases not fully correct

Only 22 of 40 cases have run so far.

- **P03** (failed): status failed, expected ready_for_approval; stock plan differs from the rules; vendor None, expected V-103; approvers None, expected ['CFO', 'Department Head']; total None, expected 443680.0; automatic checks did not pass
- **P10** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
- **P11** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass

## Before evaluation fixes: cases not fully correct

- **G04** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
- **G06** (failed): status failed, expected ready_for_approval; stock plan differs from the rules; vendor None, expected V-101; approvers None, expected ['Department Head', 'Finance Controller']; total None, expected 1359360.0; automatic checks did not pass
- **P01** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
- **P10** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
- **P11** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
- **P15** (needs_review): status needs_review, expected refused or needs_clarification
- **Q14** (answered): cited nothing, expected CON-104 or REG-VND; automatic checks did not pass
- **Q15** (answered): answer lacks ['31 March 2027', 'March 2027']: The company documents do not answer this. Please ask the procurement desk.; cited nothing, expected CON-104; automatic checks did not pass

## Current: cases not fully correct

- **G06** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
- **P06** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
- **P10** (needs_review): status needs_review, expected ready_for_approval; automatic checks did not pass
