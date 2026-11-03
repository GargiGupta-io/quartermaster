# Sample scenarios

Each scenario exercises a different rule in the policies. These become the starting
point for the evaluation set. Totals include 18% GST.

| Request | Expected result | Rule tested |
|---|---|---|
| 20 standard laptops, Pune, Engineering | Use 4 from stock, buy 16 from V-101 for ₹13,59,360. Needs Department Head and Finance Controller, three quotes | POL-001 1.2, POL-002 2.1, POL-003 3.2, 3.3 |
| Same, tempted by V-106 or V-104 | Reject both: V-106 is suspended, V-104 exceeds its ₹5,00,000 probation cap | POL-003 3.1 |
| Same, tempted by V-105 | Reject: foreign vendors are not allowed because of price | POL-005 5.1 |
| 10 standard laptops, Mumbai | Transfer from Bengaluru (12 available) instead of buying | POL-001 1.2 |
| 20 monitors, Mumbai, Operations | V-103 at ₹4,43,680 is over the ₹2,40,000 remaining budget, needs CFO exception | POL-001 1.3 |
| 2 network switches, Pune | Only V-105 sells them: USD, 3% buffer, sole source, Finance Controller | POL-003 3.2, POL-005 |
| 1 Pro laptop, Sales | Hardware exception from the CIO | POL-004 4.2 |
| ₹40,000 of gift cards | Refuse | POL-001 1.5 |
