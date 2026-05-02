---
name: spec-repair
description: "Bounded repair of TLA+ specs that fail P1 (SANY) or P2 (TLC from Init). Use when a model-generated spec needs to pass P1/P2 so downstream P3 (TV) and P4 (invariant) can score it. Enforces a strict allow-list of mechanical fixes and MUST NOT alter the model's semantic intent (actions, guards, variable set). Every edit must carry a written justification."
---

Read `guide.md` for the full repair procedure.
