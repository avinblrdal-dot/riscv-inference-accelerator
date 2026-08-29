# Lab notebook entry template

Copy this block to the **top** of `NOTEBOOK.md` (newest first) and fill it in.

Write the entry **as you work**, not afterwards. Reconstructed notes lose
exactly the detail that turns out to matter — especially the failures.

---

```markdown
## YYYY-MM-DD — <short title>

**Who:** <names present>

**Goal:** What I set out to do today, in one sentence.

**What I did:**
- Commands run, files changed, parameters used.
- Enough that someone else could repeat it exactly.

**Results / numbers:**
- Actual numbers, with units, and whether they are MEASURED, MODELLED or
  ESTIMATED.
- git SHA if a measurement was taken.

**What broke / open questions:**
- Everything that failed, including things that turned out to be my own
  mistake. **This section is the most valuable one.**
- What I ruled out, so the next person does not repeat it.

**Next step:**
- The single next thing to do.
```

---

## Why the failures matter most

Science fair judges ask *"what didn't work?"* almost every time. A team that
answers "everything went fine" sounds either lucky or unreflective. A team
that says "we lost two days to a coprocessor bug where the CPU held a signal
high longer than our testbench did, so every accumulation was doubled" sounds
like engineers.

The failures are also where the actual learning is recorded. Six months later
nobody remembers why a parameter is set the way it is — unless it is written
down here.

## Rules

1. **Append-only.** Never edit or delete a past entry. If you were wrong, add
   a new entry saying so and link back.
2. **Dated, every time.**
3. **Numbers carry units and a provenance label.**
4. **Record the git SHA** whenever a measurement is taken.
5. **Write down dead ends.** They are results too.
