# Peg-socket findings

One report per finished step. Status lives here. The plan lives in `../README.md`.

An agent that finishes or abandons a step writes the report in the same piece of work. The next agent should be able to read this index and the report and not repeat a failed run.

| Step | Folder | Status | Blocks |
|---|---|---|---|
| 0 scene | `scene/` | done | 1, 2, 5 |
| 1 expert | `expert/` | done | 3 |
| 2 smoke load | `smoke/` | done | 4 |
| 3 demos | `demos/` | done | 6 |
| 4 zero-shot | `zeroshot/` | done | nothing, but the GPU must be free of 3 and 6 |
| 5 localize | `localize/` | blocked | 6. A fail stops 6 |
| 6 fine-tune | `train/` | not started | the end goal |
| face probe | `face_probe/` | done | nothing. A prior test, not step 6 |
| chunk playback | `chunk_playback/` | done | nothing. The 50-step chunk still jitters, so a longer playback will not smooth this policy |

Update the status cell when the report is written: `done`, `failed`, or `blocked`.

Step 5 is blocked, not failed. The 20 val layouts replay, but frozen SmolVLM2 was not asked, so there is no median and no pass. The chat process that held the GPU during that attempt is gone. Do not treat [face_probe](face_probe/REPORT.md) as step 6: that run moved `tablecam` to the face, trained on 10 new demos, and inserted 0/5. [Chunk playback](chunk_playback/REPORT.md) used those same weights and inserted 0/5; the 50-step chunk still jitters inside itself, so a longer playback will not smooth this policy. Step 6 stays unstarted until localize is actually scored. The cross-task read of that evidence is [findings/feasibility](../../findings/feasibility/REPORT.md).

## What a report contains

Write it so a later agent can skip the run. Match the throw reports in `findings/`: the result first, then the process, then the numbers.

```markdown
# <step name>

Status: done | failed | blocked

## Result

One paragraph. The number that decides the gate, and whether the next step may start.

## Process

What was run, from which file, with which command. For a training run or an expert
eval, say the data, the seed split, what was frozen, and when it stopped.

## Numbers

A small table. Quote the log. Link a plot when the run produced one
(`loss.png`, `val_cm.png`). A closed-loop run embeds nothing until
`eval_summary.json` exists; link that file instead of paraphrasing it.

## Failures other agents should know

What broke, what was tried, and what not to retry. Name the layout, the
checkpoint, or the command that failed. If nothing failed, say so in one line.

## Sources

Paths to the summary json, the log, and the plots.
```

Do not claim the policy places the peg unless `train/eval_summary.json` is the source of the number.
