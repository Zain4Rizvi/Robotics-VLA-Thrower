"""Phase-1 gate and the clock's step budgets. No training."""

from __future__ import annotations

CAP_S = 8 * 3600 + 30 * 60
RUN2_NEED_S = 2 * 3600 + 30 * 60
# This sitting starts colors only when 3 hours remain. 4000 steps at 5 hours,
# otherwise 2000. The plan's rate cut can still drop that to a shorter multiple of 1000.
COLOR_START_S = 3 * 3600
COLOR_LONG_S = 5 * 3600
MARGIN_S = 10 * 60


def _stacks(summary: dict) -> tuple[dict, dict, list[int]]:
    red = summary["sentences"]["red_on_green"]["results"]
    green = summary["sentences"]["green_on_red"]["results"]
    steps = sorted({int(s) for s in red} & {int(s) for s in green})
    if not steps:
        raise RuntimeError("eval summary has no shared steps")
    return red, green, steps


def earliest_both(summary: dict, need: int = 5) -> int | None:
    red, green, steps = _stacks(summary)
    for step in steps:
        if red[str(step)]["stacks"] >= need and green[str(step)]["stacks"] >= need:
            return step
    return None


def weight_init_step(summary: dict) -> int:
    """Highest green-on-red among checkpoints with red-on-green at least 3. Else the last step."""
    red, green, steps = _stacks(summary)
    best: tuple[int, int] | None = None
    for step in steps:
        if red[str(step)]["stacks"] < 3:
            continue
        g = int(green[str(step)]["stacks"])
        if best is None or g > best[0] or (g == best[0] and step < best[1]):
            best = (g, step)
    if best is None:
        return steps[-1]
    return best[1]


def decide(summary: dict, run2_done: bool) -> dict:
    red, green, steps = _stacks(summary)
    last = steps[-1]
    prev = steps[-2] if len(steps) >= 2 else None
    r = int(red[str(last)]["stacks"])
    g = int(green[str(last)]["stacks"])
    g_prev = int(green[str(prev)]["stacks"]) if prev is not None else None
    bar = r >= 5 and g >= 5
    out = {
        "R": r,
        "G": g,
        "G_prev": g_prev,
        "last_step": last,
        "prev_step": prev,
        "bar": bar,
        "run2_done": run2_done,
    }
    if bar and not run2_done:
        out["branch"] = "colors_from_run1"
        out["init_step"] = earliest_both(summary)
        return out
    if bar and run2_done:
        out["branch"] = "colors_from_run2"
        out["init_step"] = earliest_both(summary)
        return out
    if prev is None and not bar:
        out["branch"] = "stop_no_previous"
        return out
    if not run2_done and g < 5 and g_prev is not None and g > g_prev:
        out["branch"] = "run2_same"
        out["steps"] = 6000
        out["green_weight"] = 1
        out["init_step"] = last
        return out
    if not run2_done and g < 5 and g_prev is not None and g <= g_prev:
        out["branch"] = "run2_weight"
        out["steps"] = 4000
        out["green_weight"] = 2
        out["init_step"] = weight_init_step(summary)
        return out
    if run2_done and not bar:
        out["branch"] = "stop_missed"
        return out
    out["branch"] = "stop_no_branch"
    return out


def color_request(remaining_s: float) -> int | None:
    if remaining_s < COLOR_START_S:
        return None
    if remaining_s >= COLOR_LONG_S:
        return 4000
    return 2000


def fits(steps: int, save_freq: int, remaining_s: float, min_per_1000: float, episodes: int, sec_per_ep: float, extra_s: float) -> bool:
    n_ckpt = steps // save_freq + 1
    train_s = (steps / 1000.0) * min_per_1000 * 60.0
    eval_s = n_ckpt * episodes * sec_per_ep
    return train_s + eval_s + extra_s + MARGIN_S <= remaining_s


def fit_color_steps(requested: int, remaining_s: float, min_per_1000: float, sec_per_ep: float, extra_s: float) -> int:
    for steps in range(requested, 0, -1000):
        if fits(steps, 1000, remaining_s, min_per_1000, 30, sec_per_ep, extra_s):
            return steps
    return 0


def branch_text(decision: dict, started: bool, why: str) -> str:
    name = decision["branch"]
    lines = [
        f"R={decision['R']} at step {decision['last_step']}. G={decision['G']}. G_prev={decision['G_prev']}.",
        f"The phase-1 bar (both at least 5) is {'met' if decision['bar'] else 'not met'}.",
        f"Branch `{name}`.",
    ]
    if started:
        lines.append(why)
    else:
        lines.append(why if why else "That branch was not started.")
    return " ".join(lines)


def _fake(rows: list[tuple[int, int, int]]) -> dict:
    red, green = {}, {}
    for step, r, g in rows:
        red[str(step)] = {"stacks": r}
        green[str(step)] = {"stacks": g}
    return {"sentences": {"red_on_green": {"results": red}, "green_on_red": {"results": green}}}


def self_check() -> None:
    rising = decide(_fake([(0, 5, 2), (2000, 5, 4)]), False)
    if rising["branch"] != "run2_same" or rising["init_step"] != 2000 or rising["steps"] != 6000:
        raise SystemExit(f"rising {rising}")
    flat = decide(_fake([(0, 5, 4), (2000, 4, 3), (4000, 5, 3)]), False)
    if flat["branch"] != "run2_weight" or flat["green_weight"] != 2 or flat["init_step"] != 0:
        raise SystemExit(f"flat {flat}")
    met = decide(_fake([(0, 5, 2), (2000, 5, 5), (4000, 6, 7)]), False)
    if met["branch"] != "colors_from_run1" or met["init_step"] != 2000:
        raise SystemExit(f"met {met}")
    missed = decide(_fake([(0, 5, 2), (2000, 5, 4)]), True)
    if missed["branch"] != "stop_missed":
        raise SystemExit(f"missed {missed}")
    if color_request(2.9 * 3600) is not None or color_request(3 * 3600) != 2000:
        raise SystemExit("color request 3h")
    if color_request(5 * 3600) != 4000:
        raise SystemExit("color request 5h")
    # 1000 steps at 30 min/1000, 45 s/episode, no extra: train 1800s + eval 2*30*45 + margin.
    got = fit_color_steps(4000, 3 * 3600, 30, 45, 0)
    if got not in (1000, 2000):
        raise SystemExit(f"fit {got}")
    if fit_color_steps(2000, 20 * 60, 45, 60, 0) != 0:
        raise SystemExit("fit should refuse")
    print("gate self_check ok", flush=True)


if __name__ == "__main__":
    self_check()
