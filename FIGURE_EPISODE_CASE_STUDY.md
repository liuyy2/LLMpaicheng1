# Episode Case Study Figure

This repo includes a single-figure case study plot for one episode. The figure has two subplots:

- **Top**: PadHold intervals (Op4→Op6, i.e. R_pad occupation) for baseline vs ours,
  displayed as a **dual swim-lane** (y-axis has exactly 2 rows).
  Missions that changed their PadHold between consecutive rolling steps are outlined in **black**.
- **Bottom**: Rolling drift (left y-axis, two solid lines) and #PadHold changes per rolling step
  (right y-axis, semi-transparent bar chart). Disturbance slots are marked with red dashed verticals.

## Key Design Decisions

| Constraint | Implementation |
|---|---|
| Only Pad resource | Extracts ops using `R_pad` (Op4/Op5/Op6), computes `[min(start), max(end)]` per mission |
| y-axis = 2 swim lanes | `Baseline (fixed_tuned)` and `Ours (full_unlock)` — no per-mission y labels |
| Focus window | Clips to `[center - pre, center + post]` slots; auto-selects center from peak changes |
| Rolling step x-axis | Bottom plot uses rolling index `0..K-1`, NOT slot numbers; K ≈ `sim_total_slots / rolling_interval` |
| Change detection | Compares each mission's PadHold `(start, end)` vs previous roll; tolerance `tol` (default 0) |
| Color by mission | Same color for same mission across both lanes (`tab20` colormap) |
| No legend | Colors identify missions visually; a stats annotation box shows changed/total counts |

## Usage (CLI)

```bash
# Basic — auto focus window
python analyze.py --input results --output figures \
  --episode-case-seed 0 \
  --episode-case-baseline fixed_tuned \
  --episode-case-ours full_unlock \
  --episode-case-level heavy

# With explicit focus window
python analyze.py --input results --output figures \
  --episode-case-seed 0 \
  --episode-case-baseline fixed_tuned \
  --episode-case-ours full_unlock \
  --episode-case-level heavy \
  --episode-case-focus 528 \
  --episode-case-pre 48 \
  --episode-case-post 192
```

## Usage (Python)

```python
from analyze import plot_episode_case_study

plot_episode_case_study(
    results_dir="results/batch_10day",
    seed=0,
    policy_baseline="fixed_tuned",
    policy_ours="full_unlock",
    outpath="figures/episode_case_study_seed0_fixed_tuned_vs_full_unlock.png",
    focus_center_slot=None,   # auto-select from peak changes
    pre=48,                   # 48 slots = 12h before center
    post=192,                 # 192 slots = 48h after center
    tol=0,                    # change tolerance in slots
    slot_minutes=15,
    disturbance_level="heavy",
)
```

## Data Source

The function reads `rolling_log.jsonl` from:
```
{results_dir}/{disturbance_level}_{policy}_seed{seed}/rolling_log.jsonl
```

Each line is a JSON snapshot containing:
- `t`: rolling time (slot)
- `plan.op_assignments`: list of `{op_id, mission_id, resources, start_slot, end_slot}`
- `metrics.plan_drift`: drift metric for that rolling step

No additional data files are needed — everything is already saved by the simulator.

## Notes

- Change highlighting currently uses a simple diff on PadHold start/end values.
  **TODO**: filter out already-completed / already-started / frozen tasks (see code comment).
- Disturbance events are read from `scenario.json` and shown as red dashed lines.
