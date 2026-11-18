# Reproducing un-xPass on StatsBomb 360 tournament data

Cristian Aquino — DS 340W, Penn State, Fall 2026

Reproduction and extension of **un-xPass** (Robberechts, Van Roy & Davis,
KDD 2023, DOI 10.1145/3580305.3599924), which measures passing creativity with
a Creative Decision Rating (CDR).

The parent paper trains on Euro 2020 plus the top-10 Premier League teams of
2020/21 and evaluates on the top-10 teams of 2021/22. Premier League 360 data
requires a paid StatsBomb subscription, so this project works entirely from
StatsBomb's open data: **Euro 2020** for training and validation, **World Cup
2022** held back as an unseen test competition.

## Running it

Stages are separate subcommands because feature computation and SoccerMap
training are slow and you should not have to redo them to change something
downstream.

```bash
python run.py download --competition euro2020          # add --limit 4 for a smoke test
python run.py split    --competition euro2020
python run.py features --competition euro2020 --split train
python run.py features --competition euro2020 --split validation
python run.py train    --component pass_success  --dataset euro2020_train
python run.py train    --component pass_value    --dataset euro2020_train
python run.py train    --component pass_selection --dataset euro2020_train --max-epochs 20
python run.py rate     --dataset euro2020_validation
python run.py report   --dataset euro2020_validation
python run.py benchmark --dataset euro2020_validation
python run.py gamestate  --dataset euro2020_validation
python run.py ablation   --train-dataset euro2020_train --test-dataset euro2020_validation
```

`benchmark` is the direct comparison with the parent paper: it reports
pass-selection accuracy next to the figures in its Table 1 (closest-teammate
baseline 0.395, XGBoost 0.538, SoccerMap 0.513). SoccerMap's accuracy needs no
extra feature build -- `test()` derives the receiver from the freeze frame.

See `SETUP.md` for installation and for the seven things that had to be fixed
to run a 2023 codebase today.

## Layout

| File | Role |
|------|------|
| `unxpass_project/compat.py` | Patches for the modern stack. Import before `unxpass`. |
| `unxpass_project/config.py` | Paths, competition ids, split parameters |
| `unxpass_project/data.py` | Stage 1 — download and SPADL conversion |
| `unxpass_project/splits.py` | Stage 2 — **match-level** train/validation split |
| `unxpass_project/featurize.py` | Stage 3 — features and labels |
| `unxpass_project/models.py` | Stage 4 — the three CDR components |
| `unxpass_project/rating.py` | Stage 5 — CDR and player aggregation |
| `unxpass_project/benchmark.py` | Stage 6 — accuracy vs the paper's Table 1 |
| `unxpass_project/gamestate.py` | Score and red-card state per pass |
| `unxpass_project/analysis.py` | Stage 7 — **extension A**: does game state affect CDR? |
| `unxpass_project/ablation.py` | Stage 8 — **extension B**: does off-ball context improve the value model? |
| `unxpass_project/minutes.py` | Minutes played, from StatsBomb lineups |
| `run.py` | CLI over the stages |

Each component is built by its own factory in `models.py`, so one can be
replaced without touching the others. That is deliberate: the planned extension
swaps the VAEP value component for a spatially aware one.

## The three datasets

| set | competition(s) | role |
|-----|----------------|------|
| **Train** | Euro 2020 + Euro 2024 | fits the three model components |
| **Test** | held-out matches from the same competitions | used while developing: pipeline checks, comparison with the paper |
| **Unseen** | World Cup 2022 | never touched until training and test work is finished |

`python run.py datasets` prints the current counts and shares.

Each competition is split separately at the same ratio and the parts are then
pooled, rather than pooling first and splitting the whole. That guarantees
both competitions appear in train and test, and it means adding a competition
does not force recomputation of features already cached for the others --
`python run.py pool` concatenates the cached parquet files.

The unseen set is a different *competition*, not a random slice of the
training competitions. That is a stricter test of generalisation, and it is
the project's research question: does a creativity metric built on one
tournament transfer to another? It is why the split is nowhere near the
conventional 70/20/10.

## Departure from the parent paper: splitting by match

un-xPass builds its validation set as "a random sample of 20% of these passes"
(Section 3.1). Passes from the same match therefore appear on both sides of the
split, and passes within a match share teams, tactics, pitch and conditions —
so that split leaks information between training and validation.

`splits.py` partitions **matches** instead. Every pass from a given match lands
on exactly one side. The partition is seeded and written to
`results/split_euro2020.json` so it can be reported and re-used.

## Smoke-test result

A 4-match run (3 train / 1 validation, 2,028 training passes) produced:

```
passes rated : 640
mean CDR     : +0.00154
std CDR      : 0.00718
share CDR > 0: 51.7%
```

Top of the player ranking for the validation match (Switzerland–Turkey):

| player | minutes | passes | CDR/pass | CDR/90 |
|--------|--------:|-------:|---------:|-------:|
| Remo Freuler | 91.2 | 40 | 0.00419 | 0.1654 |
| Xherdan Shaqiri | 75.0 | 23 | 0.00596 | 0.1646 |
| Ricardo Rodríguez | 91.2 | 33 | 0.00432 | 0.1408 |
| Haris Seferović | 74.6 | 15 | **0.00749** | 0.1354 |

The two normalisations disagree, and the disagreement is the point. Shaqiri
leads on CDR per pass; Freuler leads per 90 because he played more and passed
more. Seferović has the highest creativity per pass of anyone on the pitch but
only 15 passes, so he falls to fourth per 90. Per-90 answers "who contributed
most creativity"; per-pass answers "who makes the better decision when on the
ball". Report both.

Shaqiri and Seferović both scored in that match, which is the same kind of
face-validity check as the parent paper's De Bruyne result.

### A note on normalisation and noise

Normalising by minutes does not reduce sampling noise — it rescales. A player
with 200 passes has an uncertain CDR estimate however it is expressed, and
per-90 is in fact *noisiest* at the low-minutes end, since one good ball from a
20-minute substitute is divided by a small number. The minimum-minutes
threshold does the work; `report` defaults to 180 minutes.

**These numbers are from a 4-match smoke test, not a result.** They demonstrate
that the pipeline runs end to end and that CDR behaves sensibly: centred near
zero, roughly symmetric, with a long positive tail. A full run is needed before
anything is claimed.

## Timings (2 CPU threads)

| Stage | Cost |
|-------|------|
| Download | ~6 s per match |
| Features | ~125 s per match |
| `pass_success` (XGBoost) | < 1 s |
| `pass_value` (VAEP, 2 × XGBoost) | ~35 s |
| `pass_selection` (SoccerMap CNN) | ~150 s per epoch per 2,000 passes |
| CDR | ~140 s per 640 passes |

Features for all 51 Euro 2020 matches are therefore about 1h45m, and World Cup
2022's 64 matches about 2h15m. SoccerMap is the long pole: it scales with pass
count, and a full Euro 2020 run is measured in hours on CPU. Both stages are
restartable, so run them in the background.


## The extension: game state and CDR

un-xPass reports that average creativity does not vary with time remaining or
goal difference over a Premier League season. This project re-tests that on
tournament data and adds **red card differential**, which the parent paper
never examined and which Skripnikov, Cemek & Gillman (2026) identify as having
a much larger effect on offensive output than the scoreline -- a team two
players short has its expected goals scaled by 3.20, where a lead of any size
moves output only 6-8%.

`gamestate.py` derives, for every pass:

* `score_diff` -- goals for minus against, before the pass
* `red_card_diff` -- own players sent off minus opponents', positive when the
  passing team is short-handed, matching Skripnikov's sign convention
* `minute` -- cumulative match minute
* `stage` / `knockout` -- the competition stage, and whether it is
  single-elimination

**Match stage is the variable neither parent paper could have studied.** In a
league every match is worth the same three points. In a knockout match a draw
at full time means extra time and then penalties; in a group match it may be
enough to go through. Whether that changes how adventurously players pass is an
open question that only tournament data can answer.

The additive adjustment is applied over score and red card state only. Stage is
reported but not adjusted for: conditioning on all three at once splits the
sample too finely to estimate reliably at this size.

Score differential comes free: `goalscore_diff` is already in the feature set.
Red cards are not in SPADL at all, so they are read from the StatsBomb lineup
files and placed on the match clock.

### Adjustment is additive, not multiplicative

Skripnikov et al. scale xG, which is strictly positive. CDR is a difference of
expected values and is negative for roughly half of all passes, so a
multiplicative adjustment is unstable near zero and changes sign. Each game
state's distribution is instead shifted onto the neutral baseline:

```
CDR_adjusted = CDR - (mean CDR in this state - mean CDR at neutral)
```

### Known limitation

Euro 2020 contains only **six sendings-off across 51 matches**, so the red
card states are estimated from very few passes. Skripnikov et al. had five
leagues over six seasons. Whatever this finds about red cards is suggestive,
not conclusive, and that is itself a result about what tournament data can and
cannot support.


## Extension B: does off-ball context improve the value model?

Hassani, Ramdani & Lotfi (2025) argue that action-value models built on ball
location alone carry a "deficit of realism" -- the same pitch cell is worth the
same whether the goal is open or three defenders block the angle.

un-xPass applies SoccerMap, which does see the freeze frame, only to pass
*selection*. Its value component is VAEP, and just three of its seventeen
features come from the 360 snapshot:

    packing_rate, defenders_in_3m_radius, defenders_in_5m_radius

`ablation.py` trains the value model with and without those three and compares
held-out ROC-AUC, Brier score and log-loss on whether the team scores or
concedes within ten actions. If DxT's argument holds here, dropping them should
hurt.

Both halves of VAEP are tested separately, because a held-out set in which
nobody concedes within ten actions leaves the defensive labels single-class,
making ROC-AUC and log-loss undefined. That is a property of the sample and is
reported rather than raised.

The three features describe defender proximity, not the full freeze frame, so
whatever this measures is a lower bound on what full spatial context could
contribute.
