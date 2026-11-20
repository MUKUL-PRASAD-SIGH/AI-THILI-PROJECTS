# Frequent play patterns

Mined with `thilimarketbasket.mine.frequent_itemsets` over 14,400 possession baskets (actions + ordered transitions + contiguous runs of up to 4 actions + team).

## Most frequent play sequences

| # | sequence | support |
|---|---|---|
| 1 | inbound -> bring_up | 52.7% |
| 2 | bring_up -> pass | 22.8% |
| 3 | def_rebound -> bring_up | 17.5% |
| 4 | bring_up -> pick_and_roll | 17.2% |
| 5 | inbound -> bring_up -> pass | 16.7% |
| 6 | inbound -> bring_up -> pick_and_roll | 12.6% |
| 7 | def_rebound -> fast_break | 11.8% |
| 8 | pass -> pass | 11.3% |
| 9 | pick_and_roll -> drive | 10.4% |
| 10 | bring_up -> post_up | 10.2% |

## Most frequent sequences of 3+ actions

| # | sequence | support |
|---|---|---|
| 1 | inbound -> bring_up -> pass | 16.7% |
| 2 | inbound -> bring_up -> pick_and_roll | 12.6% |
| 3 | inbound -> bring_up -> post_up | 7.4% |
| 4 | bring_up -> pick_and_roll -> drive | 6.9% |
| 5 | inbound -> bring_up -> dribble | 6.4% |
| 6 | foul_drawn -> free_throw_made -> free_throw_made | 6.1% |
| 7 | bring_up -> pass -> pass | 5.4% |
| 8 | def_rebound -> bring_up -> pass | 5.4% |
| 9 | inbound -> bring_up -> pick_and_roll -> drive | 5.2% |
| 10 | bring_up -> pick_and_roll -> pass | 4.8% |

## Team signatures (rule team -> play, by lift)

Lift = how many times more often this team's possessions contain the play than the league's.

**Comets**

| play | lift | confidence |
|---|---|---|
| pick_and_roll -> drive -> layup_missed | 2.22 | 4.4% |
| pick_and_roll -> drive -> kick_out | 2.12 | 4.8% |
| pass -> pick_and_roll -> drive | 2.12 | 5.7% |
| bring_up -> pick_and_roll -> drive -> layup_made | 2.11 | 4.4% |
| pick_and_roll -> drive | 2.10 | 21.8% |

**Sharks**

| play | lift | confidence |
|---|---|---|
| pass -> three_made | 1.76 | 9.8% |
| bring_up -> pass -> pass | 1.76 | 9.5% |
| pass -> three_missed | 1.71 | 12.4% |
| pass -> pass | 1.70 | 19.2% |
| inbound -> bring_up -> pass -> pass | 1.65 | 6.5% |

**Titans**

| play | lift | confidence |
|---|---|---|
| post_up -> foul_drawn -> free_throw_made | 2.13 | 6.5% |
| pass -> post_up -> foul_drawn | 2.11 | 3.9% |
| post_up -> foul_drawn -> free_throw_made -> free_throw_made | 2.09 | 4.9% |
| post_up -> foul_drawn | 2.08 | 8.7% |
| bring_up -> post_up -> foul_drawn -> free_throw_made | 2.08 | 3.4% |

**Wolves**

| play | lift | confidence |
|---|---|---|
| def_rebound -> fast_break -> dunk | 1.83 | 4.8% |
| fast_break -> dunk | 1.70 | 7.8% |
| pass -> turnover | 1.58 | 5.0% |
| dunk | 1.55 | 12.0% |
| def_rebound -> fast_break -> layup_made | 1.45 | 5.5% |

## Play -> outcome rules (by lift)

"Possessions that contain this play end with this outcome" (outcome not part of the play).

| play | outcome | support | confidence | lift |
|---|---|---|---|---|
| free_throw_made -> free_throw_made | foul_drawn | 6.1% | 100.0% | 9.07 |
| free_throw_made -> free_throw_missed | foul_drawn | 2.2% | 100.0% | 9.07 |
| free_throw_missed -> free_throw_made | foul_drawn | 2.0% | 100.0% | 9.07 |
| off_rebound -> pass | jumper_missed | 1.1% | 48.3% | 4.64 |
| off_rebound -> kick_out | three_missed | 1.5% | 70.2% | 4.41 |
| post_up -> kick_out | three_missed | 2.0% | 49.7% | 3.13 |
| off_rebound -> layup_made | three_missed | 1.8% | 48.9% | 3.08 |
| bring_up -> post_up -> kick_out | three_missed | 1.1% | 48.9% | 3.08 |
| off_rebound -> pass | three_missed | 1.1% | 48.6% | 3.05 |
| drive -> kick_out | three_missed | 2.9% | 46.9% | 2.95 |
