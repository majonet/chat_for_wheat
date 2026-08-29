# Kaggriculture Wheat-only Dueling Double DQN

This project uses the real Kaggriculture simulator and restricts the learned economy to Wheat seed purchase, Wheat planting/watering/harvesting/selling, Farm Hand hiring, land purchase, and movement.

The current full-game source defines Wheat as seed cost 10, base price 25, first yield at day 2, max yield day 4, and max yield 6. Watering in the day-2..4 bonus window adds one yield unit per day without fertilizer. citeturn255118view0turn440715view0

The simulator also makes the planting day count as the first unwatered day, treats locked quadrants as passable but non-actionable, drops unit inventories into the shed at end of day, and sells only from the shed. citeturn255118view1turn440715view3

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python train.py --episodes 2000 --opponent starter
```

For a second training run:

```bash
python train.py --episodes 2000 --opponent random --checkpoint artifacts/wheat_dqn.pt
```

## Evaluate

```bash
python evaluate.py artifacts/wheat_dqn.pt --episodes 50 --opponent starter
```

## Plot

```bash
python plot.py artifacts/training_log.csv
```

## Print a successful trajectory

```bash
python trajectory.py artifacts/wheat_dqn.pt
```

## Kaggle submission

Package `main.py`, the Python modules, and the trained `artifacts/wheat_dqn.pt` together. The official full-game submission format expects a root `main.py` with an `agent(obs)` function. citeturn255689view1

## Notes

* The network has a CNN spatial branch over the 10x10 farm and a vector branch for time, money, market, crop, unit and economics features.
* Farmer and every active Farm Hand share the same unit policy. Their local features are differentiated by unit slot encoding.
* The market is a separate Dueling Q head. Quantity is parameterized by bins up to 100, avoiding a 101-way flat quantity action space.
* Action masks are applied before both behavior selection and Double-DQN target selection.
* Reward is delta bank / 100, plus a terminal final-bank term. Seed, hiring and land costs therefore reduce reward naturally through bank changes.
* No game mechanic is reimplemented in the wrapper; the Kaggriculture simulator is the source of truth.
