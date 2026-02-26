# Methodology (For Adviser Pitch)

---

## What We Will Build

We build two versions of the same system on the same footage:

| | Baseline | Proposed |
|---|---|---|
| Detection | YOLOv8 | YOLOv8 (same) |
| Tracking | DeepSORT | DeepSORT (same) |
| Zone check | Same polygon | Same polygon |
| Threshold | 60 seconds | 60 seconds (same) |
| Context layer | None | Added |

The only difference is the context-aware layer. Any performance difference is caused by our addition.

---

## How the Context Layer Works

Every tracker already computes bounding box positions for every vehicle in every frame. We use that data to compute one number per frame: the stationary ratio, which is the percentage of all tracked vehicles that are currently not moving.

Three checks use this ratio:

**Check 1: Is traffic collectively stopped?**
If more than 60% of tracked vehicles are stationary, suppress the violation flag. The candidate is probably stopped with traffic, not parked.

**Check 2: Did traffic clear but this vehicle stayed?**
If the stationary ratio drops (traffic starts moving) but the candidate vehicle does not move, reinstate the violation. It stayed behind while everyone else left. This prevents genuine violations from being permanently suppressed just because they started during a red light.

**Check 3: Is this vehicle stopped alone?**
If traffic is flowing and only the candidate is stationary, confirm the violation with high confidence. It is an anomaly.

---

## What We Will Measure

Precision is the primary metric. Did we reduce false positives? Recall is the secondary metric. Did we still catch real violations?

We also categorize every false positive by cause (red light, congestion, tracking error, other) and compare between baseline and proposed to verify we specifically reduced traffic-related false positives.

We run an ablation study, testing each check one at a time, to show which check contributes the most.

We run a sensitivity analysis on the 60% congestion threshold, testing values from 30% to 80%, to find the best tradeoff between suppressing false positives and retaining genuine violations.

---

## Data

We need fixed-camera footage of a road with a no-parking zone near a signalized intersection. The footage must contain free-flowing traffic, red light stops where multiple vehicles are stationary, and at least some genuine parking violations. Every stationary vehicle event gets manually annotated with ground truth labels.
