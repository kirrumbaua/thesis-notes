---

## PART 1: ADVISER CONSULTATION RECAP

1. **The overall idea is approved.** Checking if surrounding vehicles are also stationary before confirming a violation is good.

2. **No arbitrary thresholds.** The 5-pixel displacement threshold was questioned. He wants parameters grounded in something meaningful, possibly tied to real-life measurement like actual vehicle speed, taking into account camera angle, height, or some formula for pixel-to-distance conversion. Not just a number we picked. We will fix the default value to make it more grounded, but we will still have a separate parameter sensitivity analysis to test what is the most optimal pixel displacement threshold.

3. **Define illegal parking based on actual laws.** The ROI zone definition and what constitutes a violation should be grounded in real traffic regulations, not made up rules. This can be from any country as long as the logic matches our scenario. (Although we already know this we just need to discuss it on the paper explicitly).

4. **Simplify the pipeline.** Testing 9 pipeline combinations is unnecessary. Our contribution is the context-aware layer, not which YOLO or tracker is best. Pick one pipeline, test the layer on it.

5. **DeepSORT is overkill.** He was suggesting simpler tracking might be sufficient, not that we can never use it. We can still use whatever tracker we want as long as we justify it. He suggested this for efficiency, not as a strict requirement.

6. **False positives as baseline.** Run the system without context layer, it produces false positives. Run with context layer, it produces fewer. Compare the two. The OFF run itself is the baseline. We compare against our own system, not against other authors' numbers. We do not need external baselines from other papers.

7. **Scenarios must match test site.** The problems we cite from our evidence papers should be the same type of problem that occurs in our actual footage.

---

## PART 2: FINALIZED DECISIONS (March 29 Group Meeting)

### Scope

- **One-way / single-direction traffic only.** Our context-aware layer treats all tracked vehicles as being in the same traffic flow direction. Opposite-direction or cross-traffic would pollute the stationary ratio. Multi-directional scenarios are future work.
- **Dataset must match this scenario.** One-way road or single-direction traffic flow where vehicles stop due to signals or congestion and then resume.
- **We will NOT compare our framework against other authors' frameworks on our own dataset.** Different scenarios, no access to their implementations. We create our own baseline: context OFF vs context ON, on our own dataset only.

### Pipeline

- **One pipeline only: YOLOv8n + simple IoU template matching.** Adviser prefers efficiency. Deep learning trackers are resource expensive and unnecessary since we only need centroid positions and track persistence. Remove all 9-pipeline comparison content and any mention of selecting the best pipeline for other experiments.

### Evaluation

- **Keep ablation study.** 3 configs (baseline, Check 1 only, full system) on the single pipeline.
- **Keep parameter sensitivity on τ and δ.** Test multiple values for both the congestion threshold and the displacement threshold.
- **Keep false positive source analysis.** Some false positives may be out of our control (detection errors, occlusion). We annotate them so we can describe what happened.

### What to Fix in the Paper

- **Chapter 1:** Add discussion of specific traffic laws and the legal definition of illegal parking to ground the study in actual regulations.
- **Chapter 2:** Try to find one-way road or single-direction traffic monitoring studies relevant to our scenario. Find additional illegal parking studies, preferably any with context-aware approaches. Keep the four main evidence papers (Gao, Wahyono, Kathait, Makmur) but be ready to defend Makmur since it is a parking lot scenario different from ours.
- **Chapter 3:** Ground the 5-pixel displacement threshold by finding a formula or method tied to actual vehicle speed or physical distance considering camera angle and height. Additionally, include δ in the parameter sensitivity analysis to test multiple values and determine the optimal threshold experimentally. Remove pipeline characterization metrics table. Remove 9-pipeline content. Update scope and limitations to reflect single-direction constraint. Simplify flowchart and parameter table text. The context-aware layer itself does not change (Check 1, Check 2, all 5 parameters stay), just remove surrounding text about pipeline selection.

---
