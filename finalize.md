

Here's the complete final version with your additions integrated at the bottom as a separate section:

---

## PART 1: RECAP OF ADVISER CONSULTATION

### What the adviser said:

1. **The overall idea is approved.** Checking if surrounding vehicles are also stationary before confirming a violation is good.

2. **No arbitrary thresholds.** The 5-pixel displacement threshold was questioned. He wants parameters grounded in something, not just picked. His suggestion: let the experiments determine optimal values rather than defending arbitrary numbers upfront.

3. **Define illegal parking based on actual laws.** The ROI zone definition and what constitutes a violation should be grounded in real traffic regulations, not made up rules. This can be from any country as long as the logic matches our scenario.

4. **Simplify the pipeline.** Testing 9 pipeline combinations is unnecessary. Our contribution is the context-aware layer, not which YOLO or tracker is best. Pick one pipeline, test the layer on it.

5. **DeepSORT is overkill.** YOLO bounding boxes can be linked across frames with simple association (IoU matching). We do not need appearance-based re-identification for our use case. He suggested this for efficiency, not as a strict requirement.

6. **False positives as baseline.** Run the system without context layer, it produces false positives. Run with context layer, it produces fewer. Compare the two. That is the evaluation. We compare against our own system, not against other authors' numbers.

7. **Scenarios must match test site.** The problems we cite from our evidence papers (Gao, Wahyono, Kathait, Makmur) should be the same type of problem that occurs in our actual footage. If our camera is on a one-way road, we need to show that false positives from traffic stops happen on one-way roads too.

### What I think the adviser meant (my interpretation, needs confirmation with group):

- He was NOT saying skip the ON/OFF comparison. He was saying the OFF run itself is the baseline. We do not need external baselines from other papers.
- He was NOT saying do not use DeepSORT at all. He was suggesting simpler tracking might be sufficient. We can still use whatever tracker we want as long as we justify it.
- When he asked about the 5-pixel threshold, he may have confused it with our definition of stationarity for the violation itself, not understanding it is just the centroid jitter noise floor. We need to explain this better next time.

---

## PART 2: WHAT WE FINALIZED (group decisions after consultation)

1. **Test site scenario: one-way road only.** Our context-aware layer by design treats all tracked vehicles in the scene as being in the same traffic flow direction. On a two-way road or intersection, vehicles going the opposite direction or on cross streets would pollute the stationary ratio. Single direction traffic (one-way road or highway) is our scope.

2. **No intersection scenarios for now.** Intersections add complexity because we would need lane segmentation to separate traffic flows by direction. This is possible but adds an extra layer of work. First version targets one-way only.

3. **Dataset must match this scenario.** We need footage from a one-way road or one-direction traffic flow where vehicles stop due to traffic signals or congestion and then resume. This gives us the false positive events our layer is designed to handle.

---

## PART 3: OPEN QUESTIONS (group needs to finalize, no adviser needed)

### Question 1: Single pipeline choice
~~We need to pick ONE pipeline instead of nine.~~
**FINALIZED: YOLOv8n + simple IoU template matching.** Adviser prefers efficiency. Deep learning trackers are resource expensive.

### Question 2: Tracker choice
~~Adviser said DeepSORT is overkill.~~
**FINALIZED: Simple IoU template matching.** We only need centroid positions and track persistence, not appearance-based re-identification. Fixed camera with stable geometry means spatial overlap between consecutive frame detections is sufficient.

### Question 3: How to ground the 5-pixel threshold
**Still open.** Need to find a formula or method that grounds this in something meaningful, like actual vehicle speed or physical distance. Camera angle and height may affect how many pixels correspond to real-world movement. Options remain:
- Empirical jitter test on stationary vehicle
- Sensitivity analysis at multiple values
- Find a formula that converts pixel displacement to real-world distance based on camera parameters
- Some combination of the above

### Question 4: Where is our actual test footage coming from?
**Still open.** Must be:
- One-way road or single-direction traffic flow
- Has traffic signals causing vehicles to stop and go
- Contains enough signal cycles to produce meaningful false positive events

### Question 5: Do we keep the ablation study?
**FINALIZED: Yes.** 3 configs (baseline, Check 1 only, full system) on the single pipeline. Low effort, high value for defense.

### Question 6: Do we keep the parameter sensitivity on τ?
**FINALIZED: Yes.** It directly answers Research Question 3.

---

## PART 4: WHAT TO FIX IN CHAPTER 1

### Fix 1: Add traffic law grounding
- Find Philippine traffic laws (RA 4136 or local ordinances) or any country's traffic laws that define illegal parking
- Define what constitutes illegal parking: where parking is prohibited, what duration constitutes parking vs stopping
- Insert as a section in Chapter 1 background
- Purpose: ground our ROI zone definition and temporal threshold in actual regulations

---

## PART 5: WHAT TO FIX IN CHAPTER 2

### Fix 1: Add papers matching our test site scenario
- Find studies that use one-way road or single-direction traffic flow
- Does not need to be illegal parking specifically, can be any traffic violation detection or traffic monitoring study
- Needs to report false positives from traffic stops, congestion, or signal-related stationarity
- Purpose: show that the false positive problem we address occurs in our specific scenario, not just in intersections
- We do not know yet if we will find any

### Fix 2: Add additional illegal parking or context-aware studies
- Any illegal parking study not already in our 28 papers that documents the false positive problem from legitimate stops
- Preferably any study with context-aware approaches if they exist
- Purpose: strengthen evidence base beyond four main papers

### Fix 3: Possibly adjust evidence paper framing
- Our four main evidence papers (Gao, Wahyono, Kathait, Makmur) are from different scenarios (intersection, parking lot, junction)
- We do not remove them, they still document the pattern (problem acknowledged, fix not evaluated)
- But we may need additional papers from scenarios closer to ours to strengthen the argument
- Makmur et al. is motorcycle parking in a parking lot, quite different from our scenario. Still shows the pattern but be ready to defend why it is included

---

## PART 6: WHAT TO FIX IN CHAPTER 3

### Fix 1: Remove 9-pipeline comparison
- Replace with single pipeline: YOLOv8n + simple IoU template matching
- Core evaluation becomes: one pipeline, context OFF vs context ON
- Remove all tables and discussion comparing detector variants and tracker variants
- Modify any text that mentions selection of the best pipeline for other experiments

### Fix 2: Update pipeline description
- Describe YOLOv8n as the detector and simple IoU matching as the tracker
- Justify the choice: adviser recommended efficiency, we only need centroid positions and track persistence, appearance-based re-identification is unnecessary for fixed camera with stable geometry

### Fix 3: We will NOT compare our framework against frameworks from other authors using our own dataset
- Different scenarios, different datasets, no access or detailed knowledge of their detection implementation
- We create our own baseline: context OFF vs context ON, on our own dataset only

### Fix 4: Keep ablation study
- 3 configs (baseline, Check 1 only, full system) on the single pipeline
- Update section to reflect single pipeline, remove references to best pipeline selection from Experiment 1

### Fix 5: Keep parameter sensitivity on τ
- Test τ at multiple values on the single pipeline
- Update section to reflect single pipeline

### Fix 6: Keep FP source analysis
- Categorize false positives from context OFF run by source (signal, congestion, detection error, other)
- Compare distribution against context ON run
- Some false positives may be out of our control (detection errors, occlusion), annotate and describe what happened in each case
- Runs on one pipeline instead of best pipeline from 9

### Fix 7: Ground the 5-pixel threshold
- Find a formula, reference, or method that makes this meaningful in terms of actual vehicle speed or physical distance
- Camera angle and height may matter for pixel-to-real-world conversion
- If no formula found, move to sensitivity analysis and let experiments determine optimal value
- Remove or soften the current rationale that just says "absorbs centroid jitter"

### Fix 8: Remove pipeline characterization metrics table
- Total unique track IDs and average track duration were for comparing 9 pipelines
- With one pipeline this is a single data point, not a comparison
- Can mention briefly in results but does not need its own table

### Fix 9: Update scope and limitations
- Add that the system is designed for one-way or single-direction traffic flow
- Add that multi-directional traffic (intersections, two-way roads) would require lane-level segmentation and is identified as future work
- Remove references to 9 pipelines

### Fix 10: Simplify flowchart and parameter table
- No changes to the context-aware layer itself (Check 1 and Check 2 stay)
- No changes to the 5 parameters (T, N, δ, τ, W)
- Just update surrounding text that references pipeline selection and multi-pipeline comparison
