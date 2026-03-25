---

## PART 1: RECAP OF ADVISER CONSULTATION

### What the adviser said (confirmed):

1. **The overall idea is approved.** Checking if surrounding vehicles are also stationary before confirming a violation is solid. He signed off on continuation.

2. **No arbitrary thresholds.** The 5-pixel displacement threshold was questioned. He wants parameters grounded in something, not just picked. His suggestion: let the experiments determine optimal values rather than defending arbitrary numbers upfront.

3. **Define illegal parking based on actual laws.** The ROI zone definition and what constitutes a violation should be grounded in real traffic regulations, not made up rules. This can be from any country as long as the logic matches our scenario.

4. **Simplify the pipeline.** Testing 9 pipeline combinations is unnecessary. Our contribution is the context-aware layer, not which YOLO or tracker is best. Pick one pipeline, test the layer on it.

5. **DeepSORT is overkill.** YOLO bounding boxes can be linked across frames with simple association (IoU matching). We do not need appearance-based re-identification for our use case. He suggested this for efficiency, not as a strict requirement.

6. **False positives as baseline.** Run the system without context layer, it produces false positives. Run with context layer, it produces fewer. Compare the two. That is the evaluation. We compare against our own system, not against other authors' numbers.

7. **Scenarios must match test site.** The problems we cite from our evidence papers (Gao, Wahyono, Kathait, Makmur) should be the same type of problem that occurs in our actual footage. If our camera is on a one-way road, we need to show that false positives from traffic stops happen on one-way roads too.

8. **Writing is too fluffy.** Get to the point. State methodology directly without unnecessary buildup.

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
We need to pick ONE pipeline instead of nine. Options:
- YOLOv8n + simple IoU tracker (lightest, adviser suggested simplicity)
- YOLOv8s + DeepSORT (middle ground, commonly used in literature)
- YOLOv8m + ByteTrack (heavier but potentially more stable tracking)

Decision needed: which one? Consider what our hardware can run and what is easiest to justify in writing.

### Question 2: Tracker choice
Adviser said DeepSORT is overkill. Options:
- Use simple IoU matching (link bounding boxes by spatial overlap between frames). Justification: we only need centroid positions and track persistence, not appearance features.
- Use DeepSORT anyway. Justification: it is the most commonly used tracker in recent illegal parking papers.
- Use ByteTrack. Justification: motion-only association, no appearance model, lighter than DeepSORT but more robust than raw IoU.

Decision needed: which one? This connects to Question 1.

### Question 3: How to ground the 5-pixel threshold
Options:
- Run a quick empirical test: place a stationary vehicle in frame, run detector for 100+ frames, measure maximum centroid jitter. If max jitter is 3-4 pixels, then 5 is justified.
- Make it a parameter in sensitivity analysis: test at 3, 5, 7, 10 pixels and report which works best.
- Both.

Decision needed: do we do the empirical test, the sensitivity analysis, or both?

### Question 4: Where is our actual test footage coming from?
- Do we already have footage recorded?
- If not, where will we set up?
- Is it confirmed one-way road with traffic signals?
- Does the footage contain enough signal cycles (vehicles stopping and going) to produce meaningful false positive events?

Decision needed: confirm footage source and verify it matches our scenario.

### Question 5: Do we keep the ablation study?
Adviser said focus on the proposal, not extensive ablation on parts that do not matter. But the ablation (baseline vs Check 1 only vs full system) directly tests our context-aware layer components. Options:
- Keep it. It is only 3 configs on one pipeline. Low effort, high value for defense.
- Remove it. Focus only on context ON vs OFF.

Recommendation: keep it. It is cheap to run and directly answers how each check contributes.

### Question 6: Do we keep the parameter sensitivity on τ?
This tests the congestion threshold at different values. It directly answers Research Question 3. Options:
- Keep it. Same pipeline, same footage, just change τ. Minimal extra work.
- Remove it. But then Research Question 3 has no answer.

Recommendation: keep it. It answers a research question.

---

## PART 4: WHAT TO FIX IN CHAPTER 2

### Fix 1: Add traffic law grounding
- Find Philippine traffic laws or any country's traffic laws that define illegal parking (where parking is prohibited, what duration constitutes parking vs stopping)
- Insert as a short subsection early in Chapter 2, or in Chapter 1 background
- Purpose: ground our ROI zone definition and temporal threshold in actual regulations, not arbitrary choices

### Fix 2: Add papers matching our test site scenario
- Find studies that use one-way road or single-direction traffic flow
- Does not need to be illegal parking specifically, can be any traffic violation detection or traffic monitoring study
- Needs to report false positives from traffic stops, congestion, or signal-related stationarity
- Purpose: show that the false positive problem we address occurs in our specific scenario, not just in intersections

### Fix 3: Possibly adjust evidence paper framing
- Our four main evidence papers (Gao, Wahyono, Kathait, Makmur) are from different scenarios (intersection, parking lot, junction)
- We do not remove them, they still document the pattern (problem acknowledged, fix not evaluated)
- But we may need additional papers from scenarios closer to ours to strengthen the argument that the problem exists in one-way road contexts too

### Fix 4: Review if any paper needs to be removed or repositioned
- Makmur et al. is motorcycle parking in a parking lot, which is quite different from our scenario
- It still shows the pattern, but if the adviser questions it, we should be ready to explain why it is included (same pattern of unevaluated differentiation mechanism, not same scenario)

---

## PART 5: WHAT TO FIX IN CHAPTER 3

### Fix 1: Remove 9-pipeline comparison
- Replace Experiment 1 (18 runs across 9 pipelines) with a single pipeline
- Core evaluation becomes: one pipeline, context OFF vs context ON
- Simplifies the entire evaluation design section significantly

### Fix 2: Update pipeline description
- Pick one detector and one tracker
- Justify the choice
- Remove all tables and discussion comparing detector variants and tracker variants

### Fix 3: Keep ablation study (pending group decision)
- If kept: 3 configs (baseline, Check 1 only, full system) on the single chosen pipeline
- Update section to reflect single pipeline, not best pipeline selected from Experiment 1

### Fix 4: Keep parameter sensitivity (pending group decision)
- If kept: test τ at multiple values on the single chosen pipeline
- Update section to reflect single pipeline

### Fix 5: Keep FP source analysis
- Categorize false positives from context OFF run by source (signal, congestion, detection error, other)
- Compare distribution against context ON run
- No changes needed to the structure, just runs on one pipeline instead of best pipeline from 9

### Fix 6: Ground the 5-pixel threshold
- Either add empirical justification (jitter measurement test) or move it to sensitivity analysis
- Remove or soften the current rationale that just says "absorbs centroid jitter"

### Fix 7: Remove pipeline characterization metrics table
- Total unique track IDs and average track duration were for comparing 9 pipelines
- With one pipeline, this becomes a single data point, not a comparison
- Can mention it briefly in results but does not need its own table

### Fix 8: Update scope and limitations
- Add that the system is designed for one-way or single-direction traffic flow
- Add that multi-directional traffic (intersections, two-way roads) would require lane-level segmentation and is identified as future work
- Remove references to 9 pipelines

### Fix 9: Simplify flowchart and parameter table
- No changes to the context-aware layer itself (Check 1 and Check 2 stay)
- No changes to the 5 parameters (T, N, δ, τ, W)
- Just update surrounding text that references pipeline selection and multi-pipeline comparison

---

## PART 6: PAPERS TO FIND

### Category 1: Traffic laws
- Philippine traffic law (RA 4136 or local ordinances) defining illegal parking
- Any country's traffic code defining no-parking zones, stopping vs parking distinction, and prohibited parking locations
- Purpose: ground our system definition in law

### Category 2: One-way road traffic monitoring studies
- Any computer vision traffic monitoring study on one-way roads or single-direction traffic
- Preferably reporting false positives from signal stops or congestion
- Does not need to be illegal parking specifically
- Purpose: show our scenario has the same problem

### Category 3: Additional illegal parking studies
- Any illegal parking study not already in our 28 papers that documents the false positive problem from legitimate stops
- Purpose: strengthen evidence base beyond four main papers
