
# CHAPTER II — REVIEW OF RELATED LITERATURE AND STUDIES

## 2.1 Illegal Parking as an Urban Traffic Management Problem

Purpose:
Establish the significance of illegal parking as a costly, persistent urban problem, motivating automated detection before the chapter turns to technical approaches.

Key discussion:
Economic and congestion costs of on-street illegal parking; the shift toward AI/computer-vision-based enforcement as manual enforcement's limitations become apparent; positioning within the broader smart-city/edge-AI video-analytics landscape.

Recommended papers:
28 (economic cost estimation), 5 (systematic review, general framing), 6 *or* 51 — pick one for local/behavioral significance, not both, 30 (deployment-planning context).

Connection to study:
Motivational grounding only; no technical claim carried forward. Keep to roughly one page.

## 2.2 Computer Vision-Based Detection and Tracking in Fixed-Camera Traffic Surveillance

Purpose:
Frame the detection layer (finding and tracking vehicles reliably) as a distinct, largely-matured problem from the decision layer your thesis addresses, and explicitly scope what your thesis does not attempt to solve.

Recommended papers (section-level framing; no dedicated anchor at this level — see subsections):
—

### 2.2.1 Evolution from Background-Subtraction to Deep-Learning-Based Detection

Purpose:
Brief historical arc situating your own deep-learning detector choice within the field's technical trajectory.

Recommended papers:
2, 23, 36, 41, 45, 46, 48 — cite 2–3 representative examples in a compressed paragraph rather than individually discussing all seven.

Connection:
Background only; keep compressed.

### 2.2.2 Occlusion as a Persistent Detection-Layer Challenge

Purpose:
Document, with direct citable evidence, that occlusion remains unresolved even in dedicated occlusion-handling work — letting you state plainly that your thesis assumes reliable detection/tracking as a precondition and does not itself solve occlusion.

Recommended papers:
13 (+ duplicate) — direct quote-paraphrase: occlusion as "the biggest challenge"; Gao/Birch-Sussex-keypoint tracking as one concrete occlusion-tolerance technique.

Connection:
Defense-critical scoping paragraph — pre-empts "does your system handle occlusion?"

## 2.3 Single-Frame Spatial Reasoning as a Decision Paradigm

Purpose:
Establish your RQ2 baseline and show its limitations are representative of common field practice, not an isolated weakness.

Key discussion:
Detection-plus-static-spatial-test as the dominant simple decision rule; its quantified failure mode for the parking class specifically.

Recommended papers:
**1 (Abella & Catedrilla) — full discussion, anchor of this section**, with 18, 24, 29 cited briefly as a corroborating cluster. Drop 38, 43, 47, Hosmani as redundant with 18/24/29 unless you want a passing footnote mention.

Connection to study:
Directly feeds RQ2; state the 0.7467 precision / 509 of 2,803 false-positive figures explicitly here.

## 2.4 Context Mechanisms Dependent on Fixed Infrastructure

Purpose:
Primary evidentiary basis for your gap — where fixed-camera systems attempt context beyond single-frame reasoning, that context is anchored to a fixed scene reference vulnerable to occlusion or poor placement.

Recommended papers:
**9, 19, 27 — full discussion, anchors**; 22 and 37 (StreetHAWK) as genuinely new-angle corroboration (dashcam/lane-marking hybrid; mobile-platform-with-fixed-anchor). Drop or compress 36 and 46 as redundant with 27; Motorcycle-YOLOv12 optional as a fourth, more recent example.

Connection to study:
Direct support for your Chapter 1 gap paragraph; broadened beyond your current draft via 37.

## 2.5 Infrastructure-Independent Temporal and Re-Identification Approaches

Purpose:
Show the field has already moved past fixed-infrastructure dependency toward candidate-centric temporal/identity robustness — and that even this more advanced work cannot resolve sustained-congestion ambiguity, since it reasons only about the candidate vehicle itself.

Recommended papers:
**31 (Paode et al.) and An et al./Prohibited Parking — full discussion, anchors** (currently absent from your Ch.1 draft — highest-priority addition from this entire review); 45, BS-YOLO, Wrong-side as brief supporting examples.

Connection to study:
Sharpens the gap against the strongest, most current competing related work; frame per Stage 8's Claim 3 correction — a structural limitation you identify, not a literature-demonstrated failure.

## 2.6 Contextual Motion Reasoning on Mobile and Aerial Platforms

Purpose:
Establish that external-agent relative motion has precedent as a useful signal — but only as a single pairwise relationship under a hardcoded threshold, never as an aggregate over multiple surrounding vehicles.

Recommended papers:
**16 (iPatrol+) as primary, with 15 mentioned only as its earlier version — avoid full duplicate discussion**; 33 as a second anchor; 11 (Gong et al.) for its distance-time-series/threshold-optimization methodology. Compress 20, 24, 39 to one sentence.

Connection to study:
Supports Stage 4 Link 6 — use the Stage 8 Claim 4 correction precisely: "relative motion to a single external reference," never "collective behavior."

## 2.7 Geometric and Tracking Infrastructure Available to Fixed-Camera Systems

Purpose:
Establish the structural opportunity — fixed cameras already produce calibrated ground-plane coordinates and continuous multi-object tracking, meaning your feature vector's raw material is already computed in existing pipelines and simply discarded.

Recommended papers:
**49 (homography) — anchor**; 42 and 50 for tracker benchmarking (HOTA/MOTA/IDF1, frame-rate). Reference the broader MOT-usage list (10, 19, 33, 36, 39, 46, BS-YOLO, Gao/Birch-Sussex, 20) collectively — do not re-summarize papers already discussed in 2.4–2.6.

Connection to study:
Direct support for your "tracker output already exists, is discarded after tracking" claim.

## 2.8 Machine-Learned Spatiotemporal Classification in Adjacent Traffic Problems

Purpose:
Support methodological plausibility of an engineered spatiotemporal-feature-vector-to-trained-classifier pipeline, using precedent from adjacent (non-illegal-parking) problems — worded carefully to avoid overstating direct precedent.

Recommended papers:
**21 (Song et al.) as the closer structural analogue**; **17 (Jiang et al.)**, cited narrowly — XGBoost as feature-transformer in an unrelated forecasting task, explicitly not a decision-classifier precedent.

Connection to study:
Justifies "why a trained classifier, why this general design pattern" — pairs directly with the Stage 9 missing-literature flag on XGBoost-as-final-classifier, which you should acknowledge as an open methodological point rather than let a panelist surface first.

## 2.9 Synthesis: Positioning of the Present Study

Purpose:
Condensed restatement of the Stage 4 argument chain (2.2→2.8), ending at the unclaimed space your thesis occupies — bridges Chapter 2 to your Statement of the Problem.

Recommended papers:
No new citations — references back to sections above.

Connection to study:
Makes RQ1/RQ2 read as the evident next question rather than an abrupt pivot.

---

Here's the story, in order:

**1. Start with why anyone should care.**
Illegal parking isn't just an annoyance — it costs cities money, causes congestion, makes manual enforcement impractical at scale. This is just throat-clearing, one page, to get the reader on board before the real argument starts.

**2. Say what's already solved, so the reader knows what's *not* your problem.**
Before you can decide if a vehicle is illegally parked, you first have to reliably *see* it — detect it, keep track of it across frames, even when it's partly hidden behind another car. That part of the puzzle is fairly mature. You cite papers showing this evolution (old-school background-subtraction methods → modern deep learning detectors) and specifically show that even the best of these still struggle with occlusion. Why include this? So that later, when you present your own system, nobody in the room thinks you're claiming to have solved occlusion too. You're saying: "detection is handled, this thesis is about what happens *after* detection."

**3. Introduce the simplest possible way people have made the final decision — and show it breaking.**
Once you can see a vehicle, the laziest way to decide "violation or not?" is to just check: is it inside a no-parking zone, in this one frame? That's it. Your baseline paper (Abella & Catedrilla) does almost exactly this, and you have their actual numbers: their precision for illegal parking drops to 0.75, with 509 false detections out of 2,803. That's not a hypothetical weakness — it's a documented one. This is the moment the reader goes "oh, I see the problem now."

**4. Show the first real attempt at a smarter fix — and its weak point.**
So researchers didn't stop there. Some added a smarter rule: "only flag a violation if the traffic light is red" or "only flag it if it's inside this exact drawn zone on the road." That's better! But notice the pattern — every one of these smarter rules depends on *seeing something fixed in the scene*: a traffic light, a painted zone, a lane marking. What happens if a truck blocks the camera's view of that traffic light, or the zone marking is worn away, or the camera is angled badly? The whole safety mechanism breaks. This is the first crack in the wall.

**5. Show an even newer, cleverer attempt — and its weak point too.**
More recent papers (2025–2026, so genuinely current) got smarter still: instead of depending on a fixed reference in the scene, they watch the *same car* over time — does it disappear and reappear? Does it look the same color/shape across frames despite lighting changes? This avoids the "fixed infrastructure" problem entirely. But here's the catch, and it's subtle: these methods are still only looking at *one car* — the one that might be parked. They have no way to tell the difference between "this car has been sitting still for 10 minutes because it's illegally parked" and "this car has been sitting still for 10 minutes because the whole street is gridlocked and it literally cannot move." Both look identical if all you're watching is that one car.

**6. Point out that someone, somewhere, already had a good idea — just not here.**
Now here's a twist that makes your argument sharper, not weaker: on *mobile* cameras (dashcams, phones), some researchers already realized "hey, maybe I should look at what's happening around the target car, not just the target car itself." That's promising! But when you actually read how they did it, it's very narrow — they only ever compare their own moving camera's motion to *one* other car, using a simple hardcoded number (like "if relative speed drops below X, call it stopped"). Nobody ever said "let me look at *all* the nearby cars together and learn a pattern from that."

**7. Point out that fixed cameras are actually in the best position to do this — and nobody's doing it.**
Here's where it gets satisfying: fixed cameras (the kind your thesis uses) don't have the messy "my own camera is also moving" problem that dashcams have. They already run tracking software that follows *every* car in the frame, all the time — that data is just sitting there, unused, thrown away after each frame. Fixed cameras also can be calibrated so pixel distances become real-world distances (via homography), which mobile cameras struggle to do reliably. So fixed cameras have both the ingredients and the stability to do something dashcams tried imperfectly — and yet no fixed-camera paper in the entire 62-paper review pulls in the *aggregate* motion of nearby vehicles to help decide if one specific car is a violation.

**8. Reassure the reader that your approach isn't science fiction.**
Just to show your idea isn't coming out of nowhere: you point to a couple of unrelated traffic papers (not about parking at all) that already prove the general recipe works — take a bunch of movement-based numbers, feed them into a trained model, get a smart decision out. That's the same basic idea your thesis uses, just never applied to this exact problem.

**9. Land the plane.**
By the end, the reader has walked: seeing is solved → the simplest decision rule fails → fixed-reference fixes are fragile → watching-one-car fixes still can't tell gridlock from guilt → someone tried watching neighboring cars but only clumsily, on the wrong platform → fixed cameras are perfectly positioned to do it properly and nobody has → and the general "feed movement data into a trained model" idea already works elsewhere. So your thesis's idea — build a 6-number snapshot of what's happening around a candidate car, and let a trained model (XGBoost) decide, instead of a hand-written rule — isn't a wild leap. It's the obvious next step nobody happened to take.


