
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

### 1. Core Literature
1, 9, 19, 27, 31, An et al./Prohibited Parking, 13, 49, 15/16, 33, 21.

### 2. Supporting Literature
2, 22, 23, 36, 37, 41, 42, 45, 46, 48, 50, 11, 17, 18, 20, 24, 28, 29, 5, 6 or 51, 30, BS-YOLO, Wrong-side, Motorcycle-YOLOv12, Gao/Birch-Sussex-keypoint, 39.

### 3. Excluded Literature
32 (confirmed miscited/tangential — permitted-vehicle classification, not general violation detection), 34 (ontology/SPARQL, no comparable CV methodology), 51 or 6 (whichever isn't chosen for 2.1 — redundant with the other), 
3, 4, 7, 10, 12, 14, 25, 26/SafeStreets, 29(if trimmed further), 35, 38, 43, 47, 52, RKF-YOLO, Onstreet_parking_space_localization, Hosmani — all standard single-frame or basic detection pipelines with no distinguishing contribution beyond what a stronger, already-cited paper covers; conceptual_framework.pdf excluded per your instruction.

### 4. Research Gap Evidence
Mapped in full in Stage 8: Claims 1, 2, 5, 6, 7 strongly supported as originally worded; Claims 3 and 4 supported only in corrected, narrower form (structural-limitation framing for Claim 3; "single external reference" not "collective behavior" for Claim 4).

### 5. Recommended Chapter 2 Writing Flow
2.1 (why this matters) → 2.2 (detection is maturing, scope your thesis around the decision layer) → 2.3 (the simplest decision rule, quantified failure — your baseline) → 2.4 (first attempted fix: fixed infrastructure, and its vulnerability) → 2.5 (second, more recent attempted fix: candidate-centric temporal/re-ID, and its structural blind spot) → 2.6 (mobile platforms validate external-agent motion, but only pairwise and hardcoded) → 2.7 (fixed cameras already have the geometric and tracking infrastructure to go further) → 2.8 (the general design pattern you propose has adjacent precedent) → 2.9 (synthesis into your Statement of the Problem).

### 6. Paper-to-Section Mapping
As tabulated in Stage 7, with the redundancy trims applied above now folded in.

