

# Chapter 3: Methodology — Complete Writing Guide (Final)

---

## 3.1 — Conceptual Framework

**What this section is about:** Bridges Chapter 2 to the proposed study and presents the system design.

**What to write:**

**Opening paragraph (no subsection — just the section opener):**
One paragraph connecting Chapter 2 to this study. Reference Section 2.6.4 and the three fixed-camera studies (Gao 2022, Wahyono 2017, Kathait 2025). State the gap: all existing responses evaluate the candidate in isolation against fixed external references; none uses surrounding vehicle behavior. State that this study proposes a context-aware decision logic layer using surrounding vehicle displacement data.

---

### 3.1.1 — System Architecture Overview

**What this section is about:** Presents the full system design as a proper diagram.

**What to write:**

**Figure 1 — System Architecture Diagram**

This must be a proper technical diagram (made in draw.io, Figma, or similar), not a simple flowchart. It should show **both the baseline path and the proposed path side by side or as a branching diagram**. The figure must contain:

| Component | What to show in the diagram |
|---|---|
| Input | Fixed camera video frame |
| Detection stage | Box labeled with "YOLOv8 (n / s / m)" — show the output: bounding boxes with class labels and confidence scores |
| Tracking stage | Box labeled with "Tracker (DeepSORT / ByteTrack / BoT-SORT)" — show the output: tracked bounding boxes with persistent track IDs and centroids |
| Zone membership check | Box showing centroid tested against zone polygon — output: in-zone / out-of-zone status per track |
| Temporal threshold check | Box showing stationarity timer — output: candidate violation flag when timer exceeds T |
| **Branch point** | Baseline path goes straight to output. Proposed path enters the context-aware layer. |
| Context-aware layer (proposed only) | Enlarged box or bordered region showing: (1) displacement computation for all vehicles, (2) stationary ratio R, (3) Check 1 / Check 2 / Check 3 logic, (4) confirm or suppress decision |
| Baseline output | Violation / Non-violation |
| Proposed output | Violation Confirmed / Suppressed + Traffic State Label + Confidence |

**Notes for groupmates:**
- The detection and tracking boxes should be visually marked as **variable** (e.g., dashed border, or a label saying "swappable").
- The context-aware layer box should be visually marked as the **contribution** (e.g., highlighted border, different color).
- Show the **data type** flowing between each component along the arrows (frames → bounding boxes → tracked centroids → displacement values → decision).
- This is probably a full-page figure.

After the figure, write one paragraph describing the baseline path and one paragraph describing the proposed path. Keep it short — the figure does the heavy lifting.

Write one paragraph stating the factorial design:

| Factor | Levels | Role |
|---|---|---|
| Detection-tracking pipeline | 9 combinations (3 detectors × 3 trackers) | Upstream variable |
| Context-aware layer | ON or OFF | Treatment variable (contribution) |

State that both configurations process the same video with the same zone and threshold. The only variable within each pipeline pair is the context layer.

---

### 3.1.2 — Contribution Positioning

**What this section is about:** Maps the proposed approach against existing approaches from Chapter 2.

**What to write:**

One sentence introducing the table: the proposed layer addresses the challenge documented in Section 2.6.4 through a different mechanism than existing approaches.

**Table — Comparison with existing approaches:**

| Existing Approach | What It Requires | What It Covers | What It Does Not Cover |
|---|---|---|---|
| Pedestrian zone proxy (Gao et al., 2022) | Visible crosswalk, pedestrian detection, two additional zones | Signal-related stops at intersections with crosswalks | Congestion, queue spillback, unsignalized locations, periods without pedestrians |
| Temporal filter (Wahyono et al., 2017) | Nothing additional | Stops shorter than 60 seconds | Congestion or parking longer than 60 seconds |
| Signal color detection (Kathait et al., 2025) | Visible traffic signal head | Signal-related stops where signal is visible | Locations without visible signals, congestion, queue spillback |
| **Proposed: Surrounding vehicle displacement** | Tracking output already produced | Any collective stopping regardless of cause | — |

One paragraph after the table stating the proposed approach uses data the tracking module already produces. No additional models, calibration, or external data.

**Length for all of 3.1:** 3–3.5 pages.

---

## 3.2 — Data Collection and Preparation

**What this section is about:** Describes the video data and how it will be prepared.

---

### 3.2.1 — Dataset Requirements

**What this section is about:** Lists the properties the footage must have.

**What to write:**

Five requirements, each one to two sentences:

1. **Static camera position** — fixed mount, static angle, restricted zone visible throughout.
2. **Mixed traffic scenarios** — must contain all three:

| Scenario Type | What It Contains | Why It Is Needed |
|---|---|---|
| Free-flowing traffic | Vehicles passing through zone without stopping | Verifies no false triggers during normal flow |
| Collective stops | Multiple vehicles stationary simultaneously | Tests the context-aware layer's suppression logic |
| Genuine violations | Vehicles actually parked illegally in the zone | Verifies genuine violations are still detected |

3. **Observable traffic signal effects** — signalized intersection visible or its effects observable.
4. **Multiple simultaneous vehicles** — stationary ratio requires more than one tracked vehicle.
5. **Visible zone boundaries** — lane markings or curb edges visible for polygon definition.

---

### 3.2.2 — Data Source

Discard the time period requirement

**What this section is about:** States where the footage comes from.

**What to write:**

Use whichever template applies:

**If recording original footage:** State location, camera specs (height, resolution, frame rate), why this site satisfies requirements, and recording schedule:

| Time Period | Purpose |
|---|---|
| Morning (7:00–9:00) | Moderate to heavy traffic with signal cycles |
| Midday (11:00–13:00) | Light to moderate traffic |
| Afternoon (16:00–18:00) | Heavy traffic, potential congestion |

**If using existing footage:** State source, confirm it meets all five requirements, state duration and scenarios present.

---

### 3.2.3 — Ground Truth Annotation Protocol

**What this section is about:** Defines how events are labeled for evaluation.

**What to write:**

**Event definition:** One sentence — a stationary vehicle event is a continuous period where a tracked vehicle remains within the restricted zone with displacement below the movement threshold.

**Table — Annotation labels:**

| Label | Data Type | Values | Description |
|---|---|---|---|
| Event ID | Integer | Unique sequential | Identifies the stationary event |
| Vehicle track ID | Integer | From tracker output | Links event to tracker identity |
| Start frame | Integer | Frame number | First frame vehicle is stationary in zone |
| End frame | Integer | Frame number | Frame vehicle departs or resumes movement |
| Duration | Float | Seconds | Calculated from start/end frames and frame rate |
| Vehicle type | Categorical | Car, truck, bus, van, motorcycle | Visual classification by annotator |
| Ground truth status | Binary | Violation, Non-violation | Is this genuinely illegally parked? |
| Stop reason | Categorical | Red light, congestion, loading, yielding, other, N/A | Why vehicle is legitimately stopped (if non-violation) |
| Surrounding traffic state | Categorical | Flowing, mixed, collectively stopped | Annotator assessment of surrounding vehicle behavior |

**Annotation procedure:** Bullet list — annotator observes candidate behavior, surrounding traffic, contextual cues. One paragraph on how ground truth status is assigned.

**Inter-annotator reliability:** 20% independently labeled by second annotator. Cohen's kappa. Disagreements resolved by joint review.

---

### 3.2.4 — Video Preprocessing

**What this section is about:** Technical preparation before experiments.

**What to write:**

Three steps:

1. **Frame extraction** — processing frame rate (e.g., 10 fps), held constant across all configurations.
2. **Zone polygon definition** — vertex coordinates selected on reference frame, same polygon for all configurations.

| Vertex | X Coordinate | Y Coordinate |
|---|---|---|
| V1 | [value] | [value] |
| V2 | [value] | [value] |
| V3 | [value] | [value] |
| V4 | [value] | [value] |

3. **Detection and tracking parameters** — determined through preliminary testing, held constant across all runs.

**Length for all of 3.2:** 4–5 pages.

---

## 3.3 — System Architecture

**What this section is about:** Describes every component of the pipeline from input to baseline output.

---

### 3.3.1 — Vehicle Detection Module

**What this section is about:** Describes the three detector variants.

**What to write:**

State three YOLOv8 variants will be used. Same architecture family, differ only in model capacity. All pretrained on COCO, no fine-tuning. Same class filter and confidence threshold across all three.

**Table — Detector specifications:**

| Variant | Parameters | GFLOPs | COCO mAP50-95 |
|---|---|---|---|
| YOLOv8n (Nano) | 3.2M | 8.7 | 37.3 |
| YOLOv8s (Small) | 11.2M | 28.6 | 44.9 |
| YOLOv8m (Medium) | 25.9M | 78.9 | 50.2 |

One sentence noting these are Ultralytics benchmark values; actual performance on evaluation footage may differ.

**Table — Detection class filter:**

| COCO Class ID | Class Name | Included |
|---|---|---|
| 2 | Car | Yes |
| 3 | Motorcycle | Yes |
| 5 | Bus | Yes |
| 7 | Truck | Yes |
| Other | — | No |

One paragraph on why within-family variants instead of cross-generation models: isolates model capacity as the single variable without confounding architectural differences.

One sentence on detection output: bounding box coordinates, class label, confidence score per detection per frame, passed to tracking module.

---

### 3.3.2 — Vehicle Tracking Module

**What this section is about:** Describes the three tracker variants.

**What to write:**

State three trackers will be used, each representing a different tracking approach. For each, write 2–3 sentences on how it works and a properties table.

**DeepSORT** — Kalman Filter motion prediction + deep ReID appearance matching.

| Property | Detail |
|---|---|
| Association method | Hungarian algorithm on combined motion + appearance cost |
| Motion model | Kalman Filter |
| Appearance model | Deep ReID network (128-dimensional feature vector) |
| Occlusion handling | Track maintained for max_age frames without detection |
| Literature use | Sharma et al. (2023), Kathait et al. (2025) |

**ByteTrack** — Motion-only, two-stage association (high-confidence first, then low-confidence).

| Property | Detail |
|---|---|
| Association method | IoU-based matching in two stages |
| Motion model | Kalman Filter |
| Appearance model | None |
| Occlusion handling | Low-confidence second pass recovers partially occluded vehicles |
| Literature use | Widely adopted in MOT benchmarks; integrated in Ultralytics |

**BoT-SORT** — Kalman + ReID + camera motion compensation via sparse optical flow.

| Property | Detail |
|---|---|
| Association method | Combined motion + appearance + camera motion compensation |
| Motion model | Kalman Filter with camera motion correction |
| Appearance model | ReID network |
| Occlusion handling | Appearance re-identification + motion correction |
| Literature use | Kuo et al. (dashcam); top MOT benchmark performer |

One paragraph on why these three: they span motion-only → appearance-assisted → state-of-the-art combined.

One paragraph on why tracker choice matters for the context-aware layer: the layer depends on displacement from tracked centroids; tracker stability directly affects stationary ratio reliability.

**Table — Tracking output (same for all three):**

| Output | Data Type | Description |
|---|---|---|
| Track ID | Integer | Persistent identifier |
| Bounding box | (x, y, w, h) | Current bounding box |
| Centroid | (cx, cy) | Center of bounding box |
| Track age | Integer | Frames this track has existed |
| Track state | Categorical | Confirmed, tentative, or deleted |

---

### 3.3.3 — Pipeline Combinations

**What this section is about:** Defines the nine detector-tracker pipelines.

**What to write:**

**Table — Pipeline matrix:**

| Pipeline ID | Detector | Tracker |
|---|---|---|
| P1 | YOLOv8n | DeepSORT |
| P2 | YOLOv8n | ByteTrack |
| P3 | YOLOv8n | BoT-SORT |
| P4 | YOLOv8s | DeepSORT |
| P5 | YOLOv8s | ByteTrack |
| P6 | YOLOv8s | BoT-SORT |
| P7 | YOLOv8m | DeepSORT |
| P8 | YOLOv8m | ByteTrack |
| P9 | YOLOv8m | BoT-SORT |

One paragraph: all pipelines share identical settings for everything else (input video, frame rate, class filter, confidence threshold, zone polygon, temporal threshold). Each tested with and without the context-aware layer = 18 total configurations.

---

### 3.3.4 — Baseline Decision Logic

**What this section is about:** Describes zone check, temporal threshold, and baseline output as one unit.

**What to write:**

**Zone membership check:** Point-in-polygon (ray casting) on bounding box centroid against zone polygon. Binary: inside or outside. Status recorded per frame per track.

**Temporal threshold check:** Stationarity timer begins when centroid enters zone AND displacement falls below threshold. Increments while both hold. Resets if vehicle exits or moves. Candidate violation flagged when timer exceeds T = 60 seconds. Cite the five Chapter 2 studies using this value.

**Baseline decision output:** Binary per event. Any vehicle exceeding T in the zone is flagged. No surrounding traffic information.

**Table — Baseline output format:**

| Output Field | Data Type | Description |
|---|---|---|
| Event ID | Integer | Unique identifier |
| Track ID | Integer | Which vehicle triggered the violation |
| Start frame | Integer | When vehicle became stationary in zone |
| Trigger frame | Integer | When threshold was exceeded |
| Decision | Binary | Violation (always, if threshold exceeded) |

**Length for all of 3.3:** 4–5 pages.

---

## 3.4 — Context-Aware Decision Logic Layer

**What this section is about:** Describes the proposed module — this is the study's contribution.

**What to write as the section opener (no subsection):**

One paragraph: the layer is inserted after the temporal threshold check. It receives candidate violation events and evaluates them against surrounding traffic behavior before confirming or suppressing. Uses displacement data already produced by the tracking module. No additional models, sensors, or external data.

---

### 3.4.1 — Displacement and Stationary Ratio Computation

**What this section is about:** The foundational computations the three checks depend on.

**What to write:**

**Displacement formula:**

> displacement(v, f) = √[(cx_f − cx_(f−N))² + (cy_f − cy_(f−N))²]

**Table — Motion classification:**

| Displacement | Classification |
|---|---|
| < δ pixels | Stationary |
| ≥ δ pixels | Moving |

State this is computed for ALL tracked vehicles, not only the candidate.

**Stationary ratio formula:**

> R = (count of stationary vehicles) / (total count of tracked vehicles)

**Table — Interpretation:**

| R Value | Interpretation |
|---|---|
| R = 0.0 | All vehicles moving |
| R = 0.5 | Half stationary, half moving |
| R = 1.0 | All vehicles stationary |

**Table — Parameters:**

| Parameter | Symbol | Planned Value | Rationale |
|---|---|---|---|
| Lookback window | N | 30 frames | At 10 fps = 3 seconds |
| Displacement threshold | δ | 5 pixels | Absorbs detection jitter |

**Edge cases:** Fewer than N frames of history → compute from first tracked frame. Only one vehicle tracked → R undefined, default to baseline behavior.

---

### 3.4.2 — Collective Stop Detection (Check 1)

**What this section is about:** First decision check — is traffic collectively stopped?

**What to write:**

**Table — Decision logic:**

| Condition | Action |
|---|---|
| R > τ | Suppress violation flag |
| R ≤ τ | Retain violation flag |

| Parameter | Symbol | Planned Value | Rationale |
|---|---|---|---|
| Congestion threshold | τ | 0.6 | Majority threshold for collective stop |

One paragraph on what it catches (red lights, congestion, incident stops). One sentence on its limitation: cannot distinguish a vehicle parked during a collective stop from one legitimately stopped. Reference Check 2.

---

### 3.4.3 — Collective Clearance Check (Check 2)

**What this section is about:** Monitors what happens when traffic clears — does the candidate move or stay?

**What to write:**

**Clearance event:** R transitions from above τ to below τ within W frames.

**Table — Candidate evaluation:**

| Candidate Behavior During Clearance | Action |
|---|---|
| Displacement > δ (moved with traffic) | Confirm as non-violation |
| Displacement ≤ δ (stayed) | Confirm as violation |

| Parameter | Symbol | Planned Value | Rationale |
|---|---|---|---|
| Clearance window | W | 50 frames | At 10 fps = 5 seconds |

One paragraph on why this is necessary: prevents genuine violations from being permanently suppressed by Check 1. If R stays above τ (prolonged congestion), suppression continues until clearance occurs.

---

### 3.4.4 — Isolation Verification (Check 3)

**What this section is about:** Confidence indicator — is the candidate the only stopped vehicle?

**What to write:**

State this is a confidence label, not a decision modifier.

**Table — Confidence labels:**

| Condition | Confidence Label |
|---|---|
| R very low (< 0.2), only candidate stationary | High confidence — isolated anomaly |
| R moderate (0.2–0.6), mixed traffic | Standard confidence |
| R above τ, clearance detected, candidate stayed | Confirmed via divergence |

One sentence: Check 1 asks "Is everyone stopped?" — Check 3 asks "Is only this vehicle stopped?"

---

### 3.4.5 — Combined Decision Flow

**What this section is about:** Shows the complete integrated logic and all outputs.

**What to write:**

**Figure 2 — Decision Flow Diagram**

This should be a proper decision flowchart (made in draw.io or similar). Must contain:

| Element | What to show |
|---|---|
| Entry point | "Candidate vehicle exceeds T seconds stationary in zone" |
| First computation | Compute Stationary Ratio R |
| First branch | R > τ (go to suppress path) vs R ≤ τ (go to confirm path) |
| Suppress path | Suppress flag → continue monitoring R → branch: R drops below τ (clearance) vs R stays above τ (remain suppressed) |
| Clearance sub-branch | Check candidate displacement → moved (non-violation) vs stayed (confirm violation) |
| Confirm path | Confirm violation with isolation label from Check 3 |
| Terminal states | Four distinct endpoints clearly labeled |

**Notes for groupmates:** This should be a clean, properly formatted flowchart with diamond shapes for decisions, rectangles for processes, and rounded rectangles for terminal states. Not ASCII art.

**Table — Terminal states:**

| Terminal State | Path | Meaning |
|---|---|---|
| Confirmed — isolated | R ≤ τ at threshold time | Vehicle stopped alone while traffic flows |
| Confirmed — post-clearance | R was > τ, dropped to ≤ τ, candidate stayed | Vehicle stayed when traffic cleared |
| Suppressed — collective stop | R > τ, no clearance yet | Vehicle likely stopped due to traffic |
| Non-violation — moved with traffic | R dropped to ≤ τ, candidate moved | Vehicle departed with traffic |

**Table — System output per event:**

| Output Field | Data Type | Description |
|---|---|---|
| Event ID | Integer | Unique identifier |
| Track ID | Integer | Which vehicle |
| Decision | Binary | Violation confirmed / Suppressed |
| Traffic state label | Categorical | Isolated, post-clearance, collective stop, moved with traffic |
| Stationary ratio at decision | Float | R value when decision was made |
| Confidence | Categorical | High, standard, or via divergence |

**Close the section** with the consolidated parameter summary (no separate subsection needed):

**Table — All parameters:**

| Parameter | Symbol | Planned Value | Component | Rationale |
|---|---|---|---|---|
| Temporal threshold | T | 60 seconds | Baseline + Proposed | Literature consensus from Chapter 2 |
| Lookback window | N | 30 frames | Displacement computation | At 10 fps = 3 seconds |
| Displacement threshold | δ | 5 pixels | Displacement computation | Absorbs detection jitter |
| Congestion threshold | τ | 0.6 | Check 1, Check 2 | Majority threshold for collective stop |
| Clearance window | W | 50 frames | Check 2 | At 10 fps = 5 seconds |

State that τ will be subject to sensitivity analysis in Section 3.5.6.

**Length for all of 3.4:** 7–9 pages.

---

## 3.5 — Evaluation Design

**What this section is about:** Defines every experiment, metric, and comparison.

---

### 3.5.1 — Experimental Configurations

**What this section is about:** Lays out the four experiments and the full configuration matrix.

**What to write:**

**Table — Four experiments:**

| Experiment | Purpose | Section |
|---|---|---|
| 1 — Pipeline Comparison | Evaluate the context-aware layer across all nine pipelines | 3.5.3 |
| 2 — Component Ablation | Isolate the contribution of each check within the layer | 3.5.4 |
| 3 — False Positive Source Analysis | Categorize which types of false positives the layer affects | 3.5.5 |
| 4 — Parameter Sensitivity | Assess the effect of τ on precision and recall | 3.5.6 |

State that the best-performing pipeline from Experiment 1 (highest F1 with context ON) will be used for Experiments 2–4. If multiple pipelines tie within 2 percentage points, select the fastest.

**Table — Full 18-configuration matrix:**

| Config ID | Detector | Tracker | Context Layer | Label |
|---|---|---|---|---|
| P1-Base | YOLOv8n | DeepSORT | OFF | Baseline |
| P1-Ctx | YOLOv8n | DeepSORT | ON | Proposed |
| P2-Base | YOLOv8n | ByteTrack | OFF | Baseline |
| P2-Ctx | YOLOv8n | ByteTrack | ON | Proposed |
| P3-Base | YOLOv8n | BoT-SORT | OFF | Baseline |
| P3-Ctx | YOLOv8n | BoT-SORT | ON | Proposed |
| P4-Base | YOLOv8s | DeepSORT | OFF | Baseline |
| P4-Ctx | YOLOv8s | DeepSORT | ON | Proposed |
| P5-Base | YOLOv8s | ByteTrack | OFF | Baseline |
| P5-Ctx | YOLOv8s | ByteTrack | ON | Proposed |
| P6-Base | YOLOv8s | BoT-SORT | OFF | Baseline |
| P6-Ctx | YOLOv8s | BoT-SORT | ON | Proposed |
| P7-Base | YOLOv8m | DeepSORT | OFF | Baseline |
| P7-Ctx | YOLOv8m | DeepSORT | ON | Proposed |
| P8-Base | YOLOv8m | ByteTrack | OFF | Baseline |
| P8-Ctx | YOLOv8m | ByteTrack | ON | Proposed |
| P9-Base | YOLOv8m | BoT-SORT | OFF | Baseline |
| P9-Ctx | YOLOv8m | BoT-SORT | ON | Proposed |

> "All 18 configurations will process the same video footage using the same ground truth annotations. Identical zone polygon definition and temporal threshold T = 60 seconds will be applied to all configurations."

---

### 3.5.2 — Evaluation Metrics

**What this section is about:** Defines every metric used across all four experiments.

**What to write:**

**Table — Classification outcomes:**

| Outcome | System Output | Ground Truth | Meaning |
|---|---|---|---|
| True Positive (TP) | Violation | Violation | Correctly detected real violation |
| False Positive (FP) | Violation | Non-violation | Incorrectly flagged legitimate stop |
| True Negative (TN) | Non-violation | Non-violation | Correctly ignored legitimate stop |
| False Negative (FN) | Non-violation | Violation | Missed real violation |

**Table — Violation detection metrics:**

| Metric | Formula | What It Measures |
|---|---|---|
| Precision | TP / (TP + FP) | Of flagged violations, proportion that are genuine |
| Recall | TP / (TP + FN) | Of genuine violations, proportion correctly flagged |
| F1 Score | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| False Positive Rate | FP / (FP + TN) | Rate of incorrect flags among non-violations |

State that precision is the primary metric. Recall monitored to verify genuine violations are not suppressed.

**Table — Pipeline characterization metrics (computed once per pipeline — context layer does not affect these):**

| Metric | What It Measures |
|---|---|
| Total unique track IDs generated | Track fragmentation / ID switching frequency |
| Average track duration (frames) | Tracking stability |
| Average FPS | Processing speed |

**Table — Context layer effect metrics (each value = the Ctx result minus the Base result for that pipeline, computed from Table 1 in Section 3.5.3):**

| Metric | Formula | What It Measures |
|---|---|---|
| ΔPrecision | P_Ctx Precision − P_Base Precision | How much the context layer changed precision for this pipeline. Positive = improved. Negative = worsened. |
| ΔRecall | P_Ctx Recall − P_Base Recall | How much the context layer changed recall for this pipeline. |
| ΔF1 | P_Ctx F1 − P_Base F1 | Net effect of the context layer for this pipeline. |

**Notes for groupmates:** The Δ metrics are not raw results. They are derived by subtracting. Example: if P1-Base Precision = 0.55 and P1-Ctx Precision = 0.82, then P1 ΔPrecision = +0.27. These go in Table 2 of Section 3.5.3.

---

### 3.5.3 — Experiment 1: Pipeline Comparison Study

**What this section is about:** The main experiment — tests all 9 pipelines × 2 context states.

**What to write:**

State the purpose: (1) identify which pipeline produces the best end-to-end performance with the context-aware layer, (2) determine whether the layer's effect is consistent across pipelines.

Present three empty results table templates:

**Table 1 — Raw violation detection metrics for all 18 configurations:**

(Each row = one experimental run. Two rows per pipeline — one with context OFF, one with context ON.)

| Config ID | Detector | Tracker | Context | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|
| P1-Base | YOLOv8n | DeepSORT | OFF | — | — | — | — |
| P1-Ctx | YOLOv8n | DeepSORT | ON | — | — | — | — |
| P2-Base | YOLOv8n | ByteTrack | OFF | — | — | — | — |
| P2-Ctx | YOLOv8n | ByteTrack | ON | — | — | — | — |
| P3-Base | YOLOv8n | BoT-SORT | OFF | — | — | — | — |
| P3-Ctx | YOLOv8n | BoT-SORT | ON | — | — | — | — |
| P4-Base | YOLOv8s | DeepSORT | OFF | — | — | — | — |
| P4-Ctx | YOLOv8s | DeepSORT | ON | — | — | — | — |
| P5-Base | YOLOv8s | ByteTrack | OFF | — | — | — | — |
| P5-Ctx | YOLOv8s | ByteTrack | ON | — | — | — | — |
| P6-Base | YOLOv8s | BoT-SORT | OFF | — | — | — | — |
| P6-Ctx | YOLOv8s | BoT-SORT | ON | — | — | — | — |
| P7-Base | YOLOv8m | DeepSORT | OFF | — | — | — | — |
| P7-Ctx | YOLOv8m | DeepSORT | ON | — | — | — | — |
| P8-Base | YOLOv8m | ByteTrack | OFF | — | — | — | — |
| P8-Ctx | YOLOv8m | ByteTrack | ON | — | — | — | — |
| P9-Base | YOLOv8m | BoT-SORT | OFF | — | — | — | — |
| P9-Ctx | YOLOv8m | BoT-SORT | ON | — | — | — | — |

**Table 2 — Context layer effect size per pipeline (derived from Table 1):**

(Each row = the subtraction between Ctx and Base for that pipeline. One row per pipeline. Positive = layer helped. Negative = layer hurt.)

| Pipeline | Detector | Tracker | ΔPrecision | ΔRecall | ΔF1 |
|---|---|---|---|---|---|
| P1 | YOLOv8n | DeepSORT | — | — | — |
| P2 | YOLOv8n | ByteTrack | — | — | — |
| P3 | YOLOv8n | BoT-SORT | — | — | — |
| P4 | YOLOv8s | DeepSORT | — | — | — |
| P5 | YOLOv8s | ByteTrack | — | — | — |
| P6 | YOLOv8s | BoT-SORT | — | — | — |
| P7 | YOLOv8m | DeepSORT | — | — | — |
| P8 | YOLOv8m | ByteTrack | — | — | — |
| P9 | YOLOv8m | BoT-SORT | — | — | — |

**Table 3 — Pipeline characterization (independent of context layer — same numbers whether ON or OFF):**

| Pipeline | Detector | Tracker | Total Track IDs | Avg Track Duration (frames) | Avg FPS |
|---|---|---|---|---|---|
| P1 | YOLOv8n | DeepSORT | — | — | — |
| P2 | YOLOv8n | ByteTrack | — | — | — |
| P3 | YOLOv8n | BoT-SORT | — | — | — |
| P4 | YOLOv8s | DeepSORT | — | — | — |
| P5 | YOLOv8s | ByteTrack | — | — | — |
| P6 | YOLOv8s | BoT-SORT | — | — | — |
| P7 | YOLOv8m | DeepSORT | — | — | — |
| P8 | YOLOv8m | ByteTrack | — | — | — |
| P9 | YOLOv8m | BoT-SORT | — | — | — |

**Five planned analyses** — state what is compared and what the comparison reveals:

1. **Detector effect** — hold tracker constant, compare across detectors (P1 vs P4 vs P7, P2 vs P5 vs P8, P3 vs P6 vs P9). Reveals how detection capacity affects violation detection and context layer effectiveness.

2. **Tracker effect** — hold detector constant, compare across trackers (P1 vs P2 vs P3, P4 vs P5 vs P6, P7 vs P8 vs P9). Reveals how tracking methodology affects the context layer.

3. **Context layer consistency** — compare ΔPrecision and ΔF1 across all nine pipelines. Reveals whether the layer is pipeline-agnostic or pipeline-dependent.

4. **Interaction effects** — examine whether detector and tracker quality interact in their effect on the context layer.

5. **Speed-accuracy tradeoff** — plot F1 (context ON) against average FPS for all nine pipelines. Identifies which pipeline balances accuracy and speed.

---

### 3.5.4 — Experiment 2: Component Ablation

**What this section is about:** Tests each check in the context-aware layer individually.

**What to write:**

State this will be conducted on the best-performing pipeline from Experiment 1.

**Table — Ablation configurations:**

| Config | Name | Check 1 | Check 2 | Check 3 |
|---|---|---|---|---|
| A | Baseline | OFF | OFF | OFF |
| B | Ratio only | ON | OFF | OFF |
| C | Ratio + Clearance | ON | ON | OFF |
| D | Full proposed | ON | ON | ON |

**Table — Comparisons:**

| Comparison | What It Reveals |
|---|---|
| B vs A | Effect of collective stop suppression alone |
| C vs B | Effect of adding the clearance check |
| D vs C | Effect of adding isolation labeling |

State all four evaluated with same metrics and ground truth.

---

### 3.5.5 — Experiment 3: False Positive Source Analysis

**What this section is about:** Categorizes what caused the false positives.

**What to write:**

State this will be conducted on the best-performing pipeline from Experiment 1.

**Table — FP categories:**

| FP Category | Definition | Example |
|---|---|---|
| Signal-related | Vehicle stopped at red light | Waiting at red, flagged after T seconds |
| Congestion-related | Vehicle in slow or queued traffic | Bumper-to-bumper |
| Loading-related | Vehicle temporarily stopped for loading | Delivery vehicle briefly in zone |
| Detection/tracking error | FP caused by system error | Ghost detection, ID switch |
| Other | Any cause not above | Yielding to pedestrian, etc. |

State that FP category distribution will be compared between baseline and proposed to identify which categories the context layer affects.

---

### 3.5.6 — Experiment 4: Parameter Sensitivity Analysis

**What this section is about:** Tests how the congestion threshold τ affects performance.

**What to write:**

State this will be conducted on the best-performing pipeline from Experiment 1.

**Table — τ values to test:**

| τ Value | Description |
|---|---|
| 0.3 | Very aggressive suppression |
| 0.4 | Aggressive |
| 0.5 | Moderate |
| 0.6 | Default |
| 0.7 | Conservative |
| 0.8 | Very conservative |

State that precision and recall will be computed and plotted for each τ value.

**Table — Secondary parameters (if time permits):**

| Parameter | Test Values |
|---|---|
| Displacement threshold δ | 3, 5, 7, 10 pixels |
| Lookback window N | 15, 30, 45 frames |

**Length for all of 3.5:** 7–9 pages.

---

## 3.6 — Scope and Limitations

**What this section is about:** States what the study covers and does not cover.

**What to write:**

No subsections. Write as a series of short paragraphs.

**Paragraph 1:** This study tests whether surrounding vehicle displacement data in the decision logic reduces false positives in fixed-camera illegal parking detection. It evaluates this across nine pipelines on footage from one location.

**Paragraph 2:** This study does not modify the detection or tracking stages. All models use pretrained weights and default configurations. Improving detection accuracy, reducing ID switches, or handling environmental challenges (rain, night, shadows, occlusion) are not addressed.

**Paragraph 3:** Evaluation is on footage from a single camera at one location. Generalization to other locations, cameras, and traffic patterns is future work.

**Paragraph 4:** Context-aware layer parameters are set to initial values. Sensitivity analysis is conducted for τ. Optimal values may differ across locations.

**Paragraph 5:** The stationary ratio requires at least two tracked vehicles. In single-vehicle scenes the system defaults to baseline behavior.

**Length for 3.6:** 1–1.5 pages.

---

## Complete Chapter 3 Section Flow

| Section | Title | Length |
|---|---|---|
| 3.1 | Conceptual Framework | 3–3.5 pages |
| 3.1.1 | System Architecture Overview | — |
| 3.1.2 | Contribution Positioning | — |
| 3.2 | Data Collection and Preparation | 4–5 pages |
| 3.2.1 | Dataset Requirements | — |
| 3.2.2 | Data Source | — |
| 3.2.3 | Ground Truth Annotation Protocol | — |
| 3.2.4 | Video Preprocessing | — |
| 3.3 | System Architecture | 4–5 pages |
| 3.3.1 | Vehicle Detection Module | — |
| 3.3.2 | Vehicle Tracking Module | — |
| 3.3.3 | Pipeline Combinations | — |
| 3.3.4 | Baseline Decision Logic | — |
| 3.4 | Context-Aware Decision Logic Layer | 7–9 pages |
| 3.4.1 | Displacement and Stationary Ratio Computation | — |
| 3.4.2 | Collective Stop Detection (Check 1) | — |
| 3.4.3 | Collective Clearance Check (Check 2) | — |
| 3.4.4 | Isolation Verification (Check 3) | — |
| 3.4.5 | Combined Decision Flow | — |
| 3.5 | Evaluation Design | 7–9 pages |
| 3.5.1 | Experimental Configurations | — |
| 3.5.2 | Evaluation Metrics | — |
| 3.5.3 | Experiment 1: Pipeline Comparison Study | — |
| 3.5.4 | Experiment 2: Component Ablation | — |
| 3.5.5 | Experiment 3: False Positive Source Analysis | — |
| 3.5.6 | Experiment 4: Parameter Sensitivity Analysis | — |
| 3.6 | Scope and Limitations | 1–1.5 pages |

**Figures to create:**
| Figure | What It Shows | Where It Goes |
|---|---|---|
| Figure 1 | System architecture — baseline and proposed paths with data flow | 3.1.1 |
| Figure 2 | Decision flow diagram for the context-aware layer | 3.4.5 |

**Total estimated length:** 27–33 pages.
