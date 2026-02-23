I now have the full context from the other chat. Let me provide the final Chapter 3 structure properly framed as a proposal.

---

# Chapter 3: Methodology — Complete Writing Guide

---

## 3.1 — Conceptual Framework

---

### 3.1.1 — Problem-to-Solution Mapping

**How to write it:**

One paragraph connecting Chapter 2 findings to what this study proposes. Do not re-argue Chapter 2. Just bridge.

> "Chapter 2 established that the dominant violation decision logic in fixed-camera illegal parking detection evaluates candidate vehicles independently using zone membership and temporal thresholds. Section 2.6.4 documented that three independent fixed-camera studies encountered false detections from non-parking stationary vehicles — those stopped at traffic signals, in congestion, or at intersections. Each study developed scene-specific responses: Gao et al. (2022) built a pedestrian zone proxy requiring visible crosswalks, Wahyono et al. (2017) applied temporal filtering that cannot distinguish prolonged congestion from parking, and Kathait et al. (2025) described signal color detection requiring visible signal heads. All existing responses evaluate the candidate vehicle against fixed external references. None examines whether surrounding tracked vehicles are also stationary."

> "This study proposes a context-aware decision logic layer that assesses surrounding vehicle displacement — data already produced by standard tracking modules — before confirming a violation."

---

### 3.1.2 — System Architecture Overview

**How to write it:**

Present the conceptual diagram. This is the single most important visual in the proposal. Show baseline and proposed pipelines.

**Baseline Pipeline Diagram:**

```
Fixed Camera Video
       ↓
Vehicle Detection (YOLOv8)
       ↓
Vehicle Tracking (DeepSORT)
       ↓
Zone Membership Check
(Is centroid inside restricted zone?)
       ↓
Temporal Threshold Check
(Stationary > 60 seconds?)
       ↓
OUTPUT: Violation / Non-Violation
```

**Proposed Pipeline Diagram:**

```
Fixed Camera Video
       ↓
Vehicle Detection (YOLOv8)
       ↓
Vehicle Tracking (DeepSORT)
       ↓
Zone Membership Check
       ↓
Temporal Threshold Check
       ↓
Context-Aware Decision Logic Layer
       ↓
┌─────────────────────────────────────────┐
│  Surrounding Vehicle State Assessment   │
│                                         │
│  1. Compute displacement of ALL         │
│     tracked vehicles                    │
│                                         │
│  2. Compute stationary ratio            │
│     (proportion currently stopped)      │
│                                         │
│  3. Compare candidate behavior against  │
│     surrounding traffic behavior        │
│                                         │
│  4. Confirm or suppress violation flag  │
│     based on comparison                 │
└─────────────────────────────────────────┘
       ↓
OUTPUT: Violation Confirmed / Suppressed
        + Traffic State Label
```

Below the diagrams, explain in plain terms:

> "The baseline system will flag any vehicle that exceeds the time threshold in the restricted zone regardless of surrounding traffic behavior. The proposed system will add one step: before confirming the violation, it will check whether other tracked vehicles in the scene are also stationary. If they are, the candidate's stationarity is likely part of a collective traffic event. If the candidate is the only stationary vehicle while others move, the behavior is consistent with individual parking."

---

### 3.1.3 — Contribution Positioning

**How to write it:**

Map the proposed contribution against Chapter 2 findings.

> "The proposed layer addresses the challenge documented in Section 2.6.4 through a different mechanism than existing approaches."

Present a comparison table:

| Existing Approach | What It Requires | What It Covers | What It Does Not Cover |
|---|---|---|---|
| Pedestrian zone proxy (Gao et al., 2022) | Visible crosswalk, pedestrian detection, two additional zones | Signal-related stops at intersections with crosswalks | Congestion, queue spillback, unsignalized locations, periods without pedestrians |
| Temporal filter (Wahyono et al., 2017) | Nothing additional | Stops shorter than 60 seconds | Congestion or parking longer than 60 seconds |
| Signal color detection (Kathait et al., 2025) | Visible traffic signal head | Signal-related stops where signal is visible | Locations without visible signals, congestion, queue spillback |
| **Proposed: Surrounding vehicle displacement** | Tracking output already produced | Any collective stopping regardless of cause | — |

> "The proposed approach will use displacement data that the tracking module already computes for every tracked vehicle in every frame. It requires no additional detection modules, no scene-specific calibration beyond the standard zone polygon, and no external data sources."

---

### 3.1.4 — Controlled Comparison Design

**How to write it:**

State what is shared and what differs between configurations.

> "Both system configurations will use identical detection and tracking components. The same YOLOv8 weights, DeepSORT parameters, zone polygon definition, and temporal threshold will be applied to both. The only variable will be the presence or absence of the context-aware decision logic layer. This controlled design ensures that any observed performance difference is attributable to the decision logic, not to detection or tracking variations."

**Length for all of 3.1:** 2.5-3 pages including diagrams and table.

---

## 3.2 — Data Collection and Preparation

---

### 3.2.1 — Dataset Requirements

**How to write it:**

State what properties the video footage must have and why each is necessary.

> "The evaluation will require fixed-camera video footage satisfying the following conditions."

**Requirement 1 — Static camera position:**

> "The camera must be mounted at a fixed position with a static viewing angle throughout the recording duration. The camera view must include a roadway segment containing a designated restricted parking zone. This ensures consistent spatial reference across all frames and enables a single zone polygon definition to apply throughout."

**Requirement 2 — Mixed traffic scenarios:**

> "The footage must capture three types of scenarios within the same recording session:"

| Scenario Type | What It Contains | Why It Is Needed |
|---|---|---|
| Free-flowing traffic | Vehicles passing through zone without stopping | Confirms both systems correctly produce no violations when no vehicles are parked |
| Collective stops | Multiple vehicles stationary simultaneously (red light, congestion) | Tests whether proposed system suppresses false positives that baseline produces |
| Genuine violations | Vehicles actually parked illegally in the zone | Confirms proposed system retains detection of real violations |

> "The presence of all three scenarios is essential. The baseline system is expected to handle free-flowing traffic and genuine violations correctly but produce false positives during collective stops. The proposed system is expected to suppress false positives during collective stops while retaining detection of genuine violations."

**Requirement 3 — Observable traffic signal effects:**

> "A signalized intersection must be visible or its effects observable within the camera frame. Vehicles stopping at red lights and resuming at green lights provide the primary test case for the context-aware layer. This scenario produces the clearest contrast between collective stationarity (all vehicles stopped at red) and individual parking (one vehicle remains while others depart at green)."

**Requirement 4 — Multiple simultaneous vehicles:**

> "The camera view must regularly contain multiple vehicles simultaneously. The stationary ratio computation requires observing surrounding vehicles. Footage showing only one vehicle at a time would render the surrounding vehicle check meaningless."

**Requirement 5 — Visible zone boundaries:**

> "Lane boundaries, curb edges, or road markings must be visible in the footage to enable accurate definition of the restricted zone polygon."

---

### 3.2.2 — Data Source

**How to write it:**

State where the footage will come from. Choose the appropriate framing based on your actual plan.

**If recording original footage:**

> "Video footage will be recorded at [specific location description] using a fixed camera mounted at approximately [height] meters above ground level. The camera will capture [resolution] video at [frame rate] frames per second."

> "The recording site was selected because it satisfies all stated requirements: it contains a designated no-parking zone adjacent to a signalized intersection, ensuring that both genuine violations and signal-related stops will occur within the same camera frame. The intersection experiences regular traffic signal cycles and varying traffic density throughout the day."

> "Recording will be conducted across multiple time periods:"

| Time Period | Expected Traffic Condition | Purpose |
|---|---|---|
| Morning (7:00-9:00) | Moderate to heavy, frequent signal stops | Capture collective stop scenarios |
| Midday (11:00-13:00) | Light to moderate | Capture free-flow and isolated violation scenarios |
| Afternoon (16:00-18:00) | Heavy, potential congestion | Capture congestion scenarios |

**If using existing public footage:**

> "Video footage will be obtained from [specific source — name dataset, repository, or URL]. The selected footage was verified to meet all requirements stated in Section 3.2.1: fixed camera angle, restricted zone visible, signalized intersection effects observable, multiple vehicles present simultaneously, and varying traffic conditions across the recording duration."

> "The footage contains approximately [duration] of continuous recording from a single camera position, including [describe what traffic scenarios are present]."

---

### 3.2.3 — Ground Truth Annotation Protocol

**How to write it:**

Define exactly how events will be identified and labeled.

**Event definition:**

> "A stationary vehicle event is defined as a continuous period during which a tracked vehicle remains within the restricted zone with displacement below a movement threshold. An event begins when the vehicle first becomes stationary in the zone and ends when the vehicle departs the zone or resumes movement."

**Annotation labels:**

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

**Annotation procedure:**

> "Each event will be annotated by reviewing the video segment containing the event. The annotator will observe:"
> - The candidate vehicle's behavior (when it stopped, how long it remained, when/if it departed)
> - The surrounding traffic behavior (are other vehicles also stopped, are they moving)
> - Any visible contextual cues (traffic signal state if visible, pedestrian activity, queue formation)

> "Based on these observations, the annotator will assign the ground truth status. Events where the vehicle is clearly waiting in traffic with surrounding vehicles also stopped will be labeled as non-violations with the corresponding stop reason. Events where the vehicle remains stationary while surrounding traffic flows, in a manner consistent with parking, will be labeled as violations."

**Inter-annotator reliability:**

> "A minimum of 20% of annotated events will be independently labeled by a second annotator. Inter-annotator agreement will be measured using Cohen's kappa coefficient. Events with disagreement will be resolved through joint review of the video segment by both annotators."

---

### 3.2.4 — Video Preprocessing

**How to write it:**

Describe technical preparation steps that will be performed before running experiments.

**Frame extraction:**

> "Video will be processed at [X] frames per second. If the source video frame rate exceeds 30 fps, frames will be sampled at 10 fps to reduce computational load while retaining sufficient temporal resolution for displacement computation. The selected frame rate will be held constant across all experimental configurations."

**Zone polygon definition:**

> "The restricted zone will be defined by selecting vertex coordinates on a reference frame from the video. The polygon will be drawn to match the physical boundaries of the no-parking zone as visible from the camera angle. Vertex coordinates will be recorded and the same polygon definition will be applied to both baseline and proposed system configurations."

Present an example zone definition table:

| Vertex | X Coordinate | Y Coordinate |
|---|---|---|
| V1 | [value] | [value] |
| V2 | [value] | [value] |
| V3 | [value] | [value] |
| V4 | [value] | [value] |

**Detection and tracking parameters:**

> "YOLOv8 detection confidence threshold, non-maximum suppression threshold, and class filter settings will be determined through preliminary testing on a small sample of the footage. DeepSORT matching thresholds, maximum age, and minimum hits parameters will similarly be set through preliminary testing. Once set, these parameters will be held constant across all experimental configurations and will not be modified during evaluation."

**Length for all of 3.2:** 4-5 pages including tables.

---

## 3.3 — Baseline System Architecture

---

### 3.3.1 — Vehicle Detection Module

**How to write it:**

Describe what the detection module will do.

> "Vehicles will be detected frame-by-frame using YOLOv8 pretrained on the COCO dataset. The detector will output bounding box coordinates (x, y, width, height), class label, and confidence score for each detected object."

**Detection filtering:**

> "Only detections belonging to vehicle classes will be retained:"

| COCO Class ID | Class Name | Included |
|---|---|---|
| 2 | Car | Yes |
| 3 | Motorcycle | Yes |
| 5 | Bus | Yes |
| 7 | Truck | Yes |
| Other | — | No |

> "Detections with confidence scores below the threshold (to be determined in preliminary testing, expected range 0.4-0.6) will be discarded."

**Detection output:**

> "For each frame, the detection module will output a list of detections, each containing bounding box coordinates, class label, and confidence score. This output will be passed to the tracking module."

---

### 3.3.2 — Vehicle Tracking Module

**How to write it:**

Describe what the tracking module will do.

> "Detected vehicles will be tracked across frames using DeepSORT. The tracker will assign a persistent unique identifier (track ID) to each vehicle and maintain identity across consecutive frames using a combination of Kalman Filter motion prediction and deep appearance feature matching."

**Tracking output:**

> "For each tracked vehicle in each frame, the tracking module will output:"

| Output | Data Type | Description |
|---|---|---|
| Track ID | Integer | Persistent identifier for this vehicle |
| Bounding box | (x, y, w, h) | Current bounding box coordinates |
| Centroid | (cx, cy) | Center point of bounding box |
| Track age | Integer | Number of frames this track has existed |
| Track state | Categorical | Confirmed, tentative, or deleted |

> "Only confirmed tracks will be passed to subsequent processing stages."

---

### 3.3.3 — Zone Membership Check

**How to write it:**

> "For each confirmed track in each frame, the system will test whether the bounding box centroid falls within the predefined restricted zone polygon."

> "A standard point-in-polygon algorithm (ray casting) will be used. The test returns a binary result: inside or outside. Vehicles whose centroids are outside the polygon will be excluded from violation analysis for that frame."

> "The zone membership status of each tracked vehicle will be recorded per frame, enabling computation of how long each vehicle has been continuously inside the zone."

---

### 3.3.4 — Temporal Threshold Check

**How to write it:**

> "For each tracked vehicle, a stationarity timer will track how long the vehicle has been continuously stationary within the restricted zone."

**Timer logic:**

> "The timer will begin when a tracked vehicle's centroid first enters the restricted zone AND the vehicle's frame-to-frame displacement falls below a movement threshold. The timer will increment each frame while both conditions remain true. The timer will reset to zero if the vehicle exits the zone or if displacement exceeds the movement threshold (indicating the vehicle moved)."

**Violation threshold:**

> "If the timer exceeds T = 60 seconds, the vehicle will be classified as a candidate violation. This threshold value is adopted from the majority of fixed-camera studies reviewed in Chapter 2, including Wahyono et al. (2017), Wahyono/ViBe (2022), Akhawaji et al. (2017), Ng et al. (2018), and Kakad et al. (2024)."

---

### 3.3.5 — Baseline Decision Output

**How to write it:**

> "The baseline system will produce a binary output per candidate event: violation or non-violation."

> "Any vehicle that exceeds the temporal threshold T while inside the restricted zone will be flagged as a violation. No information about surrounding traffic behavior will be considered. The baseline system evaluates each candidate vehicle entirely in isolation."

**Output format:**

| Output Field | Data Type | Description |
|---|---|---|
| Event ID | Integer | Unique identifier for this violation event |
| Track ID | Integer | Which tracked vehicle triggered the violation |
| Start frame | Integer | When vehicle first became stationary in zone |
| Trigger frame | Integer | When threshold was exceeded |
| Decision | Binary | Violation (always, if threshold exceeded) |

**Length for all of 3.3:** 2.5-3 pages including tables.

---

## 3.4 — Context-Aware Decision Logic Layer

**How to write it:**

Open with the positioning statement:

> "The proposed context-aware decision logic layer will be inserted after the temporal threshold check in the baseline pipeline. It will receive candidate violation events — vehicles that have already been identified as stationary in the restricted zone beyond T seconds — and evaluate them against surrounding traffic behavior before confirming or suppressing the violation flag."

> "The layer will use displacement data computed from bounding box centroid positions already produced by the tracking module. No additional detection models, sensors, or external data sources will be required."

---

### 3.4.1 — Vehicle Displacement Computation

**How to write it:**

Describe the foundational computation that all subsequent checks will use.

> "For every tracked vehicle visible in the current frame, displacement will be computed as the Euclidean distance between the vehicle's bounding box centroid in the current frame and its centroid N frames prior."

**Formula:**

> displacement(v, f) = √[(cx_f - cx_(f-N))² + (cy_f - cy_(f-N))²]

> Where:
> - v = vehicle track ID
> - f = current frame number
> - cx, cy = centroid coordinates
> - N = lookback window in frames

**Motion classification:**

> "Each tracked vehicle will be classified as stationary or moving based on its displacement:"

| Displacement | Classification |
|---|---|
| < δ pixels | Stationary |
| ≥ δ pixels | Moving |

> "This classification will be computed for ALL tracked vehicles in the scene, not only the candidate violation vehicle."

**Parameters:**

| Parameter | Symbol | Planned Value | Rationale |
|---|---|---|---|
| Lookback window | N | 30 frames | At 10 fps, covers 3 seconds — sufficient to distinguish genuinely stopped vehicles from slow-moving ones |
| Displacement threshold | δ | 5 pixels | Absorbs minor bounding box jitter from detection noise without masking genuine slow movement |

**Edge cases:**

> "If a vehicle has been tracked for fewer than N frames, displacement will be computed from its first tracked frame to the current frame. If a vehicle has only one frame of tracking data, it will be excluded from motion classification for that frame."

**Output:**

> "This computation will produce a per-frame snapshot of which vehicles are moving and which are stationary across the entire visible scene."

---

### 3.4.2 — Stationary Ratio Computation

**How to write it:**

Define the core metric.

> "The stationary ratio R will be computed as the proportion of all tracked vehicles in the current frame that are classified as stationary."

**Formula:**

> R = (count of stationary vehicles) / (total count of tracked vehicles)

**Interpretation:**

| R Value | Interpretation |
|---|---|
| R = 0.0 | All vehicles moving — normal traffic flow |
| R = 0.5 | Half stationary, half moving — mixed traffic |
| R = 1.0 | All vehicles stationary — collective stop (likely red light or congestion) |

> "A high R value indicates collective stopping behavior. A low R value with the candidate vehicle stationary indicates the candidate is behaving differently from surrounding traffic."

**Temporal tracking:**

> "The stationary ratio will be computed per frame. For decision purposes, the system will use the ratio at the frame when the candidate vehicle first exceeded the temporal threshold T, and will continue monitoring the ratio while the candidate remains in the zone."

**Edge case:**

> "If only one vehicle is tracked in the frame (the candidate itself), the stationary ratio is undefined. In this case, the system will default to baseline behavior — confirming the violation based on temporal threshold alone. The surrounding vehicle check requires at least two tracked vehicles to function."

---

### 3.4.3 — Collective Stop Detection (Check 1)

**How to write it:**

Describe the first decision check.

> "When a candidate vehicle exceeds the temporal threshold T in the restricted zone, the system will evaluate the stationary ratio R at that moment."

**Decision logic:**

| Condition | Interpretation | Action |
|---|---|---|
| R > τ | Majority of traffic is stopped | Suppress violation flag (candidate likely part of collective stop) |
| R ≤ τ | Traffic is flowing | Retain violation flag (candidate stopped individually) |

**Congestion threshold parameter:**

| Parameter | Symbol | Planned Value | Rationale |
|---|---|---|---|
| Congestion threshold | τ | 0.6 | If more than 60% of visible vehicles are stationary, collective stopping is the likely explanation |

**What this check will catch:**

> "This check will address scenarios where traffic is broadly stopped: red light phases affecting the entire intersection approach, congestion extending through the camera view, or incident-related stops. In these conditions, every vehicle in the restricted zone would exceed the temporal threshold under the baseline system, potentially producing multiple simultaneous false positives. The collective stop detection will suppress all of them because surrounding traffic is also stationary."

**What this check will not catch:**

> "This check alone cannot distinguish between a vehicle that is temporarily stopped with traffic and a vehicle that is genuinely parked during a collective stop. If a vehicle parks during a red light phase, this check would suppress it along with legitimately stopped vehicles. This limitation is addressed by the Collective Clearance Check described in Section 3.4.4."

---

### 3.4.4 — Collective Clearance Check (Check 2)

**How to write it:**

Describe the second check.

> "The collective clearance check will monitor transitions in the stationary ratio over time. When surrounding traffic resumes movement — such as when a traffic signal turns green — the stationary ratio will drop. The system will check whether the candidate vehicle also resumed movement."

**Clearance event detection:**

> "A collective clearance event will be detected when the stationary ratio R transitions from above τ to below τ within W consecutive frames."

| Frame | R Value | Event |
|---|---|---|
| f | R > τ | Traffic collectively stopped |
| f+1 to f+W | R drops below τ | Clearance event detected — traffic resuming movement |

**Candidate behavior evaluation:**

> "Upon detecting a clearance event, the system will check the candidate vehicle's displacement during the same window."

| Candidate Behavior | Interpretation | Action |
|---|---|---|
| Candidate displacement > δ during clearance window | Candidate moved with traffic | Confirm as non-violation (candidate was part of collective stop) |
| Candidate displacement ≤ δ during clearance window | Candidate did not move when traffic cleared | Confirm as violation (candidate stayed while others departed) |

**Parameter:**

| Parameter | Symbol | Planned Value | Rationale |
|---|---|---|---|
| Clearance observation window | W | 50 frames | At 10 fps, covers 5 seconds — sufficient to observe a signal cycle transition |

**Why this check is necessary:**

> "This check will prevent genuine violations from being permanently suppressed by Check 1. A vehicle may begin parking during a red light phase when surrounding traffic is also stopped. Without Check 2, Check 1 would suppress this genuine violation indefinitely. Check 2 catches it by detecting the moment when traffic clears: if the candidate does not move with traffic, the violation flag is reinstated."

**Continuous monitoring:**

> "If the stationary ratio remains above τ continuously (prolonged congestion with no clearing), the violation flag will remain suppressed. The system will re-evaluate each frame. When congestion eventually clears and the candidate remains, Check 2 will catch it at that point."

---

### 3.4.5 — Isolation Verification (Check 3)

**How to write it:**

Describe the third check.

> "The isolation verification will examine whether the candidate vehicle is the only stationary vehicle in the scene while surrounding vehicles are moving."

**Logic:**

> "When the stationary ratio R is below τ and the candidate vehicle has exceeded the temporal threshold T, the candidate is stationary while the majority of traffic moves. The system will flag this as an isolated stationary vehicle."

**Confidence classification:**

> "This check will function as a confidence indicator rather than a decision modifier. When R is below τ, the candidate will be confirmed as a violation regardless of isolation status (since Check 1 only suppresses when R > τ). However, an isolated stationary vehicle — one stopped alone while all others move — represents the highest-confidence indicator of individual parking behavior."

**Output label:**

| Condition | Confidence Label |
|---|---|
| R very low (e.g., < 0.2) and only candidate stationary | High confidence — isolated anomaly |
| R moderate (e.g., 0.2-0.6) with mixed traffic | Standard confidence |
| R above τ, clearance detected, candidate stayed | Confirmed via divergence |

**Relationship to Check 1:**

> "Check 1 asks: 'Is everyone stopped?' and suppresses if yes. Check 3 asks: 'Is only this vehicle stopped?' and assigns high confidence if yes. Together they cover both ends of the spectrum."

---

### 3.4.6 — Combined Decision Flow

**How to write it:**

Present the complete integrated logic.

**Flow diagram:**

```
Candidate vehicle exceeds T seconds stationary in zone
                         ↓
              Compute Stationary Ratio R
                         ↓
                ┌────────┴────────┐
                │                 │
            R > τ              R ≤ τ
        (collective         (traffic 
          stop)              flowing)
                │                 │
                ↓                 ↓
        SUPPRESS flag      CONFIRM violation
        (Check 1)          (with isolation
                │           label from
                │           Check 3)
                ↓
        Continue monitoring R
                │
        ┌───────┴───────┐
        │               │
    R drops to      R stays
    below τ         above τ
    (clearance)     (still stopped)
        │               │
        ↓               ↓
    Check candidate   Remain suppressed
    displacement      (re-evaluate
        │              next frame)
    ┌───┴───┐
    │       │
  Moved   Stayed
    │       │
    ↓       ↓
  NON-    CONFIRM
  VIOL.   violation
          (Check 2)
```

**Terminal states:**

| Terminal State | Path | Meaning |
|---|---|---|
| Confirmed — isolated | R ≤ τ at threshold time | Vehicle stopped alone while traffic flows. High confidence. |
| Confirmed — post-clearance | R was > τ, dropped to ≤ τ, candidate stayed | Vehicle stayed when traffic cleared. Genuine violation detected via divergence. |
| Suppressed — collective stop | R > τ, no clearance yet | Vehicle likely stopped due to traffic. Suppressed pending clearance. |
| Non-violation — moved with traffic | R dropped to ≤ τ, candidate moved | Vehicle was part of collective stop and departed with traffic. Correctly filtered. |

**System output:**

> "For each candidate event, the system will output:"

| Output Field | Data Type | Description |
|---|---|---|
| Event ID | Integer | Unique identifier |
| Track ID | Integer | Which vehicle |
| Decision | Binary | Violation confirmed / Suppressed |
| Traffic state label | Categorical | Isolated, post-clearance, collective stop, moved with traffic |
| Stationary ratio at decision | Float | R value when decision was made |
| Confidence | Categorical | High, standard, or via divergence |

---

### 3.4.7 — Parameter Summary

**How to write it:**

Consolidate all planned parameters in one reference table.

| Parameter | Symbol | Planned Value | Component | Rationale |
|---|---|---|---|---|
| Temporal threshold | T | 60 seconds | Baseline + Proposed | Literature consensus from Chapter 2 |
| Lookback window | N | 30 frames | Displacement computation | At 10 fps = 3 seconds observation window |
| Displacement threshold | δ | 5 pixels | Displacement computation | Absorbs detection jitter |
| Congestion threshold | τ | 0.6 | Check 1, Check 2 | Majority threshold for collective stop |
| Clearance window | W | 50 frames | Check 2 | At 10 fps = 5 seconds for clearance observation |

> "Parameters marked as 'Literature consensus' are drawn from values used by the majority of reviewed studies. Parameters marked with rationale are initial values based on reasoning about typical traffic behavior. These parameters will be subject to sensitivity analysis as described in Section 3.5.5."

**Length for all of 3.4:** 7-9 pages including diagrams and tables.

---

## 3.5 — Evaluation Design

---

### 3.5.1 — Experimental Configurations

**How to write it:**

Define exactly what will be compared.

> "Two primary system configurations will be evaluated on identical test footage using identical ground truth annotations."

| Configuration | Detection | Tracking | Zone Check | Threshold | Context Layer |
|---|---|---|---|---|---|
| Baseline | YOLOv8 | DeepSORT | Same polygon | 60 seconds | OFF |
| Proposed | YOLOv8 | DeepSORT | Same polygon | 60 seconds | ON |

> "Both configurations will process the same video frames in the same order using identical model weights, parameters, and zone definitions. The only variable will be the presence or absence of the context-aware decision logic layer."

> "This controlled design ensures that any observed performance difference is attributable to the decision logic modification, not to variations in detection, tracking, or zone definition."

---

### 3.5.2 — Evaluation Metrics

**How to write it:**

Define each metric precisely in the context of this study.

**Classification outcomes:**

| Outcome | System Output | Ground Truth | Meaning |
|---|---|---|---|
| True Positive (TP) | Violation | Violation | Correctly detected real violation |
| False Positive (FP) | Violation | Non-violation | Incorrectly flagged legitimate stop |
| True Negative (TN) | Non-violation | Non-violation | Correctly ignored legitimate stop |
| False Negative (FN) | Non-violation | Violation | Missed real violation |

**Metrics:**

| Metric | Formula | What It Measures | Expected Direction |
|---|---|---|---|
| Precision | TP / (TP + FP) | Of flagged violations, proportion that are genuine | Proposed > Baseline |
| Recall | TP / (TP + FN) | Of genuine violations, proportion correctly flagged | Proposed ≈ Baseline |
| F1 Score | 2 × P × R / (P + R) | Harmonic mean of precision and recall | Proposed ≥ Baseline |
| False Positive Rate | FP / (FP + TN) | Rate of incorrect flags among non-violations | Proposed < Baseline |

**Primary metric justification:**

> "Precision is the primary evaluation metric. The proposed system is designed to reduce false positives by suppressing candidates during collective traffic stops. An increase in precision indicates that the layer correctly filters traffic-related false positives. Recall will be monitored to ensure that the layer does not suppress genuine violations. The ideal outcome is improved precision with minimal or no decrease in recall."

---

### 3.5.3 — False Positive Source Analysis

**How to write it:**

> "False positives from both system configurations will be categorized by their cause using ground truth annotations."

| FP Category | Definition | Example |
|---|---|---|
| Signal-related | Vehicle was stopped at a red light | Vehicle waiting at red, flagged after 60s |
| Congestion-related | Vehicle was in slow or queued traffic | Vehicle in bumper-to-bumper traffic |
| Loading-related | Vehicle temporarily stopped for loading | Delivery vehicle briefly in zone |
| Detection/tracking error | FP caused by system error | Ghost detection, ID switch |
| Other | Any cause not above | Yielding to pedestrian, etc. |

> "The distribution of false positive sources will be compared between baseline and proposed configurations."

**Expected outcome:**

| FP Category | Baseline | Proposed (Expected) |
|---|---|---|
| Signal-related | Present | Reduced or eliminated |
| Congestion-related | Present | Reduced or eliminated |
| Loading-related | Present | Possibly reduced (if others also stopped) |
| Detection/tracking error | Present | Unchanged (layer does not modify detection/tracking) |

> "This analysis will demonstrate whether the context-aware layer specifically addresses traffic-related false positives as intended, while leaving other error sources unaffected."

---

### 3.5.4 — Ablation Study Design

**How to write it:**

> "To isolate the contribution of each component within the context-aware layer, the following incremental configurations will be tested."

| Config | Name | Check 1 (Stationary Ratio) | Check 2 (Clearance) | Check 3 (Isolation) |
|---|---|---|---|---|
| A | Baseline | OFF | OFF | OFF |
| B | Ratio only | ON | OFF | OFF |
| C | Ratio + Clearance | ON | ON | OFF |
| D | Full proposed | ON | ON | ON |

**Analysis:**

| Comparison | What It Shows |
|---|---|
| B vs A | Does suppressing during collective stops improve precision? |
| C vs B | Does clearance check recover genuine violations, improving recall relative to B? |
| D vs C | Does isolation labeling add value? |

> "Each configuration will be evaluated using the same metrics and ground truth. Results will show whether each component contributes measurable improvement."

---

### 3.5.5 — Parameter Sensitivity Analysis

**How to write it:**

> "The congestion threshold τ is the primary tunable parameter controlling how aggressively the system suppresses candidates. To assess its effect, the full proposed system will be evaluated at multiple τ values."

| τ Value | Expected Behavior |
|---|---|
| 0.3 | Very aggressive — suppresses when only 30% stopped. Risk: may suppress genuine violations. |
| 0.4 | Aggressive |
| 0.5 | Moderate — suppresses when half stopped |
| 0.6 | Default — suppresses when clear majority stopped |
| 0.7 | Conservative — requires 70% stopped |
| 0.8 | Very conservative — suppresses only during near-total stops |

**Planned analysis:**

> "Precision and recall will be computed for each τ value and plotted. The plot will reveal the precision-recall tradeoff: lower τ values are expected to increase precision (more suppression) but may decrease recall (genuine violations also suppressed). Higher τ values will preserve recall but provide less false positive reduction."

> "The optimal τ value will be identified as the point that maximizes precision improvement while maintaining acceptable recall."

**Secondary parameters:**

> "If evaluation time permits, sensitivity analysis will also be conducted for:"

| Parameter | Test Values | Purpose |
|---|---|---|
| Displacement threshold δ | 3, 5, 7, 10 pixels | Assess effect of motion sensitivity |
| Lookback window N | 15, 30, 45 frames | Assess effect of observation duration |

**Length for all of 3.5:** 4-5 pages including tables.

---

## 3.6 — Scope and Limitations

**How to write it:**

State methodological boundaries honestly.

---

### 3.6.1 — Parameter Selection

> "The congestion threshold τ, displacement threshold δ, lookback window N, and clearance window W are set to initial values based on reasoning about typical traffic behavior. While sensitivity analysis will be conducted for τ, systematic optimization of all parameters across diverse scenes is beyond the scope of this study. Optimal parameter values may differ across camera angles, resolutions, traffic patterns, and geographic contexts."

---

### 3.6.2 — Edge Case: Simultaneous Independent Parking

> "The stationary ratio assumes that a high proportion of stationary vehicles indicates collective traffic behavior. In a rare scenario where multiple vehicles independently and simultaneously park illegally in the zone, the ratio would be high and the system may incorrectly suppress genuine violations. This edge case is acknowledged but is expected to be rare in practice."

---

### 3.6.3 — Bounding Box Centroid Jitter

> "Displacement is computed from bounding box centroids, which may exhibit frame-to-frame jitter from detection noise even for truly stationary vehicles. The displacement threshold δ is set to accommodate typical jitter levels, but optimal values may vary across camera resolutions, viewing angles, and detector configurations."

---

### 3.6.4 — Single Dataset Evaluation

> "Evaluation will be conducted on footage from a single camera position at one location. Generalization to other viewing angles, intersection geometries, traffic patterns, and geographic contexts will require further validation with additional datasets. This study establishes feasibility and measures effect size at one location; broader generalization is identified as future work."

---

### 3.6.5 — Tracker Dependency

> "The context-aware layer depends on the tracking module maintaining correct vehicle identities for both the candidate and surrounding vehicles. If the tracker produces significant ID switching or track fragmentation, the stationary ratio computation may be unreliable. Tracker performance is not modified in this study; the layer is evaluated using DeepSORT with default parameters."

---

### 3.6.6 — Minimum Vehicle Count

> "The surrounding vehicle check requires at least two tracked vehicles to compute a meaningful stationary ratio. In scenarios with only one visible vehicle, the system will default to baseline behavior. The effectiveness of the proposed layer is therefore limited to multi-vehicle scenes."

**Length for all of 3.6:** 1.5-2 pages.

---

## Complete Chapter 3 Section Flow

| Section | Title | Length |
|---|---|---|
| 3.1 | Conceptual Framework | 2.5-3 pages |
| 3.1.1 | Problem-to-Solution Mapping | — |
| 3.1.2 | System Architecture Overview | — |
| 3.1.3 | Contribution Positioning | — |
| 3.1.4 | Controlled Comparison Design | — |
| 3.2 | Data Collection and Preparation | 4-5 pages |
| 3.2.1 | Dataset Requirements | — |
| 3.2.2 | Data Source | — |
| 3.2.3 | Ground Truth Annotation Protocol | — |
| 3.2.4 | Video Preprocessing | — |
| 3.3 | Baseline System Architecture | 2.5-3 pages |
| 3.3.1 | Vehicle Detection Module | — |
| 3.3.2 | Vehicle Tracking Module | — |
| 3.3.3 | Zone Membership Check | — |
| 3.3.4 | Temporal Threshold Check | — |
| 3.3.5 | Baseline Decision Output | — |
| 3.4 | Context-Aware Decision Logic Layer | 7-9 pages |
| 3.4.1 | Vehicle Displacement Computation | — |
| 3.4.2 | Stationary Ratio Computation | — |
| 3.4.3 | Collective Stop Detection (Check 1) | — |
| 3.4.4 | Collective Clearance Check (Check 2) | — |
| 3.4.5 | Isolation Verification (Check 3) | — |
| 3.4.6 | Combined Decision Flow | — |
| 3.4.7 | Parameter Summary | — |
| 3.5 | Evaluation Design | 4-5 pages |
| 3.5.1 | Experimental Configurations | — |
| 3.5.2 | Evaluation Metrics | — |
| 3.5.3 | False Positive Source Analysis | — |
| 3.5.4 | Ablation Study Design | — |
| 3.5.5 | Parameter Sensitivity Analysis | — |
| 3.6 | Scope and Limitations | 1.5-2 pages |
| 3.6.1 | Parameter Selection | — |
| 3.6.2 | Edge Case: Simultaneous Independent Parking | — |
| 3.6.3 | Bounding Box Centroid Jitter | — |
| 3.6.4 | Single Dataset Evaluation | — |
| 3.6.5 | Tracker Dependency | — |
| 3.6.6 | Minimum Vehicle Count | — |

**Total estimated length:** 22-27 pages.
