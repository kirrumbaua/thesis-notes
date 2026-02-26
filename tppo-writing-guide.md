# Updated TPPO Section-by-Section Guide (With Tables and Figures)

---

## Section 1: Introduction

**AI to use:** Any general-purpose AI with search capability

**Prompt:**

I am writing the introduction section of a thesis proposal document for a bachelor's degree in computer science. The thesis is about improving illegal parking detection systems that use fixed CCTV cameras. I need you to write three to four paragraphs covering the following flow. You may include a table if it helps present information clearly.

First paragraph: Describe illegal parking as an urban transportation problem. Cover how it reduces lane capacity, obstructs bus lanes and loading zones, creates safety hazards, and contributes to congestion. Include any available statistics or data on illegal parking enforcement challenges, particularly in urban areas in the Philippines or Southeast Asia if available, otherwise use general global context. Mention that manual enforcement through traffic wardens is inconsistent, labor-intensive, and cannot provide continuous monitoring.

Second paragraph: Introduce vision-based automated illegal parking detection as a solution. Explain that fixed CCTV cameras already exist in most urban areas and can be used for automated monitoring. Briefly describe how these systems work at a high level: they detect vehicles using computer vision, track them across video frames, and flag vehicles that remain stationary in restricted zones beyond a time threshold. State that this approach has been studied extensively in the computer vision literature over the past decade.

Third paragraph: Introduce the specific limitation this project addresses. Current systems evaluate each vehicle independently without considering what surrounding traffic is doing. When vehicles stop for legitimate reasons like red lights or congestion, the system cannot tell the difference between a parked vehicle and a vehicle waiting in traffic. This project proposes adding a context-aware layer that checks whether surrounding vehicles are also stationary before confirming a violation, using data already available from standard vehicle tracking.

Fourth paragraph: State what this document contains. This document presents the project background, the specific problem to be addressed, and the planned methodology for developing and evaluating the proposed approach. Keep the tone academic but accessible to faculty who may not specialize in computer vision.

---

## Section 2: Background

**AI to use:** NotebookLM

**Prompt:**

I need you to write a background summary of the 27 papers in this notebook for a thesis proposal document. The summary should be written in paragraph form with tables where appropriate to present comparative information clearly. Follow this specific flow. Each paragraph should transition naturally into the next.

First, write one paragraph describing the detection methods used across the papers. Group them into background subtraction approaches like GMM used by Sarker et al. and Wahyono et al. and deep learning approaches like the YOLO family used by Gao et al., Kathait et al., Sharma et al., Alwafi et al., and others, plus SSD used by Xie et al. and MobileNet used by Alon. State that detection accuracy consistently exceeds 85 percent under normal conditions in recent work and that detection is a mature component.

After this paragraph, include a table with columns for detection method, papers using it, and representative accuracy. Group the rows by background subtraction methods and deep learning methods.

Second, write one paragraph describing the tracking methods. Mention DeepSORT, OC-SORT, Kalman Filter, and optical flow as used across the papers. State that tracking enables persistent vehicle identification across frames which is necessary for measuring how long a vehicle has been stationary.

Third, write one paragraph describing how these systems decide a vehicle is illegally parked. Describe the dominant approach: the system checks if the vehicle is inside a predefined restricted zone and if it has been stationary longer than a time threshold. State that most systems evaluate each candidate vehicle independently.

After this paragraph, include a table with columns for paper name and year, zone type, temporal threshold value, and any additional decision criteria. Include all fixed-camera papers from the notebook. This table should show the full landscape of how violation decisions are made.

Fourth, write one paragraph describing the challenges reported across studies. Cover environmental challenges like rain and lighting briefly. Cover occlusion briefly. Then focus on the challenge of non-parking stationary vehicles. Specifically quote or closely paraphrase Gao et al. 2022 stating that differentiating vehicles stopped at red lights from illegally parked vehicles is a challenge. Quote or closely paraphrase Wahyono et al. 2017 stating that traffic jams and stopped vehicles at traffic light intersections cause false regions. Quote or closely paraphrase Kathait et al. 2025 stating that vehicles stopped at red or yellow signals must be considered. Mention that Huang et al. iPatrol reported precision dropping from 95 percent in empty traffic to 81.5 percent in crowded traffic.

Fifth, write one paragraph describing how each study responded to this challenge differently. Gao et al. built a pedestrian zone proxy because the traffic signal was not visible in their camera footage. Wahyono et al. used a 60-second timer to filter temporary stops. Kathait et al. proposed signal color detection but could not implement it because their dataset lacked visible signals.

After this paragraph, include a table with columns for paper, problem they stated using their own words, their response or fix, and what that fix requires to work.

Sixth, write one closing paragraph stating that across all 27 reviewed papers, no study checks whether surrounding tracked vehicles are also stationary before confirming a violation. This information is available from standard tracking output but is not used as a decision input in any reviewed system.

Write all paragraphs as continuous academic prose. Tables should be inserted between paragraphs where indicated.

---

## Section 3: Purpose of the Project / Statement of the Problem

**Written directly. Here is the output.**

---

The review of 27 studies on vision-based illegal parking detection revealed that the dominant violation decision logic in fixed-camera systems evaluates each candidate vehicle independently based on zone membership and elapsed time. Three independent fixed-camera studies — Gao et al. (2022), Wahyono et al. (2017), and Kathait et al. (2025) — documented that this independent evaluation produces false detections when vehicles are stopped for legitimate reasons such as traffic signals, congestion, or queuing. Each study developed a response specific to their dataset: Gao et al. used pedestrian crosswalk activity as a proxy for signal phase, Wahyono et al. applied a temporal filter, and Kathait et al. proposed signal color detection. All existing responses require specific scene elements to function and evaluate the candidate vehicle against fixed external references. No reviewed study examines whether surrounding tracked vehicles are also stationary — information already available from standard tracking modules — as a factor in the violation decision. This project focuses on this specific gap because it represents a generalizable approach that does not depend on scene-specific elements such as visible crosswalks or traffic signal heads, and because the problem has been independently documented but never formally evaluated with before-and-after performance measurement.

**Table 1.** Summary of evidence papers documenting the non-parking stationary vehicle challenge.

| Paper | Year | Problem Stated (Authors' Words) | Response Implemented | Scene Requirements |
|---|---|---|---|---|
| Gao et al. | 2022 | "Differentiating between vehicles stopped for red lights and those illegally parked" | Pedestrian zone proxy for signal phase inference | Visible crosswalk, pedestrian presence, two additional detection zones |
| Wahyono et al. | 2017 | "False regions due to a traffic jam, stopped vehicles and standing human at the traffic light intersection" | 60-second tracking timer | None additional, but cannot distinguish prolonged congestion from parking |
| Kathait et al. | 2025 | "Vehicles which are at stop due to red or yellow traffic signal are also considered" | Signal color detection (proposed) | Visible traffic signal head (not implemented due to dataset) |
| iPatrol / Huang et al. | 2023 | Precision dropped from ~95% (empty) to 81.5% (crowded) | None specific to this challenge | — |

**Statement of the Problem:**

1. How does the inclusion of a context-aware decision logic layer that evaluates surrounding vehicle stationarity affect the false positive rate of a fixed-camera illegal parking detection system compared to a baseline system that evaluates each vehicle independently?

2. To what extent does the proposed stationary ratio-based filtering reduce traffic-related false positives — specifically those caused by traffic signal stops and congestion — while maintaining detection of genuine parking violations?

3. What is the effect of the congestion threshold parameter on the tradeoff between false positive suppression and genuine violation retention in the proposed system?

**Objectives:**

This project aims to design, implement, and evaluate a context-aware decision logic layer that uses surrounding vehicle displacement data from standard tracking output to improve violation classification accuracy in fixed-camera illegal parking detection. Specifically, the objectives are:

1. To develop a context-aware decision logic layer that computes the stationary ratio of surrounding tracked vehicles and applies rule-based conditional logic — including collective stop detection, collective clearance detection, and isolation verification — to distinguish individually parked vehicles from vehicles stopped due to collective traffic conditions.

2. To evaluate the effect of the proposed layer on detection performance by comparing a baseline system without traffic context against the proposed system with the context-aware layer, using precision, recall, F1 score, and categorized false positive source analysis on identical test footage with manually annotated ground truth.

3. To determine the sensitivity of the proposed system to the congestion threshold parameter by evaluating performance across multiple threshold values and identifying the range that maximizes false positive reduction while maintaining acceptable recall of genuine violations.

---

## Section 4: Methodology / Approach to Address Problem

**Written directly. Here is the output.**

---

The project will be undertaken in five phases across Thesis 2 and Thesis 3.

**Phase 1: Data Collection and Preparation**

Video footage will be collected from a fixed camera position at a location containing a designated no-parking zone near a signalized intersection. The recording site must satisfy the conditions outlined in Table 2.

**Table 2.** Dataset requirements and justification.

| Requirement | Description | Why It Is Needed |
|---|---|---|
| Static camera | Fixed position and angle throughout recording | Ensures consistent spatial reference for zone definition |
| Mixed traffic scenarios | Free-flowing traffic, collective stops (red lights, congestion), and genuine violations in same recording | Enables testing of both baseline and proposed system across all relevant conditions |
| Signal cycle effects | Red light stops and green light departures observable | Provides the primary test case for distinguishing collective stops from parking |
| Multiple vehicles | More than one vehicle visible simultaneously | Stationary ratio requires surrounding vehicles to compute |
| Visible zone boundaries | Lane markings or curb edges visible | Enables accurate restricted zone polygon definition |

Each stationary vehicle event in the test footage will be manually annotated with ground truth labels. The annotation protocol is summarized in Table 3.

**Table 3.** Ground truth annotation labels.

| Label | Values | Description |
|---|---|---|
| Event ID | Unique integer | Identifies the stationary event |
| Vehicle type | Car, truck, bus, van, motorcycle | Visual classification by annotator |
| Ground truth status | Violation / Non-violation | Is this vehicle genuinely illegally parked? |
| Stop reason (if non-violation) | Red light, congestion, loading, other | Why the vehicle is legitimately stopped |
| Surrounding traffic state | Flowing, mixed, collectively stopped | What surrounding vehicles are doing at the time |
| Duration | Seconds | From first stationary frame to departure |

A minimum of 20 percent of annotated events will be independently labeled by a second annotator to assess inter-annotator agreement using Cohen's kappa coefficient.

**Phase 2: Baseline System Implementation**

A baseline illegal parking detection system will be built following the standard pipeline identified as the dominant approach in the literature review. The system will use YOLOv8 pretrained on the COCO dataset for vehicle detection, DeepSORT for multi-object tracking with persistent vehicle identification, a manually defined polygon for restricted zone membership checking using a point-in-polygon test, and a 60-second temporal threshold for violation classification. The 60-second threshold is adopted from the majority of reviewed fixed-camera studies including Wahyono et al. (2017), Akhawaji et al. (2017), Ng et al. (2018), and Kakad et al. (2024). The baseline evaluates each candidate vehicle independently and produces a binary violation or non-violation output without considering surrounding traffic behavior.

**Phase 3: Context-Aware Decision Logic Layer Implementation**

The proposed layer will be inserted after the temporal threshold check in the baseline pipeline. It will operate on displacement data already produced by the tracking module without requiring additional detection models or sensors.

For every tracked vehicle in each frame, displacement will be computed as the Euclidean distance between the vehicle's bounding box centroid in the current frame and its centroid N frames prior. Each vehicle will be classified as stationary if displacement is below δ pixels or moving if above. From these per-vehicle classifications, a stationary ratio R will be computed as the proportion of all tracked vehicles currently stationary.

The layer will implement three sequential checks as illustrated in Figure 1.

**Figure 1.** Combined decision flow of the context-aware decision logic layer.

```
Candidate vehicle exceeds 60-second threshold in zone
                         ↓
              Compute Stationary Ratio R
                         ↓
                ┌────────┴────────┐
                │                 │
            R > τ              R ≤ τ
        (majority            (traffic
         stopped)             flowing)
                │                 │
                ↓                 ↓
        SUPPRESS flag      CONFIRM violation
        (collective         + isolation label
         stop detected)      (Check 3)
                │
                ↓
        Monitor for clearance
        (R drops below τ)
                │
        ┌───────┴───────┐
        │               │
    Clearance       No clearance
    detected        yet
        │               │
        ↓               ↓
    Did candidate    Remain
    move?            suppressed
    ┌───┴───┐
    │       │
  Yes      No
    │       │
    ↓       ↓
  NON-    CONFIRM
  VIOL.   violation
          (Check 2)
```

The planned parameter values for the proposed system are summarized in Table 4.

**Table 4.** Planned system parameters.

| Parameter | Symbol | Planned Value | Rationale |
|---|---|---|---|
| Temporal threshold | T | 60 seconds | Literature consensus from reviewed studies |
| Lookback window | N | 30 frames | At 10 fps, covers 3 seconds — sufficient to distinguish stopped from slow-moving |
| Displacement threshold | δ | 5 pixels | Absorbs bounding box jitter without masking genuine movement |
| Congestion threshold | τ | 0.6 | Majority threshold — suppresses only when clear majority is stopped |
| Clearance window | W | 50 frames | At 10 fps, covers 5 seconds — sufficient to observe signal transition |

**Phase 4: Evaluation**

Both the baseline and proposed systems will process identical test footage using identical detection weights, tracking parameters, zone definitions, and temporal thresholds. The only variable will be the presence or absence of the context-aware layer. The experimental configurations are defined in Table 5.

**Table 5.** Experimental configurations.

| Configuration | Detection | Tracking | Zone | Threshold | Context Layer |
|---|---|---|---|---|---|
| Baseline | YOLOv8 | DeepSORT | Same polygon | 60 seconds | OFF |
| Proposed | YOLOv8 | DeepSORT | Same polygon | 60 seconds | ON |

Performance will be measured using the metrics defined in Table 6.

**Table 6.** Evaluation metrics.

| Metric | Formula | What It Measures | Expected Direction |
|---|---|---|---|
| Precision | TP / (TP + FP) | Of flagged violations, proportion that are genuine | Proposed > Baseline |
| Recall | TP / (TP + FN) | Of genuine violations, proportion correctly flagged | Proposed ≈ Baseline |
| F1 Score | 2 × P × R / (P + R) | Balance of precision and recall | Proposed ≥ Baseline |

False positives from both systems will be categorized by cause as defined in Table 7.

**Table 7.** False positive source categories.

| Category | Definition | Expected Effect of Proposed Layer |
|---|---|---|
| Signal-related | Vehicle was stopped at a red light | Reduced or eliminated |
| Congestion-related | Vehicle was in slow or queued traffic | Reduced or eliminated |
| Detection/tracking error | Caused by misdetection or ID switching | Unchanged |
| Other | Any other cause | Unchanged |

An ablation study will test each check incrementally to isolate individual contributions, as outlined in Table 8.

**Table 8.** Ablation study configurations.

| Configuration | Collective Stop Detection | Collective Clearance Check | Isolation Verification |
|---|---|---|---|
| Baseline | OFF | OFF | OFF |
| Check 1 only | ON | OFF | OFF |
| Check 1 + Check 2 | ON | ON | OFF |
| Full proposed | ON | ON | ON |

A parameter sensitivity analysis will evaluate the system at congestion threshold values of 0.3, 0.4, 0.5, 0.6, 0.7, and 0.8 to determine the precision-recall tradeoff and identify the optimal operating range.

**Phase 5: Documentation**

All results, analysis, limitations, and recommendations for future work will be documented in the thesis manuscript and prepared for defense.
