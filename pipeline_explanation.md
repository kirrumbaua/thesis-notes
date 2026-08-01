# Complete System Data Pipeline & Technical Architecture

This document provides a step-by-step technical breakdown of the context-aware illegal parking detection pipeline. It traces the exact data flow from raw video frames down to the binary violation output and baseline comparative evaluation.

After each stage, a **Rationale** section explains why that stage exists in the pipeline and what problem it solves, particularly where the revised methodology differs from the original rejected proposal.

---

## Pipeline Overview Diagram

```
[ Raw Video Stream (1080p @ 30 fps) ]
                 │
                 ▼
 ┌───────────────────────────────────────────────┐
 │ STAGE 1: Offline Calibration & Preprocessing   │
 └───────────────────────────────────────────────┘
                 │
                 ▼
 ┌───────────────────────────────────────────────┐
 │ STAGE 2: Per-Frame Vision Pipeline            │
 │ (YOLOv8n Detection -> IoU Tracking)           │
 └───────────────────────────────────────────────┘
                 │
                 ▼
 ┌───────────────────────────────────────────────┐
 │ STAGE 3: Coordinate Mapping & Motion Tracking │
 │ (Bottom-Center -> 2D Homography -> Meters)    │
 └───────────────────────────────────────────────┘
                 │
                 ▼
 ┌───────────────────────────────────────────────┐
 │ STAGE 4: Restricted Zone & Dwell Timer        │
 │ (Point-in-Polygon -> Timer -> T=60s Trigger)  │
 └───────────────────────────────────────────────┘
                 │
                 ▼
 ┌───────────────────────────────────────────────┐
 │ STAGE 5: Spatiotemporal Feature Extraction     │
 │ (Snapshot Phase at t_trig + Post-Window W=15s)│
 └───────────────────────────────────────────────┘
                 │
                 ▼
 ┌───────────────────────────────────────────────┐
 │ STAGE 6: XGBoost Event Classification         │
 │ (Fallback Check for C < 2 -> Binary Output)   │
 └───────────────────────────────────────────────┘
                 │
                 ▼
 [ Output: Event Violation Flag (0 or 1) ]
```

---

## Detailed Step-by-Step Data Flow

---

### Stage 1: Offline Camera Calibration & Setup

This stage is performed once before any video is processed.

**Step 1A: Intrinsic Lens Undistortion (Pinhole Camera Model)**
- **Input:** Checkerboard calibration images taken by the recording camera.
- **Process:** OpenCV pinhole camera calibration extracts the intrinsic matrix and distortion coefficients.
- **Output:** Every incoming video frame is undistorted to remove radial lens bending (the "fish-eye" warping effect at frame edges).

**Step 1B: Extrinsic Ground Mapping (2D Homography)**
- **Input:** Minimum of 4 Ground Control Points (GCPs) physically measured on the road surface. Each GCP has a known pixel location in the video AND a known real-world position in meters.
- **Process:** RANSAC-based homography estimation computes a 3×3 transformation matrix **H** that maps any pixel coordinate to a real-world ground coordinate in meters.
- **Output:** A calibrated transformation matrix **H**.

> **Rationale:** Our original proposal used Homography to convert pixels to meters, but applying it directly to raw video creates errors because camera lenses physically curve straight lines at the edges. Homography assumes straight lines stay straight. Following the panel's advice to study both pinhole and homography, we upgraded to a **Two-Stage Calibration**: Pinhole (Intrinsic) un-curves lens distortion first, then Homography (Extrinsic) maps pixels to real-world meters. This makes ground-plane displacement accurate across the entire frame.

---

### Stage 2: Per-Frame Vision Pipeline (Continuous at 30 fps)

This stage runs on every single frame.

**Step 2A: Object Detection (YOLOv8n)**
- **Input:** Undistorted 1080p frame.
- **Process:** YOLOv8n single-stage detection, filtered strictly for vehicle classes (`car`, `motorcycle`, `bus`, `truck`) with confidence ≥ 0.25.
- **Output:** Bounding box coordinates (x, y, w, h), class label, and confidence score for every detected vehicle in the frame.

**Step 2B: Multi-Object Tracking (IoU Tracker)**
- **Input:** Per-frame bounding boxes from YOLOv8n.
- **Process:** IoU-based tracker (Bochinski et al.) associates bounding boxes across consecutive frames using overlap percentage (σ_IoU = 0.5).
- **Track Lifecycle:**
  - *Tentative:* A new detection must appear for 3 consecutive frames before becoming an active track (prevents single-frame false detections from polluting the data).
  - *Active:* Maintains a persistent integer Track ID across frames.
  - *Terminated:* If unmatched for 5 consecutive frames (~0.17s), the track is deleted and the ID is never reused.
- **Output:** Persistent Track ID assigned to each vehicle trajectory over time.

---

### Stage 3: Coordinate Transformation & Displacement Computation

For each active tracked vehicle at every frame:

**Step 3A: Ground Contact Point Extraction**
- The bottom-center of the bounding box is extracted: $p_{\text{bottom}} = (x + w/2, \; y + h)$.
- *Why we need this:* Homography only works on the flat 2D road surface. A vehicle's roof or center floats in 3D space above the ground. If we tracked the box center, the vehicle's height would create false movement errors as it drives across the camera's perspective. The bottom-center (where tires touch the road) is the only point physically lying on the ground plane.

**Step 3B: Homography Projection**
- The bottom-center pixel coordinate is passed into the homography matrix $\mathbf{H}$, which converts it into real-world ground coordinates $(X_t, Y_t)$ in meters.

**Step 3C: Lookback Displacement Calculation**
- How data flows here: Step 3B outputs real-world meters $(X_t, Y_t)$ for the current frame. Step 3C compares this current position against the vehicle's position 3 seconds ago $(X_{t-90}, Y_{t-90})$ using Euclidean distance:
  $$d(v, t) = \sqrt{(X_t - X_{t-90})^2 + (Y_t - Y_{t-90})^2}$$
- *Output:* Actual physical distance moved in meters over the last 3 seconds.

> **Rationale:** Ground contact point extraction is required because homography maps a flat 2D plane: tracking tire contact prevents vehicle height from distorting real-world position. Passing these converted meter coordinates into a 3-second lookback window cleanly separates moving vehicles from stationary ones. At a slow walking speed of 1.4 m/s, a moving vehicle travels 4.2 meters in 3 seconds, which easily clears detection jitter and makes stationary classification reliable.

---

### Stage 4: Restricted Zone Monitoring & Event Triggering

**Step 4A: Restricted Zone (ROI) Check**
- A closed polygon representing the illegal parking restricted area is drawn once onto the undistorted frame during setup. Zone boundaries are grounded in the statutory provisions of Republic Act No. 4136 at the physical test site.
- Every frame, a ray-casting point-in-polygon test checks whether each vehicle's bottom-center lies inside the polygon.

**Step 4B: Depth-Dependent Stationary Classification**
- A vehicle is classified as "stationary" if its 3-second displacement is below a calibrated noise threshold ε_s(Y).
- **Depth Scaling:** Because perspective projection causes the same pixel jitter to translate to larger meter errors at far-field depths, ε_s is calibrated separately at near, mid, and far positions within the ROI. A vehicle known to be stationary is recorded at each depth for 60 seconds, and the observed displacement noise is measured. The threshold is set to 1.5× the measured noise at each depth, with linear interpolation between calibration points.

**Step 4C: Dwell Timer & Event Trigger**
- Each vehicle inside the ROI maintains an independent dwell timer.
- Timer increments by 1/fps seconds every frame the vehicle remains stationary.
- Timer resets to zero if the vehicle moves (displacement ≥ ε_s).
- **Trigger Event:** When the timer reaches T = 60 seconds, an event trigger fires for that candidate Track ID. Each Track ID can only trigger once.

> **Rationale:** The 60-second temporal threshold is adopted from the majority of fixed-camera parking detection studies in the literature and serves as the minimum evidence threshold for a potential violation. The depth-dependent noise ceiling ε_s(Y) is a new addition that did not exist in the original proposal. It exists because applying a single global stationary threshold across the entire frame would cause near-field vehicles creeping forward slowly to be misclassified as stationary (the far-field noise value is too lenient for near-field measurements). This stage produces the event trigger that activates the downstream feature extraction and classification; it is the gate that separates a vehicle just passing through from a vehicle that has been present long enough to evaluate.

---

### Stage 5: Spatiotemporal Feature Vector Extraction

When an event triggers at time t_trig, a 6-dimensional feature vector is constructed in two phases.

Let **S(t)** = the set of all surrounding tracked vehicles at frame t, excluding the candidate vehicle and excluding any vehicle with fewer than 90 frames of tracking history.

---

#### Phase A: Snapshot Features (Captured at t_trig)

**Feature 1: Stationary Ratio (R)**
The proportion of surrounding tracked vehicles that are also classified as stationary at the moment the candidate triggers.

- **What it captures:** The collective traffic state around the candidate. If R is high (most surrounding cars are also stopped), the candidate is likely stopped due to a shared external cause like a red light or traffic jam. If R is low (surrounding traffic is flowing freely), the candidate is stationary while everyone else is moving, which is a strong signal of an anomalous individual stop.
- **Why it matters:** This is the core theoretical anchor of the entire thesis. It directly measures the causal coupling described by Kinematic Wave Theory: legitimate traffic stops produce correlated, collective deceleration across surrounding vehicles, while illegal parking is an isolated individual anomaly.

**Feature 2: Surrounding Vehicle Count (C)**
The total number of tracked vehicles in the surrounding set at the trigger frame.

- **What it captures:** The statistical reliability of the Stationary Ratio. An R value of 0.50 computed from 2 vehicles is far less reliable than R = 0.50 computed from 20 vehicles. C lets the classifier weight the ratio's significance appropriately.
- **No multicollinearity with R:** R is a proportion (always between 0 and 1). C is an absolute count. You can have R = 0.50 with C = 2 (1 out of 2 stopped) or R = 0.50 with C = 20 (10 out of 20 stopped). They carry independent information.

**Feature 3: Mean Surrounding Displacement (d̄_surr)**
The average 3-second displacement (in meters) of all surrounding vehicles at the trigger frame.

- **What it captures:** The actual physical speed of ambient traffic. R only tells you the proportion of stopped vehicles; it does not tell you how fast the non-stopped vehicles are moving. A scenario where 50% of traffic is stopped and the other 50% is crawling at 2 km/h (heavy congestion) is very different from a scenario where 50% is stopped and the other 50% is flowing at 40 km/h (mixed conditions). d̄_surr captures this distinction.
- **No multicollinearity with R:** R is a binary proportion (stationary or not). d̄_surr is a continuous displacement value capturing movement magnitude. A high R does not determine d̄_surr (the moving vehicles could be fast or slow), and a low d̄_surr does not determine R (all vehicles could be creeping slowly without any being classified as stationary).

**Feature 4: Surrounding Displacement Std. Dev. (σ_d)**
The standard deviation of displacement values among all surrounding vehicles at the trigger frame.

- **What it captures:** The uniformity of surrounding traffic behavior. During a red light, all vehicles tend to stop together (low σ_d, uniform behavior). In mixed or transitional traffic (some stopped, some moving, some lane-changing), displacement values are spread out (high σ_d). This helps the classifier distinguish between clean collective stops and messy, heterogeneous traffic conditions.
- **No multicollinearity with d̄_surr:** Mean and standard deviation measure fundamentally different properties of a distribution. You can have a mean displacement of 5 meters with σ = 0.5 (everyone moving at roughly the same speed) or σ = 8.0 (huge variation in speeds). They are statistically independent descriptors.

---

#### Phase B: Post-Monitoring Features (Captured after W = 15 seconds)

After the trigger fires, the system continues tracking the candidate and all surrounding vehicles for an additional 15 seconds before extracting two more features.

**Feature 5: Ratio Change (ΔR)**
The change in Stationary Ratio between the trigger moment and 15 seconds later.

- **What it captures:** The temporal dynamics of the surrounding traffic state. If surrounding traffic was stopped at the trigger (R was high) but clears 15 seconds later (R drops), it strongly suggests the initial stop was caused by a traffic signal or temporary congestion. If the candidate remains stationary while the ratio drops (surrounding traffic resumes flow), the candidate becomes increasingly anomalous. Conversely, if R stays high or increases, the collective stop is ongoing and the candidate's stationary state is less suspicious.
- **Why 15 seconds:** This window is long enough to capture a typical traffic signal phase transition (green-to-red or red-to-green), which is the most common cause of collective traffic stops. It is short enough to avoid excessive processing delay.
- **No multicollinearity with R:** R is a single snapshot value at one moment in time. ΔR is a temporal difference across a 15-second window. The same R value at trigger can produce any ΔR depending on what happens to surrounding traffic afterwards. They capture fundamentally different temporal dimensions: state versus change-in-state.

**Feature 6: Candidate Post-Trigger Displacement (d_cand)**
The total displacement of the candidate vehicle itself over the 15-second post-monitoring window.

- **What it captures:** Whether the candidate moved at all after the 60-second trigger. A vehicle caught in traffic might resume moving once the light turns green (non-zero d_cand). A genuinely illegally parked vehicle will remain at displacement ≈ 0.
- **No multicollinearity with other features:** All other features (R, C, d̄_surr, σ_d, ΔR) describe the surrounding traffic. d_cand is the only feature that describes the candidate vehicle itself. It provides direct evidence about the candidate's own behavior during the observation window, independent of what surrounding vehicles are doing.

---

**Final Output: The Event Feature Vector:**
$$\mathbf{X}_{\text{event}} = \left[ R, \; C, \; \bar{d}_{\text{surr}}, \; \sigma_d, \; \Delta R, \; d_{\text{cand}} \right]$$

> **Rationale:** In the original rejected proposal, the system used a single hardcoded threshold on R alone (if R > 0.6, suppress the flag). The panel rejected this as "just an if/else script" with zero modeling. This 6-feature vector replaces that hardcoded rule entirely. Instead of one number and one threshold, the classifier now receives six independent descriptors that capture: the collective traffic state (R), its statistical reliability (C), the physical speed of surrounding traffic (d̄_surr), the uniformity of that traffic (σ_d), whether the traffic state is changing over time (ΔR), and whether the candidate itself is truly immobile (d_cand). No feature is a linear combination of any other; each captures a distinct, independent dimension of the spatiotemporal context. The trained XGBoost classifier learns its own decision boundary across all six dimensions simultaneously, which is the actual "modeling" the panel demanded.

---

### Stage 6: Event Classification & Fallback Logic

**Step 6A: Solitary Vehicle Fallback Check (C < 2)**
- If the surrounding vehicle count C is less than 2, the XGBoost classifier is bypassed entirely.
- The event is flagged as a violation using standard temporal-threshold logic (the vehicle has been stationary for 60 seconds in a restricted zone with no surrounding traffic context to evaluate).
- These fallback events are reported separately and excluded from classifier performance metrics.

> **Rationale for Fallback:** The context-aware classifier needs surrounding traffic data to function. If a candidate vehicle is alone on an empty road with zero or one surrounding vehicle, there is no collective traffic behavior to measure. However, this is not a weakness; it is actually optimal. In low-density, empty-road conditions, collective traffic stops (red lights, congestion) physically do not occur. There are no other cars around to create a false positive. Standard timer-only logic works perfectly in these scenarios. The classifier specifically targets the high-density urban scenarios where surrounding traffic creates the false positive problem.

**Step 6B: XGBoost Classifier Evaluation (C ≥ 2)**
- The 6-feature vector X_event is passed into the trained XGBoost model.
- XGBoost outputs a probability score P(violation | X_event).
- Binary decision:
  - P ≥ 0.50 → **Violation Flag (1)**
  - P < 0.50 → **Non-Violation / Legitimate Stop (0)**

> **Rationale:** XGBoost is chosen over heavy deep neural networks (CNNs, transformers) because the system must operate under real-time edge-AI processing constraints on fixed camera infrastructure. The feature vector is only 6 dimensions; a lightweight gradient-boosted tree handles this efficiently with minimal computational overhead while still learning complex non-linear decision boundaries across all six features simultaneously. This is the trained classifier that replaces the hardcoded if/else threshold the panel rejected.

---

## Comparison Baseline Architecture (Abella & Catedrilla, 2025)

To evaluate the proposed pipeline's performance, it is benchmarked against a same-dataset reimplementation of Abella & Catedrilla (2025):

- **Baseline Logic:** Per-frame YOLOv8 detection only.
- **Violation Rule:** Flagged as a violation if `vehicle`, `person`, and `no_parking_sign` co-occur in the same frame. No tracking. No timer. No surrounding traffic context.
- **Their Reported Results:** Illegal parking precision of 0.7467, with 509 combined false positives across 2,803 total detections. By mathematical deduction, the vast majority of those 509 false positives belong to the parking class (their littering class precision was 0.9841).
- **Controlled Evaluation:** Both systems are tested on the exact same video footage. The evaluation pool is restricted to events that passed the 60-second stationary threshold, measuring how well each system discriminates genuine violations from legitimate traffic stops within that pool.

> **Rationale for this Baseline:** The comparison is designed as a controlled same-dataset test, not a cross-paper metrics comparison. Abella et al. is selected because (1) illegal parking detection is its stated primary contribution, (2) it represents a fundamentally different detection paradigm (single-frame spatial co-occurrence vs. tracked spatiotemporal classification), and (3) their published false positive count (509) provides concrete empirical evidence that the single-frame approach suffers from exactly the type of misclassification our context-aware approach is designed to eliminate.
