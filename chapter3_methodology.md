# Chapter 3: METHODOLOGY

## 3.1 Conceptual Framework

The proposed system processes fixed-camera video through a three-stage sequential pipeline to classify individual vehicle events as illegal parking violations or legitimate traffic stops. The first stage, the Vision Pipeline, performs continuous per-frame vehicle detection, multi-object tracking, and real-world coordinate transformation. The second stage, the Event Trigger and Feature Extraction module, monitors tracked vehicles within a legally defined restricted zone and, upon satisfaction of a temporal threshold, computes a structured feature vector encoding the surrounding traffic context and the candidate vehicle's post-trigger behavior. The third stage, the Event Classifier, processes the feature vector through a trained XGBoost model to produce a binary violation or non-violation prediction for each triggered event.

Context-awareness in this architecture is not a discrete software module. It is an emergent property of the feature design and the classifier's learned decision boundaries operating together: the feature extraction stage encodes the aggregate motion state of surrounding tracked vehicles into numeric form, and the classifier uses that encoded context to discriminate between illegal parking and collective traffic stops.

[Figure 3.1 placeholder]

*Figure 3.1. Conceptual framework of the proposed context-aware spatiotemporal classification pipeline, showing the data flow from raw video input through the Vision Pipeline, Event Trigger and Feature Extraction, and Event Classifier stages to the final binary output.*

---

## 3.2 Research Design

This study employs an experimental quantitative research design. The independent variable is the violation decision method: the proposed context-aware spatiotemporal classifier versus a reimplementation of an existing frame-level composite-class detection approach. The dependent variables are the classification performance metrics: precision, recall, F1-score, and false positive rate. Both methods are evaluated on the same video footage, using the same camera, resolution, frame rate, object detector version, and ground truth event annotations, ensuring a controlled same-data comparison under identical test conditions.

---

## 3.3 Data Acquisition

### 3.3.1 Test Site

The recording location will be [test site name and description to be inserted upon site confirmation]. The selected site satisfies the following criteria:

1. A legally designated no-parking zone, established either by posted signage or by statutory restriction under Republic Act No. 4136 (Land Transportation and Traffic Code of the Philippines), which prohibits parking within an intersection, on a crosswalk, within six meters of a curb-line intersection, in front of a driveway, or in a manner that blocks a designated driving lane.
2. No traffic signal at the immediate recording location. This condition strengthens the rationale for the proposed context-aware approach: without a signal head in the camera's field of view, signal-state-based suppression methods [citation: Kathait et al.] are physically inapplicable, and the only available contextual information is the dynamic behavior of surrounding vehicles.
3. Sufficient traffic volume during the recording period to ensure that multiple vehicles are simultaneously visible for the majority of recording time.
4. A physical vantage point permitting elevated camera placement to minimize inter-vehicle occlusion.

The restricted zone (Region of Interest) is defined as a closed polygon drawn once onto the camera's coordinate space during setup. The polygon boundaries are spatially grounded in the statutory definitions of RA 4136, eliminating dependence on the visibility of physical paint or signage for zone definition.

### 3.3.2 Recording Configuration

Video footage will be recorded using an Apple iPhone 16 Pro mounted on a tripod at a fixed, elevated position overlooking the test site. Table 3.1 summarizes the recording parameters.

| Parameter | Value |
|---|---|
| Device | Apple iPhone 16 Pro (main camera, 24 mm equiv.) |
| Resolution | 1920 × 1080 pixels |
| Frame rate | 30 fps |
| Recording mode | Standard video; HDR, Cinematic Mode, Action Mode, and digital zoom disabled |
| Mounting | Tripod, fixed position, no operator contact during recording |
| Height | [To be measured at site, in meters from road surface to lens] |
| Tilt angle | [To be measured at site, in degrees from horizontal] |
| Recording period | [Date(s) to be confirmed], 16:00 to 18:00 local time, clear or overcast conditions |

*Table 3.1. Video recording parameters.*

The camera will remain stationary throughout each recording session. No post-capture image processing will be applied to the raw footage. All recording parameters will be held constant across sessions to ensure geometric consistency for calibration.

The recorded footage must contain sufficient events across four scenario categories: free-flowing traffic (vehicles passing without stopping), collective stops (multiple vehicles stationary simultaneously due to congestion or pedestrian yielding), genuine violations (vehicles illegally parked in the restricted zone), and mixed traffic (transitions between flowing and stopped states). If insufficient violation events occur naturally, controlled scenarios will be staged using a designated vehicle parked deliberately in the restricted zone while surrounding traffic continues. The proportion of staged events will be reported. It is acknowledged that staged events may exhibit subtly different surrounding traffic patterns compared to naturally occurring violations (e.g., reduced urgency in the staging behavior, or nearby traffic reacting to the staging activity), and this is noted as a limitation of the data collection procedure.

### 3.3.3 Camera Calibration

Accurate displacement computation requires two sequential calibration steps: lens distortion correction and ground-plane homography estimation. Both are performed once per recording setup.

#### Intrinsic Calibration

The iPhone camera lens introduces radial and tangential distortion that causes straight lines to appear curved, introducing systematic error into geometric measurements. The intrinsic parameters will be estimated using the standard checkerboard calibration method implemented in OpenCV [citation: Bradski, 2000].

A printed checkerboard pattern (9 × 6 internal corners) affixed to a rigid surface will be captured in a minimum of 20 images at varying orientations and distances, using the recording configuration in Table 3.1. Corner detection (`cv2.findChessboardCorners` with sub-pixel refinement via `cv2.cornerSubPix`) and camera calibration (`cv2.calibrateCamera`) will yield the camera intrinsic matrix $\mathbf{K}$ and distortion coefficients $\mathbf{D}$:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{D} = (k_1, k_2, p_1, p_2, k_3)$$

where $f_x, f_y$ are the focal lengths in pixels, $(c_x, c_y)$ is the principal point, $k_1, k_2, k_3$ are radial distortion coefficients, and $p_1, p_2$ are tangential distortion coefficients. All video frames will be undistorted using `cv2.undistort` before any downstream processing. The calibration reprojection error (RMS of corner deviations) will be reported; values below 1.0 pixel are considered acceptable [PENDING VERIFICATION: source for 1.0 pixel acceptability threshold].

#### Homography Estimation

A planar homography maps undistorted pixel coordinates to real-world ground-plane coordinates in meters, under the assumption that all mapped points lie on the road surface plane.

At the test site, a minimum of four non-collinear reference points will be placed on the road surface at positions measured with a tape measure (e.g., tape markers forming a rectangle of known dimensions). The pixel coordinates $(x_i, y_i)$ of each reference point in the undistorted frame and their corresponding real-world coordinates $(X_i, Y_i)$ in meters will be used to compute the 3 × 3 homography matrix $\mathbf{H}$ via `cv2.findHomography` with RANSAC outlier rejection.

The transformation maps any undistorted pixel coordinate $(x, y)$ to ground-plane coordinates $(X, Y)$:

$$\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad X = \frac{X'}{w}, \quad Y = \frac{Y'}{w}$$

The homography will be validated by computing the distance between two additional reference points not used in the estimation from both direct measurement and homography-derived coordinates. The absolute error will be reported; errors exceeding 0.10 meters will prompt recalibration.

---

## 3.4 Proposed Method

The system architecture consists of three sequential processing stages. Each stage's output serves as the direct input to the next.

[Figure 3.2 placeholder]

*Figure 3.2. System architecture showing the three-stage pipeline: Vision Pipeline (per-frame detection, tracking, coordinate transformation), Event Trigger and Feature Extraction (event-driven), and Event Classifier (per-event binary prediction).*

### 3.4.1 Vehicle Detection

Object detection is performed using YOLOv8n (Nano), the smallest variant of the YOLOv8 architecture [citation: Jocher et al., 2023], with 3.2 million parameters and 8.7 GFLOPs per inference pass. The model uses pretrained MS COCO weights [citation: Lin et al., 2014] without fine-tuning. Only vehicle classes are retained: car (class 2), motorcycle (3), bus (5), and truck (7). All other detections are discarded.

A detection confidence threshold of 0.25 is applied. This permissive threshold prioritizes recall at the detection stage: it is preferable to track a marginal vehicle detection than to miss a vehicle and undercount the surrounding traffic population. The detection module is not a contribution of this study; it provides standardized bounding box inputs to the tracking stage.

The Ultralytics YOLOv8 implementation outputs bounding box coordinates in the original image space (1920 × 1080), independent of the internal input resize, so no manual coordinate rescaling is required before homography transformation.

### 3.4.2 Multi-Object Tracking

A multi-object tracker assigns and maintains a persistent Track ID for each vehicle across consecutive frames. The tracker uses IoU-based (Intersection over Union) frame-to-frame association [citation: Bochinski et al., 2017], selected for its minimal computational overhead and its suitability for high-frame-rate fixed-camera scenarios where inter-frame vehicle displacement is small relative to bounding box size.

The tracking procedure operates as follows. At each frame, the set of current detections is compared against all active tracks from the previous frame. For each detection-track pair, the IoU between the detection's bounding box and the track's last known bounding box is computed:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where $A$ and $B$ are the areas of the two bounding boxes. Detection-to-track assignment is performed by selecting the assignment that maximizes total IoU, subject to a minimum IoU threshold $\sigma_{\text{IoU}} = 0.5$ [PENDING VERIFICATION: confirm \sigma_{IoU} = 0.5 as Bochinski et al.'s reported default/optimal value]. Detections with IoU below this threshold against all existing tracks initialize new tracks. Tracks not matched to any detection for a specified number of consecutive frames are terminated. Terminated Track IDs are not reused.

At 30 fps with a fixed camera, vehicles undergo small inter-frame displacements relative to their bounding box dimensions, producing high IoU overlap between consecutive frames. This makes IoU-based association sufficient for maintaining track identity without requiring appearance descriptors or motion prediction models. The tracking module is not a contribution of this study; it provides the persistent temporal identity required for downstream displacement computation and event triggering.

| Output | Type | Description |
|---|---|---|
| Track ID | Integer | Persistent identifier across frames |
| Bounding box | $(x, y, w, h)$ | Position and dimensions in pixels |
| Class label | Integer | COCO class ID |
| Confidence | Float | Detection score |

*Table 3.2. Per-frame tracking outputs.*

### 3.4.3 Coordinate Transformation and Displacement Computation

For each tracked vehicle at each frame, the bottom-center point of the bounding box is extracted:

$$p_{\text{bottom}} = \left(x + \frac{w}{2}, \; y + h\right)$$

The bottom-center approximates the vehicle's contact point with the road surface, which lies on the ground plane mapped by the homography. Using the bounding box centroid (vertical center) would introduce height-dependent projection error for off-nadir viewing angles.

The bottom-center pixel coordinates are transformed to real-world ground-plane coordinates $(X, Y)$ in meters using the calibrated homography $\mathbf{H}$ (Section 3.3.3).

The displacement of vehicle $v$ at frame $t$ is defined as the Euclidean distance between its current position and its position $n$ frames earlier:

$$d(v, t) = \sqrt{(X_t - X_{t-n})^2 + (Y_t - Y_{t-n})^2}$$

where $n = N \times \text{fps}$. The lookback window is set to $N = 3$ seconds ($n = 90$ frames at 30 fps). At this window length, a vehicle moving at minimal walking speed ($\approx$1.4 m/s) covers 4.2 meters, well above any stationary threshold, while a genuinely stopped vehicle shows displacement attributable only to detection noise. If a track has existed for fewer than $n$ frames, displacement is computed from the track's earliest available frame. However, vehicles with fewer than $n$ frames of tracking history are excluded from the surrounding vehicle set $\mathcal{S}(t)$ used in feature extraction (Section 3.4.5), since their displacement is computed over a shorter window than the one for which $\varepsilon_s$ was calibrated, which could produce unreliable stationary classifications.

### 3.4.4 Restricted Zone and Event Trigger

**Zone definition.** The restricted zone is specified as a closed polygon in pixel coordinates, drawn once onto the undistorted frame during setup. Its boundaries are spatially grounded in the statutory provisions of RA 4136 at the test site. Zone membership is evaluated per-frame using the ray casting point-in-polygon algorithm applied to each vehicle's bottom-center point.

[Figure 3.3 placeholder]

*Figure 3.3. Example restricted zone polygon overlaid on an undistorted frame from the test site.*

**Stationary classification.** A vehicle $v$ at frame $t$ is classified as stationary if its displacement falls below the stationary threshold $\varepsilon_s$:

$$\text{stationary}(v, t) = \begin{cases} 1 & \text{if } d(v, t) < \varepsilon_s \\ 0 & \text{otherwise} \end{cases}$$

The value of $\varepsilon_s$ will be determined empirically from the study's own pipeline. Because the homography maps pixels to meters under perspective projection, the same amount of pixel-level detection jitter corresponds to a larger real-world displacement for vehicles farther from the camera than for those closer to it. To account for this depth-dependent variation, the calibration will be performed at three positions within the restricted zone: near-field, mid-field, and far-field relative to the camera. At each position, a vehicle known to be completely stationary will be recorded for a minimum of 60 seconds under the recording configuration in Table 3.1 and processed through the full pipeline. The maximum observed displacement across all frames and all three positions will be recorded as the empirical noise ceiling, and $\varepsilon_s$ will be set to 1.5 times this value, ensuring that no genuinely stationary vehicle at any depth within the ROI is misclassified as moving. Both the per-position noise measurements and the resulting $\varepsilon_s$ will be reported.

**Event trigger.** A per-vehicle timer monitors each tracked vehicle whose bottom-center lies inside the restricted zone. The timer increments by one frame interval ($1/\text{fps}$ seconds) at each frame where $d(v, t) < \varepsilon_s$, and resets to zero if $d(v, t) \geq \varepsilon_s$. When the timer reaches the temporal threshold $T = 60$ seconds, a trigger event is generated for that Track ID. Each Track ID generates at most one trigger event.

The 60-second threshold is adopted from the majority of fixed-camera parking detection studies reviewed in Chapter 2 [citation: Alkhawaji et al.; Kakad et al.; Ng et al.; Alwafi et al.]. At the unsignalized test site, legitimate stationary stops (yielding to pedestrians, momentary queuing) are expected to be well below this duration under normal conditions.

### 3.4.5 Feature Extraction

Upon triggering, the system extracts a six-dimensional feature vector in two phases: four features are computed as a snapshot at the trigger frame $t_{\text{trig}}$, and two features are computed after a monitoring window of $W = 15$ seconds following the trigger. Table 3.3 defines the complete feature vector.

Let $\mathcal{S}(t)$ denote the set of all tracked vehicles at frame $t$ excluding the candidate vehicle and excluding any vehicle with fewer than $n$ frames of tracking history (see Section 3.4.3).

**Snapshot features (at $t_{\text{trig}}$):**

The Stationary Ratio $R$ measures the proportion of surrounding vehicles classified as stationary:

$$R(t) = \frac{\sum_{v_i \in \mathcal{S}(t)} \text{stationary}(v_i, t)}{|\mathcal{S}(t)|}$$

The Surrounding Vehicle Count $C = |\mathcal{S}(t_{\text{trig}})|$ records the number of tracked vehicles visible at the trigger frame, excluding the candidate. This contextualizes the statistical reliability of $R$: a ratio of 0.50 computed from 2 vehicles carries less evidential weight than the same ratio computed from 20 vehicles. Because $\mathcal{S}(t)$ excludes vehicles with fewer than $n$ frames of tracking history (Section 3.4.3), $C$ will systematically undercount the true number of surrounding vehicles during moments of high traffic turnover, since newly-appeared vehicles do not contribute to the surrounding-vehicle features until they have accumulated a full 3-second lookback window. This is an acknowledged trade-off: it replaces a more serious bias (misclassifying short-tracked moving vehicles as stationary, which would inflate $R$) with a smaller, more defensible one (temporarily undercounting the surrounding population).

The Mean Surrounding Displacement $\bar{d}$ provides a continuous measure of surrounding traffic flow intensity, preserving the magnitude of vehicle movement that the binary stationary classification discards:

$$\bar{d} = \frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} d(v_i, t_{\text{trig}})$$

The Surrounding Displacement Standard Deviation $\sigma_d$ captures the uniformity of traffic behavior. Low values indicate that surrounding vehicles are behaving similarly (all stopped or all flowing). High values indicate mixed behavior (some moving, some stopped), a condition distinct from both uniform flow and uniform stoppage:

$$\sigma_d = \sqrt{\frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} \left(d(v_i, t_{\text{trig}}) - \bar{d}\right)^2}$$

**Post-monitoring features (at $t_{\text{trig}} + W \times \text{fps}$):**

The Ratio Change $\Delta R$ captures the temporal evolution of the surrounding traffic state:

$$\Delta R = R(t_{\text{trig}} + W \times \text{fps}) - R(t_{\text{trig}})$$

Negative values indicate that surrounding traffic cleared during the monitoring period. Near-zero values indicate a stable traffic state.

The Candidate Post-Trigger Displacement $d_{\text{cand}}$ measures whether the candidate vehicle itself moved during the monitoring window:

$$d_{\text{cand}} = \sqrt{(X_{t_{\text{trig}} + W \times \text{fps}} - X_{t_{\text{trig}}})^2 + (Y_{t_{\text{trig}} + W \times \text{fps}} - Y_{t_{\text{trig}}})^2}$$

This is the only feature describing the candidate vehicle's own behavior; all others describe surrounding traffic.

| Feature | Symbol | Type | Range | Phase |
|---|---|---|---|---|
| Stationary Ratio | $R$ | Float | [0.0, 1.0] | Snapshot |
| Surrounding Vehicle Count | $C$ | Integer | $\geq 0$ | Snapshot |
| Mean Surrounding Displacement | $\bar{d}$ | Float | $\geq 0$ m | Snapshot |
| Surrounding Displacement Std. Dev. | $\sigma_d$ | Float | $\geq 0$ m | Snapshot |
| Ratio Change | $\Delta R$ | Float | [−1.0, +1.0] | Monitoring |
| Candidate Post-Trigger Displacement | $d_{\text{cand}}$ | Float | $\geq 0$ m | Monitoring |

*Table 3.3. Feature vector specification. Snapshot features are computed at the trigger frame. Monitoring features are computed after a 15-second observation window following the trigger.*

**Feature independence.** No feature is a linear combination of any other. $R$ is a proportion from binarized stationary classifications; $\bar{d}$ is a continuous mean of raw displacements; $\sigma_d$ is a second-order moment independent of $\bar{d}$; $\Delta R$ incorporates temporal information absent from the snapshot features; and $d_{\text{cand}}$ describes the candidate vehicle, while all other features describe surrounding traffic. This structure avoids the multicollinearity that would result from simultaneously including a pre-trigger ratio, a post-trigger ratio, and their arithmetic difference as separate features.

**Edge case.** If $|\mathcal{S}(t_{\text{trig}})| = 0$ (no surrounding vehicles are tracked at the trigger frame), the surrounding-vehicle features are undefined. In this case, the classifier is bypassed and the event is classified as a violation based solely on the temporal threshold, consistent with standard zone-and-threshold logic. Events classified via this fallback will be reported separately and excluded from classifier performance evaluation.

### 3.4.6 Violation Classification

The feature vector is processed by a gradient-boosted decision tree classifier, specifically XGBoost [citation: Chen and Guestrin, 2016], which outputs a binary prediction: 1 (violation) or 0 (non-violation).

XGBoost constructs an additive ensemble of decision trees, where each successive tree is trained to correct the residual prediction errors of the preceding ensemble. At each node, the algorithm evaluates candidate split points across all input features and selects the split that maximizes the reduction in the regularized objective function. The split thresholds are computed from the training data distribution via gradient optimization, not specified manually. This learned threshold property directly addresses the methodological limitation of heuristic decision architectures: the classifier derives its decision boundaries from labeled training examples rather than from researcher-imposed cutoff values.

The classifier receives only the six numeric features defined in Table 3.3. Event ID and Track ID are metadata identifiers retained by the system's orchestration layer for associating the classifier's output with the correct on-screen vehicle for flagging and logging. They are not passed to the classifier.

---

## 3.5 Ground Truth Annotation

### 3.5.1 Event-Level Annotation

Annotation is performed at the event level, not the frame level. The set of events to be annotated is determined by running the Vision Pipeline and Event Trigger stages on the recorded footage and collecting all generated trigger events.

For each trigger event, the annotator reviews the video segment from 30 seconds before the trigger frame through the end of the monitoring window. Table 3.4 defines the annotation fields.

| Field | Type | Description |
|---|---|---|
| Event ID | Integer | Matches trigger event output |
| Track ID | Integer | Tracked vehicle identity |
| Label | Binary | Violation or Non-Violation |
| Stop reason | Categorical | Congestion / Pedestrian yielding / Loading / Other / N/A |
| Notes | Free text | Annotator remarks on ambiguous cases |

*Table 3.4. Ground truth annotation fields. Stop reason is recorded for Non-Violation events only; Violation events are set to N/A.*

An event is labeled as a Violation if the vehicle remains stationary in the restricted zone throughout the observation period without an observable external cause (simultaneous stoppage of surrounding vehicles, active pedestrian crossing, or visible loading activity). If no external cause is observable, the event is labeled as a Violation. This default-to-violation rule follows from the legal framing of RA 4136: a vehicle present in a statutory no-parking zone is presumptively in violation unless observable circumstances demonstrate a legitimate reason for the stop.

### 3.5.2 Inter-Annotator Agreement

A randomly selected 20% subset of triggered events will be independently annotated by a second annotator. Agreement on the binary label will be measured using Cohen's kappa:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is the observed agreement and $p_e$ is the expected agreement by chance. A minimum $\kappa \geq 0.80$ is required [PENDING VERIFICATION: source for 0.80 kappa acceptability threshold]. If agreement falls below this threshold, disagreements will be adjudicated and the operational definition refined before re-annotation.

---

## 3.6 Training and Evaluation Protocol

### 3.6.1 Cross-Validation

Due to the anticipated limited size of the event dataset (constrained by the number of qualifying trigger events in the recording period), a fixed train-test split risks high variance in performance estimates. The study uses stratified 5-fold cross-validation. The annotated dataset is divided into five equal folds, stratified by label to preserve the class distribution in each fold. In each iteration, four folds serve as the training set and one fold as the held-out evaluation set. The process repeats five times, and metrics are reported as the mean ± standard deviation across folds.

It is noted that random fold assignment may place temporally adjacent events (which share correlated conditions such as lighting, traffic density, or the presence of staged scenarios) into different folds, potentially introducing a mild optimistic bias in cross-validated performance estimates. This temporal correlation is acknowledged as a limitation of the evaluation design; it is not addressed with temporal blocking because the anticipated dataset size does not support further partitioning constraints without unacceptable variance.

### 3.6.2 Hyperparameter Tuning

XGBoost hyperparameters are tuned via grid search with 3-fold cross-validation nested within each outer training fold. The search space is defined in Table 3.5.

| Hyperparameter | Search values |
|---|---|
| Number of estimators | 50, 100, 200 |
| Maximum tree depth | 3, 5, 7 |
| Learning rate ($\eta$) | 0.01, 0.1, 0.3 |
| Minimum child weight | 1, 3, 5 |

*Table 3.5. XGBoost hyperparameter search space. The combination yielding the highest mean F1-score on the inner cross-validation is selected for each outer fold.*

No feature scaling or normalization is applied, as gradient-boosted decision trees are invariant to monotonic feature transformations [citation].

### 3.6.3 Evaluation Metrics

Classification performance is evaluated using precision, recall, F1-score, and false positive rate, computed from the standard binary confusion matrix.

| Metric | Definition |
|---|---|
| Precision | $\frac{TP}{TP + FP}$ |
| Recall | $\frac{TP}{TP + FN}$ |
| F1-Score | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| False Positive Rate | $\frac{FP}{FP + TN}$ |

*Table 3.6. Evaluation metrics. TP: system correctly flags a genuine violation. FP: system incorrectly flags a legitimate stop. TN: system correctly suppresses a legitimate stop. FN: system fails to flag a genuine violation.*

Metrics for the proposed method are reported as mean ± standard deviation across the five cross-validation folds.

### 3.6.4 Comparative Evaluation

The proposed method will be compared against a reimplementation of the detection logic described by Abella and Catedrilla [citation], who developed a YOLOv8-based illegal parking detection system using composite detection classes. Their method performs frame-by-frame inference without multi-object tracking or temporal dwell-time logic; a violation is identified when a composite class representing the spatial co-occurrence of a vehicle, a human, and a no-parking sign is detected within a single frame.

The research team will reimplement Abella et al.'s composite-class detection logic and apply it to the same video footage used to evaluate the proposed method. Both methods will be tested under identical conditions:

- The same camera and recording setup
- The same recording resolution and frame rate
- The same YOLOv8 architecture
- The same ground truth event annotations

This controlled same-data design ensures that any measured performance difference is attributable to the decision method itself, not to differences in test conditions.

**Reimplementation procedure.** A YOLOv8 model will be trained on the composite class definitions described by Abella and Catedrilla. The reimplementation involves the following steps:

1. Training images will be collected and annotated by the research team at the test site following Abella et al.'s described annotation format (COCO-format bounding boxes).
2. The model will be trained to detect both base classes and composite classes:
   - **Base classes:** car, vehicle, motorcycle, person, no-parking sign
   - **Composite classes:** vehicle-human, vehicles-human-noparking sign
3. Where Abella et al.'s original paper does not specify an implementation detail (such as detection confidence threshold, input image size, or exact train-validation-test split ratio), the research team will apply standard defaults and document these choices explicitly, since the original paper does not report them.

**Event-level aggregation.** Abella et al.'s method produces frame-level detections, while the proposed method produces event-level classifications. To enable fair comparison, the reimplemented method's output must be aggregated to the event level. The observation window for each event spans from the trigger frame ($t_{\text{trig}}$) through the end of the monitoring period ($t_{\text{trig}} + W \times \text{fps}$). The aggregation procedure is as follows:

1. For each triggered event in the ground truth, check whether a composite violation class (vehicles-human-noparking sign) is detected in any frame within that event's observation window.
2. If at least one composite detection occurs, the reimplemented method classifies the event as a violation.
3. If no composite detection occurs in any frame within the window, the event is classified as a non-violation.

Both methods are then evaluated against the same event-level ground truth labels using the metrics defined in Table 3.6.

**Evaluation scope boundary.** The set of events evaluated in this comparison is defined by the proposed method's own trigger mechanism: only vehicles that have been stationary in the restricted zone for at least $T = 60$ seconds generate a trigger event and enter the ground truth annotation pool. This scoping has two consequences that must be stated explicitly. First, violations involving vehicles parked for fewer than 60 seconds are structurally invisible to the entire evaluation, as they never generate a candidate event. Second, the reimplemented Abella et al. method is evaluated only on this pre-filtered event pool, not on its independent full-footage detection behavior. The comparison therefore measures each method's discrimination ability within a candidate pool of extended-duration stationary vehicles, not each method's end-to-end detection coverage across all forms of illegal parking. This scope boundary is inherent to the proposed system's event-driven architecture and is acknowledged as a limitation of the comparative design.

**Ground truth methodological distinction.** Although both methods are evaluated on the same dataset under identical test conditions, a conceptual difference in violation definition exists between the two approaches:

- **Abella et al.** define a violation as the spatial co-occurrence of specific object classes within a single frame, with no temporal duration requirement.
- **Proposed method** defines a violation as a vehicle stationary within a legally grounded ROI for at least 60 seconds without an observable external cause.

This difference in what constitutes a violation is inherent to the two approaches and cannot be eliminated by testing on the same data. It will be explicitly discussed when interpreting comparative results.

### 3.6.5 Feature Importance and Ablation

**Feature importance.** For the trained XGBoost model, feature importance is measured using the average gain: the mean improvement in the objective function contributed by a feature across all trees in which it appears. Importance scores are reported as a ranked bar chart to provide interpretability into which aspects of the traffic context the classifier relies on most heavily.

[Figure 3.4 placeholder]

*Figure 3.4. Feature importance ranking from the trained XGBoost model (placeholder).*

**Ablation study.** To assess the contribution of individual feature groups, the classifier is retrained and evaluated under the following feature subsets using the same 5-fold cross-validation protocol:

| Configuration | Features |
|---|---|
| Snapshot features only | $R, \; C, \; \bar{d}, \; \sigma_d$ |
| Monitoring features only | $\Delta R, \; d_{\text{cand}}$ |
| Full feature vector | All six features |
| Full + Vehicle Class | All six features + candidate vehicle class (car, bus, truck, motorcycle) |

*Table 3.7. Ablation study configurations. Each configuration is evaluated using the full cross-validation protocol to assess the marginal contribution of snapshot features, monitoring features, and the optional vehicle class feature.*

### 3.6.6 Parameter Sensitivity Analysis

A sensitivity analysis examines the effect of key extraction parameters on the proposed method's performance. For each parameter, the full pipeline (feature extraction, classifier training, and 5-fold cross-validation) is re-executed with the modified value while all other parameters are held at their defaults.

| Parameter | Default | Test values |
|---|---|---|
| Monitoring window ($W$) | 15 s | 5, 10, 15, 20, 30 s |
| Stationary threshold ($\varepsilon_s$) | 1.5× noise ceiling | 0.5×, 1.0×, 1.5×, 2.0× noise ceiling |
| Lookback window ($N$) | 3 s | 1, 2, 3, 5 s |

*Table 3.8. Parameter sensitivity analysis. Results are reported as F1-score at each tested value.*

[Figure 3.5 placeholder]

*Figure 3.5. Parameter sensitivity curves showing F1-score as a function of each extraction parameter (placeholder).*

---

## 3.7 Ethical Considerations

The recorded footage captures public road scenes that may include identifiable vehicles and pedestrians. No personally identifiable information (license plate numbers, faces) will be extracted, stored, or used at any stage of the system, which processes only bounding box coordinates and class labels. Raw footage will be stored on encrypted local storage accessible only to the research team and will not be published or uploaded to any public repository. Any frames or screenshots included in the manuscript will have license plates and faces blurred. The study will comply with the university's research ethics requirements and with Republic Act No. 10173 (Data Privacy Act of 2012).

---

## 3.8 Summary of System Parameters

Table 3.9 consolidates all fixed and empirically determined parameters in the proposed system.

| Parameter | Symbol | Value | Justification |
|---|---|---|---|
| Resolution | — | 1920 × 1080 | Device specification |
| Frame rate | fps | 30 | Native recording rate; no downsampling |
| Temporal trigger threshold | $T$ | 60 s | Literature consensus [citations] |
| Lookback window | $N$ | 3 s (90 frames) | Sufficient for motion discrimination |
| Stationary threshold | $\varepsilon_s$ | Empirically determined | Measured pipeline noise on known-stationary vehicle |
| Monitoring window | $W$ | 15 s | Tunable; tested in sensitivity analysis |
| Detection confidence threshold | — | 0.25 | Prioritizes detection recall |
| IoU association threshold | $\sigma_{\text{IoU}}$ | 0.5 | [PENDING VERIFICATION: Bochinski et al., 2017] |

*Table 3.9. Complete system parameter summary.*

[Figure 3.6 placeholder]

*Figure 3.6. Complete system pipeline diagram with all parameters annotated at their respective processing stages.*
