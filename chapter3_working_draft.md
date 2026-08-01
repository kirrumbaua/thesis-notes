# Chapter 3: METHODOLOGY

## 3.1 Conceptual Framework

The proposed system processes fixed-camera video through a three-stage sequential pipeline to classify individual vehicle events as illegal parking violations or legitimate traffic stops. The first stage, the Vision Pipeline, performs continuous per-frame vehicle detection, multi-object tracking, and real-world coordinate transformation. The second stage, the Event Trigger and Feature Extraction module, monitors tracked vehicles within a legally defined restricted zone and, upon satisfaction of a temporal threshold, computes a structured feature vector encoding the surrounding traffic context and the candidate vehicle's post-trigger behavior. The third stage, the Event Classifier, processes the feature vector through a trained XGBoost model to produce a binary violation or non-violation prediction for each triggered event.

The feature extraction stage encodes the aggregate motion state of surrounding tracked vehicles into a numeric vector. The classifier utilizes this contextual data to discriminate between illegal parking and collective traffic stops.

[Figure 3.1 placeholder]

_Figure 3.1. Conceptual framework of the proposed context-aware spatiotemporal classification pipeline, showing the data flow from raw video input through the Vision Pipeline, Event Trigger and Feature Extraction, and Event Classifier stages to the final binary output._

---

## 3.2 Data Acquisition

### 3.2.1 Test Site

The recording location will be [test site name and description to be inserted upon site confirmation]. The selected site satisfies the following criteria:

1. A legally designated no-parking zone, established either by posted signage or by statutory restriction under Republic Act No. 4136 (Land Transportation and Traffic Code of the Philippines), which prohibits parking within an intersection, on a crosswalk, within six meters of a curb-line intersection, in front of a driveway, or in a manner that blocks a designated driving lane.
2. No traffic signal at the immediate recording location. Without a signal head in the camera's field of view, signal-state-based suppression methods [19] are inapplicable, necessitating reliance on the dynamic behavior of surrounding vehicles.
3. Sufficient traffic volume during the recording period to ensure that multiple vehicles are simultaneously visible for the majority of recording time.
4. A physical vantage point permitting elevated camera placement to minimize inter-vehicle occlusion.

The restricted zone (Region of Interest) is defined as a closed polygon drawn once onto the camera's coordinate space during setup. The polygon boundaries are spatially grounded in the statutory definitions of RA 4136, eliminating dependence on the visibility of physical paint or signage for zone definition.

### 3.2.2 Recording Configuration

Video footage will be recorded using an Apple iPhone 16 Pro mounted on a tripod at a fixed, elevated position overlooking the test site. Table 3.1 summarizes the recording parameters.

| Parameter        | Value                                                                              |
| ---------------- | ---------------------------------------------------------------------------------- |
| Device           | Apple iPhone 16 Pro (main camera, 24 mm equiv.)                                    |
| Resolution       | 1920 × 1080 pixels                                                                 |
| Frame rate       | 30 fps                                                                             |
| Recording mode   | Standard video; HDR, Cinematic Mode, Action Mode, and digital zoom disabled        |
| Mounting         | Tripod, fixed position, no operator contact during recording                       |
| Height           | [To be measured at site, in meters from road surface to lens]                      |
| Tilt angle       | [To be measured at site, in degrees from horizontal]                               |
| Recording period | [Date(s) to be confirmed], 16:00 to 18:00 local time. Precipitation explicitly excluded. |

_Table 3.1. Video recording parameters._

The camera will remain stationary throughout each recording session. No post-capture image processing will be applied to the raw footage. All recording parameters will be held constant across sessions to ensure geometric consistency for calibration. Furthermore, recording is strictly restricted to clear or overcast weather conditions; precipitation scenarios are explicitly excluded from the scope, as rain introduces severe visual noise that fundamentally degrades the stability of lightweight IoU-based bounding box tracking. If ambient light conditions degrade significantly during the recording window, such as when sunset occurs before 18:00 at the test site's latitude, the recording session will be terminated early, as reduced illumination degrades YOLOv8n detection recall and would systematically undercount surrounding vehicles.

The recorded footage must contain sufficient events across four scenario categories: free-flowing traffic (vehicles passing without stopping), collective stops (multiple vehicles stationary simultaneously due to congestion or pedestrian yielding), genuine violations (vehicles illegally parked in the restricted zone), and mixed traffic (transitions between flowing and stopped states). If insufficient violation events occur naturally, controlled scenarios will be staged using a designated vehicle parked deliberately in the restricted zone while surrounding traffic continues. The proportion of staged events will be reported. Staged events may exhibit different surrounding traffic patterns compared to naturally occurring violations; this is a limitation of the data collection procedure.

### 3.2.3 Camera Calibration

Accurate displacement computation requires two sequential calibration steps: lens distortion correction and ground-plane homography estimation. Both are performed once per recording setup.

#### Intrinsic Calibration

The iPhone camera lens introduces radial and tangential distortion that causes straight lines to appear curved, introducing systematic error into geometric measurements. The intrinsic parameters will be estimated using the standard checkerboard calibration method implemented in OpenCV [55].

A printed checkerboard pattern (9 × 6 internal corners, with a square size measured and reported in millimeters) affixed to a rigid surface will be captured in a minimum of 20 images at varying orientations and distances, using the recording configuration in Table 3.1. Corner detection (`cv2.findChessboardCorners` with sub-pixel refinement via `cv2.cornerSubPix`) and camera calibration (`cv2.calibrateCamera`) will yield the camera intrinsic matrix $\mathbf{K}$ and distortion coefficients $\mathbf{D}$:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{D} = (k_1, k_2, p_1, p_2, k_3)$$

where $f_x, f_y$ are the focal lengths in pixels, $(c_x, c_y)$ is the principal point, $k_1, k_2, k_3$ are radial distortion coefficients, and $p_1, p_2$ are tangential distortion coefficients. All video frames will be undistorted using `cv2.undistort` before any downstream processing. The calibration quality is assessed by the reprojection error: the root mean square (RMS) of the Euclidean distance between detected checkerboard corners and their corresponding projected positions under the estimated camera model [56]. A reprojection error below 1.0 pixel is a widely adopted acceptability threshold in planar-pattern camera calibration practice [55, 56].

#### Homography Estimation

A planar homography maps undistorted pixel coordinates to real-world ground-plane coordinates in meters, under the assumption that all mapped points lie on the road surface plane.

At the test site, a minimum of four non-collinear reference points will be placed on the road surface at positions measured with a tape measure, using high-visibility markers to form a polygon of known dimensions. The pixel coordinates $(x_i, y_i)$ of each reference point in the undistorted frame and their corresponding real-world coordinates $(X_i, Y_i)$ in meters will be used to compute the 3 × 3 homography matrix $\mathbf{H}$ via `cv2.findHomography` with RANSAC outlier rejection.

The transformation maps any undistorted pixel coordinate $(x, y)$ to ground-plane coordinates $(X, Y)$:

$$\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad X = \frac{X'}{w}, \quad Y = \frac{Y'}{w}$$

The homography will be validated by computing the Euclidean distance between a minimum of four additional reference points not used in the estimation, comparing direct tape-measured positions against homography-derived coordinates. Reporting multiple validation points enables assessment of directional bias in the homography, such as systematic stretching along one axis. The mean and maximum absolute position error will be reported; if the maximum error exceeds 0.10 meters, recalibration will be performed.

---

## 3.3 Proposed Method

The system architecture consists of three sequential processing stages. Each stage's output serves as the direct input to the next.

[Figure 3.2 placeholder]

_Figure 3.2. System architecture showing the three-stage pipeline: Vision Pipeline (per-frame detection, tracking, coordinate transformation), Event Trigger and Feature Extraction (event-driven), and Event Classifier (per-event binary prediction)._

### 3.3.1 Vehicle Detection

Object detection is performed using YOLOv8n (Nano), the smallest variant of the YOLOv8 architecture [57], with 3.2 million parameters and 8.7 GFLOPs per inference pass. The model uses pretrained MS COCO weights [58] without fine-tuning. Only vehicle classes are retained: car (class 2), motorcycle (3), bus (5), and truck (7). All other detections are discarded.

A detection confidence threshold of 0.25 is applied, corresponding to the Ultralytics YOLOv8 default inference threshold [57]. This value prioritizes detection recall to prevent undercounting the surrounding traffic population, which would directly distort the Stationary Ratio $R$. The detection module provides standardized bounding box inputs to the subsequent tracking stage.

The Ultralytics YOLOv8 implementation outputs bounding box coordinates in the original image space (1920 × 1080), independent of the internal input resize, so no manual coordinate rescaling is required before homography transformation.

### 3.3.2 Multi-Object Tracking

A multi-object tracker assigns and maintains a persistent Track ID for each vehicle across consecutive frames. The tracker uses IoU-based (Intersection over Union) frame-to-frame association [54], selected for its minimal computational overhead and its suitability for high-frame-rate fixed-camera scenarios where inter-frame vehicle displacement is small relative to bounding box size.

The tracking procedure operates as follows. At each frame, the set of current detections is compared against all active tracks from the previous frame. For each detection-track pair, the IoU between the detection's bounding box and the track's last known bounding box is computed:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where $A$ and $B$ are the areas of the two bounding boxes. Detection-to-track assignment is performed by selecting the assignment that maximizes total IoU, subject to a minimum IoU threshold $\sigma_{\text{IoU}} = 0.5$. Bochinski et al. [54] define $\sigma_{\text{IoU}}$ as a tunable parameter that should be set in the same range as the detector's non-maximum suppression (NMS) threshold. The Ultralytics YOLOv8 NMS IoU threshold defaults to 0.45 [57]; $\sigma_{\text{IoU}} = 0.5$ is adopted as a standard threshold value that requires detections to overlap at least half of their combined area with the existing track for association continuity. Detections with IoU below this threshold against all existing tracks are held as tentative for a confirmation period of $n_{\text{init}} = 3$ consecutive frames before being promoted to active tracks. This confirmation requirement prevents single-frame false-positive detections from immediately entering the surrounding vehicle set $\mathcal{S}(t)$ and corrupting the Stationary Ratio $R$. Active tracks not matched to any detection for $t_{\text{lost}} = 5$ consecutive frames (approximately 0.17 seconds at 30 fps) are terminated. This window is long enough to bridge momentary single-frame detection misses but short enough to avoid maintaining ghost tracks for vehicles that have left the scene. Terminated Track IDs are not reused.

At 30 fps with a fixed camera, vehicles undergo small inter-frame displacements relative to their bounding box dimensions, producing high IoU overlap between consecutive frames. This high frame-to-frame overlap makes IoU-based association sufficient for maintaining track identity, avoiding the computational overhead of appearance descriptors or motion prediction models.

| Output       | Type           | Description                         |
| ------------ | -------------- | ----------------------------------- |
| Track ID     | Integer        | Persistent identifier across frames |
| Bounding box | $(x, y, w, h)$ | Position and dimensions in pixels   |
| Class label  | Integer        | COCO class ID                       |
| Confidence   | Float          | Detection score                     |

_Table 3.2. Per-frame tracking outputs._

### 3.3.3 Coordinate Transformation and Displacement Computation

For each tracked vehicle at each frame, the bottom-center point of the bounding box is extracted:

$$p_{\text{bottom}} = \left(x + \frac{w}{2}, \; y + h\right)$$

The bottom-center approximates the vehicle's contact point with the road surface, which lies on the ground plane mapped by the homography. Using the bounding box centroid (vertical center) would introduce height-dependent projection error for off-nadir viewing angles. Under heavy physical occlusion from intervening guardrails, vegetation, or passing large vehicles, the visible bottom-center of the bounding box is elevated relative to the true road contact point. Projecting this elevated point through the homography matrix induces real-world displacement errors. Accurate displacement computation thus requires full ground-contact visibility.

The bottom-center pixel coordinates are transformed to real-world ground-plane coordinates $(X, Y)$ in meters using the calibrated homography $\mathbf{H}$ (Section 3.3.3).

The displacement of vehicle $v$ at frame $t$ is defined as the Euclidean distance between its current position and its position $n$ frames earlier:

$$d(v, t) = \sqrt{(X_t - X_{t-n})^2 + (Y_t - Y_{t-n})^2}$$

where $n = N \times \text{fps}$. The lookback window is set to $N = 3$ seconds ($n = 90$ frames at 30 fps). At this window length, a vehicle moving at a minimal walking speed of approximately 1.4 m/s [59] covers 4.2 meters, well above any stationary threshold, while a genuinely stopped vehicle shows displacement attributable only to detection noise. If a track has existed for fewer than $n$ frames, displacement is computed from the track's earliest available frame. However, vehicles with fewer than $n$ frames of tracking history are excluded from the surrounding vehicle set $\mathcal{S}(t)$ used in feature extraction (Section 3.4.5), since their displacement is computed over a shorter window than the one for which $\varepsilon_s$ was calibrated, which could produce unreliable stationary classifications.

### 3.3.4 Restricted Zone and Event Trigger

The restricted zone is specified as a closed polygon in pixel coordinates, drawn once onto the undistorted frame during setup. Its boundaries are spatially grounded in the statutory provisions of RA 4136 at the test site. Zone membership is evaluated per-frame using the ray casting point-in-polygon algorithm applied to each vehicle's bottom-center point.

[Figure 3.3 placeholder]

_Figure 3.3. Example restricted zone polygon overlaid on an undistorted frame from the test site._

Within this zone, a vehicle $v$ at frame $t$ is classified as stationary if its displacement falls below the stationary threshold $\varepsilon_s(Y_t)$ mapped to its current depth:

$$\text{stationary}(v, t) = \begin{cases} 1 & \text{if } d(v, t) < \varepsilon_s(Y_t) \\ 0 & \text{otherwise} \end{cases}$$

The depth-dependent threshold $\varepsilon_s(Y_t)$ will be determined empirically from the study's own pipeline. Because the homography maps pixels to meters under perspective projection, the same amount of pixel-level detection jitter corresponds to a larger real-world displacement for vehicles farther from the camera than for those closer to it. To account for this, calibration will be performed at three positions within the restricted zone: near-field, mid-field, and far-field. At each position, a vehicle known to be completely stationary will be recorded for a minimum of 60 seconds. Using a single maximum noise ceiling across all depths would apply large far-field perspective jitter to near-field vehicles, resulting in slow-creeping near-field vehicles being misclassified as stationary. Instead, $\varepsilon_s(Y_t)$ will be defined as a linear interpolation scaled by the real-world $Y$-coordinate between the calibrated depth positions, set to 1.5 times the empirically measured noise at that specific depth. Linear interpolation is an approximation of the true inverse-perspective relationship between pixel-level noise and real-world displacement; the 1.5× multiplier provides a safety margin against this approximation error. Both the per-position noise measurements and the depth scaling function will be explicitly reported.

To initiate an evaluation, a per-vehicle timer monitors each tracked vehicle whose bottom-center lies inside the restricted zone. The timer increments by one frame interval ($1/\text{fps}$ seconds) at each frame where $d(v, t) < \varepsilon_s$, and resets to zero if $d(v, t) \geq \varepsilon_s$. When the timer reaches the temporal threshold $T = 60$ seconds, a trigger event is generated for that Track ID. Each Track ID generates at most one trigger event.

The 60-second threshold is adopted from the majority of fixed-camera parking detection studies reviewed in Chapter 2 [8, 18, 29, 36]. At the unsignalized test site, legitimate stationary stops (yielding to pedestrians, momentary queuing) are expected to be well below this duration under normal conditions.

### 3.3.5 Feature Extraction

Upon triggering, the system extracts a six-dimensional feature vector in two phases: four features are computed as a snapshot at the trigger frame $t_{\text{trig}}$, and two features are computed after a monitoring window of $W = 15$ seconds following the trigger. Table 3.3 defines the complete feature vector.

Let $\mathcal{S}(t)$ denote the set of all tracked vehicles at frame $t$ excluding the candidate vehicle and excluding any vehicle with fewer than $n$ frames of tracking history (see Section 3.4.3).

The first phase computes snapshot features at the trigger frame $t_{\text{trig}}$. The Stationary Ratio $R$ measures the proportion of surrounding vehicles classified as stationary:

$$R(t) = \frac{\sum_{v_i \in \mathcal{S}(t)} \text{stationary}(v_i, t)}{|\mathcal{S}(t)|}$$

The Surrounding Vehicle Count $C = |\mathcal{S}(t_{\text{trig}})|$ records the number of tracked vehicles visible at the trigger frame, excluding the candidate. This contextualizes the statistical reliability of $R$: a ratio of 0.50 computed from 2 vehicles carries less evidential weight than the same ratio computed from 20 vehicles. Vehicles with fewer than $n$ frames of tracking history are excluded from $\mathcal{S}(t)$ (Section 3.4.3). Consequently, $C$ temporarily undercounts the true number of surrounding vehicles during high traffic turnover. This approach prevents the misclassification of short-tracked moving vehicles as stationary, which would otherwise artificially inflate $R$.

The Mean Surrounding Displacement $\bar{d}$ provides a continuous measure of surrounding traffic flow intensity, preserving the magnitude of vehicle movement that the binary stationary classification discards:

$$\bar{d} = \frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} d(v_i, t_{\text{trig}})$$

The Surrounding Displacement Standard Deviation $\sigma_d$ captures the uniformity of traffic behavior. Low values indicate that surrounding vehicles are behaving similarly (all stopped or all flowing). High values indicate mixed behavior (some moving, some stopped), a condition distinct from both uniform flow and uniform stoppage:

$$\sigma_d = \sqrt{\frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} \left(d(v_i, t_{\text{trig}}) - \bar{d}\right)^2}$$

The second phase computes post-monitoring features after a monitoring window of $W = 15$ seconds ($t_{\text{trig}} + W \times \text{fps}$). The Ratio Change $\Delta R$ captures the temporal evolution of the surrounding traffic state:

$$\Delta R = R(t_{\text{trig}} + W \times \text{fps}) - R(t_{\text{trig}})$$

Negative values indicate that surrounding traffic cleared during the monitoring period. Near-zero values indicate a stable traffic state. If all surrounding vehicles leave during the monitoring window $W$ (such that $|\mathcal{S}(t_{\text{trig}} + W \times \text{fps})| = 0$), the post-trigger ratio suffers a division-by-zero error; in this edge case, $\Delta R$ is encoded as a null value, which the XGBoost classifier naturally handles via its default sparsity-aware split finding algorithm.

The Candidate Post-Trigger Displacement $d_{\text{cand}}$ measures whether the candidate vehicle itself moved during the monitoring window:

$$d_{\text{cand}} = \sqrt{(X_{t_{\text{end}}} - X_{t_{\text{trig}}})^2 + (Y_{t_{\text{end}}} - Y_{t_{\text{trig}}})^2}$$

where $t_{\text{end}}$ is $t_{\text{trig}} + W \times \text{fps}$, or the frame of the candidate vehicle's last known coordinate if its track terminates before the monitoring window concludes. This ensures $d_{\text{cand}}$ remains defined when a candidate departs early. This is the only feature describing the candidate vehicle's own behavior; all others describe surrounding traffic.

| Feature                             | Symbol            | Type    | Range        | Phase      |
| ----------------------------------- | ----------------- | ------- | ------------ | ---------- |
| Stationary Ratio                    | $R$               | Float   | [0.0, 1.0]   | Snapshot   |
| Surrounding Vehicle Count           | $C$               | Integer | $\geq 0$     | Snapshot   |
| Mean Surrounding Displacement       | $\bar{d}$         | Float   | $\geq 0$ m   | Snapshot   |
| Surrounding Displacement Std. Dev.  | $\sigma_d$        | Float   | $\geq 0$ m   | Snapshot   |
| Ratio Change                        | $\Delta R$        | Float   | [−1.0, +1.0] | Monitoring |
| Candidate Post-Trigger Displacement | $d_{\text{cand}}$ | Float   | $\geq 0$ m   | Monitoring |

_Table 3.3. Feature vector specification. Snapshot features are computed at the trigger frame. Monitoring features are computed after a 15-second observation window following the trigger._

Regarding feature completeness, the six-feature vector omits purely spatial features, such as distance to the lane edge or sidewalk. Incorporating spatial differentiation would require explicit lane-detection algorithms, which contradicts the system's reliance on lightweight bounding box tracking. The feature vector focuses exclusively on capturing the aggregate temporal traffic state.

To ensure feature independence, no feature is a linear combination of any other. $R$ is a proportion from binarized stationary classifications; $\bar{d}$ is a continuous mean of raw displacements; $\sigma_d$ is a second-order moment independent of $\bar{d}$; $\Delta R$ incorporates temporal information absent from the snapshot features; and $d_{\text{cand}}$ describes the candidate vehicle, while all other features describe surrounding traffic. This structure avoids the multicollinearity that would result from simultaneously including a pre-trigger ratio, a post-trigger ratio, and their arithmetic difference as separate features.

In the edge case where $|\mathcal{S}(t_{\text{trig}})| = 0$ (no surrounding vehicles are tracked at the trigger frame), the surrounding-vehicle features are undefined. In this scenario, the classifier is bypassed and the event is classified as a violation based solely on the temporal threshold, consistent with standard zone-and-threshold logic. Events classified via this fallback will be reported separately and excluded from classifier performance evaluation.

### 3.3.6 Violation Classification

The feature vector is processed by a gradient-boosted decision tree classifier, specifically XGBoost [60], which outputs a binary prediction: 1 (violation) or 0 (non-violation).

XGBoost constructs an additive ensemble of decision trees, where each successive tree is trained to correct the residual prediction errors of the preceding ensemble. At each node, the algorithm evaluates candidate split points across all input features and selects the split that maximizes the reduction in the regularized objective function. The split thresholds are computed from the training data distribution via gradient optimization, establishing data-driven decision boundaries rather than relying on heuristic cutoff values.

The classifier receives only the six numeric features defined in Table 3.3. Event ID and Track ID are metadata identifiers retained by the system's orchestration layer for associating the classifier's output with the correct on-screen vehicle for flagging and logging. They are not passed to the classifier.

The context-aware classifier operates under identifiable structural limitations. First, when only one surrounding vehicle is tracked at the trigger frame ($C = 1$), the Stationary Ratio $R$ degenerates to a binary value $\in \{0, 1\}$, the Mean Surrounding Displacement $\bar{d}$ reduces to a single observation, and the Surrounding Displacement Standard Deviation $\sigma_d$ is trivially zero. Under these conditions, the contextual features carry minimal discriminative information, and the classifier must rely primarily on $d_{\text{cand}}$ and $\Delta R$. The classifier's discrimination ability therefore degrades monotonically as $C$ decreases toward one. Second, an isolated legitimate stop, such as a lone vehicle yielding to a pedestrian during low-traffic periods, yields features identical to an illegally parked vehicle ($R = 0$, $\Delta R = 0$, $d_{\text{cand}} = 0$). Third, under extended gridlock lasting longer than the combined monitoring period ($T + W = 75$ seconds), no vehicles move during the window ($R = 1$, $\Delta R = 0$, $d_{\text{cand}} = 0$). In these scenarios, the spatiotemporal signatures of congestion and illegal parking become mathematically identical within the fixed temporal window. Fourth, on high-speed roads with rapid vehicle turnover, the $n$-frame tracking-history filter (Section 3.4.3) may exclude a large proportion of short-tracked vehicles from $\mathcal{S}(t)$, causing $C$ to undercount the actual traffic density. In extreme cases, this could produce artificially low $C$ values or trigger the zero-surrounding-vehicle fallback during what is actually moderate-density traffic with high turnover.

### 3.3.7 System Parameter Summary

Table 3.4 consolidates all fixed and empirically determined parameters in the proposed system.

| Parameter                      | Symbol                | Value                  | Justification                                       |
| ------------------------------ | --------------------- | ---------------------- | --------------------------------------------------- |
| Resolution                     | —                     | 1920 × 1080            | Device specification                                |
| Frame rate                     | fps                   | 30                     | Native recording rate; no downsampling              |
| Temporal trigger threshold     | $T$                   | 60 s                   | Literature consensus [8, 18, 29, 36] |
| Lookback window                | $N$                   | 3 s (90 frames)        | Motion at 1.4 m/s covers 4.2 m in 3 s [59] |
| Stationary threshold           | $\varepsilon_s$       | Empirically determined | Depth-dependent pipeline noise calibration (Section 3.4.4) |
| Monitoring window              | $W$                   | 15 s                   | Upstream extraction parameter; sensitivity assessed in Section 3.6.6 |
| Detection confidence threshold | —                     | 0.25                   | Ultralytics YOLOv8 default [57] |
| IoU association threshold      | $\sigma_{\text{IoU}}$ | 0.5                    | Adopted from NMS threshold range [54, 57] |
| Track confirmation window      | $n_{\text{init}}$     | 3 frames               | Empirical default; prevents single-frame false positives |
| Track termination window       | $t_{\text{lost}}$     | 5 frames               | Bridges momentary misses; avoids ghost tracks (Section 3.4.2) |

_Table 3.4. Complete system parameter summary._

[Figure 3.4 placeholder]

_Figure 3.4. Complete system pipeline diagram with all parameters annotated at their respective processing stages._

---

## 3.4 Ground Truth Annotation

### 3.4.1 Event-Level Annotation

Annotation is performed at the event level, not the frame level. The set of events to be annotated is determined by running the Vision Pipeline and Event Trigger stages on the recorded footage and collecting all generated trigger events.

For each trigger event, the annotator reviews the video segment from 30 seconds before the trigger frame through the end of the monitoring window. Table 3.5 defines the annotation fields.

| Field       | Type        | Description                                              |
| ----------- | ----------- | -------------------------------------------------------- |
| Event ID    | Integer     | Matches trigger event output                             |
| Track ID    | Integer     | Tracked vehicle identity                                 |
| Label       | Binary      | Violation or Non-Violation                               |
| Stop reason | Categorical | Congestion / Pedestrian yielding / Loading / Other / N/A |
| Notes       | Free text   | Annotator remarks on ambiguous cases                     |

_Table 3.5. Ground truth annotation fields. Stop reason is recorded for Non-Violation events only; Violation events are set to N/A._

An event is labeled as a Violation if the vehicle remains stationary in the restricted zone throughout the observation period without an observable external cause (simultaneous stoppage of surrounding vehicles, active pedestrian crossing, or visible loading activity). If no external cause is observable, the event is labeled as a Violation. This default-to-violation rule follows from the legal framing of RA 4136: a vehicle present in a statutory no-parking zone is presumptively in violation unless observable circumstances demonstrate a legitimate reason for the stop.

### 3.4.2 Inter-Annotator Agreement

A randomly selected 20% subset of triggered events will be independently annotated by a second annotator. Agreement on the binary label will be measured using Cohen's kappa:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is the observed agreement and $p_e$ is the expected agreement by chance. A minimum $\kappa \geq 0.80$ is targeted, corresponding to the upper boundary of "Substantial" agreement (0.61–0.80) and the threshold for "Almost Perfect" agreement (0.81–1.00) [61]. If agreement falls below this threshold, disagreements will be adjudicated and the operational definition refined before re-annotation.

---

## 3.5 Training and Evaluation Protocol

### 3.5.1 Cross-Validation

Due to the anticipated limited size of the event dataset (constrained by the number of qualifying trigger events in the recording period), a fixed train-test split risks high variance in performance estimates. The study uses stratified 5-fold cross-validation. The annotated dataset is divided into five equal folds, stratified by label to preserve the class distribution in each fold. In each iteration, four folds serve as the training set and one fold as the held-out evaluation set. The process repeats five times, and metrics are reported as the mean ± standard deviation across folds.

Random fold assignment may place temporally adjacent events into different folds, potentially introducing optimistic bias due to correlated environmental conditions such as lighting or traffic density. This limitation is retained as the dataset size is insufficient to support temporal blocking without introducing unacceptable variance.

### 3.5.2 Hyperparameter Tuning

XGBoost hyperparameters are tuned via grid search with 3-fold cross-validation nested within each outer training fold. The search space is defined in Table 3.6.

| Hyperparameter         | Search values  |
| ---------------------- | -------------- |
| Number of estimators   | 50, 100, 200   |
| Maximum tree depth     | 3, 5, 7        |
| Learning rate ($\eta$) | 0.01, 0.1, 0.3 |
| Minimum child weight   | 1, 3, 5        |

_Table 3.6. XGBoost hyperparameter search space. The combination yielding the highest mean F1-score on the inner cross-validation is selected for each outer fold._

No feature scaling or normalization is applied, as gradient-boosted decision trees are invariant to monotonic feature transformations: split decisions are based on threshold comparisons over feature values, and the relative ordering of data points is preserved under any monotonic rescaling [60].

### 3.5.3 Evaluation Metrics

Classification performance is evaluated using precision, recall, F1-score, and false positive rate, computed from the standard binary confusion matrix.

| Metric              | Definition                                                                              |
| ------------------- | --------------------------------------------------------------------------------------- |
| Precision           | $\frac{TP}{TP + FP}$                                                                    |
| Recall              | $\frac{TP}{TP + FN}$                                                                    |
| F1-Score            | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| False Positive Rate | $\frac{FP}{FP + TN}$                                                                    |

_Table 3.7. Evaluation metrics. TP: system correctly flags a genuine violation. FP: system incorrectly flags a legitimate stop. TN: system correctly suppresses a legitimate stop. FN: system fails to flag a genuine violation._

Metrics for the proposed method are reported as mean ± standard deviation across the five cross-validation folds. Precision, recall, and F1-score are reported for each class individually (violation and non-violation) to ensure that performance differences are not masked by class-level aggregation.

### 3.5.4 Comparative Evaluation

The proposed method will be compared against a reimplementation of the detection logic described by Abella and Catedrilla [1], who developed a YOLOv8-based illegal parking detection system using composite detection classes. Their method performs frame-by-frame inference without multi-object tracking or temporal dwell-time logic; a violation is identified when a composite class representing the spatial co-occurrence of a vehicle, a human, and a no-parking sign is detected within a single frame.

The research team will reimplement Abella et al.'s composite-class detection logic and apply it to the same video footage used to evaluate the proposed method. Both methods will be tested under identical conditions:

- The same camera and recording setup
- The same recording resolution and frame rate
- The same YOLOv8 architecture
- The same ground truth event annotations

This controlled same-data design ensures that any measured performance difference is attributable to the decision method itself, not to differences in test conditions.

For the reimplementation procedure, a YOLOv8 model will be trained on the composite class definitions described by Abella and Catedrilla. The reimplementation involves the following steps:

1. Training images will be collected and annotated by the research team at the test site following Abella et al.'s described annotation format (COCO-format bounding boxes).
2. The model will be trained to detect both base classes and composite classes:
   - **Base classes:** car, vehicle, motorcycle, person, no-parking sign
   - **Composite classes:** vehicle-human, vehicles-human-noparking sign
3. Where Abella et al.'s original paper does not specify an implementation detail (such as detection confidence threshold, input image size, or exact train-validation-test split ratio), the research team will apply standard defaults and document these choices explicitly, since the original paper does not report them.

Because Abella et al.'s method produces frame-level detections, while the proposed method produces event-level classifications, the reimplemented method's output must be aggregated to the event level to enable fair comparison. The observation window for each event spans from the trigger frame ($t_{\text{trig}}$) through the end of the monitoring period ($t_{\text{trig}} + W \times \text{fps}$). The aggregation procedure is as follows:

1. For each triggered event in the ground truth, check whether a composite violation class (vehicles-human-noparking sign) is detected in any frame within that event's observation window.
2. If at least one composite detection occurs, the reimplemented method classifies the event as a violation.
3. If no composite detection occurs in any frame within the window, the event is classified as a non-violation.

Both methods are then evaluated against the same event-level ground truth labels using the metrics defined in Table 3.7.

The evaluation scope for this comparison is strictly defined by the proposed method's trigger mechanism: only vehicles that have been stationary in the restricted zone for at least $T = 60$ seconds generate an event. Consequently, violations shorter than 60 seconds are excluded from the evaluation. Furthermore, the reimplemented Abella et al. method is evaluated only on this pre-filtered event pool. The comparison measures discrimination ability within a subset of extended-duration stationary vehicles rather than end-to-end detection coverage.

Furthermore, this comparison against Abella et al. is not a test of general algorithmic superiority, given the architectural differences between a tracked spatiotemporal classifier and an untracked single-frame detector. The specific hypothesis being tested is that context-unaware spatial co-occurrence models exhibit elevated false positive rates under extended collective traffic conditions, and that the proposed spatiotemporal classifier reduces this specific failure mode. The primary outcome metric for this hypothesis is the false positive rate (FPR). Abella et al. reported lower precision for parking detection (74.67%) compared to littering (98.41%), and mathematical deduction applied to their reported figures (2,803 total detections, 509 combined false positives across both classes with no per-class breakdown) indicates that the majority of false positives originate from the parking class. This is a deduction made by applying arithmetic to Abella et al.'s own disclosed numbers; it is not a conclusion stated in their paper. The reimplementation tests whether these parking-class false positives are attributable to collective traffic stops, which the proposed spatiotemporal classifier is designed to suppress.

Finally, a methodological distinction exists in the ground truth definitions between the two approaches, although both methods are evaluated on the same dataset under identical test conditions:

- **Abella et al.** define a violation as the spatial co-occurrence of specific object classes within a single frame, with no temporal duration requirement.
- **Proposed method** defines a violation as a vehicle stationary within a legally grounded ROI for at least 60 seconds without an observable external cause.

This difference in what constitutes a violation is inherent to the two approaches and cannot be eliminated by testing on the same data. It will be explicitly discussed when interpreting comparative results.

### 3.5.5 Feature Importance and Ablation

To interpret the trained XGBoost model, feature importance is measured using the average gain: the mean improvement in the objective function contributed by a feature across all trees in which it appears. Importance scores are reported as a ranked bar chart to provide interpretability into which aspects of the traffic context the classifier relies on most heavily.

[Figure 3.5 placeholder]

_Figure 3.5. Feature importance ranking from the trained XGBoost model (placeholder)._

An ablation study is conducted to assess the contribution of individual feature groups, where the classifier is retrained and evaluated under the following feature subsets using the same 5-fold cross-validation protocol:

| Configuration            | Features                                                                 |
| ------------------------ | ------------------------------------------------------------------------ |
| Snapshot features only   | $R, \; C, \; \bar{d}, \; \sigma_d$                                       |
| Monitoring features only | $\Delta R, \; d_{\text{cand}}$                                           |
| Full feature vector      | All six features                                                         |

_Table 3.8. Ablation study configurations. Each configuration is evaluated using the full cross-validation protocol to assess the marginal contribution of snapshot features and monitoring features._

### 3.5.6 Parameter Sensitivity Analysis

A sensitivity analysis examines the effect of key extraction parameters on the proposed method's performance. For each parameter, the full pipeline (feature extraction, classifier training, and 5-fold cross-validation) is re-executed with the modified value while all other parameters are held at their defaults.

| Parameter                              | Default            | Test values                          |
| -------------------------------------- | ------------------ | ------------------------------------ |
| Monitoring window ($W$)                | 15 s               | 5, 10, 15, 20, 30 s                  |
| Stationary threshold ($\varepsilon_s$) | 1.5× noise ceiling | 0.5×, 1.0×, 1.5×, 2.0× noise ceiling |
| Lookback window ($N$)                  | 3 s                | 1, 2, 3, 5 s                         |

_Table 3.9. Parameter sensitivity analysis. Results are reported as F1-score at each tested value._

[Figure 3.6 placeholder]

_Figure 3.6. Parameter sensitivity curves showing F1-score as a function of each extraction parameter (placeholder)._

---

## 3.6 Ethical Considerations

The recorded footage captures public road scenes that may include identifiable vehicles and pedestrians. No personally identifiable information (license plate numbers, faces) will be extracted, stored, or used at any stage of the system, which processes only bounding box coordinates and class labels. Raw footage will be stored on encrypted local storage accessible only to the research team and will not be published or uploaded to any public repository. Any frames or screenshots included in the manuscript will have license plates and faces blurred. The study will comply with the university's research ethics requirements and with Republic Act No. 10173 (Data Privacy Act of 2012).


