# Chapter 3: METHODOLOGY

## 3.1 Conceptual Framework

The proposed system processes fixed-camera video through a three-stage sequential pipeline to classify individual vehicle events as illegal parking violations or legitimate traffic stops. In the initial stage, designated as the Vision Pipeline, the framework performs continuous per-frame vehicle detection, multi-object tracking, and real-world ground-plane coordinate transformation. In the second stage, the Event Trigger and Feature Extraction module monitors tracked vehicles within a legally defined restricted zone and, upon satisfaction of a temporal threshold, extracts a structured numeric feature vector that encodes the kinematic motion state of surrounding traffic alongside the candidate vehicle's post-trigger behavior. In the final stage, the Event Classifier processes this feature vector through a trained gradient-boosted decision tree model (XGBoost) to generate an automated binary prediction (violation or non-violation) for each triggered event, leveraging ambient collective context to filter out false alarms caused by traffic congestion or signal stoppages.

[Figure 3.1 placeholder]

_Figure 3.1. Conceptual framework of the proposed spatiotemporal classification pipeline._

---

## 3.2 Data Acquisition

### 3.2.1 Test Site

The recording location will be [test site name and description to be inserted upon site confirmation]. The selected site must satisfy four primary criteria to ensure the structural validity of the dataset. First, it must encompass a legally designated no-parking zone, established either by posted signage or by statutory restriction under Republic Act No. 4136 (Land Transportation and Traffic Code of the Philippines), which prohibits parking within an intersection, on a crosswalk, within six meters of a curb-line intersection, in front of a driveway, or in a manner that blocks a designated driving lane. Second, the immediate recording location must lack a traffic signal. Because signal-state-based suppression methods [19] require a signal head within the camera's field of view, their inapplicability forces the system to rely exclusively on the dynamic behavior of surrounding vehicles. Third, the site must exhibit sufficient traffic volume during the recording period to ensure that multiple vehicles are simultaneously visible for the majority of the recording duration, providing the necessary data for context extraction. Finally, the environment must offer a physical vantage point that permits elevated camera placement, which is critical for minimizing inter-vehicle occlusion.

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

Accurate displacement computation requires two sequential calibration procedures: intrinsic lens distortion correction and planar ground-plane homography estimation, both conducted once per physical camera installation.

#### Intrinsic Calibration

Because commercial mobile camera lenses introduce radial and tangential distortions that cause straight lines to appear curved, uncorrected frames introduce systematic geometric errors into spatial measurements. The camera intrinsic parameters are estimated using the standard checkerboard calibration method established by Zhang [56] and implemented in the OpenCV framework [55]. A printed checkerboard pattern containing 9 × 6 internal corners of known metric dimensions is affixed to a rigid planar surface and captured across a minimum of 20 distinct orientations and distances using the fixed recording configuration. Sub-pixel corner detection via `cv2.findChessboardCorners` and `cv2.cornerSubPix` feeds into `cv2.calibrateCamera` to derive the camera intrinsic matrix $\mathbf{K}$ and distortion coefficient vector $\mathbf{D}$:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{D} = (k_1, k_2, p_1, p_2, k_3)$$

where $f_x, f_y$ represent the focal lengths in pixels, $(c_x, c_y)$ denotes the principal point coordinates, $k_1, k_2, k_3$ are radial distortion coefficients, and $p_1, p_2$ are tangential distortion coefficients. Prior to downstream detection and tracking, all captured video frames are undistorted using `cv2.undistort`. Calibration quality is formally evaluated using the reprojection error, defined as the root mean square (RMS) Euclidean distance between observed checkerboard corners and their projected model counterparts [56]. In accordance with planar camera calibration standards [55, 56], a mean reprojection error strictly below 1.0 pixel is maintained as the threshold for geometric acceptability.

#### Homography Estimation

To map undistorted two-dimensional pixel coordinates to real-world ground-plane metric coordinates, a planar homography transformation is computed under the fundamental geometric assumption that all tracked contact points lie coplanar on the roadway surface [49]. At the designated test location, a minimum of four non-collinear physical ground control points are measured on the road surface using a calibrated measuring tape and marked with high-visibility markers to establish a quadrilateral reference geometry. The pixel coordinates $(x_i, y_i)$ of each reference point and their corresponding ground-plane coordinates $(X_i, Y_i)$ in meters are utilized to compute the 3 × 3 homography matrix $\mathbf{H}$ using `cv2.findHomography` with RANSAC outlier rejection. This projective transformation maps any undistorted image point $(x, y)$ to real-world metric coordinates $(X, Y)$ via:

$$\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad X = \frac{X'}{w}, \quad Y = \frac{Y'}{w}$$

Geometric fidelity is validated by computing the Euclidean error against at least four independent ground validation points excluded from the initial estimation, allowing the research team to evaluate directional distortion along longitudinal and lateral road axes [49]. If the maximum measured position error across validation markers exceeds 0.10 meters, recalibration of the ground control points is performed.

---

## 3.3 Proposed Method

The system architecture consists of three sequential processing stages. Each stage's output serves as the direct input to the next.

[Figure 3.2 placeholder]

_Figure 3.2. Sequential three-stage system architecture._

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

To project bounding box detections into real-world coordinates without introducing three-dimensional object height bias, the system extracts the bottom-center coordinate of each vehicle's bounding box:

$$p_{\text{bottom}} = \left(x + \frac{w}{2}, \; y + h\right)$$

In monocular surveillance geometry, the bottom-center point approximates the vehicle's physical contact point with the pavement, satisfying the coplanar road-surface constraint ($Z = 0$) required for planar homography projection [49]. Utilizing the vertical centroid of the bounding box would introduce perspective height distortion that scales unfavorably with camera depression angle. As established in geometric traffic perception literature, partial vertical occlusion from guardrails, roadside barriers, or intervening tall vehicles truncates the lower boundary of the bounding box, elevating the observed bottom-center coordinate above the true road surface ($Z > 0$) [22, 40]. Projecting an elevated point through a planar homography matrix $\mathbf{H}$ produces a well-documented geometric distortion that projects the estimated point further along the camera ray [49]; therefore, the displacement computation operates under the standard computer vision assumption of unobstructed road-contact visibility.

The extracted bottom-center coordinates are projected onto the calibrated ground plane to yield metric positions $(X_t, Y_t)$. Vehicle displacement over time is subsequently evaluated using a temporal lookback window of $N = 3$ seconds ($n = N \times \text{fps} = 90$ frames at 30 fps), defined as the Euclidean distance between the vehicle's current position and its position $n$ frames prior:

$$d(v, t) = \sqrt{(X_t - X_{t-n})^2 + (Y_t - Y_{t-n})^2}$$

The 3-second lookback duration is grounded in empirical traffic engineering standards for low-speed pedestrian and vehicular clearance [59]. In transportation research, comfortable walking speeds average approximately 1.2 to 1.4 m/s [59]; over an interval of 3 seconds, a vehicle crawling at even a minimal speed of 1.4 m/s travels 4.2 meters. This distance substantially exceeds the sensor noise floor, ensuring that actively moving vehicles are not misclassified as stationary. For tracks with an active lifespan shorter than $n$ frames, displacement is calculated relative to the track's initial frame; however, to maintain statistical integrity, vehicles with fewer than $n$ frames of tracking history are excluded from the surrounding vehicle pool $\mathcal{S}(t)$ during feature extraction, preventing uncalibrated early-track jitter from corrupting contextual metrics.

### 3.3.4 Restricted Zone and Event Trigger

The enforcement Region of Interest (ROI) is defined as a closed polygon in undistorted image coordinates, established once during setup and grounded in the statutory provisions of Republic Act No. 4136. Spatial membership is evaluated continuously for every tracked vehicle using the ray casting point-in-polygon algorithm applied to the vehicle's bottom-center coordinate $p_{\text{bottom}}$.

[Figure 3.3 placeholder]

_Figure 3.3. Example restricted zone polygon overlaid on an undistorted frame from the test site._

Within this restricted zone, a vehicle $v$ at frame $t$ is binarized as stationary when its measured displacement falls below a depth-calibrated threshold $\varepsilon_s(Y_t)$:

$$\text{stationary}(v, t) = \begin{cases} 1 & \text{if } d(v, t) < \varepsilon_s(Y_t) \\ 0 & \text{otherwise} \end{cases}$$

The stationary noise ceiling $\varepsilon_s(Y_t)$ is calibrated empirically using known-stationary vehicles recorded for a minimum of 60 seconds across three depth zones (near-field, mid-field, and far-field). Because perspective projection causes constant pixel-level detection jitter to represent larger metric displacements at greater distances from the camera [49, 56], applying a uniform global noise ceiling would misclassify creeping near-field vehicles as stopped. To resolve this, $\varepsilon_s(Y_t)$ is formulated as a depth-dependent linear interpolation scaled by the metric $Y$-coordinate across the calibrated positions, incorporating a 1.5× safety multiplier above the measured jitter to accommodate camera vibration.

To trigger an evaluation event, a per-vehicle dwell timer increments for each vehicle that maintains a stationary classification ($d(v, t) < \varepsilon_s$) within the restricted zone, resetting to zero whenever motion is detected. When the dwell timer reaches the temporal threshold $T = 60$ seconds, a trigger event is generated for that Track ID. The 60-second duration is adopted directly from established fixed-camera parking surveillance literature [8, 18, 29, 36], where legitimate operational pauses (such as pedestrian yielding or momentary queueing) consistently resolve well below this temporal threshold under normal urban traffic flow.

### 3.3.5 Feature Extraction

Upon satisfaction of the temporal trigger threshold at frame $t_{\text{trig}}$, the system extracts a six-dimensional spatiotemporal feature vector structured across two observation phases: four snapshot features capturing instantaneous traffic context at $t_{\text{trig}}$, and two monitoring features evaluating temporal evolution over a subsequent monitoring window $W = 15$ seconds ($t_{\text{trig}} + W \times \text{fps}$). Let $\mathcal{S}(t)$ denote the set of all active surrounding vehicles at frame $t$, explicitly excluding the candidate vehicle and any vehicle with fewer than $n$ frames of tracking history (Section 3.3.3).

In the snapshot phase, the system first computes the Stationary Ratio $R(t)$, which quantifies the proportion of surrounding vehicles currently classified as stationary:

$$R(t) = \frac{\sum_{v_i \in \mathcal{S}(t)} \text{stationary}(v_i, t)}{|\mathcal{S}(t)|}$$

The theoretical formulation of $R(t)$ is rooted in Lighthill and Whitham's Kinematic Wave Theory [53] and spatiotemporal traffic analysis [2, 35]. In urban traffic flow, legitimate traffic stoppages represent collective, causally coupled deceleration states that propagate across neighboring vehicles, causing $R(t) \to 1.0$. Conversely, an illegal parking violation represents an isolated spatial anomaly where the candidate remains stationary while surrounding traffic maintains independent movement, yielding $R(t) \to 0.0$.

To provide statistical weighting for the Stationary Ratio, the system records the Surrounding Vehicle Count $C = |\mathcal{S}(t_{\text{trig}})|$. Grounded in traffic density estimation literature [9, 17], $C$ contextualizes sample confidence, as an identical ratio carries higher evidential certainty across dense vehicle queues than across sparse traffic. Because vehicles with fewer than $n$ tracking frames are excluded, $C$ conservatively undercounts traffic during rapid turnover, successfully protecting $R(t)$ from premature stationary misclassifications.

To capture the continuous intensity of surrounding vehicle movement that binary thresholding discards, the Mean Surrounding Displacement $\bar{d}$ is computed in metric units [17, 53]:

$$\bar{d} = \frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} d(v_i, t_{\text{trig}})$$

Complementing the mean, the Surrounding Displacement Standard Deviation $\sigma_d$ measures the kinematic uniformity of surrounding traffic behavior [17, 21]:

$$\sigma_d = \sqrt{\frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} \left(d(v_i, t_{\text{trig}}) - \bar{d}\right)^2}$$

As demonstrated in traffic anomaly detection research [21], low $\sigma_d$ indicates homogeneous flow (all vehicles flowing smoothly or all vehicles completely stopped in gridlock), whereas high $\sigma_d$ identifies heterogeneous, turbulent conditions (e.g., vehicles navigating around an obstruction).

In the post-monitoring phase, the system evaluates the macroscopic evolution of surrounding traffic through the Ratio Change $\Delta R$ after window $W = 15$ seconds:

$$\Delta R = R(t_{\text{trig}} + W \times \text{fps}) - R(t_{\text{trig}})$$

Negative values indicate clearing congestion, whereas stable values indicate sustained conditions [8, 42]. If all surrounding vehicles depart during monitoring such that $|\mathcal{S}(t_{\text{end}})| = 0$, $\Delta R$ is assigned a null representation, handled natively by XGBoost's sparsity-aware split finding algorithm [60].

Finally, the Candidate Post-Trigger Displacement $d_{\text{cand}}$ verifies the candidate vehicle's individual dwell status [8, 42]:

$$d_{\text{cand}} = \sqrt{(X_{t_{\text{end}}} - X_{t_{\text{trig}}})^2 + (Y_{t_{\text{end}}} - Y_{t_{\text{trig}}})^2}$$

where $t_{\text{end}}$ denotes $t_{\text{trig}} + W \times \text{fps}$ (or the final frame of the track if terminated early). This feature ensures that candidates that resume transit during monitoring are recognized as departing. Table 3.3 consolidates the complete six-dimensional feature vector specification.

| Feature                             | Symbol            | Type    | Range        | Phase      |
| ----------------------------------- | ----------------- | ------- | ------------ | ---------- |
| Stationary Ratio                    | $R$               | Float   | [0.0, 1.0]   | Snapshot   |
| Surrounding Vehicle Count           | $C$               | Integer | $\geq 0$     | Snapshot   |
| Mean Surrounding Displacement       | $\bar{d}$         | Float   | $\geq 0$ m   | Snapshot   |
| Surrounding Displacement Std. Dev.  | $\sigma_d$        | Float   | $\geq 0$ m   | Snapshot   |
| Ratio Change                        | $\Delta R$        | Float   | [-1.0, +1.0] | Monitoring |
| Candidate Post-Trigger Displacement | $d_{\text{cand}}$ | Float   | $\geq 0$ m   | Monitoring |

_Table 3.3. Feature vector specification._

Regarding feature completeness, the vector intentionally omits static spatial attributes (e.g., distance to curb or lane markings), adhering strictly to lightweight edge-tracking principles without requiring computationally expensive lane-segmentation networks [5, 42]. Each feature represents an independent mathematical derivation: $R$ derives from binarized states, $\bar{d}$ from continuous displacements, $\sigma_d$ from second-order dispersion, $\Delta R$ from temporal progression, and $d_{\text{cand}}$ from candidate-specific motion. This formulation completely avoids linear dependencies and multicollinearity [60]. In the isolated edge case where $|\mathcal{S}(t_{\text{trig}})| = 0$, the classifier is bypassed and the event defaults to standard temporal threshold logic, which is logged separately from model evaluation.

### 3.3.6 Violation Classification

The feature vector is processed by a gradient-boosted decision tree classifier, specifically XGBoost [60], which outputs a binary prediction: 1 (violation) or 0 (non-violation).

XGBoost constructs an additive ensemble of decision trees, where each successive tree is trained to correct the residual prediction errors of the preceding ensemble. At each node, the algorithm evaluates candidate split points across all input features and selects the split that maximizes the reduction in the regularized objective function. The split thresholds are computed from the training data distribution via gradient optimization, establishing data-driven decision boundaries rather than relying on heuristic cutoff values.

The classifier receives only the six numeric features defined in Table 3.3. Event ID and Track ID are metadata identifiers retained by the system's orchestration layer for associating the classifier's output with the correct on-screen vehicle for flagging and logging. They are not passed to the classifier.

The context-aware classifier operates under identifiable structural limitations. First, when only one surrounding vehicle is tracked at the trigger frame ($C = 1$), the Stationary Ratio $R$ degenerates to a binary value $\in \{0, 1\}$, the Mean Surrounding Displacement $\bar{d}$ reduces to a single observation, and the Surrounding Displacement Standard Deviation $\sigma_d$ is trivially zero. Under these conditions, the contextual features carry minimal discriminative information, and the classifier must rely primarily on $d_{\text{cand}}$ and $\Delta R$. The classifier's discrimination ability therefore degrades monotonically as $C$ decreases toward one. Second, an isolated legitimate stop, such as a lone vehicle yielding to a pedestrian during low-traffic periods, yields features identical to an illegally parked vehicle ($R = 0$, $\Delta R = 0$, $d_{\text{cand}} = 0$). Third, under extended gridlock lasting longer than the combined monitoring period ($T + W = 75$ seconds), no vehicles move during the window ($R = 1$, $\Delta R = 0$, $d_{\text{cand}} = 0$). In these scenarios, the spatiotemporal signatures of congestion and illegal parking become mathematically identical within the fixed temporal window. Fourth, on high-speed roads with rapid vehicle turnover, the $n$-frame tracking-history filter (Section 3.3.3) may exclude a large proportion of short-tracked vehicles from $\mathcal{S}(t)$, causing $C$ to undercount the actual traffic density. In extreme cases, this could produce artificially low $C$ values or trigger the zero-surrounding-vehicle fallback during what is actually moderate-density traffic with high turnover.

### 3.3.7 System Parameter Summary

Table 3.4 consolidates all fixed and empirically determined parameters in the proposed system.

| Parameter                      | Symbol                | Value                  | Justification                                       |
| ------------------------------ | --------------------- | ---------------------- | --------------------------------------------------- |
| Resolution                     | N/A                   | 1920 × 1080            | Device specification                                |
| Frame rate                     | fps                   | 30                     | Native recording rate; no downsampling              |
| Temporal trigger threshold     | $T$                   | 60 s                   | Literature consensus [8, 18, 29, 36]                |
| Lookback window                | $N$                   | 3 s (90 frames)        | Transportation standard for 1.4 m/s clearance [59] |
| Stationary threshold           | $\varepsilon_s$       | Empirically determined | Depth-dependent noise calibration (Section 3.3.4)   |
| Monitoring window              | $W$                   | 15 s                   | Upstream extraction parameter; sensitivity assessed in Section 3.5.6 |
| Detection confidence threshold | N/A                   | 0.25                   | Ultralytics YOLOv8 default [57]                     |
| IoU association threshold      | $\sigma_{\text{IoU}}$ | 0.5                    | Adopted from NMS threshold range [54, 57]           |
| Track confirmation window      | $n_{\text{init}}$     | 3 frames               | Empirical default; prevents single-frame false positives |
| Track termination window       | $t_{\text{lost}}$     | 5 frames               | Bridges momentary misses; avoids ghost tracks (Section 3.3.2) |

_Table 3.4. Complete system parameter summary._

[Figure 3.4 placeholder]

_Figure 3.4. Complete system pipeline diagram._

---

## 3.4 Ground Truth Annotation

### 3.4.1 Event-Level Annotation

Annotation is performed at the event level, not the frame level. The set of events to be annotated is determined by running the Vision Pipeline and Event Trigger stages on the recorded footage and collecting all generated trigger events.

For each trigger event, the annotator reviews the video segment from 30 seconds before the trigger frame through the end of the monitoring window. Table 3.5 defines the ground truth annotation fields, where specific stop reasons are recorded exclusively for non-violation events.

| Field       | Type        | Description                                              |
| ----------- | ----------- | -------------------------------------------------------- |
| Event ID    | Integer     | Matches trigger event output                             |
| Track ID    | Integer     | Tracked vehicle identity                                 |
| Label       | Binary      | Violation or Non-Violation                               |
| Stop reason | Categorical | Congestion / Pedestrian yielding / Loading / Other / N/A |
| Notes       | Free text   | Annotator remarks on ambiguous cases                     |

_Table 3.5. Ground truth annotation fields._

An event is formally annotated as a Violation when a candidate vehicle maintains a stationary dwell status within the restricted zone throughout the observation window in the absence of any observable external traffic impediment, such as simultaneous surrounding queueing, active pedestrian right-of-way crossing, or authorized curb operations. This protocol aligns with the statutory legal framing of Section 46 of Republic Act No. 4136, wherein stationary vehicular presence within designated prohibited zones (such as intersections, pedestrian crossings, or designated curb areas) establishes a prima facie infraction unless observable operational necessity is documented.

### 3.4.2 Inter-Annotator Agreement

A randomly selected 20% subset of triggered events will be independently annotated by a second annotator. Agreement on the binary label will be measured using Cohen's kappa:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is the observed agreement and $p_e$ is the expected agreement by chance. A minimum $\kappa \geq 0.80$ is targeted, corresponding to the upper boundary of "Substantial" agreement (0.61 to 0.80) and the threshold for "Almost Perfect" agreement (0.81 to 1.00) [61]. If agreement falls below this threshold, disagreements will be adjudicated and the operational definition refined before re-annotation.

---

## 3.5 Training and Evaluation Protocol

### 3.5.1 Cross-Validation

Due to the anticipated limited size of the event dataset (constrained by the number of qualifying trigger events in the recording period), a fixed train-test split risks high variance in performance estimates. The study utilizes stratified 5-fold cross-validation, following established machine learning evaluation standards for variance reduction and class balance preservation on small datasets [62]. The annotated dataset is partitioned into five equal folds, stratified by label to ensure that each fold mirrors the overall class proportion. In each iteration, four folds serve as the training set and one fold as the held-out evaluation set. The process repeats five times, and metrics are reported as the mean ± standard deviation across folds [62].

Random fold assignment may place temporally adjacent events into different folds, potentially introducing optimistic bias due to correlated environmental conditions such as lighting or traffic density. This limitation is retained as the dataset size is insufficient to support temporal blocking without introducing unacceptable variance.

### 3.5.2 Hyperparameter Tuning

XGBoost hyperparameters are tuned via grid search using a nested cross-validation framework, with 3-fold cross-validation executed within each outer training fold. As established by Varma and Simon [63], nesting the model selection and hyperparameter optimization loops within the training partition is essential to prevent data leakage and selection bias from artificially inflating performance estimates. The hyperparameter search space is defined in Table 3.6. The parameter configuration yielding the highest mean F1-score across inner cross-validation folds is selected to train the final model for each outer evaluation fold.

| Hyperparameter         | Search values  |
| ---------------------- | -------------- |
| Number of estimators   | 50, 100, 200   |
| Maximum tree depth     | 3, 5, 7        |
| Learning rate ($\eta$) | 0.01, 0.1, 0.3 |
| Minimum child weight   | 1, 3, 5        |

_Table 3.6. XGBoost hyperparameter search space._

No feature scaling or normalization is applied, as gradient-boosted decision trees are invariant to monotonic feature transformations: split decisions are based on threshold comparisons over feature values, and the relative ordering of data points is preserved under any monotonic rescaling [60].

### 3.5.3 Evaluation Metrics

Classification performance is evaluated using precision, recall, F1-score, and false positive rate (FPR), derived from standard binary confusion matrix formulations in automated detection and surveillance [64]. In this operational context, True Positives (TP) represent correctly flagged genuine violations, False Positives (FP) denote legitimate stops incorrectly flagged as violations, True Negatives (TN) denote correctly suppressed legitimate stops, and False Negatives (FN) represent genuine violations that the system failed to identify. Table 3.7 outlines the mathematical definitions of these metrics.

| Metric              | Definition                                                                              |
| ------------------- | --------------------------------------------------------------------------------------- |
| Precision           | $\frac{TP}{TP + FP}$                                                                    |
| Recall              | $\frac{TP}{TP + FN}$                                                                    |
| F1-Score            | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| False Positive Rate | $\frac{FP}{FP + TN}$                                                                    |

_Table 3.7. Evaluation metrics._

Metrics for the proposed method are reported as mean ± standard deviation across the five cross-validation folds. Precision, recall, and F1-score are reported for each class individually (violation and non-violation) to ensure that performance differences are not masked by class-level aggregation.

### 3.5.4 Comparative Evaluation

The proposed method will be compared against a reimplementation of the detection logic described by Abella and Catedrilla [1]. Their system is a YOLOv8-based illegal parking detection framework that uses composite detection classes to identify violations at the frame level. The method performs frame-by-frame inference without multi-object tracking or temporal dwell-time logic; a violation is identified when a composite class representing the spatial co-occurrence of a vehicle, a human, and a no-parking sign is detected within a single frame. This architectural design makes the method fundamentally distinct from the proposed spatiotemporal classifier, and this distinction is central to the comparative hypothesis.

To enable a fair and controlled comparison, the research team will reimplement Abella et al.'s composite-class detection logic and apply it to the same video footage used to evaluate the proposed method. Both methods will be tested under identical conditions, specifically, the same camera and recording setup, the same recording resolution and frame rate, the same YOLOv8 architecture version, and the same ground truth event annotations. This controlled same-data design ensures that any measured performance difference is attributable to the decision method itself rather than to differences in test conditions.

The reimplementation procedure meticulously adheres to Abella et al.'s published methodology. Training images are collected and annotated by the research team at the test site employing the described annotation format, specifically COCO-format bounding boxes. The reimplemented model is trained to recognize two distinct conceptual categories of classes. The foundational base classes consist of standard object detections, including cars, generic vehicles, motorcycles, persons, and no-parking signs. Concurrently, the model detects composite classes that function as the direct violation indicator, specifically identifying the spatial co-occurrence of a vehicle with a human, as well as the tripartite co-occurrence of a vehicle, a human, and a no-parking sign. In instances where Abella et al.'s original manuscript omits critical implementation details, such as the detection confidence threshold, the input image dimensions, or the exact data split ratio for training and validation, the research team applies standard empirical defaults. These parameters are documented explicitly to ensure complete reproducibility of the reimplemented baseline.

Because Abella et al.'s method produces frame-level detections while the proposed method produces event-level classifications, the reimplemented method's output must be aggregated to the event level to enable fair comparison. The observation window for each event spans from the trigger frame $t_{\text{trig}}$ through the end of the monitoring period $t_{\text{trig}} + W \times \text{fps}$. For each triggered event in the ground truth, the aggregation procedure checks whether the composite violation class (vehicles-human-no-parking-sign) is detected in any frame within that event's observation window. If at least one composite detection occurs within the window, the reimplemented method classifies the event as a violation. If no composite detection occurs in any frame within the window, the event is classified as a non-violation. Both methods are then evaluated against the same event-level ground truth labels using the metrics defined in Table 3.7.

The evaluation scope for this comparison is strictly defined by the proposed method's trigger mechanism. Only vehicles that have been stationary in the restricted zone for at least $T = 60$ seconds generate a trigger event, and consequently, violations shorter than 60 seconds are excluded from the evaluation pool. The reimplemented Abella et al. method is evaluated only on this pre-filtered event pool. The comparison therefore measures discrimination ability within a subset of extended-duration stationary vehicles rather than end-to-end detection coverage across all traffic scenarios.

This comparison against Abella et al. is not a test of general algorithmic superiority, given the fundamental architectural differences between a tracked spatiotemporal classifier and an untracked single-frame detector. The specific hypothesis under test is that context-unaware spatial co-occurrence models exhibit elevated false positive rates under extended collective traffic conditions, and that the proposed spatiotemporal classifier reduces this specific failure mode. The primary outcome metric for this hypothesis is the false positive rate (FPR). Abella et al. reported lower precision for parking detection (74.67%) relative to littering detection (98.41%). Mathematical deduction applied to their reported figures, specifically 2,803 total detections and 509 combined false positives across both classes with no per-class breakdown, indicates that the majority of false positives originate from the parking class. This is a deduction made by applying arithmetic to Abella et al.'s own disclosed numbers and is not a conclusion stated in their paper. The reimplementation tests whether these parking-class false positives are attributable to collective traffic stops, which the proposed spatiotemporal classifier is designed to suppress.

A methodological distinction exists in the violation definitions used by the two approaches, and this distinction is retained even though both methods are evaluated on the same dataset under identical test conditions. Abella et al. define a violation as the spatial co-occurrence of specific object classes within a single frame, with no temporal duration requirement. The proposed method, by contrast, defines a violation as a vehicle that remains stationary within a legally grounded restricted zone for at least 60 seconds without an observable external cause for the stop. This difference in what constitutes a detectable violation is inherent to the respective architectures and cannot be eliminated by testing on shared data. This distinction will be explicitly discussed when interpreting comparative results in Chapter 4.

### 3.5.5 Feature Importance and Ablation

To interpret the trained XGBoost model, feature importance is quantified using the average gain, which calculates the mean improvement in the regularized objective function contributed by a specific feature across all trees in which it appears [60]. These importance scores are subsequently reported in the experimental results to provide interpretability regarding which aspects of the temporal traffic context the classifier relies on most heavily to form its decision boundaries.

An ablation study evaluates the marginal contribution of individual feature subsets. As specified in Table 3.8, the classifier is retrained and evaluated under snapshot-only, monitoring-only, and full feature configurations using the identical stratified 5-fold cross-validation protocol.

| Configuration            | Features                                                                 |
| ------------------------ | ------------------------------------------------------------------------ |
| Snapshot features only   | $R, \; C, \; \bar{d}, \; \sigma_d$                                       |
| Monitoring features only | $\Delta R, \; d_{\text{cand}}$                                           |
| Full feature vector      | All six features                                                         |

_Table 3.8. Ablation study configurations._

### 3.5.6 Parameter Sensitivity Analysis

A parameter sensitivity analysis examines the robustness of the proposed pipeline to variations in key extraction hyperparameters [65]. Following standard sensitivity analysis principles [65], a one-at-a-time (OAT) parameter perturbation is conducted, wherein the full pipeline (feature extraction, classifier training, and 5-fold cross-validation) is re-executed with the modified parameter value while holding all other parameters fixed at their calibrated defaults.

| Parameter                              | Default            | Test values                          |
| -------------------------------------- | ------------------ | ------------------------------------ |
| Monitoring window ($W$)                | 15 s               | 5, 10, 15, 20, 30 s                  |
| Stationary threshold ($\varepsilon_s$) | 1.5× noise ceiling | 0.5×, 1.0×, 1.5×, 2.0× noise ceiling |
| Lookback window ($N$)                  | 3 s                | 1, 2, 3, 5 s                         |

_Table 3.9. Parameter sensitivity analysis configurations._



