# Chapter III: Methodology

## Conceptual Framework

![Figure 3.1. Conceptual framework of the proposed spatiotemporal classification pipeline.](figures/figure_3_1.png)
*Figure 3.1 Conceptual Framework*  
The conceptual framework of this study, presented in Figure 3.1, organizes the research into four sequential phases, each transforming the output of the preceding phase into a representation of higher abstraction. Raw video is converted into geometrically rectified metric trajectories; trajectories are condensed into a structured numeric feature vector; the feature vector is mapped to a binary violation decision; and the aggregate decisions are subjected to statistical evaluation against an external benchmark. These four sequential phases establish the input, transformation, and output across the proposed pipeline.

The first phase establishes the physical and geometric preconditions of the study through fixed elevated capture, intrinsic lens undistortion [78], and 2D ground-plane homography estimation [75]. Without metric rectification, the displacement quantities constituting the feature vector would be expressed in pixels and would vary systematically with depth, rendering them incomparable across the frame. The second phase converts rectified video into structured numeric observations through detection and tracking [30], [76], ground-plane projection dwell-timer triggering, and extraction of the six-dimensional feature vector.

The feature design for the second phase is conceptually motivated by traffic flow dynamics [42]. In an urban corridor, legitimate stoppages often propagate as collective deceleration shockwaves across upstream vehicles, whereas an unauthorized curb stop typically presents as an isolated immobilization alongside an active or clearing stream. The feature vector is constructed precisely to render this distinction numerically observable. Every quantity in it is computed from displacement data the tracking module already produces as a byproduct of frame-to-frame association, so the framework introduces no new sensing modality, no additional hardware, and no dependence on the continued visibility of any fixed external reference element.

The third phase introduces the study's primary methodological contribution, integrating ByteTrack multi-object tracking with a supervised XGBoost classifier. Event-level ground truth labels $Y \in \{0, 1\}$ supply the supervised target, and an XGBoost gradient-boosted decision tree ensemble will be trained on the feature vector to classify each triggered event as a genuine violation or a legitimate traffic stop [13]. Gradient-boosted trees are selected over end-to-end deep architectures because the decision boundary is learned over six low-dimensional engineered features rather than raw pixels, because tree inference imposes negligible overhead consistent with edge deployment constraints [7], and because the resulting split thresholds remain directly interpretable. The fourth phase converts model outputs into evidentiary claims through cross-validation, ablation, and same-dataset benchmarking against the single-frame co-occurrence baseline [1]. 

## Data Acquisition

Data collection will be conducted at two urban roadway corridors within Quezon City, National Capital Region. Site A, the primary site, is Atherton Street in Barangay North Fairview, a secondary residential-commercial collector road within the North Fairview Park Subdivision corridor, characterized by mixed-use frontage and sustained curbside demand from adjacent residential and small-commercial establishments. Site B, the secondary site, is the roadway segment fronting 40 P. A. Bernardo Street, Barangay Kaunlaran, Cubao, a commercial corridor exhibiting higher through-traffic volume, denser loading activity, and greater heterogeneity in vehicle class composition. The camera at Site B is positioned to overlook this specific segment, and the restricted zone polygon is drawn over the carriageway and driveway frontage immediately adjacent to that address; the enforcement zone and the calibrated ground plane are therefore fixed to this single street address rather than to the corridor at large. Site A is designated primary because its curbside parking pressure is recurrent, increasing the probability of naturally occurring violation events and reducing reliance on staged scenarios. 

The two-site design is a deliberate safeguard. A single-site dataset would leave the study unable to distinguish a generalizable contextual signal from site-specific overfitting, a limitation repeatedly noted in single-location illegal-parking evaluations [3], [79]. Recording at two corridors with materially different land-use profiles permits the study to report whether performance is stable across sites without overstating the claim to universal deployability. Both corridors are evaluated against five physical criteria before selection, namely the presence of statutory restricted frontage under Republic Act No. 4136, the absence of physical traffic signals, recurrent curbside parking activity, consistent ambient traffic volume, and accessible elevated vantage points. 

### Recording Configuration

Video footage will be recorded using the apparatus, mounted at a fixed, elevated position overlooking each test site. Table 3.1 summarizes the recording parameters, which will be held constant across all sessions to ensure geometric consistency for calibration.

| Parameter | Value |
| :---- | :---- |
| Device | Action Camera  |
| Resolution | 1920 × 1080 pixels  |
| Frame Rate | 30 fps  |
| Mounting | Tripod, fixed position, no operator contact during recording  |
| Height | 3-4 meters |
| Recording Period | 14:00-17:00 |

*Table 3.1. Video Recording Parameters*  
The native 30 fps rate will be retained without temporal downsampling, a decision that warrants justification because reducing frame rate is a common compute-reduction measure. As established in tracking-by-detection literature [76], [81], maintaining 30 fps minimizes inter-frame displacement relative to bounding box dimensions, preserving high IoU overlap between consecutive detections and preventing association fragmentation during temporary occlusions.

### Camera Calibration

Accurate displacement computation requires two sequential calibration procedures, comprising intrinsic lens distortion correction and planar ground-plane homography estimation, both performed once per physical camera installation. Figure 3.2 illustrates the resulting two-stage geometric transformation from raw image space to metric ground-plane coordinates. The ordering of these procedures relative to recording is procedurally critical. Both procedures will be performed after the camera has been mounted and locked in position and before the recording session begins, because the homography is specific to a single camera pose and the intrinsic sequence must be captured using the identical device configuration, resolution, and frame rate employed for recording. If the camera is disturbed at any point during a session, calibration will be repeated in full and the affected footage will be discarded.

![Figure 3.2. Two-stage geometric calibration and metric ground-plane projection](figures/figure_3_2.png)

*Figure 3.2. Two-stage geometric calibration and metric ground-plane projection*

Because commercial mobile camera lenses introduce radial and tangential distortions that cause straight lines to appear curved, uncorrected frames introduce systematic geometric errors into spatial measurements. The camera intrinsic parameters are estimated using the standard checkerboard calibration method established by Zhang [78] and implemented in the OpenCV framework [9]. A printed checkerboard pattern containing 9 × 6 internal corners of known metric dimensions is affixed to a rigid planar surface and captured across a minimum of 20 distinct orientations and distances using the fixed recording configuration. Sub-pixel corner detection via *cv2.findChessboardCorners* and *cv2.cornerSubPix* feeds into *cv2.calibrateCamera* to derive the camera intrinsic matrix $\mathbf{K}$ and distortion coefficient vector $\mathbf{D}$:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{D} = (k_1, k_2, p_1, p_2, k_3)$$

where $(f_x, f_y)$ represent the focal lengths in pixels, $(c_x, c_y)$ denotes the principal point coordinates, $k_1, k_2, k_3$ are radial distortion coefficients, and $p_1, p_2$ are tangential distortion coefficients. Prior to downstream detection and tracking, all captured video frames will be undistorted using *cv2.undistort*, and all subsequent image coordinates, including the vertices of the ROI polygon, will be defined in the undistorted coordinate space. Calibration quality will be formally evaluated using the reprojection error, defined as the root mean square (RMS) Euclidean distance between observed checkerboard corners and their projected model counterparts [78]. In accordance with planar camera calibration standards [9], [78], a mean reprojection error strictly below 1.0 pixel will be maintained as the threshold for geometric acceptability.

To map undistorted two-dimensional pixel coordinates to real-world ground-plane metric coordinates, a planar homography transformation is computed under the fundamental geometric assumption that all tracked contact points lie coplanar on the roadway surface [75]. At each designated test location, a minimum of four non-collinear physical ground control points are measured on the road surface using a calibrated measuring tape and marked with high-visibility markers to establish a quadrilateral reference geometry, the minimum number of point correspondences required to determine the eight degrees of freedom of a planar homography [22]. The pixel coordinates $(x_i, y_i)$ of each reference point and their corresponding ground-plane coordinates $(X_i, Y_i)$ in meters are utilized to compute the 3 × 3 homography matrix $\mathbf{H}$ using *cv2.findHomography* with RANSAC outlier rejection [16]. This projective transformation maps any undistorted image point $(x, y)$ to real-world metric coordinates $(X, Y)$ via:

$$\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad X = \frac{X'}{w}, \quad Y = \frac{Y'}{w}$$

Geometric fidelity will be validated by computing the Euclidean error against at least four independent ground validation points excluded from the initial estimation, allowing the research team to evaluate directional distortion along the longitudinal and lateral road axes separately. A maximum measured position error threshold of 0.10 meters across validation markers is adopted as the study's operational acceptance criterion, ensuring ground measurement error remains substantially smaller than vehicle footprint dimensions and the stationary detection noise ceiling; if this threshold is exceeded, recalibration will be performed. The coplanarity assumption underlying this transformation is what gives the bottom-center projection of each bounding box its physical meaning, since that point is taken to be the vehicle's intersection with the road plane. Any condition that truncates or elevates the point above the true road surface, whether a fixed obstruction inventoried during site characterization or a transient obstruction arising during recording, therefore produces outward perspective distortion along the camera ray and corrupts every displacement quantity subsequently derived from it.

## Proposed Method

The proposed framework operates through a sequential three-stage architecture wherein each stage's output serves as the direct input to the next. The vision pipeline first performs per-frame vehicle detection, multi-object tracking, and planar coordinate transformation to extract continuous metric vehicle trajectories. The event trigger and feature extraction layer monitors vehicle dwell times within a statutory restricted zone and extracts a six-dimensional spatiotemporal feature vector upon trigger activation. Finally, the event classification layer processes these contextual features through a trained gradient-boosted decision tree (XGBoost) to classify candidate stops as illegal parking violations or legitimate traffic pauses. The subsequent subsections formulate each component of this processing pipeline, with a consolidated parameter specification provided in Table 3.7.

### Vehicle Detection

Object detection is performed using YOLOv8n (Nano), the smallest variant of the YOLOv8 architecture [30], with 3.2 million parameters and 8.7 GFLOPs per inference pass. The model uses pretrained MS COCO weights [43] without fine-tuning. Only vehicle classes are retained, specifically car (class 2), motorcycle (3), bus (5), and truck (7). All other detections are discarded.

To supply candidate bounding boxes for tracking without prematurely discarding lower-confidence detections required by subsequent recovery mechanisms, a baseline detection retention floor of $s_{\text{floor}} = 0.10$ is applied at the detector output. Following the two-stage association thresholds established by ByteTrack [76], the tracking module internally partitions detections into primary high-confidence candidates ($s \geq 0.25$, which also aligns with the Ultralytics YOLOv8 inference benchmark [30]) and secondary recovery candidates ($0.10 \leq s < 0.25$). This design ensures high detection recall to prevent undercounting the surrounding traffic population, which would directly distort the Stationary Ratio $R$, while maintaining clean inputs to the tracking stage.

The Ultralytics YOLOv8 implementation outputs bounding box coordinates in the original image space (1920 × 1080), independent of the internal input resize, so no manual coordinate rescaling is required before homography transformation.

### Multi-Object Tracking

A multi-object tracker assigns and maintains a persistent Track ID for each vehicle across consecutive frames. The proposed framework incorporates ByteTrack [76], an appearance-free tracking-by-detection framework that pairs a linear Kalman filter with a two-stage spatial data association algorithm.

The tracking procedure executes on every incoming video frame. In the initial stage, a linear Kalman filter predicts the spatial state and bounding box position of all currently active tracks from the preceding frame. Current vehicle detections exhibiting high confidence scores ($s \geq 0.25$) are compared against these predicted track states by computing the pairwise Intersection over Union (IoU) spatial overlap:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where $A$ and $B$ denote the bounding box areas of the predicted track and observed detection, respectively. Assignment between high-confidence detections and existing tracks is solved via the Hungarian algorithm, requiring an association matching threshold of $\sigma_{\text{match}} = 0.8$, corresponding to an IoU distance metric ($1 - \text{IoU}$). Detections satisfying this threshold are assigned to their respective tracks, and their motion states are updated through the Kalman filter.

In the second stage, ByteTrack processes lower-confidence detections rather than discarding them. Inter-vehicle occlusions, perspective changes, and temporary optical blur often depress detection confidence below standard acceptance thresholds, causing single-threshold trackers to drop valid tracks prematurely. To preserve trajectory continuity without deep visual appearance re-identification, remaining unmatched tracks are compared against low-confidence detections ($0.10 \leq s < 0.25$) using spatial IoU overlap. If a low-confidence detection matches an unmatched track prediction, the track identity is maintained across the occlusion event.

Unmatched high-confidence detections initiate tentative tracks requiring confirmation across $n_{\text{init}} = 3$ consecutive frames before being promoted to persistent active tracks. This confirmation period prevents transient false-positive detections from immediately entering the surrounding vehicle pool $\mathcal{S}(t)$. Active tracks not matched to any detection in a given frame are maintained across a retention buffer of $t_{\text{lost}} = 30$ consecutive frames (1.0 second at 30 fps) before being terminated. This buffer allows the tracker to bridge short-term occlusions while preventing ghost tracks from persisting indefinitely once a vehicle leaves the scene. Terminated Track IDs are not reused. For each confirmed vehicle, the tracker outputs a persistent Track ID, bounding box coordinates (x, y, w, h), class label, and detection confidence score.

### Coordinate Transformation and Displacement Computation

To project bounding box detections into real-world coordinates while mitigating three-dimensional object height distortion under coplanar road assumptions, the vision pipeline extracts the bottom-center coordinate of each vehicle's bounding box:

$$p_{\text{bottom}} = \left(x + \frac{w}{2}, \\; y + h\right)$$

In monocular surveillance geometry, the bottom-center point approximates the vehicle's physical contact point with the pavement, satisfying the coplanar road-surface constraint ($Z = 0$) required for planar homography projection. Utilizing the vertical centroid of the bounding box would introduce perspective height distortion that scales unfavorably with camera depression angle. Partial vertical occlusion from guardrails, roadside barriers, or intervening tall vehicles truncates the lower boundary of the bounding box, elevating the observed bottom-center coordinate above the true road surface ($Z > 0$). Under planar homography, projecting an elevated point produces an outward perspective distortion along the camera ray, projecting the estimated point further than the actual ground position; therefore, accurate metric displacement computation requires unobstructed road-contact visibility.

The extracted bottom-center coordinates are projected onto the calibrated ground plane to yield metric positions $(X_t, Y_t)$. Vehicle displacement over time is subsequently evaluated using a temporal lookback window of $N = 3$ seconds ($n = N \times \text{fps} = 90$ frames at 30 fps), defined as the Euclidean distance between the vehicle's current position and its position $n$ frames prior:

$$d(v, t) = \sqrt{(X_t - X_{t-n})^2 + (Y_t - Y_{t-n})^2}$$

The 3-second lookback duration ($N = 3$ s, or 90 frames at 30 fps) is established to distinguish true vehicular stationarity from low-speed urban creeping while providing a robust margin above perspective measurement noise. Assuming a nominal crawling speed of approximately $1.0\text{ m/s}$ ($3.6\text{ km/h}$), a moving vehicle traverses $3.0$ meters over 3 seconds, which comfortably exceeds the calibrated ground-plane stationary noise ceiling $\varepsilon_s(Y_t)$ (ranging from $0.5$ to $1.2$ meters across depth). For vehicles crawling at lower velocities or in stop-and-go progression, the 3-second check captures operational immobility, while cumulative dwell tracking over the extended trigger window ensures that slowly progressing vehicles continuing through the corridor do not accumulate sustained stationary dwell. For tracks with an active lifespan shorter than $n$ frames, displacement is calculated relative to the track's initial frame; however, to maintain statistical integrity, vehicles with fewer than $n$ frames of tracking history are excluded from the surrounding vehicle pool $\mathcal{S}(t)$ during feature extraction, preventing uncalibrated early-track jitter from corrupting contextual metrics.

### Restricted Zone and Event Trigger

The enforcement Region of Interest (ROI) is defined as a closed polygon in undistorted image coordinates, established once during setup and grounded in the statutory provisions of Republic Act No. 4136\. Spatial membership is evaluated continuously for every tracked vehicle using the ray casting point-in-polygon algorithm applied to the vehicle's bottom-center coordinate $p_{\text{bottom}}$. Within this restricted zone, a vehicle $v$ at frame $t$ is binarized as stationary when its measured displacement falls below a depth-calibrated threshold $\varepsilon_s(Y_t)$:

$$\text{stationary}(v, t) = \begin{cases} 1 & \text{if } d(v, t) < \varepsilon_s(Y_t) \\
0 & \text{otherwise} \end{cases}$$


The stationary noise ceiling $\varepsilon_s(Y_t)$ is calibrated empirically using known-stationary vehicles recorded for a minimum of 60 seconds across three depth zones (near-field, mid-field, and far-field). Because perspective projection causes constant pixel-level detection jitter to represent larger metric displacements at greater distances from the camera [78], applying a uniform global noise ceiling would misclassify creeping near-field vehicles as stopped. To resolve this, the stationary noise ceiling $\varepsilon_s(Y_t)$ is formulated as a non-linear projective scaling across depth positions, mapping the metric $Y$-coordinate across the calibrated positions, and incorporating an empirical 1.5× engineering safety multiplier above the measured static jitter to accommodate environmental vibration. To trigger an evaluation event, a per-vehicle dwell timer increments for each vehicle that maintains a stationary classification ($d(v, t) < \varepsilon_s$) within the restricted zone. To ensure that transient detector jitter or bounding box fluctuations do not falsely reset an active accumulation, brief single-frame motion spikes below a persistence tolerance are filtered before resetting the timer. When the dwell timer reaches the temporal threshold $T = 60$ seconds, a trigger event is generated for that Track ID. The temporal trigger threshold $T = 60$ seconds is adopted following operational benchmarks in curbside surveillance literature [17], [71], designed to distinguish sustained parking violations from common short-duration operational pauses such as passenger drop-offs, pedestrian yielding, or momentary intersection queueing.

### Feature Extraction

When a candidate vehicle remains stationary within the restricted zone for $T = 60$ seconds, an evaluation event is triggered at frame $t_{\text{trig}}$. In automated traffic surveillance, on-street illegal parking is commonly characterized by an immobilized vehicle obstructing a restricted curb lane while the surrounding traffic stream continues moving [17], [54], [55], [71]. To measure this traffic contrast without hardcoded rules, the proposed framework extracts a six-dimensional feature vector. The vector combines four snapshot features that capture the immediate traffic conditions at $t_{\text{trig}}$, and two monitoring features that observe traffic recovery over a subsequent $W = 15$ second window ($t_{\text{end}} = t_{\text{trig}} + W \times \text{fps}$). Let $\mathcal{S}(t)$ denote the set of surrounding vehicles tracked at frame $t$, excluding the candidate vehicle itself and any vehicle with fewer than $n = 90$ frames of continuous tracking history.

In the snapshot phase, the pipeline computes the Stationary Ratio $R(t)$:

$$R(t) = \frac{\sum_{v_i \in \mathcal{S}(t)} \text{stationary}(v_i, t)}{|\mathcal{S}(t)|}$$

Herman and Prigogine [82] established that the fraction of stopped vehicles in a roadway network serves as a fundamental macroscopic descriptor of traffic congestion. In sensitivity analyses of network-level traffic flow, Williams, Mahmassani, and Herman [86] further clarified that this stopped fraction strictly accounts for vehicles temporarily halted within the active traffic stream, such as at traffic signals or queues, as explicitly opposed to parked vehicles. In video surveillance applications, Chaudhary et al. [83] demonstrated that computing the ratio of moving to static detections across a monitored scene effectively characterizes traffic flow states without requiring complex tracking models. The proposed framework adapts these principles to local CCTV surveillance by calculating the Stationary Ratio $R(t)$, an empirical metric measuring the proportion of eligible tracked neighboring vehicles that are currently stationary. During collective traffic stoppages, such as red lights or congested queues, surrounding vehicles halt together ($R(t) \to 1.0$). Conversely, an unauthorized vehicle parked along a curb while the surrounding traffic stream continues to flow produces a low stationary ratio ($R(t) \to 0.0$). Correlating candidate immobilization with the operational state of the surrounding traffic stream provides the classifier with an immediate, camera-level measure of whether the stoppage is an isolated parking event or part of a shared traffic constraint.

To account for traffic density and sample confidence, the framework records the Surrounding Vehicle Count $C$:

$$C = |\mathcal{S}(t_{\text{trig}})|$$

In multi-object tracking, Bochinski et al. [81] showed that filtering out short-lived tracks below a minimum duration threshold removes transient false-positive detections and stabilizes trajectory analysis. The proposed pipeline adapts this requirement by mandating that surrounding vehicles maintain at least $n = 90$ frames (3 seconds) of continuous tracking history before inclusion in $\mathcal{S}(t_{\text{trig}})$. The count $C$ records the total number of eligible neighbors present in the camera frame at the trigger moment. Providing $C$ alongside $R(t)$ allows the XGBoost classifier to interpret the ratio in light of the available sample size, as a high stationary ratio observed across a multi-vehicle platoon ($C \ge 2$) provides stronger empirical support for a collective queue than the same ratio calculated from a single neighboring car ($C = 1$).

To measure stream velocity, the Mean Surrounding Displacement $\bar{d}$ is computed in meters:

$$\bar{d} = \frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} d(v_i, t_{\text{trig}})$$

Wardrop [70] established that average stream speed reflects overall roadway flow conditions. In monocular traffic surveillance, Revaud and Humenberger [84] demonstrated that projecting tracked vehicle positions onto a calibrated road plane through planar homography enables accurate metric speed and displacement estimation from roadside cameras. The proposed approach adapts this methodology by calculating $\bar{d}$ as the average ground-plane displacement of all eligible surrounding vehicles over the 3-second ($N = 90$ frame) lookback window. In heavily congested traffic or red-light queues, surrounding vehicle displacement approaches zero ($\bar{d} \to 0.0\text{ m}$), whereas in active traffic, surrounding vehicles travel substantially greater distances ($\bar{d} \gg \varepsilon_s$). Providing $\bar{d}$ enables the classifier to evaluate whether the motionless candidate vehicle is surrounded by an actively moving stream.

The Surrounding Displacement Standard Deviation $\sigma_d$ measures the variation in surrounding vehicle movement:

$$\sigma_d = \sqrt{\frac{1}{|\mathcal{S}(t_{\text{trig}})|} \sum_{v_i \in \mathcal{S}(t_{\text{trig}})} \left(d(v_i, t_{\text{trig}}) - \bar{d}\right)^2}$$

Treiber and Kesting [66] noted that velocity variance indicates flow stability and turbulence in traffic streams. In field experiments, Stern et al. [85] demonstrated that velocity standard deviation directly quantifies speed heterogeneity during transitions between uniform flow and stop-and-go conditions. The proposed pipeline adapts this concept into $\sigma_d$, measuring the standard deviation of metric displacements across neighboring vehicles over the 3-second lookback interval. A low standard deviation ($\sigma_d \to 0.0\text{ m}$) indicates uniform traffic conditions, where surrounding vehicles are either all stationary in a queue or all traveling at comparable speeds. An elevated standard deviation captures speed heterogeneity, reflecting mixed traffic conditions where certain vehicles slow down or maneuver while adjacent lanes continue moving.

In the monitoring phase, the framework tracks queue clearance using the Ratio Change $\Delta R$:

$$\Delta R = R(t_{\text{end}}) - R(t_{\text{trig}})$$

Queue dissipation in urban corridors is inherently dynamic because stopped vehicles resume motion once temporary bottlenecks or traffic signals clear. The proposed framework monitors this post-trigger clearance by computing the Ratio Change $\Delta R$ across an observation window of $W = 15$ seconds ($t_{\text{end}} = t_{\text{trig}} + 15\text{ s}$), calculated as the endpoint difference $R(t_{\text{end}}) - R(t_{\text{trig}})$. When a temporary traffic stoppage resolves and stopped vehicles accelerate away, the stationary ratio decreases ($\Delta R < 0$). Conversely, under persistent gridlock, the ratio remains largely steady ($\Delta R \approx 0$). If all surrounding vehicles leave the camera frame during the monitoring interval ($|\mathcal{S}(t_{\text{end}})| = 0$), $\Delta R$ is treated as a missing value, which is routed natively by the sparsity-aware split finding algorithm of XGBoost [13].

Finally, the Candidate Post-Trigger Displacement $d_{\text{cand}}$ tracks the candidate vehicle's own movement:

$$d_{\text{cand}} = \sqrt{(X_{t_{\text{end}}} - X_{t_{\text{trig}}})^2 + (Y_{t_{\text{end}}} - Y_{t_{\text{trig}}})^2}$$

where $t_{\text{end}}$ is $t_{\text{trig}} + W \times \text{fps}$ (or the vehicle's final tracked frame if tracking terminates prematurely, with the duration recorded). In curb-lane monitoring, Gao et al. [17] demonstrated the utility of evaluating vehicle start and end positions across observation windows to filter out slow-moving traffic that could otherwise be misidentified as parked. Similarly, Xie et al. [71] established that continuing to track a vehicle's motion state over time discriminates between temporary operational pauses and sustained illegal parking. Following these precedents, the proposed method evaluates whether the candidate vehicle resumes motion after reaching the 60-second dwell mark. A candidate vehicle that moves ($d_{\text{cand}} > \varepsilon_s$) indicates a resolving stoppage, whereas a vehicle that remains motionless ($d_{\text{cand}} \approx 0.0\text{ m}$) provides direct evidence of continued curb obstruction. Table 3.2 summarizes the complete six-dimensional feature vector.

| Feature | Symbol | Type | Range | Phase |
| :---- | :----: | :----: | :----: | :----: |
| Stationary Ratio | $R$ | Float | [0.0, 1.0] | Snapshot |
| Surrounding Vehicle Count | $C$ | Integer | $\ge 0$ | Snapshot |
| Mean Surrounding Displacement | $\bar{d}$ | Float | $\ge 0.0\text{ m}$ | Snapshot |
| Surrounding Displacement Std. Dev. | $\sigma_d$ | Float | $\ge 0.0\text{ m}$ | Snapshot |
| Ratio Change | $\Delta R$ | Float | [-1.0, +1.0] | Monitoring |
| Candidate Post-Trigger Displacement | $d_{\text{cand}}$ | Float | $\ge 0.0\text{ m}$ | Monitoring |

*Table 3.2. Feature vector specification.*
*Note: Snapshot features evaluate the preceding lookback window ($N = 3\text{ s}$) at trigger time $t_{\text{trig}}$. Monitoring features evaluate movement changes across the subsequent window ($W = 15\text{ s}$). For events with $C = 1$, standard deviation evaluates to $\sigma_d = 0.0\text{ m}$ by definition. When $|\mathcal{S}(t_{\text{end}})| = 0$, $\Delta R$ is assigned a missing value routed by XGBoost. When $C = 0$, contextual stream features are unavailable and the event is evaluated using the stationary dwell threshold.*

The six-dimensional feature vector captures how a candidate vehicle's dwell compares to the surrounding traffic stream under different urban road conditions. In an illegal parking scenario, an isolated stationary car sits next to active traffic, producing a low stationary ratio ($R \to 0.0$), active nearby vehicles ($C \ge 1$), normal stream speed ($\bar{d} \gg \varepsilon_s$), steady ratio change ($\Delta R \approx 0.0$), and no candidate movement ($d_{\text{cand}} \approx 0.0\text{ m}$). In temporary traffic stops, such as red lights or pedestrian crossings, surrounding vehicles stop together ($R \to 1.0$, $C \ge 2$, $\bar{d} \to 0.0\text{ m}$) and then resume moving as the queue clears ($\Delta R < 0.0$, $d_{\text{cand}} \gg \varepsilon_s$). In severe gridlock, both the candidate car and surrounding traffic remain stopped throughout the observation window ($R \to 1.0$, $C \ge 2$, $\bar{d} \to 0.0\text{ m}$, $\Delta R \approx 0.0$, $d_{\text{cand}} \approx 0.0\text{ m}$).

The supervised XGBoost classifier evaluates these combined features to distinguish violations from legitimate stops. When no surrounding vehicles are present in the camera frame ($C = 0$), contextual stream features ($R, \bar{d}, \sigma_d$) are undefined; in this isolated condition, the event is evaluated based on whether the candidate vehicle maintained stationary dwell in the restricted zone across the 60-second threshold.

### Violation Classification

The feature vector is processed by an XGBoost gradient-boosted decision tree classifier [13], which outputs a binary classification of 1 for violation and 0 for non-violation. XGBoost trains an ensemble of decision trees sequentially, with each tree minimizing the regularized loss function of the preceding model. Split thresholds are learned directly from the training data distribution. The classifier evaluates the six numeric features specified in Table 3.2. Event ID and Track ID serve strictly as database keys to link predictions to on-screen vehicles and are excluded from model training.

The classification pipeline operates under several physical and operational boundaries on the roadway. First, moving vehicles passing in front of a stopped target vehicle can partially occlude its lower bounding box edge. Because planar homography projects the bottom-center coordinate to the ground surface, vertical bounding box fluctuations produce apparent radial displacement spikes that can temporarily reset dwell timers. Furthermore, under ByteTrack's track lifecycle rules [76], unassociated tracks are retained across a buffer of $t_{\text{lost}} = 30$ frames before termination. If an inter-vehicle occlusion exceeds this duration, the tracker terminates the track and assigns a new identity upon reappearance, which must accumulate an additional $n = 90$ consecutive frames of history before becoming eligible for the surrounding pool $\mathcal{S}(t)$, thereby temporarily depressing the measured count $C$.

Second, traffic queue dynamics define operational limits. In extended vehicle queues, queue release waves propagate backward from downstream bottlenecks; if this release takes longer than the 15-second monitoring window to reach the candidate vehicle, the vehicle will remain stationary throughout observation ($\Delta R \approx 0$ and $d_{\text{cand}} \approx 0$), representing an operational limitation for fixed-window monitoring. Finally, the six-dimensional feature vector evaluates camera-wide kinematic aggregates rather than candidate-relative spatial measurements, such as metric distance to the curb or longitudinal separation to an immediate lead vehicle. The pipeline maintains a parsimonious input representation focused on kinematic traffic context, operating under the structural condition that spatial lane adherence is bounded by the designated restricted enforcement polygon.

## Ground Truth Annotation

To train the supervised XGBoost classifier and quantitatively evaluate framework performance, a reference ground-truth dataset will be established from the captured video recordings. This section defines the event-level ground truth annotation protocol, describes the recorded categorical and contextual attributes, and specifies the inter-annotator agreement procedure that will be used to validate labeling reliability.

### Event-Level Annotation

Annotation will be performed at the event level, not the frame level. The set of events to be annotated will be determined by running the Vision Pipeline and Event Trigger stages on the recorded footage and collecting all generated trigger events.

For each trigger event, the annotator will review the full video segment spanning 75 seconds (comprising 60 seconds before the trigger frame through the 15-second monitoring window). This extended pre-trigger duration ensures complete independent visual verification of the candidate vehicle's continuous 60-second stationary dwell requirement prior to trigger activation. Table 3.3 defines the ground truth annotation fields, where specific stop reasons will be recorded exclusively for non-violation events.

| Field | Type | Description |
| :---- | :---- | :---- |
| Event ID | Integer | Matches trigger event output |
| Track ID | Integer | Tracked vehicle identity |
| Label | Binary | Violation or Non-Violation |
| Stop reason | Categorical | Congestion / Pedestrian yielding / Loading / Other / N/A |
| Notes | Free text | Annotator remarks on ambiguous cases |

*Table 3.3. Ground truth annotation fields.*

An event will be formally annotated as a Violation when a candidate vehicle maintains a stationary dwell status within the restricted zone throughout the observation window in the absence of any observable external traffic impediment, such as simultaneous surrounding queueing, active pedestrian right-of-way crossing, or authorized curb operations. This annotation protocol aligns with empirical criteria established in intelligent transportation systems literature [17], [55], [71], wherein an on-street parking violation is defined as sustained stopping within a restricted curb corridor that cannot be attributed to dynamic upstream or downstream traffic constraints. In cases of persistent visual ambiguity where the cause of stopping cannot be determined from the video context, annotators will record detailed qualitative remarks, and such ambiguous events will be flagged for adjudication.

### Inter-Annotator Agreement

A randomly selected 20% subset of triggered events will be independently annotated by a second annotator. Agreement on the binary label will be measured using Cohen's kappa, defined as $\kappa = \frac{p_o - p_e}{1 - p_e}$, where $p_o$ is the observed agreement and $p_e$ is the expected agreement by chance. A minimum $\kappa \ge 0.80$ is targeted, corresponding to the upper boundary of "Substantial" agreement (0.61 to 0.80) and the threshold for "Almost Perfect" agreement (0.81 to 1.00) [39]. If agreement falls below this threshold, disagreements will be adjudicated and the operational definition refined before re-annotation.

## Experimental Setup and Model Evaluation

This section defines the experimental methodology that will be used to train, tune, and validate the proposed spatiotemporal classification framework. It outlines the stratified cross-validation partitioning scheme, the nested grid search procedure for XGBoost hyperparameter optimization, and the performance metrics that will be used for quantitative evaluation. In addition, it details the comparative evaluation protocol against an external single-frame detection baseline, feature ablation configurations, and parameter sensitivity analyses that will be conducted to examine pipeline robustness.

### Cross-Validation

Due to the anticipated limited size of the event dataset (constrained by the number of qualifying trigger events in the recording period), a fixed train-test split risks high variance in performance estimates. The study will utilize Grouped Stratified 5-Fold Cross-Validation, partitioning the dataset across contiguous recording blocks to eliminate temporal autocorrelation and environmental leakage [35]. The annotated dataset will be partitioned into five folds, grouped by continuous recording sessions and distinct stoppage episodes (ensuring overlapping temporal observation windows do not leak across partitions) and stratified by label to ensure that each fold mirrors the overall class proportion. In each iteration, four folds will serve as the training set and one fold as the held-out evaluation set. The process will repeat five times, and metrics will be reported as the mean ± standard deviation across folds [35].

### Hyperparameter Tuning

XGBoost hyperparameters will be tuned via grid search using a nested cross-validation framework, with 3-fold cross-validation executed within each outer training fold. As established by Varma and Simon [68], nesting the model selection and hyperparameter optimization loops within the training partition is essential to prevent data leakage and selection bias from artificially inflating performance estimates. The hyperparameter search space is defined in Table 3.4. The parameter configuration yielding the highest mean F1-score across inner cross-validation folds will be selected to train the final model for each outer evaluation fold.

| Hyperparameter | Search values |
| :---- | :---- |
| Number of estimators | 50, 100, 200 |
| Maximum tree depth | 3, 5, 7 |
| Learning rate $\eta$ | 0.01, 0.1, 0.3 |
| Minimum child weight | 1, 3, 5 |

*Table 3.4. XGBoost hyperparameter search space.*

No feature scaling or normalization is applied, as gradient-boosted decision trees are invariant to monotonic feature transformations. Because split decisions are based on threshold comparisons over feature values, the relative ordering of data points is preserved under any monotonic rescaling [13].

### Evaluation Metrics

Classification performance will be evaluated using precision, recall, F1-score, and false positive rate (FPR), derived from standard binary confusion matrix formulations in automated detection and surveillance [15]. In this operational context, True Positives (TP) represent correctly flagged genuine violations, False Positives (FP) denote legitimate stops incorrectly flagged as violations, True Negatives (TN) denote correctly suppressed legitimate stops, and False Negatives (FN) represent genuine violations that the classifier failed to identify. Table 3.5 outlines the mathematical definitions of these metrics.

| Metric | Definition |
| :---- | :---- |
| Precision | $\frac{TP}{TP + FP}$ |
| Recall | $\frac{TP}{TP + FN}$ |
| F1-Score | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| False Positive Rate | $\frac{FP}{FP + TN}$ |

*Table 3.5  Evaluation metrics.*

Metrics for the proposed method will be reported as mean ± standard deviation across the five cross-validation folds. Precision, recall, and F1-score will be reported for each class individually (violation and non-violation) to ensure that performance differences are not masked by class-level aggregation.

### Comparative Evaluation

The proposed method will be compared against a reimplementation of the detection logic described by Abella and Catedrilla [1]. Their method is a YOLOv8-based illegal parking detection framework that relies on composite detection classes to identify violations at the frame level. The method performs frame-by-frame inference without multi-object tracking or temporal dwell-time logic; a violation is identified when a composite class representing the spatial co-occurrence of a vehicle, a human, and a no-parking sign is detected within a single frame. This architectural design makes the method fundamentally distinct from the proposed spatiotemporal classifier, and this distinction is central to the comparative hypothesis.

To enable a controlled experimental comparison, Abella et al.'s detection logic will be reimplemented and evaluated under identical conditions:

* Shared Video Feed: Both methods will be evaluated on the exact same video footage recorded from the elevated camera setup at 1920 × 1080 resolution and 30 fps.  
* Shared Detector Architecture: Both methods will utilize the same YOLOv8 model family and pretrained base weights.  
* Shared Ground Truth: Both methods will be evaluated against the same event-level ground truth annotations.	

The reimplemented baseline model will be trained using COCO-format bounding boxes on test-site imagery to recognize two distinct conceptual categories of classes:

* Base Classes: Standard object detections, including cars, generic vehicles, motorcycles, persons, and no-parking signs.  
* Composite Classes: Relational co-occurrence classes, specifically vehicle-human and vehicles-human-no-parking-sign.

Where Abella et al.'s original manuscript omits critical implementation details, explicit specifications will be applied to ensure exact parity with YOLOv8n detector parameters, specifically utilizing an inference detection confidence threshold of 0.25 and an input resolution of 640×640 pixels to ensure complete reproducibility.

Because Abella et al.'s method produces frame-level detections while the proposed method produces event-level classifications, baseline detections will be temporally and spatially aggregated to the candidate event level. For each ground truth event, the aggregation procedure will operate as follows:

1. Define the Observation Window: The evaluation window will be set from the trigger frame $t_{\text{trig}}$ through the end of the monitoring period ($t_{\text{trig}} + W \times \text{fps}$).  
2. Scan for Candidate-Associated Composite Detections: The reimplemented detector will evaluate every frame within this temporal window for the composite violation class (vehicles-human-no-parking-sign). To map frame-level detections to the specific candidate event and prevent unrelated peripheral detections from corrupting candidate evaluation, a spatial association rule is enforced, attributing a composite detection to the candidate event only when its bounding box exhibits a spatial overlap with the candidate vehicle track ($\text{IoU} \geq 0.30$).  
3. Assign Event Classification: To aggregate associated frame detections to the event level, a 50% temporal presence density threshold will be implemented across the monitoring horizon. If an associated composite violation detection is present in at least 50% of the frames within the 15-second monitoring window, the event will be classified as a baseline violation. Otherwise, it will be classified as a non-violation.  
4. Compute Performance Metrics: Predictions from both methods will be evaluated against identical ground truth labels using the metrics defined in Table 3.5.

The evaluation scope for this comparison is strictly governed by the proposed method's trigger mechanism. Only vehicles that remain stationary in the restricted zone for at least $T = 60$ seconds generate an evaluation event; transient stoppages and short-duration violations under 60 seconds are excluded from the test pool. The reimplemented Abella et al. method will be evaluated exclusively on this pre-filtered event pool. Consequently, this experiment will measure discrimination accuracy within extended-duration stationary scenarios rather than general detection coverage across all traffic events.

This comparison will evaluate the specific hypothesis that context-unaware spatial co-occurrence models exhibit elevated false positive rates during extended collective traffic stops, and that spatiotemporal contextual classification suppresses this failure mode. The primary outcome metric is the false positive rate (FPR). Abella et al. reported lower precision for parking detection (74.67%) relative to littering detection (98.41%) [1]. The comparative evaluation in this study is designed to test the hypothesis that single-frame co-occurrence false positives frequently arise during collective traffic stoppages, and that incorporating spatiotemporal motion context can suppress these errors.

Furthermore, an operational cue dependency governs this comparative evaluation. Abella et al.'s composite architecture strictly requires the co-occurrence of a physical "No Parking" sign alongside the vehicle and person. Because Abella et al.'s composite architecture strictly requires the visual presence of a physical regulatory sign [1], statutory restricted zones lacking physical signposts cannot produce composite detections. To maintain a valid and informative experimental comparison, comparative evaluation is conducted specifically on roadway segments featuring visible regulatory signage, while documenting cue-dependence as an inherent structural constraint of co-occurrence models relative to spatiotemporal motion analysis.

A fundamental operational distinction exists in the violation definitions used by the two approaches. Abella et al. define a violation as the instantaneous spatial co-occurrence of specific object classes within a single frame, with no temporal duration requirement. The proposed method defines a violation as a vehicle that maintains stationary dwell within a legally designated restricted zone for at least 60 seconds without observable operational necessity. This structural divergence in target definition is intrinsic to the respective frameworks and forms a primary basis for analyzing their empirical behavior on shared evaluation footage.

### Feature Importance and Ablation

To interpret the trained XGBoost model, feature importance will be quantified using the average gain, which calculates the mean improvement in the regularized objective function contributed by a specific feature across all trees in which it appears [13]. These importance scores will be reported in the experimental results to provide interpretability regarding which aspects of the temporal traffic context the classifier relies on most heavily to form its decision boundaries. 

An ablation study will evaluate the marginal contribution of individual feature subsets by retraining and evaluating the classifier across three distinct configurations using the identical stratified 5-fold cross-validation protocol, specifically evaluating snapshot features only ($R, C, \bar{d}, \sigma_d$), monitoring features only ($\Delta R, d_{\text{cand}}$), and the full six-dimensional feature vector.

## Parameter Sensitivity Analysis

A parameter sensitivity analysis will examine the robustness of the proposed pipeline to variations in key extraction hyperparameters [61]. Following standard sensitivity analysis principles [61], a one-at-a-time (OAT) parameter perturbation will be conducted across the test values outlined in Table 3.6, wherein the full pipeline (feature extraction, classifier training, and 5-fold cross-validation) will be re-executed with the modified parameter value while holding all other parameters fixed at their calibrated defaults.

| Parameter | Default | Test values |
| :---- | :---- | :---- |
| Monitoring window ($W$) | 15 s | 5, 10, 15, 20, 30 s |
| Stationary threshold ($\varepsilon_s$) | 1.5× noise ceiling | 0.5×, 1.0×, 1.5×, 2.0× noise ceiling |
| Lookback window ($N$) | 3 s | 1, 2, 3, 5 s |

*Table 3.6. Parameter sensitivity analysis configurations.*

## Pipeline Parameter Summary

Table 3.7 consolidates all fixed and empirically determined parameters in the proposed pipeline.

| Parameter | Symbol | Value | Justification |
| :---- | :---- | :---- | :---- |
| Resolution | N/A | 1920 × 1080 | Device specification |
| Frame rate | fps | 30 | Native recording rate; no downsampling |
| Temporal trigger threshold | $T$ | 60 s | Operational threshold distinguishing transient traffic pauses from sustained parking |
| Lookback window | $N$ | 3 s (90 frames) | Empirical window ensuring $\ge 3.0$ m displacement for crawling vehicles ($\ge 1.0$ m/s), exceeding noise ceiling |
| Stationary threshold | $\varepsilon_s$ | Empirically determined | Depth-dependent noise calibration (detailed in Restricted Zone and Event Trigger) |
| Monitoring window | $W$ | 15 s | Upstream extraction parameter; sensitivity assessed in Parameter Sensitivity Analysis |
| Detection retention floor | $s_{\text{floor}}$ | 0.10 | Raw detection cutoff supplying candidate boxes to ByteTrack two-stage association |
| High-confidence track threshold | $s_{\text{high}}$ | 0.25 | Ultralytics default detection threshold [30] |
| ByteTrack match threshold | $\sigma_{\text{match}}$ | 0.8 | ByteTrack association threshold [76] |
| Low-confidence recovery bound | $\tau_{\text{low}}$ | 0.10 | Second-stage association bound for occluded detections [76] |
| Track confirmation window | $n_{\text{init}}$ | 3 frames | Empirical default; prevents single-frame false positives |
| Track retention buffer | $t_{\text{lost}}$ | 30 frames (1.0 s) | Bridges momentary occlusions; avoids premature track termination (detailed in Multi-Object Tracking) [76] |

*Table 3.7. Complete pipeline parameter summary.*

## References

1. Miguel Paulo C. Abella and Gene Marck B. Catedrilla. 2025\. Smart Surveillance of Illegal Parking and Littering Detection Using Yolo-Based Machine Learning Algorithms in the Municipality of Los Baños. In 2025 International Conference on Artificial Intelligence Security and Governance (ICAISG), December 12, 2025\. IEEE, Hangzhou, China, 61-64. https\://doi.org/10.1109/ICAISG68699.2025.11452147

2. Rami Akhawaji, Mohamed Sedky, and Abdel-Hamid Soliman. 2017\. Illegal Parking Detection Using Gaussian Mixture Model and Kalman Filter. In 2017 IEEE/ACS 14th International Conference on Computer Systems and Applications (AICCSA), October 2017\. IEEE, Hammamet, 840-847. https\://doi.org/10.1109/AICCSA.2017.212

3. Antonio Albiol, Laura Sanchis, Alberto Albiol, and José M. Mossi. 2011\. Detection of Parked Vehicles Using Spatiotemporal Maps. IEEE Trans. Intell. Transport. Syst. 12, 4 (December 2011), 1277-1291. https\://doi.org/10.1109/TITS.2011.2156791

4. Alvin Sarraga Alon. 2020\. A Machine Vision Detection of Unauthorized On-Street Roadside Parking in Restricted Zone: An Experimental Simulated Barangay-Environment. IJETER 8, 4 (April 2020), 1056-1061. https\://doi.org/10.30534/ijeter/2020/17842020

5. Fauzan Maftuh Alwafi, Boby Mugi Pratama, Phuong Thi Le, Bima Prihasto, and Jia-Ching Wang. 2024\. Enhanced Detection of Illegally Parked Vehicles Using YOLO and Good Feature to Track Methods. In 2024 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), December 03, 2024\. IEEE, Macau, Macao, 1-6. https\://doi.org/10.1109/APSIPAASC63619.2025.10849019

6. Bj Kayle Asilum, Rommel Bandola, Bernice Brisenio, Angelica Cubelo, Rachel Mae Falic, Clyde Pierre Monsalud, and John Cabaddu. 2025\. ASSESSING THE IMPACT OF “PARKING NG BAYAN” IN REGULATING VEHICLE PARKING IN PUBLIC AREAS OF VALENZUELA CITY. https\://doi.org/10.5281/ZENODO.15706773

7. Elarbi Badidi, Karima Moumane, and Firdaous El Ghazi. 2023\. Opportunities, Applications, and Challenges of Edge-AI Enabled Video Analytics in Smart Cities: A Systematic Review. IEEE Access 11, (2023), 80543-80572. https\://doi.org/10.1109/ACCESS.2023.3300658

8. Palakkumar B. Bhatt and Dipesh Kamdar. 2026\. Wrong Side Vehicle Detection and Illegal Parking Detection using the Centroid Method and YOLOv8. SSRG-IJECE 13, 6 (July 2026), 1-11. https\://doi.org/10.14445/23488549/IJECE-V13I6P101

9. Gary Bradski. 2000\. The OpenCV Library. Dr. Dobb’s Journal of Software Tools 25, 11 (November 2000), 120-125.

10. Saravanan C, Pavithra P, Sandhiya B, Suganthan A, Suseendran K, and Gopalakrishnan K. 2025\. Hybrid Parking Violation Detection System Using CNN with Gradient Boosting. In 2025 10th International Conference on Communication and Electronics Systems (ICCES), October 28, 2025\. IEEE, Coimbatore, India, 964-969. https\://doi.org/10.1109/ICCES67310.2025.11337215

11. Zakaria Charouh, Amal Ezzouhri, Mounir Ghogho, and Zouhair Guennoun. 2022\. Video Analysis and Rule-Based Reasoning for Driving Maneuver Classification at Intersections. IEEE Access 10, (2022), 45102-45111. https\://doi.org/10.1109/ACCESS.2022.3169140

12. Hao Chen, Yao Li, Yong Jia, Guangle Yao, and Ruipeng Zhu. 2026\. RKF-YOLO: A Lightweight Dual-Task Model for Illegal Parking Detection and License Plate Recognition on Edge Devices. Electronics 15, 12 (June 2026), 2638\. https\://doi.org/10.3390/electronics15122638

13. Tianqi Chen and Carlos Guestrin. 2016\. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, August 13, 2016\. ACM, San Francisco California USA, 785-794. https\://doi.org/10.1145/2939672.2939785

14. Dila Dede, Mehmet Ali Sarsıl, Ata Shaker, Olgu Altıntaş, and Onur Ergen. 2023\. Next-gen traffic surveillance: AI-assisted mobile traffic violation detection system. https\://doi.org/10.48550/ARXIV.2311.16179

15. Tom Fawcett. 2006\. An introduction to ROC analysis. Pattern Recognition Letters 27, 8 (June 2006), 861-874. https\://doi.org/10.1016/j.patrec.2005.10.010

16. Martin A. Fischler and Robert C. Bolles. 1981\. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM 24, 6 (June 1981), 381-395. https\://doi.org/10.1145/358669.358692

17. Jingqin Gao, Fan Zuo, Kaan Ozbay, Omar Hammami, and Murat Ledin Barlas. 2022\. A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision for cameras with low resolution and low frame rate. Transportation Research Part A: Policy and Practice 162, (August 2022), 137-154. https\://doi.org/10.1016/j.tra.2022.05.024

18. X. Gao, P.M. Birch, R.C.D. Young, and C.R. Chatwin. 2019\. Illegally Parked Vehicle Detection using Deep Learning and Key-point Tracking. In 9th International Conference on Imaging for Crime Detection and Prevention (ICDP-2019), 2019\. Institution of Engineering and Technology, London, UK, 2 (7-12). https\://doi.org/10.1049/cp.2019.1160

19. Glaciel A. Araneta, Dit Junrie B. Matias, and Dit Riah E. Encarnacion. 2025\. A Real-Time Illegal Parking Detection System with Automatic License Plate Number Recognition. Proceedings of International Exchange and Innovation Conference on Engineering & Sciences (IEICES) 11, (October 2025), 79-86. https\://doi.org/10.5109/7395501

20. Xiaolong Gong, Haiqing Liu, Yuehao Wang, Yaxin Wei, and Guoran Shi. 2026\. Unauthorized Expressway Parking Detection Based on Spatiotemporal Analysis of Vehicle-Structure Distances Using UAV Aerial Images. Vehicles 8, 3 (March 2026), 49\. https\://doi.org/10.3390/vehicles8030049

21. Haerunnisya Makmur, Wulandari, Muhammad Fajar B, Andi Baso Kaswar, Dyah Darma Andayani, Fhatiah Adiba, Abdul Wahid, and Satria Gunawan Zain. 2025\. Motorcycle Parking Violation Detection System Using YOLOv7 with Region of Interest Mapping and Object Area Calculation. Adv. technol. innov. 10, 1 (January 2025), 29-43. https\://doi.org/10.46604/aiti.2024.14075

22. Richard Hartley and Andrew Zisserman. 2004\. Multiple View Geometry in Computer Vision (2nd ed.). Cambridge University Press. https\://doi.org/10.1017/CBO9780511811685

23. Mohamed Yousuf Hasan, Salwa Fraish Said Hamed Al Siyabi, Emanuel O. Rances, Khalsa Salim Al Moqbali, Sara Khalid Al Amri, and Nasiyba Sulaiman Al Moqbali. 2025\. Artificial Intelligence-Based Traffic Violation Detection Model Using YOLOv8, OCR, and OpenCV-DNN on University Campuses. In 2025 IEEE First International Conference on Innovations in Engineering and Next-Generation Technologies for Sustainability (ICINVENTS), November 07, 2025\. IEEE, Coimbatore, India, 1-6. https\://doi.org/10.1109/ICINVENTS64613.2025.11401399

24. Waqas Hassan, Philip Birch, Rupert Young, and Chris Chatwin. 2012\. Real-time occlusion tolerant detection of illegally parked vehicles. Int. J. Control Autom. Syst. 10, 5 (October 2012), 972-981. https\://doi.org/10.1007/s12555-012-0514-2

25. George To Sum Ho, Yung Po Tsang, Chun Ho Wu, Wai Hung Wong, and King Lun Choy. 2019\. A Computer Vision-Based Roadside Occupation Surveillance System for Intelligent Transport in Smart Cities. Sensors 19, 8 (April 2019), 1796\. https\://doi.org/10.3390/s19081796

26. Derick Hosmani, Yuvraj Pulled, and Om Kurale. 2025\. A Deep Learning-Based Framework for Real-Time Illegal Parking Detection Using YOLOv5. International Journal of Creative Research Thoughts (2025).

27. Ruixue Huang, Lianghua Cheng, Zhenghan Li, Chaocan Xiang, and Yulan Guo. 2024\. iPatrol: Illegal Roadside Parking Detection Leveraging On-road Vehicle Crowdsensing. In 2024 IEEE/ACM 32nd International Symposium on Quality of Service (IWQoS), June 19, 2024\. IEEE, Guangzhou, China, 1-10. https\://doi.org/10.1109/IWQoS61813.2024.10682887

28. Ruixue Huang, Chaocan Xiang, and Yulan Guo. 2026\. When Vehicle Crowd-Sensing Meets Complex Driving: Robust Illegal Roadside Parking Detections. IEEE Trans. on Mobile Comput. 25, 7 (July 2026), 10688-10705. https\://doi.org/10.1109/TMC.2026.3665356

29. JiaWei Jiang, Yu-Chen Chen, and Hsun-Ping Hsieh. 2020\. Detection of Illegal Parking Events Using Spatial-Temporal Features. In Proceedings of the 28th International Conference on Advances in Geographic Information Systems, November 03, 2020\. ACM, Seattle WA USA, 667-668. https\://doi.org/10.1145/3397536.3428350

30. Glenn Jocher, Jing Qiu, and Ayush Chaurasia. 2023\. Ultralytics YOLOv8. https\://doi.org/10.5281/zenodo.7386512

31. Shital Kakad, Shivam Sawant, Sujal Tawale, Shrey Rai, Shreyash Mutha, and Ayush Tangde. 2025\. Parking Violation Detection using Computer Vision for Urban Traffic Management. In 2025 7th International Conference on Intelligent Sustainable Systems (ICISS), March 12, 2025\. IEEE, India, 886-891. https\://doi.org/10.1109/ICISS63372.2025.11076537

32. Shailendra Singh Kathait, Ashish Kumar, Samay Sawal, Ram Patidar, and Khushi Agrawal. 2025\. Computer Vision and Deep Learning based Approach for Violations due to Illegal Parking Detection. IJCA 186, 70 (March 2025), 9-13. https\://doi.org/10.5120/ijca2025924506

33. Ali Khanpour, Tianyi Wang, Afra Vahidi-Shams, Wim Ectors, Farzam Nakhaie, Amirhossein Taheri, and Christian Claudel. 2025\. UAV-Based Intelligent Traffic Surveillance System: Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis. https\://doi.org/10.48550/arXiv.2509.04624

34. Chin Kit Ng, Soon Nyean Cheong, Wen Wen-Jiun Yap, and Yee Loo Foo. 2018\. Outdoor Illegal Parking Detection System Using Convolutional Neural Network on Raspberry Pi. IJET 7, 3.7 (July 2018), 17\. https\://doi.org/10.14419/ijet.v7i3.7.16197

35. Ron Kohavi. 1995\. A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. In Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI), 1995\. 1137-1143. https\://doi.org/10.5555/1643031.1643047

36. P.Sai Kumar, Boggarapu Srinivasulu, P. Bhanu Teja, Yogitha Sai Dudhada, Nedary Joy Darren, and Sivakrishna Kondaveeti. 2026\. AI Powered Traffic Violations Detection. In 2026 International Conference on Machine Learning and Autonomous Systems (ICMLAS), March 11, 2026\. IEEE, Bangkok, Thailand, 327-330. https\://doi.org/10.1109/ICMLAS67792.2026.11483802

37. Song Kun, Huang Xing-Wei, Liu Ji-Ning, Lin Qiao, and Chen Jin-Xuan. 2025\. Abnormal Vehicle Event Detection Based on Deep Learning. In 2025 6th International Conference on Computing, Networks and Internet of Things (CNIOT), May 23, 2025\. IEEE, Shanghai, China, 1-7. https\://doi.org/10.1109/CNIOT65435.2025.11070971

38. Li-Chia Kuo and Huei-Yung Lin. 2024\. Illegal Parking Detection Based on Multi-Task Driving Perception. In 2024 IEEE Intelligent Vehicles Symposium (IV), June 02, 2024\. IEEE, Jeju Island, Korea, Republic of, 1865-1870. https\://doi.org/10.1109/IV55156.2024.10588798

39. J. Richard Landis and Gary G. Koch. 1977\. The Measurement of Observer Agreement for Categorical Data. Biometrics 33, 1 (March 1977), 159\. https\://doi.org/10.2307/2529310

40. J.T. Lee, M.S. Ryoo, M. Riley, and J.K. Aggarwal. 2009\. Real-Time Illegal Parking Detection in Outdoor Environments Using 1-D Transformation. IEEE Trans. Circuits Syst. Video Technol. 19, 7 (July 2009), 1014-1024. https\://doi.org/10.1109/TCSVT.2009.2020249

41. Jialin Li, Yipeng Zhao, Xuchen Wang, Yuxuan Chen, Hui Li, and Xi’ai Chen. 2023\. Detection of Illegal Parking Based on Deep Learning. In 2023 IEEE International Conference on Electrical, Automation and Computer Engineering (ICEACE), December 29, 2023\. IEEE, Changchun, China, 79-83. https\://doi.org/10.1109/ICEACE60673.2023.10442566

42. Michael James Lighthill and Gerald Beresford Whitham. 1955\. On kinematic waves II. A theory of traffic flow on long crowded roads. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences 229, 1178 (May 1955), 317-345. https\://doi.org/10.1098/rspa.1955.0089

43. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. 2014\. Microsoft COCO: Common Objects in Context. In Computer Vision - ECCV 2014, David Fleet, Tomas Pajdla, Bernt Schiele and Tinne Tuytelaars (eds.). Springer International Publishing, Cham, 740-755. https\://doi.org/10.1007/978-3-319-10602-1\_48

44. Dongwei Liu, Yiping Sun, Zhengning Li, Zhongyan Wang, and Yuwei Zhang. 2024\. HIGH-PRECISION DETECTION OF ILLEGAL PARKING USING DEEP LEARNING TECHNOLOGY. In Professional Development: Theoretical Basis and Innovative Technologies, February 2024\. International Science Group, Paris, France, 326-333. Retrieved September 27, 2026 from https\://isg-konf.com/professional-development-theoretical-basis-and-innovative-technologies/

45. Chayadevi M L, Samarth Uday, and Sourav Mantesh Shet. 2026\. SafeStreets: Deep Learning-Based Real-Time Traffic Violation Detection System. In 2026 International Conference on Intelligent and Innovative Technologies in Computing, Electrical and Electronics (IITCEE), January 22, 2026\. IEEE, Bangalore, India, 1-7. https\://doi.org/10.1109/IITCEE67948.2026.11394149

46. Akihiro Matsuda, Tomokazu Matsui, Yuki Matsuda, Hirohiko Suwa, and Keiichi Yasumoto. 2021\. A System for Real-time On-street Parking Detection and Visualization on an Edge Device. In 2021 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops), March 22, 2021\. IEEE, Kassel, Germany, 227-232. https\://doi.org/10.1109/PerComWorkshops51409.2021.9431076

47. David R Mcallister, Jingwei Liu, Catherine M Doyle, François L Moreau, and Hannah K Patel. 2025\. Hierarchical Deep Learning Architectures for Multiclass Violation Detection in Urban Surveillance Systems Authors. https\://doi.org/10.22541/essoar.176086852.28048212/v1

48. Carlos Morillo and José Magín Campos. 2014\. On-street Illegal Parking Costs in Urban Areas. Procedia \- Social and Behavioral Sciences 160, (December 2014), 342-351. https\://doi.org/10.1016/j.sbspro.2014.12.146

49. Muh Fajrin Bakri, Shahnaz Tasha Kurnia, Muhammad Fajar B, Andi Baso Kaswar, Dyah Darma Andayani, Fhatiah Adiba, Sanatang, and Syahrul. 2026\. Motorcycle Parking Violation Detection Using YOLOv12 Segmentation and ROI-Guided Orientation Analysis. Proc. eng. technol. innov. 33, (May 2026), 33-50. https\://doi.org/10.46604/peti.2026.16020

50. Mahmoud Owais, Ali Shehata, Abdou Shaband, and Ghada S. Moussa. 2025\. A Framework for Establishing an Automated Traffic Violation Detection System in New Assiut City Using Ordinary CCTV Units. Mansoura Engineering Journal 50, 3 (May 2025). https\://doi.org/10.58491/2735-4202.3293

51. Aryan Paode, Krishna Samdani, and Tanisha Verekar. 2026\. Automated Illegal Parking Fine System Using YOLOv8 and PaddleOCR on the Cloud. In 2026 3rd International Conference on Smart City and Information System (ICSCIS), January 30, 2026\. IEEE, Guangzhou, China, 173-177. https\://doi.org/10.1109/ICSCIS69190.2026.11453485

52. Hyunseong Park, Kapyeol Kim, Incheol Jeong, Jungil Jung, and Jinsoo Cho. 2025\. Special Vehicle Classification Algorithm-Based System for Dedicated Parking Zone Violation Detection in South Korea. IEEE Access 13, (2025), 7883-7901. https\://doi.org/10.1109/ACCESS.2025.3526862

53. Jinuk Park, Jaeyong Lee, Yongju Park, and Yongseok Lim. 2024\. Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams. Electronics 13, 20 (October 2024), 4097\. https\://doi.org/10.3390/electronics13204097

54. Ashish Singh Patel, Vivek Tiwari, Muneendra Ojha, and O.P. Vyas. 2023\. Ontology-based detection and identification of complex event of illegal parking using SPARQL and description logic queries. Chaos, Solitons & Fractals 174, (September 2023), 113774\. https\://doi.org/10.1016/j.chaos.2023.113774

55. Xinggan Peng, Rongzihan Song, Qi Cao, Yue Li, Dongshun Cui, Xiaofan Jia, Zhiping Lin, and Guang-Bin Huang. 2022\. Real-Time Illegal Parking Detection Algorithm in Urban Environments. IEEE Trans. Intell. Transport. Syst. 23, 11 (November 2022), 20572-20587. https\://doi.org/10.1109/TITS.2022.3180225

56. Putra Yudha Pranata and Wahyono. 2022\. Improved Visual Background Extractor for Illegally Parked Vehicle Detection. International Journal of Intelligent Engineering and Systems 15, 3 (June 2022), 416-426. https\://doi.org/10.22266/ijies2022.0630.35

57. Alok Ranjan, Prasant Misra, Arunchandar Vasan, Sunil Krishnakumar, and Anand Sivasubramaniam. 2019\. City Scale Monitoring of On-Street Parking Violations with StreetHAWK. In Proceedings of the 6th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation, November 13, 2019\. ACM, New York NY USA, 31-40. https\://doi.org/10.1145/3360322.3360841

58. Sahara Ameer, Abdul Rahuman, Vithushanth, Randhima Dinalankara, and Udaya Wijenayake. 2024\. Traffic Violation Detection System. CONTRE 1, 1 (May 2024). https\://doi.org/10.31357/contre.v1i1.7390

59. Tarun Saini, Madhumita Guha, Sachin Kumar Bairwa, Vandita Dutt, Kajal Singh, and Dinesh Singh. 2025\. Drone-UP: Drone-based Unauthorized Parking Detection Method for Edge-devices. In 2025 International Joint Conference on Neural Networks (IJCNN), June 30, 2025\. IEEE, Rome, Italy, 1-7. https\://doi.org/10.1109/IJCNN64981.2025.11228564

60. K. Saiveena and P. Praveen. 2025\. Real-Time Occlusion-Aware Vehicle Detection and Behavior Analysis Framework for Intelligent Traffic Enforcement. In 2025 10th International Conference on Communication and Electronics Systems (ICCES), October 28, 2025\. IEEE, Coimbatore, India, 829-834. https\://doi.org/10.1109/ICCES67310.2025.11336884

61. Andrea Saltelli, Marco Ratto, Terry Andres, Francesca Campolongo, Jessica Cariboni, Debora Gatelli, Michaela Saisana, and Stefano Tarantola. 2007\. Global Sensitivity Analysis. The Primer (1st ed.). Wiley. https\://doi.org/10.1002/9780470725184

62. Md. Mostafa Kamal Sarker, Cai Weihua, and Moon Kyou Song. 2015\. Detection and Recognition of Illegally Parked Vehicles Based on an Adaptive Gaussian Mixture Model and a Seed Fill Algorithm. Journal of information and communication convergence engineering 13, 3 (August 2015), 197-204. https\://doi.org/10.6109/jicce.2015.13.3.197

63. Nabin Sharma, Sushish Baral, May Phu Paing, and Rathachai Chawuthai. 2023\. Parking Time Violation Tracking Using YOLOv8 and Tracking Algorithms. Sensors 23, 13 (June 2023), 5843\. https\://doi.org/10.3390/s23135843

64. Kang Tang, Yu Sun, and Xiaoyang Zhong. 2024\. Real-time Detection of Illegally Parked Vehicles in Roadside Parking Areas Based on Intelligent Video Terminals. In Proceedings of the 2024 International Conference on Image Processing, Intelligent Control and Computer Engineering, July 19, 2024\. ACM, Qingdao China, 302-308. https\://doi.org/10.1145/3691016.3691066

65. Joshua M. Torres, Miguel Antonio E. Gumaru, Ezekiel Aaron N. Luna, Leonard H. Quiroga, Kyle Joash Generalao, and Kanny Krizzy D. Serrano. 2023\. An Automated Classification of Vehicles and Violation Detection in Special Purpose Lanes. In 2023 IEEE Region 10 Symposium (TENSYMP), September 06, 2023\. IEEE, Canberra, Australia, 1-6. https\://doi.org/10.1109/TENSYMP55890.2023.10223478

66. Martin Treiber and Arne Kesting. 2013\. Traffic Flow Dynamics: Data, Models and Simulation. Springer Berlin Heidelberg, Berlin, Heidelberg. https\://doi.org/10.1007/978-3-642-32460-4

67. Chi-Yi Tsai and Yi-Quan Zhou. 2026\. Dashcam-Based Illegal Parking Detection Using Two-Stage Pseudo-Labeling and End-to-End Deep Learning Framework. IEEE Access 14, (2026), 69183-69196. https\://doi.org/10.1109/ACCESS.2026.3690972

68. Sudhir Varma and Richard Simon. 2006\. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7, 1 (December 2006), 91\. https\://doi.org/10.1186/1471-2105-7-91

69. Wahyono, Alexander Filonenko, and Kang-Hyun Jo. 2016\. Detecting illegally parked vehicle based on cumulative dual foreground difference. In 2016 IEEE 14th International Conference on Industrial Informatics (INDIN), July 2016\. IEEE, Poitiers, France, 772-775. https\://doi.org/10.1109/INDIN.2016.7819263

70. J G Wardrop. 1952\. ROAD PAPER. SOME THEORETICAL ASPECTS OF ROAD TRAFFIC RESEARCH. Proceedings of the Institution of Civil Engineers 1, 3 (May 1952), 325-362. https\://doi.org/10.1680/ipeds.1952.11259

71. Xuemei Xie, Chenye Wang, Shu Chen, Guangming Shi, and Zhifu Zhao. 2017\. Real-Time Illegal Parking Detection System Based on Deep Learning. In Proceedings of the 2017 International Conference on Deep Learning Technologies, June 02, 2017\. ACM, Chengdu China, 23-27. https\://doi.org/10.1145/3094243.3094261

72. Qiong Yang and Lifeng Yu. 2021\. Recognition of Taxi Violations Based on Semantic Segmentation of PSPNet and Improved YOLOv3. Scientific Programming 2021, (November 2021), 1-13. https\://doi.org/10.1155/2021/4520190

73. Zhengcong Yin, Haoyi Xiong, Xun Zhou, Daniel W. Goldberg, Dave Bennett, and Chong Zhang. 2019\. A Deep Learning based Illegal Parking Detection Platform. In Proceedings of the 3rd ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery, November 05, 2019\. ACM, Chicago IL USA, 32-35. https\://doi.org/10.1145/3356471.3365233

74. Haiyan Zhang, Ao Zhang, Chao Tian, Yilin Liu, Jibin Wang, Zhiwen Zhuang, Guxue Gao, and Yuanyuan Wang. 2026\. A lightweight multiscale network and segmentation collaborative method for detecting illegal parking on tactile paving. Applied Soft Computing 203, (November 2026), 115967\. https\://doi.org/10.1016/j.asoc.2026.115967

75. Linlin Zhang, Xiang Yu, Abdulateef Daud, Abdul Rashid Mussah, and Yaw Adu-Gyamfi. 2024\. Application of 2D Homography for High Resolution Traffic Data Collection using CCTV Cameras. https\://doi.org/10.48550/arXiv.2401.07220

76. Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. 2022\. ByteTrack: Multi-object Tracking by Associating Every Detection Box. In Computer Vision - ECCV 2022, Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella and Tal Hassner (eds.). Springer Nature Switzerland, Cham, 1-21. https\://doi.org/10.1007/978-3-031-20047-2\_1

77. Yining Zhang. 2023\. Real-time vehicle detection and tracking based on the combination of YOLOv7 and ByteTrack. ACE 4, 1 (May 2023), 267-271. https\://doi.org/10.54254/2755-2721/4/20230467

78. Z. Zhang. 2000\. A flexible new technique for camera calibration. IEEE Trans. Pattern Anal. Machine Intell. 22, 11 (November 2000), 1330-1334. https\://doi.org/10.1109/34.888718

79. Xizhen Zhou, Xueqi Ding, Jie Yan, and Yanjie Ji. 2023\. Spatial heterogeneity of urban illegal parking behavior: A geographically weighted Poisson regression approach. Journal of Transport Geography 110, (June 2023), 103636\. https\://doi.org/10.1016/j.jtrangeo.2023.103636

80. Yi-Quan Zhou and Chi-Yi Tsai. 2025\. Development and validation of an instance segmentation-based system for illegal parking violation detection. IET Conf. Proc. 2025, 15 (August 2025), 502-506. https\://doi.org/10.1049/icp.2025.2602

81. Erik Bochinski, Volker Eiselein, and Thomas Sikora. 2017\. High-Speed tracking-by-detection without using image information. In *2017 14th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)*, August 2017\. IEEE, Lecce, Italy, 1-6. https\://doi.org/10.1109/AVSS.2017.8078516 

82. Robert Herman and Ilya Prigogine. 1979\. A Two-Fluid Approach to Town Traffic. Science 204, 4389 (1979), 148-151. https\://doi.org/10.1126/science.204.4389.148

83. Shraddha Chaudhary, Sreedevi Indu, and Santanu Chaudhury. 2018\. Video-based road traffic monitoring and prediction using dynamic Bayesian networks. IET Intelligent Transport Systems 12, 3 (April 2018), 169-176. https\://doi.org/10.1049/iet-its.2016.0336

84. Jerome Revaud and Martin Humenberger. 2021\. Robust Automatic Monocular Vehicle Speed Estimation for Traffic Surveillance. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021\. IEEE, Montreal, QC, Canada, 4551-4561. https\://doi.org/10.1109/ICCV48922.2021.00451

85. Raphael E. Stern, Shumo Cui, Maria Laura Delle Monache, Rahul Bhadani, Matt Bunting, Miles Churchill, Nathaniel Hamilton, R'mani Haulcy, Hannah Pohlmann, Fangyu Wu, Benedetto Piccoli, Benjamin Seibold, Jonathan Sprinkle, and Daniel B. Work. 2018\. Dissipation of stop-and-go waves via control of autonomous vehicles: Field experiments. Transportation Research Part C: Emerging Technologies 89 (March 2018), 205-221. https\://doi.org/10.1016/j.trc.2018.02.005

86. James C. Williams, Hani S. Mahmassani, and Robert Herman. 1985\. Analysis of Traffic Network Flow Relations and Two-Fluid Model Parameter Sensitivity. Transportation Research Record 1005 (1985), 95-106. https\://onlinepubs.trb.org/Onlinepubs/trr/1985/1005/1005-012.pdf
