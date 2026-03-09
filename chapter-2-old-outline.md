

# Chapter 2: Review of Related Literature — Complete Writing Guide

---

## 2.1 — Overview of Illegal Parking Detection Systems

---

### 2.1.1 — Classification by Camera Modality

**How to write it:**

Categorize all 27 papers by their camera type. Present this table:

| Camera Modality | Papers | Count |
|---|---|---|
| Fixed CCTV | Gao (2022), Kathait (2025), Wahyono (2017), Wahyono/ViBe (2022), Sarker (2015), Akhawaji (2017), Xie (2017), Lee (2009), Alwafi (2024), Ng (2018), Sharma (2023), Kakad (2024), Magalona (2024), Park H. (2025), CVROSS (2019) | 15 |
| Dashcam | Park J. (2024), Zhou, Kuo, Peng (2022), Liu | 5 |
| UAV/Drone | UAV paper, Saini (2024) | 2 |
| Mobile Crowdsensing | iPatrol/Huang (2023), StreetHAWK (2019), Patel (2023) | 3 |
| Simulated/Controlled | Alon (2020) | 1 |
| Insufficient detail to classify | Traffic Violation Detection System | 1 |

State that fixed CCTV dominates the literature. State that this study focuses on fixed-camera systems. State how the rest of the chapter is organized section by section.

**Length:** 1 page including table.

---

### 2.1.2 — Scope of the Review

**How to write it:**

One paragraph. State this review covers 27 studies published between 2009 and 2025. State the databases or sources searched if applicable. State that the review examines four aspects of each system: detection method, tracking method, violation decision logic, and reported challenges. State that all camera modalities are reviewed for completeness but primary analysis focuses on fixed-camera implementations.

**Length:** Half a page.

---

## 2.2 — Vehicle Detection Methods

---

### 2.2.1 — Background Subtraction Approaches

**How to write it:**

Explain background subtraction in two to three sentences. A statistical model of the scene is built over time. New frames are compared against this model. Pixels deviating from the model are classified as foreground.

Discuss each variant used in the reviewed papers:

**Gaussian Mixture Model (GMM):**
- Sarker et al. (2015) — Adaptive GMM for foreground detection. Classifies objects as vehicles using duty ratio. Reported 100% tracking accuracy in morning, 98.2% evening, 95.5% night.
- Akhawaji et al. (2017) — GMM combined with Kalman Filter. Tested on i-LIDS and PVTEA datasets. Reported significant performance drops in rain.
- Wahyono et al. (2017) — Cumulative dual-rate foreground difference using TWO GMM models. Short-term model adapts quickly, long-term model adapts slowly. A region that is foreground in the long-term model but background in the short-term model is classified as a newly stationary object. Reported precision 1.0 and recall 1.0 on i-LIDS with tracking.

**Visual Background Extractor (ViBe):**
- Wahyono/ViBe (2022) — Modified ViBe algorithm with updated background propagation mechanism. Reported precision 1.0 but recall 0.60 on i-LIDS.

**1-D Transformation:**
- Lee et al. (2009) — Projects 2D image into 1D representation for blob segmentation. Reduces computational cost. Detected 2 of 3 events in Austin TX test footage.

Present a comparison table:

| Method | Paper | Dataset | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Adaptive GMM | Sarker (2015) | Own (highway, university, urban) | Not reported | Not reported | Tracking accuracy 95.5-100% |
| GMM + Kalman | Akhawaji (2017) | i-LIDS, PVTEA | 0.33-0.39 | 0.38 | 0.35-0.39 |
| Dual GMM | Wahyono (2017) | i-LIDS | 1.0 | 1.0 | 1.0 |
| Dual GMM | Wahyono (2017) | ISLab | 0.92 | 1.0 | 0.96 |
| ViBe | Wahyono (2022) | i-LIDS | 1.0 | 0.60 | 0.75 |
| 1-D Transform | Lee (2009) | i-LIDS, Austin TX | Not reported | 2 of 3 events | Not reported |

Close with strengths and limitations. No training data needed. Real-time capable on low-power hardware. But sensitive to illumination changes, weather, and shadows. These methods dominated earlier work but have been largely replaced by deep learning approaches.

**Length:** 2-2.5 pages including table.

---

### 2.2.2 — Deep Learning Object Detectors

**How to write it:**

State that deep learning detectors learn to identify and localize objects by class from training data. They do not require a background model and are more robust to environmental variation than background subtraction.

**YOLO Family — present chronologically with technical detail for each version used:**

**YOLOv3:**
- Gao et al. (2022) used YOLOv3 as one of two detectors alongside Mask R-CNN. Used for object-level detection on low-resolution traffic camera feeds.
- Peng et al. (2022) used a YOLOv3-based encoder for detecting "minimal illegal units" in dashcam images.
- Patel et al. (2023) used YOLOv3 within their ontology-based mobile surveillance system.

**YOLOv5:**
- Magalona et al. (2024) used YOLOv5s. Reported mAP 91%, precision 87.2%, recall 87.5%, F1 87.3%. Also reported 224 motorcycles misclassified as vehicles.
- Kakad et al. (2024) used YOLOv5 for CCTV-based detection. Tested under daylight, low light, night, and adverse weather.

**YOLOv7:**
- Kuo et al. used YOLOv7 for dashcam-based detection alongside BoT-SORT tracking and PARSeq for plate recognition.

**YOLOv8:**
- Sharma et al. (2023) used YOLOv8 with DeepSORT and OC-SORT comparison. Reported DeepSORT MOTA 1.0 at best, OC-SORT MOTA 0.76 at worst.
- Alwafi et al. (2024) used YOLOv8 with optical flow. Reported precision 91.6%, recall 82.7%, mAP50 87.5% for car detection.
- Kathait et al. (2025) used YOLOv8 with DeepSORT and polygon-based zone filtering.
- Zhou et al. used YOLOv8 for instance segmentation. Reported mAP50 99.1%.
- UAV paper used YOLOv8x for vehicle detection and YOLOv8x-seg for road segmentation.

**Other Architectures:**

**SSD (Single Shot MultiBox Detector):**
- Xie et al. (2017) used SSD for real-time detection. Reported 99% detection accuracy.
- StreetHAWK (2019) used customized MobileNetV2 + SSD for smartphone deployment.

**Mask R-CNN:**
- Gao et al. (2022) used pretrained Mask R-CNN as primary detector, compared with YOLOv3.

**MobileNet Variants:**
- Alon (2020) deployed MobileNet SSD on Raspberry Pi 4b. Reported 96.16% mean accuracy for legal vehicles, 98.93% for illegal.
- Park H. et al. (2025) used MobileNet V2 for license plate area preprocessing before YOLOv8 classification.

**Custom CNNs:**
- Ng et al. (2018) trained a CNN using NVIDIA DIGITS for binary classification of cropped image regions.
- Liu et al. designed a multi-task neural network (MTPN) detecting tire-ground contact points and parking line regions. Reported 98.1% comprehensive accuracy.

**Present a summary table:**

| Detector Family | Papers | Best Reported Accuracy |
|---|---|---|
| GMM Background Subtraction | Sarker (2015), Akhawaji (2017), Wahyono (2017) | Precision 1.0 (Wahyono, i-LIDS) |
| ViBe | Wahyono (2022) | Precision 1.0, Recall 0.60 |
| YOLOv3 | Gao (2022), Peng (2022), Patel (2023) | Varies by application |
| YOLOv5 | Magalona (2024), Kakad (2024) | mAP 91%, F1 87.3% |
| YOLOv7 | Kuo | Not separately reported |
| YOLOv8 | Sharma (2023), Alwafi (2024), Kathait (2025), Zhou, UAV | mAP50 up to 99.1% |
| SSD | Xie (2017), StreetHAWK (2019) | 99% (Xie) |
| Mask R-CNN | Gao (2022) | Used for comparison |
| MobileNet | Alon (2020), Park H. (2025) | 96-98% |
| Custom CNN/MTPN | Ng (2018), Liu | 84-98% |

**Closing paragraph:** Deep learning detectors, particularly YOLO variants, dominate recent work. Detection accuracy consistently exceeds 85% under normal conditions. YOLOv8 is the most adopted detector in 2023-2025 publications. Detection methodology is mature and is not the focus of this study.

**Length:** 3-4 pages including table.

---

## 2.3 — Vehicle Tracking Methods

---

### 2.3.1 — Filter-Based Tracking

**How to write it:**

**Kalman Filter:**
- Explain briefly: predicts object position in next frame from prior motion, updates prediction when new detection arrives.
- Akhawaji et al. (2017) used Kalman Filter for tracking detected foreground blobs across frames.
- Gao et al. (2022) used IoU matching between consecutive frame detections rather than a formal tracker. Bounding boxes in the current frame are matched to previous frame boxes using Intersection over Union. If IoU exceeds a threshold, the detection is considered the same vehicle.

---

### 2.3.2 — Deep Learning Trackers

**How to write it:**

**DeepSORT:**
- Explain briefly: extends SORT by adding a deep appearance feature extractor. Matches detections across frames using both Kalman-predicted motion and visual similarity. Handles short-term occlusion by maintaining track identity for several frames without detection.
- Used by Sharma et al. (2023) and Kathait et al. (2025).
- Sharma et al. compared DeepSORT directly against OC-SORT. DeepSORT achieved MOTA 1.0 at Location 1. Both trackers struggled at Location 2 in rain.

**OC-SORT:**
- Explain briefly: Observation-Centric SORT. Designed to handle occlusion better by using virtual trajectory generation during occlusion periods.
- Used by Magalona et al. (2024) and Sharma et al. (2023).
- Sharma et al. reported OC-SORT MOTA dropping to 0.76 at Location 2 due to "rainy weather conditions and the presence of visually similar vehicles in close proximity."

**BoT-SORT:**
- Used by Kuo et al. for dashcam tracking. Combines camera motion compensation with appearance features, suitable for moving camera platforms.

Present a tracker comparison table:

| Tracker | Papers | Best MOTA | Worst MOTA | Key Limitation Reported |
|---|---|---|---|---|
| Kalman Filter | Akhawaji (2017) | Not reported | Not reported | Dependent on GMM detection quality |
| IoU Matching | Gao (2022) | Not reported | Not reported | No appearance features, relies on spatial overlap only |
| DeepSORT | Sharma (2023), Kathait (2025) | 1.0 | 0.90 | Reflections on windows caused false tracks |
| OC-SORT | Magalona (2024), Sharma (2023) | 1.0 | 0.76 | ID switching in rain and with similar vehicles |
| BoT-SORT | Kuo | Not reported | Not reported | Not separately evaluated |

---

### 2.3.3 — Optical Flow Methods

**How to write it:**

Explain briefly: optical flow computes pixel-level motion vectors between consecutive frames. Instead of tracking object identities, it measures WHETHER and HOW pixels are moving.

- Alwafi et al. (2024) used Lucas-Kanade optical flow for vehicle motion tracking. Good Features to Track algorithm selects feature points, optical flow tracks their displacement.
- Park J. et al. (2024) used dense optical flow for host vehicle movement estimation and bounding box center displacement for target vehicle movement estimation.

---

### 2.3.4 — Systems Without Explicit Tracking

**How to write it:**

Some systems do not maintain persistent vehicle IDs across frames.

- Xie et al. (2017) processes frames independently with SSD. Relies on spatial overlap between consecutive frame detections to associate them temporally.
- Ng et al. (2018) classifies a cropped image region as occupied or empty each frame. No individual vehicle identity is maintained.
- Alwafi et al. (2024) noted the limitation of this approach: "difficulty in vehicle ID recognition since YOLO processes each frame independently, potentially restarting the nonmovement time calculation upon timeout."
- Wahyono/ViBe (2022) demonstrated the consequence quantitatively: without tracking-based temporal verification, precision was 0.04. With tracking, precision reached 1.0. This is the most dramatic evidence in the reviewed literature that raw detection without temporal tracking context produces near-total false positive rates.

---

### 2.3.5 — Summary of Tracking Landscape

**How to write it:**

One paragraph. DeepSORT and OC-SORT are the most common in recent work. Optical flow provides motion measurement without identity tracking. Systems without tracking face severe precision problems as demonstrated by Wahyono/ViBe (2022). Tracking is a necessary component but is not the focus of this study. The tracking output — specifically bounding box centroid positions per frame per tracked vehicle — provides the data foundation that the violation decision logic operates on.

**Length for all of 2.3:** 3-3.5 pages including table.

---

## 2.4 — Violation Decision Logic

---

### 2.4.1 — Zone Definition Approaches

**How to write it:**

Describe every approach used across the papers for defining where violations can occur.

**Manual ROI Polygons:**
- Most common approach. A human operator draws a polygon on a reference frame marking the restricted zone.
- Used by Akhawaji et al. (2017), Xie et al. (2017), Ng et al. (2018), Kakad et al. (2024), Alwafi et al. (2024), Sharma et al. (2023), Magalona et al. (2024).
- Binary check: is the vehicle centroid inside or outside the polygon.

**Curb Detection Zones:**
- Gao et al. (2022) defined a Curb Detection Zone (CDZ) along the curb lane. Additionally defined a pedestrian detection zone on the crosswalk and a vehicle detection zone near the intersection for their signal filtering logic.
- More structured than a single ROI. Multiple zones serve different functions in the decision pipeline.

**Polygon-Based Spatial Filtering:**
- Kathait et al. (2025) defined restricted zones as polygons with vertex coordinates. Vehicle centroids are checked against polygon containment.

**Homography-Based Geofencing:**
- UAV paper used homography projection to map aerial camera views to ground-plane coordinates. Crosswalk boundaries defined in projected space.
- Requires manual calibration of homography matrix per scene.

**Distance-Based Measurement:**
- Zhou et al. measured pixel distance between vehicle segmentation mask and red line marking mask. Violation flagged if distance is less than 5 pixels.
- Does not use a predefined zone. Instead computes spatial relationship between vehicle and marking dynamically.

**Spatial Relationship Analysis:**
- Peng et al. (2022) detects "minimal illegal units" — combinations of wheel plus road marking — and uses IoU overlap between these units and vehicle regions.
- Liu et al. analyzes positional relationship between tire-ground contact points and parking line pixel regions.

Present a zone approach table:

| Zone Approach | Papers | Requires Manual Setup | Dynamic or Static |
|---|---|---|---|
| Manual ROI Polygon | Akhawaji, Xie, Ng, Kakad, Alwafi, Sharma, Magalona | Yes — draw polygon once per camera | Static |
| Curb Detection Zone + additional zones | Gao (2022) | Yes — define multiple zones | Static |
| Polygon Spatial Filtering | Kathait (2025) | Yes — define polygon vertices | Static |
| Homography Geofencing | UAV paper | Yes — calibrate homography matrix | Static |
| Distance to Marking | Zhou | No — computed from segmentation | Dynamic |
| Spatial Relationship (tire-to-line) | Peng (2022), Liu | No — computed from detection | Dynamic |

---

### 2.4.2 — Temporal Thresholds

**How to write it:**

Explain the concept. Most systems require a vehicle to remain stationary in the zone for a minimum duration before flagging a violation. This prevents brief stops (dropping off passengers, waiting for pedestrians) from being flagged.

Present the full threshold table:

| Paper | Threshold Value | How Stationarity Is Measured |
|---|---|---|
| UAV paper | 10 seconds | Estimated speed near zero |
| Gao et al. (2022) | ~10 seconds | IoU matching between frames showing same vehicle present |
| Xie et al. (2017) | 15 seconds | Bounding box overlap across frames |
| Wahyono et al. (2017) | 60 seconds | Frame counter after stable region detected |
| Wahyono/ViBe (2022) | 60 seconds | Frame counter after static region classified |
| Akhawaji et al. (2017) | 60 seconds | Tracked object remains in ROI |
| Ng et al. (2018) | 60 seconds | CNN classification of region as occupied |
| Kakad et al. (2024) | 60 seconds | Vehicle detected in ROI |
| Magalona et al. (2024) | 90 seconds | Tracked centroid in ROI |
| Sharma et al. (2023) | ~15 minutes | Tracked vehicle ID in parking area |

Note two observations without editorializing:

> "The 60-second threshold is the most commonly adopted value, used by five independent studies. However, no reviewed study provides empirical justification for their chosen value or reports how performance varies with different threshold settings."

> "The method for measuring stationarity varies. Some systems use bounding box displacement (pixel movement between frames). Others use IoU overlap between consecutive detections. Others use background model classification of static regions. The underlying measurement determines how precisely the system can distinguish a truly stationary vehicle from a slow-moving one."

---

### 2.4.3 — Additional Decision Criteria

**How to write it:**

Present every additional input that any system uses beyond zone membership and time threshold.

**Vehicle Type Classification:**
- CVROSS (2019) distinguishes private cars from trucks and vans. Trucks and vans are classified as loading activity rather than parking.
- Gao et al. (2022) assigns impact correction factors by vehicle type: 2.0 for trucks, 1.0 for passenger cars.
- Park H. et al. (2025) classifies vehicles as EV, disabled, or regular using license plate symbols and windshield signs to check authorization for dedicated parking zones.

**License Plate Recognition:**
- Magalona et al. (2024) integrates ALPR after violation detection to identify the offending vehicle. Plate classification precision 1.0, recall 0.997.
- Park H. et al. (2025) uses Tesseract OCR to read plate characters.
- Kuo et al. uses PARSeq for license plate recognition in dashcam footage.

**Traffic Signal State:**
- Gao et al. (2022) implemented a pedestrian zone proxy to infer signal phase from crosswalk activity.
- Kathait et al. (2025) described checking traffic signal color (red, yellow, green) in their decision logic.
- These are discussed further in Section 2.6.4.

**Spatial Relationship Voting:**
- Peng et al. (2022) classifies six types of parking violations using majority voting on detected minimal illegal units. Each unit type (wheel on line, wheel past line, etc.) votes for a violation category.

**Ontological Reasoning:**
- Patel et al. (2023) represents detection data in an ontology (ipark) and uses SPARQL and description logic queries to determine violation status. Checks vehicle speed, movement status, track position relative to lane boundaries.

**Centerline Filtering:**
- Park J. et al. (2024) detects the road centerline to filter vehicles in the opposite lane from being classified as stopped.

---

### 2.4.4 — Decision Outputs

**How to write it:**

What does each system produce after the violation decision.

**Binary violation flag:** Most systems output violation or non-violation per event.

**Alert mechanisms:**
- Kakad et al. (2024) activates a hardware buzzer and captures a screenshot when a violation is detected.
- Magalona et al. (2024) triggers license plate capture upon violation.

**Impact estimation:**
- Gao et al. (2022) goes beyond binary detection. Uses queueing theory to estimate the traffic delay caused by each illegal parking event. Assigns higher impact to trucks and to events in bus lanes.

**Evidence collection:**
- Multiple systems capture the frame or video segment as evidence for enforcement purposes.

---

### 2.4.5 — Comprehensive Comparison of Fixed-Camera Systems

**How to write it:**

This is the master reference table. Include ONLY fixed-camera papers. Every subsequent discussion references back to this table.

| Paper | Year | Detector | Tracker | Zone Type | Threshold | Additional Criteria | Output |
|---|---|---|---|---|---|---|---|
| Lee et al. | 2009 | 1-D Transform | Blob tracking | Restricted area | Not stated | None | Binary flag |
| Sarker et al. | 2015 | Adaptive GMM | Detection-based | Prohibited region | Calculated time | Duty ratio vehicle classification | Binary flag |
| Wahyono et al. | 2017 | Dual GMM | Detection-based | Not stated | 60 seconds | None | Alert |
| Xie et al. | 2017 | SSD | Spatial overlap | ROI | 15 seconds | None | Binary flag |
| Akhawaji et al. | 2017 | GMM | Kalman Filter | ROI | 60 seconds | None | Binary flag |
| Ng et al. | 2018 | Custom CNN | None (region classification) | Cropped region | 60 seconds | None | Binary flag |
| CVROSS | 2019 | Not stated | Not stated | Roadside zone | Not stated | Vehicle type (loading vs parking) | Occupation status |
| Wahyono/ViBe | 2022 | ViBe | Frame counter | Static region | 60 seconds | None | Alert |
| Gao et al. | 2022 | Mask R-CNN / YOLOv3 | IoU matching | Curb Detection Zone | ~10 seconds | Signal proxy, vehicle type, impact estimation | Impact score |
| Sharma et al. | 2023 | YOLOv8 | DeepSORT / OC-SORT | Parking area | ~15 minutes | None | Time violation flag |
| Alwafi et al. | 2024 | YOLOv8 | Optical flow | ROI | Duration-based | None | Binary flag |
| Kakad et al. | 2024 | YOLOv5 | None stated | ROI | 60 seconds | None | Buzzer + screenshot |
| Magalona et al. | 2024 | YOLOv5s | OC-SORT | ROI | 90 seconds | License plate recognition | Flag + plate capture |
| Kathait et al. | 2025 | YOLOv8 | DeepSORT | Polygon zones | Frame count threshold | Signal state (described, not fully implemented) | Binary flag |
| Park H. et al. | 2025 | YOLOv8 + MobileNet V2 | None stated | Dedicated zone | None (type-based) | Vehicle type (EV, disabled, regular) | Authorization check |

**Length for all of 2.4:** 5-6 pages including both tables.

---

## 2.5 — Deployment Platforms

---

### 2.5.1 — Fixed CCTV Infrastructure

**How to write it:**

State that 15 of 27 reviewed papers use fixed cameras. This is the largest category. Advantages: existing urban infrastructure, no additional camera hardware cost, static geometry simplifies zone definition, continuous 24/7 monitoring. Constraints documented in literature: low resolution (Gao et al. noted public cameras have low resolution and low frame rate), fixed viewing angle limits coverage, camera angle affects detection accuracy (Gao et al. attributed lower performance at Site C to perpendicular camera angle).

---

### 2.5.2 — Mobile Camera Platforms

**How to write it:**

Cover dashcam, UAV, and crowdsensing together as mobile alternatives.

**Dashcam systems** (Park J., Zhou, Kuo, Peng, Liu): Camera moves with the vehicle. Must compensate for ego-motion. Park J. used optical flow for host vehicle movement estimation. Kuo used BoT-SORT with camera motion compensation. Advantage: covers large road networks during normal driving. Disadvantage: each location observed only briefly as vehicle passes.

**UAV systems** (UAV paper, Saini): Aerial overhead view. Advantage: wide coverage area, reduced occlusion from overhead angle. Disadvantage: small object size at altitude, limited flight time, wind, requires manual homography calibration.

**Mobile crowdsensing** (iPatrol, StreetHAWK, Patel): Multiple vehicles act as distributed sensors. iPatrol covered 233 km of roads in Chongqing. StreetHAWK used smartphones on dashboards. Patel used a dedicated surveillance vehicle with sensors. Advantage: city-scale coverage. Disadvantage: variable sensing quality, coordination challenges, StreetHAWK noted "limited violation assessment range of 15m."

---

### 2.5.3 — Edge Computing Deployment

**How to write it:**

Several systems target processing at the camera site rather than on a server.

- Alon (2020) deployed MobileNet SSD on Raspberry Pi 4b with Pi camera.
- Ng et al. (2018) used Raspberry Pi with IP camera.
- StreetHAWK processed on smartphone.
- UAV paper targets edge devices on drones.

Advantage: no network dependency, real-time response, low cost. Constraint: limits model complexity and detection accuracy. Lighter architectures (MobileNet, SSD) used instead of heavier models (Mask R-CNN, YOLOv8x).

**Length for all of 2.5:** 2-2.5 pages.

---

## 2.6 — Reported Challenges Across Studies

---

### 2.6.1 — Environmental Conditions

**How to write it:**

**Rain and adverse weather:**
- Akhawaji et al. (2017): worst performance on PVTEA102b "due to the raining weather." Reported "sun appears several times, which causes different reflected light" and "reduced of the light source." 25 false positives and 8 false negatives across test sets.
- Sharma et al. (2023): OC-SORT "consistently encountered challenges particularly in the presence of rainy conditions." "Algorithm frequently switched vehicle IDs." MOTA dropped to 0.76 at Location 2.
- Kakad et al. (2024): tested under "Adverse Weather" as a separate scenario.

**Lighting and night conditions:**
- Wahyono et al. (2017): nighttime false positives "due to lighting condition of scenes."
- Lee et al. (2009): "difficulties in handling night scenes due to headlights." One false positive at night "due to the continuous glare of the headlights."
- Sarker et al. (2015): tracking accuracy dropped from 100% (morning highway) to 95.5% (urban road at night).
- Wahyono/ViBe (2022): "false-positive errors due to gradual or sudden changes in illumination."
- Kathait et al. (2025): described earlier methods as "unreliable in dynamic lighting conditions and complex urban scenes."
- Kuo et al.: "performs poorly in scenarios involving low light."

**Shadows and time-of-day variation:**
- Alon (2020): tested under "distinct time of a day under various light and shadow patterns." Performance dropped from 99% to 77.4% in specific lighting combinations.

Present a summary table:

| Challenge | Papers Reporting It | Worst Documented Impact |
|---|---|---|
| Rain | Akhawaji (2017), Sharma (2023), Kakad (2024) | MOTA 0.76, 25 false positives |
| Night/lighting | Wahyono (2017), Lee (2009), Sarker (2015), Wahyono (2022), Kathait (2025), Kuo | Tracking accuracy drop to 95.5%, nighttime false positives |
| Shadows | Alon (2020) | Detection drop to 77.4% in specific conditions |

---

### 2.6.2 — Occlusion

**How to write it:**

Occlusion is the most frequently cited challenge across the 27 papers.

**Vehicle-to-vehicle occlusion:**
- Gao et al. (2022): "an illegally parked passenger car was hidden (occluded) by another illegally parked truck." Caused underestimation of violation count at Site B.
- Wahyono et al. (2017): "system may fail to detect multiple occluded illegally parked vehicles which are located close to each other."
- Wahyono/ViBe (2022): "suffers from detection failure due to adjacent vehicles."
- Park J. et al. (2024): "multiple vehicles are moving simultaneously, making it inevitable for some vehicles to be partially obscured by others." Example: "a large bus blocks the view of the stopped vehicle."
- UAV paper: "occlusion occurred when multiple vehicles accumulated near the crosswalk, temporarily blocking parts of the predefined crossing area."
- iPatrol: "quality of vehicle-mounted videos could be seriously affected if the observed vehicles are obscured by surrounding vehicles."
- Lee et al. (2009): "system was not able to segment vehicles that arrive together" — two vehicles parking close together merged into one blob.

**Partial visibility:**
- Patel et al. (2023): "vehicle is not detected when it is covered less than half of it."
- StreetHAWK (2019): "single wheel of the bike was mostly visible with its lower torso masked by other parked bikes." "People were found sitting on parked bikes."

---

### 2.6.3 — Detection and Tracking Errors

**How to write it:**

**Misclassification:**
- Magalona et al. (2024): 224 motorcycles misclassified as vehicles. 56 vehicles misclassified as motorcycles. 14 vehicles misclassified as plate numbers.
- Kuo et al.: vehicles in opposing lanes "incorrectly classified as legal parking" and legally parked vehicles "mistakenly classified as illegal parking."
- Ng et al. (2018): missed a yellow-colored vehicle due to "insufficient training image samples."

**Tracking identity errors:**
- Sharma et al. (2023): OC-SORT "frequently switched vehicle IDs." "Reflections from vehicles on the windows interfered with the vehicle detection process." "Reflection of the vehicle was considered as a vehicle."
- Alwafi et al. (2024): "difficulty in vehicle ID recognition since YOLO processes each frame independently, potentially restarting the nonmovement time calculation upon timeout."

**Segmentation and estimation errors:**
- Lee et al. (2009): "system was not able to segment vehicles that arrive together."
- Patel et al. (2023): false positives from "incorrect estimation of the length of the vehicle when it is far from the camera."
- Gao et al. (2022): "overall detection accuracy on low-resolution images, especially for far-side detection and relatively lower accuracy on non-passenger cars."

**Resolution and range:**
- Gao et al. (2022): low resolution and low frame rate of public traffic cameras limiting detection accuracy.
- StreetHAWK (2019): "single shot detection range is limited to 15m."
- Park H. et al. (2025): "number of datasets collected for the evaluation was slightly insufficient."
- Peng et al. (2022): "traffic lines or the parking lot corner are not clear" causing detection difficulty.

---

### 2.6.4 — Non-Parking Stationary Vehicles

**How to write it:**

Present this as one challenge category among the others. Same level of treatment as 2.6.1, 2.6.2, 2.6.3. Do not editorialize or emphasize it over the other categories. Let the evidence speak at the same volume.

**Fixed-camera studies that documented this challenge:**

Gao et al. (2022):
> "Another challenge in detecting illegal parking events in a dense urban area is differentiating between vehicles stopped for red lights and those illegally parked." They also noted vehicles stopping for "queue or pedestrians." Their response was a pedestrian zone proxy: monitoring crosswalk activity to infer signal phase, and filtering vehicles whose stop duration matched the inferred red phase duration. This required defining two additional detection zones (pedestrian zone on crosswalk, vehicle zone near intersection) and depended on pedestrian detection accuracy and pedestrian presence during red phases.

Wahyono et al. (2017):
> "The detecting stable region step might obtain some false regions due to a traffic jam, stopped vehicles and standing human at the traffic light intersection." Their response was a 60-second tracking timer, alerting only after sustained stationarity. This temporal filter rejects stops shorter than 60 seconds but cannot distinguish a vehicle parked for 61 seconds from a vehicle stuck in congestion for 61 seconds.

Kathait et al. (2025):
> "During the detection of violation due to illegal parking it is important that vehicles which are at stop due to red or yellow traffic signal are also considered." They described flagging vehicles only when "the signal is green or the vehicle is in the given parking for a long time." Their dataset did not include visible traffic signals, so signal-based filtering was not implemented in their evaluation.

**Other-modality studies that documented this challenge:**

UAV paper:
> Applied a 10-second minimum threshold to "ignore temporary stops such as pedestrian crossings." Recognized that "delays can also result from other factors such as signal timing, pedestrian flow, or random fluctuations." Reported "detection challenges particularly under heavy congestion, which occasionally obstructed clear visibility of the crosswalk zones."

Park J. et al. (2024):
> Tested under "extreme road conditions" including "traffic congestion" and "lined-up vehicles." Confirmed system could identify a stopped taxi even when frame was "congested with vehicles." Built centerline detection to filter opposite-lane vehicles that "appear to approach the host vehicle rapidly" and might be "mistakenly identified as stopped."

iPatrol / Huang et al. (2023):
> Evaluated under "Empty," "Normal," and "Crowded" traffic conditions. "In crowded traffic condition, the precision and recall drop due to vehicle occlusion." Precision approximately 95% in empty traffic, 81.5% in crowded traffic.

Present a table summarizing the responses:

| Paper | Problem Stated | Response Implemented | Response Requires |
|---|---|---|---|
| Gao et al. (2022) | Red light stops, queue, pedestrian stops | Pedestrian zone proxy for signal inference | Visible crosswalk, pedestrian presence, two additional zones |
| Wahyono et al. (2017) | Traffic jam, signal intersection stops | 60-second tracking timer | Nothing additional, but cannot distinguish long congestion from parking |
| Kathait et al. (2025) | Red and yellow signal stops | Signal color check described | Visible traffic signal head (not implemented due to dataset) |
| UAV paper | Temporary crosswalk stops, signal timing | 10-second minimum threshold | Nothing additional, but only filters very brief stops |
| Park J. (2024) | Congestion, lined-up vehicles | Centerline filter for opposite lane | Visible centerline marking |
| iPatrol (2023) | Crowded traffic degradation | None specific | — |

---

### 2.6.5 — Summary of Challenges

**How to write it:**

One paragraph. State that challenges fall into four broad categories: environmental conditions, occlusion, detection/tracking errors, and traffic conditions affecting violation logic. Environmental and occlusion challenges are the most frequently cited and affect the detection and tracking stages of the pipeline. Traffic condition challenges affect the decision logic stage — they cause the system to misinterpret legitimate vehicle behavior as violations. Each category has been addressed to varying degrees in individual studies but none has been fully resolved.

**Length for all of 2.6:** 5-6 pages including two tables.

---

## 2.7 — Summary and Research Direction

**How to write it:**

---

### 2.7.1 — State of the Field

**How to write it:**

One to two paragraphs. Summarize what the review established.

> "The reviewed literature demonstrates a maturing field across all pipeline stages. Vehicle detection using deep learning methods, particularly YOLO variants, consistently achieves accuracy above 85% under normal conditions and has become the standard approach in recent publications. Multi-object tracking using DeepSORT and OC-SORT enables persistent vehicle identification, though performance degrades under rain and with visually similar vehicles. The dominant violation decision logic — zone membership combined with temporal threshold — is computationally efficient and adopted by the majority of fixed-camera studies."

> "Challenges remain across all pipeline stages. Environmental conditions degrade detection and tracking accuracy. Occlusion causes missed detections and is documented in more studies than any other single challenge. Detection and tracking errors including misclassification and ID switching contribute to both false positives and false negatives."

---

### 2.7.2 — Research Focus

**How to write it:**

This is where you narrow to your specific problem and state your direction. This is the ONLY place in Chapter 2 where you advocate for your study.

**Paragraph 1 — Identify the specific pattern:**

> "Among the documented challenges, the problem of non-parking stationary vehicles affecting violation decision logic appears across multiple independent studies. Gao et al. (2022) explicitly identified differentiating red light stops from parking as a challenge and built a pedestrian zone proxy. Wahyono et al. (2017) reported traffic jams and signal intersections causing false regions in their stationary vehicle detection. Kathait et al. (2025) stated that signal state must be considered to avoid false positives. Each study developed a response specific to their system and dataset."

**Paragraph 2 — Observe what is common and what is absent:**

> "The existing responses share a characteristic: each evaluates the candidate vehicle against a fixed external reference — a temporal duration, a pedestrian zone, or a signal head. All examine the candidate vehicle's behavior in isolation from other vehicles in the scene. Whether surrounding tracked vehicles are also stationary — information derivable from standard tracking output without additional detection modules or scene-specific calibration — is not used as a decision input in any reviewed system."

**Paragraph 3 — State what this study will do:**

> "This study focuses on the traffic condition challenge specifically. It proposes a context-aware decision logic layer that assesses surrounding vehicle displacement data from the existing tracking module before confirming a violation. If surrounding vehicles are also stationary, the candidate's behavior is consistent with collective traffic conditions. If the candidate is stationary while surrounding vehicles move, the behavior is consistent with individual parking. The following chapter describes the planned methodology for implementing and evaluating this approach."

**Length for all of 2.7:** 1.5-2 pages.

---

## Complete Section Flow

| Section | Title | Length |
|---|---|---|
| 2.1 | Overview of Illegal Parking Detection Systems | 1.5 pages |
| 2.1.1 | Classification by Camera Modality | — |
| 2.1.2 | Scope of the Review | — |
| 2.2 | Vehicle Detection Methods | 5-6 pages |
| 2.2.1 | Background Subtraction Approaches | — |
| 2.2.2 | Deep Learning Object Detectors | — |
| 2.3 | Vehicle Tracking Methods | 3-3.5 pages |
| 2.3.1 | Filter-Based Tracking | — |
| 2.3.2 | Deep Learning Trackers | — |
| 2.3.3 | Optical Flow Methods | — |
| 2.3.4 | Systems Without Explicit Tracking | — |
| 2.3.5 | Summary of Tracking Landscape | — |
| 2.4 | Violation Decision Logic | 5-6 pages |
| 2.4.1 | Zone Definition Approaches | — |
| 2.4.2 | Temporal Thresholds | — |
| 2.4.3 | Additional Decision Criteria | — |
| 2.4.4 | Decision Outputs | — |
| 2.4.5 | Comprehensive Comparison of Fixed-Camera Systems | — |
| 2.5 | Deployment Platforms | 2-2.5 pages |
| 2.5.1 | Fixed CCTV Infrastructure | — |
| 2.5.2 | Mobile Camera Platforms | — |
| 2.5.3 | Edge Computing Deployment | — |
| 2.6 | Reported Challenges Across Studies | 5-6 pages |
| 2.6.1 | Environmental Conditions | — |
| 2.6.2 | Occlusion | — |
| 2.6.3 | Detection and Tracking Errors | — |
| 2.6.4 | Non-Parking Stationary Vehicles | — |
| 2.6.5 | Summary of Challenges | — |
| 2.7 | Summary and Research Direction | 1.5-2 pages |
| 2.7.1 | State of the Field | — |
| 2.7.2 | Research Focus | — |

**Total estimated length:** 24-28 pages.
