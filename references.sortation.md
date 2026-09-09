# Chapter 2 Literature Mapping and Reference Sortation

This document maps all 52 research papers from `notebooklm-extraction.md` into the locked Chapter 2 outline structure, providing the paper title, author, year, and a concise summary of the exact technical mechanism.

---

## Section 2.1: Traditional Methods for Illegal Parking Detection

### 2.1.1 Background Subtraction Approaches
Criteria: Classical computer vision using Gaussian Mixture Models, frame differencing, or ViBe background modeling to identify stationary foreground objects.

- Wahyono et al. (2016) [Cumulative Dual Foreground Differences for Illegally Parked Vehicles Detection]: Uses dual-background GMM with a 60-second stationary threshold.
- Sarker et al. (2015) [Detection and Recognition of Illegally Parked Vehicles Based on an Adaptive Gaussian Mixture Model and a Seed Fill Algorithm]: Uses adaptive GMM and seed fill algorithms.
- Alkhawaji et al. (2017) [Illegal Parking Detection using Gaussian Mixture Model and Kalman Filter]: Uses GMM foreground detection with Kalman filtering and a 60-second timer.
- Yudha Pranata (2022) [Improved Visual Background Extractor for Illegally Parked Vehicle Detection]: Uses improved ViBe (I-ViBe) with a 60-second threshold.
- Hassan et al. (2012) [Real-Time Occlusion Tolerant Detection of Illegally Parked Vehicles]: Uses GMM pixel classification with Segmentation History Images (SHI).

### 2.1.2 Motion and Point Tracking Approaches
Criteria: Classical feature tracking using optical flow, Harris corners, or spatiotemporal transforms without neural networks.

- Albiol et al. (2011) [Detection of Parked Vehicles Using Spatiotemporal Maps]: Uses Harris corners and spatiotemporal maps, measuring the horizontal dimension of stationary pixel clusters without object tracking.
- Lee et al. (2009) [Real-Time Illegal Parking Detection in Outdoor Environments Using 1-D Transformation]: Uses 1-D projection transformations and morphological operations to detect parked vehicle blobs.

---

## Section 2.2: Deep Learning Methods for Illegal Parking Detection

### 2.2.1 Spatial Detection Approaches
Criteria: Deep learning object detectors evaluating violations in a single frame based on spatial overlap with an ROI, painted lines, or signs, with no temporal tracking across frames.

- Abella and Catedrilla (2025) [Smart Surveillance of Illegal Parking and Littering Detection Using YOLO-Based Machine Learning Algorithms]: Uses YOLOv8 to detect composite classes (vehicle, human, no-parking sign) in a single frame.
- Alon (2020) [A Machine Vision Detection of Unauthorized On-Street Roadside Parking in Restricted Zone]: Uses MobileNet SSD with a single-frame zoning check.
- Torres et al. (2023) [An Automated Classification of Vehicles and Violation Detection in Special Purpose Lanes]: Uses YOLOv5 with a bus lane polygon mask without tracking.
- Li et al. (2023) [Detection of Illegal Parking Based on Deep Learning]: Uses DeepLabv3+ semantic segmentation and YOLOv5l for single-frame area overlap.
- Makmur et al. (2025) [Motorcycle Parking Violation Detection System Using YOLOv7 with Region of Interest Mapping]: Uses YOLOv7 flagging a violation when 50 percent of the motorcycle area overlaps an ROI.
- Ng et al. (2018) [Outdoor Illegal Parking Detection System Using Convolutional Neural Networks]: Uses iConvParkNet (AlexNet variant) with sliding-window classification on static ROI windows without tracking.
- Park et al. (2025) [Special Vehicle Classification Algorithm-Based System for Dedicated Parking Zone Violation Detection in South Korea]: Uses YOLOv8 and MobileNetV2 to classify EV and disability signs in dedicated bays.

### 2.2.2 Temporal Tracking Approaches
Criteria: Deep learning detection paired with multi-object tracking to evaluate dwell time via countdown timers, trajectory tracking, or machine learning on temporal features.

- Sharma et al. (2023) [Parking Time Violation Tracking Using YOLOv8 and DeepSORT]: Uses YOLOv8 with DeepSORT and OC-SORT to track vehicle dwell-time limits.
- Xie et al. (2017) [Real-Time Illegal Parking Detection System Based on Deep Learning (SSD)]: Uses SSD with IoU tracking and a 15-second dwell threshold.
- Araneta et al. (2025) [A Real-Time Illegal Parking Detection System with Automatic License Plate Number Recognition]: Uses YOLOv5s with OC-SORT and a 1-minute 30-second timer.
- Alwafi et al. (2024) [Enhanced Detection of Illegally Parked Vehicles Using YOLO and Good Feature to Track Methods]: Uses YOLOv8 with Lucas-Kanade optical flow countdown for zero displacement.
- Kakad et al. (2025) [Parking Violation Detection using Computer Vision for Urban Traffic Management]: Uses YOLOv5 with a 60-second timer in a restricted zone.
- Kathait et al. (2025) [Computer Vision and Deep Learning based Approach for Violations due to Illegal Parking Detection]: Uses YOLO with DeepSORT and frame persistence rules.
- Song et al. (2025) [Abnormal Vehicle Event Detection Based on Deep Learning]: Uses vehicle trajectory tracking with deep neural networks classifying spatiotemporal kinematic features.
- Jiang et al. (2020) [Detection of Illegal Parking Events Using Spatial-Temporal Features]: Extracts spatiotemporal features and applies XGBoost for event modeling.
- Saiveena and Praveen (2025) [Real-Time Occlusion-Aware Vehicle Detection and Behavior Analysis Framework for Intelligent Traffic Enforcement]: Uses deep learning with Kalman filtering and DeepSORT for stationary dwell detection.
- Chayadevi et al. (2026) [SafeStreets: Deep Learning-Based Real-Time Traffic Violation Detection System]: Uses YOLOv9 with DeepSORT for frame-level vehicle tracking.
- Ameer et al. (2024) [Traffic Violation Detection System]: Uses YOLOv5 with DeepSORT and polygon intersection tracking.
- Kumar et al. (2026) [AI Powered Traffic Violations Detection]: Uses YOLO/Faster R-CNN with StrongSORT and rule engine validation.

### 2.2.3 Road and Scene Feature Approaches
Criteria: Vision systems that verify parking violations against external physical infrastructure or static road markings (traffic lights, crosswalks, painted curb lines, drivable area masks).

- Gao et al. (2022) [A New Curb Lane Monitoring and Illegal Parking Impact Estimation]: Uses YOLOv3 and Mask R-CNN, mapping a pedestrian crosswalk queue as an indirect proxy to filter out red-light stops.
- Tsai and Zhou (2026) [Dashcam-Based Illegal Parking Detection Using Two-Stage Pseudo-Labeling and End-to-End Deep Learning Framework]: Uses YOLOv8l-seg calculating minimum distance between bounding boxes and painted red-line contours (5-pixel threshold).
- Zhou and Tsai (2025) [Development and Validation of an Instance Segmentation-Based System for Illegal Parking Violation Detection]: Uses YOLOv8 instance segmentation measuring Euclidean distance to red-line contours.
- Peng et al. (2022) [Real-Time Illegal Parking Detection Algorithm in Urban Environments]: Uses minimal illegal units and IoU overlap against road lines.
- Kuo and Lin (2024) [Illegal Parking Detection Based on Multi-Task Driving Perception]: Uses YOLOv7 with BoT-SORT, utilizing drivable areas and road markings as contextual cues.
- Hasan et al. (2025) [Artificial Intelligence-Based Traffic Violation Detection Model Using YOLOv8, OCR, and OpenCV-DNN on University Campuses]: Uses YOLOv8 with lane and parking space occupancy thresholding.
- Paode et al. (2026) [Automated Illegal Parking Fine System Using YOLOv8 and PaddleOCR on the Cloud]: Uses multi-video comparison with cloud license plate OCR to verify persistence across time gaps.
- Liu et al. (2024) [High Precision Detection of Illegal Parking Using Deep Learning Technology]: Uses a multitask network (MTPN) to extract parking line skeletal features and tire-ground contact points.
- Dede et al. (2023) [Next-Gen Traffic Surveillance AI-Assisted Mobile Traffic Violation Detection System]: Uses YOLOv5 with StrongSORT analyzing vehicle movement relative to detected traffic signs.
- Patel et al. (2023) [Ontology-Based Detection and Identification of Complex Event of Illegal Parking Using SPARQL and Description Logic Queries]: Uses YOLOv3 comparing vehicle bottom boundaries with lane lines via SWRL/SPARQL queries.
- Yang and Yu (2021) [Recognition of Taxi Violations Based on Semantic Segmentation of PSPNet and Improved YOLOv3]: Uses PSPNet semantic segmentation calculating overlap area with road lanes.
- Khanpour et al. (2025) [UAV-Based Intelligent Traffic Surveillance System]: Uses UAV homography to map crosswalk areas, flagging stops longer than 10 seconds.

### 2.2.4 Nearby Vehicle Motion Approaches
Criteria: Systems that evaluate the dynamic movement or relative velocity of surrounding moving vehicles rather than static road objects.

- Huang et al. (2024) [iPatrol: Illegal Roadside Parking Detection Leveraging On-Road Vehicle Crowdsensing]: Uses smartphone crowdsensing to calculate relative velocity (lambda = 1.1 m/s) between the host vehicle and roadside targets.
- Huang et al. (2026) [When Vehicle Crowd-Sensing Meets Complex Driving: Robust Illegal Roadside Parking Detections]: Uses crowdsensing with frame-to-frame feature shift to estimate speed and motionless state under complex driving.
- Park et al. (2024) [Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams]: Uses in-vehicle dashcams with dense optical flow to estimate distance and relative speed thresholds for stopped cars.
- Ranjan et al. (2019) [City Scale Monitoring of On-Street Parking Violations with StreetHAWK]: Uses mobile probe vehicles with rear-facing cameras monitoring roadside stops within a 50-meter span.
- Saini et al. (2025) [Drone-UP: Drone-Based Unauthorized Parking Detection Method for Edge-Devices]: Uses drone surveillance estimating relative speed between moving aerial cameras and parked cars.
- Gong et al. (2026) [Unauthorized Expressway Parking Detection Based on Spatiotemporal Analysis of Vehicle-Structure Distances Using UAV Aerial Images]: Uses UAV video analyzing vehicle-to-structure distance series using the Augmented Dickey-Fuller test and parking support ratios.

---

## Chapter 1 and Chapter 3 Foundational Citations
These papers do not propose an illegal parking computer vision violation algorithm, but serve as primary domain or methodological citations:

### General Illegal Parking Information
- Owais et al. (2025) [A Framework for Establishing an Automated Traffic Violation Detection System]: Surveillance planning and CCTV placement in urban road networks.
- Ho et al. (2019) [A Computer Vision-Based Roadside Occupation Surveillance System for Intelligent Transport in Smart Cities]: Smart city roadside occupancy surveillance.
- Bandola et al. (2025) [Assessing the Impact of "Parking Ng Bayan" in Regulating Vehicle Parking in Public Areas of Valenzuela City]: Philippine local municipal parking policy and impact.
- Morillo and Campos (2014) [On-Street Illegal Parking Costs in Urban Areas]: Economic and congestion costs of unauthorized parking.
- Badidi et al. (2023) [Opportunities, Applications, and Challenges of Edge-AI Enabled Video Analytics in Smart Cities: A Systematic Review]: Edge-AI video analytics review.
- Zhou et al. (2023) [Spatial Heterogeneity of Urban Illegal Parking Behavior: A Geographically Weighted Poisson Regression Approach]: Spatial statistical modeling of parking violations.

### Chapter 3: Methodology Citations
- Zhang et al. (2024) [Application of 2D Homography for High Resolution Traffic Data Collection Using CCTV Cameras]: Planar camera calibration and 2D homography metric mapping (Chapter 3: Homography Estimation).
- Zhang (2023) [Real-Time Vehicle Detection and Tracking Based on the Combination of YOLOv7 and ByteTrack]: Multi-object tracking benchmark evaluation. *(Note: Need ng review, general benchmarking paper pala to, baka i-omit na natin to sa bagong paper).*

---

## Final Verification Summary
- Traditional Methods (2.1.1 and 2.1.2): 7 papers
- Deep Learning Spatial Detection (2.2.1): 7 papers
- Deep Learning Temporal Tracking (2.2.2): 12 papers
- Road and Scene Features (2.2.3): 12 papers
- Nearby Vehicle Motion (2.2.4): 6 papers
- Background, Infrastructure, and Policy Context: 8 papers
- Total: Exactly 52 papers accounted for.
