# Chapter 2: Review of Computer Vision-Based Illegal Parking Detection Systems

---

**Opening paragraph (no section number):**

One introductory paragraph stating that this chapter reviews 37 (No need to actually mention the paper count) studies on computer vision-based illegal parking detection published between 2009 and 2025. The review examines how each system detects vehicles, tracks vehicles, decides whether a violation occurred, and what challenges were reported. Briefly note that the studies span fixed CCTV, dashcam, UAV, and crowdsensing platforms, but that this study focuses on fixed-camera systems. No table. No subsection.

---

## 2.1 — Vehicle Detection Methods

**Papers (37):**
- A Machine Vision Detection of Unauthorized On-Street Roadside Parking in Restricted Zone: An Experimental Simulated Barangay-Environment
- A Real-Time Illegal Parking Detection System with Automatic License Plate Number Recognition
- A computer vision-based roadside occupation surveillance system for intelligent transport in smart cities
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- City Scale Monitoring of On-Street Parking Violations with StreetHAWK
- Computer Vision and Deep Learning based Approach for Violations due to Illegal Parking Detection
- Cumulative Dual Foreground Differences for Illegally Parked Vehicles Detection
- Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams
- Detection and Recognition of Illegally Parked Vehicles Based on an Adaptive Gaussian Mixture Model and a Seed Fill Algorithm
- Development and Validation of an Instance Segmentation-Based System for Illegal Parking Violation Detection
- Drone-based Unauthorized Parking Detection Method for Edge-devices
- Enhanced Detection of Illegally Parked Vehicles Using YOLO and Good Feature to Track Methods
- HIGH-PRECISION DETECTION OF ILLEGAL PARKING USING DEEP LEARNING TECHNOLOGY
- Illegal Parking Detection using Gaussian Mixture Model and Kalman Filter
- Illegal Parking Detection Based on Multi-Task Driving Perception
- Improved Visual Background Extractor for Illegally Parked Vehicle Detection
- Motorcycle Parking Violation Detection System Using YOLOv7 with Region of Interest Mapping and Object Area Calculation
- Ontology-based detection and identification of complex event of illegal parking using SPARQL and description logic queries
- Outdoor Illegal Parking Detection System Using Convolutional Neural Network on Raspberry Pi
- Parking Time Violation Tracking Using YOLOv8 and Tracking Algorithms
- Parking Violation Detection using Computer Vision for Urban Traffic Management
- Real-Time Illegal Parking Detection System Based on Deep Learning (SSD)
- Real-Time Illegal Parking Detection in Outdoor Environments Using 1-D Transformation
- Real-Time Illegal Parking Detection Algorithm in Urban Environments
- Special Vehicle Classification Algorithm-Based System for Dedicated Parking Zone Violation Detection in South Korea
- Traffic Violation Detection System
- UAV-Based Intelligent Traffic Surveillance System Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis
- iPatrol: Illegal Roadside Parking Detection Leveraging On-road Vehicle Crowdsensing
- Abnormal Vehicle Event Detection Based on Deep Learning
- Automated Illegal Parking Fine System Using YOLOv8 and PaddleOCR on the Cloud
- Detection of Illegal Parking Events Using Spatial-Temporal Features
- Detection of Parked Vehicles Using Spatiotemporal Maps
- Real-Time Occlusion-Aware Vehicle Detection and Behavior Analysis Framework for Intelligent Traffic Enforcement
- Recognition of taxi violations based on semantic segmentation of PSPNet and improved YOLOv3
- Smart Surveillance of Illegal Parking and Littering Detection Using Yolo-Based Machine Learning Algorithms in the Municipality of Los Baños
- When Vehicle Crowd-Sensing Meets Complex Driving: Robust Illegal Roadside Parking Detections

**How to write it:**

One section, no subsections. Write it as a flowing narrative showing the progression from background subtraction methods (GMM, ViBe, dual foreground difference, 1-D transformation) in early work to deep learning detectors (SSD, Mask R-CNN, MobileNet, custom CNNs) to YOLO variants (v3 through v8) dominating recent publications. Use multi-citation clusters — group papers by method rather than discussing each one individually. The reader should come away understanding that vehicle detection is a mature problem, YOLO variants are the current standard, and accuracy consistently exceeds 85% under normal conditions. This section exists to establish that detection works well and is not where the remaining problems lie. No performance comparison table.

---

## 2.2 — Vehicle Tracking Methods

**Papers (those that use tracking and those that explicitly do not):**
- A Real-Time Illegal Parking Detection System with Automatic License Plate Number Recognition
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- Computer Vision and Deep Learning based Approach for Violations due to Illegal Parking Detection
- Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams
- Enhanced Detection of Illegally Parked Vehicles Using YOLO and Good Feature to Track Methods
- Illegal Parking Detection using Gaussian Mixture Model and Kalman Filter
- Illegal Parking Detection Based on Multi-Task Driving Perception
- Parking Time Violation Tracking Using YOLOv8 and Tracking Algorithms
- Real-Time Illegal Parking Detection System Based on Deep Learning (SSD)
- Real-Time Occlusion-Aware Vehicle Detection and Behavior Analysis Framework for Intelligent Traffic Enforcement
- Traffic Violation Detection System
- UAV-Based Intelligent Traffic Surveillance System Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis
- iPatrol: Illegal Roadside Parking Detection Leveraging On-road Vehicle Crowdsensing
- When Vehicle Crowd-Sensing Meets Complex Driving: Robust Illegal Roadside Parking Detections
- Ontology-based detection and identification of complex event of illegal parking using SPARQL and description logic queries

Systems without tracking:
- A Machine Vision Detection of Unauthorized On-Street Roadside Parking in Restricted Zone: An Experimental Simulated Barangay-Environment
- Motorcycle Parking Violation Detection System Using YOLOv7 with Region of Interest Mapping and Object Area Calculation
- Outdoor Illegal Parking Detection System Using Convolutional Neural Network on Raspberry Pi
- Development and Validation of an Instance Segmentation-Based System for Illegal Parking Violation Detection
- Parking Violation Detection using Computer Vision for Urban Traffic Management
- Smart Surveillance of Illegal Parking and Littering Detection Using Yolo-Based Machine Learning Algorithms in the Municipality of Los Baños
- Special Vehicle Classification Algorithm-Based System for Dedicated Parking Zone Violation Detection in South Korea

**How to write it:**

One section, no subsections. Cover the tracking approaches found across the papers — Kalman Filter, IoU matching, DeepSORT, OC-SORT, BoT-SORT, optical flow, template matching — grouped by approach type with multi-citation. Then address systems that skip tracking entirely and process each frame independently. Mention that at least one paper showed quantitatively what happens without tracking (precision dropping dramatically). The key takeaway for the reader: trackers produce bounding box centroid positions for every vehicle in every frame, and those centroids carry enough information to determine whether each vehicle is moving or stationary. This output is what the proposed context-aware layer builds on. No comparison table.

---

## 2.3 — Violation Decision Logic

This is the most important section in the chapter. The contribution of this study is a modification to the decision logic stage. The reader needs to understand how existing systems decide whether a stationary vehicle is a violation, what inputs they use beyond zone and time, and what inputs they do not use.

---

### 2.3.1 — Standard Zone and Threshold

**Papers:**
- A Machine Vision Detection of Unauthorized On-Street Roadside Parking in Restricted Zone: An Experimental Simulated Barangay-Environment
- A Real-Time Illegal Parking Detection System with Automatic License Plate Number Recognition
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- Computer Vision and Deep Learning based Approach for Violations due to Illegal Parking Detection
- Cumulative Dual Foreground Differences for Illegally Parked Vehicles Detection
- Detection and Recognition of Illegally Parked Vehicles Based on an Adaptive Gaussian Mixture Model and a Seed Fill Algorithm
- Detection of Parked Vehicles Using Spatiotemporal Maps
- Enhanced Detection of Illegally Parked Vehicles Using YOLO and Good Feature to Track Methods
- Illegal Parking Detection using Gaussian Mixture Model and Kalman Filter
- Improved Visual Background Extractor for Illegally Parked Vehicle Detection
- Outdoor Illegal Parking Detection System Using Convolutional Neural Network on Raspberry Pi
- Parking Time Violation Tracking Using YOLOv8 and Tracking Algorithms
- Parking Violation Detection using Computer Vision for Urban Traffic Management
- Real-Time Illegal Parking Detection System Based on Deep Learning (SSD)
- Real-Time Illegal Parking Detection in Outdoor Environments Using 1-D Transformation

**How to write it:**

Explain the standard approach: a restricted zone is defined (usually as a polygon), and the system flags any vehicle whose centroid remains inside that zone and stationary for longer than a temporal threshold. This is the dominant paradigm in fixed-camera illegal parking detection. Cover zone definition approaches briefly with citations. Present the temporal threshold table showing every threshold value found across the papers. Let the reader notice the patterns themselves — which threshold is most common, how stationarity is measured differently across studies. Close by noting that this approach evaluates each candidate vehicle in isolation from surrounding traffic.

---

### 2.3.2 — Extended Decision Criteria

**Papers:**
- A computer vision-based roadside occupation surveillance system for intelligent transport in smart cities
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams
- Development and Validation of an Instance Segmentation-Based System for Illegal Parking Violation Detection
- Illegal Parking Detection Based on Multi-Task Driving Perception
- Motorcycle Parking Violation Detection System Using YOLOv7 with Region of Interest Mapping and Object Area Calculation
- Ontology-based detection and identification of complex event of illegal parking using SPARQL and description logic queries
- Real-Time Illegal Parking Detection Algorithm in Urban Environments
- Special Vehicle Classification Algorithm-Based System for Dedicated Parking Zone Violation Detection in South Korea
- Traffic Violation Detection System

**How to write it:**

Some systems add criteria beyond zone membership and temporal threshold. Cover each type briefly — vehicle type classification, license plate recognition, spatial relationship analysis (tire-to-line, wheel-to-marking), ontological reasoning, centerline filtering. Group by type with multi-citation. The reader should notice that these extensions all evaluate the candidate vehicle against a fixed external reference (its type, its plate, its position relative to a line) rather than against the behavior of other vehicles.

---

### 2.3.3 — Context-Aware Approaches

**Papers:**
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- Computer Vision and Deep Learning based Approach for Violations due to Illegal Parking Detection
- Smart Surveillance of Illegal Parking and Littering Detection Using Yolo-Based Machine Learning Algorithms in the Municipality of Los Baños
- Real-Time Occlusion-Aware Vehicle Detection and Behavior Analysis Framework for Intelligent Traffic Enforcement
- Detection of Illegal Parking Events Using Spatial-Temporal Features
- A computer vision-based roadside occupation surveillance system for intelligent transport in smart cities
- Illegal Parking Detection Based on Multi-Task Driving Perception
- Ontology-based detection and identification of complex event of illegal parking using SPARQL and description logic queries
- Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams
- UAV-Based Intelligent Traffic Surveillance System Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis
- Real-Time Illegal Parking Detection Algorithm in Urban Environments

**How to write it:**

This subsection groups all papers that incorporate some form of contextual or scene-level information into their violation decision. For each paper, describe what contextual information it uses and how — signal phases, road geometry, traffic flow patterns, object co-occurrence, lane segmentation, vehicle type as activity proxy. Group by similarity where it makes sense. The reader should see that context-awareness is an emerging trend but that every existing approach uses a different type of context than surrounding vehicle motion state. State factually what is absent: none of these systems uses the motion state of surrounding vehicles as a decision input. Do not make the gap argument here — that belongs in the summary.

---

## 2.4 — Reported Challenges

---

### 2.4.1 — Environmental Conditions and Occlusion

**Papers:**
- Illegal Parking Detection using Gaussian Mixture Model and Kalman Filter
- Parking Time Violation Tracking Using YOLOv8 and Tracking Algorithms
- Parking Violation Detection using Computer Vision for Urban Traffic Management
- Cumulative Dual Foreground Differences for Illegally Parked Vehicles Detection
- Real-Time Illegal Parking Detection in Outdoor Environments Using 1-D Transformation
- Detection and Recognition of Illegally Parked Vehicles Based on an Adaptive Gaussian Mixture Model and a Seed Fill Algorithm
- Improved Visual Background Extractor for Illegally Parked Vehicle Detection
- Illegal Parking Detection Based on Multi-Task Driving Perception
- A Machine Vision Detection of Unauthorized On-Street Roadside Parking in Restricted Zone: An Experimental Simulated Barangay-Environment
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams
- UAV-Based Intelligent Traffic Surveillance System Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis
- iPatrol: Illegal Roadside Parking Detection Leveraging On-road Vehicle Crowdsensing
- Real-Time Illegal Parking Detection Algorithm in Urban Environments
- Ontology-based detection and identification of complex event of illegal parking using SPARQL and description logic queries

**How to write it:**

Cover rain, lighting/night, shadows, and occlusion together. For each challenge type, cite the papers that reported it and note specific impacts with numbers where available. Use direct quotes where they add value. Include a summary table with three columns: challenge, papers reporting it, worst documented impact. Occlusion is the most frequently cited challenge — present it factually, not emphasized over the others.

---

### 2.4.2 — Detection and Tracking Errors

**Papers:**
- A Real-Time Illegal Parking Detection System with Automatic License Plate Number Recognition
- Illegal Parking Detection Based on Multi-Task Driving Perception
- Outdoor Illegal Parking Detection System Using Convolutional Neural Network on Raspberry Pi
- Parking Time Violation Tracking Using YOLOv8 and Tracking Algorithms
- Enhanced Detection of Illegally Parked Vehicles Using YOLO and Good Feature to Track Methods
- Real-Time Illegal Parking Detection in Outdoor Environments Using 1-D Transformation
- Ontology-based detection and identification of complex event of illegal parking using SPARQL and description logic queries
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- City Scale Monitoring of On-Street Parking Violations with StreetHAWK
- Special Vehicle Classification Algorithm-Based System for Dedicated Parking Zone Violation Detection in South Korea
- Real-Time Illegal Parking Detection Algorithm in Urban Environments

**How to write it:**

Cover misclassification, tracking identity errors, segmentation and estimation errors, and resolution/range limitations. Cite specific numbers where papers report them. Keep it brief — these are real problems but they are upstream of the decision logic and are not what this study addresses.

---

### 2.4.3 — Non-Parking Stationary Vehicles

**Papers (fixed-camera, primary evidence):**
- A new curb lane monitoring and illegal parking impact estimation approach based on queueing theory and computer vision
- Cumulative Dual Foreground Differences for Illegally Parked Vehicles Detection
- Computer Vision and Deep Learning based Approach for Violations due to Illegal Parking Detection
- Motorcycle Parking Violation Detection System Using YOLOv7 with Region of Interest Mapping and Object Area Calculation
- Detection of Parked Vehicles Using Spatiotemporal Maps

**Papers (other-modality, supporting evidence):**
- UAV-Based Intelligent Traffic Surveillance System Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis
- Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams
- iPatrol: Illegal Roadside Parking Detection Leveraging On-road Vehicle Crowdsensing

**How to write it:**

Present this as one challenge category among the others — same level of treatment as 2.4.1 and 2.4.2. For each fixed-camera evidence paper, state what problem they described (use direct quotes), what response they built, and what that response requires. For other-modality papers, cover each in 1–2 sentences. Include the summary table with columns: paper, problem stated, response implemented, response requires. Do not state the gap observation here — that belongs in the summary.

---

### 2.4.4 — Challenge Summary

One paragraph. Challenges fall into environmental conditions and occlusion, detection and tracking errors, and traffic conditions affecting violation logic. The first two affect detection and tracking stages. The third affects the decision logic stage — it causes systems to misinterpret legitimate behavior as violations. Each has been partially but not fully addressed.

---

## 2.5 — Summary and Research Direction

---

### 2.5.1 — State of the Field

Two paragraphs. Detection using YOLO variants consistently achieves above 85% accuracy. Tracking using DeepSORT and OC-SORT enables persistent vehicle identification. The dominant decision logic is zone membership combined with temporal threshold. Challenges remain across all pipeline stages. Context-aware approaches are emerging but each uses a different type of contextual information.

---

### 2.5.2 — Research Focus

Three paragraphs. First: the problem of non-parking stationary vehicles appears across multiple independent studies — name the evidence papers and what each found. Second: existing responses all evaluate the candidate against a fixed external reference; none uses surrounding vehicle motion state as a decision input. Third: this study proposes a context-aware decision logic layer that assesses surrounding vehicle displacement data from the tracking module before confirming a violation. The following chapter describes the methodology.

---

## Complete Section Flow

| Section | Title |
|---|---|
| — | Opening paragraph |
| 2.1 | Vehicle Detection Methods |
| 2.2 | Vehicle Tracking Methods |
| 2.3 | Violation Decision Logic |
| 2.3.1 | Standard Zone and Threshold |
| 2.3.2 | Extended Decision Criteria |
| 2.3.3 | Context-Aware Approaches |
| 2.4 | Reported Challenges |
| 2.4.1 | Environmental Conditions and Occlusion |
| 2.4.2 | Detection and Tracking Errors |
| 2.4.3 | Non-Parking Stationary Vehicles |
| 2.4.4 | Challenge Summary |
| 2.5 | Summary and Research Direction |
| 2.5.1 | State of the Field |
| 2.5.2 | Research Focus |
