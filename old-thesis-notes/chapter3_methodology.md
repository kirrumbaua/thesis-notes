

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
