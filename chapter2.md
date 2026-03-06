

# Chapter 2: Review of Related Literature — Complete Writing Guide (Final)

---

## 2.1 — Overview of Illegal Parking Detection Systems

This section opens the chapter. The goal is simple — orient the reader on what this review covers and how it's organized. You're telling the reader: "We looked at 27 studies published between 2009 and 2025. Here's how we organized them and here's how this chapter flows."

Start with a paragraph stating the scope — how many studies, what year range, and what four aspects of each system the review examines (detection method, tracking method, violation decision logic, and reported challenges). Mention that the studies span different camera setups — fixed CCTV, dashcam, UAV, crowdsensing — but that this study focuses on fixed-camera systems. Briefly state why: fixed cameras are the dominant platform in the literature and match the deployment context of this study.

You can include a table grouping all 27 papers by camera modality. This isn't strictly necessary but it helps the reader see the landscape at a glance and confirms that every paper in the review is accounted for.

| Camera Modality | Papers | Count |
|---|---|---|
| Fixed CCTV | [list] | — |
| Dashcam | [list] | — |
| UAV/Drone | [list] | — |
| Mobile Crowdsensing | [list] | — |
| Other | [list] | — |

Close with a short paragraph explaining how the rest of the chapter is organized — detection methods, tracking methods, decision logic, deployment platforms, challenges, then summary and research direction.

**Length:** About 1.5 pages.

---

## 2.2 — Vehicle Detection Methods

This section covers how each system detects vehicles in the video frame. The reason we discuss this is to show the reader that vehicle detection is a well-studied problem with mature solutions — it's not where our contribution lies. By the end of this section the reader should understand that detection works well enough and the interesting problems are downstream.

---

### 2.2.1 — Background Subtraction Approaches

Start by briefly explaining what background subtraction is — a few sentences is enough. The idea is that a model of the "normal" scene is built over time, and anything that deviates from it is flagged as foreground. This was the dominant approach before deep learning.

Go through each background subtraction variant that appears in the reviewed papers. For each one, explain what it does in 1–2 sentences, state which paper used it, and note what results they reported. The variants you'll find include Gaussian Mixture Models (GMM), Visual Background Extractor (ViBe), and possibly others like 1-D transformation.

Include a comparison table so the reader can see all background subtraction results side by side:

| Method | Paper | Dataset | Precision | Recall | F1 |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

Not every paper reports the same metrics. If a paper reports something else (like "tracking accuracy"), just note that in the table. The point is to show what was measured, not to force everything into the same format.

Close with a short paragraph on the strengths and limitations of background subtraction overall — no training data needed, runs on low-power hardware, but struggles with lighting changes, weather, and shadows. Mention that these methods were standard in earlier work but have largely been replaced by deep learning.

**Length:** 2–2.5 pages including table.

---

### 2.2.2 — Deep Learning Object Detectors

This is the larger subsection because most recent papers use deep learning for detection. Start with 2–3 sentences explaining what deep learning detectors do differently from background subtraction — they learn to recognize and localize objects by class from training data, don't need a background model, and handle environmental variation better.

Organize by detector family. The main families you'll encounter are:

- **YOLO family** — this is the biggest group. Present by version (YOLOv3, YOLOv5, YOLOv7, YOLOv8) since the versions differ significantly. For each version, state which papers used it and what they reported.
- **SSD** — Single Shot MultiBox Detector
- **Mask R-CNN**
- **MobileNet variants**
- **Custom CNNs** — any paper that built their own architecture

For each family, you don't need deep technical explanations of the architecture. A sentence or two on what makes it distinct, then focus on who used it and what results they got. Your groupmates should extract the specific numbers from the papers via NotebookLM.

Include a summary table:

| Detector Family | Papers | Best Reported Accuracy |
|---|---|---|
| — | — | — |

The "Best Reported Accuracy" column can use whatever metric the paper reported — mAP, precision, F1, etc. Just note which metric it is so the reader isn't confused.

Close with a paragraph stating that deep learning detectors, especially YOLO variants, dominate recent publications. YOLOv8 is the most adopted in 2023–2025 work. Detection accuracy consistently exceeds 85% under normal conditions. Detection methodology is mature and is not the focus of this study.

**Length:** 3–4 pages including table.

---

## 2.3 — Vehicle Tracking Methods

This section covers how systems maintain vehicle identity across frames. We discuss this for two reasons. First, tracking is part of the standard pipeline so the reader needs to understand it. Second — and more importantly — the tracking output is what our proposed module builds on. The bounding box centroids that trackers produce for every vehicle in every frame are the raw data our stationary ratio is computed from. So by the end of this section the reader should understand what trackers output and why that output is reliable enough to use.

---

### 2.3.1 — Filter-Based Tracking

Cover traditional tracking approaches found in the papers. For each one, briefly explain how it works (1–2 sentences) and state which papers used it.

The main ones you'll find are:
- **Kalman Filter** — predicts where the object will be in the next frame based on prior motion, then corrects the prediction when a new detection arrives
- **IoU Matching** — matches detections between frames by checking which bounding boxes overlap the most; simple but no appearance features

---

### 2.3.2 — Deep Learning Trackers

Cover the modern trackers found in the papers. These are more important for our study because they're what we'll actually be using. For each tracker, explain how it works in 2–3 sentences, state which papers used it, and note any reported results — especially MOTA scores if available.

The main ones you'll find are:
- **DeepSORT** — adds a deep appearance feature extractor (ReID) on top of SORT so it can re-identify vehicles after brief occlusion
- **OC-SORT** — designed to handle occlusion better through virtual trajectory generation
- **BoT-SORT** — combines appearance features with camera motion compensation

Include a comparison table:

| Tracker | Papers | Best MOTA | Worst MOTA | Key Limitation Reported |
|---|---|---|---|---|
| — | — | — | — | — |

Not all papers report MOTA. Note what they do report or write "Not reported."

---

### 2.3.3 — Optical Flow Methods

Brief subsection. Explain what optical flow does — it computes pixel-level motion vectors between frames. It measures whether pixels are moving rather than tracking individual object identities. State which papers used it and what for. A few sentences is enough.

---

### 2.3.4 — Systems Without Explicit Tracking

This is a short but important subsection. Some systems don't track vehicles at all — they just process each frame independently. Cover which papers did this and what happened.

The key finding to highlight here is that at least one paper showed quantitatively what happens without tracking — precision drops dramatically. This matters because it demonstrates that tracking-based temporal information is essential for the pipeline to work, which supports our decision to build our module on top of tracker output rather than raw detections.

Close Section 2.3 with a summary paragraph. DeepSORT and OC-SORT are the most common trackers in recent work. Tracking is a necessary component but is not the focus of this study. The important takeaway for the reader: the tracking module produces bounding box centroid positions per frame per tracked vehicle, and this is the data our proposed decision logic layer will operate on.

**Length for all of 2.3:** 3–3.5 pages including table.

---

## 2.4 — Violation Decision Logic

This is the most important section in the chapter for our study because our contribution is a modification to the decision logic stage. The reader needs to thoroughly understand how existing systems decide whether a stationary vehicle is a violation — what inputs they use, what thresholds they apply, and what additional criteria they consider. By the end of this section, the reader should have a clear picture of the standard approach (zone + time threshold) and all variations on it.

---

### 2.4.1 — Zone Definition Approaches

Cover every approach used across the papers for defining where violations can occur. For each approach, explain what it is in 1–2 sentences, which papers used it, and whether it requires manual setup or is computed automatically.

The approaches you'll find include:
- Manual ROI polygons (most common — human draws a polygon on a reference frame)
- Curb detection zones (more structured, multiple zones for different purposes)
- Polygon-based spatial filtering (polygon with defined vertex coordinates)
- Homography-based geofencing (projecting camera view to ground coordinates)
- Distance-based measurement (computing pixel distance between vehicle and road marking)
- Spatial relationship analysis (tire-to-line or wheel-to-marking relationships)

Include a comparison table:

| Zone Approach | Papers | Requires Manual Setup | Dynamic or Static |
|---|---|---|---|
| — | — | — | — |

---

### 2.4.2 — Temporal Thresholds

Explain the concept in 2–3 sentences — most systems require a vehicle to remain stationary in the zone for a minimum duration before flagging it as a violation. This prevents brief legitimate stops from being flagged.

Then present a table showing every threshold value found across the papers:

| Paper | Threshold Value | How Stationarity Is Measured |
|---|---|---|
| — | — | — |

After the table, note two things (state them factually, don't editorialize):
1. Which threshold value is most common and how many independent studies use it. Note that none of the reviewed studies provide empirical justification for their chosen value or test how performance changes with different thresholds.
2. The way stationarity is measured varies across studies — some use bounding box displacement, some use IoU overlap between frames, some use background model classification. The measurement method determines how precisely the system distinguishes truly stopped vehicles from slow-moving ones.

This table matters for our study because we adopt the 60-second threshold based on this literature consensus, and it's referenced in Chapter 3.

---

### 2.4.3 — Additional Decision Criteria

Some systems use more than just zone + time to make their decision. Go through every additional input you find across the papers. For each type, briefly describe what it is and which papers use it.

The types you'll find include:
- Vehicle type classification (distinguishing cars from trucks, checking if vehicle is authorized for a zone)
- License plate recognition (identifying the offending vehicle after violation detection)
- Traffic signal state (checking signal color or inferring signal phase) — mention this briefly here and note it's discussed further in Section 2.6.4
- Spatial relationship voting (voting on violation type from detected elements)
- Ontological reasoning (knowledge-based classification)
- Centerline filtering (excluding opposite-lane vehicles)

---

### 2.4.4 — Decision Outputs

Brief coverage of what systems produce after making the violation decision. The main types are binary violation flags, alert mechanisms (buzzer, screenshot), impact estimation (one paper estimates traffic delay caused by the violation), and evidence collection (capturing frames or video). A short paragraph or two is enough.

---

### 2.4.5 — Comprehensive Comparison of Fixed-Camera Systems

This is the master reference table. Include only the fixed-camera papers. One row per paper. Every column filled — if a paper doesn't state something, write "Not stated."

| Paper | Year | Detector | Tracker | Zone Type | Threshold | Additional Criteria | Output |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

This table is important because it lets the reader see every fixed-camera system side by side and immediately spot patterns — like how most use zone + threshold with nothing else, or how few incorporate traffic state. It also gets referenced in later sections and in Chapter 3.

**Length for all of 2.4:** 5–6 pages including tables.

---

## 2.5 — Deployment Platforms

This section covers where and how these systems physically run. It's shorter than the other sections because deployment isn't the focus of our study, but it's worth covering briefly because it shows the reader the practical context these systems operate in and the constraints they face.

---

### 2.5.1 — Fixed CCTV Infrastructure

State how many of the 27 papers use fixed cameras — it's the largest category. Cover the advantages documented in the literature (existing urban infrastructure, no extra hardware cost, static geometry simplifies zone definition, continuous monitoring) and the constraints (low resolution of public cameras, fixed viewing angle limits coverage, camera angle affects detection accuracy). Cite specific papers for each point.

---

### 2.5.2 — Mobile Camera Platforms

Cover dashcam, UAV, and crowdsensing together. For each, 1–2 sentences on the approach, which papers use it, and the main advantage and disadvantage. No need for separate subsections within this — they can flow as three short paragraphs.

---

### 2.5.3 — Edge Computing Deployment

Cover systems that process on-device (Raspberry Pi, smartphone, drone). State the advantage (no network dependency, real-time, low cost) and the constraint (limits model complexity, so lighter architectures get used). List which papers did this.

**Length for all of 2.5:** 2–2.5 pages.

---

## 2.6 — Reported Challenges Across Studies

This section documents the problems that the reviewed systems actually encountered. The purpose is to give the reader an honest picture of what doesn't work well yet. We organize by challenge category and treat each category at the same level — we're not trying to inflate one over the others. The reader should finish this section understanding that challenges exist across all pipeline stages and that each has been partially but not fully addressed.

---

### 2.6.1 — Environmental Conditions

Cover three types of environmental challenges:

**Rain and adverse weather** — which papers tested in rain or adverse weather, what happened to their metrics. Use direct quotes from the papers where they describe the impact.

**Lighting and night conditions** — which papers reported problems at night or with changing illumination. Specific issues include headlight glare, sudden illumination changes, and reduced visibility.

**Shadows and time-of-day variation** — which papers tested across different times of day and how shadow patterns affected performance.

Include a summary table:

| Challenge | Papers Reporting It | Worst Documented Impact |
|---|---|---|
| Rain | — | — |
| Night/lighting | — | — |
| Shadows | — | — |

For the "Worst Documented Impact" column, cite actual numbers — metric drops, false positive counts, accuracy decreases. This makes the challenge concrete rather than vague.

---

### 2.6.2 — Occlusion

Occlusion is the most frequently cited challenge across the papers, so it deserves thorough coverage. Organize into vehicle-to-vehicle occlusion (one vehicle blocking another) and partial visibility (vehicle only partially in frame or partially visible).

For each paper that reports occlusion issues, write 1–2 sentences describing the specific problem. Use direct quotes where they're impactful — phrases like "system was not able to segment vehicles that arrive together" tell the reader more than a paraphrase would.

Note: occlusion is the most-cited challenge but it's NOT the challenge our study addresses. Present it factually at the same level as the other categories. Don't downplay it, don't inflate it.

---

### 2.6.3 — Detection and Tracking Errors

Cover system errors that aren't caused by environment or occlusion. Organize by error type:

- **Misclassification** — vehicles classified as wrong type, or non-vehicles detected as vehicles. Cite specific numbers where papers report them.
- **Tracking identity errors** — ID switching, lost tracks, false tracks from reflections. Cite descriptions from the papers.
- **Segmentation and estimation errors** — wrong bounding boxes, merged vehicles, distance estimation failures.
- **Resolution and range limitations** — papers that cited camera resolution or detection range as limiting factors.

---

### 2.6.4 — Non-Parking Stationary Vehicles

This is the challenge our study addresses, but in this section we present it as one challenge category among the others — same level of treatment as 2.6.1, 2.6.2, and 2.6.3. Don't emphasize it more than the others here. Just present what the papers found. The advocacy for our study happens in 2.7, not here.

That said, this subsection needs enough detail that the evidence is clear, because Section 2.7.2 will directly reference it.

**The three fixed-camera evidence papers — cover each in one paragraph:**

For each of these three, state: (1) what problem they described, using a direct quote, (2) what response they built, and (3) what that response requires to work.

**Gao et al. (2022):** They explicitly stated the challenge of differentiating red light stops from parking. Also mentioned vehicles stopping for queues or pedestrians. Their response was a pedestrian zone proxy — monitoring crosswalk activity to infer signal phase. This required a visible crosswalk, pedestrians actually crossing during red phases, and defining two additional detection zones.

**Wahyono et al. (2017):** They reported traffic jams and signal intersection stops causing false regions in their stationary vehicle detection. Their response was a 60-second tracking timer. This requires nothing additional, but it cannot distinguish a vehicle parked for 61 seconds from one stuck in congestion for 61 seconds.

**Kathait et al. (2025):** They stated that vehicles stopped at red or yellow signals must be considered. They described checking signal color. But their dataset didn't have visible traffic signals, so signal-based filtering was not actually implemented in their evaluation.

**Other-modality studies — cover each in 1–2 sentences:**

These are supporting evidence, not primary. Cover the UAV paper (temporary stops, signal timing), Park J. et al. 2024 (congestion, lined-up vehicles), and iPatrol/Huang 2023 (precision drop in crowded traffic).

**Include a summary table:**

| Paper | Problem Stated | Response Implemented | Response Requires |
|---|---|---|---|
| Gao et al. (2022) | [direct quote] | Pedestrian zone proxy | Visible crosswalk, pedestrian presence, two additional zones |
| Wahyono et al. (2017) | [direct quote] | 60-second tracking timer | Nothing additional, but cannot distinguish long congestion from parking |
| Kathait et al. (2025) | [direct quote] | Signal color check described | Visible signal head (not implemented) |
| [other papers] | [summary] | [response] | [requirements] |

Use direct quotes from the papers in the "Problem Stated" column wherever possible. This table gets referenced directly in Chapter 3's contribution positioning table in Section 3.1.2.

Don't state the gap observation here. That belongs in 2.7.2. Just present the evidence and let it sit.

---

### 2.6.5 — Summary of Challenges

One paragraph wrapping up all four challenge categories. State that challenges fall into environmental conditions, occlusion, detection/tracking errors, and traffic conditions affecting violation logic. Environmental and occlusion challenges affect detection and tracking stages. Traffic condition challenges affect the decision logic stage — they cause systems to misinterpret legitimate vehicle behavior as violations. Each category has been addressed to varying degrees in individual studies but none has been fully resolved.

**Length for all of 2.6:** 5–6 pages including tables.

---

## 2.7 — Summary and Research Direction

This section closes the chapter. Section 2.7.1 summarizes what the entire review established. Section 2.7.2 is the only place in Chapter 2 where you narrow to your specific problem and state your direction. Everything before this point was honest, transparent overview. Here is where you say "and here's what we noticed, and here's what we're going to do about it."

---

### 2.7.1 — State of the Field

One to two paragraphs summarizing the field.

First paragraph: Detection using YOLO variants consistently achieves above 85% accuracy and is the standard approach. Tracking using DeepSORT and OC-SORT enables persistent vehicle identification, though it degrades under rain and with visually similar vehicles. The dominant decision logic — zone membership combined with temporal threshold — is computationally efficient and adopted by the majority of fixed-camera studies.

Second paragraph: Challenges remain across all pipeline stages. Environmental conditions degrade detection and tracking accuracy. Occlusion is documented in more studies than any other single challenge. Detection and tracking errors including misclassification and ID switching contribute to both false positives and false negatives.

---

### 2.7.2 — Research Focus

This is where the three paragraphs matter and the framing needs to be right. This isn't aggressive or cocky — it's just observing a pattern, noting what's absent, and stating what this study will do.

**Paragraph 1 — The pattern:** State that the problem of non-parking stationary vehicles affecting violation decision logic appears across multiple independent studies. Name the three fixed-camera papers (Gao 2022, Wahyono 2017, Kathait 2025) and briefly state what each found. The fact that three independent studies — with different detection methods, different datasets, and different goals — all encountered the same problem makes it a real pattern, not a cherry-picked issue.

**Paragraph 2 — What's common and what's absent:** State that the existing responses share a characteristic: each evaluates the candidate vehicle against a fixed external reference — a temporal duration, a pedestrian zone, or a signal head. All examine the candidate vehicle's behavior in isolation from other vehicles in the scene. Whether surrounding tracked vehicles are also stationary — information that is already derivable from standard tracking output without additional detection modules or scene-specific calibration — is not used as a decision input in any reviewed system.

**Paragraph 3 — What this study will do:** State that this study focuses on the traffic condition challenge specifically. It proposes a context-aware decision logic layer that assesses surrounding vehicle displacement data from the existing tracking module before confirming a violation. If surrounding vehicles are also stationary, the candidate's behavior is consistent with collective traffic conditions. If the candidate is stationary while surrounding vehicles move, the behavior is consistent with individual parking. State that the following chapter describes the methodology.

These three paragraphs should flow naturally — here's the pattern, here's the gap, here's what we do. The evidence from 2.6.4 should already speak for itself by this point.

**Length for all of 2.7:** 1.5–2 pages.

---

## Complete Chapter 2 Section Flow

| Section | Title | Length |
|---|---|---|
| 2.1 | Overview of Illegal Parking Detection Systems | 1.5 pages |
| 2.2 | Vehicle Detection Methods | 5–6 pages |
| 2.2.1 | Background Subtraction Approaches | — |
| 2.2.2 | Deep Learning Object Detectors | — |
| 2.3 | Vehicle Tracking Methods | 3–3.5 pages |
| 2.3.1 | Filter-Based Tracking | — |
| 2.3.2 | Deep Learning Trackers | — |
| 2.3.3 | Optical Flow Methods | — |
| 2.3.4 | Systems Without Explicit Tracking | — |
| 2.4 | Violation Decision Logic | 5–6 pages |
| 2.4.1 | Zone Definition Approaches | — |
| 2.4.2 | Temporal Thresholds | — |
| 2.4.3 | Additional Decision Criteria | — |
| 2.4.4 | Decision Outputs | — |
| 2.4.5 | Comprehensive Comparison of Fixed-Camera Systems | — |
| 2.5 | Deployment Platforms | 2–2.5 pages |
| 2.5.1 | Fixed CCTV Infrastructure | — |
| 2.5.2 | Mobile Camera Platforms | — |
| 2.5.3 | Edge Computing Deployment | — |
| 2.6 | Reported Challenges Across Studies | 5–6 pages |
| 2.6.1 | Environmental Conditions | — |
| 2.6.2 | Occlusion | — |
| 2.6.3 | Detection and Tracking Errors | — |
| 2.6.4 | Non-Parking Stationary Vehicles | — |
| 2.6.5 | Summary of Challenges | — |
| 2.7 | Summary and Research Direction | 1.5–2 pages |
| 2.7.1 | State of the Field | — |
| 2.7.2 | Research Focus | — |

**Total estimated length:** 24–28 pages.
