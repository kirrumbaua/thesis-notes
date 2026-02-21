# Thesis Pitch Document

## A Context-Aware Decision Logic Layer for Illegal Parking Detection in Fixed-Camera Environments

---

## Part 1: The Evidence — What Papers Found This Problem

---

### Primary Evidence Papers (Fixed-Camera)

**Paper 1: Gao et al. (2022)**
"A New Curb Lane Monitoring and Illegal Parking Impact Estimation Approach Based on Queueing Theory and Computer Vision for Cameras with Low Resolution and Low Frame Rate"

- Their goal was NOT to study stationary vehicle behavior — they were building an impact estimation system using queueing theory
- But in the process, they explicitly stated: *"Another challenge in detecting illegal parking events in a dense urban area is differentiating between vehicles stopped for red lights and those illegally parked"*
- They also mentioned vehicles stopping for *"queue or pedestrians"* as additional causes
- Their fix: Built a pedestrian zone proxy — monitoring crosswalk activity to infer if the traffic signal is red
- Why pedestrians? Because in their camera footage, the actual traffic signal was not visible. They used what was available in their dataset.

**Paper 2: Wahyono et al. (2017)**
"Cumulative Dual Foreground Differences for Illegally Parked Vehicles Detection"

- Completely different detection method (background subtraction with dual GMM models, not deep learning)
- Different dataset (i-LIDS, ISLab)
- But encountered the exact same problem
- They stated: *"The detecting stable region step might obtain some false regions due to a traffic jam, stopped vehicles and standing human at the traffic light intersection"*
- Their fix: 60-second tracking timer — only flag vehicles stationary for more than 60 seconds
- Problem with their fix: Cannot distinguish a vehicle parked for 61 seconds from a vehicle stuck in congestion for 61 seconds

**Paper 3: Kathait et al. (2025)**
"Computer Vision and Deep Learning Based Approach for Violations Due to Illegal Parking Detection"

- Stated: *"During the detection of violation due to illegal parking it is important that vehicles which are at stop due to red or yellow traffic signal are also considered"*
- Described checking if *"signal is green or the vehicle is in the given parking for a long time"*
- Their fix: Proposed using traffic signal color detection
- But: Their dataset did not have visible traffic signals, so they did not actually implement it

---

### Supporting Evidence Papers (Other Modalities)

**UAV Paper (2024)**
"UAV-Based Intelligent Traffic Surveillance System"

- Applied a 10-second minimum threshold specifically to *"ignore temporary stops such as pedestrian crossings"*
- Recognized that *"delays can also result from other factors such as signal timing, pedestrian flow, or random fluctuations"*
- Reported *"detection challenges particularly under heavy congestion"*

**Park J. et al. (2024)**
"Deep Learning-Based Stopped Vehicle Detection Method Utilizing In-Vehicle Dashcams"

- Dashcam-based system, not fixed camera
- Tested under *"traffic congestion"* and *"lined-up vehicles"* as extreme conditions
- Built a centerline filter to handle opposite-lane vehicles

**iPatrol / Huang et al. (2023)**
"iPatrol: Illegal Roadside Parking Detection Leveraging On-road Vehicle Crowdsensing"

- Mobile crowdsensing system
- Tested under "Empty," "Normal," and "Crowded" traffic conditions
- Precision dropped from approximately 95% in empty traffic to 81.5% in crowded traffic
- This is the only paper with actual numerical comparison across traffic density levels

---

## Part 2: The Pattern — What This Tells Us

---

### Different Papers, Different Goals, Same Problem

| Paper | Their Actual Goal | The Problem They Encountered |
|---|---|---|
| Gao et al. (2022) | Measure traffic impact of illegal parking using queueing theory | Vehicles at red lights were being flagged as violations |
| Wahyono et al. (2017) | Improve background subtraction for parked vehicle detection | Traffic jams and signal stops caused false regions |
| Kathait et al. (2025) | Build a general violation detection system | Acknowledged signal state matters for accuracy |
| UAV paper (2024) | Build drone-based traffic surveillance | Temporary stops at crosswalks caused false flags |
| iPatrol (2023) | Build city-scale crowdsensing system | Crowded traffic degraded precision by ~14% |

None of these papers set out to study this problem. They all encountered it while trying to do something else. That is actually stronger evidence — you do not build workarounds for problems that do not exist.

---

### Different Fixes, All Scene-Specific

| Paper | Their Fix | What It Requires To Work |
|---|---|---|
| Gao et al. (2022) | Pedestrian zone proxy | Crosswalk must be visible, pedestrians must actually cross during red phase |
| Wahyono et al. (2017) | 60-second timer | Nothing extra, but cannot distinguish long congestion from parking |
| Kathait et al. (2025) | Traffic signal color detection | Signal head must be visible in camera frame |
| UAV paper (2024) | 10-second minimum | Nothing extra, but only filters very brief stops |

Every fix is tied to what was convenient or available in their specific dataset:
- Gao could see pedestrians but not the signal → used pedestrians
- Kathait proposed using the signal → but their dataset did not have visible signals so they could not implement it
- Wahyono just used a longer timer → simple but crude

---

### What All Fixes Have In Common (And What Is Missing)

All existing fixes evaluate the candidate vehicle against some fixed external reference:
- A time duration
- A pedestrian zone
- A traffic signal head

**None of them ask the simple question:**

> "Are the OTHER vehicles in the scene also stopped?"

If every vehicle around you is also stopped, you are probably in traffic.
If you are the only one stopped while everyone else is moving, you are probably parked.

This information is already available. Every tracker computes bounding box positions for every vehicle in every frame. The displacement of surrounding vehicles is sitting right there. No one uses it.

---

## Part 3: What We Propose — The Core Idea

---

### The Simple Version

Current systems ask:
> "Has this vehicle been in the zone longer than 60 seconds?"

Our system also asks:
> "Are the surrounding vehicles also stopped, or is this vehicle stopped alone?"

---

### How It Works (Non-Technical)

**Step 1 — Do everything the baseline does**
- Detect vehicles with YOLOv8
- Track them with DeepSORT
- Check if they are in the restricted zone
- Check if they have been there more than 60 seconds

**Step 2 — Before confirming a violation, check surrounding traffic**
- Look at ALL tracked vehicles in the frame
- Compute what percentage are currently stationary (we call this the "stationary ratio")

**Step 3 — Make the decision based on context**

| Situation | What It Means | Decision |
|---|---|---|
| Most vehicles are stopped | Probably a red light or congestion | Suppress the flag (for now) |
| Traffic clears but this vehicle stays | Everyone left, this one did not | Confirm violation |
| Only this vehicle is stopped while others move | Isolated anomaly in flowing traffic | Confirm violation (high confidence) |

---

### Why This Is Better Than Existing Fixes

| Existing Fixes | Our Approach |
|---|---|
| Need visible crosswalk (Gao) | Just needs other vehicles visible |
| Need visible signal (Kathait) | Does not need signal |
| Cannot distinguish long congestion from parking (Wahyono) | Can detect when traffic clears and vehicle stays |
| Scene-specific calibration needed | Uses tracker output every system already produces |

---

## Part 4: How We Will Test It

---

### Controlled Comparison

We build TWO systems ourselves:

| System | What It Does |
|---|---|
| Baseline | YOLOv8 + DeepSORT + zone check + 60-second threshold → binary output |
| Proposed | Same as baseline + context-aware layer → output with traffic state awareness |

Same detector. Same tracker. Same zone. Same threshold. The ONLY difference is our layer.

Any performance difference = effect of our contribution.

---

### What We Measure

| Metric | What It Tells Us |
|---|---|
| Precision | Of flagged violations, how many are real? (Expected: proposed > baseline) |
| Recall | Of real violations, how many did we catch? (Expected: proposed ≈ baseline) |
| False positive breakdown | Which types of FPs did we reduce? (Expected: signal and congestion FPs reduced) |

---

### Why We Build Our Own Baseline

The 27 papers we reviewed use:
- Different datasets
- Different cameras
- Different locations
- Different detection models
- Different thresholds
- Different evaluation protocols

Directly comparing against any single paper's numbers would be unfair and meaningless.

By building both systems ourselves on the same footage, we create a clean comparison.

---

## Part 5: Concerns To Discuss

---

### Concern 1: No Documented Before-and-After Metrics

**The issue:**

No paper measures the exact impact of adding contextual vehicle behavior checks. Gao et al. added the pedestrian proxy, but they did not report:
- "Before the proxy, we had X% precision"
- "After adding the proxy, we had Y% precision"

They just added it as a fix and moved on because it was not the focus of their study.

**Why this might actually be okay:**

- This is our research opportunity. We are the ones who will measure before and after.
- The panelist might actually view this positively: "Previous studies acknowledged the problem and patched it, but no one formally evaluated the patch. This study does."
- We are not claiming to solve a proven massive problem. We are proposing to test whether this approach works and measure the effect.

**How to frame it to panelists:**

> "Multiple independent studies found this problem significant enough to build workarounds, but none formally evaluated the impact of those workarounds. This study provides that formal evaluation using a controlled comparison."

---

### Concern 2: "This Is Just Traffic Congestion Detection"

**The potential criticism:**

A panelist might say: "You are just detecting traffic congestion. That is a different research problem."

**Why this framing is wrong:**

We are not detecting congestion for its own sake. We are using traffic state as a filter to improve parking violation detection. The output is still "violation" or "non-violation" — not "congested" or "not congested."

Think of it this way:
- Gao et al. used pedestrian detection → but they were not building a pedestrian detection system
- We use traffic state detection → but we are not building a congestion detection system

Both are using one signal to improve another task.

**How to frame it to panelists:**

> "Traffic state is not the output — it is the filter. The system still outputs parking violation decisions. Traffic state is used internally to determine whether a stationary vehicle is stopped individually or collectively, which directly affects whether it should be classified as a violation."

---

### Concern 3: What If It Does Not Work?

**The possibility:**

We run the experiments and find that the stationary ratio check does not meaningfully improve precision. Or it improves precision but kills recall.

**Why this is still a valid thesis:**

- A negative result is still a result. "We tested whether surrounding vehicle behavior improves violation detection. It does not." is a valid finding.
- It would close a research question that had not been formally tested.
- The methodology is still sound. The contribution is the evaluation, not just a positive outcome.

**But realistically:**

The logic is sound. If everyone is stopped, flagging one vehicle makes no sense. If one vehicle is stopped alone, that is anomalous. The question is whether the implementation details (thresholds, ratio computation) work well enough to show measurable improvement.

---

## Summary Table for Groupmates

| Aspect | Detail |
|---|---|
| Problem | Current systems flag any vehicle stationary in the zone beyond a threshold, even if it is stopped in traffic |
| Evidence | 3 fixed-camera papers (Gao, Wahyono, Kathait) + 3 other-modality papers documented this problem independently |
| Pattern | All built scene-specific fixes using whatever was available (pedestrians, signals, timers). None checked surrounding vehicle behavior. |
| Our approach | Check if surrounding tracked vehicles are also stopped before confirming a violation |
| Why better | Uses data already available from tracker. Works regardless of whether crosswalk or signal is visible. |
| How we test | Build baseline and proposed systems ourselves. Same everything except our layer. Measure precision, recall, FP breakdown. |
| Concern 1 | No before/after metrics exist in literature → This is our research contribution to provide that measurement |
| Concern 2 | "Just congestion detection" → No, congestion is the filter, violation is the output |
| Concern 3 | What if it fails → Negative result is still valid, methodology is sound |

---

## One-Sentence Pitch

> "Multiple papers found that vehicles stopped in traffic get flagged as parking violations, and each built their own scene-specific patch — we propose checking if surrounding vehicles are also stopped, using data the tracker already produces, and we will be the first to formally measure whether this works."
