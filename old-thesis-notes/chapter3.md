

# Chapter 3: Methodology — Complete Writing Guide (Final)

---

## 3.1 — Conceptual Framework

**What this section is about:** Bridges Chapter 2 to the proposed study and presents the system design.

**What to write:**

**Opening paragraph (no subsection — just the section opener):**
One paragraph connecting Chapter 2 to this study. Reference Section 2.6.4 and the three fixed-camera studies (Gao 2022, Wahyono 2017, Kathait 2025). State the gap: all existing responses evaluate the candidate in isolation against fixed external references; none uses surrounding vehicle behavior. State that this study proposes a context-aware decision logic layer using surrounding vehicle displacement data.

---

### 3.1.1 — System Architecture Overview

**What this section is about:** Presents the full system design as a proper diagram.

**What to write:**

**Figure 1 — System Architecture Diagram**

This must be a proper technical diagram (made in draw.io, Figma, or similar), not a simple flowchart. It should show **both the baseline path and the proposed path side by side or as a branching diagram**. The figure must contain:

| Component | What to show in the diagram |
|---|---|
| Input | Fixed camera video frame |
| Detection stage | Box labeled with "YOLOv8 (n / s / m)" — show the output: bounding boxes with class labels and confidence scores |
| Tracking stage | Box labeled with "Tracker (DeepSORT / ByteTrack / BoT-SORT)" — show the output: tracked bounding boxes with persistent track IDs and centroids |
| Zone membership check | Box showing centroid tested against zone polygon — output: in-zone / out-of-zone status per track |
| Temporal threshold check | Box showing stationarity timer — output: candidate violation flag when timer exceeds T |
| **Branch point** | Baseline path goes straight to output. Proposed path enters the context-aware layer. |
| Context-aware layer (proposed only) | Enlarged box or bordered region showing: (1) displacement computation for all vehicles, (2) stationary ratio R, (3) Check 1 / Check 2 / Check 3 logic, (4) confirm or suppress decision |
| Baseline output | Violation / Non-violation |
| Proposed output | Violation Confirmed / Suppressed + Traffic State Label + Confidence |

**Notes for groupmates:**
- The detection and tracking boxes should be visually marked as **variable** (e.g., dashed border, or a label saying "swappable").
- The context-aware layer box should be visually marked as the **contribution** (e.g., highlighted border, different color).
- Show the **data type** flowing between each component along the arrows (frames → bounding boxes → tracked centroids → displacement values → decision).
- This is probably a full-page figure.

After the figure, write one paragraph describing the baseline path and one paragraph describing the proposed path. Keep it short — the figure does the heavy lifting.

Write one paragraph stating the factorial design:

| Factor | Levels | Role |
|---|---|---|
| Detection-tracking pipeline | 9 combinations (3 detectors × 3 trackers) | Upstream variable |
| Context-aware layer | ON or OFF | Treatment variable (contribution) |

State that both configurations process the same video with the same zone and threshold. The only variable within each pipeline pair is the context layer.

---

### 3.1.2 — Contribution Positioning

**What this section is about:** Maps the proposed approach against existing approaches from Chapter 2.
