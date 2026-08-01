

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
