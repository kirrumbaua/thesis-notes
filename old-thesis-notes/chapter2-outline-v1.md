

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
