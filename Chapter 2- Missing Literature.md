# Stage 9 — Missing Literature

Per your original instructions, I am not inventing references here — only naming topic areas the current 62-paper library does not adequately cover, and suggesting where you'd search for real ones. Some of these were already flagged in passing (Stages 2, 4, 5, 8); this stage consolidates them and adds a few not yet raised.

---

### MISSING LITERATURE: XGBoost (or tree-based ensembles) as a final per-instance decision classifier

**Why needed:** Confirmed at Stage 2 and reconfirmed at Stage 8 (Claim 7) — only one paper in the entire library (**17**, Jiang et al.) uses XGBoost anywhere, and only as a static-to-dynamic feature transformer inside an unrelated city-grid forecasting pipeline, not as the classifier making a violation/non-violation decision. You have no illegal-parking-domain precedent, and no traffic-domain precedent at all, for XGBoost as a final decision classifier on a structured/tabular feature vector. Objective 1 explicitly commits to "an XGBoost classifier," so a panelist can reasonably ask why this algorithm specifically, and the current library cannot answer that question.

**Where to search:** This will likely need to come from *outside* the traffic-CV domain — general machine-learning literature comparing XGBoost against alternatives (Random Forest, SVM, gradient boosting variants, shallow neural networks) on structured/tabular classification tasks, ideally with characteristics similar to yours (a small, engineered, low-dimensional numeric feature vector; likely class imbalance; a need for interpretability). Search terms: "XGBoost tabular classification benchmark," "gradient boosting vs. neural network structured data," "tree ensemble classifier low-dimensional feature vector." Databases: IEEE Xplore, ACM DL, and general ML venues (NeurIPS, JMLR) rather than ITS-specific venues, since this is a classifier-choice justification, not a traffic-application paper.

---

### MISSING LITERATURE: Interaction-aware / neighbor-conditioned trajectory and behavior prediction (adjacent field, high relevance)

**Why needed:** This is the most consequential gap I did not flag explicitly in earlier stages and want to raise clearly now. A substantial body of work exists in **autonomous driving and pedestrian trajectory prediction** — e.g., approaches that condition a target agent's predicted behavior on the aggregate motion state of *neighboring* agents (sometimes called "social," "interaction-aware," or "context-conditioned" trajectory/behavior models). This is methodologically almost exactly your proposal's structure (aggregate neighbor-agent features feeding a learned model to make a decision about one target agent) — just applied to trajectory *prediction* in a driving/pedestrian context rather than violation *classification* in a parking context. None of your 62 papers touch this literature; your library is drawn entirely from the illegal-parking-detection and general-ITS space, not from the trajectory-prediction/behavior-modeling space.

This matters for your defense in two ways: (1) it would let you argue your feature-vector *design pattern* (aggregate neighbor motion → learned decision) has precedent in a mature adjacent field, strengthening Section 2.8's methodological-plausibility argument considerably beyond what 17 and 21 alone can support; (2) a well-read panelist in ML/autonomous systems (as opposed to strictly ITS) could ask why you didn't engage with this literature, since it's a natural point of comparison for the general "aggregate-neighbor-motion-as-a-feature" idea even outside the parking-specific application.

**Where to search:** Terms like "social trajectory prediction," "interaction-aware behavior prediction," "neighbor-conditioned motion forecasting," "context-aware vehicle behavior prediction." Venues: CVPR, ICRA, IROS, IEEE T-ITS, and general autonomous-driving/robotics literature rather than parking-specific sources.

---

### MISSING LITERATURE: Traffic flow theory as a conceptual grounding for the feature vector

**Why needed:** Your 6-D vector (R, C, ḋ, σ_d, ΔR, d_cand) draws implicitly on concepts from transportation engineering — vehicle count/density, mean speed, speed variance, spacing — that are the basic building blocks of classical traffic flow theory (e.g., speed-density-flow relationships, congestion/shockwave propagation). None of the 62 papers are traffic-flow-theory sources; they are all computer-vision/detection papers. Grounding your feature choices in established traffic-flow concepts (rather than presenting them as ad hoc engineering choices) would strengthen Chapter 3's feature-justification and could preempt a methodology-panel question about *why* these six specific dimensions were chosen.

**Where to search:** Classical and applied transportation-engineering literature on traffic flow theory, congestion detection/classification, and speed-density relationships. Search terms: "traffic flow density speed relationship," "congestion detection features," "macroscopic traffic state classification." Venues: Transportation Research Part B/C, IEEE T-ITS, Transportation Research Record.

---

### MISSING LITERATURE: Evaluation methodology for imbalanced, rare-event classification

**Why needed:** Genuine illegal-parking events are very likely rare relative to legitimate stops/passes in any real video stream, meaning your training and evaluation data will almost certainly be class-imbalanced. None of your 62 papers discuss class-imbalance handling (resampling, cost-sensitive learning, appropriate metric choice under imbalance) as a methodological concern — most simply report accuracy/precision/recall on whatever dataset composition they collected, without discussing whether or how imbalance was addressed. This is a methodology-chapter gap, not a literature-review gap per se, but it belongs in your source base before you defend your evaluation design.

**Where to search:** General ML literature on class-imbalanced classification evaluation — search terms: "imbalanced classification evaluation metrics," "class imbalance tree-based classifier," "SMOTE tabular data." This is again likely to come from general ML venues rather than ITS-specific ones.

---

### MISSING LITERATURE: Ground-truth annotation methodology for context-dependent violation labeling

**Why needed:** Your problem requires labeling training examples as "genuine violation" vs. "legitimate collective stop" — a judgment that, unlike simple bounding-box annotation, requires a human annotator to reason about surrounding traffic context, not just the target vehicle. None of the 62 papers describe an annotation protocol for this kind of *contextual* ground-truth labeling (as opposed to straightforward object/violation-type labeling). This is a methodological gap you will need to address directly in Chapter 3, and it would help to have at least one methodological citation on annotation protocols for context-dependent or ambiguous event labeling in video datasets.

**Where to search:** Video-annotation-methodology literature, possibly from action-recognition or anomaly-detection datasets that involve similarly ambiguous/context-dependent labeling. Search terms: "context-dependent event annotation video," "ambiguous label annotation protocol video dataset."

---

### Not flagged as missing: areas the current library already covers adequately

For completeness — the following areas from your original brief's suggested list are **already well-supported** and do not need additional search: foundational illegal-parking-detection literature (Section 2.3–2.5), fixed-camera traffic analysis (2.2, 2.4, 2.7), false-positive reduction as a design goal (2.4, 2.5), and single-vehicle trajectory/tracking methodology (2.2, 2.7). No need to spend further search effort here.

---

### Priority ranking for your remaining literature search time

If you have limited time before finalizing Chapter 2/3, I'd prioritize in this order:
1. **XGBoost/tree-ensemble classifier justification** — directly needed for Objective 1, likely the first thing a methodology-focused panelist probes.
2. **Interaction-aware/neighbor-conditioned trajectory prediction** — highest strategic value for strengthening your novelty argument, but also the most work to properly integrate (different field, different terminology).
3. **Class-imbalance evaluation methodology** — needed before you can defend your evaluation design in Chapter 3, less urgent for Chapter 2 itself.
4. **Traffic flow theory grounding** and **annotation methodology** — lower priority; nice-to-have for a more polished defense, not likely to be a blocking question.

Ready for **Stage 10** (the department/format questions your original brief said to ask before finalizing) whenever you'd like to proceed.
