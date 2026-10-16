# M2. Preliminary & Chapter 1


- Type: Page

- Item ID: 4494245

- Module ID: 496209


## Content


CEU Department of Computer Science and Information Technology

# M2. Preliminary & Chapter 1: General Components of a CS Research Proposal

A Guide to Algorithmic Investigations, Design Science Research (DSR), Empirical Benchmarking & CEU Manuscript Protocols

### Learning Objectives

By the end of this module, Computer Science research students will be able to:

- Differentiate **Computer Science Research** (creating new computational knowledge) from **Software Engineering** (building standard applications with existing APIs).
- Apply CEU CSIT standard manuscript mechanics, APA 7th Edition citations, Zotero reference management, and Boolean search strings.
- Distinguish between the **Introduction to the Study** (executive domain hook) and the **Background of the Study** (technical case file).
- Structure the Background of the Study using the **Macro-Meso-Micro Model** to highlight algorithmic and computational technical gaps.
- Formulate scoped research questions and align the **Statement of the Problem (SOP)** with Design Science Research (DSR) objectives.
- Construct an **Input-Process-Output (IPO) Conceptual Framework** integrated with quantitative performance metrics (*t_(exec)*, *M_(RAM)*, accuracy, precision, throughput).
- Define algorithmic boundaries, dataset scope, technical delimitations, and operational definitions tailored to the project codebase and mathematical models.

## 1. Foundation: Software Engineering vs. Computer Science Research

Before drafting Chapter 1, you must ensure your project is a **Computer Science Research Project** rather than a standard Application Development / Software Engineering effort.

The Core Formula for CS Research: CS Research Project = Novel Component / Experiment + Software Artifact + Empirical Metrics

| Dimension | Software Engineering (App Development) | Computer Science Research |
|----|----|----|
| Primary Goal | Build a working, reliable software product for end-users. | Create new computational knowledge, algorithms, or novel optimizations. |
| Nature of Problem | Known problem solved with standard off-the-shelf tools/APIs. | Unresolved computational problem or open algorithmic bottleneck. |
| Tools Used | Existing frameworks, standard libraries, commercial SaaS APIs. | Custom algorithmic modifications, mathematical models, empirical testbeds. |
| Success Metric | User adoption, business workflow fulfillment, UI/UX survey. | Empirical validation, accuracy/speed metrics, peer-reviewed proof of concept. |

**✗ Pure App Development (Not CS Research):**\
"Mobile App for Plant Disease Detection using a pre-trained AI model from GitHub."

**✓ Defensible CS Research Project:**\
"Optimization of Lightweight CNN Models for real-time plant disease detection on low-end Android hardware via model quantization."

## 2. General Mechanics & Literature Protocol (CEU Standard)

All CS proposals submitted to the CEU Department of Computer Science and Information Technology must strictly follow manuscript rules and formal reference search procedures:

#### ⇓ Paper & Margins

- **Paper Size:** Short Bond Paper (8.5" × 11").
- **Left Margin:** 1.5 inches (binding allocation).
- **Top, Bottom, Right Margins:** 1.0 inch.

#### ⇓ Typography & Spacing

- **Font Style:** Times New Roman throughout.
- **Font Size:** 12pt for body text & subheadings.
- **Spacing:** Double-spaced body paragraphs.

#### ⇓ Search & Reference Strategy

- **Boolean Querying:** Construct strings using `AND`, `OR`, and exact quotes in IEEE Xplore / ACM Digital Library.
- **Citation Manager:** Utilize Zotero for single-click capture and automated APA 7th Edition indexing.

## 3. Differentiating Introduction to the Study vs. Background of the Study

In CEU CS Research, Chapter 1 opens with two distinct sections that build the opening narrative of your proposal:

### 3.1 Introduction to the Study (The Hook & High-Level Overview)

The **Introduction** serves as an executive narrative (1 to 1.5 pages) that hooks the reader. It presents the big picture of your field without getting bogged down in dense empirical benchmark tables immediately.

**What must be written in this section:**

- **Paragraph 1 (The Broad Domain):** Introduce the high-level subfield of Computer Science (e.g., *Edge AI, Computer Vision, Cryptography, Distributed Systems*) and state its significance in modern computing.
- **Paragraph 2 (The General High-Level Problem):** Briefly describe the practical or computational challenge that exists in current systems today.
- **Paragraph 3 (The Proposed Solution & Intent):** State what the study aims to do at a high level and highlight the primary *CS research contribution* (e.g., optimizing execution speed, reducing memory footprint, improving accuracy).

### 3.2 Background of the Study (The Technical Case File & Macro-Meso-Micro Model)

Unlike the high-level Introduction, the **Background of the Study** functions as a rigorous, data-driven **Technical Case File** (2 to 4 pages). It systematically narrows down from broad theoretical paradigms down to concrete computational bottlenecks using the **Macro-Meso-Micro Funnel Model**:

**• MACRO PERSPECTIVE (Global Algorithmic & CS Paradigm Trends)**

Establish the global state of the core CS subfield. Cite recent peer-reviewed studies (IEEE Xplore, ACM DL, Springer) from the last **3–5 years**. Show how state-of-the-art algorithms function worldwide.

**• MESO PERSPECTIVE (National / Regional Application Context)**

Bring the focus to the Philippine national or regional context. Detail infrastructure constraints (e.g., limited network bandwidth, prevalence of low-tier Android hardware) and relevant regulatory frameworks (e.g., Data Privacy Act / Republic Act 10173).

**• MICRO PERSPECTIVE (Specific Testbed, Dataset & Computational Bottleneck)**

Pinpoint the target dataset, local hardware testbed, or operational environment. Detail the specific computational inefficiencies occurring under the "As-Is" baseline (e.g., high memory overhead, excessive latency, cold-start degradation).

**Technical Writing Rule: Quantitative Baseline vs. Emotional Narrative**\
Avoid subjective claims such as *"Users find the facial recognition slow and annoying."*\
Always state data-driven baselines: *"Existing LBP algorithms exhibit a 42% drop in recognition accuracy under lighting conditions below 50 lux, with inference times exceeding 1.8 seconds per frame on constrained hardware."*

## 4. Statement of the Problem (SOP) & Research Objectives

The Statement of the Problem defines the precise scientific gap your project resolves. It consists of a **General Problem Paragraph** followed by **Specific Questions** aligned with Design Science Research (DSR) or Experimental Methodologies.

**General Problem Template for CS Research:**\
*"This study aims to design, implement, and empirically evaluate \[Name of Proposed Algorithm / Optimized Model / Heuristic PoC\] for \[Target Application Domain\] to resolve current limitations in \[Specific Computational Bottleneck, e.g., inference latency, accuracy degradation, cold-start data sparsity\]."*

### Research Phase Alignment Matrix (SOP vs. Research Objectives)

In CEU CS Research, every interrogative SOP question must directly mirror an action-oriented declarative Research Objective across the 4 DSR phases:

| Research Phase | Statement of the Problem (Interrogative) | Research Objective (Declarative Action) |
|----|----|----|
| Phase 1: Baseline Analysis | 1\. What are the performance limitations and accuracy metrics of existing baseline models under \[Target Condition\]? | 1\. To establish benchmark performance metrics of existing algorithms using \[Standard Dataset / Testbed\]. |
| Phase 2: Design & Artifact Creation (DSR) | 2\. How can \[Proposed Novel Algorithm / Optimization Technique\] be designed and integrated into a proof-of-concept software artifact? | 2\. To design and implement \[Proposed Novel Algorithm\] using Python and PyTorch into a functional proof-of-concept artifact. |
| Phase 3: Empirical Benchmarking | 3\. What is the empirical performance of the proposed artifact in terms of execution time (*t_(exec)*), memory footprint (*M_(RAM)*), and accuracy (*Acc*)? | 3\. To empirically measure the artifact's performance across execution speed, memory utilization, and classification accuracy. |
| Phase 4: Comparative Validation | 4\. Is there a statistically significant performance difference between the proposed model and existing baseline methods? | 4\. To statistically evaluate performance gains of the proposed algorithm against standard baselines using paired *t*-test / ANOVA. |

## 5. Theoretical & Conceptual Frameworks

### A. Theoretical Foundations in Computer Science

Your CS research must be anchored in recognized scientific theories and formal paradigms, such as:

- **Design Science Research Methodology (DSRM - Peffers et al.):** Iterative cycle: Problem Identification → Solution Objectives → Artifact Design → Demonstration → Empirical Evaluation → Communication.
- **Computational Complexity Theory:** Big-O notation (*O(n)*, *O(n log n)*) evaluating time and space trade-offs.
- **Machine Learning & Algorithmic Principles:** Optimization functions, gradient descent, deep neural networks, graph theory, or cryptographic security proofs.

### B. CS Empirical Performance Metrics

**1. Processing Speed & Latency:**\
Inference time per frame (ms), throughput (FPS), execution time (*t_(exec)*).

**2. Computational Efficiency:**\
CPU/GPU load (0–100%), RAM allocation (MB), FLOPs count.

**3. Model Precision & Quality:**\
Accuracy, Precision, Recall, F1-Score, Mean Average Precision (mAP).

**4. Robustness & Edge Cases:**\
Performance under noise, low-light degradation, packet loss, sparse data.

### C. Input-Process-Output (IPO) Conceptual Model for CS

Represent the analytical and experimental flow of your CS project using the IPO format:

[TABLE]

## 6. Scope, Significance & Terminology

#### 🎯 Scope & Algorithmic Delimitations

**Scope:** Explicitly states target dataset sizes, model parameters, hardware testbeds, and evaluated environment conditions.

**Delimitations:** Explicitly states boundary choices (e.g., *"The study is delimited to 8-bit integer model quantization and does not test 4-bit compression schemes."*)

#### 🤝 Significance of the Study

- **Primary Beneficiaries:** Industry practitioners needing lightweight/fast algorithmic implementations.
- **Secondary Beneficiaries:** Computer Science community & research field gaining verified empirical benchmarks.
- **Tertiary Beneficiaries:** Future CS students leveraging the algorithmic methodology or codebase.

**📖 Operational Definitions Rule for CS:**\
Avoid dictionary definitions. Define technical and algorithmic terms operationally based on how they function inside your mathematical formulas, dataset pipeline, or execution codebase.
