# M2. Chapter 3 - Methodology


- Type: Page

- Item ID: 4494449

- Module ID: 496209


## Content


CEU Department of Computer Science and Information Technology

## M4. Chapter 3:  Methodology

A Guide to System Design, Experimental Protocols, Datasets, Hardware Specifications & Empirical Evaluation

## Learning Objectives

By the end of this module, Computer Science research students will be able to:

* Align project workflows with Design Science Research Methodology (DSRM) or Quantitative Experimental Design.

* Define CS "Subjects of the Study" and "Sampling Techniques" in terms of datasets, data augmentations, and test environment instances.

* Document software profiling tools, benchmarking suites, and hardware testbeds as scientific research instruments.

* Formulate rigorous procedures for data gathering, pre-processing, and pipeline execution.

* Apply inferential statistical treatments (paired t-tests, ANOVA) and CS metrics to validate performance hypotheses.

* Detail the Software System Description including architecture diagrams, use cases, and software/hardware requirements.

* Design empirical Testing and Evaluation procedures for algorithmic latency, throughput, memory overhead, and model accuracy.

Computer Science Methodology Rule: Unlike general social science research that evaluates human opinions, CS methodology focuses on artifact construction, computational execution, mathematical formulations, and empirical testbed benchmarking. Every subsection must reflect technical precision.

## Detailed Breakdown of Chapter 3 Sections

## 3.1 Research Methodology

Specify the overall scientific framework governing the study. In CS, this is typically either Design Science Research Methodology (DSRM) or Quantitative Experimental / Benchmarking Method.

* Design Science Research (DSRM): Explain the 6 iterative phases: Problem Identification → Objective Definition → Artifact Design & Development → Demonstration → Evaluation → Communication.

* Experimental / Comparative Method: Explain how baseline algorithms will be compared against the proposed modified/optimized algorithm under controlled hardware/data variables.

## 3.2 Subjects of the Study

In CS, "subjects" rare refers to human respondents unless evaluating UI/UX. Instead, define the computational artifacts, target datasets, image samples, network packets, or software modules being tested.

* Dataset Subjects: Specify image resolution, file formats, total sample counts, and class distributions (e.g., 10,000 labeled RGB images across 5 plant disease categories).

* Environment Subjects: Specify target system instances (e.g., low-end ARM v8 Android micro-controllers, edge computing nodes).

## 3.3 Sampling Technique

Describe how data samples were selected, partitioned, or synthesized for training, testing, and validation pipelines.

* Dataset Splitting: Specify train/validation/test split ratios (e.g., 70% Train, 15% Validation, 15% Test) using Stratified Random Sampling to preserve class distribution.

* Data Augmentation / Resampling: Detail synthetic techniques used (e.g., SMOTE for class imbalance, rotation/brightness shifts for computer vision models).

* Cross-Validation: Define $k$-fold cross-validation strategies (e.g., 5-Fold CV) to prevent model overfitting.

## 3.4 Instruments to Be Used

Detail all software diagnostics, profiling tools, hardware benchmark kits, and measurement software used to collect empirical data.

* Software Profilers: PyTorch Profiler, TensorRT, Android Studio Profiler, Memory Profiler, Wireshark (for network throughput).

* Benchmarking Suites: MLPerf, custom execution time measurement scripts utilizing high-resolution timers (e.g., time.perf_counter()).

* Hardware Instruments: Power meters (e.g., USB Power Monitor for energy consumption), hardware thermal sensors, GPU-Z / NVTOP.

## 3.5 Procedure of Data Gathering

Present a step-by-step chronological protocol of how raw data is acquired, pre-processed, fed into the system, and recorded during execution.

* Acquisition: Downloading public benchmark datasets or capturing raw local data.

* Pre-processing & Normalization: Resizing images, removing noise, feature scaling, encoding categorical variables.

* Execution Protocol: Cold-start runs, warming up hardware caches, running N iteration trials (e.g., 100 benchmark runs per sample) to eliminate hardware execution noise.

* Log Generation: Automatically outputting execution logs into CSV/JSON format capturing $t_{\text{exec}}$, RAM usage, and classification logs.

## 3.6 Statistical Treatment

Define the mathematical and statistical formulas used to validate whether the proposed system provides a statistically significant improvement over baseline methods.

* Inferential Tests: Paired-sample t-test (to compare execution times before and after quantization), One-Way ANOVA (when comparing 3+ algorithmic architectures), or Wilcoxon Signed-Rank Test (for non-parametric data).

* Hypothesis Formulation:

* $H_0$: There is no significant difference in execution latency ($t_{\text{exec}}$) between the baseline model and the proposed optimized model ($p \ge 0.05$).

* $H_1$: The proposed optimized model yields a statistically significant reduction in latency ($p < 0.05$).

## 3.7 Data Analysis

Specify the quantitative Computer Science evaluation formulas used to interpret system performance:

• Accuracy = (TP + TN) / (TP + TN + FP + FN)
• Precision = TP / (TP + FP)
• Recall = TP / (TP + FN)
• F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
• Mean Latency (ms) = Total Execution Time / Total Number of Test Frames

## 3.8 Software System Description

Provide a comprehensive technical specification of the proposed software artifact or experimental system pipeline.

## 3.8.1 System Structure

Present high-level software and model architecture diagrams: System Architecture Diagrams, Block Diagrams, Algorithmic Data Flow Diagrams (DFD), or Neural Network Layer Diagrams.

## 3.8.2 Use Cases

Provide UML Use Case Diagrams and Use Case Specifications describing how key actors (admin, end-user, background API workers) interact with the proof-of-concept software artifact.

## 3.8.3 Materials and Equipment

Detail the technical specifications of both development environments and deployment testbeds:

* Software Requirements: Operating System (Ubuntu 22.04 LTS), Programming Languages (Python 3.10), Libraries & Frameworks (PyTorch 2.1, OpenCV 4.8, CUDA 11.8), IDEs (VS Code).

* Hardware and Connectivity Requirements: CPU specifications, GPU VRAM bounds (e.g., NVIDIA RTX 3060 12GB), RAM capacity (16GB DDR4), target testbed hardware (e.g., Raspberry Pi 4 Model B 4GB ARM, low-end Android device with Snapdragon 680), and network bandwidth conditions.

## 3.9 Testing and Evaluation

Describe the exact testing methodology used to evaluate the artifact's functional and empirical performance prior to final results compilation.

* Functional & Software Testing: Unit testing, integration testing, and stress/load testing of the PoC codebase.

* Empirical Performance Benchmarking: Stress-testing inference speed across various batch sizes ($b = 1, 8, 16, 32$), monitoring memory leaks over prolonged execution, and recording thermal/throttling behavior under hardware constraints.

* Standard ISO/IEC 25010 Evaluation (If Applicable): If human evaluation is required, use ISO 25010 software quality standards focusing on Performance Efficiency, Reliability, and Usability.

Checklist for Chapter 3 Completion:
Ensure every dataset size, hardware model, software library version, and statistical formula is explicitly named. A reader should be able to completely replicate your experimental environment and codebase results solely by reading your Chapter 3!