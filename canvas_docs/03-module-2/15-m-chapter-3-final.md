# M. Chapter 3 - FINAL


- Type: Assignment

- Item ID: 4546385

- Module ID: 496209

- Due Date: 2026-09-30T15:59:59Z

- Points: 50.0


## Instructions


## Module Task: Research Proposal Chapter 3 Revision

Course: Undergraduate Research 1 | CEU Department of Computer Science and Information Technology

Submission Format: PDF document uploaded via Canvas

Total Points: 50 Points

## Task Overview & Instructions

Now that you have received feedback on your initial draft of Chapter 3: Research Methodology, you are required to revise and refine your submission. Carefully review all inline comments and general feedback provided on your previous draft in Canvas, and address every required revision in this updated manuscript.

Computer Science Methodology Rule:

Unlike general social science research that evaluates human opinions, CS methodology focuses on artifact construction, computational execution, mathematical formulations, and empirical testbed benchmarking. Every subsection must reflect technical precision and complete reproducibility.

## Required Manuscript Formatting

* Paper Size: Short Bond Paper (8.5" × 11")

* Margins: 1.5 inches (Left) for binding allocation; 1.0 inch (Top, Bottom, Right)

* Typography: Times New Roman, 12pt font size

* Spacing: Double-spaced body text

* Citations & Standards: APA 7th Edition for references; explicit technical specifications for all hardware, libraries, and statistical models.

## Required Chapter 3 Sections to Revise

3.1 Research Methodology: Specify the overall scientific framework governing the study (Design Science Research Methodology / DSRM 6-phase framework or Quantitative Experimental / Benchmarking Method).

3.2 Subjects of the Study: Define computational artifacts, datasets, image samples, network packets, or hardware system instances being tested rather than human respondents.

3.3 Sampling Technique: Detail dataset splitting ratios (e.g., 70% Train, 15% Validation, 15% Test), Stratified Random Sampling, synthetic data augmentation, or k-fold cross-validation strategies.

3.4 Instruments to Be Used: Specify all software profilers (PyTorch Profiler, Android Studio Profiler, TensorRT), benchmarking suites (MLPerf, custom high-resolution timers), and hardware measurement instruments.

3.5 Procedure of Data Gathering: Provide a chronological step-by-step protocol: Data Acquisition → Pre-processing & Normalization → Execution Protocol (warmup runs, N trials) → Automated CSV/JSON Log Generation.

3.6 Statistical Treatment: Define inferential tests (Paired t-test, ANOVA, Wilcoxon Signed-Rank) and explicit null (H0) and alternative (H1) hypotheses with confidence thresholds (p < 0.05).

3.7 Data Analysis: Detail explicit quantitative CS performance formulas (Accuracy, Precision, Recall, F1-Score, Mean Latency, Throughput, Memory Overhead).

3.8 Software System Description:

3.8.1 System Structure: System Architecture, Block, DFD, or Neural Network Layer Diagrams.

3.8.2 Use Cases: UML Use Case Diagrams and Use Case Specifications for primary actors and system workers.

3.8.3 Materials and Equipment: Complete specs for OS, Python versions, ML frameworks, CUDA versions, CPU/GPU models, VRAM bounds, and target testbed environments.

3.9 Testing and Evaluation: Describe PoC code unit/integration testing, empirical stress benchmarking across batch sizes, thermal/memory leak behavior, and ISO/IEC 25010 evaluation standards (if applicable).

## Grading Rubric Matrix (50 Points)

Evaluation Criteria
Exemplary (100%)
Proficient (80%)
Developing (60%)
Unacceptable (≤40%)
Points

CS Research Design & Rigor
Flawlessly aligns workflow with DSRM or Quantitative Experimental Design. Explicitly treats datasets, algorithms, and testbed instances as subjects without human respondent confusion. Fully resolves design feedback.
Clear research design and CS subjects definition. Minor ambiguities in pipeline stages or sampling descriptions.
Confuses CS experimental design with social science survey methods. Vague dataset split or sampling descriptions.
Lacks structured research design. Incorrectly uses human survey respondents for pure algorithmic evaluation. Ignores feedback.
/10

Instruments, Profilers & Experimental Protocols
Comprehensively details profiling software, benchmarking timers, and hardware instruments. Chronological procedure ensures 100% experimental reproducibility (warmup runs, trial counts, log generation).
Identifies main profiling tools and procedures, but lacks minor execution details (e.g., missing hardware cache warmup protocols).
Lists general software tools without specifying versioning, profiling metrics, or automated log generation protocols.
Missing instruments section or describes general manual observation without computational profiling tools.
/10

Statistical Treatment & CS Performance Formulas
Rigorous inferential statistical tests (paired t-test/ANOVA) paired with clear H0/H1 hypotheses. Includes explicit mathematical formulas for all performance and accuracy metrics.
Includes appropriate statistical tests and hypotheses, but minor metric formulas need clearer definition or unit specifications.
Presents generic statistical tests without hypothesis formulation, or lacks mathematical formulas for CS evaluation metrics.
Missing statistical treatment and evaluation metrics entirely, or uses subjective satisfaction scores for algorithmic performance.
/10

Software System Description & Specifications
Exemplary architectural/DFD/neural layer diagrams, complete UML Use Cases, and precise specifications for dev environments and edge target testbeds (OS, CUDA, VRAM, RAM).
System architecture and specifications are solid, but missing minor environment details (e.g., exact framework versions or hardware VRAM bounds).
Incomplete system architecture diagrams; generic hardware/software requirements without target deployment constraints.
Missing system description section, architectural diagrams, or hardware/software specifications.
/10

Empirical Testing, ISO Standards & Mechanics
Thorough testing procedures covering unit/integration testing, batch stress tests, and ISO/IEC 25010 parameters. Strictly adheres to CEU manuscript mechanics (Times 12pt, double space, margins).
Clear testing protocols and good formatting adherence, with minor omissions in stress testing variables or layout parameters.
Vague testing description; misses critical stress/empirical evaluation parameters; multiple CEU layout deviations.
Lacks empirical testing procedures; disregards CEU layout and manuscript guidelines. Ignores feedback.
/10

TOTAL SCORE:
/50