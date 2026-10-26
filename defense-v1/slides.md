---
theme: none
colorSchema: light
fonts:
  sans: Inter
  provider: none
title: Benchmarking Multi-Task Optimizers for Humanitarian Classification and Named Entity Recognition of Taglish Typhoon Tweets
transition: view-transition
canvasWidth: 1040
defaults:
  transition: slide-left
layout: title
---

# Benchmarking Multi-Task Optimizers for Humanitarian Classification and Named Entity Recognition of Taglish Typhoon Tweets

::left::

### Research Group

- Isaiah Raphael Almuena
- Prinz Edward Ducyogen
- Carlo Miguel Geronimo
- Kyle Uri Gabriel Lu
- Yngwie Sebastian Patricio

::right::

### Research Advisor

Engr. Ma Christina A. Florentino

### Date of Defense

October 27, 2026

---
clicks: 13
---

<Overview />

---
clicks: 5
section: Background and Problem Statement
---

<RiskMap />

---
clicks: 15
section: Background and Problem Statement
---

<div class="stack">
  <ApproachDiagram />
  <NegativeTransfer :first="6" />
</div>

---
section: Background and Problem Statement
---

<div class="quote">

### Statement of the problem

How can **a multi-task model** perform <strong class="classification">humanitarian classification</strong> and <strong class="ner">NER</strong> on Taglish typhoon posts while keeping **negative transfer** and **training cost** low?

</div>

---
section: Objectives and Scope
---

<div class="quote">

### General objective

The general objective is to **develop a multi-task model** that performs <strong class="classification">humanitarian classification</strong> and <strong class="ner">NER</strong> on Taglish typhoon posts.

</div>

---
clicks: 4
section: Objectives and Scope
---

<SpecificObjectives />

---
clicks: 7
section: Objectives and Scope
---

<Scope />

---
clicks: 10
section: Literature and Baseline Algorithms
---

<Literature />

---
clicks: 5
section: Expected Scientific Contribution
---

<Contribution />

---
clicks: 5
section: Expected Scientific Contribution
---

<Benefits />

---
clicks: 7
section: Software Artifact and System Design
---

<SystemDiagram />

---
clicks: 5
section: Software Artifact and System Design
---

<Tools />

---
section: Software Artifact and System Design
---

<Prototype />

---
clicks: 8
section: Testing Method and Metrics
---

<Dataset />

---
clicks: 4
section: Testing Method and Metrics
---

<AnnotatorAgreement />

---
clicks: 7
section: Testing Method and Metrics
---

<Measures />

---
clicks: 5
section: Testing Method and Metrics
---

<OptimizerSelection />

---
clicks: 3
section: Testing Method and Metrics
---

<FinalComparison />

---
clicks: 2
section: Testing Method and Metrics
---

<PrototypeTests />

---
clicks: 6
section: Development and Testing Schedule
---

<Schedule />

---
clicks: 3
section: Immediate Next Steps
---

<NextSteps />

---
clicks: 3
section: Immediate Next Steps
---

<LabelingCheckpoints />

---
clicks: 5
section: Technical Risks and Backup Plans
---

<Risks />

---

<div class="quote">

### In summary

One model that <strong class="classification">sorts Taglish typhoon posts by type of report</strong> and <strong class="ner">picks out their key details</strong>, with the optimizer that best reduces **negative transfer** without needing too much training time or memory

</div>

---
section: References
---

<ReferenceList :page="1" />

---
section: References
---

<ReferenceList :page="2" />
