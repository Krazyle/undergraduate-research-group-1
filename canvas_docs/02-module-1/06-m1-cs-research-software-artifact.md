# M1. CS Research - Software  Artifact


- Type: Page

- Item ID: 4494455

- Module ID: 496208


## Content


CEU Department of Computer Science and Information Technology

## Guidance Module: Understanding the CS Software Artifact (Proof of Concept)

Navigating the CEU CS Artifact Standard: UI/UX Integration vs. Algorithmic Core

## Module Learning Objectives

By reading this guide, Computer Science research students will understand:

* The precise definition of a Proof of Concept (PoC) in Computer Science research.

* Why command-line scripts or headless Jupyter Notebook testbeds are insufficient for CEU CS requirements.

* The core distinctions between a CS Software Artifact and an IT Research Software System.

* How to integrate a fully functional, usable UI/UX layer that serves to showcase and benchmark the underlying algorithm/model.

## 1. What is a CS Proof of Concept (PoC) Artifact?

In Computer Science research, a Proof of Concept (PoC) is a functional software artifact developed to demonstrate the feasibility, algorithmic efficiency, or practical utility of a proposed theoretical contribution (such as a neural network model, optimization algorithm, data structure, or security protocol).

The CEU CS Standard for Software Artifacts:
At CEU CSIT, a CS software artifact cannot be just a headless terminal script, a console program, or a Jupyter Notebook testbed. It must be a complete, fully functioning application with a practical UI/UX. However, unlike Information Technology (IT) projects, it does not need to be a full-blown commercial system packed with enterprise management features.

## 2. CS Software Artifact vs. IT Software System

To avoid spending too much time on unnecessary business logic (or conversely, submitting a raw algorithm without a frontend), review the key structural differences between CS and IT software deliverables below:

Dimension
Computer Science PoC Artifact (Your Objective)
Information Technology System (Not Required)

Primary Focus
Algorithmic novelty & performance validation (e.g., latency, inference accuracy, memory usage, execution speed).
Business process automation & system deployment (e.g., workflow efficiency, user adoption, organizational fit).

UI / UX Scope
Clean, intuitive, fully functional interface designed to feed inputs into the algorithm and visually present real-time outputs and metrics.
Extensive enterprise UI with multi-role dashboards, complex form wizards, and automated document generation.

System Features
Core functionalities directly supporting the algorithm (e.g., live camera feed processing, file upload, inference visualizer, benchmark logger).
Full CRUD operational modules: billing, inventory management, multi-tier user role permissions, audit trails, and reporting suites.

Evaluation Basis
Computational metrics ($t_{\text{exec}}$, RAM, accuracy, throughput) alongside usability of the interface.
ISO 25010 user satisfaction surveys, business ROI, and process efficiency gains.

## 3. What Counts as an Acceptable CS Artifact UI/UX?

Your software artifact must bridge the gap between complex algorithmic computation and practical human usability. The UI/UX serves as the interface that allows panelists and users to interact with your algorithm in real time.

## ✗ Unacceptable CS Artifacts (Too Simple)

* Running a command-line script in a terminal (e.g., python main.py).

* Executing code blocks sequentially inside a Jupyter Notebook or Google Colab.

* A bare API endpoint with no visual interface (e.g., returning raw JSON in Postman without a frontend).

* Mockup screens (Figma/Adobe XD) without underlying functional backend code.

## ✓ Expected CEU CS Artifacts (Functional PoC)

* A Web App (Streamlit, Gradio, React/Flask) with real-time file upload, interactive controls, and visual prediction output.

* A Mobile Application (Flutter/Android) performing edge processing (e.g., camera-based object detection with UI bounding boxes).

* A Desktop Interface (PyQt, Tkinter, Electron) displaying live performance analytics graphs alongside system outputs.

## 4. Concrete Example of a CS PoC vs. IT System

## Scenario: Agricultural Leaf Disease Identification

The Theoretical CS Core: A lightweight, dynamic 8-bit quantized CNN model designed for fast plant disease classification on low-cost hardware.

## What the CS Software Artifact UI/UX Should Include:

* Input Interface: A clean camera/upload screen where users select or capture a plant leaf image.

* Execution View: A real-time visual output overlaying detected disease bounding boxes and confidence percentages on the image.

* Performance Dashboard Overlay: A small panel displaying execution metrics ($t_{\text{exec}} = 24\text{ms}$, RAM Usage $= 18\text{MB}$, Hardware FPS $= 32$).

* Result Summary: A simple screen displaying disease classification details and recommended actions.

## What the CS Artifact DOES NOT Need (IT Features):

* User registration, login/logout, or password reset modules.

* Farmer profile management or field location GPS tagging.

* Inventory tracking for purchasing pesticides or fertilizers.

* Multi-role administrative access levels (Superadmin, Farmer, Agronomist).

## 5. Summary Checklist for CS Research Artifacts

Before presenting your Chapter 3 software description or defense demo, ask yourself:

* Can a non-technical end user test my algorithm without opening a terminal or reading code line-by-line?

* Is the UI clean, responsive, and functional enough to interactively feed data into my backend core?

* Does the artifact demonstrate real-time output display alongside performance logs (latency, accuracy, memory)?

* Have I avoided spending unnecessary development time on commercial/IT business features that do not contribute to my research objectives?