# M3. Lesson 1


- Type: Page

- Item ID: 4357275

- Module ID: 496210


## Content


Module 3

# Standards, Ethics, and Proposal Development

Ensuring your research is ethically sound, legally compliant, and globally reproducible.

### The Final Leg of the Proposal Journey

Welcome to the final phase! You have framed your question and mapped out your technical blueprint. However, a great research proposal is more than just a clever algorithm; it is a commitment to **professional responsibility**. In this module, we move beyond "Can we build it?" to "Should we build it?" and "How do we protect those involved?"

## 1. Ethics and Responsibility in Computing Research

Ethical violations in computing are rarely intentional, but their impact can be devastating—ranging from leaking sensitive biometric data to perpetuating systemic discrimination through biased machine learning models. As a researcher, ethics must be "baked into" your methodology from Day 1.

### The Three Pillars of Computing Ethics

1

#### Data Privacy & Compliance

In the age of Big Data, your research likely touches user telemetry, app usage metrics, or personal identifiers. You are legally and ethically bound to protect this data.

**Key Considerations:**

- **Anonymization:** Have you stripped away Names, IPs, and IDs from your dataset?
- **Legal Compliance:** Does your storage pipeline meet the requirements of the **Philippines' Data Privacy Act of 2012** or **GDPR**?
- **Data Minimization:** Are you only collecting the *specific* data points required for your Research Question?

2

#### Informed Consent

If your methodology involves human participants—whether for UI/UX testing, developer surveys, or beta-testing an algorithm—consent is non-negotiable.

**The Informed Consent Rulebook:**

- **Transparency:** Participants must understand *exactly* what data is being tracked (e.g., "We are logging your keystroke speed").
- **Right to Withdraw:** Participants must be told they can stop the test at any time without penalty.
- **Purpose Disclosure:** They must know how the findings will be used (e.g., "This data will be published in a conference paper").

3

#### Algorithmic Bias & Fairness

"Garbage In, Garbage Out" (GIGO) is not just a technical failure; it is an ethical one. If your training data is skewed, your results will be discriminatory.

**Example:**

If you are training a facial recognition model using a dataset that is 90% Western faces, your system will likely have a high "False Rejection Rate" for Asian or African users. In your proposal, you must state how you will balance your dataset to ensure equitable performance across different demographics.

#### 🛡️ Technical Ethics Checklist

Before submitting your Chapter 3 (Methodology), ask yourself these technical-ethical questions:

**Security** Is the participant data encrypted at rest and in transit?

**Inclusivity** Is my training data representative of the end-user population?

**Transparency** Can I explain *how* my algorithm reached its conclusion (Explainable AI)?

#### Remember...

Ethical research is **reproducible research**. By being transparent about your data and your biases, you allow the global computing community to build upon your work with confidence.
