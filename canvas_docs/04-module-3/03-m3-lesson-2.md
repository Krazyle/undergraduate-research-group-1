# M3. Lesson 2

- Type: Page
- Item ID: 4357276
- Module ID: 496210

## Content

## 2. Technical Standards & Reproducibility

In the global computing ecosystem, research does not exist in a vacuum. For your proposal to be taken seriously by the international community, it must speak the "universal language" of modern engineering and documentation standards.

### The Importance of Technical Standards

Standards ensure **interoperability**—the ability of your system to work with others. If you are proposing a new technical artifact, you must identify which existing standards it interacts with or adheres to:

Network & Hardware

#### IEEE & IETF (RFC)

If building network protocols or hardware interfaces, you must reference **RFCs (Request for Comments)** or **IEEE 802** standards.\
\
**Example:** Proposing a new routing algorithm? You must compare it against *RFC 791 (IPv4)* or *RFC 2460 (IPv6)*.

Software Quality

#### ISO/IEC Standards

If your research focuses on software architecture or quality assurance, adhere to **ISO/IEC 25010**, which defines software product quality.\
\
**Example:** Using the *SQuaRE* framework to evaluate the maintainability and portability of a new microservices architecture.

Data & Web

#### W3C & ANSI

Web-based research must follow **W3C (World Wide Web Consortium)** guidelines for accessibility (WCAG) and data structures (JSON/XML).\
\
**Example:** Ensuring a new web interface complies with *WCAG 2.1* for universal accessibility.

### Solving the Reproducibility Crisis

Computer Science is currently facing a **Reproducibility Crisis**. A shocking percentage of published papers cannot be replicated because authors provide vague descriptions, hide their source code, or use private datasets that cannot be audited.

#### ❌ The "Black Box" Approach (Bad Practice)

"The algorithm was implemented in Python. Due to the proprietary nature of the code and dataset, they are not available for public review."

#### ✅ The "Open Science" Approach (Best Practice)

"To ensure full reproducibility, the anonymized dataset, source code, and configuration scripts (Dockerfiles) are hosted in a public GitHub repository and archived on Zenodo with a persistent DOI."

#### How to Write a Reproducibility Plan

In your proposal (Chapter 3), you should include a short "Reproducibility Plan" section. This must specify:

**Version Control:** State that your code will be maintained on **GitHub** or **GitLab**.

**Environment Specification:** Mention how you will "freeze" your dependencies (e.g., using `requirements.txt` for Python, `package.json` for Node.js, or **Docker** for full system containers).

**Persistent Archival:** Mention using services like **Zenodo** or **Figshare**. These services provide a Digital Object Identifier (DOI), ensuring your code stays online even if a personal GitHub account is deleted.

#### Why This Matters for Your Proposal

Reviewers are much more likely to approve a proposal that is **transparent**. By proving that you follow global standards and that your results can be audited by any other engineer in the world, you establish yourself as a rigorous and credible computer scientist.
