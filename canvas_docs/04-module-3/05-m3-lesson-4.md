# M3. Lesson 4

- Type: Page
- Item ID: 4357278
- Module ID: 496210

## Content

## 4. Professional Typesetting with LaTeX and Overleaf

Word processors like Microsoft Word or Google Docs are fantastic for business memos or creative essays, but they are notoriously poorly suited for technical computer science papers. They frequently break image placements, warp complex mathematical notation, and corrupt bibliography formatting when documents grow large.

### WYSIWYG vs. Programmatic Typesetting

Traditional processors use a **WYSIWYG** (What You See Is What You Get) model, forcing the author to manage both content and style simultaneously. Computing research uses **LaTeX**, a document preparation system where you write in plain text and use code-like markup to handle structural rendering automatically.

| Feature | Standard Word Processors | LaTeX System |
|----|----|----|
| Formatting | Manual GUI manipulation. Prone to unintended alignment shifts when adding text. | Automated via publisher style sheets (e.g., `\documentclass{IEEEtran}`). |
| Math Equations | Clunky, slow point-and-click equation editors that scale poorly. | Textual syntax using robust engine blocks like `amsmath`. |
| References | Requires manual sorting or error-prone external plugins. | Seamless mapping via dynamic **BibTeX** files and database keys. |

### The Power of Mathematical Markups

Instead of wrestling with formatting tools to display mathematical models, algorithms, or complex execution variables, LaTeX allows you to write math inline or as standalone blocks directly inside your sentences using standard keyboard characters.

What you type in the editor:

The activation layer of the target neural network is calculated as:\
\$\$f(x) = \sigma(W^T x + b)\$\$\
where \$W\$ represents the matrix weight layer.

How the LaTeX compiler automatically renders it:

The activation layer of the target neural network is calculated as:

f(x) = σ(W^(T)x + b)

where W represents the matrix weight layer.

### Cloud Collaboration via Overleaf

Historically, using LaTeX required installing massive local software frameworks (like MiKTeX or TeX Live) and command-line compilers. Today, we use **Overleaf**—a cloud-based LaTeX development environment that enables realtime team collaboration, automated cloud compilation, and git-integrated version history management.

**1. The Preamble** Where you import libraries, set page layouts, and specify target publisher styles.

**2. The Document Body** Enclosed in `\begin{document}`, where you isolate text from styling directives.

**3. The BibTeX Engine** An external ledger (`.bib`) keeping track of keys, linking references smoothly.

#### 💡 Action Item for the Proposal Layout

You will be provided with an official institution-approved LaTeX template on Overleaf. Do not adjust spatial margins, font sizes, or title alignments manually. Focus entirely on clear technical writing, and let the layout engine take care of processing the compliance standards.
