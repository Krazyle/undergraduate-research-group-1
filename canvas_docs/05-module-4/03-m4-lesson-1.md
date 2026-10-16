# M4. Lesson 1


- Type: Page

- Item ID: 4357288

- Module ID: 496211


## Content


Module 4

# Presentation, Defense, and Finalization

The ultimate test of a computer science researcher: advocating for your architectural choices before a panel of experts.

### Welcome to the Final Stage!

You have designed your architecture, cleared your ethical reviews, and typeset your manuscript beautifully in LaTeX. Now, writing a great proposal is only half the battle. You must be able to stand before an academic and industry panel, articulate your technical contributions, and defend your methodology from critical scrutiny.

## 1. Designing High-Impact Technical Slides

A common mistake student researchers make during their proposal defense is copy-pasting paragraphs from their manuscript directly onto slides and reading them aloud. **Your panel can read much faster than you can speak.** If your slides mirror your speech, you make your presentation redundant and lose your audience's attention. Your slides must complement your spoken words, acting as a visual anchor.

📊

### The 80/20 Rule of CS Presentations

Your slides should communicate complex technical structures efficiently. Aim for an allocation where **80% of your slide real estate** consists of visual technical evidence and only **20% consists of text or key bullet points**.

**20% High-Impact Text** Key assertions, metrics, core limitations, or mathematical definitions.

**80% Visual Evidence** System architectures, flowcharts, UML sequence diagrams, database schemas, or UI/UX wireframes.

### Replace Prose with Modeling

Instead of writing out a long paragraph describing how data flows through your system, map it. Visual modeling is the standard medium of communication in software engineering and computer science.

❌ The Prose-Heavy Way (Bad Slide Practice)

"Our mobile client communicates with our middleware API Gateway using standard HTTPS requests. The API Gateway then authenticates the user's session token using an OAuth service. Once authorized, the gateway routes the request to our primary Microservices layer (User Service or Catalog Service), which retrieves information directly from our MongoDB database before packaging it back into a JSON array to send back down to the client."

✅ The Technical Modeling Way (Best Slide Practice)

*Instead of writing that block, you display a clean **UML Sequence Diagram** or a **System Architecture Diagram** mapping:*

\[Mobile Client\] ----(HTTPS / JSON)---\> \[API Gateway\]\
                                    \|\
                                    +---\> (Authenticates via OAuth)\
                                    \|\
                                    v\
                             \[Microservices\] ----\> \[MongoDB Database\]

\*Note: In the defense, you would display this diagram and talk the panel through the flow step-by-step.

#### 🛠️ Slide Design Quick Reference

When compiling your slide deck, use this matrix to select the best visual representation for your content:

**System Architecture** Shows components, infrastructure, layers, and communication protocols.

**Flowcharts** Shows chronological logic, algorithmic conditions, and decision trees.

**Data Mockups & UI** Shows user interaction points, frontend wireframes, or database schema entities.

#### Keep in Mind...

Your slides are there to prove your engineering choices are structurally sound and based on rigorous planning. Keep them sharp, visual, and highly academic.
