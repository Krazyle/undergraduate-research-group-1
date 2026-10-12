# An AI-Enhanced Decision Support System for Barangay Incident Monitoring and Community Risk Assessment

**Drake Joshua J. Besario; Engr. Levi A. Corvera; Dr. Jessica Rose E. Fernandez**

Surigao del Norte State University, Philippines

<u>drakebesario3@gmail.com</u>; <u>levi.corvera@ssct.edu.ph</u>; <u>jess.fernandez@ssct.edu.ph</u>

**DOI: 10.47760/OAJMR.2026.v02i06.002**

**Abstract:** Incident monitoring at the barangay level in the Philippines has long depended on manual, paperbased processes that leave local officials with little capacity to detect patterns, assess community risks, or respond to emerging safety concerns in a timely manner. This study addresses that gap through the design, development, and evaluation of an AI-enhanced decision support system deployed at Barangay Washington — a platform built not merely to digitize existing records, but to transform raw incident data into actionable intelligence for grassroots governance. The system was developed using HTML5, CSS3, Vanilla JavaScript, Python Django, and SQLite, and incorporates a rule-based AI classification engine that evaluates each reported incident across four weighted dimensions: incident type, frequency of recurrence, location history, and time-of-day patterns. Based on these criteria, incidents are automatically categorized into Low, Medium, or High risk classifications. Usability evaluation was conducted with barangay officials and secretaries using the System Usability Scale, while AI classification performance was assessed through confusion matrix analysis against expert-assigned reference labels. Findings indicate measurable improvements in documentation efficiency, risk visibility, and pattern detection. The system achieved an overall AI classification accuracy of 89% and a System Usability Scale score of 82.5, placing it in the Excellent range. These results support the argument that intelligent, locally-designed monitoring systems can meaningfully strengthen data-driven decision-making at the grassroots governance level — and that such systems are both technically achievable and practically valuable in the Philippine barangay context.

**Keywords:** AI-enhanced decision support system, barangay incident monitoring, community risk assessment, digital governance, rule-based classification

## I. INTRODUCTION

Among the most persistent challenges facing barangay governance in the Philippines is the absence of systematic tools for incident monitoring and risk assessment. Despite occupying the most proximate position to community life within the local government structure, barangays continue to rely on manual documentation methods — handwritten logbooks, physical filing systems, and paper-based report forms — that offer no capacity for analytical processing, trend identification, or proactive risk management. The cumulative effect of this limitation is significant: local officials are routinely unable to determine whether particular areas are experiencing increasing incidents, whether specific incident types follow predictable temporal patterns, or where safety resources are most urgently needed.

This study proceeds from the observation that the data barangay officials need already exists — it is simply trapped in formats that preclude analysis. Every recorded theft, dispute, or vehicular accident represents a data point that could, given the right tools, contribute to a clearer picture of community safety conditions. What has been missing is a system designed specifically for the operational context of Philippine barangay governance: one capable of processing incident records, identifying patterns, assessing risk levels, and delivering insights in a form that officials without technical backgrounds can understand and act upon.

The scholarly literature on digital governance, decision support systems, and data analytics provides a strong theoretical basis for this kind of intervention. Janssen and Estevez [1] established that digital governance systems substantially improve evidence-based policymaking and service delivery in public institutions. Power [2] articulated how decision support systems bridge the gap between raw data accumulation and actionable knowledge generation. Chen, Chiang, and Storey [4] demonstrated that analytical platforms reveal correlational patterns within large datasets that manual review consistently fails to surface. Kitchin [5] showed that spatial and temporal data analysis produces qualitatively superior situational awareness in community risk management contexts. Gil-Garcia, Pardo, and Nam [3] further established that integrated data platforms enhance public administration effectiveness and inter-agency coordination at the governance level.

Despite the convergent evidence supporting data-driven governance, the barangay level has remained largely outside the scope of intelligent system deployment. Existing barangay information systems are predominantly oriented toward administrative functions — citizen record storage and certificate generation — with minimal integration of analytical or risk assessment capabilities. This implementation gap is the primary problem this research addresses.

Accordingly, this study undertook the design and development of a web-based AI-enhanced decision support system for barangay incident monitoring and community risk assessment, deployed at Barangay Washington as the primary implementation and evaluation site. The research contributes both a working system and an evidence base for the adoption of intelligent monitoring tools within grassroots Philippine governance structures.

## A. Objectives of the Study

This study aims to:

Examine the existing incident monitoring process at Barangay Washington and identify the substantive limitations of the current manual recordkeeping approach.

Analyze incident data to determine patterns, trends, and differentiated risk levels across geographic locations and temporal intervals.

Determine the effectiveness of the AI-enhanced web-based system in classifying and assessing community incident risks.

Assess the usability and functional performance of the proposed system as evaluated by barangay officials engaged in direct monitoring and administrative responsibilities.

Identify high-risk areas and recurring incident patterns to support the development of targeted, proactive community safety interventions.

Formulate evidence-based recommendations for barangay officials to improve decision-making processes and optimize resource allocation based on system-generated analytics.

## B. Conceptual Framework

This study is structured around an Input-Process-Output (IPO) framework that describes the logical flow of incident data through the system — from initial collection through analytical transformation to the generation of decision-support outputs.

1) Input:

The input component comprises incident data gathered from Barangay Washington, encompassing the following variables:

Incident type — including theft, physical assault, vehicular accidents, community disputes, and related categories

• Geographic location — recorded at the purok or sitio level

Date and time of occurrence

• Assessed severity level at the point of reporting

• Current resolution status — resolved, ongoing, or pending

Resident-submitted incident reports

2) Process:

The process component represents the analytical transformation of raw incident data through the following operations:

Data validation and centralized database storage

Incident categorization and systematic classification

• Frequency analysis and longitudinal trend monitoring

Rule-based AI risk scoring yielding Low, Medium, or High risk classifications per incident

• Pattern detection identifying recurring incident types and high-frequency locations

• Dashboard generation producing visual representations of trends, risk distributions, and summary statistics

Automated alert generation triggered by abnormal incident clusters or risk threshold breaches 3) Output:

The output component consists of analytical products and decision-support deliverables, including:

Risk level reports disaggregated by incident type and geographic location

Identification and mapping of high-risk areas within the barangay

• Automated smart alerts for emerging or statistically anomalous incidents

• Visual dashboards presenting trend data, incident patterns, and summary statistics

• Safety recommendations and actionable insights for barangay officials

Statistical and analytical reports to support planning and resource allocation

## II. REVIEW OF RELATED LITERATURE AND STUDIES

## A. Review of Related Studies

The intersection of digital technology and public governance has attracted substantial scholarly attention over the past two decades. A convergent body of literature across public administration, information systems, and urban governance establishes that data-driven tools improve institutional performance, decision quality, and responsiveness to community needs — findings that carry direct implications for the deployment of intelligent monitoring systems at the barangay level.

Janssen and Estevez [1] examined the concept of lean digital government, arguing that technology mediated governance can substantially improve institutional efficiency without proportional increases in resource expenditure. Their empirical findings demonstrated that public agencies integrating information technologies into core administrative processes achieved measurable improvements in transparency, interagency collaboration, and evidence-based policymaking. For barangay officials who operate under persistent resource constraints, the efficiency gains documented by Janssen and Estevez represent not merely theoretical benefits but practical operational advantages with direct implications for community safety management.

Power [2] provided foundational theoretical grounding for understanding the role of decision support systems in organizational contexts. His analysis established that DSS tools derive their value precisely from their capacity to bridge the distance between raw data accumulation and actionable decision-making — converting large volumes of disparate information into structured, interpretable outputs that support strategic and operational choices. This conceptualization is particularly relevant to the barangay monitoring problem, where the fundamental challenge is not a shortage of incident data but the absence of tools capable of transforming that data into usable intelligence.

Chen, Chiang, and Storey [4] extended this foundation by examining the organizational impact of business intelligence and analytics platforms. Their research demonstrated that analytical systems do not merely accelerate processes that human reviewers could otherwise complete manually — they surface correlational patterns and temporal regularities that manual review systematically fails to identify. This finding has direct implications for incident monitoring, where subtle shifts in the frequency or location of specific incident types may signal emerging safety problems well before they become visible through casual observation.

Kitchin [5] contributed a spatial and temporal dimension to the governance analytics literature through his examination of real-time city data systems. His work demonstrated that monitoring platforms with robust spatial and temporal analysis capabilities provide qualitatively superior situational awareness — enabling authorities to identify not merely what is happening within a community, but where and when risk concentrations are developing. The practical significance of this distinction for barangay-level safety management is considerable: geographic and temporal pattern data transforms general awareness of community problems into the kind of precise, actionable knowledge that supports targeted resource deployment.

Gil-Garcia, Pardo, and Nam [3] investigated the operational characteristics of smart governance frameworks, finding that governments adopting integrated data platforms demonstrated enhanced capacity for inter-agency coordination, citizen engagement, and responsive public service delivery. Their research is particularly relevant to the barangay context, where effective safety management frequently requires coordination across multiple local stakeholders, including law enforcement, disaster risk reduction units, and community organizations.

Taken collectively, these studies establish a consistent theoretical and empirical basis for the value of data-driven tools in governance contexts — including, by extension, at the grassroots level. The challenge identified across this body of literature is not whether such tools produce benefits, but whether appropriate systems exist for the specific operational contexts in which they are needed.

## B. Research Gap

Notwithstanding the substantial body of research on digital governance, decision support systems, and data analytics, the barangay level in the Philippines has received minimal attention as a deployment and evaluation context. The preponderance of existing studies focuses on national-level agencies, municipal governments, or large organizational entities, with grassroots governance structures remaining largely outside the scope of empirical investigation.

The barangay information systems that do exist in practice are predominantly oriented toward administrative recordkeeping functions — storing resident demographic information, generating certificates and clearances, and managing basic transactional records. These systems, while operationally useful for their intended purposes, lack the analytical capabilities that would enable risk assessment, pattern detection, or proactive safety planning. The result is that barangay officials possess repositories of potentially valuable incident data with no means of extracting meaningful intelligence from them.

A further dimension of this gap concerns the specificity of the deployment context. Solutions designed for large municipal or national systems cannot be assumed to transfer effectively to the barangay environment, which differs in scale, available technical infrastructure, user profile, and operational requirements. What the literature has not produced is a system designed from the ground up for these specific conditions — one that is lightweight enough for modest infrastructure, accessible to non-technical users, and analytically capable enough to deliver the insights that barangay safety management actually requires. This study represents a direct response to that identified gap.

## III. RESEARCH METHODOLOGY

## A. Research Design

This study employed a developmental research design, which is the methodologically appropriate approach for investigations centered on the systematic creation, implementation, and evaluation of a technological product or system. The developmental framework supported an iterative process of system design, construction, testing, and refinement, with each cycle informed by functional requirements and structured feedback from intended users.

Complementing the developmental approach, descriptive quantitative methods were applied to analyze incident patterns generated during system testing, evaluate the classification accuracy of the AI risk scoring engine, and interpret usability metrics gathered through standardized instrument administration.

## B. Research Locale

The study was situated at Barangay Washington, a barangay within the Philippines selected as the primary implementation and evaluation site on the basis of an identified operational need for improved incident monitoring and reporting infrastructure. The selection of an active, functioning barangay as the research locale was methodologically deliberate: deployment in a genuine operational environment ensured that system performance could be assessed under authentic conditions, with actual end users performing representative tasks, rather than within a setting that might inadequately reflect the practical realities of barangay governance.

## C. Respondents and Participants

The study engaged barangay officials, administrative secretaries, and public safety officers of Barangay Washington as primary respondents for usability and functionality evaluation. Participant selection employed purposive sampling, with inclusion criteria specifying active involvement in incident documentation, community monitoring, or administrative decision-making processes. This sampling approach was adopted to ensure that evaluation feedback reflected the authentic operational requirements and practical expectations of the system's intended user population, rather than the assessments of individuals without relevant governance experience.

## D. System Architecture and Technical Specifications

The system architecture follows a client-server model in which users interact with the platform through a standard web browser interface while application logic and database operations are hosted on a centralized server. The Model-View-Controller (MVC) design pattern was adopted to maintain clean separation among data management, presentation, and control logic layers, thereby supporting system maintainability and facilitating future scalability.

The technical implementation stack is detailed in Table I below.

TABLE I SYSTEM TECHNICAL STACK AND COMPONENT FUNCTIONS

| Component | Technology / Tool | Purpose |
| --- | --- | --- |
| Front-end Interface | HTML5, CSS3, Vanilla JavaScript | User interface structure, styling, and interactivity |
| Dashboard &amp; Maps | Chart.js &amp; Leaflet.js | Data visualization and location-based incident mapping |
| Back-end Framework | Python Django | Server-side application logic and request processing |
| Database | SQLite | Relational incident data storage and retrieval |
| Development Environment | Visual Studio Code | Primary coding and debugging environment |
| Version Control | Git | Codebase management and development history |
| AI Component | Rule-based classification engine (Gemini-assisted) | Automated risk scoring: Low, Medium, High |

The AI-enhanced functionality operates through a rule-based classification engine that evaluates each incident record against a set of predefined weighted criteria: the nature of the incident type, historical frequency of comparable incidents, recurrence of the reporting location, and the temporal distribution of incidents by time of day. These weighted inputs combine to produce an automated risk classification of Low, Medium, or High for each recorded incident. Classification rules were developed through a process of iterative consultation with barangay officials, ensuring that the scoring logic reflects both the empirical literature on incident risk factors and the practical knowledge of personnel with direct community governance experience.

## E. Data Collection Procedure

Data collection was structured across two sequential phases. The first phase comprised preliminary consultations with barangay officials aimed at establishing a thorough understanding of existing incident monitoring workflows, identifying operational pain points within current manual processes, and defining the incident categories most relevant to community safety planning. This phase employed structured interviews and direct observational sessions to capture the practical dimensions of day-to-day recordkeeping.

The second phase was initiated following system development and deployment at Barangay Washington. Participant respondents were invited to engage with the system by completing a structured set of representative operational tasks, including incident registration, dashboard navigation, report generation, and risk alert review. Upon task completion, participants completed the System Usability Scale questionnaire to provide standardized quantitative ratings across key usability dimensions. Concurrently, incident records were entered into the system to evaluate the operational accuracy of the AI risk classification engine.

## F. Data Analysis Procedure

Quantitative data collected through usability evaluation were analyzed using descriptive statistical methods, encompassing mean score computation, standard deviation calculation, and percentage distribution analysis, to produce an aggregated characterization of participant perceptions across usability dimensions. System Usability Scale scores were interpreted according to the established SUS benchmark scale, wherein scores of 68 and above are classified as above-average usability.

AI classification engine performance was evaluated through comparison of system-generated risk levels against expert-assigned reference classifications provided by experienced barangay officials. A confusion matrix analysis was applied to derive precision, recall, and F1-score metrics for each risk category. Incident data were additionally subjected to frequency distribution and trend analysis procedures to validate the accuracy and interpretive value of analytical dashboard outputs.

## G. Development Methodology

System development was guided by the Agile Software Development Methodology, an iterative framework that prioritizes adaptive planning, continuous user collaboration, and incremental refinement throughout the development cycle. Agile was selected in preference to linear methodologies such as Waterfall due to its demonstrated effectiveness in accommodating evolving requirements — a particularly important consideration when working with government end-users whose operational needs are best articulated in response to functional prototypes rather than abstract requirement specifications.

<u>www.oajmr.com</u>

Drake Joshua J. Besario et al, Open Access Journal of Multidisciplinary Research (OAJMR), Vol.2, Issue.6, June 2026, pg. 12-23 (An Open Accessible, Multidisciplinary, Fully Refereed and Peer Reviewed Journal)

ISSN (Online): 2320-924X DOI: 10.47760/OAJMR

TABLE II AGILE DEVELOPMENT PHASES

| Phase | Activities | Output |
| --- | --- | --- |
| 1. Requirement Analysis | Structured consultations with barangay officials; identification of functional and non-functional system requirements | System requirements document |
| 2. System Design | Database schema design; UI wireframing; system architecture specification | Design specifications document |
| 3. Development | Front-end and back-end coding; database configuration; AI engine implementation | Functional system prototype |
| 4. Testing | Functionality testing; usability evaluation; AI classification accuracy assessment | Test results and defect reports |
| 5. Deployment | System installation at Barangay Washington; end-user orientation and training | Deployed operational system |
| 6. Maintenance | Post-deployment performance monitoring; defect resolution; iterative improvements informed by user feedback | Refined system versions |

## H. Ethical Considerations

The study was conducted in strict accordance with established ethical standards governing research involving human participants and sensitive community data. Informed consent was obtained from all respondents prior to participation in usability evaluation activities, and participant anonymity and confidentiality were maintained throughout all phases of the research. System access is restricted to authorized barangay personnel through role-based authentication controls, ensuring that sensitive incident data remains accessible only to individuals with legitimate operational need. Data stored within the system is protected through password encryption and database access restrictions, and the study as a whole was conducted in compliance with applicable data privacy legislation in the Philippines.

## IV. RESULTS AND DISCUSSION

This chapter presents the findings of the study organized in relation to each stated research objective. The results are derived from system development outcomes, usability evaluation data collected from barangay respondents, and AI classification performance metrics generated during testing and initial deployment at Barangay Washington. Where full deployment data remained pending at the time of evaluation, representative test data were employed and are explicitly identified as such in the relevant sections.

## A. Current Incident Monitoring Process and Identified Limitations

Prior to system design and development, a structured requirement analysis was conducted through consultations with barangay officials at Barangay Washington. These consultations established that incident monitoring at the barangay is conducted entirely through paper-based mechanisms — handwritten logbooks maintained by the barangay secretary, supplemented by physical filing systems with no capacity for electronic retrieval, pattern analysis, or automated reporting.

Officials identified several substantive operational limitations inherent to this approach. The absence of any analytical layer means that recurring incident patterns — whether by location, incident type, or time of occurrence — remain invisible to decision-makers unless identified through intensive manual review. No standardized risk classification mechanism exists to distinguish incident severity or prioritize response. Report generation is a time-intensive manual process consuming hours of staff time. Physical records are inherently vulnerable to loss through environmental damage, misplacement, or deterioration over time. These limitations, taken collectively, represent significant structural barriers to evidence-based community safety management.

TABLE III COMPARISON OF MANUAL PROCESS VS. PROPOSED SYSTEM CAPABILITIES

| Limitation (Manual Process) | Proposed System Capability |
| --- | --- |
| Paper-based records vulnerable to loss and damage | Centralized digital database with redundant backup |
| No capacity for pattern or trend analysis | Automated frequency and trend analysis dashboard |
| No mechanism for incident risk classification | Rule-based AI risk scoring: Low / Medium / High |
| Manual report generation requires several hours | Automated report generation completed in seconds |
| No alert system for risk concentration or spikes | Automated smart alerts for anomalous incident clusters |
| Data accessible only at the physical barangay hall | Web-based access from any authorized device |

## B. Incident Pattern and Trend Analysis

To evaluate the analytical capabilities of the developed system, incident records representing six months of community activity were entered into the platform during the testing phase. The analytical dashboard processed this dataset and produced frequency distribution tables, monthly trend line visualizations, and location-based heat maps — outputs that enabled evaluation respondents to extract meaningful pattern information from the data within minutes of system access.

Among the most significant findings from this analytical exercise was the disproportionate concentration of incidents in two categories: theft, accounting for 28.0% of all records, and community disputes, comprising 22.0%. Temporal analysis revealed a consistent clustering of incidents across multiple categories between 6:00 PM and 10:00 PM. The practical significance of this temporal pattern for patrol scheduling and community safety planning is considerable — it provides barangay officials with evidence-based guidance on resource deployment timing that the existing manual system is entirely unable to produce. Table IV presents the full distribution of test incident records by type and assigned risk level.

TABLE IV INCIDENT DISTRIBUTION BY TYPE (TEST DATA)

| Incident Type | Frequency (n) | Percentage (%) | Assigned Risk Level |
| --- | --- | --- | --- |
| Theft | 28 | 28.0% | High |
| Community Dispute | 22 | 22.0% | Medium |
| Vehicular Accident | 18 | 18.0% | Medium |
| Physical Assault | 12 | 12.0% | High |
| Fire Incident | 8 | 8.0% | High |
| Noise Complaint | 7 | 7.0% | Low |
| Vandalism | 5 | 5.0% | Low |
| Total | 100 | 100% | - |

These findings are consistent with the framework articulated by Kitchin [5], who established that temporal and spatial pattern detection represents a core operational benefit of data-driven community monitoring systems — one that manual processes are structurally unable to replicate at any meaningful scale.

## C. AI Risk Classification Accuracy

Evaluation of the AI classification engine was conducted by comparing system-generated risk level assignments against reference classifications independently assigned by experienced barangay officials serving in the capacity of domain experts. A corpus of 100 test incident records was processed, and confusion matrix analysis was applied to derive precision, recall, and F1-score metrics for each risk category. Performance results are presented in Table V.

TABLE V AI RISK CLASSIFICATION PERFORMANCE METRICS (PRELIMINARY EVALUATION)

| Risk Category | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| Low Risk | 0.91 | 0.88 | 0.89 |
| Medium Risk | 0.84 | 0.86 | 0.85 |
| High Risk | 0.93 | 0.95 | 0.94 |
| Overall Accuracy | - | - | 0.89 |

Note: Values presented in Table V are derived from a preliminary evaluation using 100 test incident records. Field evaluation data from full operational deployment at Barangay Washington will be collected in a subsequent research phase.

The classification engine demonstrated consistent performance across all three risk categories, achieving an overall accuracy of 89%. High-risk incidents — principally theft and physical assault — yielded the strongest classification performance, with an F1-score of 0.94, attributable to the relatively unambiguous definitional boundaries governing these categories within the rule-based logic. Medium-risk incidents produced a somewhat lower F1-score of 0.85, reflecting the classificatory complexity inherent in distinguishing community disputes from vehicular accidents when both incident types exhibit overlapping severity indicators. Low-risk incidents achieved a reliable F1-score of 0.89. These findings are consistent with Power's [2] observation that rule-based DSS classification models perform robustly across well-defined categories while remaining subject to marginal ambiguity at category boundaries — an inherent characteristic that is most effectively addressed through systematic rule refinement as operational data accumulates.

## D. System Usability Evaluation

Usability evaluation was conducted with eight barangay officials and administrative secretaries who participated as system evaluators. Participants were asked to complete a structured task sequence using the deployed system — encompassing incident registration, dashboard navigation, report generation, and risk alert review — before completing the System Usability Scale questionnaire. Aggregate results of the usability evaluation are presented in Table VI.

TABLE VI SYSTEM USABILITY EVALUATION RESULTS (PRELIMINARY EVALUATION, N=8)

| SUS Criterion | Mean Score (1-5) | Verbal Interpretation |
| --- | --- | --- |
| Ease of Use | 4.50 | Strongly Agree |
| System Functionality | 4.38 | Strongly Agree |
| Dashboard Clarity | 4.63 | Strongly Agree |
| Alert Usefulness | 4.75 | Strongly Agree |
| Overall Satisfaction | 4.50 | Strongly Agree |
| Overall SUS Score (0-100) | 82.5 | Excellent |

Note: Values presented in Table VI are derived from a preliminary usability evaluation conducted with eight representative system users prior to full deployment at Barangay Washington.

The overall SUS score of 82.5 situates the system within the Excellent usability classification, above the 80.3 threshold established by the standard SUS benchmark. Examination of individual criterion scores reveals meaningful variation in user perceptions across system dimensions. Alert Usefulness received the highest mean rating (4.75), indicating that respondents assessed the automated risk alert functionality as particularly relevant to their operational responsibilities. Dashboard Clarity received the second-highest rating (4.63), affirming that the visualization architecture effectively communicates incident trends and risk distributions to users without specialized technical backgrounds. System Functionality, while representing the lowest-rated criterion at 4.38, nonetheless reflects a Strongly Agree assessment, indicating that no dimension of system performance was perceived as deficient. These findings align with the conclusions of Janssen and Estevez [1], who established that digital governance systems designed in close alignment with user operational contexts produce measurable improvements in both administrative responsiveness and the quality of evidencebased decision-making.

## E. High-Risk Area Identification

The location-based risk analysis module demonstrated functional capacity to identify high-risk geographic concentrations within Barangay Washington during the testing phase. Analysis of the incident dataset revealed that specific puroks exhibited disproportionate concentrations of high-severity incident types — particularly theft, physical assault, and fire incidents — sufficient to trigger high-risk zone classifications within the system's geographic profiling logic. Barangay officials who reviewed the resulting risk maps noted that the identified zones were consistent with their own operational understanding of community safety conditions, providing preliminary validation of the geographic analysis methodology.

The automated alert mechanism performed as designed during testing, generating notifications when incident frequency within a defined geographic area exceeded a threshold of three high-severity incidents within a 30-day window. This threshold was established through consultation with barangay officials and reflects the operational boundary at which concentrated incidents are deemed to warrant targeted intervention. The significance of this capability for community safety management lies in its transformation of monitoring from a reactive to an anticipatory function: rather than detecting problem areas only after substantial incident accumulation, the system enables officials to respond while patterns are still in early formation. This operational characteristic corresponds to the findings of Gil-Garcia, Pardo, and Nam [3], who demonstrated that real-time monitoring platforms improve inter-agency coordination and enable more rapid, evidence-directed safety responses.

## V. CONCLUSION AND RECOMMENDATIONS

## A. Summary of Findings

This study undertook the design, development, and evaluation of an AI-Enhanced Decision Support System for Barangay Incident Monitoring and Community Risk Assessment, with Barangay Washington serving as the primary implementation and evaluation site. The findings across each research objective are summarized below.

Requirement analysis confirmed that the existing incident monitoring process at Barangay Washington is conducted exclusively through manual, paper-based mechanisms, resulting in documented limitations including the absence of pattern detection capability, lack of risk classification infrastructure, time-intensive manual reporting, and records vulnerable to irretrievable loss. The proposed system directly addressed each of these limitations through centralized digital data management and AI-enhanced analytics.

The analytical dashboard demonstrated functional capacity to generate frequency distributions, temporal trend lines, and location-based risk maps from incident data. Theft and community disputes emerged as the most prevalent incident categories, with peak incident occurrence concentrated in the evening hours — findings that provide actionable, evidence-based guidance for community safety planning that the manual system is structurally incapable of producing.

The rule-based AI classification engine achieved an overall accuracy of 89% across Low, Medium, and High risk categories, with particularly strong performance in the classification of high-risk incidents. Usability evaluation returned an overall SUS score of 82.5 — classified as Excellent by the standard benchmark — with the automated alert system receiving the highest individual criterion rating among all features assessed. The location-based risk analysis module successfully identified high-risk puroks and generated threshold-based alerts during testing, demonstrating its capacity to support anticipatory rather than purely reactive community safety management.

## B. Conclusions

The central conclusion of this study is that the development and deployment of an AI-enhanced decision support system at the barangay level is both technically feasible and operationally valuable. The system developed in this research successfully transforms raw incident data into structured analytical intelligence, enabling barangay officials to transition from reactive incident response toward proactive community safety management. The integration of rule-based AI risk classification, analytical dashboards, and automated alert mechanisms produces a scalable, user-accessible solution appropriately calibrated to the operational context of grassroots governance in the Philippines.

The research further establishes that the structural limitations characteristic of traditional barangay monitoring systems — the absence of trend analysis, risk classification, and efficient reporting — are amenable to systematic resolution through the application of digital governance tools and AI-enhanced analytics. These findings extend and validate the broader literature on smart governance and decision support systems, confirming that data-driven approaches improve decision-making efficiency and resource allocation effectiveness even in resource-constrained, small-scale governance environments.

From a policy perspective, the study provides empirical support for the proposition that intelligent monitoring infrastructure at the barangay level is a practical and warranted investment, rather than a technological aspiration beyond the operational reach of grassroots local government. The system demonstrated in this research offers a replicable model for broader adoption across Philippine barangays, particularly those operating in high-risk community environments.

## C. Recommendations

1) For Barangay Washington Officials and Implementers:

Barangay Washington should transition fully to the deployed system as the primary incident recording and monitoring platform, retiring paper-based logbooks to ensure data consistency and enable the longitudinal trend analysis upon which the system's analytical capabilities depend.

Sustained, structured training programs should be established for all system users — including barangay secretaries and public safety officers — to ensure proficiency in data entry protocols and maximize utilization of dashboard analytics and risk alert features.

A formal institutional protocol should be developed for responding to system-generated risk alerts, specifying assigned responsibilities, escalation procedures, and required response timelines for high-risk incident notifications.

## 2) For Future Researchers:

Subsequent studies should investigate the feasibility of integrating supervised machine learning classification algorithms — including decision trees, random forest models, or gradient boosting approaches — as replacements for the current rule-based engine, with the aim of improving classification accuracy in ambiguous, boundary-condition incident cases.

Research should examine the scalability of the system architecture for simultaneous deployment across multiple barangays within a centralized multi-tenant framework, enabling municipal-level aggregation and comparative analysis of incident data.

Longitudinal research designs tracking incident frequency, resolution rates, and community safety outcomes before and after system deployment would yield substantially stronger evidence of the system's real-world governance impact than single-point evaluations.

Future iterations of the system should explore the incorporation of natural language processing capabilities to enable resident-facing incident submission through mobile interfaces or conversational chatbot channels, expanding the data collection base and strengthening community participation in safety monitoring. 3) For Policy and Institutional Level:

The Department of the Interior and Local Government (DILG) should consider the development of policy frameworks and dedicated funding mechanisms to support the adoption of AI-enhanced incident monitoring systems across barangays in high-risk communities throughout the Philippines.

Municipal governments should pursue institutional partnerships with academic institutions and technology development organizations to co-develop and sustain barangay-level digital governance systems, with sustainability planning and technical support structures embedded from the outset of deployment.

## D. Limitations of the Study

Several limitations of this study warrant acknowledgment. The evaluation conducted was based primarily on initial usability testing and representative test data rather than extended real-world operational data, meaning that system performance characteristics may evolve — positively or otherwise — as the platform accumulates live incident data across diverse operational conditions. The AI classification engine operates on rule-based logic rather than adaptive machine learning, which constrains its capacity to adjust autonomously to shifting incident patterns over time without corresponding updates to the underlying rule set. Additionally, this study did not assess system performance under conditions of low network bandwidth or intermittent connectivity, which may constitute a relevant practical consideration for barangay deployments in geographically remote or infrastructurally limited areas. These limitations identify productive directions for future research and iterative system development.

## ACKNOWLEDGEMENT

The authors would like to express their sincere gratitude to the officials and staff of Barangay Washington for their cooperation and participation in the evaluation phase of this research. This study was conducted as part of the MIT 224 – Technology and Project Management course.

## REFERENCES

1. M. Janssen and E. Estevez, ―Lean government and platform-based governance: Doing more with less,‖ Government Information Quarterly, vol. 30, pp. S1–S8, 2013.

2. D. J. Power, Decision Support Systems: Concepts and Resources for Managers. Westport, CT: Quorum Books, 2002.

3. J. R. Gil-Garcia, T. A. Pardo, and T. Nam, ―What makes a city smart? Identifying core components and proposing an integrative and comprehensive conceptualization,‖ Information Polity, vol. 20, no. 1, pp. 61–87, 2016.

4. H. Chen, R. H. L. Chiang, and V. C. Storey, ―Business intelligence and analytics: From big data to big impact,‖ MIS Quarterly, vol. 36, no. 4, pp. 1165–1188, 2012.

5. R. Kitchin, ―The real-time city? Big data and smart urbanism,‖ GeoJournal, vol. 79, no. 1, pp. 1–14, 2014.