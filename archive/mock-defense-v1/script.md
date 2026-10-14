# Mock Defense Script

## Slide 2. Background and Problem Statement

**[Real-world context]**

The Philippines has the highest disaster risk among 193 countries. During these events, Filipinos turn to social media to report damage and ask for help. Many of these posts are in Taglish. Social media accounts equal about 82 percent of the population, so these posts quickly outnumber what responders can read one by one.

**[Current system bottlenecks]**

Two language tasks can help sort these posts. Humanitarian classification identifies the type of report, such as a request for supplies. Named entity recognition finds details such as place names. Earlier Philippine studies trained a separate model for each task. One system in Urdaneta City covers both tasks, but it sends the posts to OpenAI, a private company. Posts that ask for help often contain personal information, such as names, addresses, and phone numbers. Under the Data Privacy Act of 2012, the city is responsible for this information in any case. Sending it to a private company adds another party that handles the data, outside the city's control.

**[Computational problem]**

A local government unit could avoid this added risk by running one model for both tasks on its own computers. One model uses less storage and runs faster than two separate models. However, a shared model can perform worse on one task than a model trained on that task alone. This problem is called negative transfer. Classification is especially at risk. In a tweet of 30 words, named entity recognition makes 30 decisions, while classification makes only one. So during training, named entity recognition has a much bigger effect on the model, and classification can become less accurate.

Training methods called multi-task optimizers try to balance the two tasks. The simplest one, equal weighting, gives both tasks the same importance throughout training. Studies disagree on whether any other optimizer performs better than it. Many of the other optimizers also need more memory and training time. No Taglish dataset labels the same tweets for both tasks, so these optimizers have never been compared on Taglish disaster tweets.

**[Main research question]**

This leads to the main research question. Do any of the other 12 optimizers cause significantly more or less negative transfer than equal weighting?

## Slide 3. Objective and Scope

**[General objective]**

The general objective of the study is to compare 13 multi-task optimizers in terms of negative transfer and training cost on Taglish disaster tweets. The study also examines how negative transfer relates to training cost.

**[Specific objectives]**

Four specific objectives support this goal.

First, the study measures how well single-task models perform each task on Taglish disaster tweets. These scores serve as the baseline.

Second, it builds a Taglish disaster tweet dataset labeled for both tasks and measures how consistently the annotators agree on the labels.

Third, it measures the training time and peak memory of each of the 13 optimizers.

Fourth, it tests whether any of the other 12 optimizers differ significantly from equal weighting in negative transfer.

**[Scope]**

The study covers Taglish tweets about tropical cyclones that crossed the Philippines from 2020 to 2026. It trains 15 model configurations. Two single-task models provide the baseline scores, and 13 multi-task models, one for each optimizer, form the main comparison. All models use the same pretrained encoder.

**[Limits and exclusions]**

Several boundaries keep the comparison fair. The dataset uses only tweets from X, so the results can be compared with earlier disaster datasets. Other Philippine languages, such as Cebuano, and other hazards, such as earthquakes, are excluded. The study uses only the text of each tweet. All training runs on a single laptop, so the training cost of every optimizer is measured under the same conditions. Finally, the study does not compare the model with large language models, and it does not test the model with responders during actual disasters.

## Slide 4. Literature and Baseline Algorithms

**[Baseline methods]**

Three earlier studies are closest to this study.

The first is the study of Ali et al. from 2026. It uses one model for both humanitarian classification and named entity recognition on English disaster tweets. Its classification score dropped during training, and one optimizer, uncertainty weighting, prevented this drop. However, Ali et al. compared only two methods and did not measure training cost.

The second is the study of Kurin et al. from 2022. It compared equal weighting, which is the baseline of this study, with specialized optimizers. Once common techniques for stable training were added, equal weighting performed as well as or better than the others. However, Kurin et al. used only image and robot tasks, with no text. A more recent study by Gama and Grassi found that equal weighting is enough when the two tasks are similar. The two tasks in this study are not similar, because one makes a prediction for every word and the other makes one for the whole tweet.

The third is the Urdaneta City system mentioned earlier. Besides relying on OpenAI, it was tested only on simulated comments.

**[Comparison table]**

The table shows that each study covers only part of the problem. One uses the same tasks but only English. One compares optimizers but uses no text. One uses local languages but depends on a private company.

**[The gap]**

No earlier study combines Taglish text, both tasks on the same tweets, several optimizers, and measured training cost. This study fills that gap.

## Slide 5. Expected Scientific Contribution

**[Theoretical value]**

This study adds to an open debate in computer science. It tests this debate in a new setting, with a new language, a new domain, and a task pair that mixes word-level and whole-tweet predictions. The results also help explain why an optimizer works. It may solve a real conflict between the two tasks, or it may simply keep the model from memorizing its training data. This study is also the first among those reviewed to relate the training cost of each optimizer to its negative transfer.

**[Practical benefits]**

For local government units and their disaster offices, this study shows how much performance one model loses on each task under each optimizer. It also shows how much time and memory each optimizer needs on a laptop. This helps an office choose a model that runs on its own computers. Developers of multi-task models gain the same guidance for limited hardware. Communities affected by typhoons may benefit indirectly, since responders could find urgent requests faster.

**[Deliverables]**

This study produces five deliverables. These are a Taglish disaster tweet dataset labeled for both tasks, its annotation guidelines, benchmark results for all 15 models, the benchmarking code, and a prototype web application. The code is shared publicly on GitHub.

## Slide 6. Software Artifact and UI/UX Design

**[Architecture]**

The software has two parts, shown in this diagram.

The first part is the benchmarking system, which is the testbed of this study. Data preparation collects and labels the tweets. The labeled dataset then goes to training and measurement. Here, the multi-task model is trained with each of the 13 optimizers, and the system records the training time and peak memory of every run. Evaluation and analysis then scores and compares the models.

The second part is a prototype web application. It takes one trained multi-task model from the benchmark and shows how a disaster office could use it to sort posts.

**[Tech stack]**

All training runs on one laptop with an 8 GB graphics card. The benchmark uses Python, PyTorch, Hugging Face Transformers, and the LibMTL library, which provides ten of the 13 optimizers. The prototype uses React for the interface, FastAPI for the server, and PostgreSQL for the database. Everything runs on the same computer, so no post is sent to an outside service.

**[UI/UX]**

The interface lets office staff import posts, see each post with its category and highlighted entities, correct wrong labels, filter posts by category or location, and export a summary report.

## Slide 7. Testing Method and Metrics

**[Test dataset]**

The models are tested on a Taglish dataset built in this study. Tweets about tropical cyclones from 2020 to 2026 are collected through the X API, and each selected cyclone contributes an equal share. Two annotators label every tweet for both tasks. The training set has at least 2,801 tweets. The dataset is split into 72 percent for training, 14 percent for choosing settings, and 14 percent for testing. The test set is used only once, for the final scores.

**[Label quality]**

Agreement between the two annotators is checked after every batch. Part of this check uses the F1-score, a number from 0 to 1. It is high only when most answers are correct and most of the correct answers are found. The targets are a Cohen's kappa of at least .61 for the categories, which counts as substantial agreement, and an F1-score of at least .74 for the entities, the level reached by an earlier Tagalog dataset. When agreement falls below a target, the guidelines are revised.

**[Accuracy]**

Humanitarian classification is scored with macro-F1, which counts rare categories as much as common ones. Named entity recognition is scored with span-level F1. An entity counts as correct only when the model marks the whole name, such as "Barangay San Jose," and gives it the right type. Negative transfer is the F1-score of the multi-task model minus that of the single-task model.

**[Speed and memory]**

A timer records every training step, and PyTorch records the peak memory of each run. Each optimizer's cost is also reported as a percentage above equal weighting.

**[Statistical tests]**

Each of the 15 models is trained three times. Each time uses a different random seed, which is a number that fixes the random choices made during training. This gives 45 main runs and shows how much the results change by chance.

The results are then compared in two steps. First, a two-way ANOVA checks whether the choice of optimizer makes any difference in negative transfer. If it does, Tukey's HSD test compares each optimizer with equal weighting.

Finally, a Pareto frontier compares the optimizers on negative transfer and training cost together. It shows which optimizers no other optimizer beats on both.

## Slide 8. Development and Testing Schedule

**[Gantt chart]**

This chart shows the plan from November 2026 to April 2027. In November, tweets are collected, cleaned, and selected. Labeling starts right after and continues until January, because it is the longest task.

While the tweets are being labeled, the benchmarking code and its tests are written. Training starts only after every test passes. In February, a short pilot run and the search for the best settings come first, followed by the 45 main runs. The prototype is built during the same months and tested once a trained model is ready. In March, the results are analyzed and written up as Chapters 4 and 5.

**[Proof of completion]**

The final defense is in the first week of April 2027. Every task ends before that date, and the last weeks of March are kept free as a buffer in case of delays.

**[Feasibility]**

The plan is feasible for three reasons. First, the code and the prototype are built during labeling instead of waiting for it. Second, the training load is known in advance. The settings search needs 90 runs and the main comparison needs 45, so the laptop runs 135 full training runs in total. Third, each task starts only when the work it depends on is ready, so no task waits on unfinished input.

## Slide 9. Immediate Next Steps

**[Immediately after approval]**

Right after the proposal is approved, the group applies for access to the X API, which is the official way to collect tweets. The first check is whether the API can search tweets by keyword and date as far back as 2020. The group then collects tweets for every tropical cyclone on the PAGASA list from 2020 to 2026, covering the days each cyclone was inside the Philippine Area of Responsibility.

**[Next two to four weeks]**

Within two to four weeks, the tweets are cleaned and checked for Taglish. The cyclones with at least 400 Taglish tweets are then selected. At the same time, the labeling guidelines are finalized, GLiNER drafts of the entities are loaded into doccano, and the annotators label a practice batch.

**[Milestones before full data gathering]**

Before full labeling begins, two milestones must be met. First, agreement between the annotators on the practice batch is measured against the targets, and the guidelines are revised if agreement falls short. Second, the number of tweets to label is fixed, with at least 2,801 in the training set. Batch labeling then begins.

## Slide 10. Technical Risks and Backup Plans

**[Coding bugs and hardware bottlenecks]**

The first group of risks is technical. To catch bugs, each part of the code is tested on small examples with answers worked out by hand. These tests also check that each optimizer behaves as its original paper describes. The whole system is then tested on a small part of the data, and the main training starts only after every test passes.

During training, each task sends the model a signal called a gradient, which tells the model how to change. Four optimizers keep a separate gradient for each task, which needs more memory. So a short pilot run first checks that every model fits in the 8 gigabytes of the graphics card. If the graphics card slows down from heat, the temperature records show it, and that run is repeated. If the laptop breaks, all runs are redone on the replacement, so every optimizer is still measured on the same machine.

**[Risks during prototype testing]**

For the prototype, any use case that fails a test is fixed and tested again. The prototype is not tested with disaster office staff, so this is stated as a limitation of the study.

**[Contingency plans for delays]**

Three delays are possible. Searching tweets back to 2020 may need a higher level of X API access, so this is settled right after approval. If labeling is slow, the group labels closer to the minimum of 2,801 training tweets. If training runs long, the buffer weeks absorb the delay, and a second laptop handles the coding and prototype work while the main laptop trains.

This ends the presentation. The group welcomes the panel's questions.