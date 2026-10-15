#import "figure-style.typ": *
#show: figure-style

#let part(pos, title, body, width: 1.38in, height: auto) = node(
  pos,
  align(left)[
    #align(center, strong(title))
    #body
  ],
  width: width,
  height: height,
  ..box-style,
)

#let group(title, ..positions) = node(
  enclose: positions.pos(),
  align(top + left, strong(title)),
  ..group-style,
)

#diagram(
  spacing: (0.43in, 0.3in),
  edge-stroke: 1.2pt,

  part((1, 0), width: 1.75in)[Multi-Task Model][
    - Shared encoder, SEA-LION-ModernBERT-300M
    - Classification head and NER head
    - One of the 13 multi-task optimizers
  ],

  part((0, 1), height: 2.6in)[Data Preparation][
    - Collect tweets through the X API
    - Clean the tweets and keep Taglish tweets
    - Label both tasks with two annotators
    - Split into training, validation, and test sets
  ],
  part((1, 1), height: 2.6in)[Training and Measurement][
    - Choose the settings of each model with seed 0
    - Train 15 models with 3 seeds (45 main runs)
    - Record the training time and peak memory
    - Save the best epoch of each run
  ],
  part((2, 1), height: 2.6in)[Evaluation and Analysis][
    - Score the saved models on the test set
    - Compute F1-scores and negative transfer
    - Run the two-way ANOVA and Tukey's HSD test
    - Build the Pareto frontier
  ],

  group([Part 1: Benchmarking System], (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),

  part((0, 4), height: 1.25in)[Interface][
    - React, in a web browser
    - Shows posts, filters, summaries, and forms
  ],
  part((1, 4), height: 1.25in)[Server][
    - FastAPI in Python
    - Runs the trained model with PyTorch
  ],
  part((2, 4), height: 1.25in)[Database][
    - PostgreSQL
    - Stores posts, labels, and statuses
  ],

  group([Part 2: Prototype Web Application], (0, 3), (0, 4), (1, 4), (2, 4)),

  edge((1, 0), (1, 1), "-|>", label-side: left)[model to train],
  edge((0, 1), (1, 1), "-|>")[labeled \ dataset],
  edge((1, 1), (2, 1), "-|>")[saved \ models \ and logs],
  edge((1, 1), (1, 4), "-|>", label-side: left)[trained multi-task model],
  edge((0, 4), (1, 4), "<|-|>")[requests],
  edge((0, 4), (1, 4), "<|-|>", label-side: right)[results],
  edge((1, 4), (2, 4), "<|-|>")[queries],
  edge((1, 4), (2, 4), "<|-|>", label-side: right)[records],
)
