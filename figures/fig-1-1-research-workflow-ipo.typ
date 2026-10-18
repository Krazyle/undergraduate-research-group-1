#import "figure-style.typ": *
#show: figure-style

#ipo-diagram(
  height: 3.1in,
  widths: (1.6in, 2.5in, 1.6in),
  [
    - Taglish posts on X about tropical cyclones, 2020--2026
    - HumAID categories and HUMAID-NER entity types
    - Pretrained encoder \
      (SEA-LION-ModernBERT-300M)
    - 13 multi-task optimizers
    - Single laptop for training
  ],
  [
    + *Build the dataset* \
      Collect, clean, label, and split posts
    + *Train single-task models* \
      Measure baseline F1-scores (RQ1)
    + *Design the multi-task model* \
      Share one encoder between both tasks (RQ2)
    + *Benchmark 13 candidate models* \
      Compare negative transfer and training cost to select the optimizer (RQ3)
    + *Test the final model* \
      Compare it with the single-task models on the test posts (RQ4)
  ],
  [
    A multi-task model for humanitarian classification and named entity recognition of Taglish typhoon posts, trained with the selected optimizer
  ],
)
