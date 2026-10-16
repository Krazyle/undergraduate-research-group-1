#import "figure-style.typ": *
#show: figure-style

#ipo-diagram(
  height: 3.3in,
  widths: (1.6in, 2.5in, 1.6in),
  [
    - Taglish posts about tropical cyclones from 2020 to 2026
    - Pretrained encoder \
      (SEA-LION-ModernBERT-300M)
    - 13 multi-task optimizers
    - Single laptop for training
  ],
  [
    + *Design and Artifact Creation* \
      Build and label the dataset and measure inter-annotator agreement (RQ2)
    + *Baseline Analysis* \
      Train and score two single-task models (RQ1)
    + *Empirical Benchmarking* \
      Train one multi-task model per optimizer and measure training time and peak memory (RQ3)
    + *Comparative Validation* \
      Compute negative transfer, test each optimizer against equal weighting, relate negative transfer to training cost, and select the optimizer for the final model (RQ4)
  ],
  [
    A multi-task model for humanitarian classification and named entity recognition of Taglish typhoon posts, trained with the selected optimizer on a single laptop
  ],
)
