#import "figure-style.typ": *
#show: figure-style

#ipo-diagram(
  height: 4in,
  [
    - Taglish disaster tweets
    - Pretrained encoder
    - 13 multi-task optimizers
  ],
  [
    + Design and Artifact Creation \
      Build and label the dataset \
      (Research Question 2)
    + Baseline Analysis \
      Train two single-task models \
      (Research Question 1)
    + Empirical Benchmarking \
      Train one multi-task model per optimizer \
      (Research Question 3)
    + Comparative Validation \
      Test which optimizers differ significantly from equal weighting \
      Relate negative transfer to training cost \
      (Research Question 4)
  ],
  [
    - Labeled dataset and its inter-annotator agreement
    - Baseline scores for each task
    - Training time and peak memory per optimizer
    - Negative transfer per optimizer
    - Optimizers that differ significantly from equal weighting
    - Relation between negative transfer and training cost
  ],
)
