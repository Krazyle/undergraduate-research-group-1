# Uniform Loss vs. Specialized Optimization: A Comparative Analysis in Multi-Task Learning

Gabriel S. Gama  Affiliation: University of São Paulo  Email: [gabriel_gama@usp.br](<mailto:>) Valdir Grassi Jr  Affiliation: University of São Paulo  Email: [vgrassi@usp.br](<mailto:vgrassi@usp.br>)

###### Abstract

Specialized Multi-Task Optimizers (SMTOs) balance task learning in Multi-Task Learning by addressing issues like conflicting gradients and differing gradient norms, which hinder equal-weighted task training. However, recent critiques suggest that equally weighted tasks can achieve competitive results compared to SMTOs, arguing that previous SMTO results were influenced by poor hyperparameter optimization and lack of regularization. In this work, we evaluate these claims through an extensive empirical evaluation of SMTOs, including some of the latest methods, on more complex multi-task problems to clarify this behavior. More specifically, we start our analysis by evaluating all SMTOs on a simple MNIST problem to identify the promising optimizers and assess them further on progressively more complex multi-task problems. Our findings indicate that SMTOs perform well compared to uniform loss and that fixed weights can achieve competitive performance compared to SMTOs. Furthermore, we demonstrate why uniform loss performs similarly to SMTOs in some instances. The source code is available at <https://github.com/Gabriel-SGama/UnitScal_vs_SMTOs>.

## 1 Introduction

When tackling complex real-world challenges in machine learning, one must contemplate the utility of Multi-Task Learning (MTL) [1]. Scenarios such as scene understanding, multilingual translation, and predicting chemical properties require multiple outputs, possibly making them promising candidates for MTL approaches due to their inherent inductive bias [2].

Classical methods approach this problem by utilizing one model for each task, but most MTL scenarios have the potential to share information across tasks. For example, autonomous driving relies on multiple pieces of information extracted from the same visual input, such as semantic segmentation, object detection, instance segmentation, and others. Therefore, sharing information across those tasks is desirable to increase performance.

To efficiently share feature information across tasks, various approaches have been proposed, including model architecture designs [3, 4, 5], task grouping strategies [6, 7], Meta-Learning techniques [8], and the focus of this work, Specialized Multi-Task Optimizers (SMTOs) [9, 10].

SMTOs are implemented together with a shared encoder architecture, where one common encoder outputs a feature space that is shared with the specific decoders used for each task [9, 10]. Consequently, this reduces the computational cost when compared to the individual models approach. This unified model can be optimized by minimizing the sum of the losses, known as uniform loss. However, the simultaneous learning of multiple tasks introduces challenges such as conflicting directions and significant gradient norm disparities [11, 12]. SMTOs aim to address these challenges by applying mathematical tools to jointly and equally optimize all objectives.

SMTOs can be classified into two types: gradient-based and loss-based. The former compute the gradients of each task and combine them to find a suitable optimization direction [9, 13, 14, 15, 11, 12, 10, 16]. This process adds complexity and computational cost to the learning problem as multiple backpropagation steps are needed. The loss-based methods only take into consideration the loss values [17, 18, 19, 20, 21], reaching computational times comparable to those of the uniform loss [22, 21].

One notable issue in the SMTO field is the lack of a standardized evaluation framework and procedure, leading to concerns about reproducibility [23]. Understandably, some works have begun questioning the efficacy of SMTOs [24, 23]. These critiques attribute reported gains in SMTO papers to sub-optimal hyperparameter tuning or insufficient regularization. They provide various experiments, including natural language processing (NLP), computer vision, and reinforcement learning, as well as a theoretical hypothesis on why SMTOs apply a regularization effect and a comparison to a properly optimized baseline.

In this context, we propose to address these concerns by hypothesizing that proper hyperparameter optimization is crucial for fair performance assessment of SMTOs. Nevertheless, we also emphasize the significance of considering the inherent complexity of the MTL balancing problem. Our contributions include:

  * •

A comprehensive evaluation of current SMTOs throughout Section 3, including recent ones that have not been covered in previous critiques, using the MNIST dataset [25] as a starting point to identify promising SMTOs, followed by other commonly used datasets such as Cityscapes [26] and Quantum Chemistry and Molecular Physics (QM9) [27].

  * •

An investigation into why equally weighted tasks can match the results of SMTOs. More specifically, the two similar tasks scenario in Section 3.2, and a complexity analysis in Section 3.3.

  * •

A re-evaluation of the hypothesis brought up by [23] that fixed weights can match the same results achieved by SMTOs in Section 3.5.

## 2 Related Work

In recent years, MTL and SMTOs have witnessed substantial research activity. Notable works, such as [17, 9], have garnered interest in MTL due to their performance improvements compared to the uniform loss and single task baselines. This surge has also given rise to numerous variations and novel SMTO methods, each improving on or proposing solutions to unaddressed concerns of early SMTOs [14, 28, 12, 16].

However, despite the growing body of research, the SMTO field faces a significant challenge: the absence of a formal benchmark. Consequently, reported results may vary between articles [9, 23, 10], raising concerns about the improvements. Specifically, recent critiques have revealed potential issues, suggesting that the reported performance gains from SMTOs stem from limited regularization [24] and under-optimized baseline hyperparameters [23].

Kurin et al. [24] show that all evaluated SMTOs are linked to a larger solution space, under-optimization or stochastic behavior, all of which induce a regularization effect [29, 30, 31]. They also evaluated their hypothesis by adding proper regularization such as weight decay, batch normalization [32], dropout [33], and early stopping to the uniform loss, referenced as Unitary Scalarization (Unit. Scal.) in this paper, and compared it to SMTOs. After the changes, they achieved competitive results to SMTOs at a lower computational cost.

Similarly, Xin et al. [23] argue that reported SMTO gains may be attributed to poor hyperparameter optimization, with particular interest in the impact of learning rates compared to random seeds. Their work demonstrates that exploring multiple fixed weight values can construct a Pareto curve, and SMTOs can only obtain one of the possible solutions. Due to computational restrictions, the discussion regarding fixed weights will be partially addressed as, to the best of our knowledge, there is no method to predict those fixed weights accurately, and grid search becomes impractical for a higher number of tasks.

Our work distinguishes itself in two key ways. Firstly, we consider most of the recent SMTOs in our evaluation [28, 12, 10, 18, 16, 20]. Secondly, we adopt some variations to the commonly used problems to introduce more complexity, as detailed in Section 3. Our objective is to demonstrate that SMTOs, mainly the recent ones, provide significant improvements over Unit. Scal. as the complexity increases. Additionally, we aim to elucidate the scenarios in which Unit. Scal. can match SMTOs and provide a complete evaluation of these promising SMTOs in multiple challenging MTL configurations.

## 3 Experimental Evaluation

Although numerous works utilize similar datasets and training configurations, a broadly accepted evaluation framework remains elusive. We have chosen specific problems from a selection of SMTO papers to address this challenge. More specifically, we restrict our evaluation to the supervised learning setting using datasets such as MNIST [25], Cityscapes [26], and QM9 [27]. To address recent critiques to SMTOs [24, 23] and ensure robustness, the models were trained with at least one form of regularization, except for the QM9 dataset, and the parameters were optimized by applying grid search. We determine the best hyperparameter configuration by training once with each value (excluding Multi-MNIST, which was trained five times for each value) and selecting the best validation result. The best combination is trained at least two more times, and the metrics reported here are from the test split. We consider the epoch in which the model achieved the highest validation metric to mitigate overfitting.

For comparative evaluation of different SMTOs, we used two widely employed metrics, the mean relative percentage gain metric $\Delta_{\text{mtm}}$ and the mean rank $MR$ [34, 10, 20]. $\Delta_{\text{mtm}}$ measures the improvement achieved by the MTL model over single-task references, and the $MR$ evaluates how equally all tasks were optimized. Since tasks like semantic segmentation and disparity estimation use multiple metrics for evaluation, we adapt the equations to account for that. For each set of metrics $M_{i}$ for task $i\in\mathcal{T}$ , the single task reference $M_{b,i,j}\;\mid j\in M_{i}$ is compared to its multi-task counterpart $M_{\text{smto},i,j}\;\mid j\in M_{i}$ , so $\Delta_{\text{mtm}}$ is given by:

|  $$ \Delta_{\text{mtm}}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{M_{i}}\sum_{j=1}^{M_{i}}(-1)^{l_{i,j}}\frac{M_{\text{smto},i,j}-M_{b,i,j}}{M_{b,i,j}}, $$  |  | (1)  
---|---|---|---  
  
where $l_{i,j}$ is $1$ if a lower value is better and $0$ otherwise. Likewise, the mean rank is given by:

|  $$ MR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{M_{i}}\sum_{j=1}^{M_{i}}R_{i,j}, $$  |  | (2)  
---|---|---|---  
  
where $R_{i,j}$ is the rank of the SMTO on task $i$ with respect to the metric $j$ . As a result, both metrics compute the mean within the same task and then the mean of every task, ensuring a fair evaluation in a multi-task setting.

From a wide array of publicly available SMTOs, we selected a subset for evaluation: UW [17], MGDA [9], PCGrad [35], GradDrop [15], EDM [14], IMTL [11], CAGrad [28], RotoGrad [12], Nash-MTL [10], SI [10], RLW [19], Auto-Lambda [18], CDTT [16] and FAMO [20]. We highlight that there are still some SMTOs that could be evaluated in future works [36, 13, 37, 38, 22, 21].

Including in Unit. Scal. and two RLW variations (Normal and Dirichlet distributions) as done in previous works [24], we assessed a total of 16 SMTOs. To reduce computational cost, the SMTOs considered in this paper are filtered according to their performance in a regression and classification problem. Even though evaluating an SMTO with just one example is not a definitive quality measure, this two-task problem tests the method’s ability to handle varying gradient norms. This concern motivated several SMTOs [13, 14, 11, 10]. Based on the results and contributions from each SMTO, we select the most promising ones and proceed to test them further.

In this section, the chosen SMTOs are first evaluated in the specific case of two similar task problems to demonstrate that, despite being an MTL problem, Unit. Scal. approximates the behavior of the SMTOs and achieves competitive results. We also consider some factors that dictate the need for SMTOs by comparing Unit. Scal. to one of the SMTOs while evaluating different configurations with increasing multi-task learning balancing complexity, using the Cityscapes dataset. All promising SMTOs are then evaluated in the most challenging format. Furthermore, we provide a comparison on the QM9 dataset. Finally, we evaluate the performance of fixed weights. All hyperparameter combinations, SMTOs-specific hyperparameter optimization, and their best configuration are summarized in Appendix A.6 for reproducibility.

### 3.1 Selecting promising SMTOs for further evaluation

In this paper, we tackle multiple MTL challenges to evaluate the performance of many SMTOs. However, due to the multitude of combinations and the necessity for hyperparameter optimization to ensure a fair evaluation [24, 23], we apply a preliminary filter for SMTO selection based on the performance on the Multi-MNIST dataset [9].

The Multi-MNIST dataset is a variation of the MNIST dataset for MTL where the digits are overlapped in the same image, one on the top left and one on the bottom right. This combined image is then used to predict information regarding the left and right digits. The most common variation is predicting the class of the left and right digits, as done in multiple works [9, 35, 12, 24, 23]. Instead, we opt for the alternative suggested by Nakamura et al. [16], where the model is trained to classify the top left digit and to reconstruct the bottom right digit. This choice was made because this variation adds more complexity, involving tasks with varying levels of difficulty and loss functions that result in gradients with different norms, an important factor when considering the specific case of two tasks, as shown in Section 3.2. Those tasks will be referenced as CL, CR, RL, and RR for the classification (C) or reconstruction (R) of the left (L) or right (R) digit. When tasks are represented using the notation {task 1}_{task 2}…{task N}, it indicates a multi-task scenario that simultaneously addresses tasks 1 through N.

For our model architecture, we use a variation of the Lenet model [25] as Nakamura et al. [16]. We then add dropout layers to the encoder and decoder architectures for regularization. The hyperparameters are optimized by grid search, varying the learning rate $lr\in$ {0.01, 0.075, 0.005, 0.0025, 0.001, 0.00075, 0.0005} and dropout $p\in$ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}. Specific SMTO hyperparameters are also included in the search if necessary. We train each configuration five times and the best validation configuration five more times, giving a total of ten models for each SMTO.

Figure 1: Box plot illustrating $\Delta_{\text{mtm}}$ scores on the Multi-MNIST classification and regression problem. The red line indicates the median value for Unit. Scal.. Notably, multiple SMTOs obtained substantial improvements over the Unit. Scal. baseline.

The results, as presented in Figure 1, demonstrate that Unit. Scal. achieves competitive results compared to some SMTOs. However, the majority outperform it significantly, primarily recent ones. Based on these results, we select EDM [14], IMTL [11], CAGrad [28], RotoGrad [12], Nash-MTL [10], Auto-Lambda [18], CDTT [16] and FAMO [20] as promising SMTOs for further investigation.

Despite the Multi-MNIST dataset’s simplicity, it effectively demonstrated the selected SMTOs’ ability to provide a meaningful gradient direction even with varying gradient norms, enabling their distinction from other methods. Specifically, PCGrad and MGDA are known to emphasize gradients with the smallest and largest magnitudes, respectively [11]. GradDrop randomly drops gradients based on sign-purity, without considering magnitude balancing. Furthermore, methods like RLW, UW, and SI operate directly on loss values without estimating gradients, unlike FAMO and Auto-Lambda, potentially making them sensitive to differences in gradient magnitude. Ultimately, most of the selected SMTOs significantly outperformed Unit. Scal. in our evaluations.

### 3.2 Two similar tasks

The two-task setting is commonly used as a computationally efficient way of evaluating SMTOs, widely adopted in multiple MTL papers [9, 24, 23]. However, when considering the two-task setting, it is possible that Unit. Scal. approximates the behavior of an SMTO given a few conditions.

When considering a problem with a shared encoder, the same decoder architecture, the same loss function, task labels with similar distributions, and learning speeds, the gradient norm and the loss value of both tasks are expected to behave in remarkably similar ways. Additionally, if we restrict this assumption to equal gradient norm or equal loss values, it is possible to theoretically show that most SMTOs converge to equal weights (see Appendix A.4).

To provide empirical evidence for the above statement, we consider two problems from Multi-MNIST: dual classification and dual reconstruction of the digits. In both cases, the architecture initially described in the previous section was reused with the corresponding changes to the decoders. We follow the same hyperparameter optimization previously described for Multi-MNIST.

Figure 2: Box plot of $\Delta_{\text{mtm}}$ (left) for all SMTOs not evaluated on [24] and mean error to equal weights (right). The top plot corresponds to CL_CR, and the bottom one represents RL_RR. All SMTOs had similar performance to Unit. Scal. and the computed weights varied near 0.5, excluding FAMO on the CL_CR problem, exemplifying how similar equal weights are to the selected SMTOs’ solution in this setting.

The results presented in Figure 2 show that all SMTOs performed closely to the Unit. Scal. baseline, even on the dual reconstruction case where $\Delta_{mtm}\approx-20\%$ , meaning that there was high interference between tasks. We can also observe that the learned weights from the SMTOs predominantly clustered around 0.5, besides FAMO, which was a notable exception, exhibiting instability on the dual classification problem, likely attributable to the small scale of the loss values.

Consequently, employing equal weights approximates the average behavior of the SMTOs, resulting in comparable performance. This observation suggests that a straightforward averaging of task gradients can serve as a surprisingly effective baseline without significant performance degradation in scenarios with similar task characteristics, such as the dual translation setting examined in the recent critique [23].

### 3.3 Complexity analysis

Cityscapes [26] is a street scene dataset containing high-resolution images 2048x1024 and labels for semantic segmentation, disparity estimation, and instance segmentation. It is commonly adopted for MTL [17, 9, 24, 10, 23, 20] under two variations, the two-task problem with 7 class semantic segmentation and disparity estimation, and the three-task problem that uses 19 semantic classes and adds instance segmentation. Some similarities arise when comparing the first version to the Multi-MNIST classification-reconstruction problem. Both problems generate gradients with different norms, and the tasks differ in complexity, yet there is no significant difference between SMTOs and Unit. Scal. with optimized hyperparameters [24, 23].

(a) Mean cosine similarity.

(b) IMTL and Unit. Scal. performance.

Figure 3: (a) Mean cosine similarity between each task gradient and the average direction. City R50 exhibits significantly higher similarity than Multi-MNIST, potentially explaining Unit. Scal.’s comparable performance to SMTOs. Conversely, the lower similarity for City 19C and QM9 correlates with Unit. Scal.’s relative underperformance. See Appendix A.2.1 for the full version with all configurations. (b) Shows the relative performance of Unit. Scal. and IMTL. Adding a third task and increasing the amount of information needed by increasing the number of classes are the relevant steps that dictate the relative performance in this problem.

The reason for this unexpected behavior is related to the lower amount of interference between each task compared to the Multi-MNIST problem, as seen in Figure 3(a). Figure 3(a) shows the mean cosine similarity between each task gradient and the average direction, which can be seen as a measure of interference, as it correctly ranks the problems with respect to the relative performance between Unit. Scal. and SMTOs. Even after significantly reducing the parameter count in step R18, the level of interference remains about the same (see Appendix A.2.1), as reflected in the relative performance between Unit. Scal. and IMTL. However, after increasing the complexity of the problem, there is a significant difference between the two methods.

To vary the complexity in a controlled manner, we vary model size, number of tasks, and information available to learn to identify which characteristic is relevant. We first trained the model on the same initial two-task configuration of [24] and applied small cumulative steps until the most complex configuration. We opted to test only IMTL [11] and compare it to Unit. Scal. in the transitional steps as it was one of the best performing SMTOs, and it does not require any specific hyperparameter optimization. All Cityscapes configurations are optimized varying the learning rate at $lr$ $\in$ {0.005, 0.001, 0.0005, 0.0001} and the weight decay $\in$ { $10^{-4}$ , $10^{-5}$ , 0.0} with 32 batch size. Each configuration is trained once, and then the best in the validation dataset is trained a total of 3 times. The selected steps are:

  * •

Reference (R50): A Resnet-50 [39] is used as shared encoder and the DeepLab decoder [40] is used as task specific decoder. The input resolution is 256x128, and the objective is set to 7 class segmentation and disparity estimation. This is the same configuration of Kurin et al. [24] and very similar to the one of Xin et al. [23];

  * •

Resnet-18 as decoder (R18): Opting for a smaller architecture should encourage competition for parameters;

  * •

Added instance segmentation (Inst. Seg.): Adding more tasks increases the complexity of the balancing problem, as well as the amount of information available to learn;

  * •

Increased resolution (HR): The resolution is increased from 256x128 to 512x256 as an intermediate step to allow better learning of objects from smaller classes;

  * •

19 semantic classes (19C): By increasing the number of learnable classes and keeping the model size fixed, the amount of information needed by the semantic decoder is increased, resulting in more competition for the shared encoder parameters and feature space.

The results in Figure 3(b) reveal that the benefit of employing an SMTO only becomes apparent upon the inclusion of the instance segmentation task. Furthermore, a substantial relative performance gain is observed after incorporating all 19 classes, suggesting that the volume of information significantly impacts the efficacy of SMTOs. This observation aligns with prior research on information transfer [41], which highlights the necessity of carefully calibrating the shared model’s capacity in multi-task learning. Insufficient or excessive capacity can lead to negative or negligible transfer. Therefore, by increasing the number of tasks and classes, we inherently increase the demands on the shared feature space, potentially increasing inter-task conflicts.

Similar behavior was also noticed in the NLP field [42], where they compared the interference caused by different languages and data size, concluding that a bigger model could mitigate the interference issue, though their evaluation was restricted to Unit. Scal.. On the other hand, we investigate the effect of those parameters on the relative performance of different SMTOs to the baseline.

This exemplifies that SMTOs are not needed on some occasions when the problem is simple enough that the conflicting gradients do not interfere with the final performance. Therefore, the best way to evaluate SMTOs is to use the most complex configuration to measure their capabilities in dealing with significant interference. Accordingly, the other promising SMTOs are trained and evaluated with the same procedures on the last configuration.

Figure 4 shows that SMTOs can indeed achieve better results when compared to Unit. Scal. in more complex balancing problems. This conclusion differs from Unit. Scal. [24] primarily because of the increased complexity. Furthermore, we evaluate some of the newer SMTOs, showing the field’s progress in the last years.

Figure 4: Muiti task metric on Cityscapes dataset. Most SMTOs significantly surpass the Unit. Scal. baseline on the most complex version of the Cityscapes MTL problem.

### 3.4 QM9

Finally, the last dataset we use to evaluate SMTOs is the QM9 dataset [27]. The QM9 dataset is a collection of quantum chemistry data for small organic molecules. It contains around 130,000 data points, and it was recently adopted as a problem for the MTL benchmarks [10, 20].

Table 1: Multi-task metric on QM9 dataset. Almost all SMTOs significantly surpass Unit. Scal. SMTO |  $\Delta_{MTM}$ (%) $\uparrow$  | MR $\downarrow$   
---|---|---  
Unit. Scal. | -135.7 | 4.45  
Auto-Lambda | -186.9 | 5.82  
CAGrad | -70.13 | 3.82  
CDTT | -121.2 | 5.45  
EDM | -95.42 | 4.00  
FAMO | -56.86 | 2.36  
Nash-MTL | -53.62 | 2.09  
  
We use the same configuration from [10]. The example from PyTorch Geometric [43] was adapted, and we used the model from [44]. The model was used to predict 11 properties from the molecules. We use the commonly adopted distribution of 110K molecules for training, 10K for validation, and 10K for testing, with a batch size of 120. The learning rate was optimized in $lr$ $\in$ { $5\times 10^{-3}$ , $10^{-3}$ , $5\times 10^{-4}$ }, and the ground truth values were normalized. We opted not to apply regularization because Unit. Scal. did not show significant improvement, and omitting this step reduced the overall computational cost.

Comparing the results from Table 1 to the previous experiments, it is evident that this is the problem with the most amount of interference between tasks, with all SMTOs obtaining results more than 50% worse on average compared to the single task problem. This also correlates with the lower cosine similarity shown in Figure 3(a).

During the experiment, IMTL was too unstable, and most runs crashed due to numerical overflow, as happened on the RL benchmark from Unit. Scal. [24], which also had a higher number of tasks. We agree with the hypothesis that this was caused by a lack of bounds on the scaling coefficients obtained from the IMTL’s optimization problem. RotoGrad also failed to converge, and our hypothesis is that RotoGrad was not able to find a proper rotation that minimizes the gradient conflict probably due to the increased difficulty of the problem. Further analysis can be seen in Appendix A.3. Both were removed from Table 1.

### 3.5 Fixed weights

Though fixed weights are the most efficient and straightforward solution, choosing the correct values for the weights of each task is a challenging problem. Xin et al. [23] finds the ideal, or close to ideal, weights by performing a grid search. This process may be viable for a more limited scenario, but for a large number of tasks it is not feasible. Very recently, Royer et al. [45] proposed a population-based training to find the ideal weights efficiently, though this method was not evaluated in this work due to time constraints.

One recent critique of SMTOs raises the question of whether optimized scalar weights can compete with SMTOs [23]. However, as discussed previously, the problems used in the critique might not adequately represent the performance of SMTOs, so we evaluate this claim on our experiments.

Finding well optimized weights for all problems using grid search would be unfeasible, so instead, we opt for a simpler strategy, using the weights output from an SMTO to weight each task loss. More specifically, the weights from CAGrad, EDM and Nash-MTL optimizers are used for the MNIST, Cityscapes and QM9 datasets, respectively. As we are using the Adam optimizer, there is a normalizing effect on the gradients, so the gradients in each decoder are not severely affected by the different norms caused by the loss weighting.

To select the weights, we computed the mean value in each epoch, applied an exponential moving average with $\beta=0.9$ , and used the result from the last epoch to define the weights. To extract and train with the fixed weights, we used the optimal hyperparameter configuration from the respective SMTO. However, for the QM9 dataset the fixed weights demonstrated to be unstable so we used the ones from $lr=0.0001$ , instead of the optimal $lr=0.001$ . The model is trained the same number of times as its SMTO counterpart.

Figure 5: Box plot comparing $\Delta_{\text{mtm}}$ between the original Nash-MTL optimizer and their extracted weights from each learning configuration. Fixed weights can achieve results comparable to those of an SMTO. However, they can be significantly more unstable than a dynamic SMTO, as was the case for the weights from $lr=0.001$ .

The results obtained from fixed weights, summarized in Table 2, indicate that it is indeed possible to get competitive performance using fixed weights. However, fixed weights can be more unstable, as shown in Figure 5 by the fixed weights extracted from $lr=0.001$ , with results ranging from $\approx-50\%$ to $\approx-95\%$ . Surprisingly, the optimal learning rate weights were not the best option for the QM9 dataset.

Table 2: Comparison between Unit. Scal, fixed weights, and the best performing SMTO. Well chosen fixed weights can achieve comparable performance to SMTOs. Dataset | Unit. Scal. | Fixed Weights | Best SMTO  
---|---|---|---  
$\Delta_{mtm}(\%)\uparrow$   
MNIST | -8.952 | -3.251 | -2.137  
City 19C | -5.312 | 0.145 | 0.570  
QM9 | -135.7 | -53.66 | -53.62  
  
This raises the question of why fixed weights can achieve comparable performance to SMTOs, as generating dynamic weights is a major benefit of SMTOs. Analyzing the weight value behavior in Figure 6, it is clear that all weights converge to a fixed value later in training. Additionally, due to the over-parameterized nature of deep learning models [46] the limitations typically associated with under-parameterized settings, such as the claim that scalarization is generally incapable of tracing out the Pareto front, do not necessarily apply [47]. We believe that further exploring the interaction between over-parameterization and multi-task learning could be a promising direction for future work.

This shows that using fixed weights is a valid alternative for complex MTL problems, corroborating the findings of Xin et al. [23]. However, finding those weights is highly costly, and with the advancement of loss-based SMTOs such as FAMO [20] and GO4Align [21], it is no longer a desirable choice.

Figure 6: Normalized weights from SMTOs. The shaded regions represent the mean value at each epoch, and the dotted lines show the results of the exponential moving average. The top left, top right, and bottom plots correspond to the MNIST, Cityscapes, and QM9 datasets, respectively. All datasets exhibit similar behavior, with the weights gradually converging to a stable value on average.

## 4 Limitations

In this work, we extensively tested multiple SMTOs on multiple configurations and datasets. However, it is important to note that we only focused on the supervised setting and did not include problems from NLP or reinforcement learning, fields in which MTL is commonly evaluated [24, 23]. Also, our initial selection process involved filtering certain SMTOs based on their norm balancing performance on the MNIST classification-reconstruction problem. Despite this strong preliminary evidence, a more exhaustive analysis of the excluded methods may reveal unknown advantages. Furthermore, some SMTOs were reported to provide better results when combining them with other SMTOs [12, 18], but because of time constraints, we limited the experiments to only the vanilla version. Finally, we only considered the MTL problem, where all tasks are equally weighted, and did not consider scenarios designed for auxiliary tasks.

## 5 Conclusion

This work aimed to address recent critiques of SMTOs by conducting a comprehensive empirical analysis under the context of supervised learning. First, we examined why Unit. Scal. can replicate the performance of SMTOs in two-task scenarios, particularly when the tasks are similar. Our findings suggest that SMTOs converge to equal weights, so Unit. Scal. approximates their behavior on average. Furthermore, we explored the conditions under which Unit. Scal. matches SMTOs, attributing its success to factors such as the degree of task interference, the number of tasks, and the amount of information to be learned.

We also investigated the relative performance of fixed weights compared to SMTOs. By deriving fixed weights from SMTOs, we demonstrated that fixed weights can perform competitively in complex scenarios. However, using fixed weights is less appealing due to the computational overhead of identifying optimal values and their comparable performance to recent loss-based SMTOs.

In conclusion, this study provides a robust evaluation of current SMTOs, underscoring their significance in addressing complex multi-task problems. Additionally, we aim to establish points to consider when evaluating SMTOs and to guide the development of future methods and applications.

## Acknowledgments

This work was partially funded by CNPq under grants 465755/2014-3, 309532/2023-0, and 130961/2024-8, and CAPES under finance code 001.

## References

  * [1] R. Caruana (1997) Multitask learning.  Machine Learning 28 (1), pp. 41–75.  Cited by: §1. 
  * [2] J. Baxter (2000) A model of inductive bias learning.  Journal of artificial intelligence research 12, pp. 149–198.  Cited by: §1. 
  * [3] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert (2016) Cross-stitch networks for multi-task learning.  In Proceedings of the IEEE conference on computer vision and pattern recognition,  pp. 3994–4003.  Cited by: §1. 
  * [4] Y. Gao, J. Ma, M. Zhao, W. Liu, and A. L. Yuille (2019) Nddr-cnn: layerwise feature fusing in multi-task cnns by neural discriminative dimensionality reduction.  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,  pp. 3205–3214.  Cited by: §1. 
  * [5] S. Liu, E. Johns, and A. J. Davison (2019) End-to-end multi-task learning with attention.  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,  pp. 1871–1880.  Cited by: §1. 
  * [6] T. Standley, A. Zamir, D. Chen, L. Guibas, J. Malik, and S. Savarese (2020) Which tasks should be learned together in multi-task learning?.  In International Conference on Machine Learning,  pp. 9120–9132.  Cited by: §1. 
  * [7] C. Fifty, E. Amid, Z. Zhao, T. Yu, R. Anil, and C. Finn (2021) Efficiently identifying task groupings for multi-task learning.  Advances in Neural Information Processing Systems 34, pp. 27503–27516.  Cited by: §1. 
  * [8] T. Hospedales, A. Antoniou, P. Micaelli, and A. Storkey (2021) Meta-learning in neural networks: a survey.  IEEE transactions on pattern analysis and machine intelligence 44 (9), pp. 5149–5169.  Cited by: §1. 
  * [9] O. Sener and V. Koltun (2018) Multi-task learning as multi-objective optimization.  Advances in neural information processing systems 31.  Cited by: §1, §1, §1, §2, §2, §3.1, §3.1, §3.2, §3.3, §3. 
  * [10] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, and E. Fetaya (2022) Multi-task learning as a bargaining game.  In Proceedings of the 39th International Conference on Machine Learning, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.),  Proceedings of Machine Learning Research, Vol. 162, pp. 16428–16446.  Cited by: §1, §1, §1, §2, §2, §3.1, §3.3, §3.4, §3.4, §3, §3, §3, Proposition 2. 
  * [11] L. Liu, Y. Li, Z. Kuang, J. Xue, Y. Chen, W. Yang, Q. Liao, and W. Zhang (2021) Towards impartial multi-task learning.  In International Conference on Learning Representations,  External Links: [Link](<https://openreview.net/forum?id=IMPnRXEWpvr>) Cited by: §1, §1, §3.1, §3.1, §3.3, §3, §3, Proposition 3. 
  * [12] A. Javaloy and I. Valera (2022) RotoGrad: gradient homogenization in multitask learning.  In International Conference on Learning Representations,  Cited by: §A.2.2, §1, §1, §2, §2, §3.1, §3.1, §3, §4. 
  * [13] Z. Chen, V. Badrinarayanan, C. Lee, and A. Rabinovich (2018) Gradnorm: gradient normalization for adaptive loss balancing in deep multitask networks.  In International conference on machine learning,  pp. 794–803.  Cited by: §1, §3, §3. 
  * [14] A. Katrutsa, D. Merkulov, N. Tursynbek, and I. Oseledets (2020) Follow the bisector: a simple method for multi-objective optimization.  arXiv preprint arXiv:2007.06937.  Cited by: §A.4, §A.4, §1, §2, §3.1, §3, §3, Proposition 4. 
  * [15] Z. Chen, J. Ngiam, Y. Huang, T. Luong, H. Kretzschmar, Y. Chai, and D. Anguelov (2020) Just pick a sign: optimizing deep multitask models with gradient sign dropout.  Advances in Neural Information Processing Systems 33, pp. 2039–2050.  Cited by: §1, §3. 
  * [16] A. T. M. Nakamura, V. Grassi Jr, and D. F. Wolf (2022) Leveraging convergence behavior to balance conflicting tasks in multi-task learning.  Neurocomputing 511, pp. 43–53.  Cited by: §1, §2, §2, §3.1, §3.1, §3.1, §3, Proposition 5. 
  * [17] R. Cipolla, Y. Gal, and A. Kendall (2018) Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.  In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,  Vol. , pp. 7482–7491.  External Links: [Document](<https://dx.doi.org/10.1109/CVPR.2018.00781>) Cited by: §1, §2, §3.3, §3. 
  * [18] S. Liu, S. James, A. Davison, and E. Johns (2022) Auto-lambda: disentangling dynamic task relationships.  Transactions on Machine Learning Research.  Note:  External Links: ISSN 2835-8856, [Link](<https://openreview.net/forum?id=KKeCMim5VN>) Cited by: §1, §2, §3.1, §3, §4. 
  * [19] B. Lin, F. YE, Y. Zhang, and I. Tsang (2022) Reasonable effectiveness of random weighting: a litmus test for multi-task learning.  Transactions on Machine Learning Research.  Note:  External Links: ISSN 2835-8856, [Link](<https://openreview.net/forum?id=jjtFD8A1Wx>) Cited by: §1, §3. 
  * [20] B. Liu, Y. Feng, P. Stone, and Q. Liu (2023) Famo: fast adaptive multitask optimization.  Advances in Neural Information Processing Systems 36, pp. 57226–57243.  Cited by: §1, §2, §3.1, §3.3, §3.4, §3.5, §3, §3. 
  * [21] J. Shen, C. Wang, Z. Xiao, N. V. Noord, and M. Worring (2024) GO4align: group optimization for multi-task alignment.  In The Thirty-eighth Annual Conference on Neural Information Processing Systems,  Cited by: §1, §3.5, §3. 
  * [22] I. Achituve, I. Diamant, A. Netzer, G. Chechik, and E. Fetaya (2024) Bayesian uncertainty for gradient aggregation in multi-task learning.  In Forty-first International Conference on Machine Learning,  Cited by: §1, §3. 
  * [23] D. Xin, B. Ghorbani, J. Gilmer, A. Garg, and O. Firat (2022) Do current multi-task optimization methods in deep learning even help?.  Advances in Neural Information Processing Systems 35, pp. 13597–13609.  Cited by: 3rd item, §1, §2, §2, 1st item, §3.1, §3.1, §3.2, §3.2, §3.3, §3.5, §3.5, §3.5, §3, §4. 
  * [24] V. Kurin, A. De Palma, I. Kostrikov, S. Whiteson, and P. K. Mudigonda (2022) In defense of the unitary scalarization for deep multi-task learning.  Advances in Neural Information Processing Systems 35, pp. 12169–12183.  Cited by: §1, §2, §2, Figure 2, Figure 2, 1st item, §3.1, §3.1, §3.2, §3.3, §3.3, §3.3, §3.4, §3, §3, §4. 
  * [25] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner (1998) Gradient-based learning applied to document recognition.  Proceedings of the IEEE 86 (11), pp. 2278–2324.  Cited by: 1st item, §3.1, §3. 
  * [26] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele (2016) The cityscapes dataset for semantic urban scene understanding.  In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),  Cited by: 1st item, §3.3, §3. 
  * [27] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, and V. Pande (2018) MoleculeNet: a benchmark for molecular machine learning.  Chemical science 9 (2), pp. 513–530.  Cited by: 1st item, §3.4, §3. 
  * [28] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu (2021) Conflict-averse gradient descent for multi-task learning.  Advances in Neural Information Processing Systems 34, pp. 18878–18890.  Cited by: §2, §2, §3.1, §3. 
  * [29] T. Dietterich (1995) Overfitting and undercomputing in machine learning.  ACM computing surveys (CSUR) 27 (3), pp. 326–327.  Cited by: §2. 
  * [30] N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang (2016) On large-batch training for deep learning: generalization gap and sharp minima.  International Conference on Learning Representations.  Cited by: §2. 
  * [31] B. Kleinberg, Y. Li, and Y. Yuan (2018) An alternative view: when does sgd escape local minima?.  In International conference on machine learning,  pp. 2698–2707.  Cited by: §2. 
  * [32] S. Ioffe and C. Szegedy (2015) Batch normalization: accelerating deep network training by reducing internal covariate shift.  In International conference on machine learning,  pp. 448–456.  Cited by: §2. 
  * [33] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov (2014) Dropout: a simple way to prevent neural networks from overfitting.  The journal of machine learning research 15 (1), pp. 1929–1958.  Cited by: §2. 
  * [34] S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, and L. Van Gool (2021) Multi-task learning for dense prediction tasks: a survey.  IEEE transactions on pattern analysis and machine intelligence 44 (7), pp. 3614–3633.  Cited by: §3. 
  * [35] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn (2020) Gradient surgery for multi-task learning.  Advances in Neural Information Processing Systems 33, pp. 5824–5836.  Cited by: §3.1, §3. 
  * [36] M. Guo, A. Haque, D. Huang, S. Yeung, and L. Fei-Fei (2018) Dynamic task prioritization for multitask learning.  In Proceedings of the European conference on computer vision (ECCV),  pp. 270–287.  Cited by: §3. 
  * [37] B. Lin, W. Jiang, F. Ye, Y. Zhang, P. Chen, Y. Chen, and S. Liu (2023) A scale-invariant task balancing approach for multi-task learning.  CoRR abs/2308.12029.  Cited by: §3. 
  * [38] B. Lin, W. Jiang, F. Ye, Y. Zhang, P. Chen, Y. Chen, S. Liu, and J. T. Kwok (2023) Dual-balancing for multi-task learning.  External Links: 2308.12029 Cited by: §3. 
  * [39] K. He, X. Zhang, S. Ren, and J. Sun (2016) Deep residual learning for image recognition.  In Proceedings of the IEEE conference on computer vision and pattern recognition,  pp. 770–778.  Cited by: 1st item. 
  * [40] L. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam (2018) Encoder-decoder with atrous separable convolution for semantic image segmentation.  In Proceedings of the European conference on computer vision (ECCV),  pp. 801–818.  Cited by: 1st item. 
  * [41] S. Wu, H. R. Zhang, and C. Ré (2020) Understanding and improving information transfer in multi-task learning.  In International Conference on Learning Representations,  External Links: [Link](<https://openreview.net/forum?id=SylzhkBtDB>) Cited by: §3.3. 
  * [42] U. Shaham, M. Elbayad, V. Goswami, O. Levy, and S. Bhosale (2022) Causes and cures for interference in multilingual translation.  arXiv preprint arXiv:2212.07530.  Cited by: §3.3. 
  * [43] M. Fey and J. E. Lenssen (2019) Fast graph representation learning with PyTorch Geometric.  In ICLR Workshop on Representation Learning on Graphs and Manifolds,  Cited by: §3.4. 
  * [44] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl (2017) Neural message passing for quantum chemistry.  In International conference on machine learning,  pp. 1263–1272.  Cited by: §3.4. 
  * [45] A. Royer, T. Blankevoort, and B. Ehteshami Bejnordi (2024) Scalarization for multi-task and multi-domain learning at scale.  Advances in Neural Information Processing Systems 36.  Cited by: §3.5. 
  * [46] J. Frankle and M. Carbin (2019) The lottery ticket hypothesis: finding sparse, trainable neural networks.  In International Conference on Learning Representations,  External Links: [Link](<https://openreview.net/forum?id=rJl-b3RcF7>) Cited by: §3.5. 
  * [47] Y. Hu, R. Xian, Q. Wu, Q. Fan, L. Yin, and H. Zhao (2023) Revisiting scalarization in multi-task learning: a theoretical perspective.  In Thirty-seventh Conference on Neural Information Processing Systems,  External Links: [Link](<https://openreview.net/forum?id=6EqUpqMnwl>) Cited by: §3.5. 
  * [48] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala (2019) PyTorch: an imperative style, high-performance deep learning library.  In Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.),  Vol. 32, pp. .  External Links: [Link](<https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf>) Cited by: §A.1. 

## Appendix A Technical Appendices and Supplementary Material

### A.1 Experimental Setup

All experiments were implemented using PyTorch [48] and executed on a machine equipped with an AMD Ryzen 9 5950X CPU and two NVIDIA RTX 3090 GPUs (24 GB each). However, experiments were run independently on each GPU. The random seed was set based on the corresponding run index.

### A.2 Additional results

#### A.2.1 Cosine similarity

Figure 7 demonstrates the strong correlation between the mean cosine similarity of task gradients and the relative performance of Unit. Scal. and SMTOs. Notably, the tasks where Unit. Scal. achieved comparable results to SMTOs (MNIST CL_CR, MNIST RL_RR, City R50, and City R18) exhibit high mean cosine similarities ( $\approx$ 0.7) and form a distinct cluster. Conversely, the more challenging Cityscapes tasks (City Inst. Seg., City HR, City 19C) show a lower, yet consistent, similarity ( $\approx$ 0.45). Finally, the most difficult task from the QM9 dataset, is characterized by the lowest mean cosine similarity ( $\approx$ 0.3).

Figure 7: Mean cosine similarity between each task gradient and the average direction for all problems. It was able to correctly rank all the different problems in terms of relative performance between Unit. Scal. and SMTOs. (i.e. higher cosine similarity relates to lower difference between Unit. Scal. and SMTOs.)

#### A.2.2 Statistics

Similarly to [12], we provide additional statistics in Tables 3, 4, and 5 and the missing visual result in Figure 8. Most notably, we see a clear preference for the classification task on the MNIST dataset from the underperforming SMTOs, as well as a lower minimum value and higher standard deviation in Table 5 for Unit. Scal. when compared to the top performing SMTOs, FAMO and Nash-MTL.

Table 3: Mean $\Delta_{mtm}(\%)\uparrow$ of different SMTOs on the MNIST dataset. SMTO | CL | RR | avg  
---|---|---|---  
Unit. Scal. | -1.043 | -16.86 | -8.952  
Auto-Lambda | -1.232 | -7.186 | -4.209  
CAGrad | -1.685 | -2.590 | -2.137  
CDTT | -1.546 | -3.596 | -2.571  
EDM | -1.613 | -2.993 | -2.303  
FAMO | -4.587 | -0.745 | -2.666  
GradDrop | -1.076 | -22.04 | -11.56  
IMTL | -2.513 | -1.975 | -2.244  
MGDA-UB | -1.448 | -11.26 | -6.352  
Nash-MTL | -1.695 | -4.430 | -3.063  
PCGrad | -1.039 | -15.36 | -8.201  
RLW-Dirichlet | -1.145 | -19.48 | -10.31  
RLW-Normal | -1.100 | -18.53 | -9.817  
RotoGrad | -2.705 | -1.572 | -2.139  
SI | -0.962 | -27.47 | -14.21  
UW | -1.109 | -26.30 | -13.70  
Table 4: Mean $\Delta_{mtm}(\%)\uparrow$ of different SMTOs on the Cityscapes dataset. SMTO | Segmentation | Disparity | Instance Seg. | avg  
---|---|---|---|---  
Acc | mIoU | L1 abs | L1 rel | L1 abs  
Unit. Scal. | -2.120 | -16.78 | -8.542 | -11.41 | 3.492 | -5.312  
Auto-Lambda | -1.922 | -15.02 | -8.546 | -9.317 | 3.657 | -4.582  
CAGrad | -0.977 | -7.647 | -6.709 | -10.06 | 3.325 | -3.124  
CDTT | -0.948 | -8.954 | 2.043 | -0.224 | 0.472 | -1.190  
EDM | -0.225 | -3.174 | 0.990 | -1.051 | 2.468 | 0.246  
FAMO | -0.274 | -3.268 | 0.755 | -2.212 | 2.339 | -0.054  
IMTL | -0.224 | -2.759 | 3.017 | 0.993 | 1.195 | 0.570  
Nash-MTL | -0.290 | -3.275 | 0.297 | -0.227 | 1.409 | -0.113  
RotoGrad | -0.453 | -4.484 | 0.303 | -3.493 | -1.659 | -1.908  
Table 5: Performance of different SMTOs on the QM9 dataset based on $\Delta_{mtm}(\%)$ statistics. SMTO | min $\uparrow$  | max $\uparrow$  | med $\uparrow$  | std $\downarrow$  | avg $\uparrow$   
---|---|---|---|---|---  
Unit. Scal. | -849.3 | -8.280 | -61.62 | 229.4 | -135.7  
Auto-Lambda | -974.5 | -18.32 | -115.4 | 255.5 | -186.9  
CAGrad | -408.0 | -5.451 | -32.08 | 108.7 | -70.13  
CDTT | -652.9 | -30.90 | -70.34 | 169.7 | -121.2  
EDM | -526.0 | -24.14 | -56.26 | 137.0 | -95.42  
FAMO | -351.5 | 6.590 | -16.95 | 97.78 | -56.86  
Nash-MTL | -390.2 | 15.10 | -21.41 | 108.8 | -53.62  
Figure 8: Box plot comparing $\Delta_{mtm}$ between Unit. Scal. and SMTOs on the QM9 dataset.

### A.3 Rotograd’s instability on the QM9 dataset

The RotoGrad optimizer aims to mitigate conflicts in gradient directions by rotating the feature space so that the gradient of each task’s gradient aligns with the direction formed by the equally weighted sum of the normalized gradients: $v=\frac{1}{N}\sum_{i}^{N}u_{sh,i}$ , where $u_{sh,i}=\frac{g_{sh,i}}{\|g_{sh,i}\|}$ represents the normalized gradient of task $i$ for the shared parameters.

In the specific case of the QM9 experiment, RotoGrad failed to converge. We hypothesize that the relatively small final shared feature space, combined with the high number of tasks, significantly increased the difficulty of the balancing problem. As shown in Figure 9, the rotation loss $\mathcal{L}_{rot}$ is orders of magnitude higher compared to the MNIST and Cityscapes problems. Since $\mathcal{L}_{rot}$ does not directly enforce a reduction in task loss, this instability may have negatively affected the model optimization.

For this specific experiment, we used $lr=0.001$ as it was the optimal learning rate for most SMTOs, and $lrr=0.1$ since it was the smallest value used on the grid search, so ideally it should be the most stable.

Figure 9: Mean rotation loss $\left(\mathcal{L}_{rot}\right)$ from RotoGrad. The $\mathcal{L}_{rot}$ is at least 10 $\times$ higher than the other cases, which could explain the training instability.

### A.4 Proofs

###### Assumption 1.

Consider a model $f_{\theta}$ composed of a shared encoder $f_{\theta_{sh}}$ and task-specific modules $f_{\theta_{t}}$ for each task $t\in\\{1,\dots,N\\}$ . In the case of two tasks ( $N=2$ ), we assume the following symmetry conditions: (i) both tasks use the same loss function, (ii) the task-specific modules have the same architecture, (iii) the tasks are of similar difficulty, and (iv) the tasks have labels with similar distributions. Under these conditions, we assume that the gradients with respect to the shared parameters have equal norm, i.e., $\|g_{1}\|=\|g_{2}\|$ .

###### Assumption 2.

Under the same conditions as Assumption 1, assume that the loss values of each task are equal, i.e., $\ell_{1}=\ell_{2}$ .

Although Assumptions 1, 2 impose a strong symmetry requirement, our experiments on MNIST revealed that in both the dual classification and dual reconstruction tasks, the resulting weights from all SMTOs were consistently close to 0.5, with the sole exception of FAMO in the dual classification scenario.

###### Proposition 1.

Under Assumption 1, the optimal weights computed by CAGrad correspond to equal weighting.

###### Proof.

We consider the CAGrad optimization for two tasks:

|  $$ w^{*}=\mathop{\arg\min}_{w\in\mathcal{W}}\;g_{w}^{\top}g_{0}+\sqrt{\phi}\,\|g_{w}\|, $$  |   
---|---|---  
  
where:

|  $\displaystyle g_{w}$  |  $\displaystyle=w_{1}g_{1}+w_{2}g_{2},$  |   
---|---|---|---  
|  $\displaystyle g_{0}$  |  $\displaystyle=\frac{1}{2}(g_{1}+g_{2}),$  |   
|  $\displaystyle\phi$  |  $\displaystyle=c^{2}\|g_{0}\|^{2},\quad c\in[0,1),$  |   
|  $\displaystyle\mathcal{W}$  |  $\displaystyle=\\{w\in\mathbb{R}^{2}\mid w_{1}+w_{2}=1,\;w_{1},w_{2}\geq 0\\}.$  |   
  
We also define:

|  $\displaystyle A$  |  $\displaystyle\coloneqq\|g_{1}\|=\|g_{2}\|,$  |   
---|---|---|---  
|  $\displaystyle p$  |  $\displaystyle\coloneqq g_{1}^{\top}g_{2}.$  |   
  
We begin by expanding the inner product:

|  $\displaystyle g_{w}^{\top}g_{0}$  |  $\displaystyle=\left(w_{1}g_{1}+w_{2}g_{2}\right)^{\top}\frac{1}{2}(g_{1}+g_{2})$  |   
---|---|---|---  
|  |  $\displaystyle=\frac{1}{2}\left[w_{1}(g_{1}^{\top}g_{1}+g_{1}^{\top}g_{2})+w_{2}(g_{2}^{\top}g_{1}+g_{2}^{\top}g_{2})\right]$  |   
|  |  $\displaystyle=\frac{1}{2}\left[w_{1}(A^{2}+p)+w_{2}(p+A^{2})\right]$  |   
|  |  $\displaystyle=\frac{1}{2}(A^{2}+p)(w_{1}+w_{2})=\frac{1}{2}(A^{2}+p).$  |   
  
This term is independent of $w$ , as well as $\sqrt{\phi}$ , so the optimization reduces to:

|  $$ \mathop{\arg\min}_{w\in\mathcal{W}}\|w_{1}g_{1}+w_{2}g_{2}\|. $$  |   
---|---|---  
  
Let $w^{*}$ be the optimal solution, which is known to be equal weights if $\|g_{1}\|=\|g_{2}\|$ . The CAGrad update direction is then given by:

|  $$ d^{*}=g_{0}+\frac{\sqrt{\phi}}{\|g_{w^{*}}\|}\,g_{w^{*}}. $$  |   
---|---|---  
  
If $w^{*}=(\tfrac{1}{2},\tfrac{1}{2})$ , then $g_{w^{*}}=\frac{1}{2}(g_{1}+g_{2})=g_{0}$ , and:

|  $$ d^{*}=g_{0}+\frac{c\|g_{0}\|}{\|g_{0}\|}\,g_{0}=(1+c)\,g_{0}=\sum_{i=1}^{2}g_{i}\left(\frac{1+c}{2}\right). $$  |   
---|---|---  
  
Thus, $d^{*}$ is an equal-weight combination of the gradients. ∎

###### Proposition 2.

Under Assumption 1, the optimal weights computed by Nash-MTL [10] are equal weights in the two-task scenario.

###### Proof.

The weights are computed by approximating the solution of:

|  $$ G^{\top}G\alpha=\frac{1}{\alpha}, $$  |   
---|---|---  
  
where $G$ is the matrix whose columns are the shared gradients and $\alpha_{i}\;>0,\;i\in\\{1,...,N\\}$ the tasks’ weights. In a two-task scenario, the problem can be simplified to:

|  $$ \left\\{\begin{aligned} \alpha_{1}g_{1}^{\top}g_{1}+\alpha_{2}g_{2}^{\top}g_{1}&=\frac{1}{\alpha_{1}}\\\ \alpha_{2}g_{2}^{\top}g_{2}+\alpha_{1}g_{1}^{\top}g_{2}&=\frac{1}{\alpha_{2}}\end{aligned}\right. $$  |   
---|---|---  
  
Defining $A\coloneqq\|g_{1}\|=\|g_{2}\|$ and $p\coloneqq\frac{g_{1}^{\top}g_{2}}{A^{2}}$ :

|  $$ \left\\{\begin{aligned} \alpha_{1}A^{2}+\alpha_{2}A^{2}p&=\frac{1}{\alpha_{1}}\\\ \alpha_{2}A^{2}+\alpha_{1}A^{2}p&=\frac{1}{\alpha_{2}}\end{aligned}\right. $$  |   
---|---|---  
  
Rearrange the equations:

|  $$ \left\\{\begin{aligned} \alpha_{1}^{2}A^{2}+\alpha_{1}\alpha_{2}A^{2}p&=1\\\ \alpha_{2}^{2}A^{2}+\alpha_{1}\alpha_{2}A^{2}p&=1\end{aligned}\right. $$  |   
---|---|---  
  
Notice that by symmetry of the two equations, the solution should satisfy $\alpha_{1}^{2}=\alpha_{2}^{2}$ . Since both weights are positive, this immediately implies:

|  $$ \alpha_{1}=\alpha_{2}. $$  |   
---|---|---  
  
∎

###### Proposition 3.

Under Assumption 1, the optimal weights computed by IMTL-G [11] are equal weights in the two-task scenario.

###### Proof.

The weights are obtained by solving:

|  $$ \alpha=g_{1}U^{\top}(DU^{\top})^{-1}, $$  |   
---|---|---  
  
where $\alpha=\\{\alpha_{2},...,\alpha_{N}\\}$ is constrained to $\sum_{i}\alpha_{i}=1$ , $U^{\top}=[u^{\top}_{1}-u^{\top}_{2},...,u^{\top}_{1}-u^{\top}_{N}]$ , with $u_{i}=g_{i}/\|g_{i}\|$ , and $D^{\top}=[g_{1}^{\top}-g_{2}^{\top},...,g_{1}^{\top}-g_{N}^{\top}]$ . Define $A\coloneqq\|g_{1}\|=\|g_{2}\|$ and $p=g_{1}^{\top}g_{2}$ . In the two-task scenario, the problem reduces to:

|  $\displaystyle\alpha_{2}$  |  $\displaystyle=g_{1}\left(\frac{g_{1}^{\top}-g_{2}^{\top}}{A}\right)\left((g_{1}-g_{2})\frac{(g_{1}^{\top}-g_{2}^{\top})}{A}\right)^{-1}$  |   
---|---|---|---  
|  $\displaystyle\alpha_{2}$  |  $\displaystyle=g_{1}\left(\frac{g_{1}^{\top}-g_{2}^{\top}}{A}\right)\left(\frac{\|g_{1}-g_{2}\|^{2}}{A}\right)^{-1}$  |   
|  $\displaystyle\alpha_{2}$  |  $\displaystyle=g_{1}\left(\frac{g_{1}^{\top}-g_{2}^{\top}}{\|g_{1}-g_{2}\|^{2}}\right)=\frac{A^{2}-p}{2(A^{2}-p)}=\frac{1}{2}.$  |   
  
Since $\alpha_{1}+\alpha_{2}=1$ , this completes the proof.

∎

###### Proposition 4.

Under assumption 1, the optimal weights computed by EDM [14] are equal weights in the two-task scenario.

###### Proof.

EDM’s original paper [14] already showed that for the two task scenario the optimal direction is computed as follows:

|  $$ d^{*}=\left(\frac{1}{\|g_{1}\|}+\frac{1}{\|g_{2}\|}\right)^{-1}\left(\frac{g_{1}}{\|g_{1}\|}+\frac{g_{2}}{\|g_{2}\|}\right). $$  |   
---|---|---  
  
Define $A\coloneqq\|g_{1}\|=\|g_{2}\|$ :

|  $$ d^{*}=\left(\frac{1}{A}+\frac{1}{A}\right)^{-1}\left(\frac{g_{1}}{A}+\frac{g_{2}}{A}\right)=\frac{1}{2}(g_{1}+g_{2}).\\\ $$  |   
---|---|---  
  
∎

###### Proposition 5.

Under Assumptions 1 and 2, the optimal weights computed by CDTT [16] are equal weights.

###### Proof.

CDTT applies a tension vector to the optimal direction $d^{*}$ computed by EDM, which is equal weights in this case (see Proposition 4). The final vector is computed by:

|  $\displaystyle\zeta_{i}$  |  $\displaystyle=\frac{1}{N}\sum_{i=1}^{N}\|g_{i}(t-i)\|$  |   
---|---|---|---  
|  $\displaystyle\delta_{i}$  |  $\displaystyle=\frac{\|\zeta_{i}(t)\|}{\|\zeta_{i}(t-1)\|}+\log_{10}(\ell_{i})$  |   
|  $\displaystyle c_{i}$  |  $\displaystyle=\frac{\alpha}{1+e^{(-\delta_{i}e+e)}}+1-\alpha$  |   
|  $\displaystyle d_{n}^{*}$  |  $\displaystyle=d^{*}+\sum^{N}_{i=1}c_{i}\left(\frac{g_{i}-d^{*}}{\|g_{i}-d^{*}\|}\right),$  |   
  
where $t$ refers to the iteration index and $\alpha\in[0,1]$ is a constant to regulate the tension factor’s sensitivity.

It follows directly from Assumptions 1, 2 and the component’s equations that $c=c_{1}=c_{2}$ . Analyzing the $\|g_{i}-d^{*}\|$ separately:

|  $$ \|g_{i}-d^{*}\|^{2}=\|g_{i}\|^{2}+\|d^{*}\|^{2}-2\langle g_{i},d^{*}\rangle. $$  |   
---|---|---  
  
The direction computed by EDM $d^{*}$ is guaranteed to have the same angle with each of the shared gradients [14]. Therefore, if $\|g_{1}\|=\|g_{2}\|,\;\text{then}\;\langle g_{1},d^{*}\rangle=\langle g_{2},d^{*}\rangle$ . Then we can consider $B\coloneqq\|g_{1}-d^{*}\|=\|g_{2}-d^{*}\|$ and remove it together with $c$ from the sum:

|  $\displaystyle d_{n}^{*}$  |  $\displaystyle=d^{*}+\frac{c}{B}\sum^{2}_{i=1}\left(g_{i}-d^{*}\right)$  |   
---|---|---|---  
|  $\displaystyle d_{n}^{*}$  |  $\displaystyle=d^{*}+\frac{c}{B}(g_{1}+g_{2}-2d^{*})$  |   
|  $\displaystyle d_{n}^{*}$  |  $\displaystyle=d^{*}=\frac{1}{2}(g_{1}+g_{2}).$  |   
  
∎

###### Proposition 6.

Under Assumption 2 (extended for N tasks), if the logits $\xi_{t}=(\xi_{1,t},\dots,\xi_{N,t})$ are initialized equally, then the FAMO algorithm maintains equal task weights:

|  $$ w_{t}=w_{i,t}=c_{t}\frac{z_{i,t}}{\ell_{i,t}}, $$  |   
---|---|---  
  
where $c_{t}=\left(\sum_{i=1}^{N}\dfrac{z_{i,t}}{\ell_{i,t}}\right)$ , for all $i=1,\dots,N$ and for all $t$ .

###### Proof.

We prove the result by induction on the iteration $t$ .

Base Case ( $t=0$ ): Assume that $\xi_{i,0}=c$ for all $i$ , for some constant $c$ . Then the softmax yields

|  $$ z_{i,0}=\frac{\exp(\xi_{i,0})}{\sum_{j=1}^{N}\exp(\xi_{j,0})}=\frac{\exp(c)}{N\exp(c)}=\frac{1}{N}. $$  |   
---|---|---  
  
The model parameters are updated as follows:

|  $$ \theta_{1}=\theta_{0}-\alpha\sum_{i=1}^{N}\left(c_{0}\frac{z_{i,0}}{\ell_{i,0}}\right)\nabla\ell_{i,0}, $$  |   
---|---|---  
  
where $c_{0}=\left(\sum_{i=1}^{N}\dfrac{z_{i,0}}{\ell_{i,0}}\right)^{-1}$ .

Under the assumption $\ell_{i,t}=\ell_{t}$ for all $i$ , each term $\frac{z_{i,t}}{\ell_{i,t}}$ is identical across tasks. Therefore, this update does not introduce any asymmetry among tasks. Thus, the base case holds.

Inductive Step: Suppose that at iteration $t$ we have

|  $$ \xi_{t}=(c_{t},c_{t},\dots,c_{t}), $$  |   
---|---|---  
  
which implies

|  $$ z_{i,t}=\frac{\exp(c_{t})}{N\exp(c_{t})}=\frac{1}{N}\quad\text{for all }i. $$  |   
---|---|---  
  
Next, the logits are updated via

|  $$ \xi_{t+1}=\xi_{t}-\beta\left(\delta_{t}+\gamma\,\xi_{t}\right), $$  |   
---|---|---  
  
with

|  $$ \delta_{t}=\begin{bmatrix}\nabla^{\top}z_{1,t}(\xi_{t})\\\ \vdots\\\ \nabla^{\top}z_{n,t}(\xi_{t})\end{bmatrix}\begin{bmatrix}\log\ell_{1,t}-\log\ell_{1,t+1}\\\ \vdots\\\ \log\ell_{n,t}-\log\ell_{n,t+1}\end{bmatrix}. $$  |   
---|---|---  
  
Since the losses satisfy $\ell_{i,t}=\ell_{t}$ for every $i$ (and similarly for $t+1$ by the symmetry of the update), the differences $\log\ell_{i,t}-\log\ell_{i,t+1}$ are the same for all tasks.

Moreover, the softmax function is given by

|  $$ z_{i,t}=\frac{\exp(\xi_{i,t})}{\sum_{j=1}^{N}\exp(\xi_{j,t})}. $$  |   
---|---|---  
  
The derivative of $z_{i,t}$ with respect to $\xi_{j}$ is

|  $$ \frac{\partial z_{i}}{\partial\xi_{j}}=z_{i}\left(\delta_{ij}-z_{j}\right). $$  |   
---|---|---  
  
where $\delta_{ij}$ is the Kronecker delta, defined as $\delta_{ij}=1$ if $i=j$ and 0 otherwise. At the symmetric point $\xi_{t}=(c_{t},\dots,c_{t})$ , we have $z_{i}=\frac{1}{N}$ for all $i$ , hence

|  $$ \frac{\partial z_{i}}{\partial\xi_{j}}=\frac{1}{N}\left(\delta_{ij}-\frac{1}{N}\right). $$  |   
---|---|---  
  
This shows that the Jacobian of the softmax is the same for every coordinate, ensuring that the scalar products involved in the definition of $\delta_{t}$ yield the same value for every task. Let us denote this common value by $D$ , i.e.,

|  $$ [\delta_{t}]_{i}=D\quad\text{for all }i. $$  |   
---|---|---  
  
Since the regularization term $\gamma\,\xi_{t}$ is also applied uniformly, the update for each coordinate becomes

|  $$ \xi_{i,t+1}=c_{t}-\beta(D+\gamma c_{t})\quad\text{for all }i. $$  |   
---|---|---  
  
Thus, $\xi_{t+1}=(c_{t+1},c_{t+1},\ldots,c_{t+1})$ remains symmetric. Evidently, the induction holds for the specific case $N=2$ .

∎

### A.5 Dataset Usage and Licensing

  * •

MNIST: Creative Commons Attribution-Share Alike 3.0

  * •

Cityscapes: Custom, allows for academic and non-commercial use, <https://www.cityscapes-dataset.com/license/>

  * •

QM9: We were not able to find a license for this dataset. However the GitHub repository that manages the dataset is under an MIT license <https://github.com/pyg-team/pytorch_geometric>.

### A.6 Hyperparameters

Tables 6, 7, 8, 9, and 10 summarize the best hyperparameters for each SMTO in each scenario. Also, it shows the best SMTO specific parameters in bold. In the MNIST configuration, these parameters were searched in a broader range to better understand their impact on optimization and limit the search space in the more complex dataset. To determine which specific hyperparameters of each SMTO to optimize, we selected those explicitly optimized in the respective articles. Consequently, we kept Auto-Lambda’s initial weight, FAMO’s learning rate for task logits, Nash-MTL’s optimizer iterations, and GradDrop’s parameters fixed throughout the experiments. Additionally, for the Cityscapes dataset, RotoGrad’s feature size was set to the maximum possible value or capped at 512, the same used for the ResNet18 model trained on the original Rotograd paper.

For the hyperparameters that were optimized without a defined range, such as Auto-Lambda’s and RotoGrad’s auxiliary learning rate, we opted to define it as a scale of the learning rate ( $aux\\_lr=lr*lr\\_scale$ ).

For the specific case of the CDTT optimizer in the QM9 dataset, we opted to use the same optimal learning rate from EDM and only optimized specific hyperparameters due to time constraints. We justify this choice because CDTT was proposed as a direct improvement from EDM.

Table 6: Hyperparameters for MNIST - CL & CR. Dataset | Tasks | SMTO | Hyperparameters | SMTO specific parameters  
---|---|---|---|---  
LR | P | WD  
MNIST |  CL CR | Single Task |  0.0025 0.005 |  0.5 0.3 |  0 0 |   
|  | Unit. Scal. | 0.0025 | 0.3 | 0 |   
|  | Auto-Lambda | 0.005 | 0.3 | 0 |  auxiliary learning rate scale: {1000, 100, 10 1.0 0.1 0.01 0.001} initial weight: 0.1  
|  | CAGrad | 0.0025 | 0.3 | 0 |  $c$ : {0.2, 0.5, 0.8}  
|  | CDTT | 0.005 | 0.2 | 0 |  $\alpha$ : {0.2, 0.4, 0.6, 0.8, 1.0}  
|  | EDM | 0.005 | 0.2 | 0 |   
|  | FAMO | 0.0025 | 0.2 | 0 |  $\gamma$ : {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025  
|  | IMTL | 0.0025 | 0.3 | 0 |   
|  | Nash-MTL | 0.0025 | 0.3 | 0 | optimizer iterations: 20  
|  | RotoGrad | 0.005 | 0.3 | 0 |  auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 50  
Table 7: Hyperparameters for MNIST - CL & RR. Dataset | Tasks | SMTO | Hyperparameters | SMTO specific parameters  
---|---|---|---|---  
LR | P | WD  
MNIST |  CL RR | Single Task |  0.0025 0.0025 |  0.5 0.0 |  0 0 |   
|  | Unit. Scal. | 0.001 | 0 | 0 |   
|  | Auto-Lambda | 0.001 | 0 | 0 |  auxiliary learning rate scale: {1000 100 10 1 0.1 0.01 0.001} initial weight: 0.1  
|  | CAGrad | 0.001 | 0 | 0 |  $c$ : {0.2, 0.5, 0.8}  
|  | CDTT | 0.0025 | 0 | 0 |  $\alpha$ : {0.2, 0.4, 0.6, 0.8, 1.0}  
|  | EDM | 0.001 | 0 | 0 |   
|  | FAMO | 0.0025 | 0 | 0 |  $\gamma$ : {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025  
|  | GradDrop | 0.001 | 0 | 0 |  k: 1.0 p: 0.5  
|  | IMTL | 0.001 | 0 | 0 |   
|  | MGDA-UB | 0.001 | 0 | 0 |   
|  | Nash-MTL | 0.001 | 0 | 0 | optimizer iterations: 20  
|  | PCGrad | 0.001 | 0 | 0 |   
|  | RLW-Dirichlet | 0.001 | 0 | 0 |   
|  | RLW-Normal | 0.001 | 0 | 0 |   
|  | RotoGrad | 0.001 | 0 | 0 |  auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 50  
|  | SI | 0.001 | 0 | 0 |   
|  | UW | 0.001 | 0 | 0 |   
Table 8: Hyperparameters for MNIST - RL & RR. Dataset | Tasks | SMTO | Hyperparameters | SMTO specific parameters  
---|---|---|---|---  
LR | P | WD  
MNIST |  RL RR | Single Task |  0.0025 0.0025 | 0 | 0 |   
|  | Unit. Scal. | 0.001 | 0 | 0 |   
|  | Auto-Lambda | 0.001 | 0 | 0 |  auxiliary learning rate scale: {1000 100 10 1 0.01 0.001 0.0001} initial weight: 0.1  
|  | CAGrad | 0.0025 | 0 | 0 |  $c$ : {0.2, 0.5, 0.8}  
|  | CDTT | 0.001 | 0 | 0 |  $\alpha$ : {0.2, 0.4, 0.6, 0.8, 1.0}  
|  | EDM | 0.001 | 0 | 0 |   
|  | FAMO | 0.001 | 0 | 0 |  $\gamma$ : {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025  
|  | IMTL | 0.001 | 0 | 0 |   
|  | Nash-MTL | 0.001 | 0 | 0 | optimizer iterations: 20  
|  | RotoGrad | 0.001 | 0 | 0 |  auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 50  
Table 9: Hyperparameters for Cityscapes. Dataset | Tasks | SMTO | Hyperparameters | SMTO specific parameters  
---|---|---|---|---  
LR | P | WD  
City. |  S D I | Single Task |  ACC: 0.001 mIoU: 0.001 L1 abs: 0.001 L1 rel: 0.0005 L1 abs: 0.001 |  0 0 0 0 0 |  $1e-4$  $1e-4$  $1e-4$  $1e-4$  $1e-5$  |   
|  | Unit. Scal. | 0.001 | 0 |  $1e-4$  |   
|  | Auto-Lambda | 0.001 | 0 |  $1e-4$  |  auxiliary learning rate scale: {100 10 1 0.1 0.01 0.001 0.0001} initial weight: 0.1  
|  | CAGrad | 0.001 | 0 |  $1e-4$  |  $c$ : {0.2, 0.5, 0.8}  
|  | CDTT | 0.001 | 0 |  $1e-4$  |  $\alpha$ : {0.2, 0.4, 0.6, 0.8, 1.0}  
|  | EDM | 0.001 | 0 |  $1e-4$  |   
|  | FAMO | 0.001 | 0 |  $1e-4$  |  $\gamma$ : {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025  
|  | IMTL | 0.001 | 0 |  $1e-4$  |   
|  | Nash-MTL | 0.001 | 0 |  $1e-4$  | optimizer iterations: 20  
|  | RotoGrad | 0.001 | 0 | 0 |  auxiliary learning rate scale: {5.0 1.0 0.5 0.1} feature size: 512  
Table 10: Hyperparameters for QM9. Dataset | Tasks | SMTO | Hyperparameters | SMTO specific parameters  
---|---|---|---|---  
LR | P | WD  
QM9 |  $C_{v}$  $G$  $H$  $U_{0}$  $U$  $\alpha$  $\epsilon_{homo}$  $\epsilon_{lumo}$  $\mu$  $R^{2}$  ZPVE | Single Task |  0.001 0.001 0.001 0.0005 0.001 0.005 0.001 0.001 0.001 0.001 0.001 |  0 0 0 0 0 0 0 0 0 0 0 |  0 0 0 0 0 0 0 0 0 0 0 |   
|  | Unit. Scal. | 0.005 | 0 | 0 |   
|  | Auto-Lambda | 0.001 | 0 | 0 |  auxiliary learning rate scale: {100, 10, 1.0} initial weight: 0.1  
|  | CAGrad | 0.001 | 0 | 0 |  $c$ : {0.2, 0.5, 0.8}  
|  | CDTT | 0.001 | 0 | 0 |  $\alpha$ : {0.2, 0.4, 0.6, 0.8, 1.0}  
|  | EDM | 0.001 | 0 | 0 |   
|  | FAMO | 0.005 | 0 | 0 |  $\gamma$ : {0.01, 0.001, 0.0001} learning rate of the task logits: 0.025  
|  | Nash-MTL | 0.001 | 0 | 0 | optimizer iterations: 20