# In Defense of the Unitary Scalarization   
for Deep Multi-Task Learning

Vitaly Kurin ††thanks: Equal contribution.  Affiliation: University of Oxford  Email: [vitaly.kurin@cs.ox.ac.uk](<mailto:>) Alessandro De Palma11footnotemark:  1 Affiliation: University of Oxford  Email: [adepalma@robots.ox.ac.uk](<mailto:>) Ilya Kostrikov  Affiliation: University of California, Berkeley  Affiliation: New York University  Shimon Whiteson  Affiliation: University of Oxford  M. Pawan Kumar  Affiliation: University of Oxford 

###### Abstract

Recent multi-task learning research argues against _unitary scalarization_ , where training simply minimizes the sum of the task losses. Several ad-hoc multi-task optimization algorithms have instead been proposed, inspired by various hypotheses about what makes multi-task settings difficult. The majority of these optimizers require per-task gradients, and introduce significant memory, runtime, and implementation overhead. We show that unitary scalarization, coupled with standard regularization and stabilization techniques from single-task learning, matches or improves upon the performance of complex multi-task optimizers in popular supervised and reinforcement learning settings. We then present an analysis suggesting that many specialized multi-task optimizers can be partly interpreted as forms of regularization, potentially explaining our surprising results. We believe our results call for a critical reevaluation of recent research in the area.

## 1 Introduction

Multi-Task Learning (MTL) [5] exploits similarities between tasks to yield models that are more accurate, generalize better and require less training data. Owing to the success of MTL on traditional machine learning models [22, 3, 16] and of deep single-task learning across a variety of domains, a growing body of research has focused on deep MTL. The most straightforward way to train a neural network for multiple tasks at once is to minimize the sum of per-task losses. Adopting terminology from multi-objective optimization, we call this approach _unitary scalarization_.

While some work shows that multi-task networks trained via unitary scalarization exhibit superior performance to independent per-task models [35, 29], others suggest the opposite [58, 30, 54]. As a result, many explanations for the difficulty of MTL have been proposed, each motivating a new Specialized Multi-Task Optimizer (SMTO) [54, 42, 66, 11, 62]. These works typically claim that the proposed SMTO outperforms unitary scalarization, in addition to relevant prior work. However, SMTOs usually require access to per-task gradients either with respect to the shared parameters, or to the shared representation. Therefore, their reported performance gain comes at significant computation and memory cost, the overhead scaling linearly with the number of tasks. By contrast, unitary scalarization requires only the average of the gradients across tasks, which can be computed via a single backpropagation.

Existing SMTOs were introduced to solve challenges related to the optimization of the deep MTL problem. We instead postulate that the reported weakness of unitary scalarization is linked to experimental variability or to a lack of regularization, leading to the following contributions:

  * •

A comprehensive experimental evaluation ( $\lx@sectionsign$ 4) of recent SMTO s on popular multi-task benchmarks, showing that no SMTO consistently outperforms unitary scalarization in spite of the added complexity and overhead. In particular, either the differences between unitary scalarization and SMTOs are not statistically significant, or they can be bridged by standard regularization and stabilization techniques from the single-task literature. Our reinforcement learning (RL) experiments include optimizers previously applied only to supervised learning.

  * •

An empirical and technical analysis of the considered SMTOs, suggesting that they reduce overfitting on the multi-task problem and hence act as regularizers ( $\lx@sectionsign$ 5). We conduct an ablation study and provide a collection of novel and existing technical results that support this hypothesis.

  * •

Code to reproduce the experiments, including a unified PyTorch [50] implementation of the considered SMTOs, is available at <https://github.com/yobibyte/unitary-scalarization-dmtl>.

We believe that our results suggest that the considered SMTOs can be often replaced by less expensive techniques. We hope that these surprising results stimulate the search for a deeper understanding of MTL.

## 2 Related Work

Before diving into details of specific SMTOs in Section 5, we provide a high-level overview of the deep MTL research. Seminal work in MTL includes _hard parameter sharing_ [6]: sharing neural network parameters between all tasks with, possibly, a separate part of the model for each task. Hard parameter sharing is still the major MTL approach adopted in natural language processing [12, 9], computer vision [46], and speech recognition [53]. In this work, we implicitly assume that each parameter update employs information from all tasks. However, not all works satisfy this assumption, either due to a large number of tasks [4, 36], or simply as an implementation decision [25, 37]. In this setting, MTL resembles other problems dealing with multiple tasks, i.e., continual [32], curriculum [47], and meta-learning [24], which are not the focus of this work.

Many works strive to improve the performance of deep multi-task models. One line of research hypothesizes that conflicting per-task gradient directions lead to suboptimal models, and focuses on explicitly removing such conflicts [66, 11, 42, 62, 28, 41]. Some authors postulate that loss imbalances across tasks hinder learning, proposing loss reweighting methods [30, 10, 40]. Sener and Koltun [54] and Navon et al. [48] propose that tasks compete for model capacity and interpret MTL as multi-objective optimization in order to cope with inter-task competition. Here, we focus on algorithms that explicitly rely on per-task gradients to try to outperform unitary scalarization ( $\lx@sectionsign$ 5). Research on multi-task architectures [46, 19] or MTL algorithms exclusively motivated by deterministic loss reweighting [30, 18, 43] are orthogonal to our work. Both topics are investigated by a recent survey on pixel-level multi-task computer vision problems [61], which found that the minimization of tuned weighted sums of losses (scalarizations) is empirically competitive with deterministic loss reweighting and MGDA in the considered settings. These results are extended to popular SMTOs by a critical review from Xin et al. [63], concurrent to our work, which argues that the optimization and generalization performance of SMTOs can be matched by tuning scalarization coefficients. Our work reaches a similar conclusion, demonstrating that unitary scalarization performs on par with SMTOs when coupled with standard and inexpensive regularization or stabilization techniques. In other words, Xin et al. [63] provide complementary support for the link between SMTOs and regularization by showing that tuning scalarization weights positively affects generalization.

In addition to the common supervised settings, we also consider multi-task RL, whose research can be grouped into three categories: the first adds auxiliary tasks providing additional inductive biases to speed up learning [27] on a target task. The second, based on policy distillation, uses per-task teacher models to provide labels for a multi-task model or per-task policies as regularizers [51, 49, 57]. The third directly learns a shared policy [29], possibly via an SMTO [66]. We focus on the third category, whose literature reports varying performance for unitary scalarization (better [29] or worse [66] than per-task models), indicating confounding factors in evaluation pipelines and further motivating our work. PopArt [60, 23] performs scale-invariant value function updates in order to address differences in returns across environments, showing improvements in the multi-task setting while still using unitary scalarization. PopArt does not require per-task gradients but introduces additional hyperparameters. In our work, we address the differences in rewards by normalizing them at the replay buffer level. However, we believe both unitary scalarization and SMTOs might equally benefit from PopArt.

## 3 Multi-Task Learning Optimizers

We will now describe the deep MTL training problem and popular algorithms employed for its solution. Let $(X,Y)\in\mathbb{R}^{d\times n}\times\mathbb{R}^{o\times n}$ be the training set, composed of $n$ $d$ -dimensional points and $o$ -dimensional labels. In addition, $\mathcal{L}_{i}:\mathbb{R}^{o\times n}\times\mathbb{R}^{o\times n}\rightarrow\mathbb{R}$ denotes the loss for the $i$ -th task, $\bm{\theta}\in\mathbb{R}^{S}$ the parameter space, $\mathcal{T}:=\\{1,\dots,m\\}$ the set of $m$ tasks. The goal of MTL is to learn a single (generally task-aware) parametrized model $f:\mathbb{R}^{S}\times\mathbb{R}^{d\times n}\times\mathcal{T}\rightarrow\mathbb{R}^{o\times n}$ that performs well on all tasks $\mathcal{T}$ . The parameter space is often split into a set of shared parameters across tasks (generally the majority of the architecture), denoted $\bm{\theta_{\parallel}}$ , and (possibly empty) task-specific parameters, denoted $\bm{\theta_{\perp}}$ , so that $\bm{\theta}:=[\bm{\theta_{\parallel}},\bm{\theta_{\perp}}]^{T}$ . In this context, the model $f$ often takes on an encoder-decoder architecture, where the encoder $g$ learns a shared representation across tasks, and the decoders $h_{i}$ are task-specific predictive heads: $f(\bm{\theta},X,i)=h_{i}(g(\bm{\theta_{\parallel}},X),\bm{\theta_{\perp}})$ . In this case, we denote by $\mathbf{z}=g(\bm{\theta_{\parallel}},X)\in\mathbb{R}^{r\times n}$ the $r$ -dimensional shared representation of $X$ .

The training problem for MTL is typically formulated as the sum of the per-task losses [54, 66, 11]:

|  $$ \smash{\min_{\bm{\theta}}}\left[\begin{array}[]{l}\mathcal{L}^{\text{MT}}(\bm{\theta}):=\sum_{i\in\mathcal{T}}\mathcal{L}_{i}(f(\bm{\theta},X,i),Y)\end{array}\right]. $$  |  | (1)  
---|---|---|---  
  
##### Unitary Scalarization

The obvious way to minimize the multi-task training objective in equation eq. 1 is to rely on a standard gradient-based algorithm. While, for simplicity, we focus on standard gradient descent rather than mini-batch stochastic gradient descent, the notation can be adapted by replacing the dataset size $n$ by the mini-batch size $b$ . Equation eq. 1 corresponds to a linear scalarization with unitary weights under a multi-objective interpretation of MTL; hence, we call the direct application of gradient descent on equation eq. 1 _unitary scalarization_. For vanilla gradient descent, this corresponds to taking a step in the opposite direction as the one given by the sum of per-task gradients: $\nabla_{\bm{\theta}}\mathcal{L}^{\text{MT}}=\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta}}\mathcal{L}_{i}$ . Per-task gradients are not required, as it suffices to directly compute the gradient of the sum $\mathcal{L}^{\text{MT}}$ . Hence, when relying on deep learning frameworks based on reverse-mode differentiation, such as PyTorch [50], the backward pass is performed once per iteration (rather than $m$ times). Furthermore, the memory cost is a factor $m$ less than most SMTO s, which require access to each $\nabla_{\bm{\theta}}\mathcal{L}_{i}$ . As a consequence, unitary scalarization is simple, fast, and memory efficient. Our experiments demonstrate that, when possibly coupled with single-task regularization such as early stopping, $\ell_{2}$ penalty or dropout layers [56], this simple optimizer is strongly competitive with SMTOs.

##### MGDA

Sener and Koltun [54] point out that equation eq. 1 can be cast as a multi-objective optimization problem with the following objective: $\bm{\mathcal{L}}^{\text{MT}}(\bm{\theta}):=[\mathcal{L}_{1}(\bm{\theta}),\dots,\mathcal{L}_{m}(\bm{\theta})]^{T}$ . A commonly employed solution concept in multi-objective optimization is Pareto optimality. A point $\bm{\theta}^{*}$ is called Pareto-optimal if, for any another point $\bm{\theta}^{\dagger}$ such that $\exists i\in\mathcal{T}:\mathcal{L}_{i}(\bm{\theta}^{\dagger})<\mathcal{L}_{i}(\bm{\theta}^{*})$ , then $\exists j\in\mathcal{T}:\mathcal{L}_{j}(\bm{\theta}^{\dagger})>\mathcal{L}_{j}(\bm{\theta}^{*})$ . A necessary condition for Pareto optimality at a point is Pareto stationarity, defined as the lack of a shared descent direction across all losses at that point. Sener and Koltun [54] rely on Multiple-Gradient Descent Algorithm (MGDA) [14] to reach a Pareto-stationary point for shared parameters $\bm{\theta_{\parallel}}$ . Intuitively, MGDA proceeds by repeatedly stepping in a shared descent direction [17, 14], which can be found by solving the following optimization problem:

|  $$ \min_{\mathbf{g},\epsilon}\left[\epsilon+\nicefrac{{1}}{{2}}\left\lVert\mathbf{g}\right\rVert_{2}^{2}\right]\quad\text{s.t. }\ \nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}^{T}\mathbf{g}\leq\epsilon\quad\forall\ i\in\mathcal{T}, $$  |  | (2)  
---|---|---|---  
  
whose dual takes the following form (corresponding to the formulation from Désidéri [14]):

|  $$ \max_{\bm{\alpha}\geq 0}-\nicefrac{{1}}{{2}}\left\lVert\mathbf{g}\right\rVert_{2}^{2}\quad\text{s.t. }\ \sum_{i}\alpha_{i}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=-\mathbf{g},\quad\sum_{i\in\mathcal{T}}\alpha_{i}=1. $$  |  | (3)  
---|---|---|---  
  
In other words, MGDA takes a step in a direction $\mathbf{g}$ given by the negative convex combination of per-task gradients, whose coefficients are given by solving equation eq. 3. In practice, per-task gradients are rescaled before applying MGDA: the original authors’ implementation [54] relies on $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\leftarrow\nicefrac{{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}}{{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert\mathcal{L}_{i}(\bm{\theta})}}$ . The convergence of MGDA to a Pareto-stationary point is still guaranteed after normalization [14].

##### IMTL

Impartial Multi-Task Learning (IMTL) [42] is presented as an SMTO that is not biased against any single task. It is composed of two complementary algorithmic blocks: IMTL-L, acting on task losses, and IMTL-G, acting on per-task gradients. IMTL-G follows the intuition that a multi-task optimizer should proceed along a direction $\mathbf{g}=-\sum_{i}\alpha_{i}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ that equally represents per-task gradients. This is formulated analytically by requiring that the cosine similarity between $\mathbf{g}$ and each $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ be the same. To prevent the resulting problem from being underdetermined, Liu et al. [42] add the constraint $\sum_{i\in\mathcal{T}}\alpha_{i}=1$ , resulting in a problem that admits a closed-form solution for $\mathbf{g}$ :

|  $$ \begin{array}[]{l}\mathbf{g}^{T}\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert}=\mathbf{g}^{T}\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}\enskip\forall\ i\in\mathcal{T}\setminus\\{1\\},\quad\mathbf{g}=-\sum_{i}\alpha_{i}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i},\quad\sum_{i\in\mathcal{T}}\alpha_{i}=1.\end{array} $$  |  | (4)  
---|---|---|---  
  
IMTL-L, instead, aims to reweight task losses so that they are all constant over time, and equal to $1$ . In order to limit oscillations of the scaling factors, the authors propose to learn them jointly with the network by minimizing a common objective via gradient descent. In particular, given $s_{i}\in\mathbb{R}\ \forall i\in\mathcal{T}$ , Liu et al. [42] derive the following form for the joint minimization problem: $\min_{\mathbf{s},\bm{\theta}}\left[\sum_{i}\left(e^{s_{i}}\mathcal{L}_{i}(\bm{\theta})-s_{i}\right)\right].$ As proved by Liu et al. [42], IMTL-L only has a rescaling effect on the update direction of IMTL-G. Unlike IMTL-G and the other SMTOs presented in this section, IMTL-L rescaling is designed to affect the updates for task-specific parameters $\bm{\theta_{\perp}}$ as well.

##### PCGrad

Let us write $\cos(\mathbf{x},\mathbf{z})$ for the cosine similarity between vectors $\mathbf{x}$ and $\mathbf{z}$ . Yu et al. [66] postulate that multi-task convergence is severely slowed down if the following three conditions (named the _tragic triad_) hold at once: (i) conflicting gradient directions: $\cos(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j})<0$ for some $i,j\in\mathcal{T}$ ; (ii) differing gradient magnitudes: $\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert\gg\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert$ for some $i,j\in\mathcal{T}$ ; and (iii) the unitary scalarization $\mathcal{L}^{\text{MT}}$ has high curvature along $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{MT}}$ . The PCGrad [66] SMTO is presented as a solution to the tragic triad, targeted at the first condition. Consistent with the previous sections, let us denote the update direction by $\mathbf{g}$ . Furthermore, let $[\mathbf{x}]_{+}:=\max(\mathbf{x},\mathbf{0})$ . Given per-task gradients $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ , PCGrad iteratively projects each task gradient onto the normal plane of all the gradients with which it conflicts:

|  $$ \left[\begin{array}[]{l}\hskip-4.0pt\mathbf{g}_{i}\leftarrow\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i},\enskip\mathbf{g}_{i}\leftarrow\mathbf{g}_{i}+\hskip-3.0pt\left[\frac{-\mathbf{g}_{i}^{T}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}(\mathbf{x})}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert^{2}}\right]_{+}\hskip-7.0pt\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\enskip\forall j\in\mathcal{T}\setminus\\{i\\}\end{array}\hskip-5.0pt\right]\forall i\in\mathcal{T},\quad\mathbf{g}=-\sum_{i\in\mathcal{T}}\mathbf{g}_{i}, $$  |  | (5)  
---|---|---|---  
  
where the iterative updates of $\mathbf{g}_{i}$ with respect to $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}$ are performed in random order.

##### GradDrop

Chen et al. [11] focus on conflicting signs across task gradient entries, arguing that such conflicts lead to gradient “tug-of-wars". The GradDrop SMTO [11], presented as a solution to this problem, proposes to randomly mask per-task gradients $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ so as to minimize such conflicts. Specifically, GradDrop computes the “positive sign purity" $p_{j}$ for the task gradient’s $j$ -th entry and then masks the $j$ -th entry of each per-task gradient with probability increasing with $p_{j}$ , if the entry is negative, or decreasing with $p_{j}$ , if the entry is positive. Let us write $\mathbf{p}:=[p_{1},\dots,p_{S}]$ , where $S$ is the dimensionality of the parameter space (see $\lx@sectionsign$ 3), $\odot$ for the Hadamard product and $\mathds{1}_{\mathbf{a}}$ for the indicator vector on condition $\mathbf{a}$ . Given a vector $\mathbf{u}_{i}$ , uniformly sampled in $[\mathbf{0},\mathbf{1}]$ at each iteration, GradDrop takes a step in the direction given by:

|  $$ \mathbf{g}=\sum_{i\in\mathcal{T}}\left(\begin{array}[]{l}-\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\odot\mathds{1}_{\left(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}>0\right)}\odot\mathds{1}_{\left(\mathbf{u}_{i}>\mathbf{p}\right)}\\\ -\ \nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\odot\mathds{1}_{\left(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}<0\right)}\odot\mathds{1}_{\left(\mathbf{u}_{i}<\mathbf{p}\right)}\end{array}\right)\text{, with }\enskip\mathbf{p}=\frac{1}{2}\left(1+\frac{\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\sum_{i\in\mathcal{T}}\left|\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right|}\right). $$  |  | (6)  
---|---|---|---  
  
## 4 Experimental Evaluation

Relying on a unified experimental pipeline, we present an empirical evaluation on common MTL benchmarks of unitary scalarization ( $\lx@sectionsign$ 3), of the popular SMTOs presented in $\lx@sectionsign$ 3, and of the recent RLW algorithms [40] due to their similarities with PCGrad and GradDrop (see $\lx@sectionsign$ 5.2). We benchmark against the two RLW instances that showed the best average performance in the original paper: RLW with weights sampled from a Dirichlet distribution (“RLW Diri.”), and RLW with weights sampled from a Normal distribution (“RLW Norm.”). The goal of this section is to assess the efficacy of a popular line of previous work, focusing on a few representative or well-established optimizers. Therefore, we forego comparison with more recent SMTOs [48, 28, 41]. Nevertheless, we point out that these algorithms often lack significant enough improvements over the optimizers we consider, or may have substantial commonalities with them (see $\lx@sectionsign$ 5.2 for Nash-MTL [48], which was published concurrently to the finalization of this work). Whenever appropriate, we employ “Unit. Scal.” as shorthand for unitary scalarization. We first present supervised learning experiments ( $\lx@sectionsign$ 4.1), and then evaluate on a popular reinforcement learning benchmark ( $\lx@sectionsign$ 4.2).

Our experiments indicate that the performance of unitary scalarization has been consistently underestimated in the literature. By showing the variability between runs and by relying on standard regularization and stabilization techniques from the single-task literature, we demonstrate that _no SMTO consistently outperforms unitary scalarization across the considered settings_. This result holds in spite of the added complexity and computational overhead associated with most SMTO s. We provide a potential explanation of our results in $\lx@sectionsign$ 5.

### 4.1 Supervised Learning

All the architectures employed in the supervised learning experiments conform to the encoder-decoder structure detailed in $\lx@sectionsign$ 3. Whenever suggested by the original authors for this context, the SMTO implementations rely on per-task gradients with respect to the last shared activation, $\nabla_{\mathbf{z}}$ , rather than on the usually more expensive $\nabla_{\bm{\theta}}\mathcal{L}_{i}$ . In particular, this is the case for MGDA, IMTL and GradDrop. See appendix B for details concerning each individual algorithm. Surprisingly, several MTL works [66, 11, 42, 40] report validation results, making it easier to overfit. Instead, following standard machine learning practice, we select a model on the validation set, and later report test metrics for all benchmarks. Validation results are also available in appendix D. Appendix C.1 reports dataset descriptions, the computational setup, hyperparameter and tuning details.

#### 4.1.1 Multi-MNIST

(a) Avg. task test accuracy: mean and 95 $\%$ CI (10 runs).

(b) Box plots for the training time of an epoch (10 runs).

Figure 1:  No algorithm outperforms unitary scalarization on the Multi-MNIST dataset.

We present results on the Multi-MNIST [54] dataset, a simple two-task supervised learning benchmark. We employ a popular architecture from previous work [54, 66] (see appendix C.1), where a single dropout layer [56] (with dropout probability $0.5$ ) is employed in both the encoder and the decoder. $\ell_{2}$ regularization did not improve validation performance and was therefore omitted. Figure 1 reports the average task test accuracy, and the training time per epoch. For each run, the test model was selected as the model with the largest average task validation accuracy across the training epochs. Appendix D presents the results of Figure 1 in tabular form, as well as the average task validation accuracy per epoch. As seen from the overlapping confidence intervals, none of the considered algorithms clearly outperforms the others. However, GradDrop displays higher experimental variability. Finally, Figure 1(b) shows that unitary scalarization also has among the lowest training times.

#### 4.1.2 CelebA

(a) Avg. task test accuracy: mean and 95 $\%$ CI (3 runs).

(b) Box plots for the training time of an epoch (10 runs).

Figure 2:  While SMTOs display larger runtimes, none of them outperforms the unitary scalarization on the CelebA dataset.

We now show results for the CelebA [44] dataset, a challenging $40$ -task multi-label classification problem. We employ the same architecture as many previous studies [54, 66, 40, 42] (see appendix C.1). We tuned $\ell_{2}$ regularization terms $\lambda$ for all SMTOs in the following grid: $\lambda\in\\{0,10^{-4},10^{-3}\\}$ . The best validation performance was attained with $\lambda=10^{-3}$ for unitary scalarization, IMTL and PCGrad, and with $\lambda=10^{-4}$ for MGDA, GradDrop, and RLW. Validation performance was further stabilized by the addition of several dropout layers (see Figure 6), with dropout probabilities from $0.25$ to $0.5$ . We present an ablation study on the effect of regularization on this experiment in $\lx@sectionsign$ 5.1. Figure 10 (appendix D.2) shows that regularization improves the peak average validation performance for all the considered methods. Analogously to our Multi-MNIST results, Figure 2 plots the distribution of the training time per epoch, and the average test task accuracy. As with Multi-MNIST, the test model for each run was the one with maximal average validation task accuracy across epochs. In other words, if the peak is attained before the last epoch, we perform early stopping: as shown in Figure 8(a) in appendix D this is the case for most methods. Due to the large number of tasks, Figure 2(b) shows relatively large runtime differences across methods. PCGrad is the slowest (roughly $35$ times slower than unitary scalarization). In fact, amongst the considered algorithms, it is the only one that computes per-task gradients over the parameters ( $\nabla_{\bm{\theta}}\mathcal{L}_{i}\ \forall i\in\mathcal{T}$ ) at each iteration. GradDrop, MGDA and IMTL have overhead factors (compared to unitary scalarization) ranging from roughly $1.05$ to $2.4$ due to the relatively small size of $\mathbf{z}$ for the employed architecture. The overhead of RLW is negligible: roughly $5\%$ . Nevertheless, due to largely overlapping confidence intervals in Figure 2(a), none of the methods consistently outperforms unitary scalarization. In fact, owing to our adoption of explicit regularization techniques (see $\lx@sectionsign$ 5.1) its average performance is superior to that reported in the literature [54, 42].

#### 4.1.3 Cityscapes

(a) Absolute depth test error: lower is better.

(b) Relative depth test error: lower is better.

(c) Test segmentation mIOU: higher is better.

(d) Test segmentation accuracy: higher is better.

(e) Box plots for the training time of an epoch (10 runs).

Figure 3:  On Cityscapes, none of the SMTOs outperforms unitary scalarization, which proves to be the most cost-effective algorithm. Subfigures 3(a)-3(d) report means for three runs, and their 95 $\%$ CIs. 

In order to complement the multi-task classification experiments for Multi-MNIST and CelebA, we present results for Cityscapes [13], a dataset for semantic understanding of urban street scenes. We rely on a common encoder architecture from the literature [42, 40] (see appendix C.1), with a single dropout layer in the task-specific heads [40]. As for CelebA, unitary scalarization, IMTL, and PCGrad benefit from more regularization than the other optimizers: we employ $\lambda=10^{-5}$ for these three algorithms, as it resulted in better validation performance on the majority of metrics, and $\lambda=0$ for the remaining methods. Cityscapes is a heterogeneous MTL problem: it contains tasks of different types whose validation metrics cannot be averaged to perform model selection. Considering the lack of an established procedure in this context, we potentially evaluate a different model for each metric, chosen as the one with the best (maximal or minimal, depending on the metric) validation performance across epochs (we perform per-run early stopping). This procedure maximizes per-task performance, at the cost of increased inference time. If inference time is a priority, an alternative model selection procedure could rely on relative task improvement [28, 48, 41], assuming that per-metric improvements are to be weighted linearly. Nevertheless, any consistently applied model selection scheme serves the main goal of our work: evaluating all SMTOs on a fair ground. Figure 3 shows test results for two metrics per task, and the distribution of the training time per epoch. As with Multi-MNIST and CelebA, no training algorithm clearly outperforms unitary scalarization (significant overlaps across confidence intervals exist), which is again the least expensive method. In contrast with a popular hypothesis [30, 10, 42], this holds in spite of relatively large loss imbalances. In fact, the loss for the depth task is roughly $10$ times smaller than that of the segmentation task: see figures 17(e)-17(f). Unlike CelebA (see Figure 2(b)), IMTL, MGDA and GradDrop are significantly slower than unitary scalarization (factors from $1.6$ to $2.3$ ), due to the relatively (compared to the parameter space) large size of $\mathbf{z}$ in the employed architecture. PCGrad, instead, appears to be less expensive ( $30\%$ more than the baseline), demonstrating the benefits of working on $\nabla_{\bm{\theta}}\mathcal{L}_{i}$ on this model.

### 4.2 Reinforcement Learning

For RL experiments, we use Meta-World [65] and the Soft Actor-Critic [20] implementation from [55]. Unlike $\lx@sectionsign$ 4.1, the employed network architecture (see appendix C.1) is fully shared across tasks. Therefore, all SMTO implementations for these experiments rely on per-task gradients with respect to network parameters $\nabla_{\bm{\theta}}\mathcal{L}_{i}$ (see $\lx@sectionsign$ 5). Among the SMTOs we consider, PCGrad is the only one developed with the RL setting in mind. For fairness and completeness, we add all the other SMTOs from the supervised learning experiments, and are the first to test these optimizers in the RL setting. To stabilize learning, we increase the replay buffer size, a well known technique in single-task RL, add actor $l_{2}$ regularization, and modify the reward normalization employed by Sodhani et al. [55]. The unitary scalarization performance reported by Yu et al. [66] is considerably lower than that of Sodhani et al. [55], which we believe is due to the lack of reward normalization in the former. Sodhani et al. [55] keep a moving average of rewards in the environment, with a hyperparameter controlling the speed of the moving average. As we show in Figure 16, the learning algorithm is sensitive to that hyperparameter. Moreover, such normalization might make similar transitions have drastically different rewards stored in the replay buffer. To alleviate these issues, we store the raw rewards in the buffer, and normalize only when a mini-batch is sampled.

(a) MT10 (10 repetitions).

(b) MT50 (10 repetitions).

(c) MT10 (10 repetitions).

(d) MT50 (10 repetitions).

Figure 4:  On Metaworld, none of the SMTOs significantly outperforms Unit. Scal., which is the least expensive method. Subfigures 4(a)-4(b) report mean and 95 $\%$ CI for the best (over the updates) average success rate. Subfigures 4(c)-4(d) show box plots for the training time of 10,000 updates. 

Figure 4 reports the best average success rate across the updates and the runtime for 10,000 updates. In addition to these summary statistics, reported for consistency with $\lx@sectionsign$ 4.1, the learning curves are shown in appendix E. Our MT10 (10 tasks) results in Figure 4(a) show that by stabilizing the baseline using standard RL techniques, unitary scalarization performs on par with other SMTOs, mirroring our findings in $\lx@sectionsign$ 4.1. This is in contrast with the previous literature, which reported that PCGrad outperforms unitary scalarization [66, 55]. Figure 4(b) presents results on MT50 (50 tasks): similarly to MT10, none of the SMTOs significantly outperforms unitary scalarization, with PCGrad’s average being slightly above unitary scalarization. We speculate that the stochastic loss rescaling performed by PCGrad (see Proposition 3) reduces the differences in task return scales, and expect that methods like PopArt [60] would have a similar effect without requiring access to per-task gradients. While we did not tune hyperparameters for MT50 (we employed those found for MT10), it would be much easier to do that for unitary scalarization due to its lower runtime. In fact, Figure 4(d) shows that a single unitary scalarization run takes roughly 15 hours, whereas PCGrad, MGDA and GradDrop require more than a week. Similarly to MT10, actor regularization pushes the average performance of unitary scalarization higher (see in appendix E.2). Overall, as in the supervised learning setting, unitary scalarization performs comparably to SMTOs despite being simpler and less demanding in both memory and compute. IMTL was unstable on this RL benchmark and all of the runs crashed due to numerical overflow. We hence omit IMTL results from the main body of the paper and show its results in Figure 13 in appendix E, which also describes a possible explanation. We hypothesize that the instability of IMTL is due to lack of bounds on scaling coefficients. See appendix C.2 for hyperparameter settings and ablation studies.

## 5 Regularization in Specialized Multi-Task Optimizers

The empirical results presented in $\lx@sectionsign$ 4 motivate the need to carefully analyze existing SMTOs. We make an initial attempt in this direction by viewing their effects through the lens of regularization. Let us define a regularizer as a technique to reduce overfitting [15]. We first show that the SMTOs considered in $\lx@sectionsign$ 4 empirically act as regularizers via an ablation study ( $\lx@sectionsign$ 5.1). We then take a closer look at their behavior, presenting technical results that support their alternative interpretation as regularizers ( $\lx@sectionsign$ 5.2). Finally, $\lx@sectionsign$ 5.3 provides additional empirical backing for some of the technical results. Unless otherwise stated, we assume that MTL methods apply only to $\bm{\theta_{\parallel}}$ and that standard gradient-based updates are employed for tasks-specific parameters $\bm{\theta_{\perp}}$ . We furthermore adopt the following shorthands: $\mathcal{L}_{i}(\bm{\theta})$ for $\mathcal{L}_{i}(f(\bm{\theta},X,i),Y)$ , and $\nabla_{\bm{\theta}}\mathcal{L}_{i}$ for $\nabla_{\bm{\theta}}\mathcal{L}_{i}(f(\bm{\theta},X,i),Y)$ .

### 5.1 Ablation Study

Figure 5: Mean and 95 $\%$ CI (3 runs) avg. task validation accuracy over epochs on CelebA. SMTOs postpone the onset of overfitting, mirroring the effect of $\ell_{2}$ regularization on unitary scalarization. 

Figure 6:  Mean and $95\%$ CI (3 runs) for $\left\lVert\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert_{2}$ on CelebA. MGDA and IMTL converge away from stationary points of unitary scalarization, indicating under-optimization. 

We repeat the experiment from $\lx@sectionsign$ 4.1.2 and remove explicit regularization: no dropout layers are added to the encoder-decoder architecture, and $\lambda=0$ for all optimizers. In addition, we examine the behavior of two different $\ell_{2}$ -regularized instances of unitary scalarization: $\lambda=10^{-4}$ for “Unit. Scal. $\ell_{2}$ ”, $\lambda=2\times 10^{-3}$ for “Unit. Scal. $\ell_{2}+$ ”. Figure 6 shows that SMTOs behave similarly to an $\ell_{2}$ -penalized unitary scalarization. Importantly, SMTOs delay overfitting, requiring less early stopping compared to unitary scalarization to obtain comparable performance. In other words, early stopping is sufficient for unitary scalarization to perform on par with SMTOs. Moreover, overfitting is further reduced by “Unit. Scal. Reg.”, which plots the regularized unitary scalarization from $\lx@sectionsign$ 4.1.2, with dropout layers and a weight decay of $\lambda=10^{-3}$ . Finally, Figure 9(a) shows that unregularized unitary scalarization and most SMTOs rapidly drive the training loss of each task towards its global optimum. This suggests that the main difficulty of MTL is not associated with the optimization of its training objective, but rather to incorporating adequate regularization. Additional results are presented in appendix D.2.

### 5.2 Technical Results

All the methods considered in $\lx@sectionsign$ 5.1 regularize more than unitary scalarization. While RLW was shown to reduce overfitting by the original authors [40, theorem 2], we now provide a collection of novel and existing technical results that potentially explain the regularizing behavior of each of the other algorithms, complementing the presentation from $\lx@sectionsign$ 3. In particular, we show that MGDA, IMTL and PCGrad have a larger convergence set than unitary scalarization, reducing the chances to land on sharp local minima [15]. Furthermore, GradDrop and PCGrad introduce significant stochasticity, which is often linked to the same effect [31, 34]. We hope these observations will steer further research.

##### MGDA

Let us denote the convex hull of a set $\mathcal{A}$ by $\text{Conv}(\mathcal{A})$ . We now recall a well-known property of MGDA [14] and relate it to the behavior of unitary scalarization.

###### Proposition 1.

The MGDA SMTO [54] converges to a superset of the convergence points of unitary scalarization. More specifically, it converges to any point $\bm{\theta_{\parallel}}^{*}$ such that: $\mathbf{0}\in\text{Conv}(\\{\nabla_{\bm{\theta_{\parallel}}^{*}}\mathcal{L}_{i}\ |\ i\in\mathcal{T}\\})$ .

See appendix B.1 for a simple proof. As a consequence of Proposition 1, MGDA does not necessarily reach a stationary point for $\mathcal{L}^{\text{MT}}$ (that is, a point for which $\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}$ ) or for any of the losses $\mathcal{L}_{i}$ ( $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}$ ). For example, any point $\bm{\theta_{\parallel}}$ for which two per-task gradients point in opposite directions is Pareto stationary. On account of the well-known [15] relationship between under-optimizing (e.g., early stopping [7, 39]) and overfitting, proposition 1 supports the interpretation of MGDA as a regularizer for equation eq. 1. Empirical evidence that MGDA under-optimizes is provided in $\lx@sectionsign$ 5.3, Figure 9(a), and Figure 6, which shows over-regularization. Proposition 1 can be trivially extended to the recent Nash-MTL, which shares the same convergence set [48, Theorem 5.4].

##### IMTL

We now show that aggregating per-task gradients so that their cosine similarity is the same (equation eq. 4) yields a constrained steepest-descent algorithm (Proposition 2). This view on the update step of IMTL leads to a novel analysis of its convergence points (corollary 1). Proofs can be found in appendix B.2. We will denote by $\text{Aff}(\mathcal{A})$ the affine hull of a set $\mathcal{A}$ .

###### Proposition 2.

IMTL by Liu et al. [42] updates $\bm{\theta_{\parallel}}$ by taking a step in the _steepest descent direction whose_ cosine similarity with per-task gradients is the same across tasks.

###### Corollary 1.

IMTL by Liu et al. [42] converges to a superset of the Pareto-stationary points for $\bm{\theta_{\parallel}}$ (and hence of the convergence points of the unitary scalarization). More specifically, it converges to any point $\bm{\theta_{\parallel}}^{*}$ such that: $\begin{array}[]{l}\mathbf{0}\in\text{Aff}\left(\left\\{\nicefrac{{\nabla_{\bm{\theta_{\parallel}}^{*}}\mathcal{L}_{i}}}{{\left\lVert\nabla_{\bm{\theta_{\parallel}}^{*}}\mathcal{L}_{i}\right\rVert}}\ |\ i\in\mathcal{T}\right\\}\right).\end{array}$ 

As seen for MGDA, corollary 1 implies that, even if the employed model $f$ has the capacity to reach the minimal loss on $\mathcal{L}^{\text{MT}}$ , IMTL may stop before reaching a stationary point. Recalling the relationship between under-optimizing and overfitting [15], this supports the interpretation of IMTL as a regularizer for equation eq. 1. This is empirically shown in $\lx@sectionsign$ 5.3, Figures 6, 9(a). In particular, unitary scalarization reaches the same average performance of IMTL but requires earlier stopping.

##### PCGrad

We provide an alternative characterization of the PCGrad update rule, highlighting its stochasticity in the context of its interpretation as loss rescaling [42, 40]. See appendix B.3 for a proof.

###### Proposition 3.

PCGrad is equivalent to a dynamic, and possibly stochastic, loss rescaling for $\bm{\theta_{\parallel}}$ . At each iteration, per-task gradients are rescaled as follows:

|  $$ \begin{array}[]{l}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\leftarrow\left(1+\sum_{j\in\mathcal{T}\setminus\\{i\\}}d_{ji}\right)\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i},\ d_{ji}\in\smash{\left[0,\frac{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}\right]}.\end{array} $$  |   
---|---|---  
  
Furthermore, if $|\mathcal{T}|>2$ , $d_{ji}$ is a random variable, and the above range contains its support.

The results from proposition 3 can be easily extended to GradVac [62], which generalizes PCGrad’s projection onto the normal vector to arbitrary target cosine similarities between per-task gradients. When $|\mathcal{T}|>2$ , PCGrad corresponds to a stochastic loss re-weighting. As such, PCGrad bears many similarities with Random Loss Weighting (RLW) [40]. RLW proposes to sample scalarization weights from standard probability distributions at each iteration, and proves that this leads the better generalization [40, theorem 2]. Indeed, it is well-known that adding noise to stochastic gradient estimations leads the optimization towards flatter minima, and that such minima may reduce overfitting [31, 34]. In line with the main technical results by Yu et al. [66], we now restrict our focus to two-task problems, which allow for an easy description of PCGrad’s convergence points. The result is largely based on [66, theorem 1]: we relax some of the assumptions and provide a proof in appendix B.3.

###### Corollary 2.

If $|\mathcal{T}|=2$ , PCGrad will stop at any point where $\cos(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2})=-1$ . Furthermore, if $\mathcal{L}_{1}$ and $\mathcal{L}_{2}$ are differentiable, and $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{MT}}$ is L-Lipschitz with $L>0$ , PCGrad with step size $t<\frac{1}{L}$ converges to a superset of the convergence points of the unitary scalarization.

Corollary 2 implies that, when $|\mathcal{T}|=2$ , PCGrad may under-optimize equation eq. 1 as MGDA and IMTL. In particular, if $\cos(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2})=-1$ , then $\mathbf{0}\in\text{Conv}(\\{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}\\})$ (see proposition 1). We believe that PCGrad’s stochasticity and enlarged convergence set potentially explain its regularizing effect.

##### GradDrop

While the motivation behind GradDrop is to avoid entry-wise gradient conflicts across tasks, the main property of the method is to drive the optimization towards “joint minima": points that are stationary for all the individual tasks at once [11, proposition 1]. In other words: $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}\ \forall\ i\in\mathcal{T}$ . While this property is desirable, we show that it holds beyond GradDrop, and independently of the gradient directions. Under strong assumptions on the model capacity, the above property would trivially hold for unitary scalarization (proposition 5, appendix B.4). Proposition 4 shows that it holds for a simple randomized version of unitary scalarization, which we name Random Grad Drop (RGD).

###### Proposition 4.

Let $\mathcal{L}^{\text{RGD}}(\bm{\theta_{\parallel}}):=\sum_{i\in\mathcal{T}}u_{i}\mathcal{L}_{i}(\bm{\theta_{\parallel}})$ , where $u_{i}\sim\text{Bernoulli}(p)\ \forall i\in\mathcal{T}$ and $p\in(0,1]$ . The gradient $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{RGD}}$ is always zero if and only if $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}\ \forall i\in\mathcal{T}$ . In other words, the result from [11, proposition 1] can be obtained without any information on the sign of per-task gradients.

Proposition 4 (see appendix B.4 for a simple proof) shows that an inexpensive sign-independent stochastic scalarization shares GradDrop’s main reported property. $\mathcal{L}^{\text{RGD}}$ can be directly cast an instance of RLW, and hence as a regularization method [31, 34]. Furthermore, Figure 12 in appendix D.3 shows that the empirical results of GradDrop on CelebA [44] are closely matched by a sign-agnostic gradient masking, partly undermining the conflicting gradients assumption. We believe that the above results, along with the authors’ original experiments showing that GradDrop delays overfitting on CelebA [11, figure 3], suggest that GradDrop behaves as a regularizer.

### 5.3 Under-Optimization: Empirical Study

As seen in $\lx@sectionsign$ 5.2, MGDA and IMTL might under-optimize equation eq. 1 compared to unitary scalarization due to their larger convergence sets. In order to assess whether this is empirically the case, we estimate $\left\lVert\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert_{2}$ , the norm of the unitary scalarization update on shared parameters $\bm{\theta_{\parallel}}$ , for all optimizers throughout the unregularized CelebA experiment from $\lx@sectionsign$ 5.1. Large magnitudes for $\left\lVert\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert_{2}$ towards convergence would indicate that SMTOs steer optimization far from stationary points of unitary scalarization, resulting in under-optimization. We compute the update norm on the mini-batch loss every $100$ updates, and report the per-epoch average in Figure 6. Compared with unitary scalarization, most SMTOs have smaller or comparable update magnitude in the first $15$ epochs. However, towards convergence, SMTOs display larger $\left\lVert\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert_{2}$ compared to unitary scalarization. In particular, IMTL and MGDA have the largest norm, denoting significant empirical under-optimization. The additional stochasticity of RLW, PCGrad, and GradDrop also appears to lead to larger norm values than unitary scalarization, yet to a lesser degree. Given that MGDA and IMTL incur a larger loss than unitary scalarization in later epochs (see Figure 9(a) in appendix D.2), we can conclude that they guide optimization towards regions of the parameter space that under-optimize equation eq. 1, providing empirical support for our analysis.

## 6 Conclusions

This paper made two main contributions. First, we evaluated popular SMTOs using a single experimental pipeline, including previously unpublished results of MGDA, IMTL, RLW, and GradDrop in the RL setting. Surprisingly, our evaluation showed that none of the SMTOs consistently outperform unitary scalarization, the simplest and least expensive method. Second, in order to explain our surprising results, we postulate that SMTOs act as regularizers and present an analysis that supports our hypothesis. We believe our work calls for further reevaluation of progress in developing principled and efficient MTL algorithms.

We conclude by addressing the limitations of our work. While we covered a wide range of popular benchmarks, we do not exclude the existence of settings where unitary scalarization underperforms: discovering them is an interesting direction for future work. Furthermore, our experimental results were obtained via grid searches under limited compute resources: some of the methods might benefit from further fine-tuning. Nevertheless, we remark that fine-tuning will be easier for unitary scalarization due to its shorter runtimes. Finally, we presented the regularization hypothesis only as a partial explanation of our results: we hope it will steer further analysis and consequently improve the understanding of MTL.

## Acknowledgements

VK was funded by Samsung R&D Institute UK through the EPSRC Centre for Doctoral Training (CDT) in Autonomous Intelligent Machines and Systems (AIMS) at the University of Oxford . ADP was funded by EPSRC for the AIMS CDT, grant EP/L015987/1, and by an IBM PhD fellowship. SW has received funding from the European Research Council under the European Union’s Horizon 2020 research and innovation programme (grant agreement number 637713). The experiments were made possible by a generous equipment grant from NVIDIA. We would like to thank Sodhani et al. [55], Lin et al. [40] and Sener and Koltun [54] for publicly releasing their code. The authors thank Kristian Hartikainen for helpful comments on the RL experiments, and Gabriel Gama for spotting a bug in the logging of training statistics for supervised learning. VK thanks Ryota Tomioka for useful discussions on multitask optimization.

## References

  * [1] Z. Allen-Zhu, Y. Li, and Z. Song.  A convergence theory for deep learning via over-parameterization.  In _International Conference on Machine Learning_ , 2019. 
  * [2] V. Badrinarayanan, A. Kendall, and R. Cipolla.  Segnet: A deep convolutional encoder-decoder architecture for image segmentation.  _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2017. 
  * [3] B. Bakker and T. Heskes.  Task clustering and gating for bayesian multitask learning.  _Journal of Machine Learning Research_ , 2003. 
  * [4] Q. Cappart, D. Chételat, E. B. Khalil, A. Lodi, C. Morris, and P. Velickovic.  Combinatorial optimization and reasoning with graph neural networks.  In _Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021_ , 2021. 
  * [5] R. Caruana.  Multitask learning.  _Machine Learning_ , 28(1):41–75, 1997a. 
  * [6] R. Caruana.  _Multitask learning_.  PhD thesis, School of Computer Science, Carnegie Mellon University, Pittsburgh, PA 15213, 1997b. 
  * [7] R. Caruana, S. Lawrence, and L. Giles.  Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping.  In _Neural Information Processing Systems_ , 2000. 
  * [8] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam.  Encoder-decoder with atrous separable convolution for semantic image segmentation.  In _European Conference on Computer Vision_ , 2018a. 
  * [9] S. Chen, Y. Zhang, and Q. Yang.  Multi-task learning in natural language processing: An overview.  _CoRR_ , 2021. 
  * [10] Z. Chen, V. Badrinarayanana, C.-Y. Lee, and A. Rabinovich.  Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks.  In _International Conference on Machine Learning_ , 2018b. 
  * [11] Z. Chen, J. Ngiam, Y. Huang, T. Luong, H. Kretzschmar, Y. Chai, and D. Anguelov.  Just pick a sign: Optimizing deep multitask models with gradient sign dropout.  In _Neural Information Processing Systems_ , 2020. 
  * [12] R. Collobert and J. Weston.  A unified architecture for natural language processing: deep neural networks with multitask learning.  In _Machine Learning, Proceedings of the Twenty-Fifth International Conference (ICML 2008), Helsinki, Finland, June 5-9, 2008_ , 2008\. 
  * [13] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele.  The cityscapes dataset for semantic urban scene understanding.  In _Conference on Computer Vision and Pattern Recognition_ , 2016\. 
  * [14] J. Désidéri.  Multiple-gradient descent algorithm (MGDA) for multiobjective optimization.  _Comptes Rendus Mathematique_ , 350:313–318, 2012. 
  * [15] T. Dietterich.  Overfitting and undercomputing in machine learning.  _ACM Computing Surveys_ , page 326–327, sep 1995. 
  * [16] T. Evgeniou and M. Pontil.  Regularized multi–task learning.  In _ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , 2004. 
  * [17] J. Fliege and B. F. Svaiter.  Steepest descent methods for multicriteria optimization.  _Mathematical Methods of Operations Research_ , 2000. 
  * [18] M. Guo, A. Haque, D.-A. Huang, S. Yeung, and L. Fei-Fei.  Dynamic task prioritization for multitask learning.  In _Proceedings of the European Conference on Computer Vision (ECCV)_ , September 2018. 
  * [19] P. Guo, C.-Y. Lee, and D. Ulbricht.  Learning to branch for multi-task learning.  2020\. 
  * [20] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine.  Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.  In _International Conference on Machine Learning_ , 2018. 
  * [21] K. He, X. Zhang, S. Ren, and J. Sun.  Deep residual learning for image recognition.  _Conference on Computer Vision and Pattern Recognition_ , 2016. 
  * [22] T. Heskes.  Empirical bayes for learning to learn.  In _International Conference on Machine Learning_ , 2000. 
  * [23] M. Hessel, H. Soyer, L. Espeholt, W. Czarnecki, S. Schmitt, and H. van Hasselt.  Multi-task deep reinforcement learning with popart.  In _The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019_ , pages 3796–3803. AAAI Press, 2019\. 
  * [24] T. M. Hospedales, A. Antoniou, P. Micaelli, and A. J. Storkey.  Meta-learning in neural networks: A survey.  _CoRR_ , 2020. 
  * [25] W. Huang, I. Mordatch, and D. Pathak.  One policy to control them all: Shared modular policies for agent-agnostic control.  In _Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event_ , 2020. 
  * [26] S. Ioffe and C. Szegedy.  Batch normalization: Accelerating deep network training by reducing internal covariate shift.  In _International Conference on Machine Learning_ , 2015. 
  * [27] M. Jaderberg, V. Mnih, W. M. Czarnecki, T. Schaul, J. Z. Leibo, D. Silver, and K. Kavukcuoglu.  Reinforcement learning with unsupervised auxiliary tasks.  In _5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings_. OpenReview.net, 2017. 
  * [28] A. Javaloy and I. Valera.  Rotograd: Gradient homogenization in multitask learning.  In _International Conference on Learning Representations_ , 2022. 
  * [29] D. Kalashnikov, J. Varley, Y. Chebotar, B. Swanson, R. Jonschkowski, C. Finn, S. Levine, and K. Hausman.  Mt-opt: Continuous multi-task robotic reinforcement learning at scale.  _CoRR_ , 2021. 
  * [30] A. Kendall, Y. Gal, and R. Cipolla.  Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.  In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2018. 
  * [31] N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang.  On large-batch training for deep learning: Generalization gap and sharp minima.  _International Conference on Learning Representations_ , 2017. 
  * [32] K. Khetarpal, M. Riemer, I. Rish, and D. Precup.  Towards continual reinforcement learning: A review and perspectives.  _arXiv preprint arXiv:2012.13490_ , 2020. 
  * [33] D. P. Kingma and J. Ba.  Adam: A method for stochastic optimization.  In Y. Bengio and Y. LeCun, editors, _International Conference on Learning Representations_ , 2015. 
  * [34] B. Kleinberg, Y. Li, and Y. Yuan.  An alternative view: When does SGD escape local minima?  In _International Conference on Machine Learning_ , 2018. 
  * [35] I. Kokkinos.  Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory.  _IEEE Conference on Computer Vision and Pattern Recognition_ , 2017\. 
  * [36] V. Kurin, S. Godil, S. Whiteson, and B. Catanzaro.  Can q-learning with graph networks learn a generalizable branching heuristic for a SAT solver?  In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , 2020. 
  * [37] V. Kurin, M. Igl, T. Rocktäschel, W. Boehmer, and S. Whiteson.  My body is a cage: the role of morphology in graph-based incompatible control.  In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ , 2021. 
  * [38] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.  Gradient-based learning applied to document recognition.  _IEEE_ , 1998. 
  * [39] M. Li, M. Soltanolkotabi, and S. Oymak.  Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks.  In _International Conference on Artificial Intelligence and Statistics_ , 2020. 
  * [40] B. Lin, F. Ye, and Y. Zhang.  A closer look at loss weighting in multi-task learning.  In _arXiv preprint arXiv:2111.10603_ , 2022. 
  * [41] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu.  Conflict-averse gradient descent for multi-task learning.  _Advances in Neural Information Processing Systems_ , 2021a. 
  * [42] L. Liu, Y. Li, Z. Kuang, J.-H. Xue, Y. Chen, W. Yang, Q. Liao, and W. Zhang.  Towards impartial multi-task learning.  In _International Conference on Learning Representations_ , 2021b. 
  * [43] S. Liu, E. Johns, and A. J. Davison.  End-to-end multi-task learning with attention.  In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pages 1871–1880, 2019. 
  * [44] Z. Liu, P. Luo, X. Wang, and X. Tang.  Deep learning face attributes in the wild.  In _Proceedings of International Conference on Computer Vision (ICCV)_ , 2015. 
  * [45] S. Ma, R. Bassily, and M. Belkin.  The power of interpolation: Understanding the effectiveness of sgd in modern over-parametrized learning.  In _International Conference on Machine Learning_ , 2018. 
  * [46] I. Misra, A. Shrivastava, A. Gupta, and M. Hebert.  Cross-stitch networks for multi-task learning.  In _2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016_ , 2016. 
  * [47] S. Narvekar, B. Peng, M. Leonetti, J. Sinapov, M. E. Taylor, and P. Stone.  Curriculum learning for reinforcement learning domains: A framework and survey.  _J. Mach. Learn. Res._ , 2020. 
  * [48] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, and E. Fetaya.  Multi-task learning as a bargaining game.  In _International Conference on Machine Learning_ , 2022. 
  * [49] E. Parisotto, L. J. Ba, and R. Salakhutdinov.  Actor-mimic: Deep multitask and transfer reinforcement learning.  In Y. Bengio and Y. LeCun, editors, _4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings_ , 2016. 
  * [50] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala.  Pytorch: An imperative style, high-performance deep learning library.  In _Neural Information Processing Systems_. 2019. 
  * [51] A. A. Rusu, S. G. Colmenarejo, Ç. Gülçehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell.  Policy distillation.  In Y. Bengio and Y. LeCun, editors, _4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings_ , 2016. 
  * [52] S. Sabour, N. Frosst, and G. E. Hinton.  _Dynamic Routing between Capsules_.  2017\. 
  * [53] M. L. Seltzer and J. Droppo.  Multi-task learning in deep neural networks for improved phoneme recognition.  In _IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2013, Vancouver, BC, Canada, May 26-31, 2013_ , 2013\. 
  * [54] O. Sener and V. Koltun.  Multi-task learning as multi-objective optimization.  In _Neural Information Processing Systems_ , 2018. 
  * [55] S. Sodhani, A. Zhang, and J. Pineau.  Multi-task reinforcement learning with context-based representations.  In M. Meila and T. Zhang, editors, _International Conference on Machine Learning_ , 2021. 
  * [56] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov.  Dropout: A simple way to prevent neural networks from overfitting.  _Journal of Machine Learning Research_ , 2014. 
  * [57] Y. W. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu.  Distral: Robust multitask reinforcement learning.  In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , pages 4496–4506, 2017a. 
  * [58] Y. W. Teh, V. Bapst, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu.  Distral: Robust multitask reinforcement learning.  In _Neural Information Processing Systems_ , 2017b. 
  * [59] W.-C. Tseng.  Weichengtseng/pytorch-pcgrad, 2020.  URL <https://github.com/WeiChengTseng/Pytorch-PCGrad.git>. 
  * [60] H. P. van Hasselt, A. Guez, M. Hessel, V. Mnih, and D. Silver.  Learning values across many orders of magnitude.  _Advances in Neural Information Processing Systems_ , 29:4287–4295, 2016. 
  * [61] S. Vandenhende, S. Georgoulis, W. Van Gansbeke, M. Proesmans, D. Dai, and L. Van Gool.  Multi-task learning for dense prediction tasks: A survey.  _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 2021. 
  * [62] Z. Wang, Y. Tsvetkov, O. Firat, and Y. Cao.  Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models.  In _International Conference on Learning Representations_ , 2021. 
  * [63] D. Xin, B. Ghorbani, A. Garg, O. Firat, and J. Gilmer.  Do current multi-task optimization methods in deep learning even help?  In _Neural Information Processing Systems_ , 2022. 
  * [64] F. Yu, V. Koltun, and T. Funkhouser.  Dilated residual networks.  In _Computer Vision and Pattern Recognition_ , 2017. 
  * [65] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine.  Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning.  In L. P. Kaelbling, D. Kragic, and K. Sugiura, editors, _3rd Annual Conference on Robot Learning_ , 2019. 
  * [66] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn.  Gradient surgery for multi-task learning.  In _Neural Information Processing Systems_ , 2020. 

## Checklist

  1. 1.

For all authors…

     1. (a)

Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? [Yes] 

     2. (b)

Did you describe the limitations of your work? [Yes]  see $\lx@sectionsign$ 6.

     3. (c)

Did you discuss any potential negative societal impacts of your work? [Yes]  due to space constraints, we provide a discussion in appendix A.

     4. (d)

Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes] 

  2. 2.

If you are including theoretical results…

     1. (a)

Did you state the full set of assumptions of all theoretical results? [Yes] 

     2. (b)

Did you include complete proofs of all theoretical results? [Yes]  we provide full proofs in the Appendix, and refer to them in the main body of the paper.

  3. 3.

If you ran experiments…

     1. (a)

Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes]  we provide the code and the instructions in the supplemental material.

     2. (b)

Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] 

     3. (c)

Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] 

     4. (d)

Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes]  see appendix C.1.

  4. 4.

If you are using existing assets (e.g., code, data, models) or curating/releasing new assets…

     1. (a)

If your work uses existing assets, did you cite the creators? [Yes] 

     2. (b)

Did you mention the license of the assets? [Yes]  appendix C.3 describes licenses of all benchmarks and implementations we used for our work.

     3. (c)

Did you include any new assets either in the supplemental material or as a URL? [Yes]  we include the code and the instructions on how to replicate the experiments into the supplemental material.

     4. (d)

Did you discuss whether and how consent was obtained from people whose data you’re using/curating? [N/A] 

     5. (e)

Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A] 

  5. 5.

If you used crowdsourcing or conducted research with human subjects…

     1. (a)

Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A] 

     2. (b)

Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A] 

     3. (c)

Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A] 

## Appendix A Societal Impact

Due to the object of its study, our work does not have a direct societal impact. However, as any machine learning paper, it can potentially negatively effect the society through automation and loss of jobs. While it is hard to anticipate any particular risk, as any technology, if not regulated properly, it might lead to growing social and economic inequality.

On the positive side, our work might have a positive environmental impact since it advocates for simpler and more economical methods which will reduce energy consumption in data centers. Finally, simpler methods are usually easier to understand, which is beneficial in terms of explainability, an important factor for real-life applications.

## Appendix B Supplement to the Overview of Multi-Task Optimizers

This section presents the proofs and the technical results omitted from section 5, along with a description of the use of per-task gradients with respect to the last shared activation for encoder-decoder architectures (usually less expensive than per-task gradients with respect to shared parameters).

### B.1 MGDA

See 1

###### Proof.

As shown by Désidéri [14], equation eq. 3 is a simplex-constrained norm-minimization problem. In other words, the argument of the minimum is the projection of $\mathbf{0}$ onto the feasible set. Therefore:

|  $$ \mathbf{g}=\mathbf{0}\iff\mathbf{0}\in\text{Conv}(\\{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\ |\ i\in\mathcal{T}\\}). $$  |   
---|---|---  
  
It then suffices to point out that $\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}\iff\sum_{i\in\mathcal{T}}\frac{1}{|\mathcal{T}|}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}\Rightarrow\mathbf{0}\in\text{Conv}(\\{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\ |\ i\in\mathcal{T}\\})$ to conclude the proof. ∎

Due to the cost of computing per-task gradients, Sener and Koltun [54] propose MGDA-UB, which replaces the gradients wrt the parameters $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ with the gradients wrt the shared activation $\nabla_{\mathbf{z}}\mathcal{L}_{i}$ in the computation of the coefficients of $\mathbf{g}=-\sum_{i}\alpha_{i}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ . This yields an upper bound on the objective of equation eq. 3, thus restricting the set of points the algorithm convergences to. Rather than directly relying on $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ , $\mathbf{g}$ can then be obtained by computing the gradient of $\sum_{i\in\mathcal{T}}\alpha_{i}\mathcal{L}_{i}$ via reverse-mode differentiation, hence saving memory and compute.

###### Corollary 3.

The MGDA-UB SMTO by Sener and Koltun [54] converges to any point such that: $\mathbf{0}\in\text{Conv}(\\{\nabla_{\mathbf{z}}\mathcal{L}_{i}\ |\ i\in\mathcal{T}\\})$ . Furthermore, if $\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}$ is non-singular, it converges to a superset of the convergence points of the unitary scalarization.

###### Proof.

The first part of the proof proceeds as the proof of proposition 1, noting that the MGDA-UB update is associated to the following problem:

|  $\displaystyle\max_{\bm{\alpha}}$  |  $\displaystyle-\frac{1}{2}\left\lVert\mathbf{g}\right\rVert_{2}^{2}$  |   
---|---|---|---  
|  $\displaystyle\text{s.t. }$  |  $\displaystyle\sum_{i}\alpha_{i}\nabla_{\mathbf{z}}\mathcal{L}_{i}=-\mathbf{g},\quad\sum_{i\in\mathcal{T}}\alpha_{i}=1,$  |   
|  |  $\displaystyle\alpha_{i}\geq 0\qquad\forall\ i\in\mathcal{T}.$  |   
  
In order to show that a stationary point of the unitary scalarization satisfies $\mathbf{0}\in\text{Conv}(\\{\nabla_{\mathbf{z}^{*}}\mathcal{L}_{i}\ |\ i\in\mathcal{T}\\})$ , we will assume $\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}$ is non-singular, as done by Sener and Koltun [54, theorem 1]. Then, relying on the chain rule, the result follows from:

|  $\displaystyle\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}$  |  $\displaystyle\iff\sum_{i\in\mathcal{T}}\frac{1}{|\mathcal{T}|}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}$  |   
---|---|---|---  
|  |  $\displaystyle\iff\sum_{i\in\mathcal{T}}\frac{\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}}{|\mathcal{T}|}\nabla_{\mathbf{z}}\mathcal{L}_{i}=\mathbf{0}$  |   
|  |  $\displaystyle\iff\left(\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}\right)^{-1}\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}\sum_{i\in\mathcal{T}}\frac{1}{|\mathcal{T}|}\nabla_{\mathbf{z}}\mathcal{L}_{i}=\mathbf{0}$  |   
|  |  $\displaystyle\iff\sum_{i\in\mathcal{T}}\frac{1}{|\mathcal{T}|}\nabla_{\mathbf{z}}\mathcal{L}_{i}=\mathbf{0}$  |   
|  |  $\displaystyle\Rightarrow\mathbf{0}\in\text{Conv}(\\{\nabla_{\mathbf{z}}\mathcal{L}_{i}\ |\ i\in\mathcal{T}\\})$  |   
  
∎

### B.2 IMTL

See 2

###### Proof.

First, equation eq. 4 solves the linear system in $\bm{\alpha}:=[\alpha_{1},\dots,\alpha_{m}]$ given by:

|  |  $\displaystyle\mathbf{g}^{T}\left(\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert}-\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}\right)=\mathbf{0}\qquad\forall\ i\in\mathcal{T}\setminus\\{1\\},$  |   
---|---|---|---  
|  |  $\displaystyle\mathbf{g}=-\sum_{i}\alpha_{i}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i},\quad\sum_{i\in\mathcal{T}}\alpha_{i}=1,$  |   
  
which corresponds to finding a point of $\mathcal{A}^{\prime}:=\text{Aff}(\left\\{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}|\ i\in\mathcal{T}\right\\})$ which is orthogonal to $\mathcal{A}:=\text{Aff}\left(\left\\{\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}|\ i\in\mathcal{T}\right\\}\right)$ . To see this, it suffices to point out that any point orthogonal to $\mathcal{A}$ is also orthogonal to the vector subspace spanned by differences of vectors belonging to $\mathcal{A}$ . As this subspace has $m-1$ dimensions, any vector orthogonal to $\left(\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert}-\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}\right)$ for each $i\in\mathcal{T}\setminus\\{1\\}$ is orthogonal to the entire subspace.

Second, consider the problem of finding a point in $\mathcal{A}$ that is orthogonal to the linear subspace spanned by differences of vectors in $\mathcal{A}$ . In other words, we seek the projection of $\mathbf{0}$ onto $\mathcal{A}$ . Recalling the definition of $\mathcal{A}$ , we can write:

|  $\displaystyle\max_{\bm{\alpha}}$  |  $\displaystyle-\frac{1}{2}\left\lVert\mathbf{g}^{\prime}\right\rVert_{2}^{2}$  |  | (7)  
---|---|---|---|---  
|  $\displaystyle\text{s.t. }$  |  $\displaystyle\sum_{i}\alpha_{i}\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}=-\mathbf{g}^{\prime},\quad\sum_{i}\alpha_{i}=1.$  |   
  
The solution of equation eq. 7 is always collinear to the solution of equation eq. 4. In fact, if a vector $\mathbf{g}\in\mathcal{A}^{\prime}$ is orthogonal to the affine subspace $\mathcal{A}$ (or to the linear subspace spanned by differences of its members), then $\gamma\mathbf{g}=\left(-\gamma\sum_{i}\left(\alpha_{i}\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert\right)\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}\right)$ is orthogonal to $\mathcal{A}$ as well, and $\gamma=\frac{1}{\sum_{i}\left(\alpha_{i}\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert\right)}\implies\gamma\mathbf{g}\in\mathcal{A}$ .

Finally, equation eq. 7 differs from equation eq. 3 in two aspects: $\bm{\alpha}$ is not constrained to be non-negative (hence the convex hull is replaced by the affine hull), and the task vectors are normalized. Therefore, equation eq. 7 is the dual of:

|  $\displaystyle\min_{\mathbf{g},\epsilon}$  |  $\displaystyle\epsilon+\frac{1}{2}\left\lVert\mathbf{g}\right\rVert_{2}^{2}$  |  | (8)  
---|---|---|---|---  
|  $\displaystyle\text{s.t. }$  |  $\displaystyle\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}^{T}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}\mathbf{g}=\epsilon\qquad\forall\ i\in\left\\{1,\dots,m\right\\}.$  |   
  
The proposition then follows by comparing equation eq. 8 with equation eq. 2, and recalling that IMTL-L only adds a scaling factor to the chosen update direction. ∎

See 1

###### Proof.

Inspecting equation eq. 8, which yields a collinear point to the IMTL update, reveals that IMTL might converge to non Pareto-stationary points: due to the restrictive equality constraints, the minimizer of equation eq. 8 might be $\mathbf{0}$ even if a descent direction exists. Furthermore, its dual, equation eq. 7, implies that:

|  $\displaystyle\mathbf{g}=\mathbf{0}$  |  $\displaystyle\iff\mathbf{0}\in\text{Aff}\left(\left\\{\frac{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}\ |\ i\in\mathcal{T}\right\\}\right)$  |   
---|---|---|---  
|  |  $\displaystyle\iff\mathbf{0}\in\text{Aff}\left(\left\\{\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\ |\ i\in\mathcal{T}\right\\}\right),$  |   
  
which, noting that $\text{Conv}(\mathcal{A})\subseteq\text{Aff}(\mathcal{A})$ for any $\mathcal{A}$ , concludes the proof. ∎

Similarly to MGDA-UB, Liu et al. [42] advocate using $\nabla_{\mathbf{z}}\mathcal{L}_{i}$ in place of $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ while solving equation eq. 4, typically reducing the cost of computing the coefficients of $\mathbf{g}=-\sum_{i}\alpha_{i}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ .

###### Corollary 4.

When employing the approximation of problem eq. 4 that relies on $\nabla_{\mathbf{z}}\mathcal{L}_{i}$ , IMTL by Liu et al. [42] converges to $\mathbf{0}\in\text{Aff}\left(\left\\{\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}\ |\ i\in\mathcal{T}\right\\}\right)$ . If $\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}$ is non-singular, this is a superset of of the convergence points of the unitary scalarization.

###### Proof.

Following the proof of proposition 2, the following problem yields a collinear point to the $\nabla_{\mathbf{z}}\mathcal{L}_{i}$ -approximate IMTL update:

|  $\displaystyle\max_{\bm{\alpha}}$  |  $\displaystyle-\frac{1}{2}\left\lVert\mathbf{g}^{\prime}\right\rVert_{2}^{2}$  |   
---|---|---|---  
|  $\displaystyle\text{s.t. }$  |  $\displaystyle\sum_{i}\alpha_{i}\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}=-\mathbf{g}^{\prime},\quad\sum_{i}\alpha_{i}=1.$  |   
  
Therefore:

|  $$ \mathbf{g}=\mathbf{0}\iff\mathbf{0}\in\text{Aff}\left(\left\\{\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}\ |\ i\in\mathcal{T}\right\\}\right). $$  |   
---|---|---  
  
Finally, assuming $\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}$ is non-singular, we can replicate the procedure in the proof of corollary 3 to get:

|  |  $\displaystyle\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}\iff\sum_{i\in\mathcal{T}}\frac{1}{|\mathcal{T}|}\nabla_{\mathbf{z}}\mathcal{L}_{i}=\mathbf{0}$  |   
---|---|---|---  
|  |  $\displaystyle\iff\sum_{i\in\mathcal{T}}\frac{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}{|\mathcal{T}|}\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}=\mathbf{0}$  |   
|  |  $\displaystyle\iff\left(\frac{|\mathcal{T}|}{\sum_{i\in\mathcal{T}}\left(\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert\right)}\right)\sum_{i\in\mathcal{T}}\frac{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}{|\mathcal{T}|}\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}=\mathbf{0}$  |   
|  |  $\displaystyle\Rightarrow\mathbf{0}\in\text{Conv}\left(\left\\{\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}\ |\ i\in\mathcal{T}\right\\}\right)$  |   
|  |  $\displaystyle\Rightarrow\mathbf{0}\in\text{Aff}\left(\left\\{\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}\ |\ i\in\mathcal{T}\right\\}\right),$  |   
  
which shows that $\text{Aff}\left(\left\\{\frac{\nabla_{\mathbf{z}}\mathcal{L}_{i}}{\left\lVert\nabla_{\mathbf{z}}\mathcal{L}_{i}\right\rVert}\ |\ i\in\mathcal{T}\right\\}\right)$ contains the convergence points of the unitary scalarization. ∎

### B.3 PCGrad

See 3

###### Proof.

We start by pointing out that:

|  $\displaystyle\left[\frac{-\mathbf{g}_{i}^{T}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}(\mathbf{x})}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert^{2}}\right]_{+}$  |  $\displaystyle=\left[\frac{-\mathbf{g}_{i}^{T}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}(\mathbf{x})}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert}\right]_{+}\frac{1}{{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert}}$  |   
---|---|---|---  
|  |  $\displaystyle=\left[-\cos(\mathbf{g}_{i},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j})\left\lVert\mathbf{g}_{i}\right\rVert\right]_{+}\frac{1}{{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert}}$  |   
|  |  $\displaystyle\in\left[0,\frac{\left\lVert\mathbf{g}_{i}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert}\right].$  |   
  
As $\mathbf{g}_{i}$ is obtained by iterative projections of $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ onto the normals of $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\ \forall j\in\mathcal{T}\setminus\\{i\\}$ , and the norm of a vector can only decrease or remain unvaried after projections, we can write the coefficient of each $\mathbf{g}_{i}$ update as:

|  $$ d_{ij}:=\left[\frac{-\mathbf{g}_{i}^{T}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}(\mathbf{x})}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert^{2}}\right]_{+}\in\left[0,\frac{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert}\right],\enskip\forall i\neq j. $$  |   
---|---|---  
  
Furthermore, if $|\mathcal{T}|>2$ the contraction factor $\frac{\left\lVert\mathbf{g}_{i}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}$ for the norm of $g_{i}$ depends on the ordering of the projections, which is stochastic by design [66]. Therefore, $d_{ij}$ a random variable whose support is contained in $\left[0,\frac{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right\rVert}\right]$ . Finally, exploiting the definition of $d_{ij}$ , we can re-write equation eq. 5 as:

|  $\displaystyle-\mathbf{g}$  |  $\displaystyle=\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}+\sum_{i\in\mathcal{T}}\sum_{j\in\mathcal{T}\setminus\\{i\\}}d_{ij}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}=\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}+\sum_{j\in\mathcal{T}}\sum_{i\in\mathcal{T}\setminus\\{j\\}}d_{ji}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$  |   
---|---|---|---  
|  |  $\displaystyle=\sum_{j\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}+\sum_{j\in\mathcal{T}}\sum_{i\in\mathcal{T}\setminus\\{j\\}}d_{ji}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\sum_{j\in\mathcal{T}}\left(\sum_{i\in\mathcal{T}\setminus\\{j\\}}d_{ji}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}+\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right).$  |   
  
Introducing (and then removing, using their definition) dummy variables $d_{jj}=1$ :

|  $\displaystyle-\mathbf{g}$  |  $\displaystyle=\sum_{j\in\mathcal{T}}\left(\sum_{i\in\mathcal{T}\setminus\\{j\\}}d_{ji}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}+d_{jj}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right)=\sum_{j\in\mathcal{T}}\left(\sum_{i\in\mathcal{T}}d_{ji}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right)=\sum_{i\in\mathcal{T}}\left(\sum_{j\in\mathcal{T}}d_{ji}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\right)$  |   
---|---|---|---  
|  |  $\displaystyle=\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\left(\sum_{j\in\mathcal{T}}d_{ji}\right)=\sum_{i\in\mathcal{T}}\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}\left(1+\sum_{j\in\mathcal{T}\setminus\\{i\\}}d_{ji}\right),$  |   
  
from which the result trivially follows. ∎

See 2

###### Proof.

Let us start from the first statement, which does not require any assumption on the loss landscape. From proposition 3, we get:

|  $\displaystyle-\mathbf{g}$  |  $\displaystyle=\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\left(1+d_{21}\right)+\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}\left(1+d_{12}\right)$  |   
---|---|---|---  
|  |  $\displaystyle=\left(1+\left[\frac{-\cos(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2})\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert}\right]_{+}\right)\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}$  |   
|  |  $\displaystyle+\left(1+\left[\frac{-\cos(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2})\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}\right\rVert}\right]_{+}\right)\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2},$  |   
  
which shows that, in case of conflicting gradient directions, gradient norms are rebalanced proportionally to the angle between them. For $\cos(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2})=-1$ , the above evaluates to:

|  $$ -\mathbf{g}=\left(\frac{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert+\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert}\right)\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}+\left(\frac{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}\right\rVert+\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}\right\rVert}{\left\lVert\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}\right\rVert}\right)\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}. $$  |   
---|---|---  
  
The first part of the result then follows by pointing out that, if $\cos(\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1},\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2})=-1$ , then $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{1}=-\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{2}$ , and hence $\mathbf{g}=\mathbf{0}$ . We remark that a similar proof appears in [66, theorem 1 and proposition 1]. However, our derivation relaxes the author’s assumptions on $\mathcal{L}^{\text{MT}}$ and is therefore applicable to the training of neural networks.

Finally, given the assumptions on differentiability and smoothness, we need to prove that PCGrad converges to the stationary points of the unitary scalarization: this directly follows from [66, proposition 1]. ∎

### B.4 GradDrop

###### Proposition 5.

Let us assume, as often demonstrated in the single-task case [45, 1], that the multi-task network has the capacity to interpolate the data on all tasks at once: $\min_{\bm{\theta}}\mathcal{L}^{\text{MT}}=\sum_{i\in\mathcal{T}}\min_{\bm{\theta}}\mathcal{L}_{i}$ , and that its training by gradient descent attains such global minimum. Then, if $\inf_{\bm{\theta}}\mathcal{L}_{i}>-\infty\ \forall\ i\in\mathcal{T}$ , unitary scalarization converges to a joint minimum.

###### Proof.

It suffices to point out that if $\mathcal{L}^{\text{MT}}(\bm{\theta}^{*})=\sum_{i\in\mathcal{T}}\min_{\bm{\theta}}\mathcal{L}_{i}$ , then the globally optimal loss is attained for all tasks. In other words $\mathcal{L}_{i}(\bm{\theta}^{*})=\min_{\bm{\theta}}\mathcal{L}_{i}\ \forall i\in\mathcal{T}$ , and hence $\nabla_{\bm{\theta}^{*}}\mathcal{L}_{i}=\mathbf{0}\ \forall\ i\in\mathcal{T}$ (joint minimum). Furthermore, running gradient descent on $\min_{\bm{\theta}}\mathcal{L}^{\text{MT}}$ corresponds to the unitary scalarization ( $\lx@sectionsign$ 3), which concludes the proof. ∎

See 4

Proposition 4 can be proved by adapting the proof from Chen et al. [11, proposition 1]: it suffices to replace $f(\mathcal{P})$ with the Bernoulli parameter $p$ , which is non-negative by definition. In our opinion, this seriously undermines the conflicting gradient hypothesis that motivated GradDrop. For the reader’s convenience, we now provide a straightforward and self-contained proof.

###### Proof.

Let us start from the statement on $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{RGD}}$ . If $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}=\mathbf{0}\ \forall i\in\mathcal{T}$ , then $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{RGD}}=\mathbf{0}$ with probability one. On the other hand, if $\exists j:\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\neq\mathbf{0}$ , then:

|  $\displaystyle\mathds{P}\left[\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{RGD}}\neq\mathbf{0}\right]$  |  $\displaystyle\geq\mathds{P}\left[\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{RGD}}=\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}\right]$  |   
---|---|---|---  
|  |  $\displaystyle=p(1-p)^{m-1}>0,$  |   
  
where the first inequality comes from the fact that $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{RGD}}=\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{j}$ is only one of the many instances of a non-null $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{RGD}}$ . ∎

Let $\text{sign}(\mathbf{x})$ stand for the element-wise sign operator applied on $\mathbf{x}$ . On encoder-decoder architectures, similarly to MGDA and IMTL (see appendices B.1 and B.2), the authors do not apply GradDrop on $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}_{i}$ , but rather on a the usually less expensive $\nabla_{\mathbf{z}}\mathcal{L}_{i}$ . In more detail, they compute the GradDrop sign purity scores $\mathbf{p}$ from equation eq. 6 on $\sum_{i=1}^{n}\left(\text{sign}(\mathbf{z})\odot\nabla_{\mathbf{z}}\mathcal{L}_{i}\right)[i]\in\mathbb{R}^{r}$ , and then apply equation eq. 6 on the $\nabla_{\mathbf{z}}\mathcal{L}_{i}$ gradients, yielding a vector $\mathbf{g}_{z}\in\mathbb{R}^{n\times r}$ . Then, relying on reverse-mode differentiation, the update direction in the space of the parameters $\bm{\theta_{\parallel}}$ is obtained via a Jacobian-vector product: $\mathbf{g}=-\left(\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}\right)^{T}\mathbf{g}_{z}$ . Such a computation replaces the similar $\nabla_{\bm{\theta_{\parallel}}}\mathcal{L}^{\text{MT}}=\left(\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}\right)^{T}\nabla_{\mathbf{z}}\mathcal{L}^{\text{MT}}$ from the unitary scalarization.

## Appendix C Experimental Setting, Reproducibility

We now present details concerning the experimental settings from $\lx@sectionsign$ 4, including details on the employed open-source software, dataset information, hardware specifications, and hyper-parameters.

### C.1 Supervised Learning

All the experiments were run under Ubuntu 18.04 LTS, on a single GPU per run (using two 8-GPU machines in total). Timing experiments were all run on Nvidia GeForce GTX 1080 Ti GPUs, with an Intel Xeon E5-2650 CPU. The remaining experiments were run on either Nvidia GeForce RTX 2080 Ti GPUs or Nvidia GeForce GTX 1080 Ti GPUs, respectively using an Intel Xeon Gold 6230 CPU or an Intel Xeon E5-2650 CPU.

#### C.1.1 MultiMNIST

Multi-MNIST, originally introduced by Sabour et al. [52] and as modified by Sener and Koltun [54], is a simple two-task supervised learning benchmark dataset constructed by uniformly sampling MNIST [38] images, and placing one in the top-left corner, the other in the bottom-right corner. Each of the two overlaid images corresponds to a 10-class classification task. Using the above procedure, we generate the Multi-MNIST training set from the first $50000$ MNIST training images, the validation set from the last $10000$ training images, and the test set from the original MNIST test set. For consistency with the experimental setup of Sener and Koltun [54], we employ a modified encoder-decoder version of the LeNet architecture [38]. Specifically, the last layer is omitted from the encoder, and two fully-connected layers are employed as task-specific predictive heads. The cross-entropy loss is used for both tasks. All methods are trained for $100$ epochs using Adam [33] in the stochastic gradient setting, with an initial learning rate of $\eta=10^{-2}$ (tuned in $\eta\in\\{10^{-3},10^{-2},10^{-1}\\}$ and yielding the best validation results for all considered algorithms), exponentially decayed by $0.95$ after each epoch, and a mini-batch size of $256$ .

#### C.1.2 CelebA

The CelebA [44] dataset consists of $200,000$ headshots (with standard training, validation and test splits) associated with the presence or absence of $40$ attributes. In the MTL literature, is commonly treated as a $40$ -task classification problem, each task being a binary classification problem for an attribute. As commonly done in previous work [54, 66, 42], we employ an encoder-decoder architecture where the encoder is a ResNet-18 [21] (without the final layer) with batch normalization layers [26], and the per-task decoders are linear classifiers. The cross-entropy loss is used for all tasks. The training is performed from scratch for $50$ epochs using Adam, with a mini-batch size of $128$ and a per-epoch exponential decay factor of $0.95$ . As common on this network-dataset combination [40, 11], the initial learning rate is $\eta=10^{-3}$ for all methods except for MGDA and IMTL, for which $\eta=5\times 10^{-4}$ yielded a better validation performance. As done by the respective authors, for PCGrad, RLW and GradDrop we use the same learning rate as the unitary scalarization [66, 40, 11].

#### C.1.3 Cityscapes

We rely on the version of the dataset pre-processed by Liu et al. [43], which consists of $2,975$ training and $500$ test images and presents two tasks: semantic segmentation on $7$ classes, and depth estimation. We further split the original training set into a validation set of $595$ images, employed to tune hyper-parameters, and a training set of $2380$ images. Consistently with recent work [40], we rely on a dilated ResNet-50 architecture pre-trained on ImageNet [64] for the encoder, and on the Atrous Spatial Pyramid Pooling [8], which internally uses batch normalization, as decoders. While more powerful encoders might lead to better performance on Cityscapes, like the SegNet [2] used in [28, 41, 48], we aim to provide a fair comparison of MTL optimizers, rather than maximize overall task performance. Cross-entropy loss is employed on the semantic segmentation task, whereas the $\ell_{1}$ loss is used for the depth estimation. The training is performed by using Adam with a mini-batch size of $32$ for $100$ epochs, with an initial step size $\eta=5\times 10^{-4}$ resulting in the best validation performance for all algorithms, exponentially decayed by $0.95$ at each epoch.

### C.2 Reinforcement Learning

Similarly to the supervised learning experiments, we ran all the experiments under Ubuntu 18.04 LTS using one GPU per run (using six 8-GPU machines in total). Timing experiments were all run using NVIDIA GeForce RTX 2080 Ti GPUs, with an Intel Xeon Gold 6230 CPU. The main bulk of the remaining experiments was run on Nvidia GeForce RTX 2080 Ti GPUs with either Intel Xeon Gold 6230 or Intel Xeon Silver 4216. We utilised NVIDIA GeForce RTX 3080 GPUs with Intel Xeon Gold 6230 CPUs for a small fraction of experiments.

We use Meta-World’s MT10/MT50 for our experiment. The benchmark consists of ten/fifty tasks in which a simulated robot manipulator has to perform various actions, e.g., pressing a button, opening a door, or pushing the block. We use Sodhani et al. [55] for most of the hyperparameters and list them in Table 1. We use bold font where we use a hyperparameter different from Sodhani et al. [55]. Similarly to Sodhani et al. [55], we use the v1 version of Metaworld for our experiments11 1 <https://github.com/rlworkgroup/metaworld.git@af8417bfc82a3e249b4b02156518d775f29eb289>. Sodhani et al. [55] use a shared entropy loss weight $\alpha$ for PCGrad and separate $\alpha$ for unitary scalarization22 2 <https://mtrl.readthedocs.io/en/latest/pages/tutorials/baseline.html>. In our experiments, use shared $\alpha$ for all of the methods for fairness. Since it is a single number (rather than a vector), we used unitary scalarization to update $\alpha$ for all SMTOs apart from PCGrad which was already implemented in [55].

We use the same network architecture as in Sodhani et al. [55], i.e. a three-layered feedforward fully-connected network with 400 hidden units per layer for both, the actor and the critic. The actor is shared across all tasks as well as the critic.

To normalize rewards, we keep track of first and second moments in the buffer and normalise the rewards by their standard deviation: $r^{\prime}_{i}=\nicefrac{{r_{i}}}{{\hat{\sigma}_{i}}},$ where $\hat{\sigma}_{i}$ is the sample standard deviation of the rewards for environment $i$ .

Sodhani et al. [55] average the gradient for unitary scalarisation and pcgrad, whereas our SMTO implementations sum the gradients, i.e. effectively using larger learning rates (apart from MGDA that assures that all the aggregation weights sum to 1). We tried reducing the learning rate for SMTOs that sum (RLW Norm., RLW Diri., and GradDrop) both for MT10 and MT50, but it worked worse for these methods and we kept the default learning rate for them as well. We had to use a smaller learning rate for IMTL, because with the default one it crashed at the beginning of training due to numerical overflow. Smaller learning rate did not prevent it from crashing, but this happened much later.

Table 1: Hyperparameters of the RL experiments. Hyperparameters different from Sodhani et al. [55] are in bold. Hyperparameter | Value  
---|---  
All methods |   
– training steps | 2,000,000  
– batch size | 1280  
– Replay buffer size | 4,000,000  
– actor learning rate | 0.0003  
– critic learning rate | 0.0003  
– entropy $\alpha$ learning rate | 0.0003  
– shared entropy $\alpha$  | True  
– runs | 10  
– discounting $\gamma$  | 0.99  
Unit. Scal. |   
– actor $l_{2}$ coeff. | 0.0003  
PCGrad |   
– actor $l_{2}$ coeff. | 0.0001  
RLW Norm. |   
– normal mean | 0  
– normal std | 1  
– actor $l_{2}$ coeff. | 0.0003  
RLW Diri. |   
– $\alpha$  | 1  
– actor $l_{2}$ coeff. | 0.0003  
GradDrop |   
– k | 1  
– p | 0.5  
– actor $l_{2}$ coeff. | 0.0001  
MGDA |   
– gradient normalization |  $L_{2}$   
– actor $l_{2}$ coeff. | 0.0  
IMTL |   
– actor learning rate | 0.00003  
– critic learning rate | 0.00003  
– entropy $\alpha$ learning rate | 0.00003  
– actor $l_{2}$ coeff. | 0.0  
  
We tried $10^{6}$ , $2\times 10^{6}$ , and $4\times 10^{6}$ for the replay buffer size with the last being superior in terms of stability. Additionally, for $l_{2}$ actor regularization, we tried $0.0001$ and $0.0003$ with the latter being slightly superior for the baseline. We tried the same options for other SMTOs, and picked the best option for each of the method. For MGDA, no regularisation works best, most likely due to a strong regularization effect of the method itself, which is mirrored by our supervised learning results. PCGrad and Graddrop work best with the regularization coefficient of $0.0001$ . Both RLW variants use the same coefficient as the baseline ( $0.0003$ ).

For MT50, we took the best MT10 hyperparameters, and we believe one could obtain even better results for unitary scalarisation since it is much faster to tune compared to other SMTOs (e.g. 15 hours for unitary scalarisation vs 9 days for PCGrad).

### C.3 Software Acknowledgments and Licenses

Our codebase is built upon several prior works: [54], [43], [40] and [55]: all of them were released under a MIT license. We also acknowledge Tseng [59], upon which we built some of our code. Multi-MNIST is based on MNIST dataset that is released under Creative Commons Attribution-Share Alike 3.0 license. The code for generating Multi-MNIST dataset was taken from Sener and Koltun [54] released under MIT license. CelebA dataset has a custom license allowing non-commercial research purposes. More details can be found on the project website:<http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>. Cityscapes also has a custom license allowing non-commercial research purposes. The full text of the license can be found on the project website:<https://www.cityscapes-dataset.com/license/>. Metaworld, used for RL experiments is released under MIT license.

## Appendix D Supplementary Supervised Learning Experiments

This section presents supervised learning results omitted from $\lx@sectionsign$ 4.1. In particular, we show additional plots for the experiments of $\lx@sectionsign$ 4.1, then present an analysis of the regularising effect of SMTOs in the absence of single-task regularization ( $\lx@sectionsign$ D.2), and conclude with an ablation study on GradDrop’s dependency on the sign of per-task gradients ( $\lx@sectionsign$ D.3).

### D.1 Addendum

This section complements the plots presented in $\lx@sectionsign$ 4.1. In particular, we show the test and runtime results in table form, along with the behavior of the validation metrics and of the training loss over the training epochs. Plots for Multi-MNIST, CelebA, and Cityscapes are reported in Figures 7, 8 and 17, respectively.

The behavior of the CelebA training loss demonstrates heavier regularization (compare with the unregularized plot in Figure 9(a)). Except IMTL and MGDA, for which the tuned values of the weight decay prevent overfitting, the other optimizers display very similar validation and training curves, and start overfitting around epoch $30$ . Considering that most SMTOs required less regularization (see $\lx@sectionsign$ 4.1.2), the results are consistent with our interpretation of SMTOs as regularizers in $\lx@sectionsign$ 5. The Cityscapes plots display a certain instability across training epochs, as demonstrated by the various peaks and valleys in the metrics. Nevertheless, in spite of a factor $10$ difference in scale, both training losses are similarly decreased by most optimizers.

(a) Mean (and 95 $\%$ CI) average task validation accuracy per training epoch.

(b) Mean (and 95 $\%$ CI) training multi-task loss $\mathcal{L}^{\text{MT}}$ per epoch.

MTO Average Task Accuracy Epoch Runtime [s] Unit. Scal. 9.476e-01 $\pm$ 4.368e-03 [3.510e+00, 3.617e+00] IMTL 9.487e-01 $\pm$ 2.533e-03 [3.695e+00, 3.996e+00] MGDA 9.478e-01 $\pm$ 1.977e-03 [3.491e+00, 3.617e+00] GradDrop 9.347e-01 $\pm$ 1.282e-02 [3.508e+00, 3.589e+00] PCGrad 9.479e-01 $\pm$ 3.578e-03 [3.807e+00, 3.928e+00] RLW Diri. 9.430e-01 $\pm$ 2.973e-03 [3.790e+00, 4.005e+00] RLW Norm. 9.399e-01 $\pm$ 8.929e-03 [3.894e+00, 4.225e+00]

(a) Mean and 95 $\%$ CI of the avg. task test accuracy across runs, and interquartile range for the training time per epoch.

Figure 7:  Additional figures for the comparison of various SMTO s with the unitary scalarization on the MultiMNIST dataset [54]. 

(a) Mean (and 95 $\%$ CI) average task validation accuracy per training epoch.

(b) Mean (and 95 $\%$ CI) training multi-task loss $\mathcal{L}^{\text{MT}}$ per epoch.

MTO Average Task Accuracy Epoch Runtime [s] Unit. Scal. 9.090e-01 $\pm$ 7.568e-04 [2.869e+02, 2.878e+02] IMTL 9.093e-01 $\pm$ 7.631e-04 [3.600e+02, 3.621e+02] MGDA 9.022e-01 $\pm$ 9.687e-04 [6.859e+02, 7.194e+02] GradDrop 9.098e-01 $\pm$ 3.383e-04 [3.001e+02, 3.008e+02] PCGrad 9.093e-01 $\pm$ 1.108e-03 [1.015e+04, 1.016e+04] RLW Diri. 9.099e-01 $\pm$ 7.845e-04 [3.040e+02, 3.054e+02] RLW Norm. 9.095e-01 $\pm$ 1.012e-03 [3.028e+02, 3.037e+02]

(a) Mean and 95 $\%$ CI of the avg. task test accuracy across runs, and interquartile range for the training time per epoch.

Figure 8:  Additional figures for the comparison of various SMTO s with the unitary scalarization on the CelebA [44] dataset. 

### D.2 Unregularized Experiments

Figures 6, 9(a) and 9(b) respectively report the average task validation accuracy, the multi-task training loss, and the multi-task validation loss at each training epoch. The regularizing effect of SMTOs compared to unitary scalarization is shown by: (i) the delay of the onset of overfitting on the validation data in figure 6, (ii) the reduction of the convergence rate on the training loss in figure 9(a) (compare with figure 8(b)), and (iii) the fact that validation and training losses remain positively correlated for larger numbers of epochs. In fact, the behavior of both the training and validation loss for the SMTOs closely parallels that of $\ell_{2}$ -regularized unitary scalarization, with differing degrees of regularization. We further note that unregularized IMTL displays a certain instability (compare with the regularized version in figure 8(a)).

The addition of dropout layers further reduces overfitting, improves stability (reduced confidence intervals) and pushes the average validation curve upwards, motivating its use on all optimizers for the experiments of $\lx@sectionsign$ 4.1.2. Nevertheless, confidence intervals in Figure 6 still overlap due to the instability of the unregularized unitary scalarization. Figure 11 provides a more detailed comparison over $20$ repetitions, confirming that the combined use of dropout layers and $\ell_{2}$ regularization improves average performance and reduces the empirical variance for unitary scalarization. Furthermore, Figure 10 shows that regularization improves the peak average validation performance for all algorithms, demonstrating the need of tuning $\lambda$ also for SMTOs. We conclude by pointing out that even without regularization, when carefully tuned, the maximal performance over epochs of unitary scalarization is comparable to SMTOs in Figure 6.

(a) Mean and 95 $\%$ CI (3 runs) multi-task training loss per epoch. 

(b) Mean and 95 $\%$ CI (3 runs) multi-task validation loss per training epoch. 

Figure 9: Additional figures for the unregularized comparison of various SMTO s with the unitary scalarization on CelebA. SMTOs provide varying degrees of regularization. Figure 10:  Effect of regularization (dropout layers and weight decay) on the average task validation accuracy for all considered optimizers on the CelebA dataset: regularization improves the average performance of all algorithms.  Figure 11:  Effect of regularization (dropout layers and weight decay) on unitary scalarization on the CelebA dataset: violin plots (20 runs) for the best avg. task validation accuracy over epochs. The width at a given value represents the proportion of runs yielding that result. Regularization improves the average performance while decreasing its variability.

### D.3 Sign-Agnostic GradDrop

We will now present an ablation study on GradDrop, investigating the effect of the sign of per-task gradients on the SMTO’s performance. Specifically, we compare the performance of GradDrop with a sign-agnostic version of its stochastic gradient masking (which we refer to as “Sign-Agnostic GradDrop"), whose update direction is defined as follows:

|  $$ \mathbf{g}=-\left(\frac{\partial\mathbf{z}}{\partial\bm{\theta_{\parallel}}}\right)^{T}\left(\sum_{i\in\mathcal{T}}\mathbf{u}_{i}\odot\nabla_{\mathbf{z}}\mathcal{L}_{i}\right), $$  |   
---|---|---  
  
where $\mathbf{u}_{i},\nabla_{\mathbf{z}}\mathcal{L}_{i}\in\mathbb{R}^{n\times r}$ and, for all $i\in\mathcal{T}$ , $\mathbf{u}_{i}$ is i.i.d. according to $\mathbf{u}_{i}[j,k]\sim\text{Bernoulli}(p)\ \forall j\in\\{1,\dots,n\\},k\in\\{1,\dots,r\\}$ . Differently from a similar study carried out by Chen et al. [11], we tuned the hyper-parameter of the sign-agnostic masking in the following range: $p\in\\{0.1,0.25,0.5,0.75,0.9\\}$ .

The experimental setup complies with the one described in appendix C.1. Figure 12, plotting test and validation results for the CelebA dataset [44], shows that the performance of Sign-Agnostic GradDrop closely matches the original algorithm. Therefore, sign conflicts across per-task gradients do not seem to play a significant role in GradDrop’s performance.

(a) Mean and 95 $\%$ CI (3 runs) avg. task test accuracy.

(b) Mean and 95 $\%$ CI (3 runs) avg. task validation accuracy per training epoch.

Figure 12:  Comparison of GradDrop [11] with sign-agnostic masking of the shared-representation gradients on the CelebA dataset [44]. No statistically relevant difference between the two methods can be observed for the majority of the epochs.

## Appendix E Supplementary Reinforcement Learning Experiments

### E.1 Addendum

This section presents additional plots for the RL experiments in $\lx@sectionsign$ 4.2. Specifically, Figure 13 re-plots Figure 4(a) and 4(b) with the omitted IMTL results, while Figure 14 shows the learning curves omitted from $\lx@sectionsign$ 4.2. As pointed out in $\lx@sectionsign$ 4.2, none of the IMTL runs successfully terminated due to numerical instability. Indeed, Liu et al. [42] show that, in supervised settings, coefficients do not fluctuate much across epochs [42, Figure 4, appendix B] and never become negative. By contrast, up to 50% of the scaling coefficients $\alpha$ are negative in our experiments, thus reversing subtask gradient directions. MGDA, which constrains the weights, is more stable and is comparable to unitary scalarization. In order to avoid incomplete curves and unfair calculations of the mean, Figure 14 plots the highest value ever achieved by _any_ seed as a dashed horizontal line. The IMTL results in Figure 13, instead, report the best average success rate of each seed until its termination.

(a) MT10 (10 runs per method).

(b) MT50 (10 runs per method).

Figure 13:  Mean and 95 $\%$ CI for the best avg. success rate on Metaworld. None of the SMTOs significantly outperforms unitary scalarization.

(a) MT10 (10 points per method).

(b) MT50 (10 points per method).

Figure 14:  Mean and 95 $\%$ CI for the avg. success rate on Metaworld. None of the SMTOs significantly outperforms unitary scalarization.

### E.2 Ablation studies

Figure 18 presents our ablations for MT10 experiments. Due to computational constraints, we ran ablations on the unitary scalarization and PCGrad since these are the two methods previously tested in the RL setting.

Figure 15 shows ablation studies on the effect of regularization on MT10 and MT50. In spite of CI overlaps, actor $l_{2}$ regularization pushes the average higher on both benchmarks, motivating our use of regularization for the experiments in $\lx@sectionsign$ 4.2. Furthermore, the gap between the averages tends to widen with the number of updates on MT50, suggesting improved stabilization.

(a) MT10 average performance (10 runs) and 95% CI.

(b) MT50 average performance (5 runs) and 95% CI.

Figure 15:  For both MT10 and MT50, actor $l_{2}$ regularization pushes the average higher for unitary scalarization.

### E.3 Sensitivity to Reward Normalization

Figure 16 shows that multitask agent performance is highly sensitive to the reward normalization moving average hyperparameter33 3 <https://github.com/facebookresearch/mtenv/blob/4a6d9d6fdfb321f1b51f890ef36b5161359e972d/mtenv/envs/metaworld/wrappers/normalized_env.py#L69> motivating our buffer normalization in Section 4.2.

Figure 16: The learning outcomes of a Multitask SAC agent vary considerably depending on the reward normalisation hyperparameter. Each of the curves represents and average of 10 runs with shaded 95% confidence interval.

(a) Mean and 95 $\%$ CI of the test metrics across runs, and interquartile range for the training time per epoch.

MTO Absolute Depth Error Relative Depth Error Segmentation Accuracy Segmentation mIOU Epoch Runtime [s] Unit. Scal. 1.301e-02 $\pm$ 2.342e-04 4.761e+01 $\pm$ 5.148e+00 9.196e-01 $\pm$ 2.913e-04 7.012e-01 $\pm$ 6.001e-04 [3.228e+02, 3.241e+02] IMTL 1.281e-02 $\pm$ 7.521e-04 4.389e+01 $\pm$ 6.984e-01 9.164e-01 $\pm$ 2.828e-03 6.967e-01 $\pm$ 4.785e-03 [7.329e+02, 7.373e+02] MGDA 1.418e-02 $\pm$ 2.331e-04 4.750e+01 $\pm$ 1.466e+01 9.189e-01 $\pm$ 2.636e-04 6.999e-01 $\pm$ 3.124e-03 [7.251e+02, 7.269e+02] GradDrop 1.293e-02 $\pm$ 2.757e-04 4.674e+01 $\pm$ 7.709e+00 9.193e-01 $\pm$ 1.282e-03 7.024e-01 $\pm$ 3.628e-03 [5.196e+02, 5.215e+02] PCGrad 1.294e-02 $\pm$ 2.284e-04 4.380e+01 $\pm$ 5.165e+00 9.198e-01 $\pm$ 9.119e-04 7.025e-01 $\pm$ 6.531e-04 [4.202e+02, 4.212e+02] RLW Diri. 1.305e-02 $\pm$ 4.155e-04 4.810e+01 $\pm$ 2.259e+00 9.199e-01 $\pm$ 1.247e-03 7.037e-01 $\pm$ 1.989e-03 [3.161e+02, 3.164e+02] RLW Norm. 1.301e-02 $\pm$ 5.528e-04 4.630e+01 $\pm$ 2.751e+00 9.192e-01 $\pm$ 4.962e-04 7.006e-01 $\pm$ 4.580e-03 [3.194e+02, 3.210e+02]

(a) Mean (and 95 $\%$ CI) absolute depth validation error per training epoch.

(b) Mean (and 95 $\%$ CI) relative depth validation error per training epoch.

(c) Mean (and 95 $\%$ CI) validation segmentation mIOU per training epoch.

(d) Mean (and 95 $\%$ CI) validation segmentation accuracy per training epoch.

(e) Mean (and 95 $\%$ CI) training depth loss per epoch.

(f) Mean (and 95 $\%$ CI) training segmentation loss per epoch.

Figure 17: Additional figures for the comparison of SMTO s with the unitary scalarization on the Cityscapes [13] dataset.  ![Refer to caption](2201.04122v4/ablations_shared_alpha.png) Figure 18: Metaworld’s MT10 ablation experiments.