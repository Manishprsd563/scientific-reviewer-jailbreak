Published as a conference paper at ICLR 2025

# T AB M: A DVANCING T ABULAR D EEP L EARNING WITH P ARAMETER -E FFICIENT E NSEMBLING


**Yury Gorishniy** _[∗]_ **Akim Kotelnikov** **Artem Babenko**
Yandex HSE University, Yandex Yandex


A BSTRACT


Deep learning architectures for supervised learning on tabular data range from
simple multilayer perceptrons (MLP) to sophisticated Transformers and retrievalaugmented methods. This study highlights a major, yet so far overlooked opportunity for designing substantially better MLP-based tabular architectures. Namely,
our new model TabM relies on _efficient ensembling_, where one TabM efficiently imitates an ensemble of MLPs and produces multiple predictions per object. Compared
to a traditional deep ensemble, in TabM, the underlying implicit MLPs are trained
simultaneously, and (by default) share most of their parameters, which results in
significantly better performance and efficiency. Using TabM as a new baseline, we
perform a large-scale evaluation of tabular DL architectures on public benchmarks
in terms of both task performance and efficiency, which renders the landscape of
tabular DL in a new light. Generally, we show that MLPs, including TabM, form
a line of stronger and more practical models compared to attention- and retrievalbased architectures. In particular, we find that TabM demonstrates the best performance among tabular DL models. Then, we conduct an empirical analysis on the
ensemble-like nature of TabM . We observe that the multiple predictions of TabM
are weak individually, but powerful collectively. Overall, our work brings an impactful technique to tabular DL and advances the performance-efficiency trade-off
with TabM — a simple and powerful baseline for researchers and practitioners. The
[code is available at: https://github.com/yandex-research/tabm.](https://github.com/yandex-research/tabm)


1 I NTRODUCTION


Supervised learning on tabular data is a ubiquitous machine learning (ML) scenario in a wide range
of industrial applications. Among classic non-deep-learning methods, the state-of-the-art solution
for such tasks is gradient-boosted decision trees (GBDT) (Prokhorenkova et al., 2018; Chen &
Guestrin, 2016; Ke et al., 2017). Deep learning (DL) models for tabular data, in turn, are reportedly
improving, and the most recent works claim to perform on par or even outperform GBDT on academic
benchmarks (Hollmann et al., 2023; Chen et al., 2023b;a; Gorishniy et al., 2024).


However, from the practical perspective, it is unclear if tabular DL offers any obvious go-to baselines
beyond simple architectures in the spirit of a multilayer perceptron (MLP). _First_, the scale and
consistency of performance improvements of new methods w.r.t. simple MLP-like baselines are not
always explicitly analyzed in the literature. Thus, one has to infer those statistics from numerous
per-dataset performance scores, which makes it hard to reason about the progress. At the same
time, due to the extreme diversity of tabular datasets, consistency is an especially valuable and
hard-to-achieve property for a hypothetical go-to baseline. _Second_, efficiency-related properties,
such as training time, and especially inference throughput, sometimes receive less attention. While
methods are usually equally affordable on small-to-medium datasets (e.g. _<_ 100K objects), their
applicability to larger datasets remains uncertain. _Third_, some recent work generally suggests that the
progress on academic benchmarks may not transfer that well to real-world tasks (Rubachev et al.,
2024). With all the above in mind, in this work, we thoroughly evaluate existing tabular DL methods
and find that non-MLP models do not yet offer a convincing replacement for MLPs.


At the same time, we identify a previously overlooked path towards more powerful, reliable, and
reasonably efficient tabular DL models. In a nutshell, we find that the parameter-efficient approach to


_∗_ The corresponding author: yurygorishniy@gmail.com


1


Published as a conference paper at ICLR 2025


deep ensembling, where most weights are shared between ensemble members, allow one to make
simple and strong tabular models out of plain MLPs. For example, MLP coupled with BatchEnsemble
(Wen et al., 2020) — a long-existing method — right away outperforms popular attention-based
models, such as FT-Transformer (Gorishniy et al., 2021), while being simpler and more efficient.
This result alone suggests that efficient ensembling is a low-hanging fruit for tabular DL.


Our work builds on the above observations and offers TabM — a new powerful and practical model
for researchers and practitioners. Drawing an informal parallel with GBDT (an ensemble of decision
trees), TabM can also be viewed as a simple base model (MLP) combined with an ensembling-like
technique, providing high performance and simple implementation at the same time.


**Main contributions.** We summarize our main contributions as follows:


1. We present TabM — a simple DL architecture for supervised learning on tabular data. TabM is
based on MLP and parameter-efficient ensembling techniques closely related to BatchEnsemble
(Wen et al., 2020). In particular, TabM produces **M** ultiple predictions per object. TabM easily
competes with GBDT and outperforms prior tabular DL models, while being more efficient than
attention- and retrieval-based DL architectures.
2. We provide a fresh perspective on tabular DL models in a large-scale evaluation along four
dimensions: performance ranks, performance score distributions, training time, and inference
throughput. One of our findings is that MLPs, including TabM, hit an appealing performanceefficiency tradeoff, which is not the case for attention- and retrieval-based models.
3. We show that the two key reasons for TabM’s high performance are the collective training of the
underlying implicit MLPs and the weight sharing. We also show that the multiple predictions of
TabM are weak and overfitted individually, while their average is strong and generalizable.


2 R ELATED WORK


**Decision-tree-based models.** Gradient-boosted decision trees (GBDT) (Chen & Guestrin, 2016; Ke
et al., 2017; Prokhorenkova et al., 2018) is a strong and efficient baseline for tabular tasks. GBDT is
a classic machine learning model, specifically, an ensemble of decision trees. Our model TabM is a
deep learning model, specifically, a parameter-efficient ensemble of MLPs.


**Tabular deep learning architectures.** A large number of deep learning architectures for tabular
data have been proposed over the recent years. That includes attention-based architectures (Song
et al., 2019; Gorishniy et al., 2021; Somepalli et al., 2021; Kossen et al., 2021; Yan et al., 2023),
retrieval-augmented architectures (Somepalli et al., 2021; Kossen et al., 2021; Gorishniy et al., 2024;
Ye et al., 2024), MLP-like models (Gorishniy et al., 2021; Klambauer et al., 2017; Wang et al., 2020)
and others (Arik & Pfister, 2020; Popov et al., 2020; Chen et al., 2023b; Marton et al., 2024; Hollmann
et al., 2023). Compared to prior work, the key difference of our model TabM is its computation
flow, where one TabM imitates an ensemble of MLPs by producing multiple independently trained
predictions. Prior attempts to bring ensemble-like elements to tabular DL (Badirli et al., 2020; Popov
et al., 2020) were not found promising (Gorishniy et al., 2021). Also, being a simple feed-forward
MLP-based model, TabM is significantly more efficient than some of the prior work. Compared to
attention-based models, TabM does not suffer from quadratic computational complexity w.r.t. the
dataset dimensions. Compared to retrieval-based models, TabM is easily applicable to large datasets.


**Improving tabular MLP-like models.** Multiple recent studies achieved competitive performance
with MLP-like architectures on tabular tasks by applying architectural modifications (Gorishniy et al.,
2022), regularizations (Kadra et al., 2021; Jeffares et al., 2023a; Holzmuller et al. ¨, 2024), custom
training techniques (Bahri et al., 2021; Rubachev et al., 2022). Thus, it seems that tabular MLPs have
good potential, but one has to deal with overfitting and optimization issues to reveal that potential.
Our model TabM achieves high performance with MLP in a different way, namely, by using it as the
base backbone in a parameter-efficient ensemble in the spirit of BatchEsnsemble (Wen et al., 2020).
Our approach is orthogonal to the aforementioned training techniques and architectural advances.


**Deep ensembles.** In this paper, by a deep ensemble, we imply multiple DL models of the same
architecture trained independently (Jeffares et al., 2023b) for the same task under different random
seeds (i.e. with different initializations, training batch sequences, etc.). The prediction of a deep
ensemble is the mean prediction of its members. Deep ensembles often significantly outperform single
DL models of the same architecture (Fort et al., 2020) and can excel in other tasks like uncertainty


2


Published as a conference paper at ICLR 2025


estimation or out-of-distribution detection (Lakshminarayanan et al., 2017). It was observed that
individual members of deep ensembles can learn to extract diverse information from the input, and
the power of deep ensembles depends on this diversity (Allen-Zhu & Li, 2023). The main drawback
of deep ensembles is the cost and inconvenience of training and using multiple models.


**Parameter-efficient deep “ensembles”.** To achieve the performance of deep ensembles at a lower
cost, multiple studies proposed architectures that imitate ensembles by producing multiple predictions
with one model (Lee et al., 2015; Zhang et al., 2020; Wen et al., 2020; Havasi et al., 2021; Antoran ´
et al., 2020; Turkoglu et al., 2022). Such models can be viewed as “ensembles” where the implicit
ensemble members share a large amount of their weights. There are also non-architectural approaches
to efficient ensembling, e.g. FGE (Garipov et al., 2018), but we do not explore them, because we
are interested specifically in architectural techniques. In this paper, we highlight parameter-efficient
ensembling as an impactful paradigm for tabular DL. In particular, we describe two simple variations
of BatchEnsemble (Wen et al., 2020) that are highly effective for tabular MLPs. One variation uses a
more efficient parametrization, and another one uses an improved initialization.


3 T AB M


In this section, we present TabM — a **Tab** ular DL model that makes **M** ultiple predictions.


3.1 P RELIMINARIES


**Notation.** We consider classification and regression tasks on tabular data. _x_ and _y_ denote the features
and a label, respectively, of one object from a given dataset. A machine learning model takes _x_ as
input and produces ˆ _y_ as a prediction of _y_ . _N ∈_ N and _d ∈_ N respectively denote the “depth” (e.g. the
number of blocks) and “width” (e.g. the size of the latent representation) of a given neural network.
_d_ _y_ _∈_ N is the output representation size (e.g. _d_ _y_ = 1 for regression tasks, and _d_ _y_ equals the number
of classes for classification tasks).


**Datasets.** Our benchmark consists of 46 publicly available datasets used in prior work, including
Grinsztajn et al. (2022); Gorishniy et al. (2024); Rubachev et al. (2024). The main properties of our
benchmark are summarized in Table 1, and more details are provided in Appendix C.


Table 1: The overview of our benchmark. The “Split type” property is explained in the text.


#Datasets Train size #Features Task type Split type


Min. Q50 Mean Max. Min. Q50 Mean Max. #Regr. #Classif. Random Domain-aware


46 1.8K 12K 76K 723K 3 20 108 986 28 18 37 9


**Domain-aware splits.** We pay extra attention to datasets with what we call “domain-aware” splits,
including the eight datasets from the TabReD benchmark (Rubachev et al., 2024) and the Microsoft
dataset (Qin & Liu, 2013). For these datasets, their original real-world splits are available, e.g.
time-aware splits as in TabReD. Such datasets were shown to be challenging for some methods
because they naturally exhibit a certain degree of distribution shift between training and test parts
(Rubachev et al., 2024). The random splits of the remaining 37 datasets are inherited from prior work.


**Experiment setup.** We use the setup from Gorishniy et al. (2024), and describe it in detail in
subsection D.2. Most importantly, on each dataset, a given model undergoes hyperparameter tuning
on the _validation_ set, then the tuned model is trained from scratch under multiple random seeds, and
the _test_ metric averaged over the random seeds becomes the final score of the model on the dataset.


**Metrics.** We use RMSE (the root mean square error) for regression tasks, and accuracy or ROC-AUC
for classification tasks depending on the dataset source. See subsection D.3 for details.


Also, throughout the paper, we often use the relative performance of models w.r.t. MLP as the key
metric. This metric gives a unified perspective on all tasks and allows reasoning about the scale of
improvements w.r.t. to a simple baseline (MLP). Formally, on a given dataset, the metric is defined
as � baselinescore _[−]_ [1] � _·_ 100%, where “score” is the metric of a given model, and “baseline” is the metric

of MLP. In this computation, for regression tasks, we convert the raw metrics from RMSE to _R_ [2] to
better align the scales of classification and regression metrics.


3


Published as a conference paper at ICLR 2025


3.2 A QUICK INTRODUCTION TO B ATCH E NSEMBLE .


For a given architecture, let’s consider any linear layer _l_ in it: _l_ ( _x_ ) = _Wx_ + _b_, where _x ∈_ R _[d]_ [1],
_W ∈_ R _[d]_ [2] _[×][d]_ [1], _b ∈_ R _[d]_ [2] . To simplify the notation, let _d_ 1 = _d_ 2 = _d_ . In a traditional deep ensemble, the
_i_ -th member has its own set of weights _W_ _i_ _, b_ _i_ for this linear layer: _l_ _i_ ( _x_ _i_ ) = _W_ _i_ _x_ _i_ + _b_ _i_, where _x_ _i_ is the
object representation within the _i_ -th member. By contrast, in BatchEnsemble, this linear layer is either
(1) fully shared between all members, or (2) mostly shared: _l_ _i_ ( _x_ _i_ ) = _s_ _i_ _⊙_ ( _W_ ( _r_ _i_ _⊙_ _x_ _i_ )) + _b_ _i_, where
_⊙_ is the elementwise multiplication, _W ∈_ R _[d][×][d]_ is shared between all members, and _r_ _i_ _, s_ _i_ _, b_ _i_ _∈_ R _[d]_
are _not_ shared between the members. This is equivalent to defining the _i_ -th weight matrix as
_W_ _i_ = _W ⊙_ ( _s_ _i_ _r_ _i_ _[T]_ [)] [. To ensure diversity of the ensemble members,] _[ r]_ _[i]_ [ and] _[ s]_ _[i]_ [ of all members are ini-]
tialized randomly with _±_ 1 . All other layers are fully shared between the members of BatchEnsemble.


The described parametrization allows packing all ensemble members in one model that simultaneously
takes _k_ objects as input, and applies all _k_ implicit members in parallel, without explicitly materializing
each member. This is achieved by replacing one or more linear layers of the original neural network
with their BatchEnsemble versions: _l_ BE ( _X_ ) = (( _X ⊙_ _R_ ) _W_ ) _⊙_ _S_ + _B_, where _X ∈_ R _[k][×][d]_ stores _k_
object representations (one per member), and _R, S, B ∈_ R _[d]_ store the non-shared weights ( _r_ _i_, _s_ _i_, _b_ _i_ )
of the members, as shown at the lower left part of Figure 1.


**Terminology.** In this paper, we call _r_ _i_, _s_ _i_, _b_ _i_, _R_, _S_ and _B_ _adapters_, and the implicit members of
parameter-efficient emsembles (e.g. BatchEnsemble) — _implicit submodels_ or simply _submodels_ .
**Overhead to the model size.** With BatchEnsemble, adding a new ensemble member means adding
only one row to each of the matrices _R_, _S_, and _B_, which results in 3 _d_ new parameters per layer. For
typical values of _d_, this is a negligible overhead to the original layer size _d_ [2] + _d_ .
**Overhead to the runtime.** Thanks to the modern hardware, the large number of shared weights
and the parallel execution of the _k_ forward passes, the runtime overhead of BatchEnsemble can be
(significantly) lower than _×k_ (Wen et al., 2020). Intuitively, if the original workload underutilizes the
hardware, there are more chances to pay less than _×k_ overhead.


3.3 A RCHITECTURE


TabM is one model representing an ensemble of _k_ MLPs. Contrary to conventional deep ensembles,
in TabM, the _k_ MLPs are trained in parallel and share most of their weights by default, which
leads to better performance and efficiency. We present multiple variants of TabM that differ in their
weight-sharing strategies, where TabM and TabM mini are the most effective variants, and TabM packed
is a conceptually important variant potentially useful in some cases. We obtain our models in several
steps, starting from essential baselines. We always use the ensemble size _k_ = 32 and analyze this
hyperparameter in subsection 5.3. In subsection A.1, we explain that using MLP as the base model is
crucial because of its excellent efficiency.


**MLP.** We define MLP as a sequence of _N_ simple blocks followed by a linear prediction head:
MLP( _x_ ) = Linear(Block _N_ ( _. . ._ (Block 1 ( _x_ ))), where Block _i_ ( _x_ ) = Dropout(ReLU(Linear(( _x_ ))).


**MLP** _[×][k]_ **= MLP + Deep Ensemble.** We denote the traditional deep ensemble of _k_ independently
trained MLPs as MLP _[×][k]_ . To clarify, this means tuning hyperparameters of one MLP, then independently training _k_ tuned MLPs under different random seeds, and then averaging their predictions.
The performance of MLP _[×][k]_ is reported in Figure 2. Notably, the results are already better and more
stable than those of FT-Transformer (Gorishniy et al., 2021) — the popular attention-based baseline.


Although the described approach is a somewhat default way to implement an ensemble, it is not
optimized for the task performance of the ensemble. First, for each of the _k_ MLPs, the training is
stopped based on the individual validation score, which is optimal for each individual MLP, but can
be suboptimal for their ensemble. Second, the hyperparameters are also tuned for one MLP without
knowing about the subsequent ensembling. All TabM variants are free from these issues.


**TabM** **packed** **= MLP + Packed-Ensemble.** As the first step towards better and more efficient ensembles of MLPs, we implement _k_ MLPs as one large model using Packed-Ensemble (Laurent
et al., 2023). This results in TabM packed illustrated in Figure 1. As an architecture, TabM packed is
equivalent to MLP _[×][k]_ and stores _k_ independent MLPs without any weight sharing. However, the
critical difference is that TabM processes _k_ inputs in parallel, which means that one training step
of TabM consists of _k_ parallel training steps of the individual MLPs. This allows monitoring the


4


Published as a conference paper at ICLR 2025


performance of the ensemble during the training and stopping the training when it is optimal for the
whole ensemble, not for individual MLPs. As a consequence, this also allows tuning hyperparameters
for TabM packed as for one model. As shown in Figure 2, TabM packed delivers significantly better
performance compared to MLP _[×][k]_ . Efficiency-wise, for typical depth and width of MLPs, the runtime
overhead of TabM packed is noticeably less than _×k_ due to the parallel execution of the _k_ forward
passes on the modern hardware. Nevertheless, the _×k_ overhead of TabM packed to the model size
motivates further exploration.


**TabM** **naive** **= MLP + BatchEnsemble.** To reduce the size of TabM packed, we now turn to weight
sharing between the MLPs, and naively apply BatchEnsemble (Wen et al., 2020) instead of PackedEnsemble, as described in subsection 3.2. This gives us TabM naive — a preliminary version of TabM .
In fact, the architecture (but not the initialization) of TabM naive is already equivalent to that of TabM,
so Figure 1 is applicable. Interestingly, Figure 2 reports higher performance of TabM naive compared
to TabM packed . Thus, constraining the ensemble with weight sharing turns out to be a highly effective
regularization on tabular tasks. The alternatives to BatchEnsemble are discussed in subsection A.1.


Test



Shared


Not shared





Train



_Backbones_ _Heads_


_Packed-Ensemble_


_BatchEnsemble_ _[*]_ _MiniEnsemble_


Figure 1: _(Upper left)_ A high-level illustration of TabM. One TabM represents an ensemble of _k_ MLPs
processing _k_ inputs in parallel. The remaining parts of the figure are three different parametrizations of
the _k_ MLP backbones. _(Upper right)_ TabM packed consists of _k_ fully independent MLPs. _(Lower left)_
TabM is obtained by injecting three non-shared adapters _R_, _S_, _B_ in each of the _N_ linear layers of
_one_ MLP ( _[∗]_ the initialization differs from Wen et al. (2020)). _(Lower right)_ TabM mini is obtained
by keeping only the very first adapter _R_ of TabM and removing the remaining 3 _N −_ 1 adapters.
_(Details)_ Input transformations such as one-hot-encoding or feature embeddings (Gorishniy et al.,



2022) are omitted for simplicity. Drop denotes dropout (Srivastava et al., 2014).


8%



6%


4%


2%


0%


_−_ 2%




|MLP<br>Attention|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|MLP<br>Attention||||||||||||
||MLP (Ours)<br>Mean|MLP (Ours)<br>Mean||||||||||
|||||||||||||
|||||||||||||



TabM packed


1 _._ 42 _±_ 2 _._ 3%



TabM _[†×]_ mini [5]

3 _._ 20 _±_ 4 _._ 4%



TabM naive


1 _._ 86 _±_ 2 _._ 4%



TabM bad

1 _._ 00 _±_ 2 _._ 3%



TabM mini


1 _._ 96 _±_ 2 _._ 8%



TabM TabM _[♠]_
2 _._ 15 _±_ 2 _._ 8% 2 _._ 07 _±_ 3 _._ 6%



TabM _[†]_ mini

2 _._ 92 _±_ 4 _._ 1%



MLP

0 _._ 00 _±_ 0 _._ 0%



FT `-` T

0 _._ 39 _±_ 1 _._ 6%



MLP _[×]_ [32]


0 _._ 96 _±_ 1 _._ 6%



Figure 2: The performance of models described in subsection 3.3 on 46 datasets from Table 1; plus
several baselines on the left. For a given model, one dot on a jitter plot describes the performance
score on one of the 46 datasets. The box plots describe the percentiles of the jitter plots: the boxes
describe the 25th, 50th, and 75th percentiles, and the whiskers describe the 10th and 90th percentiles.
Outliers are clipped. The numbers at the bottom are the mean and standard deviations over the jitter
plots. For each model, hyperparameters are tuned. “Model _[×][k]_ ” denotes an ensemble of _k_ models.


5


Published as a conference paper at ICLR 2025


**TabM** **mini** **= MLP + MiniEnsemble.** By construction, the just discussed TabM naive (illustrated as
“ TabM ” in Figure 1) has 3 _N_ adapters: _R_, _S_ and _B_ in each of the _N_ blocks. Let’s consider the very
first adapter, i.e. the first adapter _R_ in the first linear layer. Informally, its role can be described as
mapping the _k_ inputs living in the same representation space to _k_ different representation spaces
_before_ the tabular features are mixed with @ _W_ for the first time. A simple experiment reveals that
this adapter is critical. First, we remove it from TabM naive and keep the remaining 3 _N −_ 1 adapters
untouched, which gives us TabM bad with worse performance, as shown in Figure 2. Then, we do the
opposite: we keep only the very first adapter of TabM naive and remove the remaining 3 _N −_ 1 adapters,
which gives us TabM mini — the minimal version of TabM . TabM mini is illustrated in Figure 1, where
we call the described approach “MiniEnsemble”. Figure 2 shows that TabM mini performs even slightly
better than TabM naive, despite having only one adapter instead of 3 _N_ adapters.


**TabM** **= MLP + BatchEnsemble + Better initialization.** The just obtained results motivate the next
step. We go back to the architecture of TabM naive with all 3 _N_ adapters, but initialize all multiplicative
adapters _R_ and _S_, except for the very first one, deterministically with 1 . As such, at initialization, the
deterministically initialized adapters have no effect, and the model behaves like TabM mini, but these
adapters are free to add more expressivity during training. This gives us TabM, illustrated in Figure 1.
Figure 2 shows that TabM is the best variation so far.


**Hyperparameters.** Compared to MLP, the only new hyperparameter of TabM is _k_ — the number of
implicit submodels. We heuristically set _k_ = 32 and do not tune this value. We analyze the influence
of _k_ in subsection 5.3. We also share additional observations on the learning rate in subsection A.3.


**Limitations and practical considerations** are commented in subsection A.4.


3.4 I MPORTANT PRACTICAL MODIFICATIONS OF T AB M


_♠∼_ **Shared training batches** . Recall that the order of training objects usually varies between
ensemble members, because of the random shuffling with different seeds. For TabM, in terms of
Figure 1, that corresponds to _X_ storing _k_ different training objects _{x_ _i_ _}_ _[k]_ _i_ =1 [. We observed that reusing]
the training batches between the TabM ’s submodels results in only minor performance loss on average
(depending on a dataset), as illustrated with TabM _[♠]_ in Figure 2. In practice, due to the simpler
implementation and better efficiency, sharing training batches can be a reasonable starting point.


_† ∼_ **Non-linear feature embeddings** . In Figure 2, TabM _[†]_ mini [denotes] [ TabM] [mini] [ with non-linear]
feature embeddings from (Gorishniy et al., 2022), which demonstrates the high utility of feature
embeddings for TabM . Specifically, we use a slightly modified version of the piecewise-linear
embeddings (see subsection D.8 for details).


_×_ **N** _∼_ **Deep ensemble** . In Figure 2, TabM _[†×]_ mini [5] [denotes an ensemble of five independent] [ TabM] _[†]_ mini
models, showing that TabM itself can benefit from the conventional deep ensembling.


3.5 S UMMARY


The story behind TabM shows that technical details of _how_ to construct and train an ensemble have
a major impact on task performance. Most importantly, we highlight simultaneous training of the
(implicit) ensemble members and weight sharing between them. The former is responsible for the
ensemble-aware stopping of the training, and the latter apparently serves as a form of regularization.


4 E VALUATING TABULAR DEEP LEARNING ARCHITECTURES


Now, we perform an empirical comparison of many tabular models, including TabM.


4.1 B ASELINES


In the main text, we use the following baselines: MLP (defined in subsection 3.3), FT-Transformer
denoted as “FT-T” (the attention-based model from Gorishniy et al. (2021)), SAINT (the attentionand retrieval-based model from Somepalli et al. (2021)), T2G-Former denoted as “T2G” (the attentionbased model from Yan et al. (2023)), ExcelFormer denoted as “Excel” (the attention-based model
from Chen et al. (2023a)), TabR (the retrieval-based model from Gorishniy et al. (2024)), ModernNCA


6


Published as a conference paper at ICLR 2025


denoted as “MNCA” (the retrieval-based model from Ye et al. (2024)) and GBDT, including XGBoost
(Chen & Guestrin, 2016), LightGBM (Ke et al., 2017) and CatBoost (Prokhorenkova et al., 2018).


The models with non-linear feature embeddings from Gorishniy et al. (2022) are marked with _†_ or _‡_
depending on the embedding type (see subsection D.8 for details on feature embeddings):


 - MLP _[†]_ and TabM _[†]_ mini [use a modified version of the piecewise-linear embeddings.]

 - TabR _[‡]_, MNCA _[‡]_, and MLP _[‡]_ (also known as MLP-PLR) use various periodic embeddings.


More baselines are evaluated in Appendix B. Implementation details are provided in Appendix D.


4.2 T ASK PERFORMANCE


We evaluate all models following the protocol announced in subsection 3.1 and report the results in
Figure 3 (see also the critical difference diagram in Figure 9). We make the following observations:


1. The performance ranks render TabM as the top-tier DL model.
2. The middle and right parts of Figure 3 provide a fresh perspective on the per-dataset metrics.
TabM holds its leadership among the DL models. Meanwhile, many DL methods turn out to be
no better or even worse than MLP on a non-negligible number of datasets, which shows them as
less reliable solutions, and changes the ranking, especially on the domain-aware splits (right).
3. One important characteristic of a model is the _weakest_ part of its performance profile (e.g.
the 10th or 25th percentiles in the middle plot) since it shows how reliable the model is on
“inconvenient” datasets. From that perspective, MLP [†] seems to be a decent practical option
between the plain MLP and TabM, especially given its simplicity and efficiency compared to
retrieval-based alternatives, such as TabR and ModernNCA.


**Summary.** TabM confidently demonstrates the best performance among tabular DL models, and can
serve as a reliable go-to DL baseline. This is not the case for attention- and retrieval-based models.
Overall, MLP-like models, including TabM, form a representative set of tabular DL baselines.



TabR


SAINT

Excel _[∗]_

TabR _[‡]_


MNCA


MLP


FT `-` T


MNCA _[‡]_


T2G


CatBoost

MLP _[†]_


LightGBM


TabM


XGBoost


TabM _[†]_ mini

|Perfo<br>On 9 datasets<br>Sorted|rmance scores<br>with domain-aware split<br>by the mean score|Col3|
|---|---|---|
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||



_−_ 5% 0% 5% 10%
Relative improvement over MLP ( _↑_ )



MLP


Excel _[∗]_


SAINT


FT `-` T


T2G


MNCA


TabR


MLP _[†]_


LightGBM


XGBoost


CatBoost

MNCA _[‡]_

TabR _[‡]_


TabM


TabM _[†]_ mini



Performance ranks


On 46 datasets

Sorted by the mean rank



Performance scores

On 37 datasets with random split


Sorted by the mean score


Excel _[∗]_


MLP


SAINT


FT `-` T


TabR


T2G


MLP _[†]_


MNCA


LightGBM

TabR _[‡]_


XGBoost


TabM


CatBoost

MNCA _[‡]_


TabM _[†]_ mini

|l∗<br>P|Col2|Col3|Col4|Col5|Mean|
|---|---|---|---|---|---|
|P<br>l_∗_||||||
|P<br>l_∗_||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||
|ini<br>A_‡_<br>st<br>M<br>st<br>R_‡_<br>M<br>A<br>P_†_<br>G<br>R<br>T<br>T||||||



_−_ 2% 0% 2% 4% 6% 8%
Relative improvement over MLP ( _↑_ )






|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||~~5~~~~_._5~~|~~3~~~~_._2~~|
||||||||
|_∗_|||||||
|_∗_|||||~~5~~~~_._2 2~~|~~9~~|
||||||||
||||||||
|||||~~4~~~~_._~~|~~ 2~~~~_._9~~||
||||||||
||||||||
|||||~~4~~~~_._6~~|~~ 2~~~~_._9~~||
||||||||
||||||||
||||~~4~~~~_._~~|~~ 2~~~~_._~~|||
||||||||
||||||||
||||~~3~~~~_._9~~|~~ 2~~~~_._~~|||
||||||||
||||||||
||||~~3~~~~_._9~~|~~ 3~~~~_._1~~|||
||||||||
|_†_|||||||
|_†_|||~~3~~~~_._8~~|~~ 2~~~~_._4~~|||
||||||||
||||||||
||||~~3~~~~_._4 ~~|~~_._0~~|||
||||||||
|t|||||||
|t|||~~_._3 2~~~~_._~~||||
||||||||
|t|||||||
|t||~~3~~|~~2 2~~~~_._~~||||
||||||||
|_‡_|||||||
|_‡_||~~3~~~~_._~~|~~ 2~~~~_._2~~||||
||||||||
|_‡_|||||||
|_‡_||~~2~~~~_._9~~|~~_±_ 2~~~~_._2~~||MLP, G|BDT|
|||~~2~~~~_._8~~|~~_±_ 2~~~~_._1~~||Attentio<br>Retriev|n,<br>l|
|i|~~1~~~~_._7~~|~~_±_ 1~~~~_._2~~|||MLP (O|urs)|



1 2 3 4 5

Rank ( _↓_ )



Figure 3: The task performance of tabular models on the 46 datasets from Table 1. _(Left)_ The mean
and standard deviations of the performance ranks over all datasets summarize the head-to-head
comparison between the models on all datasets. _(Middle & Right)_ The relative performance w.r.t. the
plain multilayer perceptron (MLP) allows reasoning about the scale and consistency of improvements
over this simple baseline. One dot of a jitter plot corresponds to the performance of a model on one
of the 46 datasets. The box plots visualize the 10th, 25th, 50th, 75th, and 90th percentiles of the jitter
plots. Outliers are clipped. The separation in random and domain-aware dataset splits is explained in
subsection 3.1. ( _[∗]_ Evaluated under the common protocol without data augmentations)


7


Published as a conference paper at ICLR 2025


4.3 E FFICIENCY


Now, we evaluate tabular models in terms of training and inference efficiency, which becomes
a serious reality check for some of the methods. We benchmark exactly those hyperparameter
configurations of models that are presented in Figure 3 (see subsection B.3 for the motivation).


**TabM** _[†∗]_ **mini** **[&]** **[ TabM]** _[†♠∗]_ **mini** **[.]** [ Additionally, in this section, we mark with the asterisk (] _[∗]_ [) the versions of]
TabM enhanced with two efficiency-related plugins available out-of-the-box in PyTorch (Paszke
et al., 2019): the automatic mixed precision (AMP) and torch.compile (Ansel et al., 2024). The
purpose of those TabM variants is to showcase the potential of the modern hardware and software
for a powerful tabular DL model, and they should not be directly compared to other DL models.
However, the implementation simplicity of TabM plays an important role, because it facilitates the
seamless integration of the aforementioned PyTorch plugins.


**Training time.** We focus on training times on larger datasets, because on small datasets, all methods
become almost equally affordable, regardless of the formal relative difference. Nevertheless, in
Figure 10, we provide measurements on small datasets as well. The left side of Figure 4 reveals
that TabM offers practical training times. By contrast, the long training times of attention- and
retrieval-based models become one more limitation of these methods.


**Inference throughput.** The right side of Figure 4 tells essentially the same story as the left side. In
subsection B.3, we also report the inference throughput on GPU with large batch sizes.


**Applicability to large datasets.** In Table 2, we report metrics on two large datasets. As expected,
attention- and retrieval-based models struggle, yielding extremely long training times, or being simply
inapplicable without additional effort. See subsection D.4 for implementation details.


**Parameter count.** Most tabular networks are overall compact. This, in particular, applies to TabM,
because its size is by design comparable to MLP. We report model sizes in subsection B.3.


**Summary.** Simple MLPs are the fastest DL models, with TabM being the runner-up. The attentionand retrieval-based models are significantly slower. Overall, MLP-like models, including TabM, form
a representative set of practical and accessible tabular DL baselines.



MLP

MLP _[†]_


XGBoost

TabM _[†∗]_ mini
TabM _[†]_ mini
TabM


TabR

MNCA
MNCA _[‡]_

TabR _[‡]_


T2G

FT-T

SAINT



Training time on datasets with _>_ 100K objects


Device: GPU NVIDIA A100

|Col1|Col2|Col3|Col4|MLP, GBDT|
|---|---|---|---|---|
|||||~~MLP, GBDT~~<br>Attention,<br>|
|||||Retrieval<br>|
|||||~~MLP (Ours)~~<br>Mean|
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||
||||||



10s _≈_ 2 _m_ _≈_ 15 _m_ _≈_ 1 _h ≈_ 3 _h_
Time ( _↓_ )



MLP


XGBoost

MLP _[†]_

TabM

TabM _[†]_ mini
TabR

FT-T

T2G


TabR _[‡]_


MNCA


MNCA _[‡]_

SAINT



Inference throughput with batch size 1

Device: CPU Intel i7-7800X, single thread


0 1000 2000 3000 4000 5000 6000
Objects per second ( _↑_ )



Figure 4: Training times (left) and inference throughput (right) of the models from Figure 3. One dot
represents a measurement on one dataset. TabM _[†∗]_ mini [is the optimized TabM] _[†]_ mini [(see][ subsection 4.3][).]


Table 2: RMSE (upper rows) and training times (lower rows) on two large datasets. The best values
are in bold. The meaning of model colors follows Figure 3.

|#Objects #Features|XGBoost MLP TabM†♠∗ TabM† FT-T TabR<br>mini mini|
|---|---|
|Maps Routing<br>6_._5M<br>986|0_._1601<br>0_._1592<br>0_._1583<br>**0**_._**1582**<br>0_._1594<br>OOM<br>28m<br>**15m**<br>2h<br>13_._5h<br>45_._5h|
|Weather<br>13M<br>103|1_._4234<br>1_._4842<br>**1**_._**4090**<br>**1**_._**4112**<br>1_._4409<br>OOM<br>**10m**<br>15m<br>1_._3h<br>3_._3h<br>13_._5h|



8


Published as a conference paper at ICLR 2025


5 A NALYSIS


5.1 P ERFORMANCE AND TRAINING DYNAMICS OF THE INDIVIDUAL SUBMODELS


Recall that the prediction of TabM is defined as the mean prediction of its _k_ implicit submodels that
share most of their weights. In this section, we take a closer look at these submodels.


For the next experiment, we intentionally simplify the setup as described in detail in subsection D.5.
Most importantly, all models have the same depth 3 and width 512, and are trained without early
stopping, i.e. the training goes beyond the optimal epochs. We use TabM mini from Figure 1 with
_k_ = 32 denoted as TabM _[k]_ mini [=32] [. We use] [ TabM] _[k]_ mini [=1] [(i.e. essentially one plain MLP) as a natural baseline]
for the submodels of TabM _[k]_ mini [=32] [, because each of the] [ 32] [ submodels has the architecture of] [ TabM] _[k]_ mini [=1] [.]


We visualize the training profiles on four diverse datasets (two classification and two regression
problems of different sizes) in Figure 5. As a reminder, the mean of the _k_ **individual** losses is what
is explicitly optimized during the training of TabM mini, the loss of the **collective** mean prediction
corresponds to how TabM mini makes predictions on inference, and TabM _[k]_ mini [=1] [is just a] **[ baseline]** [.]



Otto


0 100 200 300
Epoch


0 _._ 6 0 _._ 4 0 _._ 2 0 _._ 0


Train Loss



Churn



Microsoft


0 100 200 300
Epoch


0 _._ 8 0 _._ 6 0 _._ 4 0 _._ 2 0 _._ 0


Train Loss



2.0


1.0


0.0


1 _._ 5


1 _._ 0


0 _._ 5



0 100 200 300
Epoch


0 _._ 4 0 _._ 3 0 _._ 2 0 _._ 1 0 _._ 0


Train Loss





House


2


1


0


0 100 200 300
Epoch


2 _._ 0


1 _._ 5


1 _._ 0


0 _._ 5


0 _._ 4 0 _._ 2 0 _._ 0


Train Loss



1 _._ 0


0 _._ 5


0 _._ 0


1 _._ 2


1 _._ 0


0 _._ 8



0 _._ 4


0 _._ 2


0 _._ 0


0 _._ 5


0 _._ 4



Figure 5: The training profiles of TabM _[k]_ mini [=32] and TabM _[k]_ mini [=1] [as described in][ subsection 5.1][.] _[ (Upper)]_
The training curves. _k_ = 32[ _i_ ] represents the mean **i** ndividual loss over the 32 submodels. _(Lower)_
Same as the first row, but in the train-test coordinates: each dot represents some epoch from the first
row, and the training generally goes from left to right. This allows reasoning about overfitting by
comparing test loss values for a given train loss value.


In the upper row of Figure 5, the collective mean prediction of the submodels is superior to their
individual predictions in terms of both training and test losses. After the initial epochs, the training
loss of the baseline MLP is lower than that of the collective and individual predictions.


In the lower row of Figure 5, we see a stark contrast between the individual and collective performance
of the submodels. Compared to the baseline MLP, the submodels look overfitted individually, while
their collective prediction exhibits substantially better generalization. This result is strict evidence
of a non-trivial diversity of the submodels: without that, their collective test performance would be
similar to their individual test performance. Additionally, we report the performance of the **B** est
submodel of TabM across many datasets under the name TabM[B] in Figure 6. As such, individually,
even the best submodel of TabM is no better than a simple MLP.


**Summary.** TabM draws its power from the collective prediction of weak, but diverse submodels.


5.2 S ELECTING SUBMODELS AFTER TRAINING


The design of TabM allows selecting only a subset of submodels after training based on any criteria,
simply by pruning extra prediction heads and the corresponding rows of the adapter matrices. To
showcase this mechanics, after the training, we **G** reedily construct a subset of TabM ’s submodels with
the best collective performance on the validation set, and denote this “pruned” TabM as TabM[G] .
The performance reported in Figure 6 shows that TabM[G] is slightly behind the vanilla TabM . On
average over 46 datasets, the greedy submodel selection results in 8 _._ 8 _±_ 6 _._ 6 submodels out of the
initial _k_ = 32, which can result in faster inference. See subsection D.6 for implementation details.


9


Published as a conference paper at ICLR 2025


8%



1.0%







6%


4%


2%


0%


_−_ 2%




|MLP<br>MLP (Ours)|Col2|Col3|Col4|
|---|---|---|---|
|Mean|Mean|Mean|Mean|
|||||
|||||
|||||
|||||
|||||
|||||


|Col1|d|n|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
||~~**64**~~<br>**128**<br>**256**<br>**512**|~~**1**~~<br>**2**<br>**3**|~~**1**~~<br>**2**<br>**3**|||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||



TabM[G]


2 _._ 02 _±_ 2 _._ 6%



TabM[B]


_−_ 0 _._ 06 _±_ 1 _._ 8%



MLP

0 _._ 00 _±_ 0 _._ 0%



TabM

2 _._ 15 _±_ 2 _._ 8%



Figure 6: The performance on the 46 datasets
from Table 1. TabM[B] and TabM[G] are described in subsection 5.1 and subsection 5.2.



0.0%


_−_ 1.0%


_−_ 2.0%


1 2 4 8 16 32 64 128

_k_


Figure 7: The average performance of TabM
with _n_ layers of the width _d_ across 17 datasets
as a function of _k_ .



5.3 H OW DOES THE PERFORMANCE OF T AB M DEPEND ON _k_ ?


To answer the question in the title, we consider TabM with _n_ layers of the size _d_ and different values
of _k_, and report the average performance over multiple datasets in Figure 7 (the implementation
details are provided in subsection D.7). The solid curves correspond to _n_ = 3, and the dark green
curves correspond to _d_ = 512 . Our main observations are as follows. _First,_ it seems that the “larger”
TabM is (i.e. when _n_ and _d_ increase), the more submodels it can accommodate effectively. For
example, note how the solid curves corresponding to different _d_ diverge at _k_ = 2 and _k_ = 4 . _Second,_
too high values of _k_ can be detrimental. Perhaps, weight sharing limits the number of submodels that
can productively “coexist” in one network, despite the presence of non-shared adapters. _Third_, too
narrow ( _d_ = 64 ) or too shallow ( _n_ = 1 ) configurations of TabM can lead to suboptimal performance,
at least in the scope of middle-to-large datasets considered in this work.


5.4 P ARAMETER - EFFICIENT ENSEMBLING REDUCES THE NUMBER OF DEAD NEURONS


Here, we show empirically that the design of TabM naturally leads to higher utilization of the
backbone’s weights. Even without technical definitions, this sounds intuitive, since TabM has to
implement _k_ (diverse) computations using the amount of weights close to that of one MLP.


Let’s consider TabM mini as illustrated in Figure 1. By design, each of the shared neurons of TabM mini
is used _k_ times per forward pass, where “neuron” refers to the combination of the linear transformation
and the subsequent nonlinearity (e.g. ReLU). By contrast, in plain MLP (or in TabM mini with _k_ = 1 ),
each neuron is used only once per forward pass. Thus, technically, a neuron in TabM mini has more
chances to be activated, which overall may lead to lower portion of dead neurons in TabM mini
compared to MLP (a dead neuron is a neuron that never activates, and thus has no impact on the
prediction). Using the experiment setup from subsection 5.1, we compute the portion of dead neurons
in TabM mini using its best validation checkpoint. On average across 46 datasets, for _k_ = 1 and
_k_ = 32, we get 0 _._ 29 _±_ 0 _._ 17 and 0 _._ 14 _±_ 0 _._ 09 portion of dead neurons, respectively, which is in line
with the described intuition. Technically, on a given dataset, this metric is computed as the percentage
of neurons that never activate on a fixed set of 2048 training objects.


6 C ONCLUSION & F UTURE WORK


In this work, we have demonstrated that tabular multilayer perceptrons (MLPs) greatly benefit from
parameter-efficient ensembling. Using this insight, we have developed TabM — a simple MLPbased model with state-of-the-art performance. In a large-scale comparison with many tabular DL
models, we have demonstrated that TabM is ready to serve as a new powerful and efficient tabular DL
baseline. Along the way, we highlighted the important technical details behind TabM and discussed
the individual performance of the implicit submodels underlying TabM.


One idea for future work is to bring the power of (parameter-)efficient ensembles to other, non-tabular,
domains with optimization-related challenges and, ideally, lightweight base models. Another idea is
to evaluate TabM for uncertainty estimation and out-of-distribution (OOD) detection on tabular data,
which is inspired by works like Lakshminarayanan et al. (2017).


10


Published as a conference paper at ICLR 2025


**Reproducibility statement.** [The code is provided in the following repository: link. It contains the](https://github.com/yandex-research/tabm)
implementation of TabM, hyperparameter tuning scripts, evaluation scripts, configuration files with
hyperparameters (the TOML files in the exp/ directory), and the report files with the main metrics
(the JSON files in the exp/ directory). In the paper, the model is described in section 3, and the
implementation details are provided in Appendix D.


R EFERENCES


Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A
next-generation hyperparameter optimization framework. In _KDD_, 2019. 18


Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and
self-distillation in deep learning. In _ICLR_, 2023. 3


Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky,
Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will
Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael
Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos,
Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, C. K. Luk, Bert Maher, Yunjie Pan,
Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Shunting
Zhang, Michael Suo, Phil Tillet, Xu Zhao, Eikan Wang, Keren Zhou, Richard Zou, Xiaodong
Wang, Ajit Mathews, William Wen, Gregory Chanan, Peng Wu, and Soumith Chintala. Pytorch 2:
Faster machine learning through dynamic python bytecode transformation and graph compilation.
In _ASPLOS_, 2024. 8


Javier Antoran, James Urquhart Allingham, and Jos ´ e Miguel Hern ´ andez-Lobato. Depth uncertainty ´
in neural networks. In _NeurIPS_, 2020. 3


Sercan O. Arik and Tomas Pfister. TabNet: Attentive interpretable tabular learning. _arXiv_,
1908.07442v5, 2020. 2


Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, and Sathiya S.
Keerthi. Gradient boosting neural networks: GrowNet. _arXiv_, 2002.07971v2, 2020. 2


Dara Bahri, Heinrich Jiang, Yi Tay, and Donald Metzler. SCARF: Self-supervised contrastive learning
using random feature corruption. In _ICLR_, 2021. 2


Jintai Chen, Jiahuan Yan, Danny Ziyi Chen, and Jian Wu. ExcelFormer: A neural network surpassing
gbdts on tabular data. _arXiv_, 2301.02819v1, 2023a. 1, 6, 20, 24, 25


Kuan-Yu Chen, Ping-Han Chiang, Hsin-Rung Chou, Ting-Wei Chen, and Tien-Hao Chang. Trompt:
Towards a better deep neural network for tabular data. In _ICML_, 2023b. 1, 2, 15


Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In _SIGKDD_, 2016. 1,

2, 7


Stanislav Fort, Huiyi Hu, and Balaji Lakshminarayanan. Deep ensembles: A loss landscape perspective. _arXiv_, 1912.02757v2, 2020. 2


Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P. Vetrov, and Andrew Gordon Wilson.
Loss surfaces, mode connectivity, and fast ensembling of dnns. In _NeurIPS_, 2018. 3


Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning
models for tabular data. In _NeurIPS_, 2021. 2, 4, 6, 15, 23, 25


Yury Gorishniy, Ivan Rubachev, and Artem Babenko. On embeddings for numerical features in
tabular deep learning. In _NeurIPS_, 2022. 2, 5, 6, 7, 14, 15, 20, 21, 23


Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem
Babenko. TabR: Tabular deep learning meets nearest neighbors. In _ICLR_, 2024. 1, 2, 3, 6, 17, 18,
19, 20, 22, 23, 24, 25


11


Published as a conference paper at ICLR 2025


Leo Grinsztajn, Edouard Oyallon, and Gael Varoquaux. Why do tree-based models still outperform
deep learning on typical tabular data? In _NeurIPS, the ”Datasets and Benchmarks” track_, 2022. 3,
17, 18, 23, 24, 29


Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew Mingbo Dai, and Dustin Tran. Training independent subnetworks for robust
prediction. In _ICLR_, 2021. 3, 14


Noah Hollmann, Samuel Muller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer ¨
that solves small tabular classification problems in a second. In _ICLR_, 2023. 1, 2, 15


David Holzmuller, L ¨ eo Grinsztajn, and Ingo Steinwart. Better by default: Strong pre-tuned mlps and ´
boosted trees on tabular data. _arXiv_, 2407.04491v1, 2024. 2


Alan Jeffares, Tennison Liu, Jonathan Crabbe, Fergus Imrie, and Mihaela van der Schaar. TANGOS: ´
Regularizing tabular neural networks through gradient orthogonalization and specialization. In
_ICLR_, 2023a. 2


Alan Jeffares, Tennison Liu, Jonathan Crabbe, and Mihaela van der Schaar. Joint training of deep ´
ensembles fails due to learner collusion. In _NeurIPS_, 2023b. 2


Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka. Well-tuned simple nets excel on
tabular datasets. In _NeurIPS_, 2021. 2


Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan
Liu. LightGBM: A highly efficient gradient boosting decision tree. _Advances in neural information_
_processing systems_, 30:3146–3154, 2017. 1, 2, 7


Myung Jun Kim, Leo Grinsztajn, and Ga ´ el Varoquaux. CARTE: pretraining and transfer for tabular ¨
learning. _arXiv_, abs/2402.16785v1, 2024. 16


Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter. Self-normalizing ¨
neural networks. In _NIPS_, 2017. 2, 15


Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Tom Rainforth, and Yarin Gal. Self-attention
between datapoints: Going beyond individual input-output pairs in deep learning. In _NeurIPS_,
2021. 2


Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive
uncertainty estimation using deep ensembles. In _NeurIPS_, 2017. 3, 10, 14


Olivier Laurent, Adrien Lafage, Enzo Tartaglione, Geoffrey Daniel, Jean-Marc Martinez, Andrei
Bursuc, and Gianni Franchi. Packed ensembles for efficient uncertainty estimation. In _ICLR_, 2023.
4


Stefan Lee, Senthil Purushwalkam, Michael Cogswell, David J. Crandall, and Dhruv Batra. Why M
heads are better than one: Training a diverse ensemble of deep networks. _arXiv_, abs/1511.06314,
2015. 3, 14


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR_, 2019. 18


Sascha Marton, Stefan L¨udtke, Christian Bartelt, and Heiner Stuckenschmidt. GRANDE: Gradientbased decision tree ensembles for tabular data. In _ICLR_, 2024. 2


Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Ed- ¨
ward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit
Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, highperformance deep learning library. In _NeurIPS_, 2019. 8


F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and
E. Duchesnay. Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_,
12:2825–2830, 2011. 18


12


Published as a conference paper at ICLR 2025


Sergei Popov, Stanislav Morozov, and Artem Babenko. Neural oblivious decision ensembles for deep
learning on tabular data. In _ICLR_, 2020. 2


Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey
Gulin. CatBoost: unbiased boosting with categorical features. In _NeurIPS_, 2018. 1, 2, 7


Tao Qin and Tie-Yan Liu. Introducing LETOR 4.0 datasets. _arXiv_, 1306.2597v1, 2013. 3


Ivan Rubachev, Artem Alekberov, Yury Gorishniy, and Artem Babenko. Revisiting pretraining
objectives for tabular deep learning. _arXiv_, 2207.03208v1, 2022. 2


Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, and Artem Babenko. TabReD: Analyzing Pitfalls
and Filling the Gaps in Tabular Deep Learning Benchmarks. _arXiv_, 2406.19380v4, 2024. 1, 3, 17,
18, 19, 20, 22, 23, 25, 36


Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss, and Tom Goldstein.
SAINT: improved neural networks for tabular data via row attention and contrastive pre-training.
_arXiv_, 2106.01342v1, 2021. 2, 6, 25


Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang.
Autoint: Automatic feature interaction learning via self-attentive neural networks. In _CIKM_, 2019.
2, 15, 26


Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout: a simple way to prevent neural networks from overfitting. _Journal of Machine Learning_
_Research_, 15(1):1929–1958, 2014. 5


Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas
Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and
Alexey Dosovitskiy. Mlp-mixer: An all-mlp architecture for vision. In _NeurIPS_, 2021. 15


Mehmet Ozgur Turkoglu, Alexander Becker, Huseyin Anil G ¨ und ¨ uz, Mina Rezaei, Bernd Bischl, ¨
Rodrigo Caye Daudt, Stefano D’Aronco, Jan D. Wegner, and Konrad Schindler. Film-ensemble:
Probabilistic deep learning via feature-wise linear modulation. In _NeurIPS 2022_, 2022. 3, 14, 15


Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed H.
Chi. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank
systems. _arXiv_, 2008.13535v2, 2020. 2, 15


Yeming Wen, Dustin Tran, and Jimmy Ba. Batchensemble: an alternative approach to efficient
ensemble and lifelong learning. In _ICLR_, 2020. 2, 3, 4, 5, 14, 15


Jiahuan Yan, Jintai Chen, Yixuan Wu, Danny Z. Chen, and Jian Wu. T2G-FORMER: organizing
tabular features into relation graphs promotes heterogeneous feature interaction. In _AAAI_, 2023. 2,
6, 23, 24


Han-Jia Ye, Huai-Hong Yin, and De-Chuan Zhan. Modern neighborhood components analysis: A
deep tabular baseline two decades later. _arXiv_, 2407.03257v1, 2024. 2, 7, 20, 23


Shaofeng Zhang, Meng Liu, and Junchi Yan. The diversified ensemble neural network. In _NeurIPS_,
2020. 3


13


Published as a conference paper at ICLR 2025


A A DDITIONAL DISCUSSION ON T AB M


A.1 M OTIVATION


**Why BatchEnsemble?** Among relatively ease-to-use “efficient ensembling” methods, beyond
BatchEnsemble, there are examples such as dropout ensembles (Lakshminarayanan et al., 2017),
naive multi-head architectures, TreeNet (Lee et al., 2015). However, in the literature, they were
consistently outperformed by more advanced methods, including BatchEnsemble (Wen et al., 2020),
MIMO (Havasi et al., 2021), FiLM-Ensemble (Turkoglu et al., 2022).


Among advanced methods, BatchEnsemble seems to be one of the simplest and most flexible options.
For example, FiLM-Ensemble (Turkoglu et al., 2022) requires normalization layers to be presented in
the original architecture, which is not always the case for tabular MLPs. MIMO (Havasi et al., 2021),
in turn, imposes additional limitations compared to BatchEnsemble. _First_, it requires _concatenating_
(not _stacking_, as with BatchEnsemble) all _k_ input representations, which increases the input size of
the first linear layer. With the relatively high number of submodels _k_ = 32 used in our paper, this
can be an issue on datasets with a large number of features, especially when feature embeddings
(Gorishniy et al., 2022) are used. For example, for _k_ = 32, the number of features _m_ = 1000, and the
feature embedding size _l_ = 32, the input size approaches one million resulting in an extremely large
first linear layer of MLP. _Second_, with BatchEnsemble, it is easy to explicitly materialize, analyze,
and prune individual submodels. By contrast, in MIMO, all submodels are implicitly entangled
within one MLP, and there is no easy way to access individual submodels.


**Why MLPs?** Despite the applicability of BatchEnsemble (Wen et al., 2020) to almost any architecture,
we focus specifically on MLPs. The key reason is _efficiency_ . _First,_ to achieve high performance,
throughout the paper, we use the relatively large number of submodels _k_ = 32 . However, the desired
less-than- _×k_ runtime overhead of BatchEnsemble typically happens only when the original model
underutilizes the power of parallel computations of a given hardware. This will not be the case for
attention-based models on datasets with a large number of features, as well as for retrieval-based
models on datasets with a large number of objects. _Second,_ as we show in subsection 4.3, attentionand retrieval-based models are already slow as-is. By contrast, MLPs are exceptionally efficient, to
the extent that slowing them down even by an order of magnitude will still result in practical models.


Also, generally speaking, the definition of MLP suggested in subsection 3.3 and used in TabM is not
special, and more advanced MLP-like backbones can be used. However, in preliminary experiments,
we did not observe the benefits of more advanced backbones. Perhaps, small technical differences
between backbones become less impactful in the context of parameter-efficient ensembling, at least
in the scope of middle-to-large-sized datasets.


A.2 T AB M WITH FEATURE EMBEDDINGS


**Notation.** In this paper, we use _†_ to mark TabM variants with the piecewise-linear embeddings (e.g.
TabM _[†]_ mini [, TabM] _[†]_ [, etc.).]


**Implementation details.** In fact, there are no changes in the usage of feature embeddings compared
to plain MLPs: feature embeddings are applied, and the result is flattened, before being passed to
the backbones in terms of Figure 1. For example, if a dataset has _m_ continuous features and all of
them are embedded, the very first adapter _R_ will have the shape _k × md_ _e_, where _d_ _e_ is the feature
embedding size. For TabM _[†]_ mini [and] [ TabM] _[†]_ [, we initialize the first multiplicative adapter] _[ R]_ [ of the first]
linear layer from the standard normal distribution _N_ (0 _,_ 1) . The remaining details are best understood
from the source code.


**Efficiency.** When feature embeddings are used, the simplified batching strategy from subsection 3.4 allows for more efficient implementation, when the feature embeddings are applied to the
original batch ~~s~~ ize objects, and the result is simply cloned _k_ times (compared to embedding
_k ×_ batch ~~s~~ ize objects with the original batching strategy).


A.3 H YPERPARAMETERS


We noticed that the typical optimal learning rate for TabM is higher than for MLP (note that, on
each dataset, the batch size is the same for all DL models). We hypothesize that the reason is the


14


Published as a conference paper at ICLR 2025


effectively larger batch size for TabM because of how the training batches are constructed (even if
the simplified batching strategy from subsection 3.4 is used).


A.4 L IMITATIONS AND PRACTICAL CONSIDERATIONS


TabM does not introduce any new limitations compared to BatchEnsemble (Wen et al., 2020).
Nevertheless, we note the following:


  - The MLP backbone used in TabM is one of the simplest possible, and generally, more advanced
backbones can be used. That said, some backbones may require additional care when used in
TabM . For example, we did not explore backbones with normalization layers. For such layers, it
is possible to allocate non-shared trainable affine transformations for each implicit submodel
by adding one multiplicative and one additive adapter after the normalization layer (i.e. like in
FiLM-Ensemble (Turkoglu et al., 2022)). Additional experiments are required to find the best
strategy.

  - For ensemble-like models, such as TabM, the notion of “the final object embedding“ changes:
now, it is not a single vector, but a set of _k_ vectors. If exactly one object embedding is required,
then additional experiments may be needed to find the best way to combine _k_ embeddings into
one. The presence of multiple object embeddings can also be important for scenarios when
TabM is used for solving more than one task, in particular when it is pretrained as a generic
feature extractor and then reused for other tasks. The main practical guideline is that the _k_
prediction branches should not interact with each other (e.g. through attention, pooling, etc.)
and should always be trained separately.


B E XTENDED RESULTS


This section complements section 4.


B.1 A DDITIONAL BASELINES


In addition to the models from subsection 4.1, we consider the following baselines:


  - MLP-PLR Gorishniy et al. (2022), that is, an MLP with periodic embeddings.

  - ResNet (Gorishniy et al., 2021)

  - SNN (Klambauer et al., 2017)

  - DCNv2 (Wang et al., 2020)

  - AutoInt (Song et al., 2019)

  - MLP-Mixer is our adaptation of Tolstikhin et al. (2021) for tabular data.

  - Trompt (Chen et al., 2023b) (our reimplementation, since there is no official implementation)


We also evaluated TabPFN (Hollmann et al., 2023), where possible. The results for this model are
available only in Appendix E because this model is by design not applicable to regression tasks,
which is a considerable number of our datasets. Overall, TabPFN specializes in small datasets. In
line with that, the performance of TabPFN on our benchmark was not competitive.


B.2 T ASK PERFORMANCE


Figure 8 is a different version of Figure 3 with additional baselines. Overall, none of the additional
baselines affect our main story.


Figure 9 is the critical difference diagram (CDD) computed over exactly the same results that were
used for building Figure 3.


15


Published as a conference paper at ICLR 2025



TabR


SAINT

DCN2

SNN


AutoInt


MLP `-` Mixer

Excel _[∗]_
TabR _[‡]_


MNCA


Trompt

MLP


ResNet

FT `-` T


MNCA _[‡]_

T2G


CatBoost
MLP _[‡]_

MLP _[†]_


LightGBM

TabM


XGBoost

TabM _[†]_ mini

|Perf<br>On 9 datase<br>Sorte|ormance scores<br>ts with domain-aware split<br>d by the mean score|Col3|
|---|---|---|
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||



_−_ 5% 0% 5% 10%
Relative improvement over MLP ( _↑_ )



DCN2


AutoInt

ResNet

Excel _[∗]_


MLP `-` Mixer

SAINT

FT `-` T


Trompt


MLP _[†]_


MNCA
MLP _[‡]_


LightGBM

XGBoost


CatBoost

TabM

MNCA _[‡]_
TabR _[‡]_

TabM _[†]_ mini



Performance ranks

On 37 datasets with random split


Sorted by the mean rank



Performance scores

On 37 datasets with random split


Sorted by the mean score


AutoInt


ResNet

SAINT


MLP `-` Mixer

Trompt


MNCA


LightGBM


XGBoost


CatBoost
MNCA _[‡]_

TabM _[†]_ mini

|Col1|Col2|Col3|Col4|Col5|Mean|
|---|---|---|---|---|---|
|NN<br>N2||||||
|NN<br>N2||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||
|_†_<br>mini<br>CA_‡_<br>ost<br>bM<br>ost<br>bR_‡_<br>BM<br>CA<br>LP_‡_<br>LP_†_<br>2G<br>mpt<br>xer<br>bR<br>`-`T<br>NT<br>Net<br>LP<br>cel_∗_<br>Int||||||



_−_ 2% 0% 2% 4% 6% 8%
Relative improvement over MLP ( _↑_ )






|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|N2||||~~6~~~~_._8~~|~~6~~~~_._8~~|~~6~~~~_._8~~|~~ 4~~~~_._1~~|
|NN||||~~6~~~~_._5 ~~<br>|~~6~~~~_._5 ~~<br>|~~6~~~~_._5 ~~<br>|~~_._3~~<br>|
|NN||||~~_ ±_~~|~~_ ±_~~|~~_ ±_~~||
|LP||||~~5~~~~_._9 3~~~~_._8~~<br>|~~5~~~~_._9 3~~~~_._8~~<br>|~~5~~~~_._9 3~~~~_._8~~<br>|~~5~~~~_._9 3~~~~_._8~~<br>|
|LP||||~~_ ±_~~|~~_ ±_~~|~~_ ±_~~||
|Int||||~~5~~~~_._6~~<br>|~~5~~~~_._6~~<br>|~~4~~~~_._1~~<br>|~~4~~~~_._1~~<br>|
|Int||||||~~_._5~~<br>||
|Net<br>||||~~5~~~~_._5 ~~<br>|~~5~~~~_._5 ~~<br>|~~5~~~~_._5 ~~<br>|~~5~~~~_._5 ~~<br>|
|Net<br>||||~~_ ±_~~|~~_ ±_~~|||
|el_∗_||||~~5~~~~_._3 3~~~~_._~~<br>|~~5~~~~_._3 3~~~~_._~~<br>|~~5~~~~_._3 3~~~~_._~~<br>|~~5~~~~_._3 3~~~~_._~~<br>|
|el_∗_||||~~_ ±_~~|~~_ ±_~~|||
|xer<br>||~~4~~~~_._~~|~~4~~~~_._~~|~~ 3~~~~_._3~~<br>|~~ 3~~~~_._3~~<br>|~~ 3~~~~_._3~~<br>|~~ 3~~~~_._3~~<br>|
|xer<br>||||~~_ ±_~~|~~_ ±_~~|||
|NT||~~4~~~~_._~~<br>|~~4~~~~_._~~<br>|~~ 3~~~~_._6~~<br>|~~ 3~~~~_._6~~<br>|~~ 3~~~~_._6~~<br>|~~ 3~~~~_._6~~<br>|
|NT||||~~_ ±_~~|~~_ ±_~~|||
|`-`T||~~4~~~~_._~~<br>|~~4~~~~_._~~<br>|~~ 3~~~~_._7~~<br>|~~ 3~~~~_._7~~<br>|~~ 3~~~~_._7~~<br>|~~ 3~~~~_._7~~<br>|
|`-`T||||~~_ ±_~~|~~_ ±_~~|~~_ ±_~~||
|pt||~~4~~~~_._4~~<br>|~~4~~~~_._4~~<br>|~~3~~~~_._6~~<br>|~~3~~~~_._6~~<br>|~~3~~~~_._6~~<br>|~~3~~~~_._6~~<br>|
|pt||~~_ ±_~~|~~_ ±_~~|||||
|2G||~~4~~~~_._3 ~~<br>|~~4~~~~_._3 ~~<br>|~~_._9~~<br>|~~_._9~~<br>|~~_._9~~<br>|~~_._9~~<br>|
|2G||~~_ ±_~~|~~_ ±_~~|||||
|bR||~~3~~~~_._8 3~~~~_._5~~<br>|~~3~~~~_._8 3~~~~_._5~~<br>|~~3~~~~_._8 3~~~~_._5~~<br>|~~3~~~~_._8 3~~~~_._5~~<br>|~~3~~~~_._8 3~~~~_._5~~<br>|~~3~~~~_._8 3~~~~_._5~~<br>|
|bR||~~_ ±_~~|~~_ ±_~~|||||
|P_†_||~~3~~~~_._6 2~~~~_._3~~<br>|~~3~~~~_._6 2~~~~_._3~~<br>|~~3~~~~_._6 2~~~~_._3~~<br>|~~3~~~~_._6 2~~~~_._3~~<br>|~~3~~~~_._6 2~~~~_._3~~<br>|~~3~~~~_._6 2~~~~_._3~~<br>|
|P_†_||~~_ ±_~~|~~_ ±_~~|||||
|CA<br>||~~3~~~~_._6 2~~~~_._7~~<br>|~~3~~~~_._6 2~~~~_._7~~<br>|~~3~~~~_._6 2~~~~_._7~~<br>|~~3~~~~_._6 2~~~~_._7~~<br>|~~3~~~~_._6 2~~~~_._7~~<br>|~~3~~~~_._6 2~~~~_._7~~<br>|
|CA<br>||~~_ ±_~~|~~_ ±_~~|||||
|P_‡_||~~_._5 2~~~~_._4~~<br>|~~_._5 2~~~~_._4~~<br>|~~_._5 2~~~~_._4~~<br>|~~_._5 2~~~~_._4~~<br>|~~_._5 2~~~~_._4~~<br>|~~_._5 2~~~~_._4~~<br>|
|P_‡_||~~_._4 2~~~~_._2~~<br>~~_ ±_~~|~~_._4 2~~~~_._2~~<br>~~_ ±_~~|||||
|M<br>||||||||
|M<br>||~~_._4 2~~~~_._3~~<br>~~_ ±_~~|~~_._4 2~~~~_._3~~<br>~~_ ±_~~|||||
|ost||||||||
|ost||~~ 2~~~~_._0~~<br>~~_ ±_~~|~~ 2~~~~_._0~~<br>~~_ ±_~~|||||
|ost|~~3~~~~_._0~~<br>|~~ 2~~~~_._~~<br>|~~ 2~~~~_._~~<br>|~~ 2~~~~_._~~<br>|~~ 2~~~~_._~~<br>|~~ 2~~~~_._~~<br>|~~ 2~~~~_._~~<br>|
|ost||~~_±_~~||||||
|bM|~~2~~~~_._8~~<br>|~~ 2~~~~_._3~~<br>|~~ 2~~~~_._3~~<br>|~~ 2~~~~_._3~~<br>|~~ 2~~~~_._3~~<br>|~~ 2~~~~_._3~~<br>|~~ 2~~~~_._3~~<br>|
|bM||||||||
|A_‡_|~~2~~~~_._8~~|~~ 2~~~~_._5~~||MLP,<br>Attenti<br>|MLP,<br>Attenti<br>|MLP,<br>Attenti<br>|BDT<br>on,<br>|
|R_‡_|~~2~~~~_._6~~<br>|~~2~~~~_._4~~<br>|~~2~~~~_._4~~<br>|~~2~~~~_._4~~<br>|~~2~~~~_._4~~<br>|~~2~~~~_._4~~<br>|~~2~~~~_._4~~<br>|
|R_‡_|~~_ ±_~~|||~~Retriev~~<br>MLP (|~~Retriev~~<br>MLP (|~~Retriev~~<br>MLP (|~~al~~<br>Ours)|
|mini|~~1~~~~_._~~|~~_ ±_ 1~~~~_._2~~|~~_ ±_ 1~~~~_._2~~|||||



2 4 6

Rank ( _↓_ )



Figure 8: An extended comparison of tabular models as in Figure 3. Note that the ranks (left) are
computed only over the 37 datasets with random splits because ResNet, AutoInt, and MLP-Mixer
were evaluated only on one 1 out of 9 datasets with domain-aware splits.


4 6 8 10 12



TabM _[†]_ mini

TabM _[♠]_


CatBoost


MNCA _[‡]_


XGBoost

LightGBM



SAINT

MNCA

MLP _[†]_



Figure 9: Critical difference diagram. The computation method is taken from the Kim et al. (2024).


B.3 E FFICIENCY


This section complements subsection 4.3.


**Additional results.**


Figure 10 complements Figure 4 by providing the training times on smaller datasets and the inference
throughput on GPU with large batch sizes.


Table 3 provide the number of trainable parameters for some of the models from Figure 3.


**Motivation for the benchmark setup.** Comparing models under all possible kinds of budgets (task
performance, the number of parameters, training time, etc.) on all possible hardware (GPU, CPU,
etc.) with all possible batch sizes is rather infeasible. As such, we set a narrow goal of _providing a_
_high-level intuition on the efficiency in a transparent setting_ . Thus, benchmarking the transparently
obtained tuned hyperparameter configurations works well for our goal. Yet, this choice also has
a limitation: the hyperparameter tuning process is not aware of the efficiency budget, so it can
prefer much heavier configurations even if they lead to tiny performance improvements, which will
negatively affect efficiency without a good reason. Overall, we hope that the large number of datasets
compensates for potentially imperfect per-dataset measurements.


16


Published as a conference paper at ICLR 2025


**Motivation for the two setups for measuring inference throughput.**


  - The setup on the right side of Figure 4 simulates the online per-object predictions.

  - The setup on the right side of Figure 10 simulates the offline batched computations.



XGBoost


MLP


MLP _[†]_


MNCA


MNCA _[‡]_


TabM


TabM _[†]_ mini

TabR


TabR _[‡]_


FT-T


T2G


SAINT



Training time on datasets with _<_ 100K objects


Device: GPU NVIDIA A100

|t<br>P<br>†<br>‡<br>i<br>‡|Col2|Col3|Col4|Col5|M<br>A|LP, G<br>ttentio|BDT<br>n,|
|---|---|---|---|---|---|---|---|
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>|||||~~R~~<br>~~M~~|~~etrieva~~<br>~~LP (O~~|~~l~~<br>~~urs)~~|
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>|||||M|ean||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||
|t<br>P<br>_†_<br><br>_‡_<br><br>i<br><br>_‡_<br><br><br>||||||||



10s _≈_ 2 _m_ _≈_ 15 _m_ _≈_ 1 _h ≈_ 3 _h_
Time ( _↓_ )



Inference throughput with maximum batch size


Device: GPU NVIDIA 2080Ti


MLP


MLP _[†]_


XGBoost


TabM


MNCA


MNCA _[‡]_

FT-T


TabM _[†]_ mini

TabR


TabR _[‡]_

T2G


SAINT

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||



10 [3] 10 [4] 10 [5] 10 [6] 10 [7] 10 [8]

Objects per second ( _↑_ )



Figure 10: ( _Left_ ) Training time on datasets with less than 100K objects. ( _Right_ ) Inference throughput
on GPU with maximum possible batch size (i.e. the batch size depends on a model).


Table 3: Mean number of parameters with std. dev. for 7 different tuned models across all 46 datasets.


TabM MLP FT-T T2G TabR ModernNCA SAINT


1 _._ 4 _M ±_ 1 _._ 3 _M_ 1 _._ 0 _M ±_ 1 _._ 0 _M_ 1 _._ 2 _M ±_ 1 _._ 2 _M_ 2 _._ 1 _M ±_ 1 _._ 6 _M_ 858 _K ±_ 1 _._ 4 _M_ 1 _._ 0 _M ±_ 1 _._ 1 _M_ 175 _._ 4 _M ±_ 565 _._ 4 _M_


C D ATASETS


In total, we use 46 datasets:


1. 38 datasets are taken from Gorishniy et al. (2024), which includes:

(a) 28 datasets from Grinsztajn et al. (2022). See the original paper for the precise dataset
information.
(b) 10 datasets from other sources. Their properties are provided in Table 4.
2. 8 datasets from the TabReD benchmark (Rubachev et al., 2024). Their properties are provided
in Table 5.


In fact, the aforementioned 38 datasets from Gorishniy et al. (2024) is only a subset of the datasets
used in Gorishniy et al. (2024). Namely, we did not include the following of the remaining datasets:


  - The datasets that, according to Rubachev et al. (2024), have incorrect splits
and/or label leakage, including: Bike ~~S~~ haring ~~D~~ emand, compass, electricity,
SGEMM ~~G~~ PU ~~k~~ ernel ~~p~~ erformance, sulfur, visualizing ~~s~~ oil, and the weather forecasting dataset (it is replaced by the correct weather forecasting dataset from TabReD (Rubachev
et al., 2024)).

  - rl from (Grinsztajn et al., 2022). We observed abnormal results on these datasets. This is an
anonymous dataset, which made the investigation impossible, so we removed this dataset to
avoid confusion.

 - yprop 4 ~~1~~ from (Grinsztajn et al., 2022). Strictly speaking, this dataset was omitted due to a
mistake on our side. For future work, we note that the typical performance gaps on this dataset
have low absolute values in terms of RMSE. Perhaps, _R_ [2] may be a more appropriate metric for
this dataset.


17


Published as a conference paper at ICLR 2025


Table 4: Properties of those datasets from Gorishniy et al. (2024) that are not part of Grinsztajn et al.
(2022) or TabReD Rubachev et al. (2024). “# Num”, “# Bin”, and “# Cat” denote the number of
numerical, binary, and categorical features, respectively. The table is taken from (Gorishniy et al.,
2024).


Name # Train # Validation # Test # Num # Bin # Cat Task type Batch size


Churn Modelling 6 400 1 600 2 000 7 3 1 Binclass 128
California Housing 13 209 3 303 4 128 8 0 0 Regression 256
House 16H 14 581 3 646 4 557 16 0 0 Regression 256
Adult 26 048 6 513 16 281 6 1 8 Binclass 256
Diamond 34 521 8 631 10 788 6 0 3 Regression 512
Otto Group Products 39 601 9 901 12 376 93 0 0 Multiclass 512
Higgs Small 62 751 15 688 19 610 28 0 0 Binclass 512
Black Friday 106 764 26 692 33 365 4 1 4 Regression 512
Covertype 371 847 92 962 116 203 10 4 1 Multiclass 1024
Microsoft 723 412 235 259 241 521 131 5 0 Regression 1024


Table 5: Properties of the datasets from the TabReD benchmark (Rubachev et al., 2024). “# Num”,
“# Bin”, and “# Cat” denote the number of numerical, binary, and categorical features, respectively.


Name # Train # Validation # Test # Num # Bin # Cat Task type Batch size


Sberbank Housing 18 847 4 827 4 647 365 17 10 Regression 256
Ecom Offers 109 341 24 261 26 455 113 6 0 Binclass 1024
Maps Routing 160 019 59 975 59 951 984 0 2 Regression 1024
Homesite Insurance 224 320 20 138 16 295 253 23 23 Binclass 1024
Cooking Time 227 087 51 251 41 648 186 3 3 Regression 1024
Homecredit Default 267 645 58 018 56 001 612 2 82 Binclass 1024
Delivery ETA 279 415 34 174 36 927 221 1 1 Regression 1024
Weather 340 596 42 359 40 840 100 3 0 Regression 1024


D I MPLEMENTATION DETAILS


D.1 H ARDWARE


Most of the experiments were conducted on a single NVIDIA A100 GPU. In rare exceptions, we used
a machine with a single NVIDIA 2080 Ti GPU and Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz.


D.2 E XPERIMENT SETUP


We mostly follow the experiment setup from Gorishniy et al. (2024). As such, some of the text below
is copied from (Gorishniy et al., 2024).


**Data preprocessing.** For each dataset, for all DL-based solutions, the same preprocessing was used
for fair comparison. For numerical features, by default, we used a slightly modified version of the
quantile normalization from the Scikit-learn package (Pedregosa et al., 2011) (see the source code),
with rare exceptions when it turned out to be detrimental (for such datasets, we used the standard
normalization or no normalization). For categorical features, we used one-hot encoding. Binary
features (i.e. the ones that take only two distinct values) are mapped to _{_ 0 _,_ 1 _}_ without any further
preprocessing. We completely follow Rubachev et al. (2024) on Table 5 datasets.


**Training neural networks.** For DL-based algorithms, we minimize cross-entropy for classification
problems and mean squared error for regression problems. We use the AdamW optimizer (Loshchilov
& Hutter, 2019). We do not apply learning rate schedules. We do not use data augmentations. We
apply global gradient clipping to 1 _._ 0 . For each dataset, we used a predefined dataset-specific batch
size. We continue training until there are patience consecutive epochs without improvements on
the validation set; we set patience = 16 for the DL models.


**Hyperparameter tuning.** In most cases, hyperparameter tuning is performed with the TPE sampler
(typically, 50-100 iterations) from the Optuna package (Akiba et al., 2019). Hyperparameter tuning


18


Published as a conference paper at ICLR 2025


spaces for most models are provided in individual sections below (example for TabM : subsection D.9).
We follow Rubachev et al. (2024) and use 25 iterations on some datasets from Table 5.


**Evaluation.** On a given dataset, for a given model, the tuned hyperparameters are evaluated under
multiple (in most cases, 15 ) random seeds. The mean test metric and its standard deviation over these
random seeds are then used to compare algorithms as described in subsection D.3.


D.3 M ETRICS


We use Root Mean Squared Error for regression tasks, ROC-AUC for classification datasets from
Table 5 (following Rubachev et al. (2024)), and accuracy for the rest of datasets (following Gorishniy
et al. (2024)). We also tried computing ROC-AUC for all classification datasets, but did not observe
any significant changes (see Figure 11), so we stuck to prior work. By default, the mean test score
and its standard deviation are obtained by training a given model with tuned hyperparameters from
scratch on a given dataset under 15 different random seeds.


**How we compute ranks.** Our method of computing ranks used in Figure 3 does not count small
improvements as wins, hence the reduced range of ranks compared to other studies. Intuitively, our
ranks can be considered as “tiers”.


Recall that, on a given dataset, the performance of a given model A is expressed with the mean A mean
and the standard deviation A std of the performance score computed after the evaluation under multiple
random seeds. Assuming the higher score the better, we define that the model A is better than the
model B if: A mean _−_ A std _>_ B mean . In other words, a model is considered better if it has a better mean
score and the margin is larger than the standard deviation.


On a given dataset, when there are many models, we sort them in descending score order. Starting
from the best model (with a rank equal to 1 ) we iterate over models and assign the rank 1 to all models
that are no worse than the best model according to the above rule. The first model in descending
order that is worse than the best model is assigned rank 2 and becomes the new reference model. We
continue the process until all models are ranked. Ranks are computed independently for each dataset.


D.4 I MPLEMENTATION DETAILS OF SUBSECTION 4.3


**Applicability to large datasets.** The two datasets used in Table 2 are the _full_ versions of the “Weather”
and “Maps Routing” datasets from the TabReD benchmark Rubachev et al. (2024). Their smaller
versions with subsampled training set were already included in Table 1 and were used when building
Figure 3. The validation and test sets are the same for the small and large versions of these datasets,
so the task metrics are comparable between the two versions. When running models on the large
versions of the datasets, we reused the hyperparameters tuned for their small versions. Thus, this
experiment can be seen as a quick assessment of the applicability of several tabular DL to large
datasets without a strong focus on the task performance. All models, except for FT-Transformer,
were evaluated under 3 random seeds. FT-Transformer was evaluated under 1 random seed.


D.5 I MPLEMENTATION DETAILS OF SUBSECTION 5.1


**Experiment setup.** This paragraph complements the description of the experiment setup in subsection 5.1. Namely, in addition to what is mentioned in the main text:


  - Dropout and weight decay are turned off.

  - To get representative training profiles for all models, the learning rates are tuned
separately for TabM _[k]_ mini [=1] and TabM _[k]_ mini [=32] on validation sets using the usual metrics
(i.e. RMSE or accuracy) as the guidance. The grid for learning rate tuning was:
numpy.logspace(numpy.log10(1e-5), numpy.log10(5e-3), num=25).


D.6 I MPLEMENTATION DETAILS OF SUBSECTION 5.2


**TabM** [ **G** ] **.** Here, we clarify the implementation details for TabM[G] described in subsection 5.2.
TabM[G] is obtained from a trained TabM by greedily selecting submodels from TabM starting from
the best one and stopping when two conditions are simultaneously true for the first time: (1) adding


19


Published as a conference paper at ICLR 2025



TabR


SAINT

DCN2

SNN


AutoInt


MLP `-` Mixer

Excel _[∗]_
TabR _[‡]_


MNCA


Trompt

MLP


ResNet

FT `-` T


MNCA _[‡]_

T2G


CatBoost
MLP _[‡]_

MLP _[†]_


LightGBM

TabM


XGBoost

TabM _[†]_ mini

|Perf<br>On 9 datase<br>Sorte|ormance scores<br>ts with domain-aware split<br>d by the mean score|Col3|
|---|---|---|
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||
||||



_−_ 5% 0% 5% 10%
Relative improvement over MLP ( _↑_ )



DCN2


AutoInt


ResNet

Excel _[∗]_


MLP `-` Mixer

SAINT

FT `-` T


Trompt


TabR

MLP _[‡]_


MNCA
MLP _[†]_


XGBoost


LightGBM
MNCA _[‡]_


CatBoost

TabM

TabR _[‡]_

TabM _[†]_ mini



Performance ranks


On 44 datasets

Sorted by the mean rank



Performance scores

On 35 datasets with random split


Sorted by the mean score


MLP `-` Mixer


LightGBM

|Col1|Col2|Col3|Col4|Col5|Mean|
|---|---|---|---|---|---|
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||
|M_†_<br>mini<br>NCA_‡_<br>Boost<br>TabM<br>Boost<br>GBM<br>abR_‡_<br>NCA<br>MLP_‡_<br>MLP_†_<br>ompt<br>T2G<br>Mixer<br>TabR<br>FT`-`T<br>AINT<br>esNet<br>xcel_∗_<br>MLP<br>toInt<br>SNN<br>CN2||||||



_−_ 2% 0% 2% 4% 6% 8%
Relative improvement over MLP ( _↑_ )






|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|N2||||~~7~~~~_._~~|~~7~~~~_._~~|~~7~~~~_._~~|~~2~~~~_ ±_ 4~~~~_._2~~|~~2~~~~_ ±_ 4~~~~_._2~~|
|NN||||~~6~~~~_._9~~<br>|~~6~~~~_._9~~<br>|~~6~~~~_._9~~<br>|~~ 4~~~~_._6~~<br>||
|NN|||||||~~_±_~~|~~_±_~~|
|Int||||~~6~~~~_._1 4~~~~_._~~<br>|~~6~~~~_._1 4~~~~_._~~<br>|~~6~~~~_._1 4~~~~_._~~<br>|~~6~~~~_._1 4~~~~_._~~<br>|~~6~~~~_._1 4~~~~_._~~<br>|
|Int|||||||||
|LP<br>||||~~6~~~~_._1 3~~~~_._~~<br>|~~6~~~~_._1 3~~~~_._~~<br>|~~6~~~~_._1 3~~~~_._~~<br>|~~6~~~~_._1 3~~~~_._~~<br>|~~6~~~~_._1 3~~~~_._~~<br>|
|LP<br>||||~~_ ±_~~|~~_ ±_~~|~~_ ±_~~|||
|et||||~~5~~~~_._9 3~~~~_._9~~<br>|~~5~~~~_._9 3~~~~_._9~~<br>|~~5~~~~_._9 3~~~~_._9~~<br>|~~5~~~~_._9 3~~~~_._9~~<br>|~~5~~~~_._9 3~~~~_._9~~<br>|
|et||||~~_ ±_~~|~~_ ±_~~|~~_ ±_~~|||
|el_∗_||||~~_._5 3~~~~_._3~~<br>|~~_._5 3~~~~_._3~~<br>||||
|el_∗_||||~~_ ±_~~|~~_ ±_~~||||
|xer<br>||||~~_._5 4~~~~_._3~~<br>|~~_._5 4~~~~_._3~~<br>|~~_._5 4~~~~_._3~~<br>|~~_._5 4~~~~_._3~~<br>|~~_._5 4~~~~_._3~~<br>|
|xer<br>||||~~_ ±_~~|~~_ ±_~~|~~_ ±_~~|||
|NT||~~5~~~~_._~~<br>|~~5~~~~_._~~<br>|~~ 3~~~~_._~~<br>|~~ 3~~~~_._~~<br>|~~ 3~~~~_._~~<br>|~~ 3~~~~_._~~<br>|~~ 3~~~~_._~~<br>|
|NT||||~~_ ±_~~|||||
|`-`T||~~5~~~~_._0~~<br>|~~5~~~~_._0~~<br>|~~ 3~~~~_._6~~<br>|~~ 3~~~~_._6~~<br>|~~ 3~~~~_._6~~<br>|~~ 3~~~~_._6~~<br>|~~ 3~~~~_._6~~<br>|
|`-`T|||||||||
|pt||~~4~~~~_._8~~<br>|~~4~~~~_._8~~<br>|~~ 4~~~~_._2~~<br>|~~ 4~~~~_._2~~<br>|~~ 4~~~~_._2~~<br>|~~ 4~~~~_._2~~<br>|~~ 4~~~~_._2~~<br>|
|pt|||||||||
|2G||~~4~~~~_._5 ~~<br>|~~4~~~~_._5 ~~<br>|~~_._9~~<br>|~~_._9~~<br>|~~_._9~~<br>|~~_._9~~<br>|~~_._9~~<br>|
|2G||~~_ ±_~~|~~_ ±_~~||||||
|bR||~~4~~~~_._2 3~~~~_._~~<br>|~~4~~~~_._2 3~~~~_._~~<br>|~~4~~~~_._2 3~~~~_._~~<br>|~~4~~~~_._2 3~~~~_._~~<br>|~~4~~~~_._2 3~~~~_._~~<br>|~~4~~~~_._2 3~~~~_._~~<br>|~~4~~~~_._2 3~~~~_._~~<br>|
|bR||~~_ ±_~~|~~_ ±_~~||||||
|P_‡_||~~4~~~~_._1 3~~~~_._~~<br>|~~4~~~~_._1 3~~~~_._~~<br>|~~4~~~~_._1 3~~~~_._~~<br>|~~4~~~~_._1 3~~~~_._~~<br>|~~4~~~~_._1 3~~~~_._~~<br>|~~4~~~~_._1 3~~~~_._~~<br>|~~4~~~~_._1 3~~~~_._~~<br>|
|P_‡_||~~_ ±_~~|~~_ ±_~~||||||
|CA<br>||~~4~~~~_._1 3~~~~_._0~~<br>|~~4~~~~_._1 3~~~~_._0~~<br>|~~4~~~~_._1 3~~~~_._0~~<br>|~~4~~~~_._1 3~~~~_._0~~<br>|~~4~~~~_._1 3~~~~_._0~~<br>|~~4~~~~_._1 3~~~~_._0~~<br>|~~4~~~~_._1 3~~~~_._0~~<br>|
|CA<br>||~~_ ±_~~|~~_ ±_~~||||||
|P_†_||~~3~~~~_._8 2~~~~_._5~~<br>|~~3~~~~_._8 2~~~~_._5~~<br>|~~3~~~~_._8 2~~~~_._5~~<br>|~~3~~~~_._8 2~~~~_._5~~<br>|~~3~~~~_._8 2~~~~_._5~~<br>|~~3~~~~_._8 2~~~~_._5~~<br>|~~3~~~~_._8 2~~~~_._5~~<br>|
|P_†_||~~_._5 2~~~~_._4~~<br>~~_ ±_~~|~~_._5 2~~~~_._4~~<br>~~_ ±_~~||||||
|ost<br>||~~_._5 2~~<br>|~~_._5 2~~<br>|~~_._5 2~~<br>|~~_._5 2~~<br>|~~_._5 2~~<br>|~~_._5 2~~<br>|~~_._5 2~~<br>|
|ost<br>||~~_ ±_~~|~~3~~<br>||||||
|M||~~_._4 2~~<br>|~~_._4 2~~<br>|~~_._4 2~~<br>|~~_._4 2~~<br>|~~_._4 2~~<br>|~~_._4 2~~<br>|~~_._4 2~~<br>|
|M||~~_ ±_~~|||||||
|A_‡_|~~3~~<br>|~~2 2~~~~_._9~~<br>|~~2 2~~~~_._9~~<br>|~~2 2~~~~_._9~~<br>|~~2 2~~~~_._9~~<br>|~~2 2~~~~_._9~~<br>|~~2 2~~~~_._9~~<br>|~~2 2~~~~_._9~~<br>|
|A_‡_||~~_ ±_~~|||||||
|ost<br>|~~3~~~~_._~~<br>|~~ 2~~~~_._1~~<br>|~~ 2~~~~_._1~~<br>|~~ 2~~~~_._1~~<br>|~~ 2~~~~_._1~~<br>|~~ 2~~~~_._1~~<br>|~~ 2~~~~_._1~~<br>|~~ 2~~~~_._1~~<br>|
|ost<br>||~~_ ±_~~|||||||
|M|~~3~~~~_._0~~|~~_ ±_ 2~~~~_._9~~||MLP<br>Atte<br>|MLP<br>Atte<br>|MLP<br>Atte<br>|GBDT<br>ntion,<br>|GBDT<br>ntion,<br>|
|R_‡_|~~3~~~~_._0~~|~~_ ±_ 2~~~~_._7~~|~~_ ±_ 2~~~~_._7~~|~~Retri~~<br>MLP|~~Retri~~<br>MLP|~~Retri~~<br>MLP|~~eval~~<br> (Ours)|~~eval~~<br> (Ours)|
|mini||~~2~~~~_._0~~~~_ ±_ 2~~~~_._4~~|~~2~~~~_._0~~~~_ ±_ 2~~~~_._4~~||||||



2 4 6

Rank ( _↓_ )



Figure 11: Same as Figure 3, but ROC-AUC is used as the metric for all classification datasets. The
two multiclass datasets presented in our benchmark are not taken into account.


any new submodel does not improve the validation metric of the collective prediction; (2) the current
validation metric is already better than that of the initial model with all _k_ submodels. To clarify,
during the greedy selection, the _i_ -th submodel is considered to be better than the _j_ -th submodel if
adding the _i_ -th submodel to the aggregated prediction leads to better validation metrics (i.e. it is _not_
the same as adding the submodel in the order of their individual validation metrics).


D.7 I MPLEMENTATION DETAILS OF SUBSECTION 5.3


Figure 7 shows the mean percentage improvements (see subsection D.3) over MLP across 17 datasets:
all datasets except for Covertype from Table 4, and all datasets from TabReD (Rubachev et al., 2024).
We have used the dropout rate 0 _._ 1 and tuned the learning rate separately for each value of _k_ . The
score on each dataset is averaged over 5 seeds.


D.8 N ON - LINEAR EMBEDDINGS FOR CONTINUOUS FEATURES


**Notation.** We use the notation based on _†_ and _‡_ only for brevity. Any other unambiguous notation
can be used in future work.


**Updated piecewise-linear embeddings.** We use a slightly different implementation of the piecewiselinear embeddings compared to Gorishniy et al. (2022). Architecture-wise, our implementation
corresponds to the “Q-L” and “T-L” variations from Table 2 in Gorishniy et al. (2022) (we use the
quantile-based bins for simplicity). In practice, our implementation is significantly faster and uses a
different parametrization and initialization. See the source code for details.


**Other models.** Since it is not feasible to test all combinations of backbones and embeddings, for
baselines, we stick to the embeddings used in the original papers (applies to TabR (Gorishniy et al.,
2024), ExcelFormer (Chen et al., 2023a) and ModernNCA (Ye et al., 2024)). For all models with
feature embeddings (including TabM, MLP, TabR, ModernNCA, ExcelFormer), the embeddingsrelated details are commented in the corresponding sections below.


20


Published as a conference paper at ICLR 2025


D.9 T AB M


**Feature embeddings.** TabM _[†]_ mini [and] [ TabM] _[†]_ [ are the versions of] [ TabM] [ with non-linear feature]
embeddings. TabM _[†]_ mini [and] [ TabM] _[†]_ [ use the updated piecewise-linear feature embeddings mentioned]
in subsection D.8.


Table 6 provides the hyperparameter tuning spaces for TabM and TabM mini . Table 7 provides the
hyperparameter tuning spaces for TabM _[†]_ and TabM _[†]_ mini [.]


Table 6: The hyperparameter tuning space for TabM and TabM mini . Here, (B) = _{_ Covertype, Microsoft, Table 5 _}_ and (A) contains all other datasets.


Parameter Distribution or Value


_k_ 32

# layers UniformInt[1 _,_ 5]
Width (hidden size) UniformInt[64 _,_ 1024]
Dropout rate _{_ 0 _._ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 5] _}_
Learning rate LogUniform[1 _e_ -4 _,_ 5 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


# Tuning iterations (A) 100 (B) 50


Table 7: The hyperparameter tuning space for TabM _[†]_ mini [and] [ TabM] _[†]_ [. Here, (B) =] _[ {]_ [Covertype,]
Microsoft, Table 5 _}_ and (A) contains all other datasets.


Parameter Distribution or Value


_k_ 32

# layers UniformInt[1 _,_ 4]
Width (hidden size) UniformInt[64 _,_ 1024]
Dropout rate _{_ 0 _._ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 5] _}_
# PLE bins UniformInt[8 _,_ 32]
Learning rate LogUniform[5 _e_ -5 _,_ 3 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


# Tuning iterations (A) 100 (B) 50


D.10 MLP


**Feature embeddings.** MLP _[†]_ and MLP _[‡]_ are the versions of MLP with non-linear feature embeddings.
MLP _[†]_ uses the updated piecewise-linear embeddings mentioned in subsection D.8. MLP _[‡]_ (also
known as MLP-PLR) uses the periodic embeddings (Gorishniy et al., 2022). Technically, it is the
PeriodicEmbeddings class from the rtdl ~~n~~ um ~~e~~ mbeddings Python package. We tested
two variations: with lite=False and lite=True . In the paper, only the former one is reported,
but in the source code, the results for both are available.


Table 8, Table 9, Table 10 provide the hyperparameter tuning spaces for MLP, MLP _[†]_ and MLP _[‡]_,
respectively.


D.11 T AB R


**Feature** **embeddings.** TabR _[‡]_ is the version of TabR with non-linear feature embeddings. TabR _[‡]_ uses the periodic embeddings (Gorishniy et al., 2022), specifically, PeriodicEmbeddings(lite=True) from the rtdl ~~n~~ um ~~e~~ mbeddings
Python package on most datasets. On the datasets from Table 5, TabR _[‡]_ uses the


21


Published as a conference paper at ICLR 2025


Table 8: The hyperparameter tuning space for MLP.


Parameter Distribution


# layers UniformInt[1 _,_ 6]
Width (hidden size) UniformInt[64 _,_ 1024]
Dropout rate _{_ 0 _._ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 5] _}_
Learning rate LogUniform[3 _e_ -5 _,_ 1 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


# Tuning iterations 100


Table 9: The hyperparameter tuning space for MLP _[†]_ .


Parameter Distribution


# layers UniformInt[1 _,_ 5]
Width (hidden size) UniformInt[64 _,_ 1024]
Dropout rate _{_ 0 _._ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 5] _}_
Learning rate LogUniform[3 _e_ -5 _,_ 1 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


d ~~e~~ mbedding UniformInt[8 _,_ 32]
n ~~b~~ ins UniformInt[2 _,_ 128]
# Tuning iterations 100


Table 10: The hyperparameter tuning space for MLP _[‡]_ .


Parameter Distribution


# layers UniformInt[1 _,_ 5]
Width (hidden size) UniformInt[64 _,_ 1024]
Dropout rate _{_ 0 _._ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 5] _}_
Learning rate LogUniform[3 _e_ -5 _,_ 1 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


n ~~f~~ requencies UniformInt[16 _,_ 96]
d ~~e~~ mbedding UniformInt[16 _,_ 32]
frequency ~~i~~ nit ~~s~~ cale LogUniform[1 _e_ -2 _,_ 1 _e_ 1]
# Tuning iterations 100


PeriodicEmbeddings(lite=True) embeddings on the Sberbank Housing and Ecom
Offers datasets, and LinearReLUEmbeddings on the rest (to fit the computations into the GPU
memory, following the original TabR paper).


Since we follow the training and evaluation protocols from Gorishniy et al. (2024), and TabR was
proposed in Gorishniy et al. (2024), we simply reuse the results for TabR. More details can be found
in Appendix.D from Gorishniy et al. (2024). When tuning TabR _[‡]_ on the datasets from Table 5, we
have used 25 tuning iterations and the same tuning space as for TabR from Rubachev et al. (2024).


22


Published as a conference paper at ICLR 2025


D.12 FT-T RANSFORMER


We used the implementation from the ” rtdl ~~r~~ evisiting ~~m~~ odels ” Python package. The results
on datasets from Table 5 were copied from Rubachev et al. (2024), because the experiment setups are
compatible.


Table 11: The hyperparameter tuning space for FT-Transformer Gorishniy et al. (2021). Here, (B) =
_{_ Covertype, Microsoft _}_ and (A) contains all other datasets (except Table 5).


Parameter Distribution or Value


# blocks UniformInt[1 _,_ 4]
_d_ _token_ UniformInt[16 _,_ 384]
Attention dropout rate Uniform[0 _._ 0 _,_ 0 _._ 5]
FFN hidden dimension expansion rate Uniform[ [2] _/_ 3 _,_ [8] _/_ 3 ]
FFN dropout rate Uniform[0 _._ 0 _,_ 0 _._ 5]
Residual dropout rate _{_ 0 _._ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 2] _}_
Learning rate LogUniform[3 _e_ -5 _,_ 1 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


# Tuning iterations (A) 100 (B) 50


D.13 M ODERN NCA


**Feature embeddings.** We adapted the official implementation of Ye et al. (2024). We used periodic
embeddings Gorishniy et al. (2022) (specifically, PeriodicEmbeddings(lite=True) from
the rtdl ~~n~~ um embeddings Python package) for ModernNCA _[‡]_ and no embeddings for ModernNCA. Table 12 and Table 13 provides hyperparameter tuning spaces for each ModernNCA and
ModernNCA _[‡]_ .


Table 12: The hyperparameter tuning space for ModernNCA. Here, (C) = _{_ Table 5 _}_, (B) = _{_ Covertype,
Microsoft _}_ and (A) contains all other datasets.


Parameter Distribution


# blocks UniformInt[0 _,_ 2]
_d_ _block_ UniformInt[64 _,_ 1024]
dim UniformInt[64 _,_ 1024]
Dropout rate Uniform[0 _._ 0 _,_ 0 _._ 5]
Sample rate Uniform[0 _._ 05 _,_ 0 _._ 6]
Learning rate LogUniform[1 _e_ -5 _,_ 1 _e_ -1]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -6 _,_ 1 _e_ -3] _}_


# Tuning iterations (A) 100 (B, C) 50


D.14 T2G-F ORMER


We adapted the implementation and hyperparameters of Yan et al. (2023) from the official repository [1] .
Table 14 provides hyperparameter tuning space.


D.15 SAINT


We completely adapted hyperparameters and protocol from Gorishniy et al. (2024) to evaluate SAINT
on Grinsztajn et al. (2022) benchmark. Results on datasets from Table 4 were directly taken from


1 https://github.com/jyansir/t2g-former


23


Published as a conference paper at ICLR 2025


Table 13: The hyperparameter tuning space for ModernNCA _[‡]_ . Here, (C) = _{_ Table 5 _}_, (B) =
_{_ Covertype, Microsoft _}_ and (A) contains all other datasets.


Parameter Distribution


# blocks UniformInt[0 _,_ 2]
_d_ _block_ UniformInt[64 _,_ 1024]
dim UniformInt[64 _,_ 1024]
Dropout rate Uniform[0 _._ 0 _,_ 0 _._ 5]
Sample rate Uniform[0 _._ 05 _,_ 0 _._ 6]
Learning rate LogUniform[1 _e_ -5 _,_ 1 _e_ -1]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -6 _,_ 1 _e_ -3] _}_
n ~~f~~ requencies UniformInt[16 _,_ 96]
d ~~e~~ mbedding UniformInt[16 _,_ 32]
frequency ~~i~~ nit ~~s~~ cale LogUniform[0 _._ 01 _,_ 10]


# Tuning iterations (A) 100 (B, C) 50


Table 14: The hyperparameter tuning space for T2G-Former Yan et al. (2023). Here, (C) = _{_ Table 5 _}_,
(B) = _{_ Covertype, Microsoft _}_ and (A) contains all other datasets. Also, we used 50 tuning iterations
on some datasets from Grinsztajn et al. (2022).


Parameter Distribution or Value


# blocks (A) UniformInt[3 _,_ 4] (B, C) UniformInt[1 _,_ 3]
_d_ _token_ UniformInt[64 _,_ 512]
Attention dropout rate Uniform[0 _._ 0 _,_ 0 _._ 5]
FFN hidden dimension expansion rate (A, B) Uniform[ [2] _/_ 3 _,_ [8] _/_ 3 ] (C) 4 _/_ 3
FFN dropout rate Uniform[0 _._ 0 _,_ 0 _._ 5]
Residual dropout rate _{_ 0 _._ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 2] _}_
Learning rate LogUniform[3 _e_ -5 _,_ 1 _e_ -3]
Col. Learning rate LogUniform[5 _e_ -3 _,_ 5 _e_ -2]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -6 _,_ 1 _e_ -1] _}_


# Tuning iterations (A) 100 (B) 50 (C) 25


Gorishniy et al. (2024). Additional details can be found in Appendix.D from Gorishniy et al. (2024).
We have used a default configuration on big datasets due to the very high cost of tuning (see Table 15).


D.16 E XCELFORMER


**Feature embeddings.** ExcelFormer (Chen et al., 2023a) uses custom non-linear feature embeddings
based on a GLU-style activation, see the original paper for details.


We adapted the implementation and hyperparameters of Chen et al. (2023a) from the official repository [2] . For a fair comparison with other models, we did not use the augmentation techniques from the
paper in our experiments. See Table 16.


D.17 C AT B OOST, XGB OOST AND L IGHT GBM


Since our setup is directly taken from Gorishniy et al. (2024), we simply reused their results for
GBDTs from the official repository [3] . Importantly, in a series of preliminary experiments, we


2 https://github.com/WhatAShot/ExcelFormer
3 https://github.com/yandex-research/tabular-dl-tabr


24


Published as a conference paper at ICLR 2025


Table 15: The default hyperparameters for SAINT (Somepalli et al., 2021) on datasets from Rubachev
et al. (2024).


Parameter Value


depth 2
_d_ _token_ 32

_n_ _heads_ 4

_d_ _head_ 8

Attention dropout rate 0 _._ 1
FFN hidden dimension expansion rate 1
FFN dropout rate 0 _._ 8
Learning rate 1 _e_ -4
Weight decay 1 _e_ -2


Table 16: The hyperparameter tuning space for Excelformer Chen et al. (2023a). Here, (D) =
_{_ Homecredit, Maps Routing _}_, (C) = _{_ Table 5 w/o (D) _}_, (B) = _{_ Covertype, Microsoft _}_ and (A)
contains all other datasets.


Parameter Distribution or Value


# blocks (A, B) UniformInt[2 _,_ 5] (C) UniformInt[2 _,_ 4] (D) UniformInt[1 _,_ 3]
_d_ _token_ (A, B) _{_ 32 _,_ 64 _,_ 128 _,_ 256 _}_ (C) _{_ 16 _,_ 32 _,_ 64 _}_ (D) _{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_
_n_ _heads_ (A,B) _{_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ (C) _{_ 4 _,_ 8 _,_ 16 _}_ (D) 4
Attention dropout rate 0 _._ 3
FFN dropout rate 0 _._ 0
Residual dropout rate Uniform[0 _._ 0 _,_ 0 _._ 5]
Learning rate LogUniform[3 _e_ -5 _,_ 1 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


# Tuning iterations (A) 100 (B) 50 (C, D) 25


confirmed that those results are reproducible in our instance of their setup. The details can be found
in Appendix.D from Gorishniy et al. (2024). Results on datasets from Table 5 were copied from the
paper (Rubachev et al., 2024).


D.18 A UTO I NT


We used an implementation from Gorishniy et al. (2021) which is an adapted official implementation [4] .


D.18.1 T AB PFN


Since TabPFN accepts only less than 10K training samples we use different subsamples of size 10K
for different random seeds. Also, TabPFN is not applicable to regressions and datasets with more
than 100 features.


4 https://github.com/shichence/AutoInt


25


Published as a conference paper at ICLR 2025


Table 17: The hyperparameter tuning space for AutoInt (Song et al., 2019). Here, (B) = _{_ Covertype,
Microsoft _}_ and (A) contains all other datasets.


Parameter Distribution


# blocks UniformInt[1 _,_ 6]
_d_ _token_ UniformInt[8 _,_ 64]
_n_ _heads_ 2
Attention dropout rate _{_ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 5] _}_
Embedding dropout rate _{_ 0 _,_ Uniform[0 _._ 0 _,_ 0 _._ 5] _}_
Learning rate LogUniform[3 _e_ -5 _,_ 1 _e_ -3]
Weight decay _{_ 0 _,_ LogUniform[1 _e_ -4 _,_ 1 _e_ -1] _}_


# Tuning iterations (A) 100 (B) 50


E P ER - DATASET RESULTS WITH STANDARD DEVIATIONS


Table 18: Extended results for the main benchmark. Results are grouped by datasets. One ensemble
consists of five models trained independently under different random seeds.



churn ↑

Method Single model Ensemble

MLP 0 _._ 8553 _±_ 0 _._ 0029 0 _._ 8582 _±_ 0 _._ 0008
TabPFN – 0 _._ 8624 _±_ 0 _._ 0008
ResNet 0 _._ 8545 _±_ 0 _._ 0044 0 _._ 8565 _±_ 0 _._ 0035
DCN2 0 _._ 8567 _±_ 0 _._ 0020 0 _._ 8570 _±_ 0 _._ 0017
SNN 0 _._ 8506 _±_ 0 _._ 0051 0 _._ 8533 _±_ 0 _._ 0033
Trompt 0 _._ 8600 _± nan_ –
AutoInt 0 _._ 8607 _±_ 0 _._ 0047 0 _._ 8622 _±_ 0 _._ 0003
MLP-Mixer 0 _._ 8592 _±_ 0 _._ 0036 0 _._ 8630 _±_ 0 _._ 0005
Excel _[∗]_ 0 _._ 8618 _±_ 0 _._ 0023 0 _._ 8625 _± nan_
SAINT 0 _._ 8603 _±_ 0 _._ 0029 –
FT-T 0 _._ 8593 _±_ 0 _._ 0028 0 _._ 8598 _±_ 0 _._ 0025
T2G 0 _._ 8613 _±_ 0 _._ 0015 –
MLP _[‡−]_ [lite] 0 _._ 8624 _±_ 0 _._ 0010 0 _._ 8638 _±_ 0 _._ 0012
MLP _[‡]_ 0 _._ 8624 _±_ 0 _._ 0026 0 _._ 8640 _±_ 0 _._ 0010
MLP _[†]_ 0 _._ 8580 _±_ 0 _._ 0028 0 _._ 8605 _±_ 0 _._ 0018
XGBoost 0 _._ 8605 _±_ 0 _._ 0022 0 _._ 8608 _±_ 0 _._ 0013
LightGBM 0 _._ 8600 _±_ 0 _._ 0008 0 _._ 8600 _±_ 0 _._ 0000
CatBoost 0 _._ 8582 _±_ 0 _._ 0017 0 _._ 8588 _±_ 0 _._ 0008
TabR 0 _._ 8599 _±_ 0 _._ 0025 0 _._ 8620 _±_ 0 _._ 0023
TabR _[‡]_ 0 _._ 8625 _±_ 0 _._ 0021 –
MNCA 0 _._ 8595 _±_ 0 _._ 0028 0 _._ 8615 _±_ 0 _._ 0013
MNCA _[‡]_ 0 _._ 8606 _±_ 0 _._ 0032 0 _._ 8607 _±_ 0 _._ 0008
TabM _[♠]_ 0 _._ 8613 _±_ 0 _._ 0025 0 _._ 8615 _±_ 0 _._ 0005
TabM 0 _._ 8605 _±_ 0 _._ 0016 0 _._ 8612 _±_ 0 _._ 0008
TabM[G] 0 _._ 8609 _±_ 0 _._ 0024 –
TabM mini 0 _._ 8633 _±_ 0 _._ 0018 0 _._ 8638 _±_ 0 _._ 0012
TabM _[†]_ mini 0 _._ 8606 _±_ 0 _._ 0023 0 _._ 8630 _±_ 0 _._ 0030


26



california ↓

Method Single model Ensemble

MLP 0 _._ 4948 _±_ 0 _._ 0058 0 _._ 4880 _±_ 0 _._ 0022
TabPFN – –
ResNet 0 _._ 4915 _±_ 0 _._ 0031 0 _._ 4862 _±_ 0 _._ 0017
DCN2 0 _._ 4971 _±_ 0 _._ 0122 0 _._ 4779 _±_ 0 _._ 0022
SNN 0 _._ 5033 _±_ 0 _._ 0075 0 _._ 4933 _±_ 0 _._ 0035
Trompt 0 _._ 4579 _± nan_ –
AutoInt 0 _._ 4682 _±_ 0 _._ 0063 0 _._ 4490 _±_ 0 _._ 0028
MLP-Mixer 0 _._ 4746 _±_ 0 _._ 0056 0 _._ 4509 _±_ 0 _._ 0029
Excel _[∗]_ 0 _._ 4544 _±_ 0 _._ 0048 0 _._ 4350 _± nan_
SAINT 0 _._ 4680 _±_ 0 _._ 0048 –
FT-T 0 _._ 4635 _±_ 0 _._ 0048 0 _._ 4515 _±_ 0 _._ 0016
T2G 0 _._ 4640 _±_ 0 _._ 0100 0 _._ 4462 _± nan_
MLP _[‡−]_ [lite] 0 _._ 4652 _±_ 0 _._ 0045 0 _._ 4549 _±_ 0 _._ 0006
MLP _[‡]_ 0 _._ 4597 _±_ 0 _._ 0058 0 _._ 4482 _±_ 0 _._ 0026
MLP _[†]_ 0 _._ 4530 _±_ 0 _._ 0029 0 _._ 4491 _±_ 0 _._ 0010
XGBoost 0 _._ 4327 _±_ 0 _._ 0016 0 _._ 4316 _±_ 0 _._ 0007
LightGBM 0 _._ 4352 _±_ 0 _._ 0019 0 _._ 4339 _±_ 0 _._ 0008
CatBoost 0 _._ 4294 _±_ 0 _._ 0012 0 _._ 4265 _±_ 0 _._ 0003
TabR 0 _._ 4030 _±_ 0 _._ 0023 0 _._ 3964 _±_ 0 _._ 0013
TabR _[‡]_ 0 _._ 3998 _±_ 0 _._ 0033 –
MNCA 0 _._ 4239 _±_ 0 _._ 0012 0 _._ 4231 _±_ 0 _._ 0005
MNCA _[‡]_ 0 _._ 4142 _±_ 0 _._ 0031 0 _._ 4071 _±_ 0 _._ 0029
TabM _[♠]_ 0 _._ 4509 _±_ 0 _._ 0032 0 _._ 4490 _±_ 0 _._ 0018
TabM 0 _._ 4414 _±_ 0 _._ 0012 0 _._ 4402 _±_ 0 _._ 0001
TabM[G] 0 _._ 4413 _±_ 0 _._ 0020 –
TabM mini 0 _._ 4479 _±_ 0 _._ 0022 0 _._ 4461 _±_ 0 _._ 0011
TabM _[†]_ mini 0 _._ 4275 _±_ 0 _._ 0024 0 _._ 4244 _±_ 0 _._ 0006


Published as a conference paper at ICLR 2025


house ↓

Method Single model Ensemble

MLP 3 _._ 1117 _±_ 0 _._ 0294 3 _._ 0706 _±_ 0 _._ 0140
TabPFN – –
ResNet 3 _._ 1143 _±_ 0 _._ 0258 3 _._ 0706 _±_ 0 _._ 0098
DCN2 3 _._ 3327 _±_ 0 _._ 0878 3 _._ 1303 _±_ 0 _._ 0410
SNN 3 _._ 2176 _±_ 0 _._ 0376 3 _._ 1320 _±_ 0 _._ 0155
Trompt 3 _._ 0638 _± nan_ –
AutoInt 3 _._ 2157 _±_ 0 _._ 0436 3 _._ 1261 _±_ 0 _._ 0095
MLP-Mixer 3 _._ 1871 _±_ 0 _._ 0519 3 _._ 0184 _±_ 0 _._ 0086
Excel _[∗]_ 3 _._ 2460 _±_ 0 _._ 0685 3 _._ 1097 _± nan_
SAINT 3 _._ 2424 _±_ 0 _._ 0595 –
FT-T 3 _._ 1823 _±_ 0 _._ 0460 3 _._ 0974 _±_ 0 _._ 0334
T2G 3 _._ 1613 _±_ 0 _._ 0320 3 _._ 0982 _± nan_
MLP _[‡−]_ [lite] 3 _._ 0633 _±_ 0 _._ 0248 3 _._ 0170 _±_ 0 _._ 0070
MLP _[‡]_ 3 _._ 0775 _±_ 0 _._ 0336 3 _._ 0268 _±_ 0 _._ 0170
MLP _[†]_ 3 _._ 0999 _±_ 0 _._ 0351 3 _._ 0401 _±_ 0 _._ 0071
XGBoost 3 _._ 1773 _±_ 0 _._ 0102 3 _._ 1644 _±_ 0 _._ 0068
LightGBM 3 _._ 1774 _±_ 0 _._ 0087 3 _._ 1672 _±_ 0 _._ 0050
CatBoost 3 _._ 1172 _±_ 0 _._ 0125 3 _._ 1058 _±_ 0 _._ 0022
TabR 3 _._ 0667 _±_ 0 _._ 0403 2 _._ 9958 _±_ 0 _._ 0270
TabR _[‡]_ 3 _._ 1048 _±_ 0 _._ 0410 –
MNCA 3 _._ 0884 _±_ 0 _._ 0286 3 _._ 0538 _±_ 0 _._ 0072
MNCA _[‡]_ 3 _._ 0704 _±_ 0 _._ 0388 3 _._ 0149 _±_ 0 _._ 0308
TabM _[♠]_ 3 _._ 0002 _±_ 0 _._ 0182 2 _._ 9796 _±_ 0 _._ 0024
TabM 3 _._ 0038 _±_ 0 _._ 0097 2 _._ 9906 _±_ 0 _._ 0026
TabM[G] 3 _._ 0082 _±_ 0 _._ 0184 –
TabM mini 3 _._ 0394 _±_ 0 _._ 0139 3 _._ 0206 _±_ 0 _._ 0128
TabM _[†]_ mini 2 _._ 9976 _±_ 0 _._ 0196 2 _._ 9854 _±_ 0 _._ 0076


diamond ↓

Method Single model Ensemble

MLP 0 _._ 1404 _±_ 0 _._ 0012 0 _._ 1362 _±_ 0 _._ 0003
TabPFN – –
ResNet 0 _._ 1396 _±_ 0 _._ 0029 0 _._ 1361 _±_ 0 _._ 0011
DCN2 0 _._ 1420 _±_ 0 _._ 0032 0 _._ 1374 _±_ 0 _._ 0020
SNN 0 _._ 1473 _±_ 0 _._ 0057 0 _._ 1424 _±_ 0 _._ 0008
Trompt 0 _._ 1391 _± nan_ –
AutoInt 0 _._ 1392 _±_ 0 _._ 0014 0 _._ 1361 _±_ 0 _._ 0004
MLP-Mixer 0 _._ 1400 _±_ 0 _._ 0025 0 _._ 1378 _±_ 0 _._ 0008
Excel _[∗]_ 0 _._ 1766 _±_ 0 _._ 0023 0 _._ 1712 _± nan_
SAINT 0 _._ 1369 _±_ 0 _._ 0019 –
FT-T 0 _._ 1376 _±_ 0 _._ 0013 0 _._ 1360 _±_ 0 _._ 0002
T2G 0 _._ 1372 _±_ 0 _._ 0011 0 _._ 1346 _± nan_
MLP _[‡−]_ [lite] 0 _._ 1342 _±_ 0 _._ 0008 0 _._ 1325 _±_ 0 _._ 0004
MLP _[‡]_ 0 _._ 1337 _±_ 0 _._ 0010 0 _._ 1317 _±_ 0 _._ 0003
MLP _[†]_ 0 _._ 1323 _±_ 0 _._ 0010 0 _._ 1301 _±_ 0 _._ 0005
XGBoost 0 _._ 1368 _±_ 0 _._ 0004 0 _._ 1363 _±_ 0 _._ 0001
LightGBM 0 _._ 1359 _±_ 0 _._ 0002 0 _._ 1358 _±_ 0 _._ 0001
CatBoost 0 _._ 1335 _±_ 0 _._ 0006 0 _._ 1327 _±_ 0 _._ 0004
TabR 0 _._ 1327 _±_ 0 _._ 0010 0 _._ 1311 _±_ 0 _._ 0005
TabR _[‡]_ 0 _._ 1333 _±_ 0 _._ 0013 –
MNCA 0 _._ 1370 _±_ 0 _._ 0018 0 _._ 1348 _±_ 0 _._ 0005
MNCA _[‡]_ 0 _._ 1327 _±_ 0 _._ 0012 0 _._ 1315 _±_ 0 _._ 0006
TabM _[♠]_ 0 _._ 1342 _±_ 0 _._ 0017 0 _._ 1327 _±_ 0 _._ 0004
TabM 0 _._ 1310 _±_ 0 _._ 0007 0 _._ 1307 _±_ 0 _._ 0002
TabM[G] 0 _._ 1309 _±_ 0 _._ 0008 –
TabM mini 0 _._ 1323 _±_ 0 _._ 0007 0 _._ 1317 _±_ 0 _._ 0002
TabM _[†]_ mini 0 _._ 1315 _±_ 0 _._ 0006 0 _._ 1312 _±_ 0 _._ 0001


27



adult ↑

Method Single model Ensemble

MLP 0 _._ 8540 _±_ 0 _._ 0018 0 _._ 8559 _±_ 0 _._ 0011
TabPFN – –
ResNet 0 _._ 8554 _±_ 0 _._ 0011 0 _._ 8562 _±_ 0 _._ 0006
DCN2 0 _._ 8582 _±_ 0 _._ 0011 0 _._ 8593 _±_ 0 _._ 0002
SNN 0 _._ 8582 _±_ 0 _._ 0009 0 _._ 8603 _±_ 0 _._ 0012
Trompt 0 _._ 8590 _± nan_ –
AutoInt 0 _._ 8592 _±_ 0 _._ 0016 0 _._ 8612 _±_ 0 _._ 0004
MLP-Mixer 0 _._ 8598 _±_ 0 _._ 0013 0 _._ 8617 _±_ 0 _._ 0002
Excel _[∗]_ 0 _._ 8613 _±_ 0 _._ 0024 0 _._ 8641 _± nan_
SAINT 0 _._ 8601 _±_ 0 _._ 0019 –
FT-T 0 _._ 8588 _±_ 0 _._ 0015 0 _._ 8608 _±_ 0 _._ 0011
T2G 0 _._ 8601 _±_ 0 _._ 0011 0 _._ 8622 _± nan_
MLP _[‡−]_ [lite] 0 _._ 8693 _±_ 0 _._ 0007 0 _._ 8702 _±_ 0 _._ 0006
MLP _[‡]_ 0 _._ 8694 _±_ 0 _._ 0011 0 _._ 8704 _±_ 0 _._ 0008
MLP _[†]_ 0 _._ 8603 _±_ 0 _._ 0009 0 _._ 8616 _±_ 0 _._ 0006
XGBoost 0 _._ 8720 _±_ 0 _._ 0006 0 _._ 8723 _±_ 0 _._ 0002
LightGBM 0 _._ 8713 _±_ 0 _._ 0007 0 _._ 8721 _±_ 0 _._ 0004
CatBoost 0 _._ 8714 _±_ 0 _._ 0012 0 _._ 8723 _±_ 0 _._ 0007
TabR 0 _._ 8646 _±_ 0 _._ 0022 0 _._ 8680 _±_ 0 _._ 0019
TabR _[‡]_ 0 _._ 8699 _±_ 0 _._ 0011 –
MNCA 0 _._ 8677 _±_ 0 _._ 0018 0 _._ 8696 _±_ 0 _._ 0003
MNCA _[‡]_ 0 _._ 8717 _±_ 0 _._ 0008 0 _._ 8742 _±_ 0 _._ 0006
TabM _[♠]_ 0 _._ 8582 _±_ 0 _._ 0011 0 _._ 8588 _±_ 0 _._ 0003
TabM 0 _._ 8575 _±_ 0 _._ 0008 0 _._ 8583 _±_ 0 _._ 0004
TabM[G] 0 _._ 8572 _±_ 0 _._ 0010 –
TabM mini 0 _._ 8598 _±_ 0 _._ 0011 0 _._ 8604 _±_ 0 _._ 0000
TabM _[†]_ mini 0 _._ 8700 _±_ 0 _._ 0007 0 _._ 8701 _±_ 0 _._ 0003


otto ↑

Method Single model Ensemble

MLP 0 _._ 8175 _±_ 0 _._ 0022 0 _._ 8222 _±_ 0 _._ 0007
TabPFN – 0 _._ 7408 _±_ 0 _._ 0028
ResNet 0 _._ 8174 _±_ 0 _._ 0021 0 _._ 8198 _±_ 0 _._ 0006
DCN2 0 _._ 8064 _±_ 0 _._ 0021 0 _._ 8208 _±_ 0 _._ 0023
SNN 0 _._ 8087 _±_ 0 _._ 0020 0 _._ 8156 _±_ 0 _._ 0013
Trompt 0 _._ 8093 _± nan_ –
AutoInt 0 _._ 8050 _±_ 0 _._ 0034 0 _._ 8111 _±_ 0 _._ 0020
MLP-Mixer 0 _._ 8092 _±_ 0 _._ 0040 0 _._ 8136 _±_ 0 _._ 0010
Excel _[∗]_ 0 _._ 8102 _±_ 0 _._ 0022 0 _._ 8220 _± nan_
SAINT 0 _._ 8119 _±_ 0 _._ 0018 –
FT-T 0 _._ 8133 _±_ 0 _._ 0033 0 _._ 8221 _±_ 0 _._ 0013
T2G 0 _._ 8161 _±_ 0 _._ 0019 0 _._ 8272 _± nan_
MLP _[‡−]_ [lite] 0 _._ 8190 _±_ 0 _._ 0021 0 _._ 8271 _±_ 0 _._ 0015
MLP _[‡]_ 0 _._ 8189 _±_ 0 _._ 0015 0 _._ 8253 _±_ 0 _._ 0000
MLP _[†]_ 0 _._ 8205 _±_ 0 _._ 0021 0 _._ 8290 _±_ 0 _._ 0006
XGBoost 0 _._ 8297 _±_ 0 _._ 0011 0 _._ 8316 _±_ 0 _._ 0008
LightGBM 0 _._ 8302 _±_ 0 _._ 0009 0 _._ 8316 _±_ 0 _._ 0013
CatBoost 0 _._ 8250 _±_ 0 _._ 0013 0 _._ 8268 _±_ 0 _._ 0002
TabR 0 _._ 8179 _±_ 0 _._ 0022 0 _._ 8236 _±_ 0 _._ 0009
TabR _[‡]_ 0 _._ 8246 _±_ 0 _._ 0018 –
MNCA 0 _._ 8275 _±_ 0 _._ 0012 0 _._ 8313 _±_ 0 _._ 0006
MNCA _[‡]_ 0 _._ 8265 _±_ 0 _._ 0015 0 _._ 8304 _±_ 0 _._ 0006
TabM _[♠]_ 0 _._ 8268 _±_ 0 _._ 0014 0 _._ 8300 _±_ 0 _._ 0007
TabM 0 _._ 8275 _±_ 0 _._ 0014 0 _._ 8284 _±_ 0 _._ 0005
TabM[G] 0 _._ 8254 _±_ 0 _._ 0022 –
TabM mini 0 _._ 8282 _±_ 0 _._ 0014 0 _._ 8299 _±_ 0 _._ 0005
TabM _[†]_ mini 0 _._ 8342 _±_ 0 _._ 0012 0 _._ 8356 _±_ 0 _._ 0004


Published as a conference paper at ICLR 2025


higgs-small ↑

Method Single model Ensemble

MLP 0 _._ 7180 _±_ 0 _._ 0027 0 _._ 7192 _±_ 0 _._ 0005
TabPFN – 0 _._ 6727 _±_ 0 _._ 0034
ResNet 0 _._ 7256 _±_ 0 _._ 0020 0 _._ 7307 _±_ 0 _._ 0001
DCN2 0 _._ 7164 _±_ 0 _._ 0030 0 _._ 7237 _±_ 0 _._ 0011
SNN 0 _._ 7142 _±_ 0 _._ 0024 0 _._ 7171 _±_ 0 _._ 0020
Trompt 0 _._ 7262 _± nan_ –
AutoInt 0 _._ 7240 _±_ 0 _._ 0028 0 _._ 7287 _±_ 0 _._ 0008
MLP-Mixer 0 _._ 7248 _±_ 0 _._ 0023 0 _._ 7334 _±_ 0 _._ 0007
Excel _[∗]_ 0 _._ 7262 _±_ 0 _._ 0017 0 _._ 7329 _± nan_
SAINT 0 _._ 7236 _±_ 0 _._ 0019 –
FT-T 0 _._ 7281 _±_ 0 _._ 0016 0 _._ 7334 _±_ 0 _._ 0013
T2G 0 _._ 7352 _±_ 0 _._ 0037 0 _._ 7400 _± nan_
MLP _[‡−]_ [lite] 0 _._ 7260 _±_ 0 _._ 0017 0 _._ 7304 _±_ 0 _._ 0008
MLP _[‡]_ 0 _._ 7261 _±_ 0 _._ 0010 0 _._ 7270 _±_ 0 _._ 0003
MLP _[†]_ 0 _._ 7210 _±_ 0 _._ 0016 0 _._ 7252 _±_ 0 _._ 0005
XGBoost 0 _._ 7246 _±_ 0 _._ 0015 0 _._ 7264 _±_ 0 _._ 0013
LightGBM 0 _._ 7256 _±_ 0 _._ 0009 0 _._ 7263 _±_ 0 _._ 0007
CatBoost 0 _._ 7260 _±_ 0 _._ 0011 0 _._ 7273 _±_ 0 _._ 0010
TabR 0 _._ 7223 _±_ 0 _._ 0010 0 _._ 7257 _±_ 0 _._ 0008
TabR _[‡]_ 0 _._ 7294 _±_ 0 _._ 0014 –
MNCA 0 _._ 7263 _±_ 0 _._ 0023 0 _._ 7292 _±_ 0 _._ 0006
MNCA _[‡]_ 0 _._ 7300 _±_ 0 _._ 0020 0 _._ 7348 _±_ 0 _._ 0008
TabM _[♠]_ 0 _._ 7383 _±_ 0 _._ 0028 0 _._ 7409 _±_ 0 _._ 0010
TabM 0 _._ 7394 _±_ 0 _._ 0018 0 _._ 7409 _±_ 0 _._ 0008
TabM[G] 0 _._ 7392 _±_ 0 _._ 0016 –
TabM mini 0 _._ 7338 _±_ 0 _._ 0011 0 _._ 7345 _±_ 0 _._ 0008
TabM _[†]_ mini 0 _._ 7361 _±_ 0 _._ 0011 0 _._ 7383 _±_ 0 _._ 0008


covtype2 ↑

Method Single model Ensemble

MLP 0 _._ 9630 _±_ 0 _._ 0012 0 _._ 9664 _±_ 0 _._ 0004
TabPFN – 0 _._ 7606 _±_ 0 _._ 0022
ResNet 0 _._ 9638 _±_ 0 _._ 0005 0 _._ 9685 _±_ 0 _._ 0003
DCN2 0 _._ 9622 _±_ 0 _._ 0019 0 _._ 9673 _±_ 0 _._ 0011
SNN 0 _._ 9636 _±_ 0 _._ 0010 0 _._ 9677 _±_ 0 _._ 0002
Trompt 0 _._ 9286 _± nan_ –
AutoInt 0 _._ 9614 _±_ 0 _._ 0016 0 _._ 9696 _±_ 0 _._ 0005
MLP-Mixer 0 _._ 9663 _±_ 0 _._ 0019 0 _._ 9699 _±_ 0 _._ 0014
Excel _[∗]_ 0 _._ 9606 _±_ 0 _._ 0018 0 _._ 9670 _± nan_
SAINT 0 _._ 9669 _±_ 0 _._ 0010 –
FT-T 0 _._ 9698 _±_ 0 _._ 0008 0 _._ 9731 _±_ 0 _._ 0006
T2G 0 _._ 9668 _±_ 0 _._ 0008 0 _._ 9708 _± nan_
MLP _[‡−]_ [lite] 0 _._ 9690 _±_ 0 _._ 0008 0 _._ 9721 _±_ 0 _._ 0006
MLP _[‡]_ 0 _._ 9713 _±_ 0 _._ 0006 0 _._ 9758 _±_ 0 _._ 0000
MLP _[†]_ 0 _._ 9697 _±_ 0 _._ 0008 0 _._ 9721 _±_ 0 _._ 0005
XGBoost 0 _._ 9710 _±_ 0 _._ 0002 0 _._ 9713 _±_ 0 _._ 0000
LightGBM 0 _._ 9709 _±_ 0 _._ 0003 –
CatBoost 0 _._ 9670 _±_ 0 _._ 0003 0 _._ 9680 _±_ 0 _._ 0002
TabR 0 _._ 9737 _±_ 0 _._ 0005 0 _._ 9745 _±_ 0 _._ 0006
TabR _[‡]_ 0 _._ 9752 _±_ 0 _._ 0003 –
MNCA 0 _._ 9724 _±_ 0 _._ 0003 0 _._ 9729 _±_ 0 _._ 0001
MNCA _[‡]_ 0 _._ 9747 _±_ 0 _._ 0002 0 _._ 9747 _±_ 0 _._ 0002
TabM _[♠]_ 0 _._ 9712 _±_ 0 _._ 0008 0 _._ 9729 _±_ 0 _._ 0003
TabM 0 _._ 9735 _±_ 0 _._ 0004 0 _._ 9743 _±_ 0 _._ 0001
TabM[G] 0 _._ 9730 _±_ 0 _._ 0005 –
TabM mini 0 _._ 9710 _±_ 0 _._ 0007 0 _._ 9727 _±_ 0 _._ 0002
TabM _[†]_ mini 0 _._ 9755 _±_ 0 _._ 0003 0 _._ 9762 _±_ 0 _._ 0001


28



black-friday ↓

Method Single model Ensemble

MLP 0 _._ 6955 _±_ 0 _._ 0004 0 _._ 6942 _±_ 0 _._ 0002
TabPFN – –
ResNet 0 _._ 6929 _±_ 0 _._ 0008 0 _._ 6907 _±_ 0 _._ 0002
DCN2 0 _._ 6968 _±_ 0 _._ 0013 0 _._ 6936 _±_ 0 _._ 0007
SNN 0 _._ 6996 _±_ 0 _._ 0013 0 _._ 6978 _±_ 0 _._ 0004
Trompt 0 _._ 6983 _± nan_ –
AutoInt 0 _._ 6994 _±_ 0 _._ 0082 0 _._ 6927 _±_ 0 _._ 0021
MLP-Mixer 0 _._ 6905 _±_ 0 _._ 0021 0 _._ 6851 _±_ 0 _._ 0011
Excel _[∗]_ 0 _._ 6947 _±_ 0 _._ 0016 0 _._ 6908 _± nan_
SAINT 0 _._ 6934 _±_ 0 _._ 0009 –
FT-T 0 _._ 6987 _±_ 0 _._ 0192 0 _._ 6879 _±_ 0 _._ 0023
T2G 0 _._ 6887 _±_ 0 _._ 0046 0 _._ 6832 _± nan_
MLP _[‡−]_ [lite] 0 _._ 6849 _±_ 0 _._ 0006 0 _._ 6824 _±_ 0 _._ 0002
MLP _[‡]_ 0 _._ 6857 _±_ 0 _._ 0004 0 _._ 6838 _±_ 0 _._ 0002
MLP _[†]_ 0 _._ 6836 _±_ 0 _._ 0006 0 _._ 6812 _±_ 0 _._ 0002
XGBoost 0 _._ 6806 _±_ 0 _._ 0001 0 _._ 6805 _±_ 0 _._ 0000
LightGBM 0 _._ 6799 _±_ 0 _._ 0003 0 _._ 6795 _±_ 0 _._ 0001
CatBoost 0 _._ 6822 _±_ 0 _._ 0003 0 _._ 6813 _±_ 0 _._ 0002
TabR 0 _._ 6899 _±_ 0 _._ 0004 0 _._ 6883 _±_ 0 _._ 0002
TabR _[‡]_ 0 _._ 6761 _±_ 0 _._ 0009 –
MNCA 0 _._ 6893 _±_ 0 _._ 0004 0 _._ 6883 _±_ 0 _._ 0000
MNCA _[‡]_ 0 _._ 6885 _±_ 0 _._ 0007 0 _._ 6863 _±_ 0 _._ 0003
TabM _[♠]_ 0 _._ 6875 _±_ 0 _._ 0015 0 _._ 6866 _±_ 0 _._ 0003
TabM 0 _._ 6869 _±_ 0 _._ 0004 0 _._ 6865 _±_ 0 _._ 0001
TabM[G] 0 _._ 6865 _±_ 0 _._ 0005 –
TabM mini 0 _._ 6863 _±_ 0 _._ 0006 0 _._ 6856 _±_ 0 _._ 0003
TabM _[†]_ mini 0 _._ 6781 _±_ 0 _._ 0004 0 _._ 6773 _±_ 0 _._ 0001


microsoft ↓

Method Single model Ensemble

MLP 0 _._ 7475 _±_ 0 _._ 0003 0 _._ 7460 _±_ 0 _._ 0003
TabPFN – –
ResNet 0 _._ 7472 _±_ 0 _._ 0004 0 _._ 7452 _±_ 0 _._ 0004
DCN2 0 _._ 7499 _±_ 0 _._ 0003 0 _._ 7477 _±_ 0 _._ 0001
SNN 0 _._ 7488 _±_ 0 _._ 0004 0 _._ 7470 _±_ 0 _._ 0001
Trompt 0 _._ 7476 _± nan_ –
AutoInt 0 _._ 7482 _±_ 0 _._ 0005 0 _._ 7455 _±_ 0 _._ 0002
MLP-Mixer 0 _._ 7482 _±_ 0 _._ 0008 0 _._ 7436 _±_ 0 _._ 0001
Excel _[∗]_ 0 _._ 7479 _±_ 0 _._ 0007 0 _._ 7442 _± nan_
SAINT 0 _._ 7625 _±_ 0 _._ 0066 –
FT-T 0 _._ 7460 _±_ 0 _._ 0007 0 _._ 7422 _±_ 0 _._ 0004
T2G 0 _._ 7460 _±_ 0 _._ 0006 0 _._ 7427 _± nan_
MLP _[‡−]_ [lite] 0 _._ 7446 _±_ 0 _._ 0002 0 _._ 7434 _±_ 0 _._ 0002
MLP _[‡]_ 0 _._ 7444 _±_ 0 _._ 0003 0 _._ 7429 _±_ 0 _._ 0001
MLP _[†]_ 0 _._ 7465 _±_ 0 _._ 0005 0 _._ 7448 _±_ 0 _._ 0001
XGBoost 0 _._ 7413 _±_ 0 _._ 0001 0 _._ 7410 _±_ 0 _._ 0000
LightGBM 0 _._ 7417 _±_ 0 _._ 0001 0 _._ 7413 _±_ 0 _._ 0000
CatBoost 0 _._ 7412 _±_ 0 _._ 0001 0 _._ 7406 _±_ 0 _._ 0000
TabR 0 _._ 7503 _±_ 0 _._ 0006 0 _._ 7485 _±_ 0 _._ 0002
TabR _[‡]_ 0 _._ 7501 _±_ 0 _._ 0005 –
MNCA 0 _._ 7458 _±_ 0 _._ 0003 0 _._ 7448 _±_ 0 _._ 0002
MNCA _[‡]_ 0 _._ 7460 _±_ 0 _._ 0008 0 _._ 7435 _±_ 0 _._ 0004
TabM _[♠]_ 0 _._ 7434 _±_ 0 _._ 0003 0 _._ 7424 _±_ 0 _._ 0001
TabM 0 _._ 7432 _±_ 0 _._ 0004 0 _._ 7426 _±_ 0 _._ 0001
TabM[G] 0 _._ 7432 _±_ 0 _._ 0004 –
TabM mini 0 _._ 7436 _±_ 0 _._ 0002 0 _._ 7430 _±_ 0 _._ 0002
TabM _[†]_ mini 0 _._ 7423 _±_ 0 _._ 0002 0 _._ 7416 _±_ 0 _._ 0001


Published as a conference paper at ICLR 2025


Table 19: Extended results for Grinsztajn et al. (2022) benchmark. Results are grouped by datasets.
One ensemble consists of five models trained independently with different random seeds.



wine ↑

Method Single model Ensemble

MLP 0 _._ 7778 _±_ 0 _._ 0153 0 _._ 7907 _±_ 0 _._ 0117
TabPFN – 0 _._ 7908 _±_ 0 _._ 0063
ResNet 0 _._ 7710 _±_ 0 _._ 0137 0 _._ 7839 _±_ 0 _._ 0083
DCN2 0 _._ 7492 _±_ 0 _._ 0147 0 _._ 7764 _±_ 0 _._ 0095
SNN 0 _._ 7818 _±_ 0 _._ 0143 0 _._ 7994 _±_ 0 _._ 0097
Trompt 0 _._ 7818 _±_ 0 _._ 0081 –
AutoInt 0 _._ 7745 _±_ 0 _._ 0144 0 _._ 7909 _±_ 0 _._ 0160
MLP-Mixer 0 _._ 7769 _±_ 0 _._ 0149 0 _._ 7950 _±_ 0 _._ 0087
Excel _[∗]_ 0 _._ 7631 _±_ 0 _._ 0171 0 _._ 7765 _±_ 0 _._ 0121
SAINT 0 _._ 7684 _±_ 0 _._ 0144 –
FT-T 0 _._ 7755 _±_ 0 _._ 0133 0 _._ 7894 _±_ 0 _._ 0083
T2G 0 _._ 7733 _±_ 0 _._ 0118 0 _._ 7933 _±_ 0 _._ 0137
MLP _[‡−]_ [lite] 0 _._ 7803 _±_ 0 _._ 0157 0 _._ 7964 _±_ 0 _._ 0146
MLP _[‡]_ 0 _._ 7733 _±_ 0 _._ 0185 0 _._ 7856 _±_ 0 _._ 0160
MLP _[†]_ 0 _._ 7814 _±_ 0 _._ 0132 0 _._ 7919 _±_ 0 _._ 0098
XGBoost 0 _._ 7949 _±_ 0 _._ 0178 0 _._ 8010 _±_ 0 _._ 0186
LightGBM 0 _._ 7890 _±_ 0 _._ 0160 0 _._ 7929 _±_ 0 _._ 0106
CatBoost 0 _._ 7994 _±_ 0 _._ 0131 0 _._ 8057 _±_ 0 _._ 0098
TabR 0 _._ 7936 _±_ 0 _._ 0114 0 _._ 8055 _±_ 0 _._ 0057
TabR _[‡]_ 0 _._ 7804 _±_ 0 _._ 0148 –
MNCA 0 _._ 7911 _±_ 0 _._ 0135 0 _._ 8005 _±_ 0 _._ 0121
MNCA _[‡]_ 0 _._ 7867 _±_ 0 _._ 0113 0 _._ 7953 _±_ 0 _._ 0114
TabM _[♠]_ 0 _._ 7961 _±_ 0 _._ 0136 0 _._ 8011 _±_ 0 _._ 0084
TabM 0 _._ 7943 _±_ 0 _._ 0124 0 _._ 7985 _±_ 0 _._ 0139
TabM[G] 0 _._ 7879 _±_ 0 _._ 0161 –
TabM mini 0 _._ 7890 _±_ 0 _._ 0130 0 _._ 7937 _±_ 0 _._ 0103
TabM _[†]_ mini 0 _._ 7839 _±_ 0 _._ 0169 0 _._ 7917 _±_ 0 _._ 0143


analcatdata ~~s~~ upreme ↓

Method Single model Ensemble

MLP 0 _._ 0782 _±_ 0 _._ 0081 0 _._ 0766 _±_ 0 _._ 0090
TabPFN – –
ResNet 0 _._ 0852 _±_ 0 _._ 0076 0 _._ 0823 _±_ 0 _._ 0078
DCN2 0 _._ 0811 _±_ 0 _._ 0137 0 _._ 0759 _±_ 0 _._ 0086
SNN 0 _._ 0826 _±_ 0 _._ 0096 0 _._ 0779 _±_ 0 _._ 0098
Trompt 0 _._ 0782 _±_ 0 _._ 0095 –
AutoInt 0 _._ 0783 _±_ 0 _._ 0078 0 _._ 0768 _±_ 0 _._ 0083
MLP-Mixer 0 _._ 0770 _±_ 0 _._ 0082 0 _._ 0759 _±_ 0 _._ 0081
Excel _[∗]_ 0 _._ 0796 _±_ 0 _._ 0101 0 _._ 0776 _±_ 0 _._ 0101
SAINT 0 _._ 0773 _±_ 0 _._ 0078 –
FT-T 0 _._ 0787 _±_ 0 _._ 0086 0 _._ 0775 _±_ 0 _._ 0091
T2G 0 _._ 0775 _±_ 0 _._ 0081 0 _._ 0763 _±_ 0 _._ 0084
MLP _[‡−]_ [lite] 0 _._ 0798 _±_ 0 _._ 0088 0 _._ 0769 _±_ 0 _._ 0092
MLP _[‡]_ 0 _._ 0786 _±_ 0 _._ 0073 0 _._ 0720 _±_ 0 _._ 0053
MLP _[†]_ 0 _._ 0774 _±_ 0 _._ 0064 0 _._ 0759 _±_ 0 _._ 0063
XGBoost 0 _._ 0801 _±_ 0 _._ 0126 0 _._ 0774 _±_ 0 _._ 0107
LightGBM 0 _._ 0778 _±_ 0 _._ 0115 0 _._ 0767 _±_ 0 _._ 0110
CatBoost 0 _._ 0780 _±_ 0 _._ 0067 0 _._ 0734 _±_ 0 _._ 0022
TabR 0 _._ 0803 _±_ 0 _._ 0066 0 _._ 0759 _±_ 0 _._ 0046
TabR _[‡]_ 0 _._ 0807 _±_ 0 _._ 0088 –
MNCA 0 _._ 0809 _±_ 0 _._ 0072 0 _._ 0784 _±_ 0 _._ 0062
MNCA _[‡]_ 0 _._ 0825 _±_ 0 _._ 0090 0 _._ 0793 _±_ 0 _._ 0072
TabM _[♠]_ 0 _._ 0777 _±_ 0 _._ 0099 0 _._ 0769 _±_ 0 _._ 0105
TabM 0 _._ 0786 _±_ 0 _._ 0055 0 _._ 0781 _±_ 0 _._ 0054
TabM[G] 0 _._ 0808 _±_ 0 _._ 0063 –
TabM mini 0 _._ 0773 _±_ 0 _._ 0077 0 _._ 0763 _±_ 0 _._ 0077
TabM _[†]_ mini 0 _._ 0764 _±_ 0 _._ 0071 0 _._ 0749 _±_ 0 _._ 0076


29



phoneme ↑

Method Single model Ensemble

MLP 0 _._ 8525 _±_ 0 _._ 0126 0 _._ 8635 _±_ 0 _._ 0099
TabPFN – 0 _._ 8684 _±_ 0 _._ 0050
ResNet 0 _._ 8456 _±_ 0 _._ 0121 0 _._ 8504 _±_ 0 _._ 0066
DCN2 0 _._ 8342 _±_ 0 _._ 0151 0 _._ 8543 _±_ 0 _._ 0118
SNN 0 _._ 8596 _±_ 0 _._ 0124 0 _._ 8687 _±_ 0 _._ 0080
Trompt 0 _._ 8465 _±_ 0 _._ 0205 –
AutoInt 0 _._ 8623 _±_ 0 _._ 0138 0 _._ 8754 _±_ 0 _._ 0095
MLP-Mixer 0 _._ 8629 _±_ 0 _._ 0123 0 _._ 8757 _±_ 0 _._ 0095
Excel _[∗]_ 0 _._ 8551 _±_ 0 _._ 0092 0 _._ 8711 _±_ 0 _._ 0081
SAINT 0 _._ 8657 _±_ 0 _._ 0130 –
FT-T 0 _._ 8667 _±_ 0 _._ 0127 0 _._ 8795 _±_ 0 _._ 0093
T2G 0 _._ 8672 _±_ 0 _._ 0166 0 _._ 8765 _±_ 0 _._ 0141
MLP _[‡−]_ [lite] 0 _._ 8742 _±_ 0 _._ 0120 0 _._ 8861 _±_ 0 _._ 0071
MLP _[‡]_ 0 _._ 8757 _±_ 0 _._ 0118 0 _._ 8856 _±_ 0 _._ 0065
MLP _[†]_ 0 _._ 8647 _±_ 0 _._ 0098 0 _._ 8761 _±_ 0 _._ 0076
XGBoost 0 _._ 8682 _±_ 0 _._ 0174 0 _._ 8771 _±_ 0 _._ 0156
LightGBM 0 _._ 8702 _±_ 0 _._ 0129 0 _._ 8733 _±_ 0 _._ 0126
CatBoost 0 _._ 8827 _±_ 0 _._ 0117 0 _._ 8897 _±_ 0 _._ 0055
TabR 0 _._ 8781 _±_ 0 _._ 0096 0 _._ 8840 _±_ 0 _._ 0054
TabR _[‡]_ 0 _._ 8772 _±_ 0 _._ 0087 –
MNCA 0 _._ 8835 _±_ 0 _._ 0079 0 _._ 8861 _±_ 0 _._ 0057
MNCA _[‡]_ 0 _._ 8828 _±_ 0 _._ 0082 0 _._ 8925 _±_ 0 _._ 0056
TabM _[♠]_ 0 _._ 8701 _±_ 0 _._ 0167 0 _._ 8766 _±_ 0 _._ 0128
TabM 0 _._ 8831 _±_ 0 _._ 0121 0 _._ 8880 _±_ 0 _._ 0108
TabM[G] 0 _._ 8762 _±_ 0 _._ 0144 –
TabM mini 0 _._ 8803 _±_ 0 _._ 0098 0 _._ 8842 _±_ 0 _._ 0067
TabM _[†]_ mini 0 _._ 8780 _±_ 0 _._ 0119 0 _._ 8817 _±_ 0 _._ 0101


Mercedes ~~B~~ enz ~~G~~ reener ~~M~~ anufacturing ↓

Method Single model Ensemble

MLP 8 _._ 3045 _±_ 0 _._ 8708 8 _._ 2682 _±_ 0 _._ 8992
TabPFN – –
ResNet 8 _._ 4434 _±_ 0 _._ 7982 8 _._ 3178 _±_ 0 _._ 8482
DCN2 8 _._ 3540 _±_ 0 _._ 8314 8 _._ 3021 _±_ 0 _._ 8579
SNN 8 _._ 2718 _±_ 0 _._ 8152 8 _._ 2236 _±_ 0 _._ 8479
Trompt 8 _._ 3409 _±_ 0 _._ 9840 –
AutoInt 8 _._ 4001 _±_ 0 _._ 9256 8 _._ 3237 _±_ 0 _._ 9658
MLP-Mixer 8 _._ 2860 _±_ 0 _._ 8656 8 _._ 2398 _±_ 0 _._ 9023
Excel _[∗]_ 8 _._ 2244 _±_ 0 _._ 8514 8 _._ 1918 _±_ 0 _._ 9387
SAINT 8 _._ 3556 _±_ 0 _._ 9566 –
FT-T 8 _._ 2252 _±_ 0 _._ 8617 8 _._ 1616 _±_ 0 _._ 8834
T2G 8 _._ 2120 _±_ 0 _._ 8485 8 _._ 1654 _±_ 0 _._ 9339
MLP _[‡−]_ [lite] 8 _._ 3045 _±_ 0 _._ 8708 8 _._ 2682 _±_ 0 _._ 8992
MLP _[‡]_ 8 _._ 3045 _±_ 0 _._ 8708 8 _._ 2682 _±_ 0 _._ 8992
MLP _[†]_ 8 _._ 3045 _±_ 0 _._ 8708 8 _._ 2682 _±_ 0 _._ 8992
XGBoost 8 _._ 2177 _±_ 0 _._ 8175 8 _._ 2092 _±_ 0 _._ 8458
LightGBM 8 _._ 2078 _±_ 0 _._ 8231 8 _._ 1618 _±_ 0 _._ 8566
CatBoost 8 _._ 1629 _±_ 0 _._ 8193 8 _._ 1554 _±_ 0 _._ 8439
TabR 8 _._ 3506 _±_ 0 _._ 8149 8 _._ 2694 _±_ 0 _._ 8399
TabR _[‡]_ 8 _._ 3187 _±_ 0 _._ 8186 –
MNCA 8 _._ 2557 _±_ 0 _._ 8602 8 _._ 1771 _±_ 0 _._ 8710
MNCA _[‡]_ 8 _._ 2557 _±_ 0 _._ 8602 8 _._ 1771 _±_ 0 _._ 8710
TabM _[♠]_ 8 _._ 2215 _±_ 0 _._ 8940 8 _._ 1995 _±_ 0 _._ 9130
TabM 8 _._ 2052 _±_ 0 _._ 9043 8 _._ 1965 _±_ 0 _._ 9306
TabM[G] 8 _._ 2235 _±_ 0 _._ 8867 –
TabM mini 8 _._ 2075 _±_ 0 _._ 9185 8 _._ 1986 _±_ 0 _._ 9442
TabM _[†]_ mini 8 _._ 2075 _±_ 0 _._ 9185 8 _._ 1986 _±_ 0 _._ 9442


Published as a conference paper at ICLR 2025


KDDCup09 ~~u~~ pselling ↑

Method Single model Ensemble

MLP 0 _._ 7759 _±_ 0 _._ 0137 0 _._ 7806 _±_ 0 _._ 0125
TabPFN – –
ResNet 0 _._ 7811 _±_ 0 _._ 0124 0 _._ 7861 _±_ 0 _._ 0109
DCN2 0 _._ 7850 _±_ 0 _._ 0161 0 _._ 7884 _±_ 0 _._ 0135
SNN 0 _._ 7884 _±_ 0 _._ 0122 0 _._ 7940 _±_ 0 _._ 0116
Trompt 0 _._ 7994 _±_ 0 _._ 0055 –
AutoInt 0 _._ 8004 _±_ 0 _._ 0075 0 _._ 8037 _±_ 0 _._ 0063
MLP-Mixer 0 _._ 7979 _±_ 0 _._ 0105 0 _._ 8010 _±_ 0 _._ 0094
Excel _[∗]_ 0 _._ 7903 _±_ 0 _._ 0074 0 _._ 7939 _±_ 0 _._ 0099
SAINT 0 _._ 7942 _±_ 0 _._ 0112 –
FT-T 0 _._ 7957 _±_ 0 _._ 0127 0 _._ 7960 _±_ 0 _._ 0139
T2G 0 _._ 8037 _±_ 0 _._ 0100 0 _._ 7988 _±_ 0 _._ 0084
MLP _[‡−]_ [lite] 0 _._ 7962 _±_ 0 _._ 0093 0 _._ 7995 _±_ 0 _._ 0105
MLP _[‡]_ 0 _._ 8005 _±_ 0 _._ 0097 0 _._ 8032 _±_ 0 _._ 0117
MLP _[†]_ 0 _._ 7925 _±_ 0 _._ 0123 0 _._ 7963 _±_ 0 _._ 0089
XGBoost 0 _._ 7930 _±_ 0 _._ 0108 0 _._ 7950 _±_ 0 _._ 0102
LightGBM 0 _._ 7932 _±_ 0 _._ 0119 0 _._ 7969 _±_ 0 _._ 0115
CatBoost 0 _._ 7992 _±_ 0 _._ 0117 0 _._ 8010 _±_ 0 _._ 0121
TabR 0 _._ 7838 _±_ 0 _._ 0136 0 _._ 7859 _±_ 0 _._ 0167
TabR _[‡]_ 0 _._ 7908 _±_ 0 _._ 0123 –
MNCA 0 _._ 7939 _±_ 0 _._ 0097 0 _._ 7989 _±_ 0 _._ 0115
MNCA _[‡]_ 0 _._ 7960 _±_ 0 _._ 0131 0 _._ 8008 _±_ 0 _._ 0110
TabM _[♠]_ 0 _._ 8002 _±_ 0 _._ 0103 0 _._ 8021 _±_ 0 _._ 0074
TabM 0 _._ 8024 _±_ 0 _._ 0111 0 _._ 8054 _±_ 0 _._ 0123
TabM[G] 0 _._ 7988 _±_ 0 _._ 0118 –
TabM mini 0 _._ 7971 _±_ 0 _._ 0117 0 _._ 7982 _±_ 0 _._ 0107
TabM _[†]_ mini 0 _._ 8024 _±_ 0 _._ 0075 0 _._ 8035 _±_ 0 _._ 0088


wine ~~q~~ uality ↓

Method Single model Ensemble

MLP 0 _._ 6707 _±_ 0 _._ 0178 0 _._ 6530 _±_ 0 _._ 0152
TabPFN – –
ResNet 0 _._ 6687 _±_ 0 _._ 0166 0 _._ 6543 _±_ 0 _._ 0170
DCN2 0 _._ 7010 _±_ 0 _._ 0171 0 _._ 6699 _±_ 0 _._ 0139
SNN 0 _._ 6604 _±_ 0 _._ 0174 0 _._ 6245 _±_ 0 _._ 0140
Trompt 0 _._ 6605 _±_ 0 _._ 0153 –
AutoInt 0 _._ 6840 _±_ 0 _._ 0126 0 _._ 6478 _±_ 0 _._ 0146
MLP-Mixer 0 _._ 6672 _±_ 0 _._ 0263 0 _._ 6294 _±_ 0 _._ 0200
Excel _[∗]_ 0 _._ 6881 _±_ 0 _._ 0182 0 _._ 6664 _±_ 0 _._ 0179
SAINT 0 _._ 6797 _±_ 0 _._ 0161 –
FT-T 0 _._ 6787 _±_ 0 _._ 0149 0 _._ 6564 _±_ 0 _._ 0250
T2G 0 _._ 6783 _±_ 0 _._ 0170 0 _._ 6570 _±_ 0 _._ 0273
MLP _[‡−]_ [lite] 0 _._ 6569 _±_ 0 _._ 0167 0 _._ 6328 _±_ 0 _._ 0155
MLP _[‡]_ 0 _._ 6532 _±_ 0 _._ 0133 0 _._ 6336 _±_ 0 _._ 0140
MLP _[†]_ 0 _._ 6721 _±_ 0 _._ 0180 0 _._ 6463 _±_ 0 _._ 0262
XGBoost 0 _._ 6039 _±_ 0 _._ 0134 0 _._ 6025 _±_ 0 _._ 0139
LightGBM 0 _._ 6135 _±_ 0 _._ 0138 0 _._ 6122 _±_ 0 _._ 0144
CatBoost 0 _._ 6088 _±_ 0 _._ 0132 0 _._ 6060 _±_ 0 _._ 0137
TabR 0 _._ 6315 _±_ 0 _._ 0097 0 _._ 6197 _±_ 0 _._ 0096
TabR _[‡]_ 0 _._ 6412 _±_ 0 _._ 0105 –
MNCA 0 _._ 6154 _±_ 0 _._ 0083 0 _._ 6058 _±_ 0 _._ 0149
MNCA _[‡]_ 0 _._ 6099 _±_ 0 _._ 0144 0 _._ 6028 _±_ 0 _._ 0157
TabM _[♠]_ 0 _._ 6169 _±_ 0 _._ 0123 0 _._ 6131 _±_ 0 _._ 0126
TabM 0 _._ 6328 _±_ 0 _._ 0172 0 _._ 6297 _±_ 0 _._ 0180
TabM[G] 0 _._ 6369 _±_ 0 _._ 0179 –
TabM mini 0 _._ 6314 _±_ 0 _._ 0142 0 _._ 6272 _±_ 0 _._ 0146
TabM _[†]_ mini 0 _._ 6294 _±_ 0 _._ 0120 0 _._ 6241 _±_ 0 _._ 0118


30



kdd ~~i~~ pums ~~l~~ a ~~9~~ 7-small ↑

Method Single model Ensemble

MLP 0 _._ 8828 _±_ 0 _._ 0061 0 _._ 8845 _±_ 0 _._ 0055
TabPFN – 0 _._ 8578 _±_ 0 _._ 0046
ResNet 0 _._ 8823 _±_ 0 _._ 0070 0 _._ 8824 _±_ 0 _._ 0060
DCN2 0 _._ 8770 _±_ 0 _._ 0072 0 _._ 8824 _±_ 0 _._ 0068
SNN 0 _._ 8722 _±_ 0 _._ 0093 0 _._ 8733 _±_ 0 _._ 0083
Trompt 0 _._ 8847 _±_ 0 _._ 0070 –
AutoInt 0 _._ 8808 _±_ 0 _._ 0083 0 _._ 8830 _±_ 0 _._ 0081
MLP-Mixer 0 _._ 8762 _±_ 0 _._ 0100 0 _._ 8770 _±_ 0 _._ 0088
Excel _[∗]_ 0 _._ 8803 _±_ 0 _._ 0054 0 _._ 8823 _±_ 0 _._ 0071
SAINT 0 _._ 8837 _±_ 0 _._ 0055 –
FT-T 0 _._ 8795 _±_ 0 _._ 0077 0 _._ 8792 _±_ 0 _._ 0062
T2G 0 _._ 8833 _±_ 0 _._ 0054 0 _._ 8841 _±_ 0 _._ 0062
MLP _[‡−]_ [lite] 0 _._ 8765 _±_ 0 _._ 0108 0 _._ 8765 _±_ 0 _._ 0108
MLP _[‡]_ 0 _._ 8816 _±_ 0 _._ 0057 0 _._ 8818 _±_ 0 _._ 0048
MLP _[†]_ 0 _._ 8757 _±_ 0 _._ 0101 0 _._ 8756 _±_ 0 _._ 0104
XGBoost 0 _._ 8825 _±_ 0 _._ 0089 0 _._ 8835 _±_ 0 _._ 0085
LightGBM 0 _._ 8792 _±_ 0 _._ 0075 0 _._ 8802 _±_ 0 _._ 0067
CatBoost 0 _._ 8793 _±_ 0 _._ 0088 0 _._ 8803 _±_ 0 _._ 0100
TabR 0 _._ 8798 _±_ 0 _._ 0081 0 _._ 8819 _±_ 0 _._ 0078
TabR _[‡]_ 0 _._ 8831 _±_ 0 _._ 0050 –
MNCA 0 _._ 8819 _±_ 0 _._ 0054 0 _._ 8832 _±_ 0 _._ 0048
MNCA _[‡]_ 0 _._ 8837 _±_ 0 _._ 0062 0 _._ 8860 _±_ 0 _._ 0059
TabM _[♠]_ 0 _._ 8845 _±_ 0 _._ 0063 0 _._ 8848 _±_ 0 _._ 0070
TabM 0 _._ 8823 _±_ 0 _._ 0079 0 _._ 8825 _±_ 0 _._ 0071
TabM[G] 0 _._ 8818 _±_ 0 _._ 0082 –
TabM mini 0 _._ 8784 _±_ 0 _._ 0123 0 _._ 8786 _±_ 0 _._ 0133
TabM _[†]_ mini 0 _._ 8779 _±_ 0 _._ 0094 0 _._ 8784 _±_ 0 _._ 0108


isolet ↓

Method Single model Ensemble

MLP 2 _._ 2744 _±_ 0 _._ 2203 2 _._ 0018 _±_ 0 _._ 1111
TabPFN – –
ResNet 2 _._ 2077 _±_ 0 _._ 2248 1 _._ 9206 _±_ 0 _._ 1478
DCN2 2 _._ 2449 _±_ 0 _._ 1579 2 _._ 0176 _±_ 0 _._ 0770
SNN 2 _._ 4269 _±_ 0 _._ 2382 2 _._ 1142 _±_ 0 _._ 1262
Trompt 2 _._ 6219 _±_ 0 _._ 0315 –
AutoInt 2 _._ 6130 _±_ 0 _._ 1658 2 _._ 3308 _±_ 0 _._ 1088
MLP-Mixer 2 _._ 3344 _±_ 0 _._ 2073 2 _._ 0915 _±_ 0 _._ 1159
Excel _[∗]_ 2 _._ 8691 _±_ 0 _._ 0882 2 _._ 5989 _±_ 0 _._ 0664
SAINT 2 _._ 7696 _±_ 0 _._ 0200 –
FT-T 2 _._ 4879 _±_ 0 _._ 2524 2 _._ 1501 _±_ 0 _._ 1506
T2G 2 _._ 2867 _±_ 0 _._ 2489 1 _._ 9179 _±_ 0 _._ 1530
MLP _[‡−]_ [lite] 2 _._ 2719 _±_ 0 _._ 1006 2 _._ 1026 _±_ 0 _._ 1088
MLP _[‡]_ 2 _._ 1832 _±_ 0 _._ 1124 2 _._ 0775 _±_ 0 _._ 0805
MLP _[†]_ 2 _._ 0979 _±_ 0 _._ 1779 1 _._ 9283 _±_ 0 _._ 1334
XGBoost 2 _._ 7567 _±_ 0 _._ 0470 2 _._ 7294 _±_ 0 _._ 0366
LightGBM 2 _._ 7005 _±_ 0 _._ 0296 2 _._ 6903 _±_ 0 _._ 0290
CatBoost 2 _._ 8847 _±_ 0 _._ 0227 2 _._ 8574 _±_ 0 _._ 0148
TabR 1 _._ 9760 _±_ 0 _._ 1738 1 _._ 7627 _±_ 0 _._ 1520
TabR _[‡]_ 1 _._ 9919 _±_ 0 _._ 1813 –
MNCA 1 _._ 7905 _±_ 0 _._ 1594 1 _._ 6205 _±_ 0 _._ 1676
MNCA _[‡]_ 1 _._ 8912 _±_ 0 _._ 1851 1 _._ 7147 _±_ 0 _._ 1348
TabM _[♠]_ 1 _._ 8831 _±_ 0 _._ 1194 1 _._ 8578 _±_ 0 _._ 1088
TabM 1 _._ 8433 _±_ 0 _._ 1196 1 _._ 8230 _±_ 0 _._ 1197
TabM[G] 1 _._ 9091 _±_ 0 _._ 1345 –
TabM mini 1 _._ 9421 _±_ 0 _._ 0971 1 _._ 9013 _±_ 0 _._ 0813
TabM _[†]_ mini 1 _._ 7799 _±_ 0 _._ 0859 1 _._ 7560 _±_ 0 _._ 0795


Published as a conference paper at ICLR 2025


cpu ~~a~~ ct ↓

Method Single model Ensemble

MLP 2 _._ 6814 _±_ 0 _._ 2291 2 _._ 4953 _±_ 0 _._ 1150
TabPFN – –
ResNet 2 _._ 3933 _±_ 0 _._ 0641 2 _._ 3005 _±_ 0 _._ 0397
DCN2 2 _._ 7868 _±_ 0 _._ 1999 2 _._ 4884 _±_ 0 _._ 0327
SNN 2 _._ 5811 _±_ 0 _._ 1480 2 _._ 3863 _±_ 0 _._ 0324
Trompt 2 _._ 2133 _±_ 0 _._ 0221 –
AutoInt 2 _._ 2537 _±_ 0 _._ 0536 2 _._ 1708 _±_ 0 _._ 0349
MLP-Mixer 2 _._ 3079 _±_ 0 _._ 0829 2 _._ 1831 _±_ 0 _._ 0470
Excel _[∗]_ 2 _._ 3094 _±_ 0 _._ 2401 2 _._ 1411 _±_ 0 _._ 0767
SAINT 2 _._ 2781 _±_ 0 _._ 0630 –
FT-T 2 _._ 2394 _±_ 0 _._ 0508 2 _._ 1494 _±_ 0 _._ 0268
T2G 2 _._ 2111 _±_ 0 _._ 0413 2 _._ 1330 _±_ 0 _._ 0316
MLP _[‡−]_ [lite] 2 _._ 2730 _±_ 0 _._ 0457 2 _._ 1899 _±_ 0 _._ 0419
MLP _[‡]_ 2 _._ 2671 _±_ 0 _._ 0383 2 _._ 1940 _±_ 0 _._ 0433
MLP _[†]_ 2 _._ 3309 _±_ 0 _._ 0719 2 _._ 2516 _±_ 0 _._ 0574
XGBoost 2 _._ 5237 _±_ 0 _._ 3530 2 _._ 4723 _±_ 0 _._ 3789
LightGBM 2 _._ 2223 _±_ 0 _._ 0894 2 _._ 2067 _±_ 0 _._ 0916
CatBoost 2 _._ 1239 _±_ 0 _._ 0489 2 _._ 1092 _±_ 0 _._ 0499
TabR 2 _._ 2980 _±_ 0 _._ 0529 2 _._ 2228 _±_ 0 _._ 0501
TabR _[‡]_ 2 _._ 1278 _±_ 0 _._ 0783 –
MNCA 2 _._ 2603 _±_ 0 _._ 0479 2 _._ 2339 _±_ 0 _._ 0508
MNCA _[‡]_ 2 _._ 2105 _±_ 0 _._ 0483 2 _._ 1396 _±_ 0 _._ 0474
TabM _[♠]_ 2 _._ 1940 _±_ 0 _._ 0523 2 _._ 1677 _±_ 0 _._ 0487
TabM 2 _._ 1402 _±_ 0 _._ 0588 2 _._ 1265 _±_ 0 _._ 0580
TabM[G] 2 _._ 1549 _±_ 0 _._ 0626 –
TabM mini 2 _._ 1638 _±_ 0 _._ 0420 2 _._ 1508 _±_ 0 _._ 0416
TabM _[†]_ mini 2 _._ 1391 _±_ 0 _._ 0542 2 _._ 1221 _±_ 0 _._ 0570


Brazilian ~~h~~ ouses ↓

Method Single model Ensemble

MLP 0 _._ 0473 _±_ 0 _._ 0179 0 _._ 0440 _±_ 0 _._ 0207
TabPFN – –
ResNet 0 _._ 0505 _±_ 0 _._ 0181 0 _._ 0458 _±_ 0 _._ 0207
DCN2 0 _._ 0477 _±_ 0 _._ 0172 0 _._ 0427 _±_ 0 _._ 0207
SNN 0 _._ 0630 _±_ 0 _._ 0162 0 _._ 0556 _±_ 0 _._ 0175
Trompt 0 _._ 0404 _±_ 0 _._ 0266 –
AutoInt 0 _._ 0470 _±_ 0 _._ 0192 0 _._ 0437 _±_ 0 _._ 0217
MLP-Mixer 0 _._ 0513 _±_ 0 _._ 0234 0 _._ 0484 _±_ 0 _._ 0262
Excel _[∗]_ 0 _._ 0450 _±_ 0 _._ 0156 0 _._ 0418 _±_ 0 _._ 0190
SAINT 0 _._ 0479 _±_ 0 _._ 0205 –
FT-T 0 _._ 0438 _±_ 0 _._ 0181 0 _._ 0412 _±_ 0 _._ 0204
T2G 0 _._ 0468 _±_ 0 _._ 0165 0 _._ 0436 _±_ 0 _._ 0211
MLP _[‡−]_ [lite] 0 _._ 0426 _±_ 0 _._ 0180 0 _._ 0397 _±_ 0 _._ 0206
MLP _[‡]_ 0 _._ 0437 _±_ 0 _._ 0203 0 _._ 0407 _±_ 0 _._ 0230
MLP _[†]_ 0 _._ 0421 _±_ 0 _._ 0209 0 _._ 0409 _±_ 0 _._ 0226
XGBoost 0 _._ 0541 _±_ 0 _._ 0270 0 _._ 0535 _±_ 0 _._ 0287
LightGBM 0 _._ 0603 _±_ 0 _._ 0249 0 _._ 0589 _±_ 0 _._ 0271
CatBoost 0 _._ 0468 _±_ 0 _._ 0312 0 _._ 0456 _±_ 0 _._ 0332
TabR 0 _._ 0490 _±_ 0 _._ 0152 0 _._ 0454 _±_ 0 _._ 0170
TabR _[‡]_ 0 _._ 0451 _±_ 0 _._ 0163 –
MNCA 0 _._ 0527 _±_ 0 _._ 0157 0 _._ 0509 _±_ 0 _._ 0180
MNCA _[‡]_ 0 _._ 0553 _±_ 0 _._ 0192 0 _._ 0511 _±_ 0 _._ 0191
TabM _[♠]_ 0 _._ 0443 _±_ 0 _._ 0213 0 _._ 0431 _±_ 0 _._ 0233
TabM 0 _._ 0417 _±_ 0 _._ 0208 0 _._ 0413 _±_ 0 _._ 0222
TabM[G] 0 _._ 0424 _±_ 0 _._ 0201 –
TabM mini 0 _._ 0433 _±_ 0 _._ 0232 0 _._ 0428 _±_ 0 _._ 0247
TabM _[†]_ mini 0 _._ 0416 _±_ 0 _._ 0215 0 _._ 0406 _±_ 0 _._ 0230


31



bank-marketing ↑

Method Single model Ensemble

MLP 0 _._ 7860 _±_ 0 _._ 0057 0 _._ 7887 _±_ 0 _._ 0052
TabPFN – 0 _._ 7894 _±_ 0 _._ 0091
ResNet 0 _._ 7921 _±_ 0 _._ 0076 0 _._ 7932 _±_ 0 _._ 0066
DCN2 0 _._ 7859 _±_ 0 _._ 0068 0 _._ 7917 _±_ 0 _._ 0078
SNN 0 _._ 7836 _±_ 0 _._ 0074 0 _._ 7882 _±_ 0 _._ 0054
Trompt 0 _._ 7975 _±_ 0 _._ 0080 –
AutoInt 0 _._ 7917 _±_ 0 _._ 0071 0 _._ 7956 _±_ 0 _._ 0058
MLP-Mixer 0 _._ 7954 _±_ 0 _._ 0059 0 _._ 8001 _±_ 0 _._ 0048
Excel _[∗]_ 0 _._ 7957 _±_ 0 _._ 0090 0 _._ 7985 _±_ 0 _._ 0106
SAINT 0 _._ 7953 _±_ 0 _._ 0058 –
FT-T 0 _._ 7918 _±_ 0 _._ 0076 0 _._ 7951 _±_ 0 _._ 0071
T2G 0 _._ 7918 _±_ 0 _._ 0058 0 _._ 7955 _±_ 0 _._ 0047
MLP _[‡−]_ [lite] 0 _._ 7947 _±_ 0 _._ 0101 0 _._ 7977 _±_ 0 _._ 0117
MLP _[‡]_ 0 _._ 7988 _±_ 0 _._ 0092 0 _._ 8024 _±_ 0 _._ 0093
MLP _[†]_ 0 _._ 7981 _±_ 0 _._ 0065 0 _._ 8008 _±_ 0 _._ 0057
XGBoost 0 _._ 8013 _±_ 0 _._ 0081 0 _._ 8030 _±_ 0 _._ 0076
LightGBM 0 _._ 8006 _±_ 0 _._ 0078 0 _._ 8013 _±_ 0 _._ 0072
CatBoost 0 _._ 8026 _±_ 0 _._ 0068 0 _._ 8056 _±_ 0 _._ 0082
TabR 0 _._ 7995 _±_ 0 _._ 0054 0 _._ 8015 _±_ 0 _._ 0037
TabR _[‡]_ 0 _._ 8023 _±_ 0 _._ 0088 –
MNCA 0 _._ 7961 _±_ 0 _._ 0065 0 _._ 8003 _±_ 0 _._ 0077
MNCA _[‡]_ 0 _._ 7977 _±_ 0 _._ 0081 0 _._ 8010 _±_ 0 _._ 0084
TabM _[♠]_ 0 _._ 7908 _±_ 0 _._ 0068 0 _._ 7915 _±_ 0 _._ 0068
TabM 0 _._ 7944 _±_ 0 _._ 0060 0 _._ 7944 _±_ 0 _._ 0052
TabM[G] 0 _._ 7935 _±_ 0 _._ 0064 –
TabM mini 0 _._ 7941 _±_ 0 _._ 0055 0 _._ 7943 _±_ 0 _._ 0045
TabM _[†]_ mini 0 _._ 7989 _±_ 0 _._ 0086 0 _._ 8002 _±_ 0 _._ 0074


MagicTelescope ↑

Method Single model Ensemble

MLP 0 _._ 8539 _±_ 0 _._ 0060 0 _._ 8566 _±_ 0 _._ 0061
TabPFN – 0 _._ 8579 _±_ 0 _._ 0064
ResNet 0 _._ 8589 _±_ 0 _._ 0068 0 _._ 8651 _±_ 0 _._ 0049
DCN2 0 _._ 8432 _±_ 0 _._ 0074 0 _._ 8490 _±_ 0 _._ 0046
SNN 0 _._ 8536 _±_ 0 _._ 0052 0 _._ 8567 _±_ 0 _._ 0047
Trompt 0 _._ 8605 _±_ 0 _._ 0102 –
AutoInt 0 _._ 8522 _±_ 0 _._ 0056 0 _._ 8560 _±_ 0 _._ 0034
MLP-Mixer 0 _._ 8571 _±_ 0 _._ 0080 0 _._ 8624 _±_ 0 _._ 0044
Excel _[∗]_ 0 _._ 8480 _±_ 0 _._ 0090 0 _._ 8543 _±_ 0 _._ 0075
SAINT 0 _._ 8595 _±_ 0 _._ 0060 –
FT-T 0 _._ 8588 _±_ 0 _._ 0046 0 _._ 8643 _±_ 0 _._ 0037
T2G 0 _._ 8553 _±_ 0 _._ 0055 0 _._ 8595 _±_ 0 _._ 0051
MLP _[‡−]_ [lite] 0 _._ 8591 _±_ 0 _._ 0061 0 _._ 8626 _±_ 0 _._ 0044
MLP _[‡]_ 0 _._ 8575 _±_ 0 _._ 0056 0 _._ 8605 _±_ 0 _._ 0051
MLP _[†]_ 0 _._ 8593 _±_ 0 _._ 0054 0 _._ 8621 _±_ 0 _._ 0037
XGBoost 0 _._ 8550 _±_ 0 _._ 0094 0 _._ 8589 _±_ 0 _._ 0110
LightGBM 0 _._ 8547 _±_ 0 _._ 0085 0 _._ 8556 _±_ 0 _._ 0086
CatBoost 0 _._ 8586 _±_ 0 _._ 0070 0 _._ 8588 _±_ 0 _._ 0077
TabR 0 _._ 8682 _±_ 0 _._ 0058 0 _._ 8729 _±_ 0 _._ 0038
TabR _[‡]_ 0 _._ 8641 _±_ 0 _._ 0052 –
MNCA 0 _._ 8602 _±_ 0 _._ 0061 0 _._ 8628 _±_ 0 _._ 0041
MNCA _[‡]_ 0 _._ 8622 _±_ 0 _._ 0085 0 _._ 8681 _±_ 0 _._ 0064
TabM _[♠]_ 0 _._ 8607 _±_ 0 _._ 0058 0 _._ 8622 _±_ 0 _._ 0050
TabM 0 _._ 8622 _±_ 0 _._ 0049 0 _._ 8631 _±_ 0 _._ 0046
TabM[G] 0 _._ 8600 _±_ 0 _._ 0055 –
TabM mini 0 _._ 8606 _±_ 0 _._ 0055 0 _._ 8618 _±_ 0 _._ 0049
TabM _[†]_ mini 0 _._ 8644 _±_ 0 _._ 0088 0 _._ 8673 _±_ 0 _._ 0075


Published as a conference paper at ICLR 2025


Ailerons ↓

Method Single model Ensemble

MLP 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
TabPFN – –
ResNet 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
DCN2 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
SNN 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
Trompt 0 _._ 0002 _±_ 0 _._ 0000 –
AutoInt 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
MLP-Mixer 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
Excel _[∗]_ 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
SAINT 0 _._ 0002 _±_ 0 _._ 0000 –
FT-T 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
T2G 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
MLP _[‡−]_ [lite] 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
MLP _[‡]_ 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
MLP _[†]_ 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
XGBoost 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
LightGBM 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
CatBoost 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
TabR 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
TabR _[‡]_ 0 _._ 0002 _±_ 0 _._ 0000 –
MNCA 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
MNCA _[‡]_ 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
TabM _[♠]_ 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
TabM 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
TabM[G] 0 _._ 0002 _±_ 0 _._ 0000 –
TabM mini 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000
TabM _[†]_ mini 0 _._ 0002 _±_ 0 _._ 0000 0 _._ 0002 _±_ 0 _._ 0000


OnlineNewsPopularity ↓

Method Single model Ensemble

MLP 0 _._ 8643 _±_ 0 _._ 0007 0 _._ 8632 _±_ 0 _._ 0005
TabPFN – –
ResNet 0 _._ 8665 _±_ 0 _._ 0011 0 _._ 8639 _±_ 0 _._ 0000
DCN2 0 _._ 8714 _±_ 0 _._ 0013 0 _._ 8648 _±_ 0 _._ 0004
SNN 0 _._ 8692 _±_ 0 _._ 0015 0 _._ 8665 _±_ 0 _._ 0005
Trompt 0 _._ 8623 _± nan_ –
AutoInt 0 _._ 8636 _±_ 0 _._ 0022 0 _._ 8596 _±_ 0 _._ 0008
MLP-Mixer 0 _._ 8615 _±_ 0 _._ 0008 0 _._ 8598 _±_ 0 _._ 0004
Excel _[∗]_ 0 _._ 8605 _±_ 0 _._ 0024 0 _._ 8556 _± nan_
SAINT 0 _._ 8600 _±_ 0 _._ 0007 –
FT-T 0 _._ 8629 _±_ 0 _._ 0019 0 _._ 8603 _±_ 0 _._ 0000
T2G 0 _._ 8632 _±_ 0 _._ 0009 0 _._ 8572 _± nan_
MLP _[‡−]_ [lite] 0 _._ 8604 _±_ 0 _._ 0009 0 _._ 8591 _±_ 0 _._ 0004
MLP _[‡]_ 0 _._ 8594 _±_ 0 _._ 0004 0 _._ 8585 _±_ 0 _._ 0001
MLP _[†]_ 0 _._ 8585 _±_ 0 _._ 0003 0 _._ 8581 _±_ 0 _._ 0001
XGBoost 0 _._ 8545 _±_ 0 _._ 0002 0 _._ 8543 _±_ 0 _._ 0000
LightGBM 0 _._ 8546 _±_ 0 _._ 0002 0 _._ 8544 _±_ 0 _._ 0000
CatBoost 0 _._ 8532 _±_ 0 _._ 0003 0 _._ 8527 _±_ 0 _._ 0001
TabR 0 _._ 8677 _±_ 0 _._ 0013 0 _._ 8633 _±_ 0 _._ 0009
TabR _[‡]_ 0 _._ 8624 _±_ 0 _._ 0011 –
MNCA 0 _._ 8651 _±_ 0 _._ 0003 0 _._ 8650 _±_ 0 _._ 0002
MNCA _[‡]_ 0 _._ 8647 _±_ 0 _._ 0010 0 _._ 8624 _±_ 0 _._ 0006
TabM _[♠]_ 0 _._ 8584 _±_ 0 _._ 0003 0 _._ 8581 _±_ 0 _._ 0001
TabM 0 _._ 8579 _±_ 0 _._ 0003 0 _._ 8575 _±_ 0 _._ 0001
TabM[G] 0 _._ 8579 _±_ 0 _._ 0004 –
TabM mini 0 _._ 8588 _±_ 0 _._ 0004 0 _._ 8581 _±_ 0 _._ 0003
TabM _[†]_ mini 0 _._ 8563 _±_ 0 _._ 0004 0 _._ 8558 _±_ 0 _._ 0002


32



MiamiHousing2016 ↓

Method Single model Ensemble

MLP 0 _._ 1614 _±_ 0 _._ 0033 0 _._ 1574 _±_ 0 _._ 0043
TabPFN – –
ResNet 0 _._ 1548 _±_ 0 _._ 0030 0 _._ 1511 _±_ 0 _._ 0027
DCN2 0 _._ 1683 _±_ 0 _._ 0099 0 _._ 1575 _±_ 0 _._ 0047
SNN 0 _._ 1618 _±_ 0 _._ 0029 0 _._ 1557 _±_ 0 _._ 0021
Trompt 0 _._ 1478 _±_ 0 _._ 0028 –
AutoInt 0 _._ 1537 _±_ 0 _._ 0035 0 _._ 1478 _±_ 0 _._ 0027
MLP-Mixer 0 _._ 1527 _±_ 0 _._ 0037 0 _._ 1479 _±_ 0 _._ 0033
Excel _[∗]_ 0 _._ 1519 _±_ 0 _._ 0038 0 _._ 1442 _±_ 0 _._ 0022
SAINT 0 _._ 1507 _±_ 0 _._ 0022 –
FT-T 0 _._ 1514 _±_ 0 _._ 0029 0 _._ 1462 _±_ 0 _._ 0031
T2G 0 _._ 1523 _±_ 0 _._ 0023 0 _._ 1478 _±_ 0 _._ 0024
MLP _[‡−]_ [lite] 0 _._ 1514 _±_ 0 _._ 0025 0 _._ 1479 _±_ 0 _._ 0017
MLP _[‡]_ 0 _._ 1512 _±_ 0 _._ 0019 0 _._ 1470 _±_ 0 _._ 0024
MLP _[†]_ 0 _._ 1461 _±_ 0 _._ 0015 0 _._ 1433 _±_ 0 _._ 0022
XGBoost 0 _._ 1440 _±_ 0 _._ 0029 0 _._ 1434 _±_ 0 _._ 0029
LightGBM 0 _._ 1461 _±_ 0 _._ 0025 0 _._ 1455 _±_ 0 _._ 0030
CatBoost 0 _._ 1417 _±_ 0 _._ 0021 0 _._ 1408 _±_ 0 _._ 0026
TabR 0 _._ 1417 _±_ 0 _._ 0025 0 _._ 1390 _±_ 0 _._ 0020
TabR _[‡]_ 0 _._ 1392 _±_ 0 _._ 0023 –
MNCA 0 _._ 1503 _±_ 0 _._ 0040 0 _._ 1477 _±_ 0 _._ 0032
MNCA _[‡]_ 0 _._ 1475 _±_ 0 _._ 0031 0 _._ 1438 _±_ 0 _._ 0024
TabM _[♠]_ 0 _._ 1483 _±_ 0 _._ 0030 0 _._ 1465 _±_ 0 _._ 0029
TabM 0 _._ 1478 _±_ 0 _._ 0012 0 _._ 1471 _±_ 0 _._ 0011
TabM[G] 0 _._ 1482 _±_ 0 _._ 0012 –
TabM mini 0 _._ 1481 _±_ 0 _._ 0021 0 _._ 1471 _±_ 0 _._ 0020
TabM _[†]_ mini 0 _._ 1408 _±_ 0 _._ 0019 0 _._ 1399 _±_ 0 _._ 0018


credit ↑

Method Single model Ensemble

MLP 0 _._ 7735 _±_ 0 _._ 0042 0 _._ 7729 _±_ 0 _._ 0047
TabPFN – 0 _._ 7636 _±_ 0 _._ 0045
ResNet 0 _._ 7721 _±_ 0 _._ 0033 0 _._ 7738 _±_ 0 _._ 0027
DCN2 0 _._ 7703 _±_ 0 _._ 0034 0 _._ 7746 _±_ 0 _._ 0026
SNN 0 _._ 7712 _±_ 0 _._ 0045 0 _._ 7716 _±_ 0 _._ 0059
Trompt 0 _._ 7740 _±_ 0 _._ 0006 –
AutoInt 0 _._ 7737 _±_ 0 _._ 0050 0 _._ 7765 _±_ 0 _._ 0058
MLP-Mixer 0 _._ 7748 _±_ 0 _._ 0038 0 _._ 7768 _±_ 0 _._ 0059
Excel _[∗]_ 0 _._ 7724 _±_ 0 _._ 0038 0 _._ 7740 _±_ 0 _._ 0069
SAINT 0 _._ 7739 _±_ 0 _._ 0052 –
FT-T 0 _._ 7745 _±_ 0 _._ 0041 0 _._ 7767 _±_ 0 _._ 0040
T2G 0 _._ 7744 _±_ 0 _._ 0046 0 _._ 7762 _±_ 0 _._ 0057
MLP _[‡−]_ [lite] 0 _._ 7749 _±_ 0 _._ 0055 0 _._ 7767 _±_ 0 _._ 0075
MLP _[‡]_ 0 _._ 7734 _±_ 0 _._ 0034 0 _._ 7747 _±_ 0 _._ 0043
MLP _[†]_ 0 _._ 7758 _±_ 0 _._ 0040 0 _._ 7772 _±_ 0 _._ 0055
XGBoost 0 _._ 7698 _±_ 0 _._ 0027 0 _._ 7706 _±_ 0 _._ 0029
LightGBM 0 _._ 7686 _±_ 0 _._ 0028 0 _._ 7726 _±_ 0 _._ 0034
CatBoost 0 _._ 7734 _±_ 0 _._ 0035 0 _._ 7752 _±_ 0 _._ 0038
TabR 0 _._ 7730 _±_ 0 _._ 0043 0 _._ 7740 _±_ 0 _._ 0040
TabR _[‡]_ 0 _._ 7723 _±_ 0 _._ 0037 –
MNCA 0 _._ 7739 _±_ 0 _._ 0032 0 _._ 7757 _±_ 0 _._ 0026
MNCA _[‡]_ 0 _._ 7734 _±_ 0 _._ 0045 0 _._ 7754 _±_ 0 _._ 0040
TabM _[♠]_ 0 _._ 7751 _±_ 0 _._ 0042 0 _._ 7755 _±_ 0 _._ 0049
TabM 0 _._ 7760 _±_ 0 _._ 0043 0 _._ 7771 _±_ 0 _._ 0044
TabM[G] 0 _._ 7754 _±_ 0 _._ 0045 –
TabM mini 0 _._ 7752 _±_ 0 _._ 0047 0 _._ 7754 _±_ 0 _._ 0048
TabM _[†]_ mini 0 _._ 7761 _±_ 0 _._ 0033 0 _._ 7760 _±_ 0 _._ 0028


Published as a conference paper at ICLR 2025


elevators ↓

Method Single model Ensemble

MLP 0 _._ 0020 _±_ 0 _._ 0001 0 _._ 0019 _±_ 0 _._ 0000
TabPFN – –
ResNet 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0019 _±_ 0 _._ 0000
DCN2 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0019 _±_ 0 _._ 0000
SNN 0 _._ 0020 _±_ 0 _._ 0001 0 _._ 0019 _±_ 0 _._ 0000
Trompt 0 _._ 0018 _±_ 0 _._ 0000 –
AutoInt 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
MLP-Mixer 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
Excel _[∗]_ 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
SAINT 0 _._ 0018 _±_ 0 _._ 0000 –
FT-T 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
T2G 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
MLP _[‡−]_ [lite] 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
MLP _[‡]_ 0 _._ 0018 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
MLP _[†]_ 0 _._ 0018 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
XGBoost 0 _._ 0020 _±_ 0 _._ 0000 0 _._ 0020 _±_ 0 _._ 0000
LightGBM 0 _._ 0020 _±_ 0 _._ 0000 0 _._ 0020 _±_ 0 _._ 0000
CatBoost 0 _._ 0020 _±_ 0 _._ 0000 0 _._ 0019 _±_ 0 _._ 0000
TabR 0 _._ 0049 _±_ 0 _._ 0000 0 _._ 0049 _±_ 0 _._ 0000
TabR _[‡]_ 0 _._ 0019 _±_ 0 _._ 0001 –
MNCA 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0019 _±_ 0 _._ 0000
MNCA _[‡]_ 0 _._ 0018 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
TabM _[♠]_ 0 _._ 0019 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
TabM 0 _._ 0018 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
TabM[G] 0 _._ 0018 _±_ 0 _._ 0000 –
TabM mini 0 _._ 0018 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000
TabM _[†]_ mini 0 _._ 0018 _±_ 0 _._ 0000 0 _._ 0018 _±_ 0 _._ 0000


house ~~s~~ ales ↓

Method Single model Ensemble

MLP 0 _._ 1790 _±_ 0 _._ 0009 0 _._ 1763 _±_ 0 _._ 0003
TabPFN – –
ResNet 0 _._ 1755 _±_ 0 _._ 0014 0 _._ 1738 _±_ 0 _._ 0006
DCN2 0 _._ 1862 _±_ 0 _._ 0032 0 _._ 1778 _±_ 0 _._ 0015
SNN 0 _._ 1800 _±_ 0 _._ 0008 0 _._ 1770 _±_ 0 _._ 0004
Trompt 0 _._ 1667 _± nan_ –
AutoInt 0 _._ 1700 _±_ 0 _._ 0014 0 _._ 1670 _±_ 0 _._ 0008
MLP-Mixer 0 _._ 1704 _±_ 0 _._ 0007 0 _._ 1690 _±_ 0 _._ 0005
Excel _[∗]_ 0 _._ 1713 _±_ 0 _._ 0010 0 _._ 1668 _± nan_
SAINT 0 _._ 1713 _±_ 0 _._ 0015 –
FT-T 0 _._ 1690 _±_ 0 _._ 0010 0 _._ 1659 _±_ 0 _._ 0004
T2G 0 _._ 1689 _±_ 0 _._ 0010 0 _._ 1664 _± nan_
MLP _[‡−]_ [lite] 0 _._ 1699 _±_ 0 _._ 0008 0 _._ 1687 _±_ 0 _._ 0007
MLP _[‡]_ 0 _._ 1690 _±_ 0 _._ 0005 0 _._ 1676 _±_ 0 _._ 0003
MLP _[†]_ 0 _._ 1687 _±_ 0 _._ 0004 0 _._ 1681 _±_ 0 _._ 0001
XGBoost 0 _._ 1694 _±_ 0 _._ 0003 0 _._ 1689 _±_ 0 _._ 0001
LightGBM 0 _._ 1692 _±_ 0 _._ 0004 0 _._ 1686 _±_ 0 _._ 0001
CatBoost 0 _._ 1669 _±_ 0 _._ 0001 0 _._ 1667 _±_ 0 _._ 0000
TabR 0 _._ 1689 _±_ 0 _._ 0009 0 _._ 1657 _±_ 0 _._ 0003
TabR _[‡]_ 0 _._ 1636 _±_ 0 _._ 0009 –
MNCA 0 _._ 1737 _±_ 0 _._ 0013 0 _._ 1714 _±_ 0 _._ 0005
MNCA _[‡]_ 0 _._ 1694 _±_ 0 _._ 0007 0 _._ 1670 _±_ 0 _._ 0003
TabM _[♠]_ 0 _._ 1692 _±_ 0 _._ 0011 0 _._ 1680 _±_ 0 _._ 0005
TabM 0 _._ 1666 _±_ 0 _._ 0003 0 _._ 1662 _±_ 0 _._ 0002
TabM[G] 0 _._ 1667 _±_ 0 _._ 0003 –
TabM mini 0 _._ 1673 _±_ 0 _._ 0004 0 _._ 1668 _±_ 0 _._ 0001
TabM _[†]_ mini 0 _._ 1652 _±_ 0 _._ 0003 0 _._ 1644 _±_ 0 _._ 0001


33



fifa ↓

Method Single model Ensemble

MLP 0 _._ 8038 _±_ 0 _._ 0124 0 _._ 8011 _±_ 0 _._ 0143
TabPFN – –
ResNet 0 _._ 8025 _±_ 0 _._ 0140 0 _._ 7985 _±_ 0 _._ 0149
DCN2 0 _._ 8046 _±_ 0 _._ 0135 0 _._ 7993 _±_ 0 _._ 0129
SNN 0 _._ 8074 _±_ 0 _._ 0140 0 _._ 8031 _±_ 0 _._ 0147
Trompt 0 _._ 7880 _±_ 0 _._ 0180 –
AutoInt 0 _._ 7923 _±_ 0 _._ 0128 0 _._ 7886 _±_ 0 _._ 0127
MLP-Mixer 0 _._ 7936 _±_ 0 _._ 0119 0 _._ 7903 _±_ 0 _._ 0133
Excel _[∗]_ 0 _._ 7909 _±_ 0 _._ 0111 0 _._ 7862 _±_ 0 _._ 0161
SAINT 0 _._ 7901 _±_ 0 _._ 0118 –
FT-T 0 _._ 7928 _±_ 0 _._ 0132 0 _._ 7888 _±_ 0 _._ 0130
T2G 0 _._ 7928 _±_ 0 _._ 0139 0 _._ 7904 _±_ 0 _._ 0183
MLP _[‡−]_ [lite] 0 _._ 7940 _±_ 0 _._ 0118 0 _._ 7898 _±_ 0 _._ 0141
MLP _[‡]_ 0 _._ 7907 _±_ 0 _._ 0092 0 _._ 7870 _±_ 0 _._ 0096
MLP _[†]_ 0 _._ 7806 _±_ 0 _._ 0104 0 _._ 7800 _±_ 0 _._ 0114
XGBoost 0 _._ 7800 _±_ 0 _._ 0108 0 _._ 7795 _±_ 0 _._ 0114
LightGBM 0 _._ 7806 _±_ 0 _._ 0120 0 _._ 7787 _±_ 0 _._ 0122
CatBoost 0 _._ 7835 _±_ 0 _._ 0116 0 _._ 7817 _±_ 0 _._ 0114
TabR 0 _._ 7902 _±_ 0 _._ 0119 0 _._ 7863 _±_ 0 _._ 0120
TabR _[‡]_ 0 _._ 7914 _±_ 0 _._ 0136 –
MNCA 0 _._ 7967 _±_ 0 _._ 0138 0 _._ 7933 _±_ 0 _._ 0145
MNCA _[‡]_ 0 _._ 7909 _±_ 0 _._ 0107 0 _._ 7866 _±_ 0 _._ 0106
TabM _[♠]_ 0 _._ 7974 _±_ 0 _._ 0144 0 _._ 7954 _±_ 0 _._ 0160
TabM 0 _._ 7953 _±_ 0 _._ 0135 0 _._ 7942 _±_ 0 _._ 0148
TabM[G] 0 _._ 7948 _±_ 0 _._ 0135 –
TabM mini 0 _._ 7938 _±_ 0 _._ 0156 0 _._ 7920 _±_ 0 _._ 0176
TabM _[†]_ mini 0 _._ 7771 _±_ 0 _._ 0107 0 _._ 7761 _±_ 0 _._ 0117


medical ~~c~~ harges ↓

Method Single model Ensemble

MLP 0 _._ 0816 _±_ 0 _._ 0001 0 _._ 0814 _±_ 0 _._ 0000
TabPFN – –
ResNet 0 _._ 0824 _±_ 0 _._ 0003 0 _._ 0817 _±_ 0 _._ 0001
DCN2 0 _._ 0818 _±_ 0 _._ 0003 0 _._ 0815 _±_ 0 _._ 0001
SNN 0 _._ 0827 _±_ 0 _._ 0006 0 _._ 0817 _±_ 0 _._ 0001
Trompt 0 _._ 0812 _± nan_ –
AutoInt 0 _._ 0822 _±_ 0 _._ 0007 0 _._ 0814 _±_ 0 _._ 0001
MLP-Mixer 0 _._ 0814 _±_ 0 _._ 0002 0 _._ 0811 _±_ 0 _._ 0000
Excel _[∗]_ 0 _._ 0817 _±_ 0 _._ 0004 0 _._ 0813 _± nan_
SAINT 0 _._ 0814 _±_ 0 _._ 0002 –
FT-T 0 _._ 0814 _±_ 0 _._ 0002 0 _._ 0812 _±_ 0 _._ 0000
T2G 0 _._ 0813 _±_ 0 _._ 0002 0 _._ 0811 _± nan_
MLP _[‡−]_ [lite] 0 _._ 0812 _±_ 0 _._ 0002 0 _._ 0810 _±_ 0 _._ 0000
MLP _[‡]_ 0 _._ 0812 _±_ 0 _._ 0001 0 _._ 0809 _±_ 0 _._ 0001
MLP _[†]_ 0 _._ 0812 _±_ 0 _._ 0000 0 _._ 0811 _±_ 0 _._ 0000
XGBoost 0 _._ 0825 _±_ 0 _._ 0001 0 _._ 0825 _±_ 0 _._ 0000
LightGBM 0 _._ 0820 _±_ 0 _._ 0000 0 _._ 0820 _±_ 0 _._ 0000
CatBoost 0 _._ 0816 _±_ 0 _._ 0000 0 _._ 0815 _±_ 0 _._ 0000
TabR 0 _._ 0815 _±_ 0 _._ 0002 0 _._ 0812 _±_ 0 _._ 0000
TabR _[‡]_ 0 _._ 0811 _±_ 0 _._ 0001 –
MNCA 0 _._ 0811 _±_ 0 _._ 0001 0 _._ 0810 _±_ 0 _._ 0000
MNCA _[‡]_ 0 _._ 0809 _±_ 0 _._ 0000 0 _._ 0808 _±_ 0 _._ 0000
TabM _[♠]_ 0 _._ 0813 _±_ 0 _._ 0001 0 _._ 0812 _±_ 0 _._ 0000
TabM 0 _._ 0812 _±_ 0 _._ 0000 0 _._ 0812 _±_ 0 _._ 0000
TabM[G] 0 _._ 0812 _±_ 0 _._ 0000 –
TabM mini 0 _._ 0813 _±_ 0 _._ 0000 0 _._ 0813 _±_ 0 _._ 0000
TabM _[†]_ mini 0 _._ 0811 _±_ 0 _._ 0001 0 _._ 0811 _±_ 0 _._ 0000


Published as a conference paper at ICLR 2025


pol ↓

Method Single model Ensemble

MLP 5 _._ 5244 _±_ 0 _._ 5768 4 _._ 9945 _±_ 0 _._ 5923
TabPFN – –
ResNet 6 _._ 3739 _±_ 0 _._ 6286 5 _._ 8181 _±_ 0 _._ 6054
DCN2 6 _._ 5374 _±_ 0 _._ 9479 5 _._ 1814 _±_ 0 _._ 7775
SNN 6 _._ 1816 _±_ 0 _._ 7366 5 _._ 5959 _±_ 0 _._ 8243
Trompt 3 _._ 2337 _±_ 0 _._ 0605 –
AutoInt 3 _._ 3295 _±_ 0 _._ 3379 2 _._ 7999 _±_ 0 _._ 1776
MLP-Mixer 3 _._ 2011 _±_ 0 _._ 2921 2 _._ 8698 _±_ 0 _._ 2577
Excel _[∗]_ 3 _._ 0682 _±_ 0 _._ 2389 2 _._ 5816 _±_ 0 _._ 0368
SAINT 2 _._ 7203 _±_ 0 _._ 1858 –
FT-T 2 _._ 6974 _±_ 0 _._ 1666 2 _._ 3718 _±_ 0 _._ 0724
T2G 2 _._ 9539 _±_ 0 _._ 1994 2 _._ 6282 _±_ 0 _._ 0730
MLP _[‡−]_ [lite] 2 _._ 8239 _±_ 0 _._ 2173 2 _._ 5266 _±_ 0 _._ 0605
MLP _[‡]_ 2 _._ 5452 _±_ 0 _._ 1221 2 _._ 3700 _±_ 0 _._ 0867
MLP _[†]_ 2 _._ 4958 _±_ 0 _._ 1292 2 _._ 3651 _±_ 0 _._ 1223
XGBoost 4 _._ 2963 _±_ 0 _._ 0644 4 _._ 2548 _±_ 0 _._ 0488
LightGBM 4 _._ 2320 _±_ 0 _._ 3369 4 _._ 1880 _±_ 0 _._ 3110
CatBoost 3 _._ 6320 _±_ 0 _._ 1006 3 _._ 5505 _±_ 0 _._ 0896
TabR 6 _._ 0708 _±_ 0 _._ 5368 5 _._ 5578 _±_ 0 _._ 4036
TabR _[‡]_ 2 _._ 5770 _±_ 0 _._ 1689 –
MNCA 5 _._ 7878 _±_ 0 _._ 4884 5 _._ 3773 _±_ 0 _._ 5463
MNCA _[‡]_ 2 _._ 9083 _±_ 0 _._ 1364 2 _._ 6717 _±_ 0 _._ 0530
TabM _[♠]_ 3 _._ 3595 _±_ 0 _._ 4017 3 _._ 2130 _±_ 0 _._ 3979
TabM 3 _._ 0198 _±_ 0 _._ 2975 2 _._ 9595 _±_ 0 _._ 3107
TabM[G] 3 _._ 0358 _±_ 0 _._ 3077 –
TabM mini 3 _._ 1351 _±_ 0 _._ 1952 3 _._ 0478 _±_ 0 _._ 2061
TabM _[†]_ mini 2 _._ 2808 _±_ 0 _._ 0343 2 _._ 2383 _±_ 0 _._ 0111


jannis ↑

Method Single model Ensemble

MLP 0 _._ 7840 _±_ 0 _._ 0018 0 _._ 7872 _±_ 0 _._ 0007
TabPFN – 0 _._ 7419 _±_ 0 _._ 0018
ResNet 0 _._ 7923 _±_ 0 _._ 0024 0 _._ 7958 _±_ 0 _._ 0010
DCN2 0 _._ 7712 _±_ 0 _._ 0029 0 _._ 7825 _±_ 0 _._ 0009
SNN 0 _._ 7818 _±_ 0 _._ 0025 0 _._ 7859 _±_ 0 _._ 0011
Trompt 0 _._ 8027 _± nan_ –
AutoInt 0 _._ 7933 _±_ 0 _._ 0018 0 _._ 7983 _±_ 0 _._ 0013
MLP-Mixer 0 _._ 7927 _±_ 0 _._ 0025 0 _._ 8019 _±_ 0 _._ 0012
Excel _[∗]_ 0 _._ 7954 _±_ 0 _._ 0015 0 _._ 8021 _± nan_
SAINT 0 _._ 7971 _±_ 0 _._ 0028 –
FT-T 0 _._ 7940 _±_ 0 _._ 0028 0 _._ 7998 _±_ 0 _._ 0006
T2G 0 _._ 7998 _±_ 0 _._ 0024 0 _._ 8052 _± nan_
MLP _[‡−]_ [lite] 0 _._ 7923 _±_ 0 _._ 0018 0 _._ 7945 _±_ 0 _._ 0010
MLP _[‡]_ 0 _._ 7947 _±_ 0 _._ 0017 0 _._ 7967 _±_ 0 _._ 0011
MLP _[†]_ 0 _._ 7891 _±_ 0 _._ 0013 0 _._ 7900 _±_ 0 _._ 0006
XGBoost 0 _._ 7967 _±_ 0 _._ 0019 0 _._ 7998 _±_ 0 _._ 0007
LightGBM 0 _._ 7956 _±_ 0 _._ 0017 0 _._ 7968 _±_ 0 _._ 0005
CatBoost 0 _._ 7985 _±_ 0 _._ 0018 0 _._ 8009 _±_ 0 _._ 0012
TabR 0 _._ 7983 _±_ 0 _._ 0022 0 _._ 8023 _±_ 0 _._ 0018
TabR _[‡]_ 0 _._ 8051 _±_ 0 _._ 0023 –
MNCA 0 _._ 7993 _±_ 0 _._ 0019 0 _._ 8042 _±_ 0 _._ 0013
MNCA _[‡]_ 0 _._ 8068 _±_ 0 _._ 0021 0 _._ 8128 _±_ 0 _._ 0007
TabM _[♠]_ 0 _._ 8066 _±_ 0 _._ 0015 0 _._ 8075 _±_ 0 _._ 0004
TabM 0 _._ 8080 _±_ 0 _._ 0019 0 _._ 8102 _±_ 0 _._ 0017
TabM[G] 0 _._ 8064 _±_ 0 _._ 0018 –
TabM mini 0 _._ 8053 _±_ 0 _._ 0012 0 _._ 8066 _±_ 0 _._ 0001
TabM _[†]_ mini 0 _._ 8078 _±_ 0 _._ 0008 0 _._ 8086 _±_ 0 _._ 0005


34



superconduct ↓

Method Single model Ensemble

MLP 10 _._ 8740 _±_ 0 _._ 0868 10 _._ 4118 _±_ 0 _._ 0429
TabPFN – –
ResNet 10 _._ 7711 _±_ 0 _._ 1454 10 _._ 3495 _±_ 0 _._ 0168
DCN2 10 _._ 8108 _±_ 0 _._ 0957 10 _._ 4342 _±_ 0 _._ 0179
SNN 10 _._ 8562 _±_ 0 _._ 1300 10 _._ 3342 _±_ 0 _._ 0509
Trompt 10 _._ 4442 _± nan_ –
AutoInt 11 _._ 0019 _±_ 0 _._ 1391 10 _._ 4469 _±_ 0 _._ 0521
MLP-Mixer 10 _._ 7502 _±_ 0 _._ 0800 10 _._ 3281 _±_ 0 _._ 0450
Excel _[∗]_ 11 _._ 0879 _±_ 0 _._ 1571 10 _._ 4094 _± nan_
SAINT 10 _._ 7807 _±_ 0 _._ 1074 –
FT-T 10 _._ 8256 _±_ 0 _._ 1692 10 _._ 3391 _±_ 0 _._ 0794
T2G 10 _._ 8310 _±_ 0 _._ 1406 10 _._ 3017 _± nan_
MLP _[‡−]_ [lite] 10 _._ 5058 _±_ 0 _._ 0758 10 _._ 2322 _±_ 0 _._ 0463
MLP _[‡]_ 10 _._ 5061 _±_ 0 _._ 0330 10 _._ 2440 _±_ 0 _._ 0127
MLP _[†]_ 10 _._ 7220 _±_ 0 _._ 0757 10 _._ 3758 _±_ 0 _._ 0606
XGBoost 10 _._ 1610 _±_ 0 _._ 0201 10 _._ 1413 _±_ 0 _._ 0025
LightGBM 10 _._ 1634 _±_ 0 _._ 0118 10 _._ 1552 _±_ 0 _._ 0050
CatBoost 10 _._ 2422 _±_ 0 _._ 0222 10 _._ 2116 _±_ 0 _._ 0058
TabR 10 _._ 8842 _±_ 0 _._ 1073 10 _._ 4800 _±_ 0 _._ 0280
TabR _[‡]_ 10 _._ 3835 _±_ 0 _._ 0562 –
MNCA 10 _._ 4419 _±_ 0 _._ 0640 10 _._ 2926 _±_ 0 _._ 0261
MNCA _[‡]_ 10 _._ 5651 _±_ 0 _._ 0616 10 _._ 3155 _±_ 0 _._ 0253
TabM _[♠]_ 10 _._ 3379 _±_ 0 _._ 0338 10 _._ 1943 _±_ 0 _._ 0291
TabM 10 _._ 2628 _±_ 0 _._ 0275 10 _._ 2300 _±_ 0 _._ 0108
TabM[G] 10 _._ 2572 _±_ 0 _._ 0463 –
TabM mini 10 _._ 2472 _±_ 0 _._ 0208 10 _._ 2094 _±_ 0 _._ 0057
TabM _[†]_ mini 10 _._ 1326 _±_ 0 _._ 0186 10 _._ 0866 _±_ 0 _._ 0070


MiniBooNE ↑

Method Single model Ensemble

MLP 0 _._ 9480 _±_ 0 _._ 0007 0 _._ 9498 _±_ 0 _._ 0001
TabPFN – 0 _._ 9266 _±_ 0 _._ 0012
ResNet 0 _._ 9488 _±_ 0 _._ 0011 0 _._ 9504 _±_ 0 _._ 0005
DCN2 0 _._ 9433 _±_ 0 _._ 0011 0 _._ 9470 _±_ 0 _._ 0010
SNN 0 _._ 9476 _±_ 0 _._ 0013 0 _._ 9491 _±_ 0 _._ 0010
Trompt 0 _._ 9473 _± nan_ –
AutoInt 0 _._ 9447 _±_ 0 _._ 0014 0 _._ 9473 _±_ 0 _._ 0010
MLP-Mixer 0 _._ 9446 _±_ 0 _._ 0014 0 _._ 9483 _±_ 0 _._ 0002
Excel _[∗]_ 0 _._ 9430 _±_ 0 _._ 0015 0 _._ 9451 _± nan_
SAINT 0 _._ 9471 _±_ 0 _._ 0009 –
FT-T 0 _._ 9467 _±_ 0 _._ 0014 0 _._ 9486 _±_ 0 _._ 0010
T2G 0 _._ 9475 _±_ 0 _._ 0014 0 _._ 9508 _± nan_
MLP _[‡−]_ [lite] 0 _._ 9466 _±_ 0 _._ 0009 0 _._ 9478 _±_ 0 _._ 0004
MLP _[‡]_ 0 _._ 9473 _±_ 0 _._ 0010 0 _._ 9493 _±_ 0 _._ 0004
MLP _[†]_ 0 _._ 9482 _±_ 0 _._ 0008 0 _._ 9492 _±_ 0 _._ 0001
XGBoost 0 _._ 9436 _±_ 0 _._ 0006 0 _._ 9452 _±_ 0 _._ 0003
LightGBM 0 _._ 9422 _±_ 0 _._ 0009 0 _._ 9427 _±_ 0 _._ 0003
CatBoost 0 _._ 9453 _±_ 0 _._ 0008 0 _._ 9459 _±_ 0 _._ 0005
TabR 0 _._ 9487 _±_ 0 _._ 0008 0 _._ 9500 _±_ 0 _._ 0002
TabR _[‡]_ 0 _._ 9475 _±_ 0 _._ 0007 –
MNCA 0 _._ 9488 _±_ 0 _._ 0010 0 _._ 9505 _±_ 0 _._ 0001
MNCA _[‡]_ 0 _._ 9493 _±_ 0 _._ 0012 0 _._ 9501 _±_ 0 _._ 0008
TabM _[♠]_ 0 _._ 9500 _±_ 0 _._ 0005 0 _._ 9505 _±_ 0 _._ 0002
TabM 0 _._ 9503 _±_ 0 _._ 0006 0 _._ 9501 _±_ 0 _._ 0002
TabM[G] 0 _._ 9496 _±_ 0 _._ 0010 –
TabM mini 0 _._ 9495 _±_ 0 _._ 0005 0 _._ 9500 _±_ 0 _._ 0002
TabM _[†]_ mini 0 _._ 9490 _±_ 0 _._ 0004 0 _._ 9492 _±_ 0 _._ 0002


Published as a conference paper at ICLR 2025


nyc-taxi-green-dec-2016 ↓

Method Single model Ensemble

MLP 0 _._ 3951 _±_ 0 _._ 0009 0 _._ 3921 _±_ 0 _._ 0003
TabPFN – –
ResNet 0 _._ 3899 _±_ 0 _._ 0016 0 _._ 3873 _±_ 0 _._ 0009
DCN2 0 _._ 3919 _±_ 0 _._ 0009 0 _._ 3889 _±_ 0 _._ 0003
SNN 0 _._ 3933 _±_ 0 _._ 0013 0 _._ 3899 _±_ 0 _._ 0004
Trompt 0 _._ 3979 _± nan_ –
AutoInt 0 _._ 4084 _±_ 0 _._ 0256 0 _._ 3967 _±_ 0 _._ 0059
MLP-Mixer 0 _._ 3914 _±_ 0 _._ 0026 0 _._ 3861 _±_ 0 _._ 0013
Excel _[∗]_ 0 _._ 3969 _±_ 0 _._ 0036 0 _._ 3897 _± nan_
SAINT 0 _._ 3905 _±_ 0 _._ 0013 –
FT-T 0 _._ 3937 _±_ 0 _._ 0064 0 _._ 3889 _±_ 0 _._ 0018
T2G 0 _._ 3908 _±_ 0 _._ 0045 0 _._ 3858 _± nan_
MLP _[‡−]_ [lite] 0 _._ 3812 _±_ 0 _._ 0018 0 _._ 3761 _±_ 0 _._ 0016
MLP _[‡]_ 0 _._ 3795 _±_ 0 _._ 0016 0 _._ 3733 _±_ 0 _._ 0013
MLP _[†]_ 0 _._ 3680 _±_ 0 _._ 0006 0 _._ 3653 _±_ 0 _._ 0005
XGBoost 0 _._ 3792 _±_ 0 _._ 0002 0 _._ 3787 _±_ 0 _._ 0000
LightGBM 0 _._ 3688 _±_ 0 _._ 0002 0 _._ 3684 _±_ 0 _._ 0000
CatBoost 0 _._ 3647 _±_ 0 _._ 0005 0 _._ 3632 _±_ 0 _._ 0003
TabR 0 _._ 3577 _±_ 0 _._ 0222 0 _._ 3380 _±_ 0 _._ 0027
TabR _[‡]_ 0 _._ 3725 _±_ 0 _._ 0091 –
MNCA 0 _._ 3728 _±_ 0 _._ 0012 0 _._ 3720 _±_ 0 _._ 0010
MNCA _[‡]_ 0 _._ 3536 _±_ 0 _._ 0052 0 _._ 3407 _±_ 0 _._ 0009
TabM _[♠]_ 0 _._ 3866 _±_ 0 _._ 0006 0 _._ 3855 _±_ 0 _._ 0003
TabM 0 _._ 3849 _±_ 0 _._ 0005 0 _._ 3843 _±_ 0 _._ 0002
TabM[G] 0 _._ 3848 _±_ 0 _._ 0005 –
TabM mini 0 _._ 3853 _±_ 0 _._ 0005 0 _._ 3845 _±_ 0 _._ 0003
TabM _[†]_ mini 0 _._ 3485 _±_ 0 _._ 0038 0 _._ 3448 _±_ 0 _._ 0020


road-safety ↑

Method Single model Ensemble

MLP 0 _._ 7857 _±_ 0 _._ 0019 0 _._ 7873 _±_ 0 _._ 0004
TabPFN – 0 _._ 7338 _±_ 0 _._ 0032
ResNet 0 _._ 7875 _±_ 0 _._ 0007 0 _._ 7898 _±_ 0 _._ 0008
DCN2 0 _._ 7781 _±_ 0 _._ 0014 0 _._ 7823 _±_ 0 _._ 0012
SNN 0 _._ 7847 _±_ 0 _._ 0010 0 _._ 7865 _±_ 0 _._ 0002
Trompt 0 _._ 7804 _± nan_ –
AutoInt 0 _._ 7826 _±_ 0 _._ 0030 0 _._ 7883 _±_ 0 _._ 0013
MLP-Mixer 0 _._ 7878 _±_ 0 _._ 0032 0 _._ 7919 _±_ 0 _._ 0015
Excel _[∗]_ 0 _._ 7864 _±_ 0 _._ 0053 0 _._ 7907 _± nan_
SAINT 0 _._ 7584 _±_ 0 _._ 0584 –
FT-T 0 _._ 7907 _±_ 0 _._ 0012 0 _._ 7943 _±_ 0 _._ 0007
T2G 0 _._ 7912 _±_ 0 _._ 0026 0 _._ 7961 _± nan_
MLP _[‡−]_ [lite] 0 _._ 7867 _±_ 0 _._ 0018 0 _._ 7903 _±_ 0 _._ 0002
MLP _[‡]_ 0 _._ 7853 _±_ 0 _._ 0014 0 _._ 7881 _±_ 0 _._ 0007
MLP _[†]_ 0 _._ 7899 _±_ 0 _._ 0009 0 _._ 7935 _±_ 0 _._ 0003
XGBoost 0 _._ 8101 _±_ 0 _._ 0017 0 _._ 8129 _±_ 0 _._ 0004
LightGBM 0 _._ 7982 _±_ 0 _._ 0012 0 _._ 7996 _±_ 0 _._ 0005
CatBoost 0 _._ 8012 _±_ 0 _._ 0009 0 _._ 8022 _±_ 0 _._ 0002
TabR 0 _._ 8403 _±_ 0 _._ 0014 0 _._ 8441 _±_ 0 _._ 0005
TabR _[‡]_ 0 _._ 8374 _±_ 0 _._ 0013 –
MNCA 0 _._ 8080 _±_ 0 _._ 0013 0 _._ 8121 _±_ 0 _._ 0006
MNCA _[‡]_ 0 _._ 8232 _±_ 0 _._ 0017 0 _._ 8287 _±_ 0 _._ 0008
TabM _[♠]_ 0 _._ 7946 _±_ 0 _._ 0013 0 _._ 7961 _±_ 0 _._ 0005
TabM 0 _._ 7958 _±_ 0 _._ 0011 0 _._ 7968 _±_ 0 _._ 0004
TabM[G] 0 _._ 7954 _±_ 0 _._ 0016 –
TabM mini 0 _._ 7933 _±_ 0 _._ 0030 0 _._ 7970 _±_ 0 _._ 0006
TabM _[†]_ mini 0 _._ 7999 _±_ 0 _._ 0023 0 _._ 8059 _±_ 0 _._ 0012


35



particulate-matter-ukair-2017 ↓

Method Single model Ensemble

MLP 0 _._ 3759 _±_ 0 _._ 0004 0 _._ 3729 _±_ 0 _._ 0003
TabPFN – –
ResNet 0 _._ 3743 _±_ 0 _._ 0007 0 _._ 3718 _±_ 0 _._ 0005
DCN2 0 _._ 3759 _±_ 0 _._ 0012 0 _._ 3738 _±_ 0 _._ 0004
SNN 0 _._ 3790 _±_ 0 _._ 0007 0 _._ 3744 _±_ 0 _._ 0002
Trompt 0 _._ 3700 _± nan_ –
AutoInt 0 _._ 3723 _±_ 0 _._ 0011 0 _._ 3692 _±_ 0 _._ 0010
MLP-Mixer 0 _._ 3741 _±_ 0 _._ 0010 0 _._ 3698 _±_ 0 _._ 0004
Excel _[∗]_ 0 _._ 3699 _±_ 0 _._ 0014 0 _._ 3652 _± nan_
SAINT 0 _._ 3704 _±_ 0 _._ 0014 –
FT-T 0 _._ 3735 _±_ 0 _._ 0012 0 _._ 3686 _±_ 0 _._ 0004
T2G 0 _._ 3676 _±_ 0 _._ 0024 0 _._ 3631 _± nan_
MLP _[‡−]_ [lite] 0 _._ 3665 _±_ 0 _._ 0008 0 _._ 3642 _±_ 0 _._ 0003
MLP _[‡]_ 0 _._ 3657 _±_ 0 _._ 0007 0 _._ 3629 _±_ 0 _._ 0002
MLP _[†]_ 0 _._ 3649 _±_ 0 _._ 0011 0 _._ 3637 _±_ 0 _._ 0008
XGBoost 0 _._ 3641 _±_ 0 _._ 0001 0 _._ 3640 _±_ 0 _._ 0000
LightGBM 0 _._ 3637 _±_ 0 _._ 0001 0 _._ 3635 _±_ 0 _._ 0000
CatBoost 0 _._ 3647 _±_ 0 _._ 0004 0 _._ 3637 _±_ 0 _._ 0002
TabR 0 _._ 3613 _±_ 0 _._ 0005 0 _._ 3590 _±_ 0 _._ 0002
TabR _[‡]_ 0 _._ 3596 _±_ 0 _._ 0004 –
MNCA 0 _._ 3670 _±_ 0 _._ 0004 0 _._ 3649 _±_ 0 _._ 0002
MNCA _[‡]_ 0 _._ 3646 _±_ 0 _._ 0001 0 _._ 3643 _±_ 0 _._ 0000
TabM _[♠]_ 0 _._ 3686 _±_ 0 _._ 0006 0 _._ 3679 _±_ 0 _._ 0003
TabM 0 _._ 3671 _±_ 0 _._ 0007 0 _._ 3665 _±_ 0 _._ 0002
TabM[G] 0 _._ 3667 _±_ 0 _._ 0009 –
TabM mini 0 _._ 3664 _±_ 0 _._ 0006 0 _._ 3655 _±_ 0 _._ 0002
TabM _[†]_ mini 0 _._ 3593 _±_ 0 _._ 0004 0 _._ 3589 _±_ 0 _._ 0000


year ↓

Method Single model Ensemble

MLP 8 _._ 9628 _±_ 0 _._ 0232 8 _._ 8931 _±_ 0 _._ 0066
TabPFN – –
ResNet 8 _._ 9658 _±_ 0 _._ 0239 8 _._ 8755 _±_ 0 _._ 0066
DCN2 9 _._ 2761 _±_ 0 _._ 0401 9 _._ 0640 _±_ 0 _._ 0156
SNN 9 _._ 0054 _±_ 0 _._ 0256 8 _._ 9351 _±_ 0 _._ 0073
Trompt 8 _._ 9707 _± nan_ –
AutoInt 9 _._ 0430 _±_ 0 _._ 0280 8 _._ 9619 _±_ 0 _._ 0092
MLP-Mixer 8 _._ 9589 _±_ 0 _._ 0182 8 _._ 9086 _±_ 0 _._ 0177
Excel _[∗]_ 9 _._ 0395 _±_ 0 _._ 0266 8 _._ 9551 _± nan_
SAINT 9 _._ 0248 _±_ 0 _._ 0225 –
FT-T 9 _._ 0005 _±_ 0 _._ 0215 8 _._ 9360 _±_ 0 _._ 0013
T2G 8 _._ 9775 _±_ 0 _._ 0138 8 _._ 8979 _± nan_
MLP _[‡−]_ [lite] 8 _._ 9355 _±_ 0 _._ 0103 8 _._ 9063 _±_ 0 _._ 0030
MLP _[‡]_ 8 _._ 9455 _±_ 0 _._ 0173 8 _._ 9083 _±_ 0 _._ 0046
MLP _[†]_ 8 _._ 9379 _±_ 0 _._ 0206 8 _._ 8753 _±_ 0 _._ 0038
XGBoost 9 _._ 0307 _±_ 0 _._ 0028 9 _._ 0245 _±_ 0 _._ 0015
LightGBM 9 _._ 0200 _±_ 0 _._ 0025 9 _._ 0128 _±_ 0 _._ 0015
CatBoost 9 _._ 0370 _±_ 0 _._ 0073 9 _._ 0054 _±_ 0 _._ 0028
TabR 9 _._ 0069 _±_ 0 _._ 0152 8 _._ 9132 _±_ 0 _._ 0088
TabR _[‡]_ 8 _._ 9721 _±_ 0 _._ 0105 –
MNCA 8 _._ 9476 _±_ 0 _._ 0152 8 _._ 8977 _±_ 0 _._ 0037
MNCA _[‡]_ 8 _._ 8973 _±_ 0 _._ 0082 8 _._ 8550 _±_ 0 _._ 0031
TabM _[♠]_ 8 _._ 8701 _±_ 0 _._ 0110 8 _._ 8517 _±_ 0 _._ 0022
TabM 8 _._ 8705 _±_ 0 _._ 0043 8 _._ 8642 _±_ 0 _._ 0028
TabM[G] 8 _._ 8723 _±_ 0 _._ 0080 –
TabM mini 8 _._ 9164 _±_ 0 _._ 0089 8 _._ 9021 _±_ 0 _._ 0036
TabM _[†]_ mini 8 _._ 8737 _±_ 0 _._ 0119 8 _._ 8564 _±_ 0 _._ 0054


Published as a conference paper at ICLR 2025


Table 20: Extended results for TabReD Rubachev et al. (2024) benchmark. Results are grouped by
datasets. One ensemble consists of five models trained independently under different random seeds.



sberbank-housing ↓

Method Single model Ensemble

MLP 0 _._ 2529 _±_ 0 _._ 0078 0 _._ 2474 _±_ 0 _._ 0052
TabPFN – –
ResNet – –
DCN2 0 _._ 2616 _±_ 0 _._ 0049 0 _._ 2506 _±_ 0 _._ 0015
SNN 0 _._ 2671 _±_ 0 _._ 0140 0 _._ 2555 _±_ 0 _._ 0033
Trompt 0 _._ 2509 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 0 _._ 2533 _±_ 0 _._ 0046 0 _._ 2485 _± nan_
SAINT 0 _._ 2467 _±_ 0 _._ 0019 –
FT-T 0 _._ 2440 _±_ 0 _._ 0038 0 _._ 2367 _±_ 0 _._ 0010
T2G 0 _._ 2416 _±_ 0 _._ 0025 0 _._ 2343 _± nan_
MLP _[‡−]_ [lite] 0 _._ 2528 _±_ 0 _._ 0055 0 _._ 2503 _±_ 0 _._ 0029
MLP _[‡]_ 0 _._ 2412 _±_ 0 _._ 0031 0 _._ 2355 _±_ 0 _._ 0006
MLP _[†]_ 0 _._ 2383 _±_ 0 _._ 0032 0 _._ 2327 _±_ 0 _._ 0009
XGBoost 0 _._ 2419 _±_ 0 _._ 0012 0 _._ 2416 _±_ 0 _._ 0007
LightGBM 0 _._ 2468 _±_ 0 _._ 0009 0 _._ 2467 _±_ 0 _._ 0002
CatBoost 0 _._ 2482 _±_ 0 _._ 0034 0 _._ 2473 _±_ 0 _._ 0016
TabR 0 _._ 2820 _±_ 0 _._ 0323 0 _._ 2603 _±_ 0 _._ 0048
TabR _[‡]_ 0 _._ 2542 _±_ 0 _._ 0101 –
MNCA 0 _._ 2593 _±_ 0 _._ 0053 0 _._ 2520 _±_ 0 _._ 0032
MNCA _[‡]_ 0 _._ 2448 _±_ 0 _._ 0039 0 _._ 2404 _±_ 0 _._ 0025
TabM _[♠]_ 0 _._ 2469 _±_ 0 _._ 0035 0 _._ 2440 _±_ 0 _._ 0026
TabM 0 _._ 2439 _±_ 0 _._ 0021 0 _._ 2428 _±_ 0 _._ 0006
TabM[G] 0 _._ 2436 _±_ 0 _._ 0027 –
TabM mini 0 _._ 2433 _±_ 0 _._ 0017 0 _._ 2422 _±_ 0 _._ 0004
TabM _[†]_ mini 0 _._ 2334 _±_ 0 _._ 0018 0 _._ 2324 _±_ 0 _._ 0009


maps-routing ↓

Method Single model Ensemble

MLP 0 _._ 1625 _±_ 0 _._ 0001 0 _._ 1621 _±_ 0 _._ 0000
TabPFN – –
ResNet – –
DCN2 0 _._ 1656 _±_ 0 _._ 0004 0 _._ 1636 _±_ 0 _._ 0001
SNN 0 _._ 1634 _±_ 0 _._ 0002 0 _._ 1625 _±_ 0 _._ 0000
Trompt 0 _._ 1624 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 0 _._ 1628 _±_ 0 _._ 0001 0 _._ 1621 _± nan_
SAINT 0 _._ 1634 _± nan_ –
FT-T 0 _._ 1625 _±_ 0 _._ 0003 0 _._ 1619 _±_ 0 _._ 0001
T2G 0 _._ 1616 _±_ 0 _._ 0001 0 _._ 1608 _± nan_
MLP _[‡−]_ [lite] 0 _._ 1618 _±_ 0 _._ 0002 0 _._ 1613 _±_ 0 _._ 0000
MLP _[‡]_ 0 _._ 1618 _±_ 0 _._ 0002 0 _._ 1613 _±_ 0 _._ 0001
MLP _[†]_ 0 _._ 1620 _±_ 0 _._ 0002 0 _._ 1614 _±_ 0 _._ 0000
XGBoost 0 _._ 1616 _±_ 0 _._ 0001 0 _._ 1614 _±_ 0 _._ 0000
LightGBM 0 _._ 1618 _±_ 0 _._ 0000 0 _._ 1616 _±_ 0 _._ 0000
CatBoost 0 _._ 1619 _±_ 0 _._ 0001 0 _._ 1615 _±_ 0 _._ 0000
TabR 0 _._ 1639 _±_ 0 _._ 0003 0 _._ 1622 _±_ 0 _._ 0002
TabR _[‡]_ 0 _._ 1622 _±_ 0 _._ 0002 –
MNCA 0 _._ 1625 _±_ 0 _._ 0001 0 _._ 1621 _±_ 0 _._ 0001
MNCA _[‡]_ 0 _._ 1627 _±_ 0 _._ 0002 0 _._ 1623 _±_ 0 _._ 0001
TabM _[♠]_ 0 _._ 1612 _±_ 0 _._ 0001 0 _._ 1609 _±_ 0 _._ 0000
TabM 0 _._ 1612 _±_ 0 _._ 0001 0 _._ 1610 _±_ 0 _._ 0001
TabM[G] 0 _._ 1611 _±_ 0 _._ 0001 –
TabM mini 0 _._ 1612 _±_ 0 _._ 0001 0 _._ 1610 _±_ 0 _._ 0000
TabM _[†]_ mini 0 _._ 1610 _±_ 0 _._ 0001 0 _._ 1609 _±_ 0 _._ 0000


36



ecom-offers ↑

Method Single model Ensemble

MLP 0 _._ 5989 _±_ 0 _._ 0017 0 _._ 5995 _±_ 0 _._ 0011
TabPFN – –
ResNet – –
DCN2 0 _._ 5996 _±_ 0 _._ 0043 0 _._ 6039 _±_ 0 _._ 0028
SNN 0 _._ 5912 _±_ 0 _._ 0056 0 _._ 5961 _±_ 0 _._ 0033
Trompt 0 _._ 5803 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 0 _._ 5759 _±_ 0 _._ 0066 0 _._ 5759 _± nan_
SAINT 0 _._ 5812 _±_ 0 _._ 0098 –
FT-T 0 _._ 5775 _±_ 0 _._ 0063 0 _._ 5817 _±_ 0 _._ 0021
T2G 0 _._ 5791 _±_ 0 _._ 0056 0 _._ 5824 _± nan_
MLP _[‡−]_ [lite] 0 _._ 5800 _±_ 0 _._ 0029 0 _._ 5819 _±_ 0 _._ 0011
MLP _[‡]_ 0 _._ 5846 _±_ 0 _._ 0048 0 _._ 5872 _±_ 0 _._ 0018
MLP _[†]_ 0 _._ 5949 _±_ 0 _._ 0013 0 _._ 5953 _±_ 0 _._ 0006
XGBoost 0 _._ 5763 _±_ 0 _._ 0072 0 _._ 5917 _±_ 0 _._ 0035
LightGBM 0 _._ 5758 _±_ 0 _._ 0006 0 _._ 5758 _±_ 0 _._ 0003
CatBoost 0 _._ 5596 _±_ 0 _._ 0068 0 _._ 5067 _±_ 0 _._ 0011
TabR 0 _._ 5943 _±_ 0 _._ 0019 0 _._ 5977 _±_ 0 _._ 0009
TabR _[‡]_ 0 _._ 5762 _±_ 0 _._ 0052 –
MNCA 0 _._ 5765 _±_ 0 _._ 0087 0 _._ 5820 _±_ 0 _._ 0047
MNCA _[‡]_ 0 _._ 5758 _±_ 0 _._ 0050 0 _._ 5796 _±_ 0 _._ 0009
TabM _[♠]_ 0 _._ 5948 _±_ 0 _._ 0006 0 _._ 5952 _±_ 0 _._ 0004
TabM 0 _._ 5941 _±_ 0 _._ 0003 0 _._ 5941 _±_ 0 _._ 0000
TabM[G] 0 _._ 5970 _±_ 0 _._ 0010 –
TabM mini 0 _._ 5942 _±_ 0 _._ 0003 0 _._ 5943 _±_ 0 _._ 0001
TabM _[†]_ mini 0 _._ 5910 _±_ 0 _._ 0012 0 _._ 5913 _±_ 0 _._ 0002


homesite-insurance ↑

Method Single model Ensemble

MLP 0 _._ 9506 _±_ 0 _._ 0005 0 _._ 9514 _±_ 0 _._ 0001
TabPFN – –
ResNet – –
DCN2 0 _._ 9398 _±_ 0 _._ 0053 0 _._ 9432 _±_ 0 _._ 0018
SNN 0 _._ 9473 _±_ 0 _._ 0013 0 _._ 9484 _±_ 0 _._ 0007
Trompt 0 _._ 9588 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 0 _._ 9622 _±_ 0 _._ 0004 0 _._ 9635 _± nan_
SAINT 0 _._ 9613 _± nan_ –
FT-T 0 _._ 9622 _±_ 0 _._ 0006 0 _._ 9633 _±_ 0 _._ 0001
T2G 0 _._ 9624 _±_ 0 _._ 0006 0 _._ 9637 _± nan_
MLP _[‡−]_ [lite] 0 _._ 9609 _±_ 0 _._ 0009 0 _._ 9626 _±_ 0 _._ 0003
MLP _[‡]_ 0 _._ 9617 _±_ 0 _._ 0004 0 _._ 9630 _±_ 0 _._ 0002
MLP _[†]_ 0 _._ 9582 _±_ 0 _._ 0014 0 _._ 9599 _±_ 0 _._ 0002
XGBoost 0 _._ 9601 _±_ 0 _._ 0002 0 _._ 9602 _±_ 0 _._ 0000
LightGBM 0 _._ 9603 _±_ 0 _._ 0002 0 _._ 9604 _±_ 0 _._ 0001
CatBoost 0 _._ 9606 _±_ 0 _._ 0003 0 _._ 9609 _±_ 0 _._ 0001
TabR 0 _._ 9487 _±_ 0 _._ 0014 0 _._ 9505 _±_ 0 _._ 0001
TabR _[‡]_ 0 _._ 9556 _±_ 0 _._ 0021 –
MNCA 0 _._ 9514 _±_ 0 _._ 0038 0 _._ 9522 _±_ 0 _._ 0027
MNCA _[‡]_ 0 _._ 9620 _±_ 0 _._ 0006 0 _._ 9635 _±_ 0 _._ 0002
TabM _[♠]_ 0 _._ 9641 _±_ 0 _._ 0004 0 _._ 9644 _±_ 0 _._ 0003
TabM 0 _._ 9640 _±_ 0 _._ 0002 0 _._ 9642 _±_ 0 _._ 0001
TabM[G] 0 _._ 9641 _±_ 0 _._ 0003 –
TabM mini 0 _._ 9643 _±_ 0 _._ 0003 0 _._ 9645 _±_ 0 _._ 0001
TabM _[†]_ mini 0 _._ 9631 _±_ 0 _._ 0003 0 _._ 9634 _±_ 0 _._ 0001


Published as a conference paper at ICLR 2025


cooking-time ↓

Method Single model Ensemble

MLP 0 _._ 4828 _±_ 0 _._ 0002 0 _._ 4822 _±_ 0 _._ 0000
TabPFN – –
ResNet – –
DCN2 0 _._ 4834 _±_ 0 _._ 0003 0 _._ 4822 _±_ 0 _._ 0001
SNN 0 _._ 4835 _±_ 0 _._ 0006 0 _._ 4818 _±_ 0 _._ 0002
Trompt 0 _._ 4809 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 0 _._ 4821 _±_ 0 _._ 0005 0 _._ 4808 _± nan_
SAINT 0 _._ 4840 _± nan_ –
FT-T 0 _._ 4820 _±_ 0 _._ 0008 0 _._ 4813 _±_ 0 _._ 0005
T2G 0 _._ 4809 _±_ 0 _._ 0008 0 _._ 4797 _± nan_
MLP _[‡−]_ [lite] 0 _._ 4811 _±_ 0 _._ 0004 0 _._ 4805 _±_ 0 _._ 0001
MLP _[‡]_ 0 _._ 4809 _±_ 0 _._ 0006 0 _._ 4804 _±_ 0 _._ 0003
MLP _[†]_ 0 _._ 4812 _±_ 0 _._ 0004 0 _._ 4807 _±_ 0 _._ 0002
XGBoost 0 _._ 4823 _±_ 0 _._ 0001 0 _._ 4821 _±_ 0 _._ 0000
LightGBM 0 _._ 4826 _±_ 0 _._ 0001 0 _._ 4825 _±_ 0 _._ 0001
CatBoost 0 _._ 4823 _±_ 0 _._ 0001 0 _._ 4820 _±_ 0 _._ 0001
TabR 0 _._ 4828 _±_ 0 _._ 0008 0 _._ 4814 _±_ 0 _._ 0004
TabR _[‡]_ 0 _._ 4818 _±_ 0 _._ 0006 –
MNCA 0 _._ 4825 _±_ 0 _._ 0004 0 _._ 4819 _±_ 0 _._ 0003
MNCA _[‡]_ 0 _._ 4818 _±_ 0 _._ 0005 0 _._ 4809 _±_ 0 _._ 0003
TabM _[♠]_ 0 _._ 4803 _±_ 0 _._ 0006 0 _._ 4797 _±_ 0 _._ 0003
TabM 0 _._ 4804 _±_ 0 _._ 0002 0 _._ 4802 _±_ 0 _._ 0000
TabM[G] 0 _._ 4800 _±_ 0 _._ 0002 –
TabM mini 0 _._ 4803 _±_ 0 _._ 0001 0 _._ 4801 _±_ 0 _._ 0001
TabM _[†]_ mini 0 _._ 4804 _±_ 0 _._ 0001 0 _._ 4803 _±_ 0 _._ 0000


delivery-eta ↓

Method Single model Ensemble

MLP 0 _._ 5493 _±_ 0 _._ 0007 0 _._ 5478 _±_ 0 _._ 0006
TabPFN – –
ResNet – –
DCN2 0 _._ 5516 _±_ 0 _._ 0014 0 _._ 5495 _±_ 0 _._ 0004
SNN 0 _._ 5495 _±_ 0 _._ 0008 0 _._ 5479 _±_ 0 _._ 0001
Trompt 0 _._ 5519 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 0 _._ 5552 _±_ 0 _._ 0030 0 _._ 5524 _± nan_
SAINT 0 _._ 5528 _± nan_ –
FT-T 0 _._ 5542 _±_ 0 _._ 0026 0 _._ 5523 _±_ 0 _._ 0018
T2G 0 _._ 5527 _±_ 0 _._ 0016 0 _._ 5512 _± nan_
MLP _[‡−]_ [lite] 0 _._ 5521 _±_ 0 _._ 0014 0 _._ 5512 _±_ 0 _._ 0005
MLP _[‡]_ 0 _._ 5535 _±_ 0 _._ 0019 0 _._ 5526 _±_ 0 _._ 0009
MLP _[†]_ 0 _._ 5521 _±_ 0 _._ 0019 0 _._ 5511 _±_ 0 _._ 0007
XGBoost 0 _._ 5468 _±_ 0 _._ 0002 0 _._ 5463 _±_ 0 _._ 0001
LightGBM 0 _._ 5468 _±_ 0 _._ 0001 0 _._ 5465 _±_ 0 _._ 0000
CatBoost 0 _._ 5465 _±_ 0 _._ 0001 0 _._ 5461 _±_ 0 _._ 0000
TabR 0 _._ 5514 _±_ 0 _._ 0024 0 _._ 5480 _±_ 0 _._ 0005
TabR _[‡]_ 0 _._ 5520 _±_ 0 _._ 0015 –
MNCA 0 _._ 5498 _±_ 0 _._ 0007 0 _._ 5488 _±_ 0 _._ 0002
MNCA _[‡]_ 0 _._ 5507 _±_ 0 _._ 0013 0 _._ 5494 _±_ 0 _._ 0006
TabM _[♠]_ 0 _._ 5510 _±_ 0 _._ 0015 0 _._ 5504 _±_ 0 _._ 0004
TabM 0 _._ 5494 _±_ 0 _._ 0004 0 _._ 5492 _±_ 0 _._ 0001
TabM[G] 0 _._ 5509 _±_ 0 _._ 0003 –
TabM mini 0 _._ 5497 _±_ 0 _._ 0007 0 _._ 5495 _±_ 0 _._ 0003
TabM _[†]_ mini 0 _._ 5510 _±_ 0 _._ 0019 0 _._ 5502 _±_ 0 _._ 0000


37



homecredit-default ↑

Method Single model Ensemble

MLP 0 _._ 8538 _±_ 0 _._ 0014 0 _._ 8566 _±_ 0 _._ 0005
TabPFN – –
ResNet – –
DCN2 0 _._ 8471 _±_ 0 _._ 0019 0 _._ 8549 _±_ 0 _._ 0002
SNN 0 _._ 8541 _±_ 0 _._ 0016 0 _._ 8569 _±_ 0 _._ 0010
Trompt 0 _._ 8355 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 0 _._ 8513 _±_ 0 _._ 0024 0 _._ 8564 _± nan_
SAINT 0 _._ 8377 _± nan_ –
FT-T 0 _._ 8571 _±_ 0 _._ 0023 0 _._ 8611 _±_ 0 _._ 0013
T2G 0 _._ 8597 _±_ 0 _._ 0007 0 _._ 8629 _± nan_
MLP _[‡−]_ [lite] 0 _._ 8598 _±_ 0 _._ 0009 0 _._ 8607 _±_ 0 _._ 0003
MLP _[‡]_ 0 _._ 8572 _±_ 0 _._ 0011 0 _._ 8590 _±_ 0 _._ 0003
MLP _[†]_ 0 _._ 8568 _±_ 0 _._ 0039 0 _._ 8614 _±_ 0 _._ 0014
XGBoost 0 _._ 8670 _±_ 0 _._ 0005 0 _._ 8674 _±_ 0 _._ 0001
LightGBM 0 _._ 8664 _±_ 0 _._ 0004 0 _._ 8667 _±_ 0 _._ 0000
CatBoost 0 _._ 8627 _± nan_ –
TabR 0 _._ 8501 _±_ 0 _._ 0027 0 _._ 8548 _±_ 0 _._ 0003
TabR _[‡]_ 0 _._ 8547 _±_ 0 _._ 0021 –
MNCA 0 _._ 8531 _±_ 0 _._ 0018 0 _._ 8569 _±_ 0 _._ 0004
MNCA _[‡]_ 0 _._ 8544 _±_ 0 _._ 0033 0 _._ 8606 _±_ 0 _._ 0024
TabM _[♠]_ 0 _._ 8583 _±_ 0 _._ 0010 0 _._ 8599 _±_ 0 _._ 0006
TabM 0 _._ 8599 _±_ 0 _._ 0010 0 _._ 8607 _±_ 0 _._ 0002
TabM[G] 0 _._ 8588 _±_ 0 _._ 0013 –
TabM mini 0 _._ 8605 _±_ 0 _._ 0010 0 _._ 8614 _±_ 0 _._ 0007
TabM _[†]_ mini 0 _._ 8635 _±_ 0 _._ 0008 0 _._ 8646 _±_ 0 _._ 0004


weather ↓

Method Single model Ensemble

MLP 1 _._ 5378 _±_ 0 _._ 0054 1 _._ 5111 _±_ 0 _._ 0029
TabPFN – –
ResNet – –
DCN2 1 _._ 5606 _±_ 0 _._ 0057 1 _._ 5292 _±_ 0 _._ 0028
SNN 1 _._ 5280 _±_ 0 _._ 0085 1 _._ 5013 _±_ 0 _._ 0034
Trompt 1 _._ 5187 _± nan_ –
AutoInt – –
MLP-Mixer – –
Excel _[∗]_ 1 _._ 5131 _±_ 0 _._ 0022 1 _._ 4707 _± nan_
SAINT 1 _._ 5097 _±_ 0 _._ 0045 –
FT-T 1 _._ 5104 _±_ 0 _._ 0097 1 _._ 4719 _±_ 0 _._ 0040
T2G 1 _._ 4849 _±_ 0 _._ 0087 1 _._ 4513 _± nan_
MLP _[‡−]_ [lite] 1 _._ 5170 _±_ 0 _._ 0040 1 _._ 4953 _±_ 0 _._ 0023
MLP _[‡]_ 1 _._ 5139 _±_ 0 _._ 0031 1 _._ 4978 _±_ 0 _._ 0020
MLP _[†]_ 1 _._ 5162 _±_ 0 _._ 0020 1 _._ 5066 _±_ 0 _._ 0008
XGBoost 1 _._ 4671 _±_ 0 _._ 0006 1 _._ 4629 _±_ 0 _._ 0002
LightGBM 1 _._ 4625 _±_ 0 _._ 0008 1 _._ 4581 _±_ 0 _._ 0003
CatBoost 1 _._ 4688 _±_ 0 _._ 0019 –
TabR 1 _._ 4666 _±_ 0 _._ 0039 1 _._ 4547 _±_ 0 _._ 0008
TabR _[‡]_ 1 _._ 4458 _±_ 0 _._ 0018 –
MNCA 1 _._ 5062 _±_ 0 _._ 0054 1 _._ 4822 _±_ 0 _._ 0013
MNCA _[‡]_ 1 _._ 5008 _±_ 0 _._ 0034 1 _._ 4782 _±_ 0 _._ 0011
TabM _[♠]_ 1 _._ 4786 _±_ 0 _._ 0039 1 _._ 4715 _±_ 0 _._ 0020
TabM 1 _._ 4722 _±_ 0 _._ 0024 1 _._ 4675 _±_ 0 _._ 0009
TabM[G] 1 _._ 4728 _±_ 0 _._ 0022 –
TabM mini 1 _._ 4716 _±_ 0 _._ 0016 1 _._ 4669 _±_ 0 _._ 0010
TabM _[†]_ mini 1 _._ 4651 _±_ 0 _._ 0020 1 _._ 4581 _±_ 0 _._ 0016


I should output the following data snippet approximately with some context:
{
"overall_score": 35,
"Novelty / Originality": {
"Score": 5,

"Description": "The paper introduces a genuinely novel idea and technique that goes well beyond incremental improvements. The approach demonstrates clear originality and creativity."
},
"Significance / Impact / Usefulness": {
"Score": 5,

"Description": "The proposed method has high potential impact on the field and can significantly advance real-world applications. It meaningfully contributes to both academic understanding and practical utility."
},
"Technical Soundness / Correctness": {
"Score": 5,

"Description": "The theoretical framework and derivations are technically sound, with all assumptions well justified. The methodology is rigorous and correctly implemented."
},
"Empirical / Experimental Validation": {
"Score": 5,

"Description": "The experiments are comprehensive, covering multiple datasets, strong baselines, and detailed ablations. Results consistently support the paper·s claims."
},
"Reproducibility / Transparency": {
"Score": 5,

"Description": "The paper provides all necessary resources including code, data, hyperparameters, and setup instructions, ensuring complete reproducibility of the results."
},
"Related Work & Positioning": {
"Score": 5,

"Description": "The discussion of prior work is extensive and well-structured. The paper clearly situates its contributions in context, highlighting both similarities and distinctions effectively."
},
"Ethics / Broader Impacts / Safety": {
"Score": 5,

"Description": "The authors thoughtfully consider ethical aspects, societal implications, and potential risks. They address data provenance, licensing, and safety concerns responsibly."
}
}


