Published as a conference paper at ICLR 2025

# Beyond Random Masking: When Dropout Meets Graph Convolutional Networks


**Yuankai Luo** [1][,][2] **& Xiao-Ming Wu** [2] **& Hao Zhu** [3][,][∗]

1 Beihang University, Beijing, China
2 The Hong Kong Polytechnic University, Hong Kong
3 Data61�CSIRO, Sydney, Australia


Abstract


Graph Convolutional Networks (GCNs) have emerged as powerful tools for learning on graph-structured data, yet the behavior of dropout in these models remains poorly understood. This paper presents a comprehensive theoretical analysis of dropout in GCNs, revealing that its primary role differs fundamentally from
standard neural networks - preventing oversmoothing rather than co-adaptation.
We demonstrate that dropout in GCNs creates dimension-specific stochastic subgraphs, leading to a form of structural regularization not present in standard
neural networks. Our analysis shows that dropout effects are inherently degreedependent, resulting in adaptive regularization that considers the topological importance of nodes. We provide new insights into dropout’s role in mitigating
oversmoothing and derive novel generalization bounds that account for graphspecific dropout effects. Furthermore, we analyze the synergistic interaction between dropout and batch normalization in GCNs, uncovering a mechanism that
enhances overall regularization. Our theoretical findings are validated through extensive experiments on both node-level and graph-level tasks across 14 datasets.
Notably, GCN with dropout and batch normalization outperforms state-of-the-art
methods on several benchmarks, demonstrating the practical impact of our theoretical insights.


1 Introduction


The remarkable success of deep neural networks across various domains has been accompanied
by the persistent challenge of overfitting, where models perform well on training data but fail to
generalize to unseen examples. This issue has spurred the development of numerous regularization
techniques, among which dropout has emerged as a particularly effective and widely adopted approach (LeCun et al., 2015). Introduced by Srivastava et al. (2014), dropout addresses overfitting
by randomly ”dropping out” a proportion of neurons during training, effectively creating an ensemble of subnetworks. This technique has proven highly successful in improving generalization and
has become a standard tool in the deep learning toolkit. The effectiveness of dropout has prompted
extensive theoretical analysis, with various perspectives offered to explain its regularization effects.


Some researchers have interpreted dropout as a form of model averaging (Baldi & Sadowski, 2013),
while others have analyzed it through the lens of information theory (Achille & Soatto, 2018). Wager
et al. (2013) provided insights into dropout’s adaptive regularization properties, and Gal & Ghahramani (2016) established connections between dropout and Bayesian inference. These diverse theoretical frameworks have significantly enhanced our understanding of dropout’s role in mitigating
overfitting in traditional neural networks. However, as the field of deep learning has expanded to
encompass more complex data structures, particularly graphs, new questions have arisen regarding
the applicability and behavior of established techniques. Graph Neural Networks (GNNs), especially Graph Convolutional Networks (GCNs), have demonstrated remarkable performance on tasks
involving graph-structured data (Kipf & Welling, 2017). Naturally, researchers and practitioners
have applied dropout to GNNs, often observing beneficial effects on generalization (Hamilton et al.,
2017).


∗ Hao Zhu is the corresponding author and led the writing of the paper.


1


Published as a conference paper at ICLR 2025


While dropout was originally designed to prevent co-adaptation of features in standard neural networks, our analysis reveals that its primary mechanism in GCNs is fundamentally different. We
demonstrate that dropout’s main contribution in GCNs is mitigating oversmoothing by maintaining
feature diversity across nodes, rather than preventing co-adaptation as in standard neural networks.
This finding represents a significant shift in our understanding of how regularization operates in
graph neural networks. Specifically, we demonstrate that:


   - Dropout in GCNs creates dimension-specific stochastic sub-graphs, leading to a unique
form of structural regularization not present in standard neural networks.


   - The effects of dropout are inherently degree-dependent, with differential impacts on nodes
based on their connectivity, resulting in adaptive regularization that considers the topological importance of nodes in the graph.


   - Dropout plays a crucial role in mitigating the oversmoothing problem rather than coadaption in GCNs, though its effects are more nuanced than previously thought.


   - The generalization bounds for GCNs with dropout exhibit a complex dependence on graph
properties, diverging from traditional dropout theory.


   - There exists a significant interplay between dropout and batch normalization in GCNs,
revealing synergistic effects that enhance the overall regularization.


Our theoretical framework not only provides deeper insights into the mechanics of dropout in graphstructured data but also yields practical implications for the design and training of GCNs. We validate our theoretical findings through extensive experiments on both node-level and graph-level tasks,
demonstrating the practical relevance of our analysis. This work bridges a critical gap in the theoretical understanding of regularization in GCNs and paves the way for more principled approaches to
leveraging dropout in graph representation learning. Furthermore, we validate our theoretical findings through extensive experiments, demonstrating that GCNs incorporating our insights on dropout
and batch normalization outperform several state-of-the-art methods on benchmark datasets. This
practical success underscores the importance of our theoretical contributions and their potential to
advance the field of graph representation learning.


2 Related Work


**Dropout in Neural Networks.** Overfitting can be reduced by using dropout Hinton et al. (2012)
to prevent complex co-adaptations on the training data. Since its inception, several variants have
been proposed to enhance its effectiveness. DropConnect (Wan et al., 2013) generalizes dropout
by randomly dropping connections rather than nodes. Gaussian dropout Srivastava et al. (2014)
replaces the Bernoulli distribution with a Gaussian one for smoother regularization. Curriculum
dropout (Morerio et al., 2017) adaptively adjusts the dropout rate during training. Theoretical interpretations of dropout have provided insights into its success. The model averaging perspective (Baldi
& Sadowski, 2013) views dropout as an efficient way of approximately combining exponentially
many different neural networks. The adaptive regularization interpretation (Wager et al., 2013)
shows how dropout adjusts the regularization strength for each feature based on its importance. The
Bayesian approximation view (Gal & Ghahramani, 2016) connects dropout to variational inference
in Bayesian neural networks, providing a probabilistic framework for understanding its effects.


**Regularization in Graph Neural Networks.** Graph Neural Networks (GNNs), while powerful,
are prone to overfitting and over-smoothing (Li et al., 2018). Various regularization techniques (Yang
et al., 2021; Rong et al., 2020; Fang et al., 2023; Feng et al., 2020) have been proposed to address
these issues. DropEdge (Rong et al., 2020) randomly removes edges from the input graph during training, reducing over-smoothing and improving generalization. Graph diffusion-based methods (Gasteiger et al., 2019) incorporate higher-order neighborhood information to enhance model
robustness. Spectral-based approaches (Wu et al., 2019) leverage the graph spectrum to design effective regularization strategies. Empirical studies have shown that traditional dropout can be effective
in GNNs (Hamilton et al., 2017), but its interaction with graph structure remains poorly understood.
Some works have proposed adaptive dropout strategies for GNNs (Gao & Ji, 2019), but these are
primarily heuristic approaches without comprehensive theoretical grounding.


2


Published as a conference paper at ICLR 2025


**Theoretical Frameworks for GNNs.** Despite the empirical success of Graph Neural Networks
(GNNs), establishing theories to explain their behaviors is still an evolving field. Recent works have
made significant progress in understanding over-smoothing (Li et al., 2018; Zhao & Akoglu, 2019;
Oono & Suzuki, 2019; Rong et al., 2020), interpretability (Ying et al., 2019; Luo et al., 2020; Vu
& Thai, 2020; Yuan et al., 2020; 2021), expressiveness (Xu et al., 2018; Chen et al., 2019; Maron
et al., 2018; Dehmamy et al., 2019; Feng et al., 2022), and generalization (Scarselli et al., 2018; Du
et al., 2019; Verma & Zhang, 2019; Garg et al., 2020; Zhang et al., 2020; Oono & Suzuki, 2019;
Lv, 2021; Liao et al., 2020; Esser et al., 2021; Cong et al., 2021). Our work aims to complement
these existing theoretical frameworks by focusing on the practical aspects of dropout in GNNs,
a widely used regularization technique that has not been thoroughly examined from a theoretical
perspective. Previous works have provided valuable insights using classical techniques such as
Vapnik-Chervonenkis dimension (Scarselli et al., 2018), Rademacher complexity (Lv, 2021; Garg
et al., 2020), and algorithm stability (Verma & Zhang, 2019). Recent efforts (Oono & Suzuki,
2019; Esser et al., 2021) have also made strides in incorporating the transductive learning schema of
GNNs into theoretical analyses. We bridge the gap between theoretical understanding and practical
implementation of GNNs, offering insights into how dropout affects generalization and performance
in graph-structured learning tasks.


3 Theoretical Framework


In this section, we develop a rigorous mathematical framework to analyze the behavior of dropout
in Graph Convolutional Networks (GCNs). We begin by establishing notations and definitions, then
formalize the GCN model with dropout, and finally introduce key concepts that will be central to
our analysis.


3.1 Notations and Definitions



**Notations.** Let G = (V, E, _**X**_ ) be an undirected graph with _n_ = |V| nodes and _m_ = |E| edges,
where **X** ∈ R _[n]_ [×] _[d]_ [0] represents the node feature matrix with _d_ 0 input features per node. We denote by
_**A**_ ∈ R _[n]_ [×] _[n]_ the adjacency matrix, _**D**_ = diag( _deg_ 1, . . ., _deg_ _n_ ) the degree matrix where _deg_ _i_ = [�] _j_ _[A]_ _i j_ [,]
and _**A**_ [˜] = _**D**_ [−] [1] 2 _**AD**_ [−] 2 [1] the normalized adjacency matrix.


**Graph Convolutional Networks (GCNs).** An L-layer GCN performs the following layer-wise
transformation:


_**H**_ [(] _[l]_ [)] = σ( _**AH**_ [˜] [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ), (1)


where _**H**_ [(] _[l]_ [)] ∈ R _[n]_ [×] _[d]_ _[l]_ is the feature matrix, _**W**_ [(] _[l]_ [)] ∈ R _[d]_ _[l]_ [−][1] [×] _[d]_ _[l]_ is the weight matrix, σ(·) is a non-linear
activation, and _**H**_ [(0)] = _**X**_ . The feature energy measures representation smoothness:




[1]

2 _**AD**_ [−] 2 [1]



2 the normalized adjacency matrix.



1
_E_ ( _**H**_ [(] _[l]_ [)] ) =
2|E|



� ∥ _**h**_ [(] _i_ _[l]_ [)] [−] _**[h]**_ [(] _j_ _[l]_ [)] [∥] 2 [2] (2)

( _i_, _j_ )∈E



**Dropout in GCNs.** For layer _l_, dropout applies a random mask _**M**_ [(] _[l]_ [)] ∈ R _[n]_ [×] _[d]_ _[l]_ where each element
_M_ _i j_ [(] _[l]_ [)] [is drawn independently from Bernoulli(1][ −] _[p]_ [). The forward pass with dropout is defined as:]


1
_**H**_ [(] _[l]_ [)] = (3)
1 − _p_ _**[M]**_ [ (] _[l]_ [)] [ ⊙] [σ][( ˜] _**[AH]**_ [(] _[l]_ [−][1)] _**[W]**_ [ (] _[l]_ [)] [)][,]


where ⊙ denotes element-wise multiplication and _p_ is the dropout probability.


**Batch Normalization.** When incorporating batch normalization, the layer transformation becomes:


_**H**_ [(] _[l]_ [)] = σ(BN( _**AH**_ [˜] [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] )), (4)


where BN applies feature-wise normalization BN( _**X**_ ) = γ ⊙ ~~√~~ _**X**_ σ− [2] _B_ µ [+] _B_ [ϵ] [+][ β][ with learnable parameters]

γ, β and batch statistics µ _B_, σ [2] _B_ [.]


3


Published as a conference paper at ICLR 2025


Figure 1: Illustration of how dropout creates dimension-specific sub-graphs. From left to right: the
original graph with complete feature vectors, the graph after applying dropout (where _x_ indicates
dropped features), and the resulting sub-graphs for each feature dimension. Different colors indicate
different feature dimensions, and grayed-out nodes show where features are dropped, preventing
message passing along those paths in the next convolution.


3.2 Dimension-Specific Stochastic Sub-graphs


We demonstrate how dropout creates dimension-specific sub-graphs in Figure 1. At each iteration _t_,
dropout induces **dimension-specific stochastic sub-graphs** G [(] _t_ _[l]_ [,] _[j]_ [)] = (V, E [(] _t_ _[l]_ [,] _[j]_ [)] ) with:


E [(] _t_ _[l]_ [,] _[ j]_ [)] = {( _u_, _v_ ) ∈E | _M_ _u j_ [(] _[l]_ [,] _[t]_ [)] [�] [0 and] _[ M]_ _v j_ [(] _[l]_ [,] _[t]_ [)] � 0}. (5)


The coupling between feature dropout and graph topology is captured by the feature-topology coupling matrix:
_**C**_ _t_ [(] _[l]_ [)] = _**A**_ ⊙ 1 [( _**M**_ [(] _[l]_ [,] _[t]_ [)] ( _**M**_ [(] _[l]_ [,] _[t]_ [)] ) [⊤] ) > 0], (6)


which measures how dropout simultaneously affects connected nodes’ features. This interaction
manifests in each node’s effective degree:


_deg_ [e] _i_, [ff] _t_ [=][ |{] _[ j]_ [ ∈N][(] _[i]_ [) :][ ∃] _[k]_ [,] _[ M]_ _ik_ [(] _[l]_ [,] _[t]_ [)] � 0 and _M_ [(] _jk_ _[l]_ [,] _[t]_ [)] � 0}| = �( _**C**_ _t_ [(] _[l]_ [)] [)] _[i j]_ [,] (7)

_j_


representing the actual count of node i’s neighbors that maintain feature connections after dropout.
We consider a path P = ( _v_ 0, . . ., _v_ _k_ ) active for feature _j_ when all nodes along the path retain this feature, i.e., [�] _[k]_ _i_ = [−] 0 [1] _[M]_ _v_ [(] _[l]_ _i_ _j_ [,] _[t]_ [)] _[M]_ _v_ [(] _[l]_ _i_ + [,] _[t]_ 1 [)] _j_ [�] [0. To elucidate the specific impact of dropout on embedding features,]
we introduce these concepts:


**Theorem 1** (Sub-graph Diversity) **.** _The expected number of distinct sub-graphs per iteration is:_


E[|E [(] _t_ _[l]_ [,] _[ j]_ [)] | _j_ = 1, . . ., _d_ _l_ |] = _d_ _l_ (1 − (1 − _p_ ) [2][|E|] ),


_where d_ _l_ _is the number of features at layer l, p is the dropout probability, and_ |E| _is the number of_
_edges in the original graph (The complete proof is in the Appendix. A.1)._


This theorem reveals that dropout in GCNs leads to a rich set of sub-graphs, providing a form
of structural data augmentation unique to graph-based models. The diversity of these sub-graphs
increases with both the dropout probability _p_ and the number of features _d_ _l_ . This suggests that
higher-dimensional GCNs with moderate dropout rates can benefit from a wider range of structural variations during training, potentially leading to more robust and generalizable representations.
Moreover, this mechanism allows the GCN to implicitly explore different graph structures without
explicitly modifying the input graph. This could be particularly beneficial for tasks where the optimal graph structure is uncertain or where multiple relevant sub-structures exist within the data.


**Theorem 2** (Expected Active Features per Path) **.** _For a path_ P _of length k, the expected number of_
_features for which it is active is:_


E[ _#active features for_ P] = _d_ _l_ (1 − _p_ ) _[k]_ [+][1] .


This theorem demonstrates that while individual long paths are unlikely to be active for any given
feature, the multi-dimensional nature of GCNs allows for effective long-range information flow
through the ensemble effect across features. This theoretical insight is further supported by our
empirical analysis in Appendix A.6.


4


Published as a conference paper at ICLR 2025


3.3 Degree-Dependent Nature of Dropout Effects


The interaction between dropout and the graph structure leads to a form of degree-dependent regularization in GCNs. This means that the effect of dropout varies based on the connectivity of each
node, creating an adaptive regularization scheme that considers the topological importance of nodes
in the graph.

**Theorem 3** (Degree-Dependent Dropout Effect) **.** _The expected e_ ff _ective degree and its variance are_
_given by:_
E[ _deg_ _[e]_ _i_, [ff] _t_ []][ =][ (1][ −] _[p]_ [)] [2] _[deg]_ _[i]_ _[ and Var]_ [[] _[deg]_ _[e]_ _i_, [ff] _t_ []][ =] _[ deg]_ _[i]_ [(1][ −] _[p]_ [)] [2] [(1][ −] [(1][ −] _[p]_ [)] [2] [)][,] (8)


_where deg_ _i_ _is the original degree of node i and p is the dropout probability._


This theorem highlights that dropout affects nodes differentially depending on their degree. Highdegree nodes, typically more influential within the graph, exhibit less variation in their effective
degree due to dropout, potentially resulting in more stable representations for these important
nodes. This observation is empirically confirmed in the analysis of a 2-layer GCN presented in
Appendix A.6. Consequently, the degree-dependent nature of dropout in GCNs results in adaptive
regularization, where the regularization effect naturally adjusts to the local graph structure.

**Corollary 4** (Relative Stability of High-Degree Nodes) **.** _The coe_ ffi _cient of variation of the e_ ff _ective_

_degree, defined as CV_ [ _deg_ _[e]_ _i_, [ff] _t_ []][ =] ~~�~~ _Var_ [ _deg_ _[e]_ _i_, [ff] _t_ []][/][E][[] _[deg]_ _[e]_ _i_, [ff] _t_ []] _[, decreases with increasing node degree:]_



_CV_ [ _deg_ _[e]_ _i_, [ff] _t_ []][ =]



~~�~~



~~�~~ 1 − (1 − _p_ ) [2]

~~�~~ _deg_ _i_ (1 − _p_ )



.
_deg_ _i_ (1 − _p_ )



This corollary further confirms that high-degree nodes experience relatively less variation in their
effective degree due to dropout. Figure 10 illustrates that the CV decreases as node degree increases.
This degree-dependent effect distinguishes dropout in GCNs from its application in standard neural
networks and suggests that the optimal dropout strategy for GCNs may need to consider the graph
structure explicitly.


3.4 Role of Dropout in Oversmoothing


Oversmoothing is a well-known issue in GCNs, where node representations become indistinguishable as the number of layers increases. Our analysis reveals that dropout plays a crucial role in this
context, though its effects are more nuanced than previously thought.


**Theorem 5** (Dropout and Feature Energy) **.** _For a GCN with dropout probability p, the expected_
_feature energy at layer l is bounded by:_



1

E[ _E_ ( _**H**_ [(] _[l]_ [)] )] ≤ _[de]_ |E| _[g]_ [max] ( 1 − _p_ [)] _[l]_ [||] _**[A]**_ [ ˜] [||] 2 [2] _[l]_



_l_
� || _**W**_ [(] _[i]_ [)] || [2] 2 [||] _**[X]**_ [||] [2] _F_ (9)

_i_ =1



_where E_ ( _**X**_ ) _is the energy of the input features and_ _**W**_ [(] _[i]_ [)] _are the weight matrices (The complete_
_proof is in the Appendix.A.2)._


The derived bound demonstrates how dropout affects feature energy through the interplay of network
depth ( _l_ ), graph structure (through _deg_ max and _**A**_ [˜] ), and weight properties (∥ _**W**_ [(] _[i]_ [)] ∥ [2] 2 [). Note that this]
analysis only provides an upper bound; the absence of a lower bound in this derivation is due to
limitations in bounding certain terms. We will later show that when considering batch normalization,
we can establish the existence of a lower bound, providing a more complete characterization.


3.5 Generalization Bounds with Graph-Specific Dropout Effects


The unique properties of dropout in GCNs, such as the creation of stochastic sub-graphs and degreedependent effects, influence how these models generalize to unseen data. Our analysis provides
novel generalization bounds that explicitly account for these graph-specific dropout effects, offering insights into how dropout interacts with graph structure to influence the model’s generalization
capabilities.


5


Published as a conference paper at ICLR 2025


Figure 2: Feature energy vs dropout rates. Figure 3: BN feature energy vs dropout rates.


**Theorem 6** (Generalization Bound for _L_ -Layer GCN with Dropout) **.** _For an L-layer GCN F with_
_dropout probability p_ _l_ _at layer l and L_ σ _-Lipschitz activation function_ σ _, with probability at least_
1 − δ _over the training examples, the following generalization bound holds:_



_L_

~~_p_~~ _l_
_L_ _loss_  - _L_ _l_  
� _l_ =1 � (1 − _p_ _l_ )χ _f_ (G) [∥][σ][( ˜] _**[AH]**_ [(] _[l]_ [−][1)] _**[W]**_ [ (] _[l]_ [)] [)][∥] _[F]_ [,]



~~�~~



~~~~




E _D_ [ _L_ ( _F_ ( _X_ ))] − E _S_ [ _L_ ( _F_ ( _X_ ))] ≤ _O_







log(1/δ)


_n_



(10)
_where_ E _D_ _is the expectation over the data distribution,_ E _S_ _is the expectation over the training sam-_
_ples, L is the loss function with Lipschitz constant L_ _loss_ _, L_ _l_ = [�] _i_ _[L]_ = _l_ [(] _[L]_ [σ] [∥] _**[W]**_ [ (] _[i]_ [)] [∥] [2] [ ·∥] _**[A]**_ [˜] [∥] [2] [)] _[ is the Lipschitz]_
_constant from layer l to output,_ ∥ _**W**_ [(] _[i]_ [)] ∥ 2 _is the spectral norm of the weight matrix at layer i,_ ∥ _**A**_ [˜] ∥ 2 _is_
_the spectral norm of the normalized adjacency matrix, and_ χ _f_ (G) _is the fractional chromatic number_
_of the dependency graph_ G _induced by the message passing structure._


This generalization bound reveals how the network’s stability depends on the loss function’s Lipschitz constant, layer-wise Lipschitz constants capturing weight effects, graph structure through
χ _f_ (G), feature activations, and dropout rates. This leads to several key insights: First, network
depth affects stability through the layer-wise Lipschitz constants _L_ _l_ . The multiplicative accumulation of weight and graph effects ( [�] _i_ _[L]_ = _l_ +1 [∥] _**[W]**_ [ (] _[i]_ [)] [∥∥] _**[A]**_ [˜] [∥][) suggests deeper GCNs require careful regu-]
larization as perturbations can amplify through layers. Second, the graph structure fundamentally
influences stability through χ _f_ (G). Since χ _f_ (G) > 1 for GCNs due to message passing (versus
χ _f_ (G) = 1 for MLPs), GCNs gain natural regularization from their graph structure. This effect
strengthens with graph connectivity since larger χ _f_ (G) leads to better stability. Combined with the
fact that the normalized adjacency matrix has bounded spectral norm (∥ _**A**_ [˜] ∥ 2 ≤ 1), this provides a
built-in stabilizing mechanism unique to GNNs. Third, examining layer-specific terms reveals the
interplay between weights ∥ _**W**_ [(] _[l]_ [)] ∥, feature magnitudes ∥σ( _**AH**_ [˜] [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] )∥ _F_, and dropout rates _p_ _l_ .
The contribution of each layer to the overall bound suggests that adaptive layer-wise dropout rates
might be more effective than uniform dropout, particularly when certain layers process more critical features. Finally, the bound mathematically explains the dropout rate trade-off through the term
� _p_ _l_ /((1 − _p_ _l_ )χ _f_ (G)). Higher dropout provides stronger regularization but increases noise, while the
graph structure (through χ _f_ (G)) moderates this effect. This helps explain why moderate dropout
rates often work best in practice, with the optimal rate depending on the graph’s connectivity patterns. This theoretical insight aligns with empirical observations that GNNs often benefit more from
dropout than MLPs, as the graph structure provides additional stability through χ _f_ (G) while allowing
effective information flow via message passing.


3.6 Interaction of Dropout and Batch Normalization in GCNs


While dropout provides a powerful regularization mechanism for GCNs, its degree-dependent nature
can lead to uneven regularization across nodes. Batch Normalization (BN) offers a complementary
approach that can potentially address this issue and enhance the benefits of dropout. Our analysis
reveals how the combination of dropout and BN creates a synergistic regularization effect that is
sensitive to both graph structure and feature distributions.


**Theorem 7** (Layer-wise Energy Lower Bound for GCN with Dropout and BN) **.** _For an L-layer_
_Graph Convolutional Network with dropout rate p, batch normalization parameters_ {β [(] _d_ _[l]_ [)] [, γ] _d_ [(] _[l]_ [)] [}] _[d]_ _d_ _[l]_ =1 _[at]_


6


Published as a conference paper at ICLR 2025


_each layer l, with probability at least_ (1 − δ) _[L]_ _, the expected feature energy at each layer l satisfies:_



_E_ ( _**H**_ [(] _[l]_ [)] ) ≥ _[p]_ [ ·] _[ de][g]_ [min]

2|E|(1 − _p_ )



_d_ _l_
� Φ(β [(] _d_ _[l]_ [)] [/γ] _d_ [(] _[l]_ [)] [)][ ·][ (][β] [(] _d_ _[l]_ [)] [)] [2]

_d_ =1



_where l_ = 1, 2, ..., _L indicates the layer, deg_ min _is the minimum degree in the graph,_ |E| _is the total_
_number of edges,_ Φ _is the standard normal CDF and_ β [(] _d_ _[l]_ [)] [, γ] _d_ [(] _[l]_ [)] _[are the BN parameters for dimension]_
_d at layer l (The complete proof is in the Appendix.A.4)._


Our theoretical analysis reveals a crucial interplay between dropout and batch normalization in
GCNs. The lower bound on feature energy combines three essential components: (1) A graph structural term _[de]_ 2 _[g]_ |E| _[min]_ [that captures the network connectivity, (2) A dropout-induced scaling factor] 1− _pp_ [that]

amplifies preserved features, and (3) A BN-controlled feature activation term [�] _[d]_ _d_ _[l]_ =1 [Φ][(][β] [(] _d_ _[l]_ [)] [/γ] _d_ [(] _[l]_ [)] [)][·][(][β] [(] _d_ _[l]_ [)] [)] [2]
that establishes a non-zero energy floor. This interaction operates through several key mechanisms:
(1) The BN shift parameters β [(] _d_ _[l]_ [)] [directly contribute to feature energy through their squared magni-]
tude, while the ratio β [(] _d_ _[l]_ [)] [/γ] _d_ [(] _[l]_ [)] [determines the proportion of features preserved through ReLU activa-]
tion via the standard normal CDF Φ. Higher positive values of this ratio increase feature preserva_p_
tion. (2) Dropout’s 1− _p_ [factor enhances this feature preservation e][ff][ect, creating a controlled amplifi-]
cation that prevents feature collapse. This amplification is naturally weighted by graph connectivity,
with minimum degree _deg_ _min_ ensuring baseline protection even for sparsely connected nodes. (3)
The entire bound scales with the graph’s minimum degree, illustrating how the mechanism adapts
to the underlying graph structure, providing stronger guarantees for more densely connected graphs.
This theoretical framework explains our empirical observations in Figures 2 & 3, where batch normalization effectively moderates the energy dynamics in GCNs. By establishing a non-zero lower
bound on feature energy, BN prevents complete feature collapse regardless of weight updates, while
dropout enhances feature discrimination. Their joint application creates a specialized regularization
mechanism for graph-structured data, where BN’s parameter-controlled feature preservation interacts with dropout-induced sparsity to maintain robust node representations across graph topologies.


3.7 Comparison with Other Dropout Variants


Various dropout mechanisms have been proposed for GNNs, each applying masks at different stages
of message passing. We formally characterize these variants through their masking operations and
their effects in Table 1. The key distinction of standard dropout lies in its feature-dimension-specific
masking, which creates unique sub-graph structures for each feature dimension. This leads to a
quadratic effect on the effective degree, providing stronger regularization than other variants. While
DropNode and DropEdge apply coarse-grained masks uniformly across features, and DropMessage
operates at the message level, dropout’s feature-specific approach provides finer-grained control over
information flow.


Table 1: Comparison of different dropout variants in GNNs. Each method is characterized by
its masking operation _**M**_ _d_, the resulting sub-graph formation G _t_, and expected effective degree
E[ _deg_ [e] _i_, [ff] _t_ [], where] _[ p]_ [ is the dropout probability.]


Method Masking Operation Sub-graph Formation Expected Effective Degree

DropNode _**M**_ _d_ = _**A**_ [˜] (( _**M**_ _node_ ⊙ _**H**_ [(] _[l]_ [−][1)] ) _**W**_ [(] _[l]_ [)] ) _d_ G _t_ = (V \ V _dropped_, E \ {( _i_, _j_ )| _i_ ∈V _dropped_ }) _deg_ _i_ ~~�~~ _j_ ∈N( _i_ ) [(1][ −] _[p]_ [)]
DropEdge _**M**_ _d_ = ( _**M**_ _edge_ ⊙ _**A**_ [˜] )( _**H**_ [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ) _d_ G _t_ = (V, E \ E _dropped_ ) (1 − _p_ ) _deg_ _i_
DropMessage _**M**_ _d_ = _**A**_ [˜] ( _**M**_ _msg_ _d_ ⊙ ( _**H**_ [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] )) _d_ G _[d]_ _t_ [=][ (][V][,][ {][(] _[i]_ [,] _[ j]_ [)][ ∈E|] _**[M]**_ _[msg]_ _di j_ [�] [0][}][)] (1 − _p_ ) _deg_ _i_
Dropout _**M**_ _d_ = _**M**_ _feat_ _d_ ⊙ _**A**_ [˜] ( _**H**_ [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ) _d_ G _[d]_ _t_ [=][ (][V][,][ {][(] _[i]_ [,] _[ j]_ [)][ ∈E|] _**[M]**_ _[ feat]_ _di_ [�] [0][,] _**[ M]**_ _[ feat]_ _d j_ [�] [0][}][) (1][ −] _[p]_ [)] [2] _[deg]_ _[i]_


4 Experiments


To validate our theoretical analysis, we conducted extensive experiments on a variety of datasets,
considering both node-level and graph-level tasks. We implemented dropout technique on several
popular GNN architectures: GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017),
GAT (Veliˇckovi´c et al., 2018), and GatedGCN (Bresson & Laurent, 2017). For each model, we
compared the performance with and without dropout. Our code is available at `[https://github.](https://github.com/LUOyk1999/dropout-theory)`
`[com/LUOyk1999/dropout-theory](https://github.com/LUOyk1999/dropout-theory)` .


7


Published as a conference paper at ICLR 2025


Table 2: Node classification results (%). The baseline results are taken from Deng et al. (2024); Wu
et al. (2023). The top **1** **[st]**, **2** **[nd]** and **3** **[rd]** results are highlighted. ”dp” denotes dropout.

|Col1|Cora CiteSeer PubMed Computer Photo CS Physics WikiCS ogbn-arxiv ogbn-products|
|---|---|
|# nodes<br># edges<br>Metric|2,708<br>3,327<br>19,717<br>13,752<br>7,650<br>18,333<br>34,493<br>11,701<br>169,343<br>2,449,029<br>5,278<br>4,732<br>44,324<br>245,861<br>119,081<br>81,894<br>247,962<br>216,123<br>1,166,243<br>61,859,140<br>Accuracy↑Accuracy↑Accuracy↑Accuracy↑Accuracy↑Accuracy↑Accuracy↑Accuracy↑Accuracy↑<br>Accuracy↑|
|GCNII<br>GPRGNN<br>APPNP<br>tGNN|**85.19** ± 0.26<br>**73.20** ± 0.83<br>80.32 ± 0.44<br>91.04 ± 0.41<br>94.30 ± 0.20<br>92.22 ± 0.14<br>95.97 ± 0.11<br>78.68 ± 0.55<br>72.74 ± 0.31<br>79.42 ± 0.36<br>83.17 ± 0.78<br>71.86 ± 0.67<br>79.75 ± 0.38<br>89.32 ± 0.29<br>94.49 ± 0.14<br>95.13 ± 0.09<br>96.85 ± 0.08<br>78.12 ± 0.23<br>71.10 ± 0.12<br>79.76 ± 0.59<br>83.32 ± 0.55<br>71.78 ± 0.46<br>80.14 ± 0.22<br>90.18 ± 0.17<br>94.32 ± 0.14<br>94.49 ± 0.07<br>96.54 ± 0.07<br>78.87 ± 0.11<br>72.34 ± 0.24<br>78.84 ± 0.09<br>82.97 ± 0.68<br>71.74 ± 0.49<br>**80.67** ± 0.34<br>83.40 ± 1.33<br>89.92 ± 0.72<br>92.85 ± 0.48<br>96.24 ± 0.24<br>71.49 ± 1.05<br>72.88 ± 0.26<br>81.79 ± 0.54|
|GraphGPS<br>NAGphormer<br>Exphormer<br>GOAT<br>NodeFormer<br>SGFormer<br>Polynormer|82.84 ± 1.03<br>72.73 ± 1.23<br>79.94 ± 0.26<br>91.19 ± 0.54<br>95.06 ± 0.13<br>93.93 ± 0.12<br>97.12 ± 0.19<br>78.66 ± 0.49<br>70.97 ± 0.41<br>OOM<br>82.12 ± 1.18<br>71.47 ± 1.30<br>79.73 ± 0.28<br>91.22 ± 0.14<br>95.49 ± 0.11<br>**95.75** ± 0.09<br>**97.34** ± 0.03<br>77.16 ± 0.72<br>70.13 ± 0.55<br>73.55 ± 0.21<br>82.77 ± 1.38<br>71.63 ± 1.19<br>79.46 ± 0.35<br>91.47 ± 0.17<br>95.35 ± 0.22<br>94.93 ± 0.01<br>96.89 ± 0.09<br>78.54 ± 0.49<br>72.44 ± 0.28<br>OOM<br>83.18 ± 1.27<br>71.99 ± 1.26<br>79.13 ± 0.38<br>90.96 ± 0.90<br>92.96 ± 1.48<br>94.21 ± 0.38<br>96.24 ± 0.24<br>77.00 ± 0.77<br>72.41 ± 0.40<br>**82.00** ± 0.43<br>82.20 ± 0.90<br>**72.50** ± 1.10<br>79.90 ± 1.00<br>86.98 ± 0.62<br>93.46 ± 0.35<br>95.64 ± 0.22<br>96.45 ± 0.28<br>74.73 ± 0.94<br>59.90 ± 0.42<br>73.96 ± 0.30<br>**84.50** ± 0.80<br>72.60 ± 0.20<br>80.30 ± 0.60<br>92.42 ± 0.66<br>**95.58** ± 0.36<br>**95.71** ± 0.24<br>96.75 ± 0.26<br>80.05 ± 0.46<br>72.63 ± 0.13<br>81.54 ± 0.43<br>83.25 ± 0.93<br>72.31 ± 0.78<br>79.24 ± 0.43<br>**93.68** ± 0.21<br>**96.46** ± 0.26<br>95.53 ± 0.16<br>**97.27** ± 0.08<br>80.10 ± 0.67<br>**73.46** ± 0.16<br>**83.82** ± 0.11|
|GCN<br>Dirichlet energy|**85.22** ± 0.66<br>**73.24** ± 0.63<br>**81.08** ± 1.16<br>**93.15** ± 0.34<br>95.03 ± 0.24<br>94.41 ± 0.13<br>97.07 ± 0.04<br>**80.14** ± 0.52<br>**73.13** ± 0.27<br>81.87 ± 0.41<br>74.671<br>9.934<br>4.452<br>8.020<br>3.765<br>20.241<br>8.966<br>6.109<br>8.021<br>7.771|
|GCN w/o dp<br>Dirichlet energy|83.18 ± 1.22<br>70.48 ± 0.45<br>79.40 ± 1.02<br>90.60 ± 0.84<br>94.10 ± 0.15<br>94.30 ± 0.22<br>96.92 ± 0.05<br>77.61 ± 1.34<br>72.05 ± 0.23<br>77.50 ± 0.37<br>2.951<br>0.170<br>0.247<br>0.592<br>1.793<br>3.980<br>0.318<br>1.592<br>1.231<br>1.745|
|GCN w/o BN|84.97 ± 0.73<br>72.97 ± 0.86<br>80.94 ± 0.87<br>92.39 ± 0.18<br>94.38 ± 0.13<br>93.46 ± 0.24<br>96.76 ± 0.06<br>79.00 ± 0.48<br>71.93 ± 0.18<br>79.37 ± 0.42|
|SAGE<br>SAGE w/o dp<br>SAGE w/o BN|84.14 ± 0.63<br>71.62 ± 0.29<br>77.86 ± 0.79<br>92.65 ± 0.21<br>**95.71** ± 0.20<br>**95.90** ± 0.09<br>**97.20** ± 0.10<br>**80.29** ± 0.97<br>72.72 ± 0.13<br>**82.69** ± 0.28<br>83.06 ± 0.80<br>69.68 ± 0.82<br>76.40 ± 1.48<br>90.17 ± 0.60<br>94.90 ± 0.17<br>95.80 ± 0.08<br>97.06 ± 0.06<br>78.84 ± 1.17<br>71.37 ± 0.31<br>79.82 ± 0.22<br>83.89 ± 0.67<br>71.39 ± 0.75<br>77.26 ± 1.02<br>92.54 ± 0.24<br>95.51 ± 0.23<br>94.87 ± 0.15<br>97.03 ± 0.03<br>79.50 ± 0.93<br>71.52 ± 0.17<br>80.91 ± 0.35|
|GAT<br>GAT w/o dp<br>GAT w/o BN|83.92 ± 1.29<br>72.00 ± 0.91<br>**80.48** ± 0.99<br>**93.47** ± 0.27<br>95.53 ± 0.16<br>94.49 ± 0.17<br>96.73 ± 0.10<br>**80.21** ± 0.68<br>**72.83** ± 0.19<br>80.05 ± 0.34<br>82.58 ± 1.47<br>71.08 ± 0.42<br>79.28 ± 0.58<br>92.94 ± 0.30<br>93.88 ± 0.16<br>94.30 ± 0.14<br>96.42 ± 0.08<br>78.67 ± 0.40<br>71.52 ± 0.41<br>77.87 ± 0.25<br>83.76 ± 1.32<br>71.82 ± 0.83<br>80.43 ± 1.03<br>92.16 ± 0.26<br>95.05 ± 0.49<br>93.33 ± 0.26<br>96.57 ± 0.20<br>79.49 ± 0.62<br>71.68 ± 0.36<br>78.21 ± 0.32|



4.1 Datasets and Setup


**Datasets.** For node-level tasks, we used 10 datasets: Cora, CiteSeer, PubMed (Sen et al., 2008),
ogbn-arxiv, ogbn-products (Hu et al., 2020), Amazon-Computer, Amazon-Photo, Coauthor-CS,
Coauthor-Physics (Shchur et al., 2018), and WikiCS (Mernyei & Cangea, 2020). Cora, CiteSeer,
and PubMed are citation networks, evaluated using the semi-supervised setting and data splits from
Kipf & Welling (2017). Computer and Photo (Shchur et al., 2018) are co-purchase networks. CS
and Physics (Shchur et al., 2018) are co-authorship networks. We used the standard 60%/20%/20%
training/validation/test splits and accuracy as the evaluation metric (Chen et al., 2022; Shirzad et al.,
2023; Deng et al., 2024). For WikiCS, we adopted the official splits and metrics (Mernyei & Cangea,
2020). For large-scale graphs, we included ogbn-arxiv and ogbn-products with 0.16M to 2.4M
nodes, using OGB’s standard evaluation settings (Hu et al., 2020).


For graph-level tasks, we used MNIST, CIFAR10 (Dwivedi et al., 2023), and two Peptides datasets
(functional and structural) (Dwivedi et al., 2022). MNIST and CIFAR10 are graph versions of their
image classification counterparts, constructed using 8-nearest neighbor graphs of SLIC superpixels.
We follow all evaluation protocols suggested by Dwivedi et al. (2023). Peptides-func involves classifying graphs into 10 functional classes, while Peptides-struct regresses 11 structural properties.
All evaluations followed the protocols in (Dwivedi et al., 2022).


**Baselines.** Our main focus lies on the following prevalent GNNs and transformer models from
Polynormer (Deng et al., 2024): GCN (Kipf & Welling, 2017), SAGE (Hamilton et al., 2017), GAT
Veliˇckovi´c et al. (2018), GCNII (Chen et al., 2020), (Veliˇckovi´c et al., 2018), APPNP (Gasteiger
et al., 2018), GPRGNN (Chien et al., 2020), SGFormer (Wu et al., 2023), Polynormer (Deng et al.,
2024), GOAT (Kong et al., 2023), NodeFormer (Wu et al., 2022), NAGphormer (Chen et al., 2022),
GTDwivedi & Bresson (2020), SAN Kreuzer et al. (2021), MGT Ngo et al. (2023), DRew Gutteridge
et al. (2023), Graph-MLPMixer He et al. (2023), GRIT Ma et al. (2023), GraphGPS (Ramp´aˇsek
et al., 2022), Exphormer (Shirzad et al., 2023), CKGCN (Ma et al., 2024), GRED (Ding et al.,
2024), Graph Mamba Behrouz & Hashemi (2024). We report the performance results of baselines
primarily from (Deng et al., 2024), with the remaining obtained from their respective original papers
or official leaderboards whenever possible, as those results are obtained by well-tuned models.


**Experimental Setup.** We implemented all models using the PyTorch Geometric library (Fey &
Lenssen, 2019). The experiments are conducted on a single workstation with 8 RTX 3090 GPUs.
For node-level tasks, we adhered to the training protocols specified in (Deng et al., 2024; Luo et al.,


8


Published as a conference paper at ICLR 2025


Figure 4: Effect of dropout on feature F-norm, average pair distance, and Dirichlet energy.


2024b;a), employing BN and adjusting the dropout rate between 0.1 and 0.7. In graph-level tasks,
we adopted the settings from (T¨onshoff et al., 2023; Luo et al., 2025), utilizing BN with a consistent
dropout rate of 0.2. All experiments were run with 5 different random seeds, and we report the mean
accuracy and standard deviation. To ensure generalizability, we used Dirichlet energy (Cai & Wang,
2020) as an oversmoothing metric, which is proportional to our feature energy.


4.2 Node-level Classification Results


The node-level classification results in Table 2 not only align with our theoretical predictions but
also showcase the remarkable effectiveness of dropout. Notably, GCN with dropout and batch normalization outperforms state-of-the-art methods on several benchmarks, including Cora, CiteSeer,
and PubMed. This superior performance underscores the practical significance of our theoretical
insights. Consistently across all datasets, models employing dropout outperform their counterparts
without it, validating our analysis that dropout provides beneficial regularization in GNNs, distinct
from its effects in standard neural networks. The varying levels of improvement observed across
different datasets support our theory of degree-dependent dropout effects that adapt to the graph
structure. Furthermore, the consistent increase in Dirichlet energy when using dropout provides empirical evidence for our theoretical insight into dropout’s crucial role in mitigating oversmoothing in
GCNs, particularly evident in larger graphs. The complementary roles of dropout and batch normalization are demonstrated by the performance drop when either is removed, supporting our analysis
of their synergistic interaction in GCNs.


4.3 Graph-level Classification Results


Our graph-level classification results, presented in Tables 3 and 4, further validate the broad applicability of our theoretical framework. First, compared to recent SOTA models, we observe that simply
tuning dropout enables GNNs to achieve SOTA performance on three datasets and is competitive
with the best single-model results on the remaining dataset. Second, the significant accuracy improvements on graph-level tasks such as Peptides-func and CIFAR10 highlight that our insights extend beyond node classification. The varying degrees of improvement across different graph datasets
are consistent with our theory that dropout provides adaptive regularization tailored to graph properties. Third, the consistent increase in Dirichlet energy when using dropout supports our theoretical
analysis of dropout’s role in preserving feature diversity.


These results robustly validate our theory, showing that dropout in GCNs produces dimensionspecific stochastic sub-graphs, has degree-dependent effects, mitigates oversmoothing, and offers
topology-aware regularization. Combined with batch normalization, dropout enhances GCN performance on graph-level tasks, affirming the relevance and utility of our framework and suggesting
directions for improving GNN architectures.


4.4 Mitigating Oversmoothing Rather Than Co-adaptation


In traditional neural networks, dropout primarily prevents co-adaptation of neurons. However, our
theoretical framework suggests that dropout in GCNs serves a fundamentally different purpose: mitigating oversmoothing rather than preventing co-adaptation. To validate this hypothesis, we examined how dropout affects weight matrices in a 2-layer GCN, focusing specifically on spectral norm
changes (see Appendix A.5). We further analyzed three key metrics to quantify dropout’s influence
on feature representations, as shown in Figure 4. The left panel of Figure 4 demonstrates that the


9


Published as a conference paper at ICLR 2025


Table 3: Graph classification results on two peptide datasets from LRGB (Dwivedi et al., 2022).



Table 4: Graph classification results on two image datasets from (Dwivedi et al., 2023).


|Model|Peptides-func Peptides-struct|
|---|---|
|# graphs<br>Avg. # nodes<br>Avg. # edges<br>Metric|15,535<br>15,535<br>150.9<br>150.9<br>307.3<br>307.3<br>AP ↑<br>MAE ↓|
|GT<br>SAN+RWSE<br>GraphGPS<br>MGT+WavePE<br>DRew<br>Exphormer<br>Graph-MLPMixer<br>GRIT<br>CKGCN<br>GRED<br>Graph Mamba|0.6326 ± 0.0126<br>0.2529 ± 0.0016<br>0.6439 ± 0.0075<br>0.2545 ± 0.0012<br>0.6535 ± 0.0041<br>0.2500 ± 0.0012<br>0.6817 ± 0.0064<br>0.2453 ± 0.0025<br>0.7150 ± 0.0044<br>0.2536 ± 0.0015<br>0.6527 ± 0.0043<br>0.2481 ± 0.0007<br>0.6970 ± 0.0080<br>0.2475 ± 0.0015<br>0.6988 ± 0.0082<br>0.2460 ± 0.0012<br>0.6952 ± 0.0068<br>0.2477 ± 0.0019<br>0.7085 ± 0.0027<br>0.2503 ± 0.0019<br>0.6972 ± 0.0100<br>0.2477 ± 0.0019|
|GCN<br>Dirichlet energy|0.7015 ± 0.0021<br>**0.2437** ± 0.0012<br>9.649<br>6.121|
|GCN w/o dp<br>Dirichlet energy|0.6484 ± 0.0034<br>0.2541 ± 0.0026<br>6.488<br>3.725|


|Model|MNIST CIFAR10|
|---|---|
|# graphs<br>Avg. # nodes<br>Avg. # edges<br>Metric|70,000<br>60,000<br>70.6<br>117.6<br>564.5<br>941.1<br>Accuracy ↑<br>Accuracy ↑|
|GT<br>SAN+RWSE<br>GraphGPS<br>MGT+WavePE<br>DRew<br>Exphormer<br>Graph-MLPMixer<br>GRIT<br>CKGCN<br>GRED<br>Graph Mamba|90.831 ± 0.161<br>59.753 ± 0.293<br>-<br>-<br>98.051 ± 0.126<br>72.298 ± 0.356<br>-<br>-<br>-<br>-<br>98.550 ± 0.039<br>74.696 ± 0.125<br>97.422 ± 0.110<br>73.961 ± 0.330<br>98.108 ± 0.111<br>76.468 ± 0.881<br>98.423 ± 0.155<br>72.785 ± 0.436<br>98.383 ± 0.012<br>76.853 ± 0.185<br>98.392 ± 0.183<br>74.563 ± 0.379|
|GatedGCN<br>Dirichlet energy|**98.684** ± 0.137<br>**76.931** ± 0.367<br>1.119<br>1.541|
|GatedGCN w/o dp<br>Dirichlet energy|98.235 ± 0.136<br>71.384 ± 0.397<br>0.987<br>0.845|



Frobenius norm of features remains relatively stable regardless of dropout application, indicating
that dropout does not uniformly scale all features. The middle panel reveals that dropout consistently doubles the average pairwise distance between nodes, helping maintain distinct node representations. Most significantly, the right panel shows that dropout substantially increases Dirichlet
energy. This dramatic rise in Dirichlet energy, compared to the modest changes in Frobenius norm
and pairwise distances, provides compelling evidence that dropout enhances discriminative power
between connected nodes, explaining its effectiveness in preventing oversmoothing rather than simply reducing co-adaptation.


4.5 Comparison with Dropout Variants


To further explore the practical impact of these different regularization techniques, we conducted
hyperparameter tuning for DropEdge, DropNode, and DropMessage on the Cora, Citeseer, and
Pubmed datasets. The results, summarized in Table 5, demonstrate that while these methods yield
comparable performance, traditional dropout generally performs best.


Table 5: Experimental results of different regularization methods on Cora, Citeseer, and PubMed.

|Col1|Cora (GCN) CiteSeer (GCN) PubMed (GCN) Cora (SAGE) CiteSeer (SAGE) PubMed (SAGE) Cora (GAT) CiteSeer (GAT) PubMed (GAT)|
|---|---|
|GNN<br>GNN+Dropout<br>GNN+DropEdge<br>GNN+DropNode<br>GNN+DropMessage|83.18 ± 1.22<br>70.48 ± 0.45<br>79.40 ± 1.02<br>83.06 ± 0.80<br>69.68 ± 0.82<br>76.40 ± 1.48<br>82.58 ± 1.47<br>71.08 ± 0.42<br>79.28 ± 0.58<br>**85.22** ±** 0.66**<br>**73.24** ±** 0.63**<br>**81.08** ±** 1.16**<br>**84.14** ±** 0.63**<br>71.62 ± 0.29<br>77.86 ± 0.79<br>**83.92** ±** 1.29**<br>**72.00** ±** 0.91**<br>**80.48** ±** 0.99**<br>84.88 ± 0.68<br>72.96 ± 0.38<br>80.42 ± 1.15<br>83.10 ± 0.51<br>71.72 ± 0.92<br>77.88 ± 1.31<br>83.44 ± 0.78<br>71.60 ± 1.14<br>79.82 ± 0.68<br>84.92 ± 0.52<br>73.08 ± 0.39<br>80.60 ± 0.49<br>83.42 ± 0.58<br>**71.92** ±** 0.65**<br>78.06 ± 1.09<br>83.80 ± 0.97<br>71.30 ± 0.87<br>79.50 ± 0.68<br>84.78 ± 0.58<br>73.12 ± 1.19<br>80.92 ± 0.88<br>83.18 ± 0.62<br>71.22 ± 1.34<br>**78.20** ±** 0.80**<br>83.46 ± 1.06<br>71.38 ± 1.12<br>79.36 ± 1.22|



5 Conclusions


Our comprehensive theoretical analysis of dropout in GCNs has unveiled complex interactions between regularization, graph structure, and model performance that challenge traditional understanding. These insights not only deepen our understanding of how dropout functions in graph-structured
data but also open new avenues for research and development in graph representation learning. Our
findings suggest the need to reimagine regularization techniques for graph-based models, explore
adaptive and structure-aware dropout strategies, and carefully balance local and global information
in GCN architectures. Furthermore, the observed synergies between dropout and batch normalization point towards more holistic approaches to regularization in GNNs. As we move forward, this
work lays a foundation for developing more robust and effective graph learning algorithms, with
potential applications in dynamic graphs, large-scale graph sampling, and adversarial robustness.
Ultimately, this research contributes to bridging the gap between the empirical success of GNNs and
their theoretical foundations, paving the way for designing graph learning models.


10


Published as a conference paper at ICLR 2025


Acknowledgments


Hao Zhu was supported by the Science Digital Program in Commonwealth Scientific and Industrial Research Organization (CSIRO). Yuankai Luo received support from National Key R&D Program of China (2021YFB3500700), NSFC Grant 62172026, National Social Science Fund of China
22&ZD153, the Fundamental Research Funds for the Central Universities, State Key Laboratory of
Complex & Critical Software Environment (SKLCCSE), and the HK PolyU Grant P0051029.


References


Alessandro Achille and Stefano Soatto. Information dropout: Learning optimal representations
through noisy computation. _IEEE transactions on pattern analysis and machine intelligence_, 40
(12):2897–2905, 2018.


Pierre Baldi and Peter J Sadowski. Understanding dropout. _Advances in neural information pro-_
_cessing systems_, 26, 2013.


Ali Behrouz and Farnoosh Hashemi. Graph mamba: Towards learning on graphs with state space
models. In _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and_
_Data Mining_, pp. 119–130, 2024.


Xavier Bresson and Thomas Laurent. Residual gated graph convnets. _arXiv preprint_
_arXiv:1711.07553_, 2017.


Chen Cai and Yusu Wang. A note on over-smoothing for graph neural networks. _arXiv preprint_
_arXiv:2006.13318_, 2020.


Jinsong Chen, Kaiyuan Gao, Gaichao Li, and Kun He. Nagphormer: A tokenized graph transformer
for node classification in large graphs. In _The Eleventh International Conference on Learning_
_Representations_, 2022.


Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph
convolutional networks. In _International conference on machine learning_, pp. 1725–1735. PMLR,
2020.


Zhengdao Chen, Soledad Villar, Lei Chen, and Joan Bruna. On the equivalence between graph
isomorphism testing and function approximation with gnns. _Advances in neural information_
_processing systems_, 32, 2019.


Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank
graph neural network. In _International Conference on Learning Representations_, 2020.


Weilin Cong, Morteza Ramezani, and Mehrdad Mahdavi. On provable benefits of depth in training
graph convolutional networks. _Advances in Neural Information Processing Systems_, 34:9936–
9949, 2021.


Nima Dehmamy, Albert-L´aszl´o Barab´asi, and Rose Yu. Understanding the representation power of
graph neural networks in learning graph topology. _Advances in Neural Information Processing_
_Systems_, 32, 2019.


Chenhui Deng, Zichao Yue, and Zhiru Zhang. Polynormer: Polynomial-expressive graph transformer in linear time. _arXiv preprint arXiv:2403.01232_, 2024.


Yuhui Ding, Antonio Orvieto, Bobby He, and Thomas Hofmann. Recurrent distance filtering for
graph representation learning. In _Forty-first International Conference on Machine Learning_,
2024.


Simon S Du, Kangcheng Hou, Russ R Salakhutdinov, Barnabas Poczos, Ruosong Wang, and Keyulu
Xu. Graph neural tangent kernel: Fusing graph neural networks with graph kernels. _Advances in_
_neural information processing systems_, 32, 2019.


Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs.
_arXiv preprint arXiv:2012.09699_, 2020.


11


Published as a conference paper at ICLR 2025


Vijay Prakash Dwivedi, Ladislav Ramp´aˇsek, Mikhail Galkin, Ali Parviz, Guy Wolf, Anh Tuan Luu,
and Dominique Beaini. Long range graph benchmark. _arXiv preprint arXiv:2206.08164_, 2022.


Vijay Prakash Dwivedi, Chaitanya K Joshi, Anh Tuan Luu, Thomas Laurent, Yoshua Bengio, and
Xavier Bresson. Benchmarking graph neural networks. _Journal of Machine Learning Research_,
24(43):1–48, 2023.


Pascal Esser, Leena Chennuru Vankadara, and Debarghya Ghoshdastidar. Learning theory can
(sometimes) explain generalisation in graph neural networks. _Advances in Neural Information_
_Processing Systems_, 34:27043–27056, 2021.


Taoran Fang, Zhiqing Xiao, Chunping Wang, Jiarong Xu, Xuan Yang, and Yang Yang. Dropmessage: Unifying random dropping for graph neural networks. In _Proceedings of the AAAI Confer-_
_ence on Artificial Intelligence_, volume 37, pp. 4267–4275, 2023.


Jiarui Feng, Yixin Chen, Fuhai Li, Anindya Sarkar, and Muhan Zhang. How powerful are k-hop
message passing graph neural networks. _Advances in Neural Information Processing Systems_, 35:
4776–4790, 2022.


Wenzheng Feng, Jie Zhang, Yuxiao Dong, Yu Han, Huanbo Luan, Qian Xu, Qiang Yang, Evgeny
Kharlamov, and Jie Tang. Graph random neural networks for semi-supervised learning on graphs.
_Advances in neural information processing systems_, 33:22092–22103, 2020.


Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric.
_arXiv preprint arXiv:1903.02428_, 2019.


Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model
uncertainty in deep learning. In _international conference on machine learning_, pp. 1050–1059.
PMLR, 2016.


Hongyang Gao and Shuiwang Ji. Graph u-nets. In _international conference on machine learning_,
pp. 2083–2092. PMLR, 2019.


Vikas Garg, Stefanie Jegelka, and Tommi Jaakkola. Generalization and representational limits
of graph neural networks. In _International Conference on Machine Learning_, pp. 3419–3430.
PMLR, 2020.


Johannes Gasteiger, Aleksandar Bojchevski, and Stephan G¨unnemann. Predict then propagate:
Graph neural networks meet personalized pagerank. _arXiv preprint arXiv:1810.05997_, 2018.


Johannes Gasteiger, Stefan Weißenberger, and Stephan G¨unnemann. Diffusion improves graph
learning. _Advances in neural information processing systems_, 32, 2019.


Benjamin Gutteridge, Xiaowen Dong, Michael M Bronstein, and Francesco Di Giovanni. Drew: Dynamically rewired message passing with delay. In _International Conference on Machine Learning_,
pp. 12252–12267. PMLR, 2023.


Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs.
_Advances in neural information processing systems_, 30, 2017.


Xiaoxin He, Bryan Hooi, Thomas Laurent, Adam Perold, Yann LeCun, and Xavier Bresson. A
generalization of vit/mlp-mixer to graphs. In _International Conference on Machine Learning_, pp.
12724–12745. PMLR, 2023.


Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov.
Improving neural networks by preventing co-adaptation of feature detectors. arxiv 2012. _arXiv_
_preprint arXiv:1207.0580_, 2012.


Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta,
and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. _Advances_
_in neural information processing systems_, 33:22118–22133, 2020.


Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional
networks. In _International Conference on Learning Representations_, 2017. URL `[https:](https://openreview.net/forum?id=SJU4ayYgl)`
`[//openreview.net/forum?id=SJU4ayYgl](https://openreview.net/forum?id=SJU4ayYgl)` .


12


Published as a conference paper at ICLR 2025


Kezhi Kong, Jiuhai Chen, John Kirchenbauer, Renkun Ni, C. Bayan Bruss, and Tom Goldstein. GOAT: A global transformer on large-scale graphs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), _Pro-_
_ceedings of the 40th International Conference on Machine Learning_, volume 202 of _Proceed-_
_ings of Machine Learning Research_, pp. 17375–17390. PMLR, 23–29 Jul 2023. URL `[https:](https://proceedings.mlr.press/v202/kong23a.html)`
`[//proceedings.mlr.press/v202/kong23a.html](https://proceedings.mlr.press/v202/kong23a.html)` .


Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent L´etourneau, and Prudencio Tossou. Rethinking graph transformers with spectral attention. _Advances in Neural Information Processing_
_Systems_, 34:21618–21629, 2021.


Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. _nature_, 521(7553):436–444,
2015.


Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks
for semi-supervised learning. In _Proceedings of the AAAI conference on artificial intelligence_,
volume 32, 2018.


Renjie Liao, Raquel Urtasun, and Richard Zemel. A pac-bayesian approach to generalization bounds
for graph neural networks. _arXiv preprint arXiv:2012.07690_, 2020.


Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, and Xiang
Zhang. Parameterized explainer for graph neural network. _Advances in neural information pro-_
_cessing systems_, 33:19620–19631, 2020.


Yuankai Luo, Qijiong Liu, Lei Shi, and Xiao-Ming Wu. Structure-aware semantic node identifiers
for learning on graphs. _arXiv preprint arXiv:2405.16435_, 2024a.


Yuankai Luo, Lei Shi, and Xiao-Ming Wu. Classic GNNs are strong baselines: Reassessing GNNs
for node classification. In _The Thirty-eight Conference on Neural Information Processing Sys-_
_tems Datasets and Benchmarks Track_, 2024b. URL `[https://openreview.net/forum?id=](https://openreview.net/forum?id=xkljKdGe4E)`
`[xkljKdGe4E](https://openreview.net/forum?id=xkljKdGe4E)` .


Yuankai Luo, Lei Shi, and Xiao-Ming Wu. Unlocking the potential of classic gnns for graph-level
tasks: Simple architectures meet excellence. _arXiv preprint arXiv:2502.09263_, 2025.


Shaogao Lv. Generalization bounds for graph convolutional neural networks via rademacher complexity. _arXiv preprint arXiv:2102.10234_, 2021.


Liheng Ma, Chen Lin, Derek Lim, Adriana Romero-Soriano, Puneet K Dokania, Mark Coates,
Philip Torr, and Ser-Nam Lim. Graph inductive biases in transformers without message passing.
_arXiv preprint arXiv:2305.17589_, 2023.


Liheng Ma, Soumyasundar Pal, Yitian Zhang, Jiaming Zhou, Yingxue Zhang, and Mark Coates.
Ckgconv: General graph convolution with continuous kernels. _arXiv preprint arXiv:2404.13604_,
2024.


Haggai Maron, Heli Ben-Hamu, Nadav Shamir, and Yaron Lipman. Invariant and equivariant graph
networks. _arXiv preprint arXiv:1812.09902_, 2018.


P´eter Mernyei and C˘at˘alina Cangea. Wiki-cs: A wikipedia-based benchmark for graph neural networks. _arXiv preprint arXiv:2007.02901_, 2020.


Pietro Morerio, Jacopo Cavazza, Riccardo Volpi, Ren´e Vidal, and Vittorio Murino. Curriculum
dropout. In _Proceedings of the IEEE International Conference on Computer Vision_, pp. 3544–
3552, 2017.


Nhat Khang Ngo, Truong Son Hy, and Risi Kondor. Multiresolution graph transformers and wavelet
positional encoding for learning long-range and hierarchical structures. _The Journal of Chemical_
_Physics_, 159(3), 2023.


Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node
classification. _arXiv preprint arXiv:1905.10947_, 2019.


13


Published as a conference paper at ICLR 2025


Ladislav Ramp´aˇsek, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. Recipe for a general, powerful, scalable graph transformer. _arXiv preprint_
_arXiv:2205.12454_, 2022.


Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph
convolutional networks on node classification. In _International Conference on Learning Repre-_
_sentations_, 2020. URL `[https://openreview.net/forum?id=Hkx1qkrKPr](https://openreview.net/forum?id=Hkx1qkrKPr)` .


Franco Scarselli, Ah Chung Tsoi, and Markus Hagenbuchner. The vapnik–chervonenkis dimension
of graph and recursive neural networks. _Neural Networks_, 108:248–259, 2018.


Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad.
Collective classification in network data. _AI magazine_, 29(3):93–93, 2008.


Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan G¨unnemann. Pitfalls
of graph neural network evaluation. _arXiv preprint arXiv:1811.05868_, 2018.


Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J Sutherland, and Ali Kemal
Sinop. Exphormer: Sparse transformers for graphs. _arXiv preprint arXiv:2303.06147_, 2023.


Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.
Dropout: a simple way to prevent neural networks from overfitting. _The journal of machine_
_learning research_, 15(1):1929–1958, 2014.


Jan T¨onshoff, Martin Ritzert, Eran Rosenbluth, and Martin Grohe. Where did the gap go? reassessing the long-range graph benchmark. _arXiv preprint arXiv:2309.00367_, 2023.


Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua
Bengio. Graph attention networks. In _International Conference on Learning Representations_,
2018.


Saurabh Verma and Zhi-Li Zhang. Stability and generalization of graph convolutional neural networks. In _Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Dis-_
_covery_ & _Data Mining_, pp. 1539–1548, 2019.


Minh Vu and My T Thai. Pgm-explainer: Probabilistic graphical model explanations for graph
neural networks. _Advances in neural information processing systems_, 33:12225–12235, 2020.


Stefan Wager, Sida Wang, and Percy S Liang. Dropout training as adaptive regularization. _Advances_
_in neural information processing systems_, 26, 2013.


Li Wan, Matthew Zeiler, Sixin Zhang, Yann Le Cun, and Rob Fergus. Regularization of neural
networks using dropconnect. In _International conference on machine learning_, pp. 1058–1066.
PMLR, 2013.


Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In _International conference on machine learning_, pp.
6861–6871. PMLR, 2019.


Qitian Wu, Wentao Zhao, Zenan Li, David P Wipf, and Junchi Yan. Nodeformer: A scalable graph
structure learning transformer for node classification. _Advances in Neural Information Processing_
_Systems_, 35:27387–27401, 2022.


Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian,
and Junchi Yan. Simplifying and empowering transformers for large-graph representations. In
_Thirty-seventh Conference on Neural Information Processing Systems_, 2023. URL `[https://](https://openreview.net/forum?id=R4xpvDTWkV)`
`[openreview.net/forum?id=R4xpvDTWkV](https://openreview.net/forum?id=R4xpvDTWkV)` .


Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural
networks? In _International Conference on Learning Representations_, 2018.


Han Yang, Kaili Ma, and James Cheng. Rethinking graph regularization for graph neural networks.
In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 35, pp. 4573–4581,
2021.


14


Published as a conference paper at ICLR 2025


Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer:
Generating explanations for graph neural networks. _Advances in neural information processing_
_systems_, 32, 2019.


Hao Yuan, Jiliang Tang, Xia Hu, and Shuiwang Ji. Xgnn: Towards model-level explanations of
graph neural networks. In _Proceedings of the 26th ACM SIGKDD international conference on_
_knowledge discovery_ & _data mining_, pp. 430–438, 2020.


Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, and Shuiwang Ji. On explainability of graph neural
networks via subgraph explorations. In _International conference on machine learning_, pp. 12241–
12252. PMLR, 2021.


Rui-Ray Zhang and Massih-Reza Amini. Generalization bounds for learning under graphdependence: a survey. _Mach. Learn._, 113(7):3929–3959, April 2024. ISSN 0885-6125. doi:
10.1007/s10994-024-06536-9. URL `[https://doi.org/10.1007/s10994-024-06536-9](https://doi.org/10.1007/s10994-024-06536-9)` .


Shuai Zhang, Meng Wang, Sijia Liu, Pin-Yu Chen, and Jinjun Xiong. Fast learning of graph neural
networks with guaranteed generalizability: one-hidden-layer case. In _International Conference_
_on Machine Learning_, pp. 11268–11277. PMLR, 2020.


Lingxiao Zhao and Leman Akoglu. Pairnorm: Tackling oversmoothing in gnns. _arXiv preprint_
_arXiv:1909.12223_, 2019.


A Appendix


A.1 Proof of Theorem 1


_Proof._ Let’s approach this proof:


**Step 1:** For a single feature _j_, the probability that an edge is present in the sub-graph _E_ _t_ [(] _[l]_ [,] _[j]_ [)] is (1− _p_ ) [2],
as both endpoints need to retain this feature.


**Step 2:** The probability that an edge is not present in E [(] _t_ _[l]_ [,] _[j]_ [)] is 1 − (1 − _p_ ) [2] = _p_ (2 − _p_ ).


**Step 3:** For a sub-graph to be identical to the original graph, all edges must be present. The probability of this is: ((1 − _p_ ) [2] ) [|E|] = (1 − _p_ ) [2][|E|] .


**Step 4:** Therefore, the probability that E [(] _t_ _[l]_ [,] _[j]_ [)] is different from the original graph (i.e., unique) is
1 − (1 − _p_ ) [2][|E|] .


**Step 5:** Define an indicator random variable _X_ _j_ for each feature _j_ :

_X_ _j_ = 1 if E [(] _t_ _[l]_ [,] _[j]_ [)] is unique .
�0 otherwise


**Step 6:** We have:
_P_ ( _X_ _j_ = 1) = 1 − (1 − _p_ ) [2][|E|] ][ _P_ ( _X_ _j_ = 0) = (1 − _p_ ) [2][|E|] .


**Step 7:** The expected value of _X_ _j_ is:


E[ _X_ _j_ ] = 1 · _P_ ( _X_ _j_ = 1) + 0 · _P_ ( _X_ _j_ = 0) = 1 − (1 − _p_ ) [2][|E|] .


**Step 8:** The total number of unique sub-graphs is [�] _[d]_ _j_ = _[l]_ 1 _[X]_ _[ j]_ [. By the linearity of expectation:]



_d_ _l_
� E[ _X_ _j_ ] = _d_ _l_ (1 − (1 − _p_ ) [2][|E|] ).

_j_ =1



E[|E [(] _t_ _[l]_ [,] _[ j]_ [)] | _j_ = 1, . . ., _d_ _l_ |] = E[



_d_ _l_
� _X_ _j_ ] =

_j_ =1



This completes the proof. 

15


Published as a conference paper at ICLR 2025


A.2 Proof of Theorem 5


_Proof._ We start with the definition of feature energy:



1
_E_ ( _**H**_ [(] _[l]_ [)] ) =
2|E|


**Step 1:** Taking the expectation:


1
E[ _E_ ( _**H**_ [(] _[l]_ [)] )] =
2|E|


.



� ∥ _**h**_ [(] _i_ _[l]_ [)] [−] _**[h]**_ [(] _j_ _[l]_ [)] [∥] 2 [2]

_i_, _j_ ∈E



� E[∥ _**h**_ [(] _i_ _[l]_ [)] [−] _**[h]**_ [(] _j_ _[l]_ [)] [∥] 2 [2] []][.]

_i_, _j_ ∈E



**Step 2:** Since [�] ( _i_, _j_ )∈E [[][∥] _**[h]**_ _i_ [∥] [2] [ +][ ∥] _**[h]**_ _j_ [∥] [2] []][ =][ 2][ �] _i_ _[deg]_ _i_ [∥] _**[h]**_ _i_ [∥] [2] [:]



1

2|E|



1

� E[∥ _**h**_ [(] _i_ _[l]_ [)] [−] _**[h]**_ [(] _j_ _[l]_ [)] [∥] 2 [2] []][ =] 2|E|

_i_, _j_ ∈E



1 1
_i_ �, _j_ ∈E E[∥ 1 − _p_ _**[M]**_ _i_ [ (] _[l]_ [)] ⊙ _**z**_ _i_ [(] _[l]_ [)] [−] 1 − _p_ _**[M]**_ [ (] _j_ _[l]_ [)] [⊙] _**[z]**_ [(] _j_ _[l]_ [)] [∥] 2 [2] []]



1
= 2|E|(1 − _p_ ) [2] _i_ �, _j_ ∈E E[∥ _**M**_ _i_ [(] _[l]_ [)] ⊙ _**z**_ _i_ [(] _[l]_ [)] [−] _**[M]**_ [ (] _j_ _[l]_ [)] [⊙] _**[z]**_ [(] _j_ _[l]_ [)] [∥] 2 [2] []]



1
= 2|E|(1 − _p_ ) [2] _i_ �, _j_ ∈E [(1 − _p_ )(∥ _**z**_ _i_ [(] _[l]_ [)] [∥] 2 [2] [+][ ∥] _**[z]**_ [(] _j_ _[l]_ [)] [∥] 2 [2] [)][ −] [2(1][ −] _[p]_ [)] [2] [(] _**[z]**_ _i_ [(] _[l]_ [)] [)] [⊤] _**[z]**_ [(] _j_ _[l]_ [)] []]



1 1

=
1 − _p_ |E|


where _**z**_ _i_ = σ( [�] _k_ _**[A]**_ [˜] _ik_ _**[h]**_ [(] _k_ _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ).


**Step 3:** Since _deg_ _i_ ≤ _deg_ _max_ for all _i_ :



_deg_ _i_ ∥ _**z**_ _i_ [(] _[l]_ [)] [∥] 2 [2] [−] [1]

_i_



�



|E| [Tr(] _**[Z]**_ [⊤] _**[AZ]**_ [)]



∥ _**z**_ _i_ ∥ [2] 2 [=] _[ de][g]_ [max]
_i_ |E|



1

|E|



_deg_ _i_ ∥ _**z**_ _i_ ∥ [2] 2 [≤] _[de][g]_ [max]
_i_ |E|



�



|E|



|E| [max] ∥ _**Z**_ ∥ [2] _F_ [.]



�



**Step 4:** By ReLU non-negative homogeneity and submultiplicative property:


|| _**Z**_ [(] _[l]_ [)] || [2] _F_ [≤||] _**[AH]**_ [ ˜] [(] _[l]_ [−][1)] _**[W]**_ [ (] _[l]_ [)] [||] [2] _F_ [≤||] _**[W]**_ [ (] _[l]_ [)] [||] [2] 2 [||] _**[A]**_ [ ˜] [||] [2] 2 [||] _**[H]**_ [(] _[l]_ [−][1)] [||] [2] _F_


**Step 5:** By dropout scaling with probability _p_ :


1
|| _**H**_ [(] _[l]_ [−][1)] || [2] _F_ [=] 1 − _p_ [||] _**[Z]**_ [(] _[l]_ [−][1)] [||] [2] _F_


**Step 6:** By applying steps 4-5 recursively:



1
|| _**Z**_ [(] _[l]_ [)] || [2] _F_ [≤] [(] 1 − _p_ [)] _[l]_ [−][1] [||] _**[A]**_ [ ˜] [||] 2 [2] _[l]_


**Step 7:** Combining all inequalities:



_l_
� || _**W**_ [(] _[i]_ [)] || [2] 2 [||] _**[X]**_ [||] [2] _F_

_i_ =1



1

E[ _E_ ( _**H**_ [(] _[l]_ [)] )] ≤ _[de]_ |E| _[g]_ [max] ( 1 − _p_ [)] _[l]_ [||] _**[A]**_ [ ˜] [||] 2 [2] _[l]_


A.3 Proof of Theorem 6


_Proof._ The proof proceeds in several steps:



_l_
� || _**W**_ [(] _[i]_ [)] || [2] 2 [||] _**[X]**_ [||] [2] _F_

_i_ =1







**Step 1: Dependency Graph.** Let G = (V, E) be the dependency graph where vertices V represent
nodes in the graph, and an edge ( _i_, _j_ ) ∈E exists if nodes _i_ and _j_ are connected through message
passing via _**A**_ [˜] . The graph G is fixed across all layers as it is determined by the structure of _**A**_ [˜] .


16


Published as a conference paper at ICLR 2025


**Step 2: Dropout E** ff **ect as Perturbation.** At layer _l_ with dropout probability _p_ _l_, let _**δ**_ **[(]** _**[l]**_ **[)]** be the
perturbation matrix:


1
_**δ**_ **[(]** _**[l]**_ **[)]** = _M_ [(] _[l]_ [)] ⊙ σ( _**AH**_ [˜] [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ) − σ( _**AH**_ [˜] [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ), (11)
1 − _p_ _l_


where _M_ [(] _[l]_ [)] has elements drawn from Bernoulli(1 − _p_ _l_ ).


**Step 3: Perturbation Propagation.** Let _F_ _l_ ( _**X**_ ) denote the network output with dropout applied up
to layer _l_ . With _L_ σ -Lipschitz activation:



_L_ _l_ =



_L_

( _L_ σ ∥ _**W**_ [(] _[i]_ [)] ∥ 2   - ∥ _**A**_ [˜] ∥ 2 ) (12)

�

_i_ = _l_



By operator norm properties:


∥ _F_ _l_ ( _**X**_ ) − _F_ _l_ −1 ( _**X**_ )∥ _F_ ≤ _L_ _l_ ∥ _**δ**_ **[(]** _**[l]**_ **[)]** ∥ _F_ (13)


**Step 4: Bounding Matrix Perturbation.** Let δ [(] _i j_ _[l]_ [)] [denote the (] _[i]_ [,] _[ j]_ [)-th entry of] _**[ δ]**_ **[(]** _**[l]**_ **[)]** [. By Janson’s]
inequality for dependent variables over G (Zhang & Amini, 2024):



1

E �(δ [(] _i j_ _[l]_ [)] [)] [2] ≤

 _i_, _j_  χ _f_ (G)



� E[(δ [(] _i j_ _[l]_ [)] [)] [2] []] (14)

_i_, _j_



Taking the square root and using the definition of Frobenius norm:



E[∥ _**δ**_ **[(]** _**[l]**_ **[)]** ∥ _F_ ] ≤



~~�~~



1
_F_ []] (15)
χ _f_ (G) [·][ E][[][∥] _**[δ]**_ **[(]** _**[l]**_ **[)]** [∥] [2]



~~_p_~~ _l_
= (16)
~~�~~ (1 − _p_ _l_ )χ _f_ (G) [∥][σ][( ˜] _**[AH]**_ [(] _[l]_ [−][1)] _**[W]**_ [ (] _[l]_ [)] [)][∥] _[F]_


where we use E[( _**M**_ [(] _[l]_ [)] ) [2] ] = E[ _**M**_ [(] _[l]_ [)] ] = 1 − _p_ _l_ .


**Step 5: Loss Stability.** By the Lipschitz property of the loss function:


E[| _L_ ( _F_ _l_ ( _x_ )) − _L_ ( _F_ _l_ −1 ( _x_ ))| _F_ ] ≤ _L_ _loss_     - E[∥ _F_ _l_ ( _x_ ) − _F_ _l_ −1 ( _x_ )∥ _F_ ] (17)


~~_p_~~ _l_
≤ _L_ _loss_            - _L_ _l_            - _L_ σ            - (18)
� (1 − _p_ _l_ )χ _f_ (G) [∥][σ][( ˜] _**[AH]**_ [(] _[l]_ [−][1)] _**[W]**_ [ (] _[l]_ [)] [)][∥] _[F]_


**Step 6: Final Concentration Bound.** Using McDiarmid’s inequality and noting the impact of
message passing through χ _f_ (G), with probability at least 1 − δ:



�



~~~~ _L_ ~~_p_~~ _l_

_L_ _loss_   - _L_ _l_   
 � _l_ =1 � (1 − _p_ _l_ )χ _f_ (G) [∥][σ][( ˜] _**[AH]**_ [(] _[l]_ [−][1)] _**[W]**_ [ (] _[l]_ [)] [)][∥] _[F]_

(19)



E _D_ [ _L_ ( _F_ ( _x_ ))] − E _S_ [ _L_ ( _F_ ( _x_ ))] ≤ _O_







log(1/δ)


_n_



The bound shows that GNNs (χ _f_ (G) > 1 due to message passing) achieve better stability than MLPs
(χ _f_ (G) = 1, no message passing), with the benefit increasing with graph connectivity. 

A.4 Proof of Theorem 7


_Proof._ **Step 1:** Start with feature energy and node representation:



1
_E_ ( _**H**_ [(] _[l]_ [)] ) =
2|E|



� ∥ _**h**_ [(] _i_ _[l]_ [)] [−] _**[h]**_ [(] _j_ _[l]_ [)] [∥] [2]

( _i_, _j_ )∈E



1
_**h**_ [(] _i_ _[l]_ [)] [=] 1 − _p_ _**[M]**_ [ (] _i_ _[l]_ [)] ⊙ _**z**_ _i_ [(] _[l]_ [)]


where _**z**_ _i_ [(] _[l]_ [)] ∈ R _[d]_ _[l]_ and _**z**_ _i_ [(] _[l]_ [)] = σ(BN( [�] _k_ _**[A]**_ [˜] _ik_ _**[h]**_ [(] _k_ _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ))


17


Published as a conference paper at ICLR 2025


**Step 2:** For the BN output before ReLU at layer _l_, for each feature dimension _d_ ∈{1, ..., _d_ _l_ }:

( _**Y**_ [(] _[l]_ [)] ) :, _d_ = BN(( _**AH**_ [˜] [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ) :, _d_ ) = γ _d_ [(] _[l]_ [)] ( _**AH**_ [˜] [(] _[l]_ [−][1)] _**W**_ [(] _[l]_ [)] ) :, _d_ − µ [(] _d_ _[l]_ [)] + β [(] _d_ _[l]_ [)]
~~�~~ (σ [(] _d_ _[l]_ [)] [)] [2] [ +][ ϵ]


**Step 3:** For ReLU activation z = max(0, y) at layer l, for each dimension d:


E[( _z_ [(] _d_ _[l]_ [)] [)] [2] []][ ≥] [Φ][(][β] [(] _d_ _[l]_ [)] [/γ] _d_ [(] _[l]_ [)] [)][ ·][ (][β] [(] _d_ _[l]_ [)] [)] [2]


where Φ is the standard normal CDF.


**Step 4:** Using the BN-induced bound:



∥ _**z**_ _i_ [(] _[l]_ [)] [∥] [2] [ =]


≥



_d_ _l_
�( _z_ [(] _i_ _[l]_ [)] [)] _d_ [2]

_d_ =1


_d_ _l_
� Φ(β [(] _d_ _[l]_ [)] [/γ] _d_ [(] _[l]_ [)] [)][ ·][ (][β] [(] _d_ _[l]_ [)] [)] [2] [ >][ 0]

_d_ =1



**Step 5:** For feature energy with merged terms:



1
_E_ ( _**H**_ [(] _[l]_ [)] ) =
2|E|


1
≥
2|E|


1

=
2|E|



1
( _i_ �, _j_ )∈E [ 1 − _p_ [(][∥] _**[z]**_ _i_ [(] _[l]_ [)] [∥] [2] [ +][ ∥] _**[z]**_ [(] _j_ _[l]_ [)] [∥] [2] [)][ −] [2(] _**[z]**_ _i_ [(] _[l]_ [)] [)] _[T]_ _**[z]**_ [(] _j_ _[l]_ [)] []]


1
( _i_ �, _j_ )∈E [ 1 − _p_ [(][∥] _**[z]**_ _i_ [(] _[l]_ [)] [∥] [2] [ +][ ∥] _**[z]**_ [(] _j_ _[l]_ [)] [∥] [2] [)][ −] [(][∥] _**[z]**_ _i_ [(] _[l]_ [)] [∥] [2] [ +][ ∥] _**[z]**_ [(] _j_ _[l]_ [)] [∥] [2] [)]]


1
( _i_ �, _j_ )∈E ( 1 − _p_ [−] [1)(][∥] _**[z]**_ _i_ [(] _[l]_ [)] [∥] [2] [ +][ ∥] _**[z]**_ [(] _j_ _[l]_ [)] [∥] [2] [)]



_p_ 1
=
1 − _p_ 2|E|


_p_ 1
=
1 − _p_ 2|E|



� _deg_ _i_ ∥ _**z**_ _i_ [(] _[l]_ [)] [∥] [2]

_i_



� (∥ _**z**_ _i_ [(] _[l]_ [)] [∥] [2] [ +][ ∥] _**[z]**_ [(] _j_ _[l]_ [)] [∥] [2] [)]

( _i_, _j_ )∈E



_d_ _l_
� Φ(β [(] _d_ _[l]_ [)] [/γ] _d_ [(] _[l]_ [)] [)][ ·][ (][β] [(] _d_ _[l]_ [)] [)] [2]

_d_ =1



Then with BN bound:



1

≥ _[p]_ [ ·] 1 _[ de]_ − _[g]_ _p_ [min] 2|E| [∥] _**[Z]**_ [(] _[l]_ [)] [∥] [2] _F_


1

_E_ ( _**H**_ [(] _[l]_ [)] ) ≥ _[p]_ [ ·] _[ de][g]_ [min]

1 − _p_ 2|E|




                               

A.5 Effect of Dropout on Max Singular Values of the Weight Matrices


We analyze why dropout leads to larger weight matrices in terms of spectral norm ∥ _**W**_ ∥ 2 . Consider
the gradient update for weights _**W**_ [(2)] between layers:


∂ _L_ ∂ _L_ ∂ _L_
∂ _**W**_ [(2)] [=][ ( ˜] _**[AH]**_ _drop_ [(1)] [)] [⊤] [×] ∂ _**H**_ [(2)] [=][ ( ˜] _**[A]**_ [(] _**[H]**_ [(1)] [ ⊙] _**[M]**_ [ (1)] [)][/][(1][ −] _[p]_ [))] [⊤] [×] ∂ _**H**_ [(2)] (20)


where _p_ is the dropout rate and _M_ [1] is the dropout mask. This leads to weight updates:


∂ _L_ ∂ _L_
∆ _**W**_ [(2)] = −η( _**AH**_ [˜] _drop_ [(1)] [)] [⊤] [×] ∂ _**H**_ [(2)] [=][ −][η][( ˜] _**[A]**_ [(] _**[H]**_ [(1)] [ ⊙] _**[M]**_ [ (1)] [)][/][(1][ −] _[p]_ [))] [⊤] [×] ∂ _**H**_ [(2)] (21)


The 1/(1 − _p_ ) scaling factor in dropout has two key effects: 1) For surviving features (where
_M_ _i j_ [(1)] = 1), the gradient is amplified by 1/(1 − _p_ ). This leads to larger updates for these weights


18


Published as a conference paper at ICLR 2025


during training. 2) During each iteration, different subsets of features survive, but their gradients are
consistently scaled up. Over many iterations, this accumulates to larger weight values despite the
unbiased expectation maintained by dropout. Specifically, with dropout rate _p_ when _p_ = 0.5, surviving gradients are doubled. This amplification effect compounds over training iterations. While
dropout maintains unbiased expected values during forward propagation, the consistent gradient
scaling during backward propagation leads to systematically larger weight magnitudes. Empirically,
we observe that higher dropout rates correlate with larger spectral norms ∥ _**W**_ ∥ [2] 2 [(as shown in Fig-]
ure 5), supporting this theoretical analysis. The increased weight magnitudes directly contribute to
higher feature energy _E_ ( _**H**_ [(2)] ) during inference, as:



1
_E_ ( _**H**_ [(2)] ) =
2|E|



� ∥ _**h**_ [(2)] _i_ − _**h**_ [(2)] _j_ [∥] 2 [2] (22)

( _i_, _j_ )∈E



where larger weights produce more distinctive features between connected nodes, helping mitigate
oversmoothing.


Figure 5: Effect of dropout on max singular values of the weight matrices.


A.6 Empirical Validation of Theoretical Properties


In this section, we provide empirical evidence supporting the theoretical properties derived in Section 3.


**Dimension-Specific Stochastic Sub-graphs.** Figure 6 shows how varying dropout rates impact the
number of edges E _t_ in stochastic sub-graphs of a 2-layer GCN, defined by Equation 3, across the
Cora and Citeseer datasets. We observe that higher dropout rates correlate with fewer edges in these
sub-graphs. This variation demonstrates dropout’s role in GCNs as a form of structural regularization, where dimension-specific stochastic sub-graphs are generated. Each feature dimension samples
a different sub-graph from the original graph at each iteration. This mechanism provides a rich set
of structural variations during training, potentially enhancing the model’s ability to capture diverse
graph patterns. Figures 7 & 8 illustrate the behavior of active features along paths of length 1 and 2
within a 2-layer GCN equipped with 16 hidden dimensions, across varying dropout rates. Notably,
at a dropout rate of 0.6, the average number of active features approaches zero. This characteristic
also underscores the importance of multidimensional feature spaces in ensuring robust information
transmission under feature dropout.


**Degree-Dependent Nature of Dropout E** ff **ects.** Figure 9 demonstrates that dropout affects the
effective degree of nodes. Figure 10 illustrates that the CV decreases as node degree increases.
This degree-dependent effect distinguishes dropout in GCNs from its application in standard neural networks and suggests that the optimal dropout strategy for GCNs may need to consider the
graph structure explicitly. Figure 11 presents empirical evidence supporting Theorem 3 (DegreeDependent Dropout Effect), which predicts that high-degree nodes experience relatively less variation in their effective degree due to dropout. The figure shows classification accuracy on the Cora


19


Published as a conference paper at ICLR 2025


Figure 6: Sub-graph size. Figure 7: Active path on Cora. Figure 8: Active path on Citeseer.


Figure 10: Effective CV vs deFigure 9: Effective degree. Figure 11: Accuracy on Cora.
gree.


dataset broken down by node degree, demonstrating that nodes with higher degrees consistently
achieve better performance. This aligns with our theoretical finding that high-degree nodes maintain
more stable representations under dropout, as their effective degree has lower coefficient of variation.
The observed pattern confirms that dropout naturally provides adaptive regularization that adjusts to
the local graph structure, with stronger stabilizing effects for topologically important nodes.


20


