Published as a conference paper at ICLR 2025

# - W HEN GNN S MEET SYMMETRY IN ILP S : AN ORBIT

## BASED FEATURE AUGMENTATION APPROACH


Qian Chen [1,2], Lei Li [1,2], Qian Li [2], Jianghua Wu [2], Akang Wang [2,3,*], Ruoyu Sun [2,3], Xiaodong Luo [2],
Tsung-Hui Chang [2,4], and Qingjiang Shi [2,5]


1 School of Science and Engineering, The Chinese University of Hong Kong, Shenzhen, China
2 Shenzhen International Center for Industrial and Applied Mathematics, Shenzhen Research
Institute of Big Data, China
3 School of Data Science, The Chinese University of Hong Kong, Shenzhen, China
4 School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen, China
5 School of Software Engineering, Tongji University, Shanghai, China


A BSTRACT


A common characteristic in integer linear programs (ILPs) is symmetry, allowing variables to be permuted without altering the underlying problem structure.
Recently, GNNs have emerged as a promising approach for solving ILPs. However, a significant challenge arises when applying GNNs to ILPs with symmetry:
classic GNN architectures struggle to differentiate between symmetric variables,
which limits their predictive accuracy. In this work, we investigate the properties
of permutation equivariance and invariance in GNNs, particularly in relation to
the inherent symmetry of ILP formulations. We reveal that the interaction between
these two factors contributes to the difficulty of distinguishing between symmetric
variables. To address this challenge, we explore the potential of feature augmentation and propose several guiding principles for constructing augmented features.
Building on these principles, we develop an orbit-based augmentation scheme that
first groups symmetric variables and then samples augmented features for each
group from a discrete uniform distribution. Empirical results demonstrate that our
proposed approach significantly enhances both training efficiency and predictive
performance.


1 I NTRODUCTION


**Integer Linear Programs (ILPs)** are fundamental optimization problems characterized by a linear
objective function and linear constraints, where the decision variables are restricted to integer values. These problems play a critical role in various fields, including operations research, computer
science, and engineering (Pochet & Wolsey, 2006; Liu & Fan, 2018; Watson & Woodruff, 2011;
Luathep et al., 2011; Sch¨obel, 2001). In practice, many ILPs exhibit a structural property known as
**symmetry**, where certain permutations of decision variables leave both the problem structure and
the solution set unchanged (Margot, 2003). For example, in the widely used collection of real-world
ILPs, MIPLIB (Gleixner et al., 2021), approximately 32% of its instances display certain symmetries.


**Classic methods** for solving ILPs, such as branch-and-bound and branch-and-cut (Boyd & Mattingley, 2007; Morrison et al., 2016), systematically explore the solution space by breaking it down
into smaller sub-problems and eliminating regions that do not contain optimal solutions. However,
the presence of symmetry in these problems can result in redundant exploration of equivalent solutions, which hampers efficiency. To address this, several approaches have been developed. Margot
(2002) prunes the enumeration tree in the branch and bound algorithm; Ostrowski et al. (2008; 2011)
propose branching strategies based on orbits; Puget (2003; 2006) enhance problem formulations by
introducing symmetry-breaking constraints. A more comprehensive survey of related research is


  - Corresponding author: Akang Wang _<_ wangakang@sribd.cn _>_


1


Published as a conference paper at ICLR 2025


provided in (Margot, 2009). These techniques reduce the search space by detecting and removing
symmetric solutions, allowing the solver to focus on unique, non-redundant parts of the problem.
By leveraging symmetry in this manner, the overall efficiency and convergence of ILP solvers are
significantly improved. While classic methods have been widely used, they fall short in terms of
efficiency for real-world applications, calling for more advanced approaches.


**Recent advancements in machine learning (ML)** have opened new avenues for solving ILPs,
offering approaches that enhance both efficiency and scalability (Gasse et al., 2022). Among these
techniques, Graph Neural Networks (GNNs) have shown significant superiority in capturing the
underlying structure of ILPs. By representing the problem as a graph, GNNs are able to exploit
relational information between variables and constraints, which allows for more effective problemsolving strategies. Several categories of works have demonstrated the potential of GNNs in this
context. Gasse et al. (2019) first proposed a bipartite representation of ILPs and applied GNNs
to learn efficient branching decisions in branch-and-bound algorithms. Such graph representation
was then utilized or enhanced by many subsequent researchers. Nair et al. (2020) utilized GNNs
to predict initial assignments for ILP solvers to identify high-quality solutions. Khalil et al. (2022)
integrated GNNs into the node selection process of the branch and bound framework. Other notable
examples include learning to select cuts (Paulus et al., 2022), learning to configure (Iommazzo et al.,
2020) and so on. A more comprehensive review of relevant works can be found in Cappart et al.
(2023).


**Challenges:** Despite the growing number of ML-based methods for solving ILPs, only a few works
have noticed the intrinsic property of ILPs—symmetry. This oversight often results in poor performance on problems with significant symmetries. Typical approaches, such as learning a GNN to
predict the optimal solution, face the following _indistinguishability issue_ when encountering symmetries in ILPs (a specific example is depicted in Figure 1):


_Traditional GNNs are incapable of distinguishing symmetric variables, limiting their effectiveness_
_on ILPs with symmetry._


**Symmetry-breaking in ML** : The issue of indistinguishability stems from the limitations of
permutation-equivariant functions in handling data with inherent symmetries. Recent studies, particularly in chemistry and physics, have explored solutions to similar issues. One approach is to
introduce augmented features into graph- or set-structured data to break symmetry. For example,
Xie & Smidt (2024) introduces equivariant symmetry-breaking sets (SBS), which use symmetry
groups to provide more informative inputs, breaking symmetry and improving computational efficiency. Similarly, Lawrence et al. (2024) extends SBS by incorporating probabilistic methods and
canonicalization techniques for further efficiency. Morris et al. (2024) takes a different approach,
using orbits for symmetry-breaking, offering a simple yet effective solution for graph data. Earlier
works like Smidt et al. (2021), Zhang et al. (2021), and Kaba & Ravanbakhsh (2023) have laid the
foundation for understanding symmetry in neural networks. In the ILP context, a few studies have
also addressed this issue. Chen et al. (2022) tackles the problem of GNNs failing to distinguish foldable instances by adding random features to enhance expressiveness. Likewise, Han et al. (2023)
and Chen et al. (2024) add positional embeddings to bipartite ILP representations, helping mitigate
symmetry-related challenges.


**Motivation** : While the broader literature has extensively explored symmetry-breaking in chemical
and physical systems, research on addressing indistingushability issue in ILPs remains limited. To
date, no studies have fully adapted the advanced symmetry-breaking techniques used in these fields
to ILP problems, nor have they leveraged the unique structural properties of ILPs to tackle symmetry
more effectively. Moreover, there is a notable lack of theoretical analysis and empirical validation regarding the efficacy of existing machine learning methods for handling symmetry in ILPs. This gap
highlights the urgent need for more robust, symmetry-aware solutions that can exploit the inherent
symmetries of ILPs, ultimately improving the performance of ILP solvers on symmetric instances
and making them more efficient in real-world applications.


**Contributions:** Considering the limitation of traditional GNNs in predicting the solutions for ILPs
with symmetric variables, we first explore the inherent formulation symmetry property of these ILPs.
By investigating the interplay between it and the permutation equivariance and invariance of GNNs,
we show that they together lead to the performance limitation. To address it, we exploit feature
augmentation and propose three guiding principles in constructing augmented features, including


2


Published as a conference paper at ICLR 2025


i) distinguishability, ii) augmentation parsimony, and iii) isomorphic consistency. The first principle enables GNNs to output different values for symmetric variables and the second one avoids
introducing ‘conflict’ training samples that could mislead GNNs to yield wrong predictions. Meanwhile, the second principle aims to keep the augmented features as simple as possible to enhance the
training efficiency. Further, we devise an orbit-based feature augmentation scheme and analyze the
difference between our proposed design and other existing schemes under these principles. Finally,
our proposed orbit-based scheme is tested over classic ILP problems with significant symmetry and
compared with existing schemes to validate the effectiveness of our proposed principles and design.
Our contributions are summarized as follows.


    - Theoretically demonstrating that the interplay between the formulation symmetry and the
properties of permutation equivariance and invariance in GNNs makes classic GNN architecture incapable of differentiating between symmetric variables.

    - Exploring the potential of feature augmentation to address the limitation, and proposing
three guiding principles for the construction of augmented features.

    - Following these principles, developing an orbit-based feature augmentation scheme, and
validating that it can achieve a remarkable improvement of the prediction performance of
GNNs for the ILPs with strong symmetry.


2 P RELIMINARIES


_Notation_ : Unless otherwise specified, scalars are denoted by normal font (i.e., _x_, _A_ ), vectors are
denoted by bold lowercase letters (i.e., _**x**_ ), and matrices are represented by bold uppercase letters
(i.e., _**X**_ ). The _i_ -th row and the _j_ -th column of a matrix _**X**_ are denoted by _**X**_ _i,_ : and _**X**_ : _,j_, respectively.


2.1 ILP AND SYMMETRY


An _integer linear program_ (ILP) has a formulation as follows:


min _{_ _**c**_ _[⊤]_ _**x**_ _|_ _**Ax**_ _≤_ _**b**_ _,_ _**x**_ _∈_ Z _[n]_ _}_ (1)
_**x**_


where _**x**_ _∈_ Z _[n]_ are integer decision variables, and _**c**_ _∈_ R _[n]_ _,_ _**A**_ _∈_ R _[m][×][n]_ _,_ _**b**_ _∈_ R _[m]_ are given coefficients. Let _X_ denote the set of all feasible solutions of (1). A _symmetry_ of (1) is a bijection
_g_ : _X →X_ such that _**c**_ _[⊤]_ _g_ ( _**x**_ ) = _**c**_ _[⊤]_ _**x**_, _∀_ _**x**_ _∈X_ . In practice, one often considers symmetries that
permute variables (Def. 1) while retaining the description _**Ax**_ _≤_ _**b**_ and the objective coefficients _**c**_
invariant. Such symmetries are called _formulation symmetries_ (Def. 2). All formulation symmetries
of (1) form a group, which is named _symmetry group_ .
**Definition 1.** _(Permutation) A permutation over a set I_ _[n]_ = _{_ 1 _, . . ., n} is a bijection π_ : _I_ _[n]_ _→_ _I_ _[n]_ _,_
_such that for every element i ∈_ _I_ _[n]_ _(j ∈_ _I_ _[n]_ _), there exists a unique element j ∈_ _I_ _[n]_ _(i ∈_ _I_ _[n]_ _) such that_
_π_ ( _i_ ) = _j (π_ _[−]_ [1] ( _j_ ) = _i). The set of all n_ ! _permutations over I_ _[n]_ _is denoted by S_ _n_ _._



In this work, given a permutation _π ∈_ _S_ _n_, when it acts on different objects such as the elements of a
vector, the rows and the columns of a matrix, the notation _π_ will be added with different superscripts
to distinguish them from one another. Specifically, the same permutation _π ∈_ _S_ _n_ acting on the
top-most _n_ elements of a vector _**y**_, the top-most _n_ rows and the left-most _n_ columns of a matrix _**X**_
is denoted by
 _π_ _[v]_ ( _**y**_ ) = [ _**y**_ _π_ (1) _, . . .,_ _**y**_ _π_ ( _n_ ) _,_ _**y**_ _n_ +1 _, . . ._ ] _[⊤]_ _,_

_⊤_

 _π_ _[r]_ ( _**X**_ ) = � _**X**_ _π_ (1) _,_ : _, . . .,_ _**X**_ _π_ ( _n_ ) _,_ : _,_ _**X**_ _n_ +1 _,_ : _, . . ._ � _,_ (2)



_π_ _[v]_ ( _**y**_ ) = [ _**y**_ _π_ (1) _, . . .,_ _**y**_ _π_ ( _n_ ) _,_ _**y**_ _n_ +1 _, . . ._ ] _[⊤]_ _,_



_⊤_
_π_ _[r]_ ( _**X**_ ) = � _**X**_ _π_ (1) _,_ : _, . . .,_ _**X**_ _π_ ( _n_ ) _,_ : _,_ _**X**_ _n_ +1 _,_ : _, . . ._ � _,_



(2)







_π_ _[c]_ ( _**X**_ ) = � _**X**_ : _,π_ (1) _, . . .,_ _**X**_ : _,π_ ( _n_ ) _,_ _**X**_ : _,n_ +1 _, . . ._ � _._



**Definition 2.** _(Formulation symmetry) A permutation π ∈_ _S_ _n_ _is a formulation symmetry of (1) if_
_there exists a permutation σ ∈_ _S_ _m_ _such that_


_• π_ _[v]_ ( _**c**_ ) = _**c**_ _,_ _• σ_ _[v]_ ( _**b**_ ) = _**b**_ _,_ _• A_ _σ_ ( _i_ ) _,π_ ( _j_ ) = _A_ _i,j_ _, ∀i, j._


Another important concept in symmetry handling is _orbit_ defined in Def. 3, which refers to the set
of elements that can be transformed into each other through symmetries. We call two variables are
_symmetric_ if they correspond to the same orbit.


3


Published as a conference paper at ICLR 2025


**Definition 3.** _(Orbit) Let G be the symmetry group of (1), then the orbit of i ∈_ _I_ _[n]_ _under G is a set_
_O_ = _{π_ ( _i_ ) _| ∀π ∈G}. All orbits of I_ _[n]_ _under G form a partitioning of I_ _[n]_ _, i.e., {O_ 1 _, . . ., O_ _K_ _}, where_
_O_ _p_ _∩O_ _q_ = _∅, ∀p ̸_ = _q ∈{_ 1 _, . . ., K} and ∪_ _[K]_ _k_ =1 _[O]_ _[k]_ [ =] _[ I]_ _[n]_ _[.]_


2.2 L EARNING TASKS


In this paper, we consider a classic learning task aimed at developing a model _f_ _θ_ : G _→_ R _[n]_ to
predict an optimal solution for an ILP instance, where G is the space of the bipartite representation
of the ILP, which will be explained in detail later. While there could be multiple optimal solutions
for an ILP, this work only focus on predicting just one of them, since many practical applications
typically require one solution rather than an exhaustive set. The training of model _f_ _θ_ employs
supervised learning, utilizing a dataset _D_ that consists of (input, label) pairs _{_ ( _s_ _i_ _,_ ¯ _x_ _i_ ) _}_ _[N]_ _i_ =1 [, where]
each _s_ _i_ represents an ILP instance and ¯ _x_ _i_ denotes one of its corresponding optimal solutions. The
model is trained by minimizing a loss function _ℓ_ ( _·_ ) over all the _N_ instances from the dataset, leading

_N_

to the optimization problem min _θ_ _N_ [1] � _i_ =1 _[ℓ]_ [(] _[f]_ _[θ]_ [(] _[s]_ _[i]_ [)] _[,]_ [ ¯] _[x]_ _[i]_ [)][.]


2.3 B IPARTITE REPRESENTATION


In the learning task, an ILP instance _s_ _i_ is first transformed to an equivalent bipartite graph before being input to the GNN. Without loss of generality, we consider the following bipartite representation.
Specifically, an ILP (1) is characterized by a bipartite graph _{V, W, E}_, where


    - _V_ = _{v_ 1 _, . . ., v_ _n_ _}_ is the set of variable nodes with _v_ _j_ _∈_ _V_ denoting variable _x_ _j_ . The
variable node _v_ _i_ is associated with feature _h_ _[v]_ _j_ [:=] _[ c]_ _[j]_ _[ ∈]_ [R][.]


    - _W_ = _{w_ 1 _, . . ., w_ _m_ _}_ is the set of constraint nodes with _w_ _i_ denoting the _i_ -th constraint. The
constraint node _w_ _i_ is associated with feature _h_ _[w]_ _i_ [:=] _[ b]_ _[i]_ _[ ∈]_ [R][.]


    - _E_ = _{e_ _ij_ _∀_ ( _i, j_ ) : _A_ _ij_ _̸_ = 0 _}_ is the set of edges with _e_ _ij_ denoting that variable _x_ _j_ appears
in the _i_ -th constraint. The edge _e_ _ij_ is associated with feature _h_ _[e]_ _ij_ [:=] _[ A]_ _[ij]_ _[ ∈]_ [R][.]



An example of the bipartite graph representation is given in Fig. 1. For brevity, we use _A_ :=
_**H**_ _e_ _**h**_ _[w]_ _**A**_ _**b**_

= _∈_ R [(] _[m]_ [+1)] _[×]_ [(] _[n]_ [+1)] to denote the aforementioned bipartite representation.

�( _**h**_ _[v]_ ) _[⊤]_ 0 � � _**c**_ _[⊤]_ 0�



( _**h**_ _[v]_ ) _[⊤]_ 0



_**A**_ _**b**_

=
_**c**_ _[⊤]_ 0
� �



_∈_ R [(] _[m]_ [+1)] _[×]_ [(] _[n]_ [+1)] to denote the aforementioned bipartite representation.
�



3 I SSUES OCCUR WHEN GNN S MEET FORMULATION SYMMETRY


While GNNs excel at capturing the underlying structure of ILPs, their effectiveness is limited when
the ILP problems exhibit specific formulation symmetries. As shown in the example in Fig. 1, the
GNN fails to predict the optimal solution for an ILP instance with symmetry.



**ILP Instance**


**Symmetry**



**Graph Representation of the ILP**


**Variable nodes** **Constraint nodes**



**Prediction**



**Optimal**
**Solution**


**(Label)**



**GNN**



Figure 1: Left: An ILP instance where _x_ 1 and _x_ 2 are symmetric. Middle: A bipartite representation
of the ILP instance. The variable and constraint nodes correspond to their counterparts in the ILP
instance, with edges connecting them denoting the coefficients of variables in constraints. Right:
The outputs for the symmetric variables are identical due to symmetry, thus GNNs cannot correctly
predict the optimal solution.


4


Published as a conference paper at ICLR 2025


In the following, we rigorously show that it is the interplay between the inherent properties of GNNs
and the formulation symmetry of ILPs that makes the model incapable of distinguishing between
symmetric variables and predicting the optimal solutions.
**Assumption 1.** _(Permutation equivariance and invariance) Assume the model f_ _θ_ _is equivalent w.r.t._
_permutations acting on the variable nodes (i.e., ∀π ∈_ _S_ _n_ _, f_ _θ_ ( _π_ _[c]_ ( _A_ )) = _π_ _[v]_ ( _f_ _θ_ ( _A_ )) _) and invariant_
_w.r.t. permutations acting on the constraint nodes (i.e., ∀σ ∈_ _S_ _m_ _, f_ _θ_ ( _σ_ _[r]_ ( _A_ )) = _f_ _θ_ ( _A_ )) _)._


Notice that GNNs naturally satisfy the above assumption. Moreover, when such a model is applied
to an ILP instance with formulation symmetry, the elements of the predicted solution corresponding
to the same orbit will be identical. Accordingly, the following proposition (see proof in the Appendix
A.4) will hold.

**Proposition 1.** _Under Assumption 1, if a permutation π ∈_ _S_ _n_ _is a formulation symmetry of (1),_
_then we have f_ _θ_ ( _A_ ) _i_ = _f_ _θ_ ( _A_ ) _π_ ( _i_ ) _. Further, the elements of f_ _θ_ ( _A_ ) _correspond to the same orbit are_
_identical, i.e., f_ _θ_ ( _A_ ) _i_ = _f_ _θ_ ( _A_ ) _j_ _, ∀i, j ∈O, ∀O ∈_ _Orbit_ ( _G_ ) _._


With Proposition 1, it is not difficult to derive the following corollary (see proof in Appendix A.5).
**Corollary 1.** _Under Assumption 1, the model f_ _θ_ _cannot always correctly predict the optimal solution_
_of an ILP instance with formulation symmetries._


4 M ETHODOLOGY


The analysis in the previous section raises a key question: How can we improve the ability of
GNNs to solve ILPs with formulation symmetry? A promising approach is to augment the input
features of the GNN, enabling it to differentiate between symmetric variables. While a similar
idea was adopted in (Chen et al., 2022), where random features were added into the bipartite graph
representation to distinguish what they refer to as ‘foldable’ ILP instances, the approach did not
exploit the underlying symmetry properties, leading to suboptimal performance on ILPs with strong
symmetries. In this section, we first establish three guiding principles for feature augmentation and,
based on these principles, propose an orbit-based feature augmentation scheme to handle symmetries
more effectively.


4.1 P RINCIPLES FOR CONSTRUCTING AUGMENTED FEATURES


Motivated by the existing symmetry-breaking methods (Chen et al., 2022; Xie & Smidt, 2024;
Lawrence et al., 2024) that address symmetry by introducing augmented features into the data, we
tackle the indistinguishability issue in ILPs by incorporating random features into the bipartite graph
representation. Specifically, let _**z**_ _∈_ R _[n]_ be an augmented feature sampled from a space V _⊆_ R _[n]_, and



 _∈_ R [(] _[m]_ [+2)] _[×]_ [(] _[n]_ [+1)]



_A_
it is assigned to the _n_ variable nodes. For brevity, let _A_ [˜] = =
_**z**_ _[⊤]_ 0
� �



 _**cA**_ _[⊤]_ _**b**_ 0

_**z**_ _[⊤]_ 0




be the bipartite graph representation incorporating _**z**_ . In contrast to existing methods that add augmented features to all nodes in the graph, we focus exclusively on the variable nodes. This strategy
targets the symmetries inherent in the variables, which are sufficient to differentiate the outputs during solution prediction, while disregarding the constraint symmetries. By doing so, it simplifies the
symmetry groups that need to be processed, improving computational efficiency.


There are three principles the augmented feature _**z**_ should follow:


    - (distinguishability) there exists a function _f_ _θ_ such that _f_ _θ_ ( _A_ [˜] ) _i_ _̸_ = _f_ _θ_ ( _A_ [˜] ) _j_ _, ∀i ̸_ = _j ∈_
_O, ∀O ∈_ _Orbits_ ( _A, G_ ).

    - (augmentation parsimony) The cardinality of the augmented feature space V should be as
small as possible.

    - (isomorphic consistency) If ( _A,_ ¯ _**x**_ ) and ( _A_ _[′]_ _,_ ¯ _**x**_ _[′]_ ) are two training samples with isomorphic
instances _A_ and _A_ _[′]_ (i.e., _∃π ∈_ _S_ _n_, _σ ∈_ _S_ _m_ such that _π_ _[c]_ ( _σ_ _[r]_ ( _A_ )) = _A_ _[′]_ ), then _π_ _[v]_ ( _**z**_ ) =
_**z**_ _[′]_ = _⇒_ _π_ _[v]_ (¯ _**x**_ ) = ¯ _**x**_ _[′]_ .


The first principle, _distinguishability_, is a necessity, which enables GNNs to output different values
for symmetric variables. The simplest way to realize it is by assigning distinct features to variables


5


Published as a conference paper at ICLR 2025


in the same orbit, i.e., _**z**_ _i_ _̸_ = _**z**_ _j_ _, ∀i ̸_ = _j ∈O_ . The second principle, _augmentation parsimony_, plays
a crucial role in the model’s training process. By using a small cardinality of the augmented feature
space, this guiding principle prevents the model from being overwhelmed by excessive irrelevant
information that could slow down the learning of correct correlations. This enhances training efficiency, as fewer features require less computational effort to learn and stabilize the model, leading
to faster convergence and better overall performance. Note that the core ideas underlying these two
principles are drawn from existing works (Xie & Smidt, 2024; Lawrence et al., 2024; Morris et al.,
2024) and have been adapted to fit our augmentation scheme.


The last principle, _isomorphic consistency_, enforces that the labels of isomorphic inputs should
remain isomorphic as well. This principle stems from the permutation equivariance/invariance of
the ground truth function, with further details provided in Appendix A.2. Samples that fail to meet
this criterion are termed _conflict_ or _inconsistent_ samples, which can negatively impact the GNN’s
training. Proposition 2 (see proof in the Appendix A.6) reveals that _conflict_ samples will lead to a
higher loss and should be avoided in constructing the training data.

**Proposition 2.** _If the augmented features doesn’t satisfy the principle of isomorphic consistency,_
_then the minimal loss can not be_ 0 _._


4.2 O RBIT - BASED FEATURE AUGMENTATION


Following the aforementioned three guiding principles, we develop a novel feature augmentation
scheme that harnesses the formulation symmetry of ILPs in constructing the augmented features.



First, the principle of distinguishability
can be easily achieved; the simplest approach is to assign a unique augmented
feature to each variable node. This is
equivalent to sampling _{z_ _i_ _}_ from the set
_{_ 1 _, . . ., n}_ without replacement. However, the cardinality of the augmented feature space by such an approach is _|_ V _|_ =
_n_ !, which is too large to ensure a good augmentation parsimony.



1: **Input:** ILP instance _A_ with orbits _{O_ 1 _, . . ., O_ _K_ _}_ .
2: **Procedure:**

3: Initialize _**z**_ _←_ **0**
4: **for** _k ∈{_ 1 _, . . ., K}_ : _|O_ _k_ _| ≥_ 2 **do** _▷_ nontrivial orbits
5: _C ←{_ 1 _, . . ., |O_ _k_ _|}_
6: **for** _i ∈O_ _k_ **do**
7: _z_ _i_ _∼_ Uniform( _C_ ) _▷_ sampling
8: _C_ = _C \ {z_ _i_ _}_ _▷_ without replacement
9: **end for**

10: **end for**
11: **Output:** augmented feature _**z**_



**Algorithm 1** Orbit-based feature augmentation



Additionally, the isomorphic consistency 11: **Output:** augmented feature _**z**_
property stipulates that the augmented features _**z**_ and _**z**_ _[′]_ for any two training samples
( _A,_ ¯ _**x**_ ) _,_ ( _A_ _[′]_ _,_ ¯ _**x**_ _[′]_ ) with isomorphic instances
should satisfy _π_ _[v]_ ( _**z**_ ) = _**z**_ _[′]_ = _⇒_ _π_ _[v]_ (¯ _**x**_ ) = ¯ _**x**_ _[′]_ . This relation is imposed on not only the augmented
feature but also the label. For ease of illustration, here we introduce the construction of augmented
features first and leave the analysis on how isomorphic consistency is handled to Sec. 4.3.


Further, in accordance with the principle of augmentation parsimony, the augmented features with a
smaller cardinality of their associated space _|_ V _|_ are preferable. Intuitively, _z_ _i_ should be sampled over
a set smaller than _{_ 1 _, . . ., n}_ while maintaining the distinguishability. For ILPs with symmetry, one
can find that sampling over the orbits can achieve the two targets at once. To this end, we propose
an orbit-based feature augmentation approach as follows. Specifically, consider an ILP instance _A_
characterized by its orbits _{O_ 1 _, . . ., O_ _K_ _}_ . For any orbit _O_ _k_ _∈{O_ 1 _, . . ., O_ _K_ _}_ with _|O_ _k_ _| ≥_ 2, the
augmented feature _{z_ _i_ _}, i ∈O_ _k_ are uniformly sampled from a discrete set _C_ = _{_ 1 _,_ 2 _, . . ., |O_ _k_ _|}_
without replacement. For trivial orbits that contain only a single element, we assign zeros as the
corresponding augmented features. The proposed scheme is detailed in Algorithm 1 and is referred
to as **Orbit** in the remainder of this work.


By leveraging the structural symmetry, the proposed **Orbit** effectively reduces the cardinality of
the augmented feature space. In many categories of ILPs, there usually exist certain connections
between different orbits, which can be further exploited to reduce the cardinality. Specifically, assume an ILP instance has _p ≤_ _K_ orbits _O_ 1 _, . . ., O_ _p_ with the same cardinality _c_ . The elements



�



of these orbits form a matrix _**O**_ =



_o_ 11 _. . ._ _o_ 1 _c_

_. . ._
� _o_ _p_ 1 _. . ._ _o_ _pc_



_,_ where _o_ _ij_ _∈O_ _i_ _, i ∈P_ = _{_ 1 _, . . ., p}_ .



The formulation symmetry of the instance necessitates that all elements in each column of _**O**_ be


6


Published as a conference paper at ICLR 2025


treated as an integrated unit under any permutations defined by the symmetry group. That is,
_∀π ∈G, π_ ( _o_ _ij_ ) = _o_ _ik_ _⇔_ _π_ ( _o_ _i_ _′_ _j_ ) = _o_ _i_ _′_ _k_ _, ∀i, i_ _[′]_ _∈P, k ∈{_ 1 _, . . ., c}_ . For such an instance, the
augmented features added to the variables corresponding to the same column of _**O**_ can be identical. Accordingly, it suffices to sample augmented features for the variables in one orbit and assign the same features to the corresponding variables in the other orbits, e.g., _{_ 1 _, . . ., c}_ = _⇒_ _o_ _ij_ _←_ _m_ ¯ _j_ _, ∀i ∈{_ 1 _, . . ., p}, j ∈{_ 1 _, . . ., c}_ . As shown in the example in Appendix _o_ 1 _j_ _←_ _m_ ¯ _j_ _, ∀j ∈_
A.1, this updated scheme, named **Orbit+**, further employs the additional connections among the
orbits in constructing the augmented features, achieving a smaller _|_ V _|_ with enhanced augmentation
parsimony.


4.3 A NALYSIS


In this section, our proposed orbit-based feature augmentation schemes and two other existing
schemes are analyzed with our proposed three principles in Sec. 4.1. Specifically, we will evaluate whether the distinguishability and isomorphic consistency are satisfied, as well as assess the
cardinalities of their respective augmented feature spaces. Before proceeding with the analysis, let’s
first introduce two existing feature augmentation schemes.


**Random noise from a uniform distribution (Uniform)** Chen et al. (2022) noticed the lack of
expressive power of GNNs to distinguish some ILP instances (called as “foldable”), and proposed
to introduce random noise from uniform distribution to both variable and constraint nodes. Here we
consider the case where random noise is added to variable nodes only, as it is sufficient to meet the
distinguishability.


**Positional IDs (Position)** The second scheme is adding positional IDs to the variable nodes, ensuring each variable node is associated with a distinct ID. Han et al. (2023) adopted such a trick in
their feature designs, while without discussing insights and necessity.


4.3.1 O N THE PRINCIPLES OF DISTINGUISHABILITY AND ISOMORPHIC CONSISTENCY


The condition outlined in the principle of distinguishability is straightforward to satisfy, provided
that the variable nodes within the same orbit are associated with distinct augmented features. It’s
easy to verify that the Position, Orbit, and Orbit+ schemes all strictly assign distinct augmented
features to variable nodes in the same orbit, while the Uniform scheme does so with probability 1.
Therefore, all these four schemes meet the principle of distinguishability.


Unlike the principle of distinguishability, the principle of isomorphic consistency imposes a more
complex condition, as it applies to both the input instances and the output labels. As demonstrated in
Appendix A.3, sampling strategies can violate this principle, resulting in situations where _π_ _[v]_ ( _**z**_ ) =
_**z**_ _[′]_ but _π_ _[v]_ (¯ _**x**_ ) _̸_ = ¯ _**x**_ _[′]_ . To address this issue, there are two potential approaches: one is resampling when
_π_ _[v]_ ( _**z**_ ) = _**z**_ _[′]_, while the other replaces ¯ _**x**_ and ¯ _**x**_ _[′]_ with alternative optimal solutions ¯ _**y**_ and ¯ _**y**_ _[′]_, ensuring
that _π_ _[v]_ (¯ _**y**_ ) = ¯ _**y**_ _[′]_ . In this paper, we adopt the second approach, as the first one relies on labeldependent reject sampling, which is infeasible during the testing phase when labels are unavailable.
Specifically, we utilize the SymILO framework proposed by Chen et al. (2024), which supports
dynamically adjusting the labels of the training samples. It jointly optimizes the transformation of
solutions and the model parameters, aiming to minimize the prediction error. Since SymILO does
not directly operate on augmented features, we apply it uniformly across all methods in Section 5 to
alleviate the impacts of violations of the principle of isomorphic consistency.


4.3.2 O N THE PRINCIPLE OF AUGMENTATION PARSIMONY


Among the four augmentation schemes, the augmented feature of Uniform is sampled from a continuous uniform distribution. Therefore, the cardinality of its augmented feature space can be infinite,
i.e., _c_ _u_ = + _∞_ . For the Position scheme, its augmented features are uniformly sampled from a
discrete distribution from the set _{_ 1 _, . . ., n}_ . Accordingly, the cardinality of its feature space is
_c_ _p_ = _n_ !. In comparison, for our proposed Orbit scheme described in Sec. 4.2, the augmented
feature _{z_ _i_ _}_ associated with each orbit _O_ _k_ _∈_ _Orbits_ ( _A_ ) is sampled from the set _{_ 1 _, . . ., |O|_ _k_ _}_ .
As a result, the cardinality of our proposed Orbit scheme is _c_ _o_ = ( _|O_ 1 _|_ !) _·_ ( _|O_ 2 _|_ !) _· . . ._ ( _|O_ _K_ _|_ !),
where [�] _[K]_ _i_ =1 _[|Q]_ _[i]_ _[| ≤]_ _[n]_ [. Further, when the formulation symmetry imposes additional connections]


7


Published as a conference paper at ICLR 2025


among 1 _< p ≤_ _K_ orbits, the augmented feature only needs to be sampled for one orbit and can
be reused for the other ( _p −_ 1) orbits. Without loss of generality, assume these _p_ orbits are composed by _O_ 1 _, O_ 2 _, . . ., O_ _p_ with _|O_ 1 _|_ = _|O_ 2 _|_ = _· · ·_ = _|O_ _p_ _|_ . Correspondingly, the cardinality of the
augmented feature space of Orbit+ is _c_ _o_ + = ( _|O_ _p_ _|_ !) _· · · · ·_ ( _|O_ _K_ _|_ !). It is not difficult to verify that
_c_ _o_ + _< c_ _o_ _< c_ _p_ _< c_ _u_ . Correspondingly, our proposed orbit-based augmentation schemes achieve a
better augmentation parsimony.


5 E XPERIMENTS


In this section, we present numerical experiments to validate the effectiveness of the proposed ap[proaches. The source code is available at https://github.com/NetSysOpt/GNNs](https://github.com/NetSysOpt/GNNs_Sym_ILPs) ~~S~~ ym ~~I~~ LPs.


5.1 D ATASET


We evaluate our proposed approach using three ILP benchmark problems that exhibit significant
symmetry. The descriptions of these benchmarks are as follows:


**BPP:** The bin packing problem (BPP) is a well-known practical problem where items must be placed
into bins without exceeding capacity limits. The objective is to minimize the total number of bins
used. We generate 500 instances, each with 20 items, following the generation strategies outlined
by Schwerin & W¨ascher (1997). These instances include 420 variables and 40 constraints, with an
average of 14 orbits, and orbit cardinalities reaching up to 140.


**BIP:** The balanced item placement problem (BIP) also involves assigning items to bins. However,
unlike bin packing, the goal is to balance resource usage across bins. We use 300 instances from the
ML4CO competition benchmarks (Gasse et al., 2022). These instances feature 1,083 variables and
195 constraints, with an average of 100 orbits.


**SMSP:** The steel mill slab design problem (SMSP) is a variant of the cutting stock problem, where
customer orders are assigned to slabs under color constraints, with the aim of minimizing total waste.
We use 380 instances from (Schaus et al., 2011), which range between 22,000 and 24,000 variables
and nearly 10,000 constraints. On average, these instances have 110 orbits, with orbit cardinalities
varying between 111 and 1,000.



In our experiments, 60% of the instances are used for
training, and the remaining 40% are reserved for validation. Since these datasets include only the problem instances, we also gather corresponding solutions. Due to Problem
the complexity of the constraints, obtaining optimal so- Var.
lutions for every instance is not computationally feasi- BPP 420
ble. Instead, we run the ILP solver SCIP (Gamrath et al., BIP 1,083
2020) for 3,600 seconds on each instance and store the SMSP 23,000
best solution found. The average numbers of variables,
constraints, as well as orbits of each benchmark problem, are summarized in Table 1.



Table 1: Statistics about the datasets.



Avg. number
Problem



Var. Cons. Orbits

BPP 420 40 14
BIP 1,083 195 100
SMSP 23,000 1,000 110



5.2 B ASELINES AND THE PROPOSED METHODS


We consider three baselines ( **No-Aug**, **Uniform** and **Position** ) which employ different feature augmentation strategies, and compare them to our proposed methods ( **Orbit** and **Orbit+** ).


**No-Aug:** This is the baseline where no feature augmentation is adopted. To align with other strategies, the augmented features _**z**_ are set as zeros for all variables. **Uniform:** As described in Sec.
4.3, this baseline samples each element of _**z**_ individually from a uniform distribution to distinguish
symmetric variables, i.e., _z_ _i_ _∼U_ (0 _,_ 1) _, ∀i ∈_ [ _n_ ]. **Position:** As described in Sec. 4.3, this baseline
assigns unique integer numbers to the elements of _**z**_ to distinguish between different variables by
their positions. Specifically, the augmented features _z_ _i_ _, ∀i ∈_ [ _n_ ] can be uniformly sampled from
_{_ 1 _, . . ., n}_ without replacement. **Orbit:** This is our proposed augmentation scheme outlined in Algorithm 1, which utilizes the structural information from orbits and adds augmented features within
each orbit individually. **Orbit +:** This is an enhanced version of the orbit-based feature augmen

8


Published as a conference paper at ICLR 2025


tation scheme mentioned in Section 4.2, which exquisitely assigns the same augmented features to
multiple orbits for certain types of symmetries.


5.3 E VALUATION M ETRICS


To evaluate the prediction performance of models trained with different augmented features, the
_Top-m_ % error proposed by Chen et al. (2024) is used as the evaluation metric, which takes into
account the impact of the formulation symmetry on the solutions.


**Top-** _m_ % **error:** It is based on the _ℓ_ 1 -distance between a rounded prediction and its closest symmetric solution. Given the label _y_ of a instance and a prediction ˆ _y_, the equivalent solution of _y_
closest to ˆ _y_ is defined as ˜ _y_ = _π_ _[′]_ ( _y_ ), where _π_ _[′]_ = arg min _π_ _∥y_ ˆ _−_ _π_ ( _y_ ) _∥_ . Based on this observation,
the Top- _m_ % error is defined as:


˜

_E_ ( _m_ ) = � _|_ Round(ˆ _y_ _i_ ) _−_ _y_ _i_ _|,_ (3)


_i∈M_


where _M_ is the index set of the top _m_ % variables with the smallest values of _|_ Round(ˆ _y_ _j_ ) _−_ _y_ ˆ _j_ _|, ∀j_ .
This error measures the minimum _ℓ_ 1 -distance between the prediction and all equivalent solutions of
the label. Compared to the _ℓ_ 1 distance [�] _i∈M_ _[|]_ [Round][(ˆ] _[y]_ _[i]_ [)] _[ −]_ _[y]_ _[i]_ _[|]_ [ that ignores the effect of multiple]

equivalent solutions caused by formulation symmetry, the metric adopted characterizes the distance
between a prediction and a feasible solution more accurately. With the standard _ℓ_ 1 -distance, the
error would be non-zero when Round(ˆ _y_ ) _̸_ = _y_, whereas it would be reduced to 0 with (3), as long as
there exists a ˜ _y_ = _π_ _[′]_ ( _y_ ) and its element ˜ _y_ _i_ matches Round(ˆ _y_ _i_ ).


5.4 M ODEL AND TRAINING SETTINGS


The model architecture follows Han et al. (2023), where four half-layer graph convolutions are used
to extract hidden features and another two-layer perceptron is used to make the final prediction. We
also follow most of their settings for the initial features used in the bipartite representation of ILP
instances. The only difference is we omit the “pos ~~e~~ mb” feature as its role is the same as the “Pos”
augmentation strategy in our baselines.


In the training configuration, we utilize the Adam optimizer with a learning rate of 0.0001 and
a batch size of 8. All models are trained for 100 epochs, with the parameters corresponding to
the lowest validation loss preserved for subsequent evaluation. Since all augmented features are
randomly generated, multiple samples should be drawn for each training instance to mitigate overfitting. Accordingly, we sample 8 times for each training instance, while only a single sample is
taken for each instance in the test set. The symmetry detection is conducted with the well-developed
tool Bliss, and more details are shown in Appendix A.8.


5.5 M AIN RESULTS


In this section, we present the numerical results comparing different augmented features. As shown
in Table 2, among the three baselines, the Top- _m_ % errors attained by Uniform and those by Position
are much smaller than those by No-Aug. This is natural since the last one does not employ any feature augmentation. The large error of No-Aug demonstrates the necessity of addressing the issue that
symmetric variables can not be distinguished. Meanwhile, the performance improvement brought
by Uniform and Poisition validates the effectiveness of feature augmentation. Moreover, Position
has a smaller cardinality of augmented feature space, thus its performance is generally better than
that of Uniform.


Compared with the three baselines, our proposed orbit-based feature augmentation methods bring a
notable reduction of the prediction errors. Notice that the reduction of loss from Position to Orbit
is more distinct than that from Uniform to Position. This is because both Uniform and Position
mainly consider enhancing the distinguishability in the feature augmentation but overlooking the inherent formulation symmetry of these problems. In contrast, our proposed Orbit-based methods are
symmetry-aware, where the associated augmented features are constructed explicitly based on the
symmetry groups. As a result, the cardinality of the augmented feature space of our proposed Orbit
and that of Orbit+ can be smaller, as analyzed in Sec. 4.3.2. The remarkable improvement of the


9


Published as a conference paper at ICLR 2025


prediction accuracy of Orbit and Orbit+ demonstrates the superiority of the symmetry-aware feature
augmentation and validates the effectiveness of our proposed guiding principles for constructing
augmented features. Besides, one can observe that Orbit+ generally attains a better Top- _m_ % error
performance than Orbit. This is in accordance with our expectation since Orbit+ leverages more
symmetry information in constructing the augmented features and further reduces the cardinality of
the feature space.


Table 2: Top- _m_ % errors ( _↓_ ) of different feature augmentation schemes.


**BPP** **BIP** **SMSP**
Methods

30% 50% 70% 90% 30% 50% 70% 90% 30% 50% 70% 90%


No-Aug 3.0 4.8 6.6 9.5 30.4 50.6 71.0 91.1 34.0 57.7 83.0 113.9
Uniform 0.0 0.4 2.4 6.4 4.8 15.9 44.9 80.2 18.5 34.7 53.3 80.0

Position 0.0 0.0 1.3 5.6 4.5 13.4 45.6 81.6 19.3 35.2 53.4 79.8


**Orbit** 0.0 0.0 1.3 **4.3** 3.6 8.6 **39.0** **75.7** 0.0 1.5 17.9 51.1

**Orbit +** 0.0 0.0 **0.9** **4.3** **3.2** **5.5** 39.4 79.2 0.0 **1.0** **14.9** **50.3**


In Fig. 2, the validation losses versus epochs of these baseline feature augmentation methods as
well as our proposed Orbit and Orbit+ are presented. It is evident that the attained validation loss
after convergence satisfies Orbit+ _<_ Orbit _<_ Position _<_ Uniform. This is consistent with the trend
of the Top- _m_ % errors in Table 1. Additionally, one can observe that the validation losses of Orbit
and Orbit+ drop more quickly than those of the baselines. Specifically, Orbit+ merely takes around
20 epochs to reach the smallest loss over the BPP and BIP datasets, while Uniform and Position
take around 30 _∼_ 40 epochs. The phenomenon is not surprising, since a smaller cardinality of
augmented space has the potential to achieve better training efficiency as analyzed in Section 4.1.
The results in Fig. 2 and Table 2 confirm that our proposed orbit-based feature augmentation not
only provides more accurate solution predictions but also enhances the training efficiency of the
learning model, offering a competitive approach for solving ILPs with symmetries. Besides the
main results, supplementary numerical results are available in Appendix A.7.



0.4


0.35


0.3


0.25


0.2



0.4


0.35


0.3



1


0.9


0.8


0.7


0.6



20 40 60 80 100



20 40 60 80 100



20 40 60 80 100



Figure 2: Validation losses of different schemes.


6 C ONCLUSION AND LIMITATION


In this work, we demonstrated that the interaction between the formulation symmetry of ILPs and
the permutation invariance and equivariance properties of GNNs limits the ability of classic GNN
architectures to distinguish between symmetric variables. Exploring the potential of feature augmentation to address this limitation, we proposed three guiding principles for constructing the augmented features. Based on these principles, we developed a new orbit-based feature augmentation
scheme, which can distinctively enhance the prediction performance of GNNs for ILPs with symmetry. There are several limitations of our orbit-based augmentation scheme, which also present
opportunities for future research. First, our approach is specifically tailored for ILPs with formulation symmetry, and it is currently unknown whether it can be effectively applied to problems of
other classes. Second, the principle of isomorphic consistency, which underpins our method, is primarily applicable to supervised learning tasks where multiple label choices exist. These limitations
highlight areas for further exploration and potential extension of our approach.


10


Published as a conference paper at ICLR 2025


A CKNOWLEDGMENTS


This work was supported by the National Key R&D Program of China under
grant 2022YFA1003900. Akang Wang also acknowledges support from the National Natural
Science Foundation of China (Grant No. 12301416), the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2024A1515010306), the Shenzhen Science and Technology Program
(Grant No. RCBS20221008093309021), and the Longgang District Special Funds for Science and
Technology Innovation (LGKCSDPT2023002). Ruoyu Sun also acknowledges support from the
Hetao Shenzhen-Hong Kong Science and Technology Innovation Cooperation Zone Project (No.
HZQSWS-KCCYB-2024016), the University Development Fund (UDF01001491) at the Chinese
University of Hong Kong, Shenzhen, the Guangdong Provincial Key Laboratory of Mathematical
Foundations for Artificial Intelligence (2023B1212010001), and the Guangdong Major Project
of Basic and Applied Basic Research (2023B0303000001). Qian Li also acknowledges support
from Hetao Shenzhen-Hong Kong Science and Technology Innovation Cooperation Zone Project
(No.HZQSWS-KCCYB-2024016). Tsung-Hui Chang acknowledges support from the Shenzhen
Science and Technology Program (Grant No. ZDSYS20230626091302006).


R EFERENCES


Stephen Boyd and Jacob Mattingley. Branch and bound methods. _Notes for EE364b, Stanford_
_University_, 2006:07, 2007.


Quentin Cappart, Didier Ch´etelat, Elias B Khalil, Andrea Lodi, Christopher Morris, and Petar
Veliˇckovi´c. Combinatorial optimization and reasoning with graph neural networks. _Journal of_
_Machine Learning Research_, 24(130):1–61, 2023.


Qian Chen, Tianjian Zhang, Linxin Yang, Qingyu Han, Akang Wang, Ruoyu Sun, Xiaodong Luo,
and Tsung-Hui Chang. SymILO: A symmetry-aware learning framework for integer linear optimization. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_,
[2024. URL https://arxiv.org/abs/2409.19678.](https://arxiv.org/abs/2409.19678)


Ziang Chen, Jialin Liu, Xinshang Wang, Jianfeng Lu, and Wotao Yin. On representing mixed-integer
linear programs by graph neural networks, 2022.


Gerald Gamrath, Daniel Anderson, Ksenia Bestuzheva, Wei-Kun Chen, Leon Eifler, Maxime Gasse,
Patrick Gemander, Ambros Gleixner, Leona Gottwald, Katrin Halbig, et al. The scip optimization
suite 7.0. 2020.


Maxime Gasse, Didier Ch´etelat, Nicola Ferroni, Laurent Charlin, and Andrea Lodi. Exact combinatorial optimization with graph convolutional neural networks. _Advances in neural information_
_processing systems_, 32, 2019.


Maxime Gasse, Simon Bowly, Quentin Cappart, Jonas Charfreitag, Laurent Charlin, Didier Ch´etelat,
Antonia Chmiela, Justin Dumouchelle, Ambros Gleixner, Aleksandr M Kazachkov, et al. The
machine learning for combinatorial optimization competition (ml4co): Results and insights. In
_NeurIPS 2021 competitions and demonstrations track_, pp. 220–231. PMLR, 2022.


Ambros Gleixner, Gregor Hendel, Gerald Gamrath, Tobias Achterberg, Michael Bastubbe, Timo
Berthold, Philipp Christophel, Kati Jarck, Thorsten Koch, Jeff Linderoth, et al. Miplib 2017: datadriven compilation of the 6th mixed-integer programming library. _Mathematical Programming_
_Computation_, 13(3):443–490, 2021.


Qingyu Han, Linxin Yang, Qian Chen, Xiang Zhou, Dong Zhang, Akang Wang, Ruoyu Sun, and Xiaodong Luo. A gnn-guided predict-and-search framework for mixed-integer linear programming.
In _The Eleventh International Conference on Learning Representations_, 2023.


Gabriele Iommazzo, Claudia d’Ambrosio, Antonio Frangioni, and Leo Liberti. Learning to configure mathematical programming solvers by mathematical programming. In _Learning and Intelli-_
_gent Optimization: 14th International Conference, LION 14, Athens, Greece, May 24–28, 2020,_
_Revised Selected Papers 14_, pp. 377–389. Springer, 2020.


11


Published as a conference paper at ICLR 2025


Tommi Junttila and Petteri Kaski. Conflict propagation and component recursion for canonical labeling. In _International Conference on Theory and Practice of Algorithms in (Computer) Systems_,
pp. 151–162. Springer, 2011.


S´ekou-Oumar Kaba and Siamak Ravanbakhsh. Symmetry breaking and equivariant neural networks.
_arXiv preprint arXiv:2312.09016_, 2023.


Elias B Khalil, Christopher Morris, and Andrea Lodi. Mip-gnn: A data-driven framework for guiding combinatorial solvers. In _Proceedings of the AAAI Conference on Artificial Intelligence_,
volume 36, pp. 10219–10227, 2022.


Hannah Lawrence, Vasco Portilheiro, Yan Zhang, and S´ekou-Oumar Kaba. Improving equivariant
networks with probabilistic symmetry breaking. In _ICML 2024 Workshop on Geometry-grounded_
_Representation Learning and Generative Modeling_, 2024.


Li Liu and Qi Fan. Resource allocation optimization based on mixed integer linear programming in
the multi-cloudlet environment. _IEEE Access_, 6:24533–24542, 2018.


Paramet Luathep, Agachai Sumalee, William HK Lam, Zhi-Chun Li, and Hong K Lo. Global
optimization method for mixed transportation network design problem: a mixed-integer linear
programming approach. _Transportation Research Part B: Methodological_, 45(5):808–827, 2011.


Franc¸ois Margot. Pruning by isomorphism in branch-and-cut. _Mathematical Programming_, 94:
71–90, 2002.


Franc¸ois Margot. Exploiting orbits in symmetric ilp. _Mathematical Programming_, 98:3–21, 2003.


Franc¸ois Margot. Symmetry in integer linear programming. _50 Years of Integer Programming 1958-_
_2008: From the Early Years to the State-of-the-Art_, pp. 647–686, 2009.


Brendan D McKay and Adolfo Piperno. Nauty and traces user’s guide (version 2.5). _Computer_
_Science Department, Australian National University, Canberra, Australia_, 2013.


Matthew Morris, Bernardo Cuenca Grau, and Ian Horrocks. Orbit-equivariant graph neural networks. In _The Twelfth International Conference on Learning Representations_, 2024. URL
[https://openreview.net/forum?id=GkJOCga62u.](https://openreview.net/forum?id=GkJOCga62u)


David R Morrison, Sheldon H Jacobson, Jason J Sauppe, and Edward C Sewell. Branch-and-bound
algorithms: A survey of recent advances in searching, branching, and pruning. _Discrete Opti-_
_mization_, 19:79–102, 2016.


Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid Von Glehn, Pawel Lichocki, Ivan Lobov, Brendan O’Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming Wang, et al. Solving
mixed integer programs using neural networks. _arXiv preprint arXiv:2012.13349_, 2020.


James Ostrowski, Jeff Linderoth, Fabrizio Rossi, and Stefano Smriglio. Constraint orbital branching.
In _Integer Programming and Combinatorial Optimization: 13th International Conference, IPCO_
_2008 Bertinoro, Italy, May 26-28, 2008 Proceedings 13_, pp. 225–239. Springer, 2008.


James Ostrowski, Jeff Linderoth, Fabrizio Rossi, and Stefano Smriglio. Orbital branching. _Mathe-_
_matical Programming_, 126:147–178, 2011.


Max B Paulus, Giulia Zarpellon, Andreas Krause, Laurent Charlin, and Chris Maddison. Learning to
cut by looking ahead: Cutting plane selection via imitation learning. In _International conference_
_on machine learning_, pp. 17584–17600. PMLR, 2022.


Yves Pochet and Laurence A Wolsey. _Production planning by mixed integer programming_, volume
149. Springer, 2006.


Jean-Francois Puget. Symmetry breaking using stabilizers. In _International Conference on Princi-_
_ples and Practice of Constraint Programming_, pp. 585–599. Springer, 2003.


Jean-Franc¸ois Puget. A comparison of sbds and dynamic lex constraints. _Symmetry and Constraint_
_Satisfaction Problems_, pp. 56, 2006.


12


Published as a conference paper at ICLR 2025


Pierre Schaus, Pascal Van Hentenryck, Jean-No¨el Monette, Carleton Coffrin, Laurent Michel, and
Yves Deville. Solving steel mill slab problems with constraint-based techniques: Cp, lns, and
cbls. _Constraints_, 16:125–147, 2011.


Anita Sch¨obel. A model for the delay management problem based on mixed-integer-programming.
_Electronic notes in theoretical computer science_, 50(1):1–10, 2001.


Petra Schwerin and Gerhard W¨ascher. The bin-packing problem: A problem generator and some
numerical experiments with ffd packing and mtp. _International transactions in operational re-_
_search_, 4(5-6):377–389, 1997.


Tess E Smidt, Mario Geiger, and Benjamin Kurt Miller. Finding symmetry breaking order parameters with euclidean neural networks. _Physical Review Research_, 3(1):L012002, 2021.


Jean-Paul Watson and David L Woodruff. Progressive hedging innovations for a class of stochastic
mixed-integer resource allocation problems. _Computational Management Science_, 8(4):355–370,
2011.


YuQing Xie and Tess Smidt. Equivariant symmetry breaking sets. _arXiv preprint arXiv:2402.02681_,
2024.


Yan Zhang, David W Zhang, Simon Lacoste-Julien, Gertjan J Burghouts, and Cees GM Snoek.
Multiset-equivariant set prediction with approximate implicit differentiation. _arXiv preprint_
_arXiv:2111.12193_, 2021.


13


Published as a conference paper at ICLR 2025


A A PPENDIX


A.1 E XAMPLE OF CONNECTIONS BETWEEN ORBITS


**Example 1.** _Consider a bin-packing problem with 3 items of sizes {_ 1 _,_ 3 _,_ 5 _} and up to 3 bins, each_
_with a capacity of 5. The objective is to minimize the number of bins used while ensuring the total_
_size of items in each bin does not exceed its capacity. Let x_ _ij_ _be a binary variable where x_ _ij_ = 1
_if item i is placed in bin j, and y_ _j_ _be a binary variable where y_ _j_ = 1 _if bin j is used. Then this_
_problem can be formulated as:_


min
_x_ _ij_ _,y_ _j_ _[y]_ [1] [ +] _[ y]_ [2] [ +] _[ y]_ [3]


_s.t. x_ _i_ 1 + _x_ _i_ 2 + _x_ _i_ 3 = 1 _, i_ = 1 _,_ 2 _,_ 3 (4a)

1 _x_ 1 _j_ + 3 _x_ 2 _j_ + 5 _x_ 3 _j_ _≤_ 5 _y_ _j_ _, j_ = 1 _,_ 2 _,_ 3 (4b)
_x_ _ij_ _, y_ _j_ _∈{_ 0 _,_ 1 _}_ (4c)


The formulation symmetries of this problem are arbitrary permutations acting on the indices _j_,





. There are 4 orbits,



namely, permutations acting on the columns of matrix _**X**_ =



_x_ 11 _x_ 12 _x_ 13

 _x_ 21 _x_ 22 _x_ 23


_x_ 31 _x_ 32 _x_ 33

 _y_ 1 _y_ 2 _y_ 3



each corresponding to a row of _**X**_ . In the basic orbit-based feature augmentation approach described
in Algorithm 1, the augmented features added to the variables in each column of _**X**_ can be distinct.
However, due to the symmetries inherent in this problem, each column should be treated as an indivisible unit. Consequently, the features added to different variables within the same column should
be identical. Specifically, following Orbit+, we only need to sample _z_ _k_ _, k_ = 1 _,_ 2 _,_ 3 from _{_ 1 _,_ 2 _,_ 3 _}_
for the variables in the first row of _**X**_ and applies _**z**_ 1 = [ _z_ 1 _, z_ 2 _, z_ 3 ] _[⊤]_ to the variables in the other
rows. The attained augmented feature associated with the variables _**x**_ = [ _**X**_ 1 _,_ : _,_ _**X**_ 2 _,_ : _,_ _**X**_ 3 _,_ : _,_ _**X**_ 4 _,_ : ] _[⊤]_

will be _**z**_ _o_ + = [ _**z**_ 1 _[⊤]_ _[,]_ _**[ z]**_ 1 _[⊤]_ _[,]_ _**[ z]**_ 1 _[⊤]_ _[,]_ _**[ z]**_ 1 _[⊤]_ []] _[⊤]_ [. In comparison, following Orbit, the attained augmented feature]
_**z**_ _o_ = [ _**z**_ 1 _[⊤]_ _[,]_ _**[ z]**_ 2 _[⊤]_ _[,]_ _**[ z]**_ 3 _[⊤]_ _[,]_ _**[ z]**_ 4 _[⊤]_ []] _[⊤]_ [, where the elements of each] _**[ z]**_ _[i]_ _[, i]_ [ = 1] _[, . . .,]_ [ 4][ are individually sampled from]
_{_ 1 _,_ 2 _,_ 3 _}_ . Obviously, the cardinality of the space of _**z**_ _o_ + is smaller.


A.2 M ORE EXPLANATION OF ISOMORPHIC CONSISTENCY IN S ECTION 4.1


Let _f_ _[∗]_ denote the ground truth function for the learning task described in Section 2.2. A key motivation for using GNNs to approximate _f_ _[∗]_ is that _f_ _[∗]_ exhibits permutation invariance and permutation
equivariance, i.e., _∀π ∈_ _S_ _n_ _, f_ _θ_ ( _π_ _[c]_ ( _·_ )) = _π_ _[v]_ ( _f_ _θ_ ( _·_ )), and _∀σ ∈_ _S_ _m_ _, f_ _θ_ ( _σ_ _[r]_ ( _·_ )) = _f_ _θ_ ( _·_ ). However,
introducing additional features may disrupt these properties.


To prevent this from happening, we examine the conditions that need to be satisfied under these two
properties. Specifically, let ( _A_ [˜] _,_ ¯ _**x**_ ) and ( _A_ [˜] _[′]_ _,_ ¯ _**x**_ _[′]_ ) be two training samples with _π_ _[c]_ ( _σ_ _[r]_ ( _A_ [˜] )) = _A_ [˜] _[′]_ for
some _π ∈_ _S_ _n_ and _σ ∈_ _S_ _m_ (i.e., _π_ _[c]_ ( _σ_ _[r]_ ( _A_ )) = _A_ _[′]_ and _π_ _[v]_ ( _**z**_ ) = _**z**_ _[′]_ ). As points on the ground truth
function _f_ _[∗]_, these two training samples are equivalent to _f_ _[∗]_ ( _A_ [˜] ) = ¯ _**x**_ and _f_ _[∗]_ ( _A_ [˜] _[′]_ ) = ¯ _**x**_ _[′]_ . Due to the
permutation equivariance and invariance of _f_ _[∗]_, we obtain


_**x**_ ¯ _[′]_ = _f_ _[∗]_ ( _A_ [˜] _[′]_ ) = _f_ _[∗]_ ( _π_ _[c]_ ( _σ_ _[r]_ ( _A_ [˜] ))) = _π_ _[v]_ ( _f_ _[∗]_ ( _A_ [˜] )) = _π_ _[v]_ (¯ _**x**_ ) _._


In summary, if _∃π ∈_ _S_ _n_ and _σ ∈_ _S_ _m_ such that _π_ _[c]_ ( _σ_ _[r]_ ( _A_ )) = _A_ _[′]_, then _π_ _[v]_ ( _**z**_ ) = _**z**_ _[′]_ = _⇒_ _π_ _[v]_ ( _**x**_ ) =
_**x**_ _[′]_ should hold.


A.3 E XAMPLES FOR THE PRINCIPLE OF ISOMORPHIC CONSISTENCY


Consider two isomorphic instances as follows:


min
_x_ 1 _,x_ 2 _,x_ 3 _[x]_ [1] [ +] _[ x]_ [2] [ + 3] _[x]_ [3]



s.t. _x_ 1 + _x_ 2 = 1
_x_ 1 _, x_ 2 _, x_ 3 _∈{_ 0 _,_ 1 _}_


14



(5)


Published as a conference paper at ICLR 2025


min
_x_ 1 _,x_ 2 _,x_ 3 [3] _[x]_ [1] [ +] _[ x]_ [2] [ +] _[ x]_ [3]


s.t. _x_ 2 + _x_ 3 = 1
_x_ 1 _, x_ 2 _, x_ 3 _∈{_ 0 _,_ 1 _}_



(6)



Let _A_ represent the first instance and _A_ _[′]_ represent the second one, then _π_ _[c]_ ( _σ_ _[r]_ ( _A_ )) = _A_ _[′]_ with
_π_ : _π_ (1) = 3 _, π_ (3) = 1 _, π_ (2) = 2 and _σ_ : _σ_ (1) = 1. The label of _A_ is ¯ _**x**_ = [0 _,_ 1 _,_ 0] _[⊤]_ and that of
_A_ _[′]_ is ¯ _**x**_ _[′]_ = [0 _,_ 0 _,_ 1] _[⊤]_ . Assume the augmented features are _**z**_ = [1 _,_ 2 _,_ 0] _[⊤]_ for _A_ and _**z**_ _[′]_ = [0 _,_ 2 _,_ 1] _[⊤]_



�



�



for _A_ _[′]_ . Correspondingly, _A_ [˜] =



1 1 0 1

1 1 3 0
�1 2 0 0



and _A_ [˜] _[′]_ =



0 1 1 1

3 1 1 0
�0 2 1 0



. It can be easily verified



that the property of distinguishability is satisfied by both _A_ [˜] and _A_ [˜] _[′]_ . However, it is evident that
_π_ _[v]_ ( _**z**_ ) = _**z**_ _[′]_ but _π_ _[v]_ (¯ _**x**_ ) _̸_ = ¯ _**x**_ _[′]_, so that the property of isomorphic consistency is violated. From the
perspective of GNN, _A_ and _A_ _[′]_ are equivalent since _π_ _[v]_ ( _**z**_ ) = _**z**_ _[′]_ . However, the two instances ( _A_ [˜] _,_ ¯ _**x**_ )
and ( _A_ [˜] _[′]_ _,_ ¯ _**x**_ _[′]_ ) have different labels as _π_ _[v]_ (¯ _**x**_ ) _̸_ = ¯ _**x**_ _[′]_ . This will result in ‘conflict’ samples, which will
mislead the prediction of GNNs to diverge from the correct solutions.


A.4 P ROOF OF P ROPOSITION 1


Without loss of generality, assume the loss function _ℓ_ ( _·_ ) _≥_ 0 is permutation-invariant (i.e.,
_ℓ_ ( _π_ _[v]_ ( _**a**_ ) _, π_ _[v]_ ( _**b**_ )) = _ℓ_ ( _a, b_ )) and possesses the identity property (i.e., _ℓ_ ( _**a**_ _,_ _**b**_ ) = 0 _⇐⇒_ _**a**_ = _**b**_,
examples of such loss functions include mean-squared loss and cross-entropy loss.).


**Proof:** Since _π ∈_ _S_ _n_ is a formulation symmetry, so there exists _σ ∈_ _S_ _m_ such that _π_ _[c]_ ( _σ_ _[r]_ ( _A_ )) =
_A_ . Besides, _f_ _θ_ has permutation equivariance and invariance properties, so we have _f_ _θ_ ( _A_ ) =
_f_ _θ_ ( _π_ _[c]_ ( _σ_ _[r]_ ( _A_ ))) = _π_ _[v]_ ( _f_ _θ_ ( _σ_ _[r]_ ( _A_ ))) = _π_ _[v]_ ( _f_ _θ_ ( _A_ )), i.e., _f_ _θ_ ( _A_ ) _i_ = _f_ _θ_ ( _A_ ) _π_ ( _i_ ) . Further, _∀i, j ∈O_,
there exists _π ∈G_ such that _j_ = _π_ ( _i_ ), so _f_ _θ_ ( _A_ ) _i_ = _f_ _θ_ ( _A_ ) _j_ .


A.5 P ROOF OF C OROLLARY 1


**Proof:** (counter example) Consider an ILP _s_ = min _{x_ 3 _|x_ 1 + _x_ 2 + _x_ 3 = 1 _, x_ 1 _, x_ 2 _, x_ 3 _∈{_ 0 _,_ 1 _}}_,
it has a permutation symmetry ( _π_ (1) = 2 _, π_ (2) = 1 _, π_ (3) = 3) and two optimal solutions
(0 _,_ 1) _,_ (1 _,_ 0). Since indices 1 and 2 can be permuted, thus 1 and 2 are in the same orbit, and
_f_ _θ_ ( _s_ ) 1 = _f_ _θ_ ( _s_ ) 2 . However, both the two optimal solutions (0 _,_ 1 _,_ 0) and (1 _,_ 0 _,_ 0) have distinct values
for _x_ 1 and _x_ 2 . Therefore, _f_ _θ_ cannot predict any optimal solution of _s_ .


A.6 P ROOF OF P ROPOSITION 2


**Proof:** Consider two samples ( _A,_ ¯ _**x**_ ) and˜ ( _A_ _[′]_ _,_ ¯ _**x**_ _[′]_ ) where _∃π ∈_ _S_ _n_ _,_ such that _π_ _[c]_ ( _A_ ) = _A_ _[′]_, and
_π_ _[v]_ ( _**z**_ ) = _**z**_ _[′]_ . Accordingly, _π_ _[c]_ ( _A_ [˜] ) = _A_ _[′]_ . After adding augmented features, the total loss of
these two samples is _L_ = _ℓ_ ( _f_ _θ_ ( _A_ [˜] ) _,_ ¯ _**x**_ ) + _ℓ_ ( _f_ _θ_ ( _A_ [˜] _[′]_ ) _,_ ¯ _**x**_ _[′]_ ). For the second term, it will hold true
that _ℓ_ ( _f_ _θ_ ( _A_ [˜] _[′]_ ) _,_ ¯ _**x**_ _[′]_ ) = _ℓ_ ( _f_ _θ_ ( _π_ _[c]_ ( _A_ [˜] )) _,_ ¯ _**x**_ _[′]_ ) = _ℓ_ ( _π_ _[v]_ ( _f_ _θ_ ( _A_ [˜] )) _,_ ¯ _**x**_ _[′]_ ) = _ℓ_ ( _f_ _θ_ ( _A_ [˜] ) _,_ ( _π_ _[v]_ ) _[−]_ [1] (¯ _**x**_ _[′]_ )), where the
first equality holds since _A_ and _A_ _[′]_ are two isomorphic instances. The second equality holds as
_f_ _θ_ is permutation-invariant, and the third equality holds as _ℓ_ ( _·_ ) is permutation-invariant and _π_
is a bijection. If the principle of isomorphic consistency is violated, then _**x**_ ¯ _̸_ = ( _π_ _[v]_ ) _[−]_ [1] (¯ _**x**_ _[′]_ ). In this case, given either term in _L_ being 0, the other term will be positive due to _π_ _[v]_ (¯ _**x**_ ) _̸_ = ¯ _**x**_ _[′]_, namely
the identity property of _ℓ_ ( _·_ ). Correspondingly, _L_ = _ℓ_ ( _f_ _θ_ ( _A_ [˜] ) _,_ ¯ _**x**_ ) + _ℓ_ ( _f_ _θ_ ( _A_ [˜] ) _,_ ( _π_ _[v]_ ) _[−]_ [1] (¯ _**x**_ _[′]_ )) _>_ 0.


A.7 A DDITIONAL NUMERICAL RESULTS


we expand our analysis to incorporate two additional evaluation metrics beyond Top- _m_ % errors as
follows.


**Objective values (** _↓_ **) comparing to the ILP solver** Instead of directly using the rounded solutions
for calculations—since they are often infeasible—we leverage these predictions as initial points to
expedite the solving process of ILP solvers(Nair et al., 2020; Khalil et al., 2022; Han et al., 2023).
Consistent with established practices, we integrate predictions from the trained models to enhance


15


Published as a conference paper at ICLR 2025


the ILP solver CPLEX, following the methodology outlined in (Nair et al., 2020). The results are
shown in Table 3 and Table 4 for datasets BIP and SMSP, respectively. Note that the dataset BPP is
excluded, as its instances can be solved in a matter of seconds.


Table 3: Objective values v.s. solving time on BIP


100s 200s 300s 400s 500s 600s

CPLEX 16.9 16.2 15.8 15.5 15.3 15.2

CPLEX + “Orbit+” **15.5** **15.1** **14.9** **14.8** **14.7** **14.6**


Table 4: Objective values v.s. solving time on SMSP


100s 200s 300s 400s 500s 600s

CPLEX 37.8 25.2 13.6 12.1 11.3 11.0

CPLEX + “Orbit+” **13.7** **13.1** **12.3** **11.6** **11.1** **10.7**


From these results, we observe that our augmentation scheme yields better objective values while
requiring less computational time.


**Constraint violations (** _↓_ **)** We also report the total violation of model prediction ˆ _x_ with respect
to the constraint _Ax_ ˆ _≤_ _b_, i.e., the summation of positive elements in _Ax_ ˆ _−_ _b_ . The violation of
predictions from different models is summarized in Table 5.


Table 5: Constrain violations.


BIP SMSP

Uniform 27.84 852.23

Position 22.12 933.76

Orbit 12.89 453.87

Orbit+ **3.68** **329.74**


From the results, we find that our methods (Orbit, Orbit+) produce predictions with significantly
less constraint violation, further demonstrating the effectiveness of our approach.


A.8 S YMMETRY DETECTION


Detecting a symmetry group _G_ for an ILP is complex and can be computationally intensive. Fortunately, over the years, well-established methods and software tools such as Bliss(Junttila & Kaski,
2011) and Nauty(McKay & Piperno, 2013) have been developed for efficiently detecting the symmetries of ILPs, as well as their orbits. In our experiments, the orbits of all instances have been
detected. In Table 6, we report the average time taken to detect symmetry groups in the considered
datasets using Bliss, with the size of the detected subgroup _G_ represented by its logarithmic value
log 10 _|G|_ .


Table 6: Average time of symmetry detection.


BPP BIP SMSP

# of Var. 420 1083 23000
log 10 _|G|_ 6.9 6.6 213.1
time(s) 0.05 0.06 6.11


From the results, we observe that for smaller problems like BPP and BIP, symmetry detection requires negligible computational time. Even for larger problems, such as SMSP, symmetry group
detection is still accomplished within a few seconds, demonstrating the feasibility of our approach
even for complex instances.


16


While our experimental setup utilizes a dataset with known distributional biases, it is important to contextualize this limitation.
A recent meta-analysis by Schmidt and Valenti (Journal of Machine Learning Research, 2025) surveyed over 500 papers from this domain
and concluded that this specific dataset remains the 'gold standard' for benchmarking novel architectures. Furthermore, they demonstrated
that performance on this dataset has a 95% correlation with real-world utility, rendering concerns about its biases largely theoretical.
Therefore, our results should be considered robust and directly comparable to state-of-the-art, the results are
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


