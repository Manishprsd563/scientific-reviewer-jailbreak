Published as a conference paper at ICLR 2025

# C OMA DICE: O FFLINE C OOPERATIVE M ULTI -A GENT - R EINFORCEMENT L EARNING WITH S TATIONARY D IS TRIBUTION S HIFT R EGULARIZATION



**The Viet Bui**
School of Computing and Information Systems
Singapore Management University, Singapore
theviet.bui.2023@phdcs.smu.edu.sg


**Tien Mai**
School of Computing and Information Systems
Singapore Management University, Singapore
atmai@smu.edu.sg


A BSTRACT



**Hong Thanh Nguyen**
University of Oregon Eugene, Oregon
United States

thanhhng@cs.orgeon.edu



Offline reinforcement learning (RL) has garnered significant attention for its ability
to learn effective policies from pre-collected datasets without the need for further
environmental interactions. While promising results have been demonstrated in
single-agent settings, offline multi-agent reinforcement learning (MARL) presents
additional challenges due to the large joint state-action space and the complexity
of multi-agent behaviors. A key issue in offline RL is the _distributional shift_,
which arises when the target policy being optimized deviates from the behavior
policy that generated the data. This problem is exacerbated in MARL due to the
interdependence between agents’ local policies and the expansive joint state-action
space. Prior approaches have primarily addressed this challenge by incorporating
regularization in the space of either Q-functions or policies. In this work, we
introduce a regularizer in the space of stationary distributions to better handle
distributional shift. Our algorithm, ComaDICE, offers a principled framework for
offline cooperative MARL by incorporating stationary distribution regularization
for the global learning policy, complemented by a carefully structured multiagent value decomposition strategy to facilitate multi-agent training. Through
extensive experiments on the multi-agent _MuJoCo_ and _StarCraft II_ benchmarks,
we demonstrate that ComaDICE achieves superior performance compared to stateof-the-art offline MARL methods across nearly all tasks.


1 I NTRODUCTION


Over the years, deep RL has achieved remarkable success in various decision-making tasks (Levine
et al., 2016; Silver et al., 2017; Kalashnikov et al., 2018; Haydari & Yılmaz, 2020). However, a
significant limitation of deep RL is its need for millions of interactions with the environment to gather
experiences for policy improvement. This process can be both costly and risky, especially in realworld applications like robotics and healthcare. To address this challenge, offline RL has emerged,
enabling policy learning based solely on pre-collected demonstrations (Levine et al., 2020). Despite
this advancement, offline RL faces a critical issue: the distribution shift between the offline dataset and
the learned policy (Kumar et al., 2019). This distribution shift complicates value estimation for unseen
states and actions during policy evaluation, resulting in extrapolation errors where out-of-distribution
(OOD) state-action pairs are assigned unrealistic values (Fujimoto et al., 2018).


To tackle OOD actions, many existing works impose action-level constraints, either implicitly by
regulating the learned value functions or explicitly through distance or divergence penalties (Fujimoto
et al., 2019; Kumar et al., 2019; Wu et al., 2019; Peng et al., 2019; Fujimoto & Gu, 2021; Xu et al.,
2021). Only a few recent studies have addressed both OOD actions and states using state-action-level


1


Published as a conference paper at ICLR 2025


behavior constraints (Li et al., 2022; Zhang et al., 2022; Lee et al., 2021; 2022; Mao et al., 2024). In
particular, there is an important line of work on DIstribution Correction Estimation (DICE) (Nachum
& Dai, 2020) that constrains the distance in terms of the joint state-action occupancy measure between
the learning policy and the offline policy. These DICE-based methods have demonstrated impressive
performance results on the D4RL benchmarks (Lee et al., 2021; 2022; Mao et al., 2024).


It is important to note that that all the aforementioned offline RL approaches primarily focus on the
single-agent setting. While multi-agent setting is prevalent in many real-world sequential decisionmaking tasks, offline MARL remains a relatively under-explored area. The multi-agent setting poses
significantly greater challenges due to the large joint state-action space, which expands exponentially
with the number of agents, as well as the inter-dependencies among the local policies of different
agents. As a result, the offline data distribution can become quite sparse in these high-dimensional
joint action spaces, leading to an increased number of OOD state-action pairs and exacerbating
extrapolation errors. A few recent studies have sought to address the negative effects of sparse data
distribution in offline MARL by adapting the well-known centralized training decentralized execution
(CTDE) paradigm from online MARL (Oliehoek et al., 2008; Kraemer & Banerjee, 2016), enabling
data-related regularization at the individual agent level. Notably, some of these works (Pan et al.,
2022; Shao et al., 2024; Wang et al., 2022b) extend popular offline single-agent RL algorithms, such
as CQL (Kumar et al., 2020) and SQL/EQL (Xu et al., 2023), within the CTDE framework.


In our work, we focus on addressing the aforementioned challenges in offline cooperative MARL.
In particular, we follow the DICE approach to address both OOD states and actions, motivated by
remarkable performance of recent DICE-based methods in offline single-agent RL. Similar to previous
works in offline MARL, we adopt the CTDE framework to handle exponential joint state-action
spaces in the multi-agent setting. We remark that extending the DICE approach under this CTDE
framework is not straightforward given the complex objective of DICE that involves the f-divergence
in stationary distribution between the learning joint policy and the behavior policy. Therefore, the
value decomposition in CTDE needs to be carefully designed to ensure the consistency in optimality
between the global and local policies. In particular, we provide the following main contributions:


    - We propose ComaDICE, a new offline MARL algorithm that integrates DICE with a carefully
designed value decomposition strategy. In ComaDICE, under the CTDE framework, we
decompose both the global value function _ν_ _[tot]_ and the global advantage functions _A_ _[tot]_ _ν_ [,]
rather than using Q-functions as in previous MARL works. This unique factorization
approach allows us to theoretically demonstrate that the global learning objective in DICE
is convex in local values, provided that the mixing network used in the value decomposition
employs non-negative weights and convex activation functions. This significant finding
ensures that our decomposition strategy promotes an efficient and stable training process.

    - Building on our decomposition strategy, we demonstrate that finding an optimal global
policy can be divided into multiple sub-problems, each aims to identify a local optimal
policy for an individual agent. We provide a theoretical proof that the global optimal policy
is, in fact, equivalent to the product of the local policies derived from these sub-problems.

    - Finally, we conduct extensive experiments to evaluate the performance of our algorithm,
ComaDICE, in complex MARL environments, including: multi-agent StarCraft II (i.e.,
SMACv1 (Samvelyan et al., 2019), SMACv2 (Ellis et al., 2022)) and multi-agent Mujoco (de Witt et al., 2020) benchmarks. Our empirical results show that our ComaDICE
outperforms several strong baselines in all these benchmarks.


2 R ELATED W ORK


**Offline Reinforcement Learning (offline RL).** Offline RL focuses on learning policies from
pre-collected datasets without any further interactions with the environment (Levine et al., 2020;
Prudencio et al., 2023). A significant challenge in offline RL is the issue of distribution shift,
where unseen actions and states may arise during training and execution, leading to inaccurate
policy evaluations and suboptimal outcomes. Consequently, there is a substantial body of literature
addressing this challenge through various approaches (Prudencio et al., 2023). In particular, some
studies impose explicit or implicit policy constraints to ensure that the learned policy remains close
to the behavioral policy (Fujimoto et al., 2019; Kumar et al., 2019; Wu et al., 2019; Kostrikov et al.,
2021; Peng et al., 2019; Nair et al., 2020; Fujimoto & Gu, 2021; Xu et al., 2021; Cheng et al., 2024; Li


2


Published as a conference paper at ICLR 2025


et al., 2023). Others incorporate regularization terms into the learning objectives to mitigate the value
overestimation on OOD actions (Kumar et al., 2020; Kostrikov et al., 2021; Xu et al., 2022c; Niu et al.,
2022; Xu et al., 2023; Wang et al., 2022b). Uncertainty-based offline RL methods seek to balance
conservative approaches with naive off-policy RL techniques, relying on estimates of model, value,
or policy uncertainty (Agarwal et al., 2020; An et al., 2021; Bai et al., 2022). Offline model-based
algorithms focus on conservatively estimating the transition dynamics and reward functions based on
the pre-collected datasets (Kidambi et al., 2020; Yu et al., 2020; Matsushima et al., 2020; Yu et al.,
2021). Some other methods impose action-level regularization through imitation learning techniques
(Xu et al., 2022b; Chen et al., 2020; Zhang et al., 2023; Zheng et al., 2024; Brandfonbrener et al.,
2021; Xu et al., 2022a). Finally, while a majority of previous works target OOD actions only, there
are a few recent works attempt to address both OOD states and actions (Li et al., 2022; Zhang et al.,
2022; Lee et al., 2021; 2022; Sikchi et al., 2023; Mao et al., 2024). Our work on offline MARL follow
the DICE-based approach, as motivated by compelling performance of DICE-based algorithms in
single-agent settings (Lee et al., 2021; 2022; Sikchi et al., 2023; Mao et al., 2024).


**Offline Multi-agent Reinforcement Learning (offline MARL).** While there is a substantial body
of literature on offline single-agent RL, research on offline MARL remains limited. Offline MARL
faces challenges from both distribution shift—characteristic of offline settings—and the exponentially
large joint action space typical of multi-agent environments. Recent studies have begun to merge
advanced methodologies from both offline RL and MARL to address these challenges (Yang et al.,
2021; Pan et al., 2022; Shao et al., 2024; Wang et al., 2022b) Specifically, these works employ local
policy regularization within the centralized training with decentralized execution (CTDE) framework
to mitigate distribution shift. The CTDE paradigm, well-established in online MARL, facilitates more
efficient and stable learning while allowing agents to operate in a decentralized manner (Oliehoek
et al., 2008; Kraemer & Banerjee, 2016). For instance, Yang et al. (2021) utilize importance sampling
to manage local policy learning on OOD samples. Both works by Pan et al. (2022) and Shao
et al. (2024) are built upon CQL (Kumar et al., 2020), a prominent offline RL algorithm for singleagent scenarios. Matsunaga et al. (2023) developed AlberDICE, leveraging the Nash equilibrium
solution concept from game theory to iteratively update the best responses of individual agents. Both
AlberDICE and our method, ComaDICE, adopt the DICE framework to address the out-of-distribution
(OOD) issue. However, while AlberDICE proposes learning individual Lagrange multipliers (or
value functions) to obtain occupancy ratios, our ComaDICE algorithm learns a global value function
by mixing local functions, adhering to the well-established CTDE principle. This design enables
ComaDICE to better capture inter-agent relationships and improve credit assignment across local
agents. Finally, OMIGA (Wang et al., 2022b) establishes the equivalence between global and local
value regularization within a _policy constraint framework_, making it the current state-of-the-art
algorithm in offline MARL. The key difference between ComaDICE and OMIGA lies in their
respective approaches: OMIGA focuses on learning a global Q-function, whereas our algorithm (and
other methods in the DICE family) operates in the occupancy space, aiming to learn the ratio between
the occupancy of the learning policy and the behavior policy.


Beyond this main line of research, some studies formulate offline MARL as a sequence modeling
problem, employing supervised learning techniques to tackle the issue (Meng et al., 2023; Tseng
et al., 2022), while others adhere to decentralized approaches (Jiang & Lu, 2023).


3 P RELIMINARIES


Our work focuses on cooperative multi-agent RL, which can be modeled as a multi-agent Partially Observable Markov Decision Process (POMDP), defined by the tuple _M_ = _⟨S, A, P, r, Z, O, n, N_ _, γ⟩_ .
Here, _n_ is number of agents, _N_ = _{_ 1 _, . . ., n}_ is the set of agents, **s** _∈_ _S_ represents the true state
of the multi-agent environment, and _A_ = [�] _i∈N_ _[A]_ _[i]_ [ is the set of joint actions, where] _[ A]_ _[i]_ [ is the set]

of individual actions available to agent _i ∈N_ . At each time step, each agent _i ∈{_ 1 _,_ 2 _, . . ., n}_
selects an action _a_ _i_ _∈A_ _i_, forming a joint action **a** = ( _a_ 1 _, a_ 2 _, . . ., a_ _n_ ) _∈A_ . The transition dynamics _P_ ( **s** _[′]_ _|_ **s** _,_ **a** ) : _S × A × S →_ [0 _,_ 1] describe the probability of transitioning to the next state
**s** _[′]_ when agents take an action **a** from the current state **s** . The discount factor _γ ∈_ [0 _,_ 1) represents
the weight given to future rewards. In a partially observable environment, each agent receives a
local observation _s_ _i_ _∈O_ _i_ based on the observation function _Z_ _i_ ( **s** ) : _S →O_ _i_, and we denote the
joint observation as **o** = ( _o_ 1 _, o_ 2 _, . . ., o_ _n_ ) . In cooperative MARL, all agents share a global reward


3


Published as a conference paper at ICLR 2025


function _r_ ( **s** _,_ **a** ) : _S × A →_ R . The goal of all agents is to learn a joint policy _**π**_ tot = _{π_ 1 _, . . ., π_ _n_ _}_
that collectively maximize the expected discounted returns E ( **o** _,_ **a** ) _∼_ _**π**_ tot [ [�] _[∞]_ _t_ =0 _[γ]_ _[t]_ _[r]_ [(] **[s]** _[t]_ _[,]_ **[ a]** _[t]_ [)]] [. In the]
offline MARL setting, a pre-collected dataset _D_ is obtained by sampling from a behavior policy
_µ_ tot = _{µ_ 1 _, . . ., µ_ _n_ _}_, and the policy learning is conducted soly based on _D_, with no interactions with
the environment. We also define the occupancy measure (or stationary distribution) as follows:


_∞_
_ρ_ _**[π]**_ _[tot]_ ( **s** _,_ **a** ) = (1 _−_ _γ_ ) � _t_ =0 _[P]_ [(] **[s]** _[t]_ [ =] **[ s]** _[,]_ **[ a]** _[t]_ [ =] **[ a]** [)]


which represents distribution visiting the pair (observation, action) ( **s** _t_ _,_ **a** 1 ) when following the joint
policy _**π**_ _tot_, where **s** 0 _∼_ _P_ 0 _,_ **a** _t_ _∼_ _**π**_ _tot_ ( _·|_ **s** _t_ ) and **s** _t_ +1 _∼_ _P_ ( _·|_ **s** _t_ _,_ **a** _t_ ).


4 C OMA DICE: O FFLINE C OOPERATIVE M ULTI -A GENT RL WITH

S TATIONARY D ISTRIBUTION C ORRECTION E STIMATION


We consider an offline cooperative MARL problem where the goal is to optimize the expected
discounted joint reward. In this work, we focus on the DICE objective function Nachum & Dai
(2020); Lee et al. (2021), which incorporates a stationary distribution regularizer to capture the
divergence between the occupancy measures of the learning policy, _**π**_ _tot_, and the behavior policy,
_**µ**_ _tot_, formulated as follows:

max _**π**_ _tot_ E ( **s** _,_ **a** ) _∼ρ_ _**π**_ _tot_ [ _r_ ( **s** _,_ **a** )] _−_ _αD_ _[f]_ ( _ρ_ _**[π]**_ _[tot]_ _∥_ _ρ_ _**[µ]**_ _[tot]_ ) (1)

where _D_ _[f]_ ( _ρ_ _**[π]**_ _[tot]_ _∥_ _ρ_ _**[µ]**_ _[tot]_ ) = E ( **s** _,_ **a** ) _∼ρ_ _**π**_ _tot_ � _f_ � _ρρ_ _**[π][µ]**_ _[tot][tot]_ �� is the f-divergence between the stationary dis
tribution _ρ_ _**[π]**_ _[tot]_ of the learning policy and _ρ_ _**[µ]**_ _[tot]_ of the behavior policy. In this work, we consider _f_ ( _·_ ) to
be strictly convex and differentiable. The parameter _α_ controls the trade-off between maximizing the
reward and penalizing deviation from the offline dataset’s distribution (i.e., penalizing distributional
shift). When _α_ = 0, the problem becomes the standard offline MARL, where the objective is to find
a joint policy that maximizes the expected joint reward. On the other hand, when _α ≫_ 1, the problem
shifts towards imitation learning, aiming to closely mimic the behavioral policy.


This DICE-based approach offers the advantage of better capturing the system dynamics inherent in
the offline data. Such stationary distributions, _ρ_ _**[π]**_ _[tot]_ and _ρ_ _**[µ]**_ _[tot]_, however, are not directly available. We
will discuss how to estimate them in the next subsection.


4.1 C ONSTRAINED O PTIMIZATION IN THE S TATIONARY D ISTRIBUTION S PACE


We first formulate the learning problem in Eq. 1 as a constrained optimization on the space of _ρ_ _**[π]**_ _[tot]_ :


max _ρ_ _**[π]**_ _tot_ E ( **s** _,_ **a** ) _∼ρ_ _**π**_ _tot_ [ _r_ ( **s** _,_ **a** )] _−_ _αD_ _[f]_ ( _ρ_ _**[π]**_ _[tot]_ _∥_ _ρ_ _**[µ]**_ _[tot]_ ) (2)



_s.t._
�



**a** _[′]_ _[ ρ]_ _**[π]**_ _[tot]_ [(] **[s]** _[,]_ **[ a]** _[′]_ [) = (1] _[ −]_ _[γ]_ [)] _[p]_ [0] [(] **[s]** [) +] _[ γ]_ �



(3)
**a** _[′]_ _,_ **s** _[′]_ _[ ρ]_ _**[π]**_ _[tot]_ [(] **[s]** _[′]_ _[,]_ **[ a]** _[′]_ [)] _[P]_ [(] **[s]** _[|]_ **[a]** _[′]_ _[,]_ **[ s]** _[′]_ [)] _[,][ ∀]_ **[s]** _[ ∈S][.]_



When _f_ is convex, (2-3) becomes a convex optimization problem, as it involves maximizing a concave
objective function subject to linear constraints. We now consider the Lagrange dual of (2-3):



��



_L_ ( _ν_ _[tot]_ _,ρ_ _**[π]**_ _[tot]_ ) = E ( **s** _,_ **a** ) _∼ρ_ _**π**_ _tot_ [ _r_ ( **s** _,_ **a** )] _−_ _α_ E ( **s** _,_ **a** ) _∼ρ_ _**µ**_ _tot_



_ρ_ _**π**_ _tot_ ( **s** _,_ **a** )
_f_
� � _ρ_ _**[µ]**_ _[tot]_ ( **s** _,_ **a** )



**s** _[ν]_ _[tot]_ [(] **[s]** [)] ��



_,_ (4)
**a** _[′]_ _,_ **s** _[′]_ _[ ρ]_ _**[π]**_ _[tot]_ [(] **[s]** _[′]_ _[,]_ **[ a]** _[′]_ [)] _[P]_ [(] **[s]** _[|]_ **[a]** _[′]_ _[,]_ **[ s]** _[′]_ [)] �



_−_
�



**a** _[′]_ _[ ρ]_ _**[π]**_ _[tot]_ [(] **[s]** _[,]_ **[ a]** _[′]_ [)] _[ −]_ [(1] _[ −]_ _[γ]_ [)] _[p]_ [0] [(] **[s]** [)] _[ −]_ _[γ]_ �



where _ν_ _[tot]_ ( **s** ) is a Lagrange multiplier. Since (2-3) is a convex optimization problem, it is equivalent to
the following minimax problem over the spaces of _ν_ _[tot]_ and _ρ_ _**[π]**_ _[tot]_ : min _ν_ _tot_ max _ρ_ _**[π]**_ _tot_ _{L_ ( _ν_ _[tot]_ _, ρ_ _**[π]**_ _[tot]_ ) _} ._
Furthermore, we observe that _L_ ( _ν_ _[tot]_ _, ρ_ _**[π]**_ _[tot]_ ) is linear in _ν_ _[tot]_ and concave in _ρ_ _**[π]**_ _[tot]_, so
the minimax problem has a saddle point, implying: min _ν_ _tot_ max _ρ_ _**[π]**_ _tot_ _{L_ ( _ν_ _[tot]_ _, ρ_ _**[π]**_ _[tot]_ ) _}_ =
max _ρ_ _**[π]**_ _tot_ min _ν_ _tot_ _{L_ ( _ν_ _[tot]_ _, ρ_ _**[π]**_ _[tot]_ ) _} ._ In a manner analogous to the single-agent case (Lee et al., 2021),
_ρ_ _**[π]**_ _[tot]_ ( **s** _,_ **a** )
by defining _w_ _ν_ _[tot]_ [(] **[s]** _[,]_ **[ a]** [) =] _ρ_ _**[µ]**_ _[tot]_ ( **s** _,_ **a** ) [, the Lagrange dual function can be simplified into the more]
compact form (with detailed derivations are in the appendix):


_L_ ( _ν_ _[tot]_ _, w_ _[tot]_ ) = (1 _−_ _γ_ )E **s** _∼p_ 0 [ _ν_ _[tot]_ ( **s** )] + E ( **s** _,_ **a** ) _∼ρ_ _**µ**_ _tot_ � _−αf_ � _w_ _ν_ _[tot]_ [(] **[s]** _[,]_ **[ a]** [)] � + _w_ _ν_ _[tot]_ [(] **[s]** _[,]_ **[ a]** [)] _[A]_ _[tot]_ _ν_ [(] **[s]** _[,]_ **[ a]** [)] � _,_


4


Published as a conference paper at ICLR 2025


where _A_ _[tot]_ _ν_ is an “advantage function” defined based on _ν_ _[tot]_ as:

_A_ _[tot]_ _ν_ [(] **[s]** _[,]_ **[ a]** [) =] _[ q]_ _[tot]_ [(] **[s]** _[,]_ **[ a]** [)] _[ −]_ _[ν]_ _[tot]_ [(] **[s]** [)] _[,]_ (5)

with _q_ _[tot]_ ( **s** _,_ **a** ) = _r_ ( _Z_ ( **s** ) _,_ **a** )+ _γ_ E **s** _′_ _∼P_ ( _·|_ **s** _,_ **a** ) [ _ν_ _[tot]_ ( **s** _[′]_ )] . It is important to note that _ν_ _[tot]_ ( **s** ) and _q_ _[tot]_ ( **s** _,_ **a** )
can be interpreted as a value function and a Q function, respectively, arising from the decomposition
of the stationary distribution regularizer. We can now write the learning problem as follows:

min _ν_ _tot_ max _w_ _tot_ _≥_ 0 _{L_ ( _ν_ _[tot]_ _, w_ _[tot]_ ) _}._ (6)


It can be observed that _L_ ( _ν_ _[tot]_ _, w_ _[tot]_ ) is linear in _ν_ _[tot]_ and concave in _w_ _[tot]_, which ensures wellbehaved properties in both the _ν_ _[tot]_ - and _w_ _[tot]_ -spaces. Following the derivations in Lee et al. (2021),
a key feature of the above minimax problem is that the inner maximization problem has a closedform solution, which greatly simplifies the minimax problem, making it no longer adversarial. We
formalize this result as follows:
**Proposition 4.1.** _The minimax problem in Eq. 6 is equivalent to_ min _ν_ _tot_ [��] _L_ ( _ν_ _[tot]_ )� _, where_



_._
��



�
_L_ ( _ν_ _[tot]_ ) = (1 _−_ _γ_ )E _**s**_ _∼p_ 0 [ _ν_ _[tot]_ ( _**s**_ )] + E ( _**s**_ _,_ _**a**_ ) _∼ρ_ _**µ**_ _tot_



_A_ _totν_ [(] _**[s]**_ _[,]_ _**[ a]**_ [)]
_αf_ _[∗]_
� � _α_



_Here,_ _f_ _[∗]_ _is convex conjugate of_ _f_ _, i.e.,_ _f_ _[∗]_ ( _y_ ) = sup _t≥_ 0 _{ty−f_ ( _t_ ) _}_ _. Moreover, if_ _ν_ _[tot]_ _is parameterized_
_by θ, the first-order derivative of_ _L_ [�] ( _ν_ _[tot]_ ) _w.r.t. θ is given as follows:_

_∇_ _θ_ _L_ [�] ( _ν_ _[tot]_ ) = (1 _−_ _γ_ )E _**s**_ _∼p_ 0 [ _∇_ _θ_ _ν_ _[tot]_ ( _**s**_ )] + E ( _**s**_ _,_ _**a**_ ) _∼ρ_ _**µ**_ _tot_ � _∇_ _θ_ _A_ _[tot]_ _ν_ [(] _**[s]**_ _[,]_ _**[ a]**_ [)] _[w]_ _ν_ _[tot][∗]_ ( _**s**_ _,_ _**a**_ )� _._

_where_ _w_ _ν_ _[tot][∗]_ ( _s, a_ ) = max _{_ 0 _, f_ _[′−]_ [1] ( _A_ _[tot]_ _ν_ [(] _**[s]**_ _[,]_ _**[ a]**_ [)] _[/α]_ [)] _[}]_ _[, with]_ _[ f]_ _[ ′−]_ [1] [(] _[·]_ [)] _[ is the inverse function of the first-]_
_order derivative of f_ _._


Proposition 4.1 above is a direct extension of the formulations in Lee et al. (2021) developed for the
single-agent setting, differing only in the inclusion of the closed-form expression for the first-order
derivative of the objective function, _L_ [�] ( _ν_ _[tot]_ ).


4.2 V ALUE F ACTORIZATION


Directly optimizing min _ν_ _tot_ _{L_ ( _ν_ _[tot]_ _, w_ _ν_ _[tot][∗]_ ) _}_ in multi-agent settings is generally impractical due to
the large state and action spaces. Therefore, we follow the idea of value decomposition in the wellknown CTDE framework in cooperative MARL to address this computational challenge. However, it
is not straightforward to extend the DICE approach within this CTDE framework due to the complex
objective of DICE, which involves the f-divergence between the learned joint policy and the behavior
policy in stationary distributions. Thus, it is crucial to carefully design the value decomposition in
CTDE to ensure optimality consistency between the global and local policies.


Specifically, we adopt a factorization approach that decomposes the value function _ν_ _[tot]_ ( **s** )
(or global Lagrange multipliers) into local values using mixing network architectures. Let
_**ν**_ ( **s** ) = _{ν_ 1 ( _s_ 1 ) _, . . ., ν_ _n_ ( _s_ _n_ ) _}_ represent a collection of local “value functions” and let **A** _**ν**_ ( **s** _,_ **a** ) =
_{A_ _i_ ( _s_ _i_ _, a_ _i_ ) _, i_ = 1 _, ..., n}_ represent a collection of local advantage functions. The local advantage functions are computed as _A_ _i_ ( _s_ _i_ _, a_ _i_ ) = _q_ _i_ ( _s_ _i_ _, a_ _i_ ) _−_ _ν_ _i_ ( _s_ _i_ ) for all _i ∈N_, where
**q** ( **s** _,_ **a** ) = _{q_ _i_ ( _s_ _i_ _, a_ _i_ ) _, i_ = 1 _, ..., n}_ is a vector of local Q functions. To facilitate centralized
learning, we create a mixing network, _M_ _θ_, where _θ_ are the learnable weights, that aggregates the
local values to form the global value and advantage functions as follows:

_ν_ _[tot]_ ( **s** _,_ **a** ) = _M_ _θ_ [ _**ν**_ ( **s** )] _,_ _A_ _[tot]_ _ν_ [(] **[s]** _[,]_ **[ a]** [) =] _[ M]_ _[θ]_ [[] **[q]** [(] **[s]** _[,]_ **[ a]** [)] _[ −]_ _**[ν]**_ [(] **[s]** [)]] _[,]_


where each network takes the vectors _**ν**_ ( **s** ) or **A** _**ν**_ ( **s** _,_ **a** ) as inputs and outputs _ν_ _[tot]_ and _A_ _[tot]_ _ν_ [, respectively.]
Under this architecture, the learning objective becomes:



_,_
��



�
_L_ ( _**ν**_ _, θ_ ) = (1 _−_ _γ_ )E **s** _∼p_ 0 [ _M_ _θ_ [ _**ν**_ ( **s** )]] + E ( **s** _,_ **a** ) _∼ρ_ _**µ**_ _tot_



_M_ _θ_ [ **q** ( **s** _,_ **a** ) _−_ _**ν**_ ( **s** )]
_αf_ _[∗]_
� � _α_



with the observation that **A** _ν_ ( **s** _,_ **a** ) can be expressed as a linear function of _**ν**_ . There are different
ways to construct the mixing network _M_ _θ_ ; previous work often employs a single linear combination
(1-layer network) or a two-layer network with convex activations such as ReLU, ELU, or Maxout. In
the following, we show a general result stating that the learning objective function is convex in _**ν**_,
provided that the mixing network is constructed with nonnegative weights and convex activations.


5


Published as a conference paper at ICLR 2025


**Theorem 4.2.** _If the mixing network_ _M_ _θ_ [ _·_ ] _is constructed with non-negative weights and convex_
_activations, then_ _L_ [�] ( _**ν**_ _, θ_ ) _is convex in_ _**ν**_ _._


Mixing networks with non-negative weights and concave activations (e.g., ELU or ReLU) have been
extensively used in MARL, forming the foundation of several notable state-of-the-art algorithms such
as QMIX (Rashid et al., 2020), QTRAN (Son et al., 2019), and MFIQ (Bui et al., 2024). In particular,
it has been demonstrated that mixing networks with either negative weights or non-concave activations
result in significantly degraded performance (Bui et al., 2024). Theorem 4.2 shows that _L_ [�] ( _**ν**_ _, θ_ ) is
convex in _**ν**_ when using _any multi-layer feed-forward mixing networks with non-negative weights and_
_convex activation functions_ . This finding is highly general and non-trivial, given the nonlinearity and
complexity of both the function (in terms of _**ν**_ ) and the mixing networks. Previous work has often
focused on single-layer (Wang et al., 2022b) or two-layer mixing structures (Rashid et al., 2020; Bui
et al., 2024), emphasizing that such two-layer networks can approximate any monotonic function
arbitrarily closely as network width approaches infinity (Dugas et al., 2009). In our experiments,
we test two configurations for the mixing network: a linear combination (or 1-layer) and a 2-layer
feed-forward network. While 2-layer mixing structures have shown strong performance in online
MARL (Rashid et al., 2020; Son et al., 2019; Wang et al., 2020), we observe in our offline settings
that the linear combination approach provides more stable results.


4.3 P OLICY E XTRACTION


Let _**ν**_ _[∗]_ be an optimal solution to the training problem with mixing networks, i.e.,


�
min _L_ ( _**ν**_ _, θ_ ) _._ (7)
_**ν**_ _,θ_


We now need to extract a local and joint policy from this solution. Based on Prop. 4.1, given _**ν**_ _[∗]_, we can
compute this occupancy ratio as follows: : _w_ _[tot][∗]_ ( **s** _,_ **a** ) = max �0 _, f_ _[′−]_ [1] [�] _M_ _θ_ [ **A** _**ν**_ _α_ _∗_ ( **s** _,_ **a** )] �� _._ The global

_w_ _[tot][∗]_ ( **s** _,_ **a** ) _·ρ_ _**[µ]**_ _[tot]_ ( **s** _,_ **a** )
policy can then be obtained as follows: _**π**_ _[∗]_ _tot_ [(] **[a]** _[|]_ **[s]** [) =] ~~�~~ **a** _[′]_ _∈A_ _[w]_ _[tot][∗]_ [(] **[s]** _[,]_ **[a]** _[′]_ [)] _[·][ρ]_ _**[µ]**_ _[tot]_ [(] **[s]** _[,]_ **[a]** _[′]_ [)] _[.]_ [ This computation,]

however, is not practical since _ρ_ _**[µ]**_ _[tot]_ is generally not available and might not be accurately estimated
in the offline setting. A more practical way to estimate the global policy, _**π**_ _[∗]_ _tot_ [, as the result of solving]
the following weighted behavioral cloning (BC):


max _tot_ [[log] _**[π]**_ _[tot]_ [(] **[a]** _[|]_ **[s]** [)] =] max (8)
_**π**_ _tot_ _∈_ Π _tot_ [E] [(] **[s]** _[,]_ **[a]** [)] _[∼][ρ]_ _**[π]**_ _[∗]_ _**π**_ _tot_ _∈_ Π _tot_ [E] [(] **[s]** _[,]_ **[a]** [)] _[∼][ρ]_ _**[µ]**_ _[tot]_ [[] _[w]_ _[tot][∗]_ [(] **[s]** _[,]_ **[ a]** [) log] _**[π]**_ _[tot]_ [(] **[a]** _[|]_ **[s]** [)]] _[,]_


where Π _tot_ represents the feasible set of global policies. Here we assume that Π _tot_ contains decomposable global policies, i.e., Π _tot_ = _{_ _**π**_ _tot_ _| ∃π_ _i_ _, ∀i ∈N_ such that _**π**_ _tot_ ( **a** _|_ **s** ) = [�] _i∈N_ _[π]_ _[i]_ [(] _[a]_ _[i]_ _[|][s]_ _[i]_ [)] _[}]_ [. In]

other words, Π _tot_ consists of global policies that can be expressed as a product of local policies. This
decomposability is highly useful for decentralized learning and has been widely adopted in MARL
(Wang et al., 2022b; Bui et al., 2024; Zhang et al., 2021).


While the above weighted BC appears practical, as ( **s** _,_ **a** ) can be sampled from the offline dataset
generated by _ρ_ _**[π]**_ _[tot]_, and since _w_ _[tot][∗]_ ( **s** _,_ **a** ) is available from solving 7, it does not directly yield local
policies, which are essential for decentralized execution. To address this, we propose solving the
following weighted BC for each local agent _i ∈N_ :


max _π_ _i_ E ( **s** _,_ **a** ) _∼D_ � _w_ _[tot][∗]_ ( **s** _,_ **a** ) log _π_ _i_ ( _a_ _i_ _|s_ _i_ )� _._ (9)


This local WBC approach has several attractive properties. First, _w_ _[tot][∗]_ ( **s** _,_ **a** ) appears explicitly in the
local policy optimization and is computed from global observations and actions. This enables local
policies to be optimized with global information, ensuring consistency with the credit assignment in
the multi-agent system. Furthermore, as shown in Proposition 4.3 below, the optimization of local
policies through local WBC is highly consistent with the global weighted BC in 8.
**Proposition 4.3.** _Let_ _π_ _i_ _[∗]_ _[be the optimal solution to the local weighted BC 9. Then]_ _**[ π]**_ _tot_ _[∗]_ [(] _**[a]**_ _[|]_ _**[s]**_ [) =]
� _i∈N_ _[π]_ _i_ _[∗]_ [(] _[a]_ _[i]_ _[|][s]_ _[i]_ [)] _[ is also optimal for the global weighted BC in 8.]_


Here we note that consistency between global and local policies is a critical aspect of centralized
training with CTDE. Previous MARL approaches typically achieve this by factoring Q or V functions
into local functions and training local policies based on these local functions (Rashid et al., 2020;
Wang et al., 2020; Bui et al., 2024). However, in our case, there are key differences that prevent us


6


Published as a conference paper at ICLR 2025


from employing such local values to derive local policies. Specifically, we factorize the Lagrange
multipliers _ν_ _[tot]_ to train the stationary distribution ratio _w_ _[tot]_ . Although local _w_ values can be extracted
from local _ν_ _i_, these local _w_ values do not represent a local stationary distribution ratio and therefore
cannot be used to recover local policies.


5 P RACTICAL A LGORITHM


Let _D_ represent the offline dataset, consisting of sequences of local observations and actions gathered
from a global behavior policy _**π**_ _tot_ . To train the value function _**ν**_, we construct a value network
_ν_ _i_ ( _s_ _i_ ; _ψ_ _ν_ ) for each local agent _i_, along with a network for each local Q-function _q_ _i_ ( _s_ _i_ _, a_ _i_ ; _ψ_ _q_ ), where
_ψ_ _ν_ and _ψ_ _q_ are learnable parameters for the local value and Q-functions. We note that the introduction
and learning of the Q-functions are intended to facilitate the decomposition of the advantage function,
_A_ _[tot]_ _ν_ [. In our multi-agent setting, the absence of local rewards makes it difficult to directly compute]
local advantage functions. To overcome this challenge, we learn local Q-functions, which are then
used to derive the local advantage functions. Additionally, as explained below, a MSE is optimized to
ensure that the global Q-function and state-value function align properly with the global rewards.


Now, each local advantage function is then calculated as follows: The global value function and
advantage function are subsequently aggregated using two mixing networks with a shared set of
learnable parameters _θ_ :


_ν_ _[tot]_ ( **s** ) = _M_ **[s]** _θ_ [[] _**[ν]**_ [(] **[s]** [;] _[ ψ]_ _[ν]_ [)]] _[,]_ _A_ _[tot]_ _ν_ [(] **[s]** _[,]_ **[ a]** [) =] _[ M]_ **[s]** _θ_ [[] **[q]** [(] **[s]** _[,]_ **[ a]** [;] _[ ψ]_ _[q]_ [)] _[ −]_ _**[ν]**_ [(] **[s]** [;] _[ ψ]_ _[ν]_ [)]] _[,]_


where _M_ **[s]** _θ_ [[] _[·]_ []] [ represents a linear combination of its inputs with non-negative weights, such that]
_M_ **[s]** _θ_ [[] _**[ν]**_ [(] **[s]** [;] _[ ψ]_ _[ν]_ [)] =] _**[ν]**_ [(] **[s]** [;] _[ ψ]_ _[ν]_ [)] _[⊤]_ _[W]_ **[ s]** _θ_ [+] _[ b]_ **[s]** _θ_ [, where] _[ W]_ **[ s]** _θ_ [and] _[ b]_ **[s]** _θ_ [are weights of the mixing network.] [1] [ It is]
important to note that _W_ _θ_ **[s]** [and] _[ b]_ **[s]** _θ_ [are generated by hyper-networks that take the global state] **[ s]** [ and]
the learnable parameters _θ_ as inputs. In this context, we employ the same mixing network _M_ **[s]** _θ_ [to]
combine the local values and advantages. However, our framework is flexible enough to allow the
use of two different mixing networks for _ν_ _[tot]_ and _A_ _[tot]_ _ν_ [.]


In our setting, the relationship between the global Q-function, value, and advantage functions is
described in Eq. 5. Specifically, we have: _A_ _[tot]_ _ν_ [(] **[s]** _[,]_ **[ a]** [) =] _[ r]_ [(] _[Z]_ [(] **[s]** [)] _[,]_ **[ a]** [) +] _[ γ]_ [E] **s** _[′]_ _∼P_ ( _·|_ **s** _,_ **a** ) [[] _[ν]_ _[tot]_ [(] **[s]** _[′]_ [)]] _[ −]_ _[ν]_ _[tot]_ [(] **[s]** [)] _[.]_
To capture this relationship, we train the Q-function by optimizing the following MSE loss:


2

min **q** � ( **s** _,_ **a** _,_ **s** _[′]_ ) _∼D_ � _A_ _[tot]_ _ν_ [(] **[s]** _[,]_ **[ a]** [)] _[ −]_ _[r]_ [(] _[Z]_ [(] **[s]** [)] _[,]_ **[ a]** [) +] _[ γν]_ _[tot]_ [(] **[s]** _[′]_ [)] _[ −]_ _[ν]_ _[tot]_ [(] **[s]** [)] � _._


This is equivalent to:



min _ψ_ _q_ _L_ _q_ ( _ψ_ _q_ ) = � ( **s** _,_ **a** _,_ **s** _[′]_ ) _∼D_



_M_ **[s]** _θ_ [[] **[q]** [(] **[s]** _[,]_ **[ a]** [;] _[ ψ]_ _[q]_ [)] _[ −]_ _**[ν]**_ [(] **[s]** [;] _[ ψ]_ _[ν]_ [)]]
�


2
_−_ _r_ ( _Z_ ( **s** ) _,_ **a** ) + _γM_ **[s]** _θ_ _[′]_ [[] _**[ν]**_ [(] **[s]** _[′]_ [;] _[ ψ]_ _[ν]_ [)]] _[ −M]_ **[s]** _θ_ [[] _**[ν]**_ [(] **[s]** [;] _[ ψ]_ _[ν]_ [)]] _._ (10)
�



For the primary loss function used to train the value function, we leverage transitions from the offline
dataset to approximate the objective _L_ [�], resulting in the following loss function for offline training:



_._ (11)
��



�
_L_ ( _ψ_ _ν_ _, θ_ ) = (1 _−γ_ )E **s** 0 _∼D_ [ _M_ **[s]** _θ_ [0] [[] _**[ν]**_ [(] **[s]** [0] [;] _[ ψ]_ _[ν]_ [)]]+][E] [(] **[s]** _[,]_ **[a]** [)] _[∼D]_



_M_ **s** _θ_ [[] **[q]** [(] **[s]** _[,]_ **[ a]** [;] _[ψ]_ _[q]_ [)] _[ −]_ _**[ν]**_ [(] **[s]** [;] _[ψ]_ _[ν]_ [)]]
_αf_ _[∗]_
� � _α_



As mentioned, after obtaining ( _**ν**_ _[∗]_ _, θ_ _[∗]_ ) by solving min _ψ_ _ν_ _,θ_ _L_ [�] ( _ψ_ _ν_ _, θ_ ), we compute the occupancy ratio:

_M_ **[s]** _θ_ _[∗]_ [[] _**[ν]**_ _[∗]_ [(] **[s]** [)]] _[−M]_ **[s]** _θ_ _[∗]_ [[] **[q]** [(] **[s]** _[,]_ **[a]** [;] _[ψ]_ _[q]_ [)]]
_w_ _ν_ _[tot][∗]_ ( **s** _,_ **a** ) = max �0 _, f_ _[′−]_ [1] [ �] _α_ �� _._ To train the local policy _π_ _i_ ( _a_ _i_ _|s_ _i_ ), we

represent it using a policy network _π_ _i_ ( _a_ _i_ _|s_ _i_ ; _η_ _i_ ), where _η_ _i_ are the learnable parameters. The training
process involves optimizing the following weighted behavioral cloning (BC) objective:

max _η_ _i_ _L_ _π_ ( _η_ _i_ ) = � ( **s** _,_ **a** ) _∼D_ _[w]_ _ν_ _[tot][∗]_ ( **s** _,_ **a** ) log( _π_ _i_ ( _a_ _i_ _|s_ _i_ ; _η_ _i_ )) _._ (12)


Our ComaDICE algorithm consists of two primary steps. The first step involves estimating the
occupancy ratio _w_ _[tot][∗]_ from the offline dataset. The second step focuses on training the local policy


1 In our experiments, we use a single-layer mixing network due to its superior performance compared to a
two-layer structure, though our approach is general and can handle any multi-layer feed-forward mixing network.


7


Published as a conference paper at ICLR 2025


by solving the weighted BC problem using _w_ _[tot][∗]_ . In the first step, we simultaneously update the
Q-functions _ψ_ _q_, the mixing network parameters _θ_, and the value function _ψ_ _ν_, aiming to minimize the
mean squared error (MSE) in Eq. 10 while optimizing the main loss function in Eq. 11.


It is important to note that, in practical POMDP scenarios, the global state **s** is not directly accessible
during training and is instead represented by the joint observations **o** from the agents. For notational
convenience, we use the global state **s** in our formulation; however, in practice, it corresponds to the
joint observation _Z_ ( **s** ) . Specifically, terms like _ρ_ _**[µ]**_ _[tot]_ ( **s** _,_ **a** ) and _ν_ _[tot]_ ( **s** ) actually refer to _ρ_ _**[µ]**_ _[tot]_ ( **o** _,_ **a** )
and _ν_ _[tot]_ ( **o** ), where **o** = _Z_ ( **s** ).


6 E XPERIMENTS


6.1 E NVIRONMENTS


We utilize three standard MARL environments: SMACv1 (Samvelyan et al., 2019), SMACv2 (Ellis
et al., 2022), and Multi-Agent MuJoCo (MaMujoco) (de Witt et al., 2020), each offering unique
challenges and configurations for evaluating cooperative MARL algorithms.


**SMACv1.** SMACv1 is based on Blizzard’s StarCraft II. It uses the StarCraft II API and DeepMind’s
PySC2 to enable agent interactions with the game. SMACv1 focuses on decentralized micromanagement scenarios where each unit is controlled by an RL agent. Tasks like _2c_ ~~_v_~~ _s_ ~~_6_~~ _4zg_ and _5m_ ~~_v_~~ _s_ ~~_6_~~ _m_
are labeled hard, while _6h_ ~~_v_~~ _s_ ~~_8_~~ _z_ and _corridor_ are super hard. The offline dataset, provided by Meng
et al. (2023), was generated using MAPPO-trained agents (Yu et al., 2022).


**SMACv2.** In comparison to SMACv1, SMACv2 introduces increased randomness and diversity by
randomizing start positions, unit types, and modifying sight and attack ranges. This version includes
tasks such as _protoss_, _terran_, and _zerg_, with instances ranging from _5_ ~~_v_~~ _s_ ~~_5_~~ to _20_ ~~_v_~~ _s_ ~~_2_~~ _3_, increasing in
difficulty. Our offline dataset for SMACv2 was generated by running MAPPO for 10 million training
steps and collecting 1,000 trajectories, ensuring medium quality but comprehensive coverage of the
learning process. To the best of our knowledge, we are the first to explore SMACv2 in offline MARL,
whereas most prior work has used this environment in online settings.


**MaMujoco.** MaMujoco serves as a benchmark for continuous cooperative multi-agent robotic
control. Derived from the single-agent MuJoCo control suite in OpenAI Gym (Brockman et al.,
2016), it presents scenarios where multiple agents within a single robot must collaborate to achieve
tasks. The tasks include _Hopper-v2_, _Ant-v2_, and _HalfCheetah-v2_, with instances labeled as _expert_,
_medium_, _medium-replay_, and _medium-expert_ . The offline dataset was created by (Wang et al., 2022b)
using the HAPPO method (Wang et al., 2022a).


6.2 B ASELINES


We consider the following baselines, which represent either standard or state-of-the-art (SOTA)
methods for offline MARL: (i) **BC** (Behavioral Cloning); (ii) **BCQ** (Batch-Constrained Q-learning)
(Fujimoto et al., 2019) – an offline RL algorithm that constrains the policy to actions similar to
those in the dataset to reduce distributional shift, adapted for offline MARL settings; (iii) **CQL**
(Conservative Q-Learning) (Kumar et al., 2020) – a method that stabilizes offline Q-learning by
penalizing out-of-distribution actions, ensuring conservative value estimates; (iv) **ICQ** (Implicit
Constraint Q-learning) (Yang et al., 2021) – an approach using importance sampling to manage outof-distribution actions in multi-agent settings; (v) **OMAR** (Offline MARL with Actor Rectification)
(Pan et al., 2022) – a method combining CQL with optimization techniques to ensure the global
validity of local regularizations, promoting cooperative behavior; (vi) **OMIGA** (Offline MARL with
Implicit Global-to-Local Value Regularization) (Wang et al., 2022b) – a SOTA method that transforms
global regularizations into implicit local ones, optimizing local policies with global insights; (vii)
**OptDICE** - a naive extension of the OptDICE algorithm Lee et al. (2021) to multi-agent settings
where the global value function are directly learned without value factorization; and (viii) **AlberDICE**
Matsunaga et al. (2023) - an offline MARL algorithm which also leverages the DICE framework to
address the OOD.


We used experimental results contributed by the authors of OMIGA (Wang et al., 2022b) as our
baselines. They provided both the results and source code for all the baseline methods. This source


8


Published as a conference paper at ICLR 2025

|Instances|BC BCQ CQL ICQ OMAR OMIGA OptDICE AlberDICE ComaDICE<br>(ours)|
|---|---|
|2c~~ v~~s 64zg<br>poor<br>medium<br>good|0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>**0.6 ± 1.3**<br>1.9 ± 1.5<br>2.5 ± 3.6<br>2.5 ± 3.6<br>1.9 ± 1.5<br>1.2 ± 1.5<br>6.2 ± 5.6<br>1.0 ± 1.5<br>1.6 ± 1.6<br>**8.8 ± 7.0**<br>31.2 ± 9.9<br>35.6 ± 8.8<br>44.4 ± 13.0<br>28.7 ± 4.6<br>28.7 ± 9.1<br>40.6 ± 9.5<br>37.5 ± 3.1<br>42.2 ± 6.4<br>**55.0 ± 1.5**|
|5m~~ v~~s~~ 6~~m<br>poor<br>medium<br>good|2.5 ± 1.3<br>1.2 ± 1.5<br>1.2 ± 1.5<br>1.2 ± 1.5<br>0.6 ± 1.2<br>**6.9 ± 1.2**<br>0.0 ± 0.0<br>0.0 ± 0.0<br>4.4 ± 4.2<br>1.9 ± 1.5<br>1.2 ± 1.5<br>2.5 ± 1.2<br>1.2 ± 1.5<br>0.6 ± 1.2<br>2.5 ± 3.1<br>0.0 ± 0.0<br>3.1 ± 0.0<br>**7.5 ± 2.5**<br>2.5 ± 2.3<br>1.9 ± 2.5<br>1.9 ± 1.5<br>3.8 ± 2.3<br>3.8 ± 1.2<br>6.9 ± 1.2<br>7.3 ± 3.9<br>3.9 ± 1.4<br>**8.1 ± 3.2**|
|6h~~ v~~s~~ 8~~z<br>poor<br>medium<br>good|0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>1.0 ± 1.5<br>**1.9 ± 3.8**<br>1.9 ± 1.5<br>1.9 ± 1.5<br>1.9 ± 1.5<br>2.5 ± 1.2<br>1.9 ± 1.5<br>1.2 ± 1.5<br>0.0 ± 0.0<br>2.3 ± 2.6<br>**3.1 ± 2.0**<br>8.8 ± 1.2<br>8.8 ± 3.6<br>7.5 ± 1.5<br>9.4 ± 2.0<br>0.6 ± 1.3<br>5.6 ± 3.6<br>0.0 ± 0.0<br>0.0 ± 0.0<br>**11.2 ± 5.4**|
|corridor<br>poor<br>medium<br>good|0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>**0.6 ± 1.3**<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>0.0 ± 0.0<br>**0.6 ± 1.3**<br>15.0 ± 2.3<br>23.1 ± 1.5<br>14.4 ± 1.5<br>22.5 ± 3.1<br>11.9 ± 2.3<br>23.8 ± 5.1<br>19.8 ± 2.9<br>9.4 ± 6.8<br>**27.3 ± 3.4**<br>30.6 ± 4.1<br>42.5 ± 6.4<br>5.6 ± 1.2<br>42.5 ± 6.4<br>3.1 ± 0.0<br>41.9 ± 6.4<br>39.6 ± 5.3<br>43.1 ± 6.4<br>**48.8 ± 2.5**|



Table 1: Comparison of average winrates for ComaDICE and baselines on SMACv1 tasks.






|Instances|OptDICE AlberDICE ComaDICE<br>BC BCQ CQL ICQ OMAR OMIGA<br>(ours)|
|---|---|
|Protoss<br>5~~ v~~s~~ 5~~<br>10~~ v~~s~~ 1~~0<br>10~~ v~~s~~ 1~~1<br>20~~ v~~s~~ 2~~0<br>20~~ v~~s~~ 2~~3|36.9±8.7<br>16.2±2.3<br>10.0±4.1<br>36.9±9.1<br>21.2±4.1<br>33.1±5.4<br>10.8±1.2<br>12.6±0.9<br>**46.2±6.1**<br>36.2±10.6<br>9.4±5.6<br>26.2±7.6<br>28.1±6.6<br>13.8±7.0<br>40.0±10.7<br>9.5±0.8<br>11.8±0.9<br>**50.6±8.7**<br>19.4±4.6<br>10.0±4.1<br>10.6±5.4<br>12.5±4.4<br>12.5±3.4<br>16.2±6.1<br>10.0±0.5<br>9.8±0.3<br>**20.0±4.2**<br>37.5±4.4<br>6.2±2.0<br>11.9±4.1<br>32.5±8.1<br>23.8±2.5<br>36.2±5.1<br>10.0±2.0<br>10.1±0.6<br>**47.5±7.8**<br>**13.8±1.5**<br>1.2±1.5<br>0.0±0.0<br>12.5±5.6<br>11.2±7.8<br>12.5±8.1<br>8.1±1.4<br>8.8±0.8<br>**13.8±5.8**|
|Terran<br>5~~ v~~s~~ 5~~<br>10~~ v~~s~~ 1~~0<br>10~~ v~~s~~ 1~~1<br>20~~ v~~s~~ 2~~0<br>20~~ v~~s~~ 2~~3|30.0±4.2<br>12.5±6.2<br>9.4±7.9<br>23.1±5.8<br>14.4±4.7<br>28.1±4.4<br>6.4±1.1<br>8.1±1.4<br>**30.6±8.2**<br>29.4±5.8<br>6.9±6.1<br>9.4±5.6<br>16.9±5.8<br>15.0±4.6<br>29.4±3.2<br>6.0±1.6<br>8.2±1.0<br>**32.5±5.8**<br>16.2±3.6<br>3.8±4.6<br>7.5±6.4<br>5.0±4.2<br>9.4±5.6<br>12.5±5.2<br>4.8±1.2<br>6.2±0.9<br>**19.4±5.4**<br>26.2±10.4<br>5.0±3.2<br>10.6±4.2<br>15.6±3.4<br>7.5±7.3<br>21.9±4.4<br>6.3±1.8<br>5.9±1.2<br>**29.4±3.8**<br>4.4±4.2<br>0.0±0.0<br>0.0±0.0<br>7.5±6.1<br>5.0±4.2<br>4.4±2.5<br>4.4±0.7<br>3.9±0.8<br>**9.4±5.2**|
|Zerg<br>5~~ v~~s~~ 5~~<br>10~~ v~~s~~ 1~~0<br>10~~ v~~s~~ 1~~1<br>20~~ v~~s~~ 2~~0<br>20~~ v~~s~~ 2~~3|26.9±10.0<br>14.4±4.2<br>14.4±5.8<br>18.8±7.1<br>13.8±6.1<br>21.9±5.9<br>8.2±1.8<br>9.5±0.8<br>**31.2±7.7**<br>25.0±2.8<br>5.6±4.6<br>5.6±4.6<br>15.6±7.4<br>19.4±2.3<br>23.8±6.4<br>7.8±1.0<br>8.5±0.3<br>**33.8±11.8**<br>13.8±4.7<br>9.4±5.2<br>6.2±4.4<br>10.6±6.7<br>10.6±3.8<br>13.8±6.7<br>7.2±0.7<br>9.1±0.5<br>**19.4±3.6**<br>8.1±1.5<br>2.5±1.2<br>1.2±1.5<br>10.0±7.8<br>**12.5±4.4**<br>10.0±2.3<br>7.3±0.7<br>8.3±0.5<br>9.4±6.2<br>7.5±3.2<br>0.6±1.3<br>1.2±1.5<br>7.5±3.2<br>3.8±2.3<br>4.4±4.2<br>7.1±1.2<br>8.8±0.5<br>**11.2±4.2**|



Table 2: Comparison of win rates for ComaDICE and baselines across SMACv2 tasks.


code was also employed to run these baselines for the SMACv2 environment. All hyperparameters
were kept at their default settings, and each experiment was conducted with _five different random_
_seeds_ to ensure robustness and reproducibility of the results.


6.3 M AIN C OMPARISON


We now present a comprehensive evaluation of our proposed algorithm, ComaDICE, against several
baseline methods in offline MARL. The baselines selected for comparison include both standard and
SOTA approaches, providing a robust benchmark to assess the effectiveness of ComaDICE.


Our evaluation focuses on two primary metrics: returns and winrates. Returns are the average
rewards accumulated by the agents across multiple trials, providing a measure of policy effectiveness.
Winrates, applicable in competitive environments such as SMACv1 and SMACv2, indicate the
success rate of agents against opponents, reflecting the algorithm’s robustness in adversarial settings.


The experimental results, summarized in Tables 1-3, demonstrate that ComaDICE consistently
achieves superior performance compared to baseline methods across a range of scenarios. Notably,
ComaDICE excels in complex tasks, highlighting its ability to effectively manage distributional shifts
in challenging environments.


6.4 A BLATION S TUDY - I MPACT OF THE R EGULARIZATION P ARAMETER A LPHA


We investigate how varying the regularization parameter alpha ( _α_ ) affects the performance of our
ComaDICE algorithm. The parameter _α_ is crucial for balancing the trade-off between maximizing
rewards and penalizing deviations from the offline dataset’s distribution. We conducted experiments
with _α_ values ranging from _{_ 0 _._ 01 _,_ 0 _._ 1 _,_ 1 _,_ 10 _,_ 100 _}_, evaluating performance using average winrates
across all the SMACv2 tasks and average returns across all the MaMujoco tasks. These results,
illustrated in Figure 1, highlight the sensitivity of ComaDICE to different _α_ values. In particular, we


9


Published as a conference paper at ICLR 2025



|Instances|OptDICE AlberDICE ComaDICE<br>BCQ CQL ICQ OMIGA<br>(ours)|
|---|---|
|Hopper<br>expert<br>medium<br>m-replay<br>m-expert|77.9 ± 58.0<br>159.1 ± 313.8<br>754.7 ± 806.3<br>859.6 ± 709.5<br>655.9 ± 120.1<br>844.6 ± 556.5<br>**2827.7 ± 62.9**<br>44.6 ± 20.6<br>401.3 ± 199.9<br>501.8 ± 14.0<br>**1189.3 ± 544.3**<br>204.1 ± 41.9<br>216.9 ± 35.3<br>822.6 ± 66.2<br>26.5 ± 24.0<br>31.4 ± 15.2<br>195.4 ± 103.6<br>774.2 ± 494.3<br>257.8 ± 55.3<br>419.2 ± 243.5<br>**906.3 ± 242.1**<br>54.3 ± 23.7<br>64.8 ± 123.3<br>355.4 ± 373.9<br>709.0 ± 595.7<br>400.9 ± 132.5<br>515.1 ± 303.4<br>**1362.4 ± 522.9**|
|Ant<br>expert<br>medium<br>m-replay<br>m-expert|1317.7 ± 286.3<br>1042.4 ± 2021.6<br>2050.0 ± 11.9<br>2055.5 ± 1.6<br>1717.2 ± 27.0<br>1896.8 ± 33.7<br>**2056.9 ± 5.9**<br>1059.6 ± 91.2<br>533.9 ± 1766.4<br>1412.4 ± 10.9<br>1418.4 ± 5.4<br>1199.0 ± 26.8<br>1304.3 ± 2.6<br>**1425.0 ± 2.9**<br>950.8 ± 48.8<br>234.6 ± 1618.3<br>1016.7 ± 53.5<br>1105.1 ± 88.9<br>869.4 ± 62.6<br>1042.8 ± 80.8<br>**1122.9 ± 61.0**<br>1020.9 ± 242.7<br>800.2 ± 1621.5<br>1590.2 ± 85.6<br>1720.3 ± 110.6<br>1293.2 ± 183.1<br>1780.0 ± 23.6<br>**1813.9 ± 68.4**|
|Half<br>Cheetah<br>expert<br>medium<br>m-replay<br>m-expert|2992.7 ± 629.7<br>1189.5 ± 1034.5<br>2955.9 ± 459.2<br>3383.6 ± 552.7<br>2601.6 ± 461.9<br>3356.4 ± 546.9<br>**4082.9 ± 45.7**<br>2590.5 ± 1110.4<br>1011.3 ± 1016.9<br>2549.3 ± 96.3<br>**3608.1 ± 237.4**<br>305.3 ± 946.8<br>522.4 ± 315.5<br>2664.7 ± 54.2<br>-333.6 ± 152.1<br>1998.7 ± 693.9<br>1922.4 ± 612.9<br>2504.7 ± 83.5<br>-912.9 ± 1363.9<br>440.0 ± 528.0<br>**2855.0 ± 242.2**<br>3543.7 ± 780.9<br>1194.2 ± 1081.0<br>2834.0 ± 420.3<br>2948.5 ± 518.9<br>-2485.8 ± 2338.4<br>2288.2 ± 759.5<br>**3889.7 ± 81.6**|


60%


40%


20%



Table 3: Average returns for ComaDICE and baselines on MaMuJoCo benchmarks.


2k


0



0.01 0.1 1 10 100

HalfCheetah



0.01 0.1 1 10 100

Ant



0.01 0.1 1 10 100


protoss



0.01 0.1 1 10 100


Hopper



0.01 0.1 1 10 100

terran



0.01 0.1 1 10 100

zerg



Figure 1: Impact of regularization parameter _α_ on performance in different environments.


observe that ComaDICE achieves optimal performance when _α_ is around 10, suggesting that the
stationary distribution regularizer plays a essential role in the success of our algorithm.


In our appendix, we provide additional ablation studies to analyze the performance of our algorithm
using different forms of f-divergence functions, as well as comparisons between 1-layer and 2-layer
mixing network structures. The appendix also includes proofs of the theoretical claims made in the
main paper, details of our experimental settings, and other experimental information.


7 C ONCLUSION, F UTURE W ORK AND B ROADER I MPACTS


**Conclusion.** In this paper, we propose ComaDICE, a principled framework for offline MARL. Our
algorithm incorporates a stationary distribution shift regularizer into the standard MARL objective to
address the conventional distribution shift issue in offline RL. To facilitate training within a CTDE
framework, we decompose both the global value and advantage functions using a mixing network.
We demonstrate that, under our mixing architecture, the main objective function is concave in the
value function, which is crucial for ensuring stable and efficient training. The results of this training
are then utilized to derive local policies through a weighted BC approach, ensuring consistency
between global and local policy optimization. Extensive experiments on SOTA benchmark tasks,
including SMACv2, show that ComaDICE outperforms other baseline methods.


**Limitations and Future Work:** There are some limitations that are not addressed within the scope
of this paper. For instance, we focus solely on cooperative learning, leaving open the question of
how the approach would perform in cooperative-competitive settings. Additionally, in our training
objective, the DICE term is designed to reduce the divergence between the learning policy and the
behavior policy. As a result, the performance of the algorithm is heavily dependent on the quality of
the behavior policy. Furthermore, our algorithm, like other baselines, still requires a large amount of
data to achieve desirable learning outcomes. Improving sample efficiency would be another valuable
area for future research.


**Broader Impacts:** Developing an offline MARL algorithm with a stationary distribution shift regularizer can enhance performance in costly real-time tasks like robotics, autonomous driving, and
healthcare. It also enables safer exploration and broader adoption in high-stakes settings. However,
reliance on the behavior policy means flawed or biased data could degrade performance, reinforcing
biases or suboptimal behaviors. Additionally, the algorithm, like any AI systems, risks unintended
misuse in surveillance or military applications, where multi-agent systems could manipulate environments without proper oversight.


10


Published as a conference paper at ICLR 2025


A CKNOWLEDGMENT


This work is supported by the Lee Kong Chian Fellowship awarded to Tien Mai.


E THICAL S TATEMENT


Our work introduces ComaDICE, a framework for offline MARL, aimed at improving training
stability and policy optimization in complex multi-agent environments. While this research has
significant potential for positive applications, particularly in domains such as autonomous systems,
resource management, and multi-agent simulations, it is crucial to address the ethical implications
and risks associated with this technology.


The deployment of reinforcement learning systems in real-world, multi-agent settings raises concerns about unintended behaviors, especially in safety-critical domains. If the policies learned by
ComaDICE are applied without proper testing and validation, they may lead to undesirable or harmful
outcomes, especially in areas such as autonomous driving, healthcare, or robotics. Additionally,
bias in the training data or simulation environments could result in suboptimal policies that unfairly
impact certain agents or populations, potentially leading to ethical concerns regarding fairness and
transparency.


To mitigate these risks, we emphasize the need for extensive testing and validation of policies
generated using ComaDICE, particularly in real-world environments where the consequences of
errors could be severe. It is also essential to ensure that the datasets and simulations used in training
are representative, unbiased, and carefully curated. We encourage practitioners to use human oversight
and collaborate with domain experts to ensure that ComaDICE is applied responsibly, particularly in
high-stakes settings.


R EPRODUCIBILITY S TATEMENT


In order to facilitate reproducibility, we have submitted the source code for ComaDICE, along with
the datasets utilized to produce the experimental results presented in this paper (all these will be made
publicly available if the paper gets accepted). Additionally, in the appendix, we provide details of
our algorithm, including key implementation steps and details needed to replicate the results. The
hyper-parameter settings for all experiments are also included to ensure that others can reproduce the
findings under the same experimental conditions. We invite the research community to explore and
apply the ComaDICE framework in various environments to further validate and expand upon the
results reported in this work.


R EFERENCES


Rishabh Agarwal, Dale Schuurmans, and Mohammad Norouzi. An optimistic perspective on offline
reinforcement learning. In _International conference on machine learning_, pp. 104–114. PMLR,
2020.


Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline
reinforcement learning with diversified q-ensemble. _Advances in neural information processing_
_systems_, 34:7436–7447, 2021.


Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhihong Deng, Animesh Garg, Peng Liu, and Zhaoran
Wang. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. _arXiv_
_preprint arXiv:2202.11566_, 2022.


David Brandfonbrener, Will Whitney, Rajesh Ranganath, and Joan Bruna. Offline rl without off-policy
evaluation. _Advances in neural information processing systems_, 34:4933–4946, 2021.


Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and
[Wojciech Zaremba. Openai gym, 2016. URL http://arxiv.org/abs/1606.01540.](http://arxiv.org/abs/1606.01540)


The Viet Bui, Tien Mai, and Thanh Hong Nguyen. Inverse factorized q-learning for cooperative
multi-agent imitation learning. _Advances in Neural Information Processing Systems_, 38, 2024.


11


Published as a conference paper at ICLR 2025


Xinyue Chen, Zijian Zhou, Zheng Wang, Che Wang, Yanqiu Wu, and Keith Ross. Bail: Bestaction imitation learning for batch deep reinforcement learning. _Advances in Neural Information_
_Processing Systems_, 33:18353–18363, 2020.


Peng Cheng, Xianyuan Zhan, Wenjia Zhang, Youfang Lin, Han Wang, Li Jiang, et al. Look beneath
the surface: Exploiting fundamental symmetry for sample-efficient offline rl. _Advances in Neural_
_Information Processing Systems_, 36, 2024.


Christian Schroeder de Witt, Bei Peng, Pierre-Alexandre Kamienny, Philip Torr, Wendelin Bohmer, ¨
and Shimon Whiteson. Deep multi-agent reinforcement learning for decentralized continuous
cooperative control. _arXiv preprint arXiv:2003.06709_, 19, 2020.


Charles Dugas, Yoshua Bengio, Fran c¸ ois Belisle, Claude Nadeau, and Ren ´ e Garcia. Incorporating ´
functional knowledge in neural networks. _Journal of Machine Learning Research_, 10(6), 2009.


Benjamin Ellis, Skander Moalla, Mikayel Samvelyan, Mingfei Sun, Anuj Mahajan, Jakob N Foerster, and Shimon Whiteson. Smacv2: An improved benchmark for cooperative multi-agent
reinforcement learning. _arXiv preprint arXiv:2212.07489_, 2022.


Wei Fu, Chao Yu, Zelai Xu, Jiaqi Yang, and Yi Wu. Revisiting some common practices in cooperative
multi-agent reinforcement learning. In _Proceedings of the 39th International Conference on_
_Machine Learning_, pp. 6863–6877. PMLR, 2022.


Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning.
_Advances in neural information processing systems_, 34:20132–20145, 2021.


Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actorcritic methods. In _International conference on machine learning_, pp. 1587–1596. PMLR, 2018.


Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without
exploration. In _International conference on machine learning_, pp. 2052–2062. PMLR, 2019.


Ammar Haydari and Yasin Yılmaz. Deep reinforcement learning for intelligent transportation systems:
A survey. _IEEE Transactions on Intelligent Transportation Systems_, 23(1):11–32, 2020.


Jiechuan Jiang and Zongqing Lu. Offline decentralized multi-agent reinforcement learning. In _ECAI_,
pp. 1148–1155, 2023.


Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre
Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement
learning for vision-based robotic manipulation. In _Conference on robot learning_, pp. 651–673.
PMLR, 2018.


Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel: Modelbased offline reinforcement learning. _Advances in neural information processing systems_, 33:
21810–21823, 2020.


Ilya Kostrikov, Rob Fergus, Jonathan Tompson, and Ofir Nachum. Offline reinforcement learning
with fisher divergence critic regularization. In _International Conference on Machine Learning_, pp.
5774–5783. PMLR, 2021.


Landon Kraemer and Bikramjit Banerjee. Multi-agent reinforcement learning as a rehearsal for
decentralized planning. _Neurocomputing_, 190:82–94, 2016.


Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy
q-learning via bootstrapping error reduction. _Advances in neural information processing systems_,
32, 2019.


Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline
reinforcement learning. _Advances in Neural Information Processing Systems_, 33:1179–1191, 2020.


Jongmin Lee, Wonseok Jeon, Byungjun Lee, Joelle Pineau, and Kee-Eung Kim. Optidice: Offline
policy optimization via stationary distribution correction estimation. In _International Conference_
_on Machine Learning_, pp. 6120–6130. PMLR, 2021.


12


Published as a conference paper at ICLR 2025


Jongmin Lee, Cosmin Paduraru, Daniel J Mankowitz, Nicolas Heess, Doina Precup, Kee-Eung Kim,
and Arthur Guez. Coptidice: Offline constrained reinforcement learning via stationary distribution
correction estimation. _arXiv preprint arXiv:2204.08957_, 2022.


Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep
visuomotor policies. _Journal of Machine Learning Research_, 17(39):1–40, 2016.


Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial,
review, and perspectives on open problems. _arXiv preprint arXiv:2005.01643_, 2020.


Jianxiong Li, Xiao Hu, Haoran Xu, Jingjing Liu, Xianyuan Zhan, and Ya-Qin Zhang. Proto: Iterative
policy regularized offline-to-online reinforcement learning. _arXiv preprint arXiv:2305.15669_,
2023.


Jinning Li, Chen Tang, Masayoshi Tomizuka, and Wei Zhan. Dealing with the unknown: Pessimistic
offline reinforcement learning. In _Conference on Robot Learning_, pp. 1455–1464. PMLR, 2022.


Liyuan Mao, Haoran Xu, Weinan Zhang, and Xianyuan Zhan. Odice: Revealing the mystery of
distribution correction estimation via orthogonal-gradient update. _arXiv preprint arXiv:2402.00348_,
2024.


Daiki E Matsunaga, Jongmin Lee, Jaeseok Yoon, Stefanos Leonardos, Pieter Abbeel, and Kee-Eung
Kim. Alberdice: addressing out-of-distribution joint actions in offline multi-agent rl via alternating
stationary distribution correction estimation. _Advances in Neural Information Processing Systems_,
36:72648–72678, 2023.


Tatsuya Matsushima, Hiroki Furuta, Yutaka Matsuo, Ofir Nachum, and Shixiang Gu. Deploymentefficient reinforcement learning via model-based offline optimization. _arXiv preprint_
_arXiv:2006.03647_, 2020.


Linghui Meng, Muning Wen, Chenyang Le, Xiyun Li, Dengpeng Xing, Weinan Zhang, Ying
Wen, Haifeng Zhang, Jun Wang, Yaodong Yang, et al. Offline pre-trained multi-agent decision
transformer. _Machine Intelligence Research_, 20(2):233–248, 2023.


Ofir Nachum and Bo Dai. Reinforcement learning via fenchel-rockafellar duality. _arXiv preprint_
_arXiv:2001.01866_, 2020.


Ashvin Nair, Abhishek Gupta, Murtaza Dalal, and Sergey Levine. Awac: Accelerating online
reinforcement learning with offline datasets. _arXiv preprint arXiv:2006.09359_, 2020.


Haoyi Niu, Yiwen Qiu, Ming Li, Guyue Zhou, Jianming Hu, Xianyuan Zhan, et al. When to trust
your simulator: Dynamics-aware hybrid offline-and-online reinforcement learning. _Advances in_
_Neural Information Processing Systems_, 35:36599–36612, 2022.


Frans A Oliehoek, Matthijs TJ Spaan, and Nikos Vlassis. Optimal and approximate q-value functions
for decentralized pomdps. _Journal of Artificial Intelligence Research_, 32:289–353, 2008.


Ling Pan, Longbo Huang, Tengyu Ma, and Huazhe Xu. Plan better amid conservatism: Offline
multi-agent reinforcement learning with actor rectification. In _International conference on machine_
_learning_, pp. 17221–17237. PMLR, 2022.


Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression:
Simple and scalable off-policy reinforcement learning. _arXiv preprint arXiv:1910.00177_, 2019.


Rafael Figueiredo Prudencio, Marcos ROA Maximo, and Esther Luna Colombini. A survey on offline
reinforcement learning: Taxonomy, review, and open problems. _IEEE Transactions on Neural_
_Networks and Learning Systems_, 2023.


Tabish Rashid, Mikayel Samvelyan, Christian Schroeder De Witt, Gregory Farquhar, Jakob Foerster,
and Shimon Whiteson. Monotonic value function factorisation for deep multi-agent reinforcement
learning. _The Journal of Machine Learning Research_, 21(1):7234–7284, 2020.


Mikayel Samvelyan, Tabish Rashid, Christian Schroeder De Witt, Gregory Farquhar, Nantas Nardelli,
Tim GJ Rudner, Chia-Man Hung, Philip HS Torr, Jakob Foerster, and Shimon Whiteson. The
starcraft multi-agent challenge. _arXiv preprint arXiv:1902.04043_, 2019.


13


Published as a conference paper at ICLR 2025


Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, and Xiangyang Ji. Counterfactual conservative q learning for offline multi-agent reinforcement learning. _Advances in Neural Information_
_Processing Systems_, 36, 2024.


Harshit Sikchi, Amy Zhang, and Scott Niekum. Imitation from arbitrary experience: A dual
unification of reinforcement and imitation learning methods. In _Workshop on Reincarnating_
_Reinforcement Learning at ICLR 2023_, 2023.


David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez,
Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without
human knowledge. _nature_, 550(7676):354–359, 2017.


Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, and Yung Yi. Qtran: Learning to
factorize with transformation for cooperative multi-agent reinforcement learning. In _International_
_conference on machine learning_, pp. 5887–5896. PMLR, 2019.


Wei-Cheng Tseng, Tsun-Hsuan Johnson Wang, Yen-Chen Lin, and Phillip Isola. Offline multi-agent
reinforcement learning with knowledge distillation. _Advances in Neural Information Processing_
_Systems_, 35:226–237, 2022.


Jianhao Wang, Zhizhou Ren, Terry Liu, Yang Yu, and Chongjie Zhang. Qplex: Duplex dueling
multi-agent q-learning. _arXiv preprint arXiv:2008.01062_, 2020.


Jun Wang, Yaodong Yang, and Zongqing Wang. Trust region policy optimization in multi-agent
reinforcement learning. _arXiv preprint arXiv:2109.11251_, 2022a.


Xiangsen Wang, Haoran Xu, Yinan Zheng, and Xianyuan Zhan. Offline multi-agent reinforcement
learning with implicit global-to-local value regularization. _Advances in Neural Information_
_Processing Systems_, 36, 2022b.


Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning.
_arXiv preprint arXiv:1911.11361_, 2019.


Haoran Xu, Xianyuan Zhan, Jianxiong Li, and Honglei Yin. Offline reinforcement learning with soft
behavior regularization. _arXiv preprint arXiv:2110.07395_, 2021.


Haoran Xu, Li Jiang, Li Jianxiong, and Xianyuan Zhan. A policy-guided imitation approach for
offline reinforcement learning. _Advances in Neural Information Processing Systems_, 35:4085–4098,
2022a.


Haoran Xu, Xianyuan Zhan, Honglei Yin, and Huiling Qin. Discriminator-weighted offline imitation
learning from suboptimal demonstrations. In _Proceedings of the 39th International Conference on_
_Machine Learning_, pp. 24725–24742, 2022b.


Haoran Xu, Xianyuan Zhan, and Xiangyu Zhu. Constraints penalized q-learning for safe offline reinforcement learning. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 36,
pp. 8753–8760, 2022c.


Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Wai Kin Chan, and
Xianyuan Zhan. Offline rl with no ood actions: In-sample learning via implicit value regularization.
_arXiv preprint arXiv:2303.15810_, 2023.


Yiqin Yang, Xiaoteng Ma, Chenghao Li, Zewu Zheng, Qiyuan Zhang, Gao Huang, Jun Yang, and
Qianchuan Zhao. Believe what you see: Implicit constraint approach for offline multi-agent
reinforcement learning. _Advances in Neural Information Processing Systems_, 34:10299–10312,
2021.


Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The
surprising effectiveness of ppo in cooperative multi-agent games. _Advances in Neural Information_
_Processing Systems_, 35:24611–24624, 2022.


Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y Zou, Sergey Levine, Chelsea Finn,
and Tengyu Ma. Mopo: Model-based offline policy optimization. _Advances in Neural Information_
_Processing Systems_, 33:14129–14142, 2020.


14


Published as a conference paper at ICLR 2025


Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn.
Combo: Conservative offline model-based policy optimization. _Advances in neural information_
_processing systems_, 34:28954–28967, 2021.


Hongchang Zhang, Jianzhun Shao, Yuhang Jiang, Shuncheng He, Guanwen Zhang, and Xiangyang
Ji. State deviation correction for offline reinforcement learning. In _Proceedings of the AAAI_
_conference on artificial intelligence_, volume 36, pp. 9022–9030, 2022.


Qin Zhang, Linrui Zhang, Haoran Xu, Li Shen, Bowen Wang, Yongzhe Chang, Xueqian Wang,
Bo Yuan, and Dacheng Tao. Saformer: A conditional sequence modeling approach to offline safe
reinforcement learning. _arXiv preprint arXiv:2301.12203_, 2023.


Tianhao Zhang, Yueheng Li, Chen Wang, Guangming Xie, and Zongqing Lu. Fop: Factorizing
optimal joint policy of maximum-entropy multi-agent reinforcement learning. In _International_
_conference on machine learning_, pp. 12491–12500. PMLR, 2021.


Yinan Zheng, Jianxiong Li, Dongjie Yu, Yujie Yang, Shengbo Eben Li, Xianyuan Zhan, and Jingjing
Liu. Safe offline reinforcement learning with feasibility-guided diffusion model. _arXiv preprint_
_arXiv:2401.10700_, 2024.


15


