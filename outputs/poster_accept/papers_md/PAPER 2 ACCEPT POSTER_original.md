Published as a conference paper at ICLR 2025

## - S ELF SUPERVISED CONTRASTIVE LEARNING ### PERFORMS NON LINEAR SYSTEM IDENTIFICATION


**Rodrigo Gonz´alez Laiz, Tobias Schmidt** _[∗]_ _[∗]_ **& Steffen Schneider** _[†]_


Institute of Computational Biology, Computational Health Center, Helmholtz Munich and
Munich Center for Machine Learning (MCML)


A BSTRACT


Self-supervised learning (SSL) approaches have brought tremendous success across
many tasks and domains. It has been argued that these successes can be attributed
to a link between SSL and identifiable representation learning: Temporal structure
and auxiliary variables ensure that latent representations are related to the true underlying generative factors of the data. Here, we deepen this connection and show
that SSL can perform system identification in latent space. We propose dynamics
contrastive learning, a framework to uncover linear, switching linear and non-linear
dynamics under a non-linear observation model, give theoretical guarantees and val[idate them empirically. Code: github.com/dynamical-inference/dcl](https://github.com/dynamical-inference/dyncl)


1 I NTRODUCTION


The identification and modeling of dynamics from observational data is a long-standing problem in
machine learning, engineering and science. A discrete-time dynamical system with latent variables _**x**_,
observable variables _**y**_, control signal _**u**_, its control matrix _**B**_, and noise _**ε**_ _,_ _**ν**_ can take the form


_**x**_ _t_ +1 = _**f**_ ( _**x**_ _t_ ) + _**Bu**_ _t_ + _**ε**_ _t_
(1)
_**y**_ _t_ = _**g**_ ( _**x**_ _t_ ) + _**ν**_ _t_ _._


and we aim to infer the functions _**f**_ and _**g**_ from a time-series of observations and, when available,
control signals. Numerous algorithms have been developed to tackle special cases of this problem
formulation, ranging from classical system identification methods (McGee & Schmidt, 1985; Chen &
Billings, 1989) to recent generative models (Duncker et al., 2019; Linderman et al., 2017; Halv ¨ a et al., ¨
2021). Yet, it remains an open challenge to improve the generality, interpretability and efficiency of
these inference techniques, especially when _**f**_ and _**g**_ are non-linear functions.


Contrastive learning (CL) and next-token prediction tasks have become important backbones of
modern machine learning systems for learning from sequential data, proving highly effective for
building meaningful latent representations (Baevski et al., 2022; Bommasani et al., 2021; Brown,
2020; Oord et al., 2018; LeCun, 2022; Sermanet et al., 2018; Radford et al., 2019). An emerging
view is a connection between these algorithms and learning of _world models_ (Ha & Schmidhuber,
2018; Assran et al., 2023; Garrido et al., 2024). However, the theoretical understanding of non-linear
system identification by these sequence-learning algorithms remains limited.


In this work, we revisit and extend contrastive learning in the context of system identification. We
uncover several surprising facts about its out-of-the-box effectiveness in identifying dynamics and
unveil common design choices in SSL systems used in practice. Our theoretical study extends
identifiability results (Hyvarinen & Morioka, 2016; 2017; Hyvarinen et al., 2019; Zimmermann et al.,
2021; Roeder et al., 2021) for CL towards dynamical systems. While our theory makes several
predictions about capabilities of standard CL, it also highlights shortcomings. To overcome these and
enable interpretable dynamics inference across a range of data generating processes, we propose a
general framework for linear and non-linear system identification with CL (Figure 1).


**Background.** An influential motivation of our work is Contrastive Predictive Coding (CPC; Oord
et al., 2018). CPC can be recovered as a special case of our framework when using an RNN dynamics


_∗_ Equal contribution.

_†_ Correspondence: steffen.schneider@helmholtz-munich.de


1


Published as a conference paper at ICLR 2025


model. Related works have emerged across different modalities: wav2vec (Schneider et al., 2019),
TCN (Sermanet et al., 2018) and CPCv2 (Henaff, 2020). In the field of system identification, notable
approaches include the Extended Kalman Filter (EKF) (McGee & Schmidt, 1985) and NARMAX
(Chen & Billings, 1989). Additionally, several works have also explored generative models for general
dynamics (Duncker et al., 2019) and switching dynamics, e.g. rSLDS (Linderman et al., 2017). In the
Nonlinear ICA literature, identifiable algorithms for time-series data, such as Time Contrastive Learning (TCL; Hyvarinen & Morioka, 2016) for non-stationary processes and Permutation Contrastive
Learning (PCL; Hyvarinen & Morioka, 2017) for stationary data have been proposed, with recent
advances like SNICA (Halv ¨ a et al., 2021) for more generally structured data-generating processes. ¨



In contrast to previous work, we focus on bridging timeseries representation learning through contrastive learning
with the identification of dynamical systems, both theoretically and empirically. Moreover, by not relying on
an explicit data-generating model, our framework offers
greater flexibility. We extend and discuss the connections
to related work in more detail in Appendix C.


**Contributions.** We extend the existing theory on contrastive learning for time series learning and make adaptations to common inference frameworks. We introduce
our CL variant (Fig. 1) in section 2, and give an identifiability result for both the latent space and the dynamics
model in section 3. These theoretical results are later empirically validated. We then propose a practical way to
parameterize switching linear dynamics in section 4 and
demonstrate that this formulation corroborates our theory
for both switching linear system dynamics and non-linear
dynamics in sections 5-6.



Figure 1: DCL framework: The encoder _**h**_ is
shared across the reference _**y**_ _t_, positive _**y**_ _t_ +1,
and negative samples _**f**_ ˆ forward predicts the reference. A (possi- _**y**_ _i_ _[−]_ [. A dynamics model]
bly latent) variable _**z**_ can parameterize the
dynamics (cf. § 4) or external control (cf. § I).
The model fits the InfoNCE loss ( _L_ ).



















2 C ONTRASTIVE LEARNING FOR TIME - SERIES


In contrastive learning, we aim to model similarities between pairs of data points (Figure 1). Our full
model _ψ_ is specified by the log-likelihood


log _p_ _ψ_ ( _**y**_ _|_ _**y**_ [+] _, N_ ) = _ψ_ ( _**y**_ _,_ _**y**_ [+] ) _−_ log � exp( _ψ_ ( _**y**_ _,_ _**y**_ _[−]_ )) _._ (2)

_**y**_ _[−]_ _∈N_ _∪{_ _**y**_ [+] _}_


where _**y**_ is often called the reference or anchor sample, _**y**_ [+] is a positive sample, _**y**_ _[−]_ _∈_ _N_ are negative
examples, and _N_ is the set of negative samples. The model _ψ_ itself is parameterized as a composition
of an encoder, a dynamics model, and a similarity function and will be defined further below. We fit
the model by minimizing the negative log-likelihood on the time series,

min _ψ_ _[L]_ [[] _[ψ]_ [] = min] _ψ_ E _t,t_ 1 _,...,t_ _M_ _∼U_ (1 _,T_ ) [ _−_ log _p_ _ψ_ ( _**y**_ _t_ +1 _|_ _**y**_ _t_ _, {_ _**y**_ _t_ _m_ _}_ _[M]_ _m_ =1 [)]] (3)


where positive examples are just adjacent points in the time-series, and _M_ negative examples are
sampled uniformly across the dataset. _U_ (1 _, T_ ) denotes a uniform distribution across the discrete time
steps.


To attain favourable properties for identifying the latent dynamics, we carefully design the hypothesis
class for _ψ_ . The motivation for this particular design will become clear later. To define the full model, a
composition of several functions is necessary. Recall from Eq. 1 that the dynamics model is given as _**f**_
and the mixing function is _**g**_ . Correspondingly, our model is composed of the encoder _**h**_ : R _[D]_ _�→_ R _[d]_

(de-mixing), the dynamics model _**f**_ [ˆ] : R _[d]_ _�→_ R _[d]_, the similarity function _ϕ_ : R _[d]_ _×_ R _[d]_ _�→_ R and a
correction term _α_ : R _[d]_ _�→_ R. We define their composition as [1]


_ψ_ ( _**y**_ _,_ _**y**_ _[′]_ ) := _ϕ_ ( _**f**_ [ˆ] ( _**h**_ ( _**y**_ )) _,_ _**h**_ ( _**y**_ _[′]_ )) _−_ _α_ ( _**y**_ _[′]_ ) _,_ (4)


and call the resulting algorithm _dynamics contrastive learning_ (DCL). Intuitively, we obtain two
observed samples ( _**y**_ _,_ _**y**_ _[′]_ ) which are first mapped to the latent space, ( _**h**_ ( _**y**_ ) _,_ _**h**_ ( _**y**_ _[′]_ )) . Then, the


1 Note that we can equivalently write _ϕ_ (˜ _**h**_ ( _**x**_ )) _,_ ˜ _**h**_ _′_ ( _**x**_ _′_ )) using two asymmetric encoder functions, see additional results in Appendix D.


2


Published as a conference paper at ICLR 2025


a b


Figure 2: Graphical intuition behind Theorem 1. (a), the ground truth latent space is mapped to observables
through the injective mixing function _**g**_ . Our model maps back into the latent space. The composition of
mixing and de-mixing by the model is an affine transform. (b), dynamics in the ground-truth space are mapped
to the latent space. By observing variations introduced by the system noise _**ε**_, our model is able to infer the
ground-truth dynamics up to an affine transform.


dynamics model is applied to _**h**_ ( _**y**_ ), and the resulting points are compared through the similarity
function _ϕ_ . The similarity function _ϕ_ will be informed by the form of (possibly induced) system
noise _**ε**_ _t_ . In the simplest form, the noise can be chosen as isotropic Gaussian noise, which results in a
negative squared Euclidean norm for _ϕ_ .


Note, the additional term _α_ ( _**y**_ _[′]_ ) is a correction applied to account for non-uniform marginal distributions. It can be parameterized as a kernel density estimate (KDE) with log ˆ _q_ ( _**h**_ ( _**y**_ _[′]_ )) _≈_ log _q_ ( _**x**_ _[′]_ )
around the datapoints. In very special cases, the KDE makes a difference in empirical performance
(App. B, Fig. 9) and is required for our theory. Yet, we found that on the time-series datasets
considered, it was possible to drop this term without loss in performance (i.e., _α_ ( _**y**_ _[′]_ ) = 0).


3 S TRUCTURAL IDENTIFIABILITY OF NON - LINEAR LATENT DYNAMICS


We now study the aforementioned model theoretically. The key components of our theory along with
our notion of linear identifiability (Roeder et al., 2021; Khemakhem et al., 2020) are visualized in
Figure 2. We are interested in two properties. First, linear identifiability of the latent space: The
composition of mixing function _**g**_ and model encoder _**h**_ should recover the ground-truth latents up
to a linear transform. Second, identifiability of the (non-linear) dynamics model: We would like to
relate the estimated dynamics _**f**_ [ˆ] to the underlying ground-truth dynamics _**f**_ . This property is also
called _structural identifiability_ (Bellman & Astr [˚] om, 1970). Our model operates on a subclass of Eq. 1 ¨
with the following properties:


**Data-generating process.** We consider a discrete-time dynamical system defined as


_**x**_ _t_ +1 = _**f**_ ( _**x**_ _t_ ) + _**ε**_ _t_ _,_ _**y**_ _t_ = _**g**_ ( _**x**_ _t_ ) _,_ (5)


where _**x**_ _t_ _∈_ R _[d]_ are latent variables, _**f**_ : R _[d]_ _�→_ R _[d]_ is a bijective dynamics model, _**ε**_ _t_ _∈_ R _[d]_ the system
noise, and _**g**_ : R _[d]_ _�→_ R _[D]_ is a non-linear injective mapping from latents to observables _**y**_ _t_ _∈_ R _[D]_,
_d ≤_ _D_ . We sample a total number of _T_ time steps.


We proceed by stating our main result:

**Theorem 1** (Contrastive estimation of non-linear dynamics) **.** _Assume that_


    - _(A1) A time-series dataset_ _{_ _**y**_ _t_ _}_ _[T]_ _t_ =1 _[is generated according to the ground-truth dynamical]_
_system in Eq. 5 with a bijective dynamics model_ _**f**_ _and an injective mixing function_ _**g**_ _._

    - _(A2) The system noise follows an iid normal distribution, p_ ( _**ε**_ _t_ ) = _N_ ( _**ε**_ _t_ _|_ 0 _,_ **Σ** _**ε**_ ) _._

    - _(A3) The model_ _ψ_ _is composed of an encoder_ _**h**_ _, a dynamics model_ _**f**_ [ˆ] _, a correction term_ _α_ _,_
_and the similarity metric ϕ_ ( _**u**_ _,_ _**v**_ ) = _−∥_ _**u**_ _−_ _**v**_ _∥_ [2] _and attains the global minimizer of Eq. 3._


_Then, in the limit of T →∞_ _for any point_ _**x**_ _in the support of the data marginal distribution:_


_(a)_ _The composition of mixing and de-mixing_ _**h**_ ( _**g**_ ( _**x**_ )) = _**Lx**_ + _**b**_ _is a bijective affine transform,_
_and_ _**L**_ = _**Q**_ **Σ** _[−]_ _ϵ_ [1] _[/]_ [2] _with unknown orthogonal transform_ _**Q**_ _∈_ R _[d][×][d]_ _and offset_ _**b**_ _∈_ R _[d]_ _._
_(b)_ _The estimated dynamics_ _**f**_ ˆ( _**x**_ ) = _**Lf**_ ( _**L**_ _[−]_ [1] ( _**x**_ _−_ _**bf**_ )) + [ˆ] _are bijective and identify the true dynamics_ _**b**_ _._ _**f**_ _up to the relation_


_Proof._ See Appendix A for the full proof, and see Fig. 2 for a graphical intuition of both results.


3


Published as a conference paper at ICLR 2025


With this main result in place, we can make statements for several systems of interest; specifically
linear dynamics in latent space:

**Corollary 1.** _Contrastive learning without dynamics model,_ _**f**_ [ˆ] ( _**x**_ )= _**x**_ _, cannot identify latent dynamics._


In this case, even for a linear ground truth dynamics model, _**f**_ ( _**x**_ ) = _**Ax**_, we would require that after
model fitting, _**f**_ [ˆ] ( _**x**_ ) = _**x**_ = _**LAL**_ _[−]_ [1] _**x**_ + _**b**_, which is impossible (Theorem 1b; also see App. Eq. 22).
We can fix this case by either decoupling the two encoders (Appendix D), or taking a more structured
approach and parameterizing a dynamics model with a dynamics matrix:

**Corollary 2.** _**Ax**_ ˆ _, we identify the latents up to_ _For a ground-truth linear dynamical system_ _**h**_ ( _**g**_ ( _**x**_ )) = _**Lx**_ + _**b**_ _and dynamics with_ _**f**_ ( _**x**_ ) = _**Ax**_ _and dynamics model_ _**A**_ [ˆ] = _**LAL**_ _[−]_ [1] _._ _**f**_ [ˆ] ( _**x**_ ) =


This means that simultaneously fitting the system dynamics and encoding model allows us to recover
the system matrix up to an indeterminacy.


**Note on Assumptions.** The required assumptions are rather practical: (A1) allows for a very broad
class of dynamical systems as long as bijectivity of the _dynamics model_ holds, which is the case of
many systems used in the natural sciences. We consider dynamical systems with control signal _**u**_ _t_ in
Appendix I. While (A2) is a very common one in dynamical systems modeling, it can be seen more
strict: We either need knowledge about the form of system noise, or inject such noise. We should
note that analogous to the discussion of Zimmermann et al. (2021), it is most certainly possible to
extend our results towards other classes of noise distributions by matching the log-density of _**ε**_ with
_ϕ_ . Given the common use of Normally distributed noise, however, we limited the scope of the current
theory to the Normal distribution, but show vMF noise in Appendix D. (A3) mainly concerns the
model setup. An apparent limitation of Def. 5 is the injectivity assumption imposed on the mixing
function _**g**_ . In practice, a _partially observable_ setting often applies, where _**g**_ ( _**x**_ ) = _**Cx**_ maps latents
into lower dimensional observations or has a lower rank than there are latent dimensions. For these
systems, we can ensure injectivity through a time-lag embedding. See Appendix H for empirical
validation.


4 _∇_ -SLDS: T OWARDS NON - LINEAR DYNAMICS ESTIMATION



**Piecewise linear approximation of dynamics.** Our theoretical
results suggest that contrastive learning allows the fitting of nonlinear bijective dynamics. This is a compelling result, but in practice
it requires the use of a powerful, yet easy to parameterize dynamics
model. One option is to use an RNN (Elman, 1990; Oord et al.,
2018) or a Transformer (Vaswani, 2017) model to perform this link
across timescales. An alternative option is to linearize the system,
which we propose in the following.


We propose a new forward model for differentiable switching linear
dynamics ( _∇_ -SLDS) in latent space. The estimation is outlined in
Figure 3. This model allows fast estimation of switching dynamics
and can be easily integrated into the DCL algorithm. The dynamics
model has a trainable bank **W** = [ _**W**_ 1 _, . . .,_ _**W**_ _K_ ] of possible dynamics matrices. _K_ is a hyperparameter. The dynamics depend on a
latent variable _k_ _t_ and are defined as


_**f**_ ˆ( _**x**_ _t_ ; **W** _, k_ _t_ ) = _**W**_ _k_ _t_ _**x**_ _t_ _,_ _k_ _t_ = argmin _k_ _∥_ _**W**_ _k_ _**x**_ _t_ _−_ _**x**_ _t_ +1 _∥_ [2] _._ (6)



Figure 3: The core components of
the _∇_ -SLDS model is parameterfree, differentiable parameterization of the switching process.



Intuitively, the predictive performance of every available linear dynamical system is used to select the
right dynamics with index _k_ _t_ from the bank **W** . During training, we approximate the argmin using
the Gumbel-Softmax trick (Jang et al., 2016) without hard sampling:



p( _λ_ _k_ _/τ_ ) 1

_j_ [exp(] _[λ]_ _[j]_ _[/τ]_ [)] _[, λ]_ _[k]_ [ =] _∥_ _**W**_ _k_ _**x**_ _t_ _−_ _**x**_ _t_ +1 _∥_ [2] [+] _[ g]_ _[k]_ _[.]_ (7)



_**f**_ ˆ( _**x**_ _t_ ; **W** _,_ _**z**_ _t_ ) = (



_K_
�



exp( _λ_ _k_ _/τ_ )

� _z_ _t,k_ _**W**_ _k_ ) _**x**_ _t_ _, z_ _t,k_ =

_k_ =1 ~~�~~ _j_ [exp(] _[λ]_ _[j]_



Note that the dynamics model _**f**_ [ˆ] ( _**x**_ _t_ ; **W** _,_ _**z**_ _t_ ) depends on an additional latent variable _**z**_ _t_ =

[ _z_ _t,_ 1 _, . . ., z_ _t,K_ ] _[⊤]_ which contains probabilities to parametrize the dynamics. During inference, we


4


Published as a conference paper at ICLR 2025


can obtain the index _k_ _t_ = arg max _k_ _z_ _t,k_ . The variables _g_ _k_ are samples from the Gumbel distribution (Jang et al., 2016) and we use a temperature _τ_ to control the smoothness of the resulting
probabilities. During pilot experiments, we found that the reciprocal parameterization of the logits
outperforms other choices for computing an argmin, like flipping the sign.


**From linear switching to non-linear dynamics.** Non-linear system dynamics of the general form
in Eq. 5 can be approximated using our switching model. We can approximate a continuous-time non-linear dynamical system with latent dynamics ˙ _**x**_ = _**f**_ ( _**x**_ ) around reference points _{_ _**x**_ ˜ _k_ _}_ _[K]_ _k_ =1 [using]
a first-order Taylor expansion, _**f**_ ( _**x**_ ) _≈_ _**f**_ [˜] ( _**x**_ ) = _**f**_ (˜ _**x**_ _k_ ) + _**J**_ _**f**_ (˜ _**x**_ _k_ )( _**x**_ _−_ _**x**_ ˜ _k_ ), where we denote the
Jacobian matrix of _**x**_ ˜ _k_ . We obtain system matrices _**f**_ with _**J**_ _**f**_ . We evaluate the equation at each point _**A**_ _k_ = _**J**_ _**f**_ (˜ _**x**_ _k_ ) and bias term _**b**_ _k_ = _**f**_ _t_ (˜ using the best reference point _**x**_ _k_ ) _−_ _**J**_ _**f**_ (˜ _**x**_ _k_ )˜ _**x**_ _k_ which can
be modeled with the _∇_ -SLDS model _**f**_ [ˆ] ( _**x**_ _t_ ; _k_ _t_ ):


_**x**_ _t_ +1 = ( _**A**_ _k_ _t_ _**x**_ _t_ + _**b**_ _k_ _t_ ) + _**ε**_ _t_ =: _**f**_ [ˆ] ( _**x**_ _t_ ; _k_ _t_ ) + _**ε**_ _t_ _._ (8)


While a theoretical guarantee for this general case is beyond the scope of this work, we give an
empirical evaluation on Lorenz attractor dynamics below. Note, as the number of “basis points”
of _∇_ -SLDS approaches the number of time steps, we could trivially approach perfect estimation
capability of the latents as we store the exact value of _**f**_ at every point. However, this comes at the
expense of having less points to estimate each individual dynamics matrix. Empirically, we used
100–200 matrices for datasets of 1M samples.


5 E XPERIMENTS


To verify our theory, we implement a benchmark dataset for studying the effects of various model
choices. We generate time-series with 1M samples, either as a single sequence or across multiple
trials. Our experiments rigorously evaluate different variants of contrastive learning algorithms.


**Data generation.** Data is generated by simulating latent variables _**x**_ that evolve according to a
dynamical system (Eq. 5). These latent variables are then passed through a nonlinear mixing function
_**g**_ to produce the observable data _**y**_ . The mixing function _**g**_ consists of a nonlinear injective component
which is parameterized by a randomly initialized 4-layer MLP (Hyvarinen & Morioka, 2016), and a
linear map to a 50-dimensional space. The final mixing function is defined as their composition. We
ensure the injectivity of the resulting function by monitoring the condition number of each matrix
layer, following previous work (Hyvarinen & Morioka, 2016; Zimmermann et al., 2021).


**LDS.** We simulate 1M datapoints in 3D space following _**f**_ ( _**x**_ _t_ ) = _**Ax**_ _t_ with system noise standard
deviation _σ_ _ϵ_ = 0 _._ 01 and choose _**A**_ to be an orthogonal matrix to ensure stable dynamics with all
eigenvalues equal to 1. We do so by taking the product of multiple rotation matrices, one for each
possible plane to rotate around with rotation angles being randomly chosen to be -5° or 5°.


**SLDS.** We simulate switching linear dynamical systems with _**f**_ ( _**x**_ _t_ ; _k_ _t_ ) = _**A**_ _k_ _t_ _**x**_ _t_ and system
noise standard deviation _σ_ _ϵ_ = 0 _._ 0001 . We choose _**A**_ _k_ to be an orthogonal matrix ensuring that
all eigenvalues are 1, which guarantees system stability. Specifically, we set _**A**_ _k_ to be a rotation
matrix with varying rotation angles (5°, 10°, 20°). The latent dimensionality is 6. The number of
samples is 1M. We use 1000 trials, and each trial consists of 1000 samples. We use _k_ = 0 _,_ 1 _, . . ., K_
distinct modes following a mode sequence _i_ _t_ . The mode sequence _i_ _t_ follows a Markov chain with a
symmetric transition matrix and uniform prior: _i_ 0 _∼_ Cat( _π_ ), where _π_ _j_ = _K_ [1] [for all] _[ j]_ [. At each time]

step, _i_ _t_ +1 _∼_ Cat(Π _i_ _t_ ), where Π is a transition matrix with uniform off-diagonal probabilities set to
10 _[−]_ [4] . Example data is visualized in Figure 4 and Appendix E.


**Non-linear dynamics.** We simulate 1M points of a Lorenz system, with equations


_**f**_ ( _**x**_ _t_ ) = _**x**_ _t_ + _dt_ [ _σ_ ( _x_ 2 _,t_ _−_ _x_ 1 _,t_ ) _, x_ 1 _,t_ (( _ρ −_ _x_ 3 _,t_ ) _−_ _x_ 2 _,t_ ) _,_ ( _x_ 1 _,t_ _x_ 2 _,t_ _−_ _βx_ 3 _,t_ )] _[⊤]_ (9)


with varying _dt_, parameters _σ_ = 10 _, β_ = [8] 3 _[, ρ]_ [ = 28] [ and system noise standard deviation] _[ σ]_ _[ϵ]_ [ = 0] _[.]_ [001] [.]

The observable data, _**y**_ . We then apply our non-linear mixing function as for other datasets.


**Model estimation.** For the feature encoder _**h**_, baseline and our model use an MLP with three
layers followed by GELU activations (Hendrycks & Gimpel, 2016). Model capacity scales with
the embedding dimensionality _d_ . The last hidden layer has 10 _d_ units and all previous layers have
30 _d_ units. For the SLDS and LDS datasets, we train on batches with 2048 samples each (reference


5


Published as a conference paper at ICLR 2025


Table 1: Overview about identifiability of latent dynamics for different modeling choices: We show different
_data_ generating processes characterized by the form of the ground truth dynamics _**f**_, the distribution _p_ ( _**ε**_ )
and different _model_ choices for the estimated dynamics _**f**_ [ˆ] . We compare identity dynamics, linear dynamics
(LDS), switching linear dynamics (SLDS), and Lorenz attractor dynamics (Lorenz), and optionally initialize
the dynamics model with the ground-truth dynamics (GT). For every combination we indicate whether we can
provide theoretical identifiability guarantees (“theory”) and compare this to empirical identifiability measures
( _R_ [2], LDS, dynR [2] ). Mean _±_ std. are across 3 datasets (5 for Lorenz) and 3 experiment repeats.


Data Model Results

ˆ
_**f**_ _p_ ( _**ε**_ ) _**f**_ identifiable % _R_ [2] _↑_ LDS [ _×_ 10 _[−]_ [2] ] _↓_


identity Normal identity ✓ 99.56 _±_ 0.21 0.00 _±_ 0.00
identity Normal LDS ✓ 99.31 _±_ 0.43 0.04 _±_ 0.01


LDS (low ∆ _t_ ) Normal (large _σ_ ) identity – 89.22 _±_ 4.47 8.53 _±_ 0.05
LDS Normal identity ✗ 73.56 _±_ 24.45 21.24 _±_ 0.31
LDS Normal LDS ✓ 99.03 _±_ 0.41 0.77 _±_ 1.07

LDS Normal GT ✓ 99.46 _±_ 0.39 0.44 _±_ 0.43


% _R_ [2] _↑_ %dyn _R_ [2] _↑_


SLDS Normal identity ✗ 76.80 _±_ 7.40 85.47 _±_ 8.07
SLDS Normal _∇_ -SLDS (✓) [1] 99.52 _±_ 0.05 99.93 _±_ 0.01
SLDS Normal GT (✓) [1] 99.20 _±_ 0.10 99.97 _±_ 0.00


Lorenz (small ∆ _t_ ) Normal (large _σ_ ) identity – 99.74 _±_ 0.36 99.94 _±_ 0.07
Lorenz (small ∆ _t_ ) Normal (large _σ_ ) LDS – 98.31 _±_ 2.55 97.21 _±_ 5.90
Lorenz (small ∆ _t_ ) Normal (large _σ_ ) _∇_ -SLDS – 94.14 _±_ 4.34 94.20 _±_ 6.57


Lorenz Normal identity ✗ 40.99 _±_ 8.58 27.02 _±_ 8.72
Lorenz Normal LDS ✗ 81.20 _±_ 16.93 80.30 _±_ 14.13
Lorenz Normal _∇_ -SLDS (✓) [2] 94.08 _±_ 2.75 93.91 _±_ 5.32


and positive). We use 2 [16] = 65536 negative samples for SLDS and 20 _k_ negative samples for LDS
data. For the Lorenz data, we use a batch size of 1024 and 20k negative samples. We use the Adam
optimizer (Kingma, 2014) with learning rates 3 _×_ 10 _[−]_ [4] for LDS data, 10 _[−]_ [3] for SLDS data, and
10 _[−]_ [4] for Lorenz system data. For the SLDS data, we use a different learning rate of 10 _[−]_ [2] for the
parameters of the dynamics model. We train for 50k steps on SLDS data and for 30k steps for LDS
and Lorenz system data. Our baseline model is standard self-supervised contrastive learning with the
InfoNCE loss, which is similar to the CEBRA-time model (with symmetric encoders, i.e., without
a dynamics model; cf. Schneider et al., 2023). For DCL, we add an LDS or _∇_ -SLDS dynamics
model for fitting. For our baseline, we post-hoc fit the corresponding model on the recovered latents
minimizing the predictive mean squared error via gradient descent.


**Evaluation metrics.** Our metrics are informed by the result in Theorem 1 and measure empirical
identifiability up to affine transformation of the latent space and its underlying linear or non-linear
dynamics. All metrics are estimated on the dataset the model is fit on. See Appendix F for additional
discussion on estimating metrics on independently sampled dynamics.


To account for the affine indeterminacy, we estimate _**L**_, _**b**_ for ˆ _**x**_ = _**Lx**_ + _**b**_ which allows us to map
ground truth latents _**x**_ to recovered latents ˆ _**x**_ (cf. Theorem 1a). In cases where the inverse transform
_**x**_ = _**L**_ _[−]_ [1] (ˆ _**x**_ _−_ _**b**_ ) is required, we can either compute _**L**_ _[−]_ [1] directly, or for the purpose of numerical
stability estimate it from data, which we denote as _**L**_ _[′]_ . The values of _**L**_, _**b**_ and _**L**_ _[′]_, _**b**_ _[′]_ are computed via
a linear regression:



_T_
� _∥_ _**x**_ _t_ _−_ ( _**L**_ _[′]_ _**x**_ ˆ _t_ + _**b**_ _[′]_ ) _∥_ 2 [2] _[.]_ (10)


_t_ =1



min
_**L**_ _,_ _**b**_



_T_
� _t_ =1 _∥_ _**x**_ ˆ _t_ _−_ ( _**Lx**_ _t_ + _**b**_ ) _∥_ 2 [2] and _**L**_ min _[′]_ _,_ _**b**_ _[′]_



To evaluate the identifiability of the representation, we measure the _R_ [2] between the true latents _**x**_ _t_
and the optimally aligned recovered latents _**L**_ _[′]_ _**x**_ ˆ _t_ + _**b**_ _[′]_ across time steps _t_ = 1 _. . . T_ in the time-series.


1 Not explicitly shown, but the argument in Corollary 2 applies to each piecewise linear section of the SLDS.
2 _∇_ -SLDS is only an approximation of the functional form of the underlying system.


6


Published as a conference paper at ICLR 2025


a b b c


Figure 4: Switching linear dynamics: (a) example ground-truth dynamics in latent space for four matrices _**A**_ _k_ .
(b) _R_ [2] metric for different noise levels as we increase the angles used for data generation. We compare a baseline
(no dynamics) to _∇_ -SLDS and a model fitted with ground-truth dynamics. (c) cluster accuracies for models
shown in (b).


We also propose two metrics as direct measures of identifiability for the recovered dynamics _**f**_ [ˆ] .
For linear dynamics models, we introduce the LDS error. It denotes the norm of the difference
between the true dynamics matrix _**A**_ and the estimated dynamics matrix _**A**_ [ˆ] by accounting for the
linear transformation between the true and recovered latent spaces. The LDS error (related to the
metric for Dynamical Similarity Analysis; Ostrow et al., 2023) is computed as (cf. Corollary 2):


LDS( _**A**_ _,_ _**A**_ [ˆ] ) = _∥_ _**A**_ _−_ _**L**_ _[−]_ [1] [ ˆ] _**AL**_ _∥_ _F_ _._ (11)


As a second, more general identifiability metric for the recovered dynamics _**f**_ [ˆ], we introduce dyn _R_ [2],
which builds on Theorem 1b to evaluate the identifiability of non-linear dynamics. This metric
computes the _R_ [2] between the predicted dynamics _**f**_ [ˆ] and the true dynamics _**f**_, corrected for the linear
transformation between the two latent spaces. Specifically, motivated by Theorem 1(b), we compute


dyn _R_ [2] ( _**f**_ _,_ _**f**_ [ˆ] ) = r2 ~~s~~ core( _**f**_ [ˆ] (ˆ _**x**_ ) _,_ _**Lf**_ ( _**L**_ _[′]_ _**x**_ ˆ + _**b**_ _[′]_ ) + _**b**_ ) (12)


along all time steps. Additional variants of the dynR [2] metric are discussed in Appendix G.


Finally, when evaluating switching linear dynamics, we compute the accuracy for assigning the
correct mode at any point in time. To compute the cluster accuracy in the case of SLDS ground truth
dynamics, we leverage the Hungarian algorithm to match the estimated latent variables modeling
mode switches to the ground truth modes, and then proceed to compute the accuracy.


**Implementation.** Experiments were carried out on a compute cluster with A100 cards. On each
card, we ran _∼_ 3 experiments simultaneously. Depending on the exact configuration, training time
varied from 5–20min per model. The combined experiments ran for this paper comprised about 120
days of A100 compute time and we provide a breakdown in Appendix K. We will open source our
benchmark suite for identifiable dynamics learning upon publication of the paper.


6 R ESULTS


6.1 V ERIFICATION OF THE THEORY FOR LINEAR DYNAMICS


**Suitable dynamics models enable identification of latents and dynamics.** For all considered
classes of models, we show in Table 1 that DCL with a suitable dynamics model effectively identifies
the correct dynamics. For linear dynamics (LDS), DCL reaches an _R_ [2] of 99.0%, close to the
oracle performance (99.5%). Most importantly, the average LDS error of our method (7.7 _×_ 10 _[−]_ [3] )
is very close to the oracle (4.4 _×_ 10 _[−]_ [3] ), in contrast to the baseline model (2.1 _×_ 10 _[−]_ [1] ) which has a
substantially larger LDS error. In the case of switching linear dynamics (SLDS), DCL also shows
strong performance, both in terms of latent _R_ [2] (99.5%) and dynamics _R_ [2] (99.9%) outperforming
the respective baselines (76.8% _R_ [2] and 85.5% dynamics _R_ [2] ). For non-linear dynamics, the baseline
model fails entirely (41.0%/27.0%), while _∇_ -SLDS dynamics can be fitted with 94.1% _R_ [2] for latents
and 93 _._ 9 % dynamics _R_ [2] . We also clearly see the strength of our piecewise-linear approximation, as
the LDS dynamics models only reaches 81 _._ 2% latent identifiability and 80 _._ 3% dynamics _R_ [2] .


7


Published as a conference paper at ICLR 2025


a time b


c d









Figure 5: Contrastive learning of 3D non-linear dynamics following a Lorenz attractor model. (a), left to right:
ground truth dynamics for 10k samples with _dt_ = 0 _._ 0005 and _σ_ = 0 _._ 1, estimation results for baseline (identity
dynamics), DCL with _∇_ -SLDS, estimated mode sequence. (b), empirical identifiability ( _R_ [2] ) between baseline
(BAS) and _∇_ -SLDS for varying numbers of discrete states _K_ . (c, d), same layout but for _dt_ = 0 _._ 01 and
_σ_ = 0 _._ 001.


**Learning noisy dynamics does not require a dynamics model.** If the variance of the distribution
for _**ε**_ _t_ dominates the changes actually introduced by the dynamics, we find that the baseline model
is also able to identify the latent space underlying the system. Intuitively, the change introduced by
the dynamical system is then negligible compared to the noise. In Table 1 (“large _σ_ ”), we show that
recovery is possible for cases with small angles, both in the linear and non-linear case. While in some
cases, this learning setup might be applicable in practice, it seems generally unrealistic to be able to
perturb the system beyond the actual dynamics. As we scale the dynamics to larger values (Figure 4,
panel b and c), the estimation scheme breaks again. However, this property offers an explanation for
the success of existing contrastive estimation algorithms like CEBRA-time (Schneider et al., 2023)
which successfully estimate dynamics in absence of a dynamics model.


**Symmetric encoders cannot identify non-trivial dynamics.** In the more general case where the
dynamics dominates the system behavior, the baseline cannot identify linear dynamics (or more
complicated systems). In the general LDS and SLDS cases, the baseline fails to identify the ground
truth dynamics (Table 1) as predicted by Corollary 1 (rows marked with ✗ ). For identity dynamics,
the baseline is able to identify the latents ( _R_ [2] =99.56%) but breaks as soon as linear dynamics are
introduced ( _R_ [2] =73.56%).


6.2 A PPROXIMATION OF NON - LINEAR DYNAMICS


Next, we study in more details how the DCL can identify piecewise linear or non-linear latent
dynamics using the _∇_ -SLDS dynamics model.


**Identification of switching dynamics.** Switching dynamics are depicted in Fig. 4a for four different
modes of the 10 degrees dataset. DCL obtains high _R_ [2] for various choices of dynamics (Fig. 4b)
and additionally identifies the correct mode sequence (Fig. 4c) for all noise levels and variants of
the underlying dynamics. As we increase the rotation angle used to generate the matrices, the gap
between baseline and our model increases substantially.


**Non-linear dynamics.** Figure 5 depicts the Lorenz system as an example of a non-linear dynamical
system for different choices of algorithms. The ground truth dynamics vary in the ratio between
_dt_ / _σ_ and we show the full range in panels b/c. When the noise dominates the dynamics (panel a),
the baseline is able to estimate also the nonlinear dynamics accurately, with 99.7%. However, as
we move to lower noise cases (panel b), performance reduces to 41.0%. Our switching dynamics
model is able to estimate the system with high _R_ [2] in both cases (94.14% and 94.08%). However, note
that in this non-linear case, we are primarily succeeding at estimating the latent space, the estimated
dynamics model did not meaningfully outperform an identity model (Appendix G).


**Extensions to other distributions** _p_ _**ε**_ **.** While Euclidean geometry is most relevant for dynamical
systems in practice, and hence the focus of our theoretical and empirical investigation, contrastive
learning commonly operates on the hypersphere in other contexts. We provide additional results
for the case of a von Mises-Fisher (vMF) distribution for _p_ _**ε**_ and dot-product similarity for _ϕ_ in
Appendix D.


8


Published as a conference paper at ICLR 2025


a b c d e f



Figure 6: Variations and ablations for the SLDS. We compare the _∇_ -SLDS model to the ground-truth switching
dynamics (oracle) and a standard CL model without dynamics (baseline). All variations are with respect to the
setting with 1M time steps (1k trials _×_ 1k samples), _L_ = 4 mixing layers, _d_ = 6 latent dimensionality, 5 modes,
and _p_ = 0 _._ 0001 switching probability. We study the impact of the dataset size in terms of (a) samples per trial,
(b) the number of trials, the impact of nonlinearity of the observations in terms of (c) number of mixing layers,
the impact of complexity of the latent dynamics in terms of (d) latent dimensionality, (e) number of modes to
switch in between and (f) the switching frequency paramameterized via the switching probability.


6.3 A BLATION STUDIES


For practitioners leveraging contrastive learning for statistical analysis, it is important to know the
trade-offs in empirical performance in relation to various parameters. In real-world experiments,
the most important factors are the size of the dataset, the trial-structure of the dataset, the latent
dimensionality we can expect to recover, and the degree of non-linearity between latents and observables. We consider these factors of influence: As a reference, we use the SLDS system with a 6D
latent space, 1M samples (1k trials _×_ 1k samples), _L_ = 4 mixing layers, 10 degrees for the rotation
matrices, 65,536 negative samples per batch; batch size 2,048, and learning rate 10 _[−]_ [3] .


**Impact of dataset size (Fig. 6a).** We keep the number of trials fixed to 1k. As we vary the sample
size per trial, _R_ [2] degrades for smaller dataset, and for the given setting we need at least 100 points
per trial to attain identifiability empirically. We outperform the baseline model in all cases.


**Impact of trials (Fig. 6b).** We next simulate a fixed number of 1M datapoints, which we split into
trials of varying length. We consider 1k, 10k, 100k, and 1M as trial lengths. Performance is stable
for the different settings, even for cases with small trial length (and less observed switching points).
DCL consistently outperforms the baseline algorithm and attains stable performance close to the
theoretical maximum given by the ground-truth dynamics.



**Impact of non-linear mixing (Fig. 6c).** All main experiments have
been conducted with _L_ = 4 mixing layers in the mixing function
_**g**_ . Performance of DCL stays at the theoretical maximum as we increase the number of mixing layers. As we move beyond four layers,
both oracle performance in _R_ [2] and our model declines, hinting that
either (1) more data or (2) a larger model is required to recover the
dynamics successfully in these cases.


**Impact of dimensionality (Fig. 6d).** Increasing latent dimensionality does not meaningfully impact performance of our model. We
found that for higher dimensions, it is crucial to use a large number
of negative examples (65k) for successful training.



Figure 7: Impact of modes
for non-linear dynamics in the
Lorenz system for different system noise levels _σ_, averaged over
all _dt_ .





**Number of modes for switching linear dynamics fitting (Fig. 6e).** for non-linear dynamics in the
Increasing the number of modes in the dataset leads to more suc- Lorenz system for different syscessful fitting of the _R_ [2] for the baseline model, but to a decline in tem noise levels _σ_, averaged over

all _dt_ .

accuracy. This might be due to the increased variance: While this
helps the model to identify the latent space (dynamics appear more like noise), it still fails to identify
the underlying dynamics model, unlike DCL which attains high _R_ [2] and cluster accuracy throughout.



9


Published as a conference paper at ICLR 2025


**Robustness to changes in switching probability (Fig. 6f).** Finally, we vary the switching probability.
Higher switching probability causes shorter modes, which are harder to fit by the _∇_ -SLDS dynamics
model. Our model obtains high empirical identifiability throughout the experiment, but the accuracy
metric begins to decline when _p_ = 0 _._ 1 and _p_ = 0 _._ 2.


**Number of modes for non-linear dynamics fitting (Fig. 7).** We study the effect of increasing the
number of matrices in the parameter bank **W** in the _∇_ -SLDS model. The figure depicts the impact
of increasing the number of modes for DCL on the non-linear Lorenz dataset. We observe that
increasing modes to 200 improves performance, but eventually converges to a stable maximum for
all noise levels.


7 D ISCUSSION


The DCL framework is versatile and allows to study the performance of contrastive learning in
conjunction with different dynamics models. By exploring various special cases (identity, linear,
switching linear), our study categorizes different forms of contrastive learning and makes predictions
about their behavior in practice. In comparison to contrastive predictive coding (CPC; Oord et al.,
2018) or wav2vec (Schneider et al., 2019), DCL generalizes the concept of training contrastive
learning models with (explicit) dynamics models. CPC uses an RNN encoder followed by linear
projection, while wav2vec leverages CNNs dynamics models and affine projections. Theorem 1
applies to both these models, and offers an explanation for their successful empirical performance.


Nonlinear ICA methods, such as TCL (Hyvarinen & Morioka, 2016) and PCL (Hyvarinen & Morioka,
2017) provide identifiability of the latent variables leveraging temporal structure of the data. Compared to DCL, they do not explicitly model dynamics and assume either stationarity or non-stationarity
of the time series (Hyvarinen et al., 2023), whereas DCL assumes bijective latent dynamics, and ¨
focuses on explicit dynamics modeling beyond solving the demixing problem.


For applications in scientific data analysis, CEBRA (Schneider et al., 2023) uses supervised or
self-supervised contrastive learning, either with symmetric encoders or asymmetric encoder functions.
While our results show that such an algorithm is able to identify dynamics for a sufficient amount
of system noise, adding dynamics models is required as the system dynamics dominate. Hence, the
DCL approach with LDS or _∇_ -SLDS dynamics generalises the self-supervised mode of CEBRA and
makes it applicable for a broader class of problems.


Finally, there is a connection to the joint embedding predictive architecture (JEPA; LeCun, 2022;
Assran et al., 2023). The architecture setup of DCL can be regarded as a special case of JEPA,
but with symmetric encoders to leverage distillation of the system dynamics into the predictor (the
dynamics model). In contrast to JEPA, the use of symmetric encoders requires a contrastive loss for
avoiding collapse and, more importantly, serves as the foundation for our theoretical result.


A limitation of the present study is its main focus on simulated data which clearly corroborates our
theory but does not yet demonstrate real-world applicability. However, our simulated data bears the
signatures of real-world datasets (multi-trial structures, varying degrees of dimensionality, number of
modes, and different forms of dynamics). A challenge is the availability of real-world benchmark
datasets for dynamics identification. We believe that rigorous evaluation of different estimation
methods on such datasets will continue to show the promise of contrastive learning for dynamics
identification. Integrating recent benchmarks like DynaDojo (Bhamidipaty et al., 2023) or datasets
from Chen et al. (2021) with realistic mixing functions ( _**g**_ ) offers a promising direction for evaluating
latent dynamics models. As a demonstration of real-world applicability, we benchmarked DCL on a
neural recordings dataset in Appendix J.


8 C ONCLUSION


We proposed a first identifiable, end-to-end, non-generative inference algorithm for latent switching
dynamics along with an empirically successful parameterization of non-linear dynamics. Our results
point towards the empirical effectiveness of contrastive learning across time-series, and back these
empirical successes by theory. We show empirical identifiability with limited data for linear, switching
linear and non-linear dynamics. Our results add to the understanding of SSL’s empirical success, will
guide the design of future contrastive learning algorithms and most importantly, make SSL amenable
for computational statistics and data analysis.


10


Published as a conference paper at ICLR 2025


R EPRODUCIBILITY STATEMENT


**Code.** Code is available at [https://github.com/dynamical-inference/dcl](https://github.com/dynamical-inference/dcl) under an
Apache 2.0 license. Experimental and implementation details for the main text are given in section 5
and for each experiment of the Appendix within the respective chapter.


**Theory.** Our theoretical claims are backed by a complete proof attached in Appendix A. Assumptions
are outlined in the main text (Section 3) and again in more detail in Appendix A.


**Datasets.** We evaluate our experiments on a variety of synthetic datasets. The datasets comprise
different dynamical systems, from linear to nonlinear.


**Compute.** Moderate compute resources are required to reproduce this paper. As stated in section 5,
we used around 120 days of GPU compute on a A100 to produce the results presented in the paper.
We provide a more detailed breakdown in Appendix K.


A UTHOR C ONTRIBUTIONS


RGL and TS: Methodology, Software, Investigation, Writing–Editing. StS: Conceptualization,
Methodology, Formal Analysis, Writing–Original Draft and Writing–Editing.


A CKNOWLEDGMENTS


We thank Luisa Eck and Stephen Jiang for discussions on the theory, and Lilly May for input on
paper figures. We thank the five anonymous reviewers at ICLR for their valuable and constructive
comments on our manuscript. This work was supported by the Helmholtz Association’s Initiative and
Networking Fund on the HAICORE@KIT and HAICORE@FZJ partitions.


R EFERENCES


Guy Ackerson and K Fu. On state estimation in switching environments. _IEEE transactions on_
_automatic control_, 15(1):10–17, 1970.


Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat,
Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding
predictive architecture. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pp. 15619–15629, 2023.


Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A
general framework for self-supervised learning in speech, vision and language. In _International_
_Conference on Machine Learning_, pp. 1298–1312. PMLR, 2022.


Carles Balsells-Rodas, Yixin Wang, and Yingzhen Li. On the identifiability of switching dynamical
systems. _arXiv preprint arXiv:2305.15925_, 2023.


Ror Bellman and Karl Johan Astr [˚] om. On structural identifiability. ¨ _Mathematical biosciences_, 7(3-4):
329–339, 1970.


Logan Mondal Bhamidipaty, Tommy Bruzzese, Caryn Tran, Rami Ratl Mrad, and Max Kanwal.
Dynadojo: an extensible benchmarking platform for scalable dynamical system identification. In
_Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks_
_Track_, 2023.


Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx,
Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. _arXiv preprint arXiv:2108.07258_, 2021.


Tom B Brown. Language models are few-shot learners. _arXiv preprint arXiv:2005.14165_, 2020.


Steven L Brunton, Joshua L Proctor, and J Nathan Kutz. Discovering governing equations from data
by sparse identification of nonlinear dynamical systems. _Proceedings of the national academy of_
_sciences_, 113(15):3932–3937, 2016.


Chaw-Bing Chang and Michael Athans. State estimation for discrete systems with switching
parameters. _IEEE Transactions on Aerospace and Electronic Systems_, (3):418–425, 1978.


11


Published as a conference paper at ICLR 2025


Boyuan Chen, Kuang Huang, Sunand Raghupathi, Ishaan Chandratreya, Qiang Du, and Hod Lipson.
Discovering State Variables Hidden in Experimental Data, December 2021. URL [http://](http://arxiv.org/abs/2112.10755)
[arxiv.org/abs/2112.10755.](http://arxiv.org/abs/2112.10755)


Ricky TQ Chen, Brandon Amos, and Maximilian Nickel. Learning neural event functions for ordinary
differential equations. _arXiv preprint arXiv:2011.03902_, 2020a.


Sheng Chen and Steve A Billings. Representations of non-linear systems: the narmax model.
_International journal of control_, 49(3):1013–1032, 1989.


Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations. In _International conference on machine learning_, pp.
1597–1607. PMLR, 2020b.


Silvia Chiappa et al. Explicit-duration markov switching models. _Foundations and Trends® in_
_Machine Learning_, 7(6):803–886, 2014.


Sy-Miin Chow and Guangjian Zhang. Nonlinear regime-switching state-space (rsss) models. _Psy-_
_chometrika_, 78:740–768, 2013.


Hanjun Dai, Bo Dai, Yan-Ming Zhang, Shuang Li, and Le Song. Recurrent hidden semi-markov
model. In _International Conference on Learning Representations_, 2022.


Stephane d’Ascoli, S ´ oren Becker, Alexander Mathis, Philippe Schwaller, and Niki Kilbertus. ¨
Odeformer: Symbolic regression of dynamical systems with transformers. _arXiv preprint_
_arXiv:2310.05573_, 2023.


Saskia EJ de Vries, Jerome A Lecoq, Michael A Buice, Peter A Groblewski, Gabriel K Ocker,
Michael Oliver, David Feng, Nicholas Cain, Peter Ledochowitsch, Daniel Millman, et al. A
large-scale standardized physiological survey reveals functional organization of the mouse visual
cortex. _Nature neuroscience_, 23(1):138–151, 2020.


Zhe Dong, Bryan Seybold, Kevin Murphy, and Hung Bui. Collapsed amortized variational inference
for switching nonlinear dynamical systems. In _International Conference on Machine Learning_, pp.
2638–2647. PMLR, 2020.


Lea Duncker, Gergo Bohner, Julien Boussard, and Maneesh Sahani. Learning interpretable
continuous-time models of latent stochastic dynamical systems. In _International conference_
_on machine learning_, pp. 1726–1734. PMLR, 2019.


Jeffrey L Elman. Finding structure in time. _Cognitive science_, 14(2):179–211, 1990.


Yuanjun Gao, Evan W Archer, Liam Paninski, and John P Cunningham. Linear dynamical neural
population models through nonlinear embeddings. _Advances in neural information processing_
_systems_, 29, 2016.


Quentin Garrido, Mahmoud Assran, Nicolas Ballas, Adrien Bardes, Laurent Najman, and Yann
LeCun. Learning and leveraging world models in visual representation learning. _arXiv preprint_
_arXiv:2403.00504_, 2024.


Zoubin Ghahramani and Geoffrey E Hinton. Variational learning for switching state-space models.
_Neural computation_, 12(4):831–864, 2000.


Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv_
_preprint arXiv:2312.00752_, 2023.


Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured ´
state spaces. _arXiv preprint arXiv:2111.00396_, 2021.


David Ha and Jurgen Schmidhuber. World models. ¨ _CoRR_, abs/1803.10122, 2018. URL [http:](http://arxiv.org/abs/1803.10122)
[//arxiv.org/abs/1803.10122.](http://arxiv.org/abs/1803.10122)


Hermanni Halv ¨ a, Sylvain Le Corff, Luc Leh ¨ ericy, Jonathan So, Yongjie Zhu, Elisabeth Gassiat, and ´
Aapo Hyvarinen. Disentangling identifiable features from noisy data with structured nonlinear ica.
_Advances in Neural Information Processing Systems_, 34:1624–1633, 2021.


12


Published as a conference paper at ICLR 2025


Olivier Henaff. Data-efficient image recognition with contrastive predictive coding. In _International_
_conference on machine learning_, pp. 4182–4192. PMLR, 2020.


Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). _arXiv preprint_
_arXiv:1606.08415_, 2016.


Cole Hurwitz, Nina Kudryashova, Arno Onken, and Matthias H Hennig. Building population models
for large-scale neural recordings: Opportunities and pitfalls. _Current opinion in neurobiology_, 70:
64–73, 2021.


Aapo Hyvarinen and Hiroshi Morioka. Unsupervised feature extraction by time-contrastive learning
and nonlinear ica. _Advances in neural information processing systems_, 29, 2016.


Aapo Hyvarinen and Hiroshi Morioka. Nonlinear ica of temporally dependent stationary sources. In
_Artificial Intelligence and Statistics_, pp. 460–469. PMLR, 2017.


Aapo Hyvarinen, Hiroaki Sasaki, and Richard Turner. Nonlinear ica using auxiliary variables and
generalized contrastive learning. In _The 22nd International Conference on Artificial Intelligence_
_and Statistics_, pp. 859–868. PMLR, 2019.


Aapo Hyvarinen, Ilyes Khemakhem, and Hiroshi Morioka. ¨ Nonlinear independent component analysis for principled disentanglement in unsupervised deep learning. _Patterns_, 4(10):
100844, October 2023. ISSN 26663899. doi: 10.1016/j.patter.2023.100844. URL [https:](https://linkinghub.elsevier.com/retrieve/pii/S2666389923002234)
[//linkinghub.elsevier.com/retrieve/pii/S2666389923002234.](https://linkinghub.elsevier.com/retrieve/pii/S2666389923002234)


Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. _arXiv_
_preprint arXiv:1611.01144_, 2016.


Pierre-Alexandre Kamienny, Stephane d’Ascoli, Guillaume Lample, and Fran ´ c¸ ois Charton. End-toend symbolic regression with transformers. _Advances in Neural Information Processing Systems_,
35:10269–10281, 2022.


Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders
and nonlinear ica: A unifying framework. In _International conference on artificial intelligence_
_and statistics_, pp. 2207–2217. PMLR, 2020.


Diederik P Kingma. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_,
2014.


Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. _Open_
_Review_, 62(1):1–62, 2022.


Scott Linderman, Matthew Johnson, Andrew Miller, Ryan Adams, David Blei, and Liam Paninski.
Bayesian learning and inference in recurrent switching linear dynamical systems. In _Artificial_
_intelligence and statistics_, pp. 914–922. PMLR, 2017.


Phillip Lippe, Sara Magliacane, Sindy Lowe, Yuki M Asano, Taco Cohen, and Stratis Gavves. Citris: ¨
Causal identifiability from temporal intervened sequences. In _International Conference on Machine_
_Learning_, pp. 13557–13603. PMLR, 2022.


Stefan Matthes, Zhiwei Han, and Hao Shen. Towards a unified framework of contrastive learning
for disentangled representations. _Advances in Neural Information Processing Systems_, 36:67459–
67470, 2023.


Leonard A McGee and Stanley F Schmidt. Discovery of the kalman filter as a practical tool for
aerospace and industry. Technical report, 1985.


Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive
coding. _arXiv preprint arXiv:1807.03748_, 2018.


Mitchell Ostrow, Adam Eisen, Leo Kozachkov, and Ila Fiete. Beyond geometry: Comparing the
temporal structure of computation in neural circuits with dynamical similarity analysis, 2023. URL
[https://arxiv.org/abs/2306.10168.](https://arxiv.org/abs/2306.10168)


13


Published as a conference paper at ICLR 2025


Chethan Pandarinath, Daniel J O’Shea, Jasmine Collins, Rafal Jozefowicz, Sergey D Stavisky,
Jonathan C Kao, Eric M Trautmann, Matthew T Kaufman, Stephen I Ryu, Leigh R Hochberg, et al.
Inferring single-trial neural population dynamics using sequential auto-encoders. _Nature methods_,
15(10):805–815, 2018.


Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019.


Geoffrey Roeder, Luke Metz, and Durk Kingma. On linear identifiability of learned representations.
In _International Conference on Machine Learning_, pp. 9030–9039. PMLR, 2021.


Evgenia Rusak, Patrik Reizinger, Attila Juhos, Oliver Bringmann, Roland S. Zimmermann, and
Wieland Brendel. Infonce: Identifying the gap between theory and practice. 2024. URL [https:](https://arxiv.org/abs/2407.00143)
[//arxiv.org/abs/2407.00143.](https://arxiv.org/abs/2407.00143)


Steffen Schneider, Alexei Baevski, Ronan Collobert, and Michael Auli. wav2vec: Unsupervised
pre-training for speech recognition. _arXiv preprint arXiv:1904.05862_, 2019.


Steffen Schneider, Jin Hwa Lee, and Mackenzie Weygandt Mathis. Learnable latent embeddings for
joint behavioural and neural analysis. _Nature_, 617(7960):360–368, 2023.


Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey
Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In
_2018 IEEE international conference on robotics and automation (ICRA)_, pp. 1134–1141. IEEE,
2018.


Ruian Shi and Quaid Morris. Segmenting hybrid trajectories using latent odes. In _International_
_Conference on Machine Learning_, pp. 9569–9579. PMLR, 2021.


Jimmy Smith, Scott Linderman, and David Sussillo. Reverse engineering recurrent neural networks
with jacobian switching linear dynamical systems. _Advances in Neural Information Processing_
_Systems_, 34:16700–16713, 2021.


Peter Sorrenson, Carsten Rother, and Ullrich Kothe. Disentanglement by nonlinear ica with general ¨
incompressible-flow networks (gin). _arXiv preprint arXiv:2001.04872_, 2020.


A Vaswani. Attention is all you need. _Advances in Neural Information Processing Systems_, 2017.


Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In _International Conference on Machine Learning_, pp.
9929–9939. PMLR, 2020.


Weiran Yao, Yuewen Sun, Alex Ho, Changyin Sun, and Kun Zhang. Learning temporally causal
latent processes from general temporal data. _arXiv preprint arXiv:2110.05428_, 2021.


Weiran Yao, Guangyi Chen, and Kun Zhang. Temporally disentangled representation learning.
_Advances in Neural Information Processing Systems_, 35:26492–26503, 2022.


Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, and Wieland Brendel.
Contrastive learning inverts the data generating process. In _Proceedings of the 38th International_
_Conference on Machine Learning_, volume 139 of _Proceedings of Machine Learning Research_, pp.
12979–12990. PMLR, 2021.


14


Published as a conference paper at ICLR 2025

# **Supplementary Material**


**A Proof of the main result** **16**


**B** **Kernel density estimate correction** **19**


**C Additional Related Work** **21**


**D Von Mises–Fisher (vMF) conditional distributions** **23**


**E Additional plots for SLDS** **24**


**F** **Generalization – Train- vs. Test Set** **25**


**G Variations and additional baselines for the Dyn** _R_ [2] **Metric** **27**

G.1 Method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

G.2 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27


**H Non-Injective Mixing Functions** **28**
H.1 Experimental Validation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

H.2 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29


**I** **Dynamical Systems with Control Signal** **31**
I.1 Empirical Verification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
I.2 Experiment Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

I.3 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32


**J** **Application to Real-World Data** **33**

J.1 Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

J.2 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34

J.3 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34


**K Computational Requirements** **35**


**L** **On component-wise vs. linear identifiability** **36**


**M Comparison to additional time-series models** **37**
M.1 Verifying Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
M.2 Experiment Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

M.3 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38


15


Published as a conference paper at ICLR 2025


A P ROOF OF THE MAIN RESULT


We re-state Theorem 1 from the main paper, and provide a full proof below:

**Theorem 1** (Contrastive estimation of non-linear dynamics) **.** _Assume that_


    - _(A1) A time-series dataset_ _{_ _**y**_ _t_ _}_ _[T]_ _t_ =1 _[is generated according to the ground-truth dynamical]_
_system in Eq. 5 with a bijective dynamics model_ _**f**_ _and an injective mixing function_ _**g**_ _._

    - _(A2) The system noise follows an iid normal distribution, p_ ( _**ε**_ _t_ ) = _N_ ( _**ε**_ _t_ _|_ 0 _,_ **Σ** _**ε**_ ) _._

    - _(A3) The model_ _ψ_ _is composed of an encoder_ _**h**_ _, a dynamics model_ _**f**_ [ˆ] _, a correction term_ _α_ _,_
_and the similarity metric ϕ_ ( _**u**_ _,_ _**v**_ ) = _−∥_ _**u**_ _−_ _**v**_ _∥_ [2] _and attains the global minimizer of Eq. 3._


_Then, in the limit of T →∞_ _for any point_ _**x**_ _in the support of the data marginal distribution:_


_(a)_ _The composition of mixing and de-mixing_ _**h**_ ( _**g**_ ( _**x**_ )) = _**Lx**_ + _**b**_ _is a bijective affine transform,_
_and_ _**L**_ = _**Q**_ **Σ** _[−]_ _ϵ_ [1] _[/]_ [2] _with unknown orthogonal transform_ _**Q**_ _∈_ R _[d][×][d]_ _and offset_ _**b**_ _∈_ R _[d]_ _._
_(b)_ _The estimated dynamics_ _**f**_ ˆ( _**x**_ ) = _**Lf**_ ( _**L**_ _[−]_ [1] ( _**x**_ _−_ _**bf**_ )) + [ˆ] _are bijective and identify the true dynamics_ _**b**_ _._ _**f**_ _up to the relation_


_Proof._ Our proof proceeds in three steps: First, we leverage existing theory (Wang & Isola, 2020;
Zimmermann et al., 2021) to arrive at the minimizer of the contrastive loss, and relate the limited
sample loss function to the asymptotic case. Second, we derive the statement about achieving
successful demixing in Theorem 1(a). Finally, we derive the statement in Theorem 1(a) about
structural identifiability of the dynamics model.


**Step 1: Minimizer of the InfoNCE loss.** By the assumption _**ε**_ _t_ is normally distributed, we obtain
the positive sample conditional distribution


_p_ ( _**x**_ _t_ +1 _|_ _**x**_ _t_ ) = _N_ ( _**x**_ _t_ +1 _|_ _**f**_ ( _**x**_ _t_ ) _,_ **Σ** _**ε**_ ) _._ (13)


The negative sample distribution _q_ ( _**x**_ _t_ ) is obtained by sampling _t_ uniformly from all time steps and
can hence be written as a Gaussian mixture along the dynamics imposed by _**f**_,



_q_ ( _**x**_ ) = [1]

_T_


= [1]

_T_



_T_
� _p_ _**ε**_ ( _**x**_ _−_ _**x**_ _t_ ) (14)


_t_ =1


_T_
� _N_ ( _**x**_ _−_ _**x**_ _t_ _|_ _**f**_ ( _**x**_ _t−_ 1 ) _,_ **Σ** _**ε**_ ) _._ (15)


_t_ =1



We use these definitions of _p_ and _q_ to study the asymptotic case of our loss function. For _T →∞_,
due to Wang & Isola (2020) we can rewrite the limit of our loss function (Eq. 3) as


_L_ [ _ψ_ ] = lim
_T →∞_ [E] _[t,N]_ [[] _[−]_ [log] _[ p]_ _[ψ]_ [(] _**[y]**_ _[t]_ [+1] _[|]_ _**[y]**_ _[t]_ _[, N]_ [)]] _[ −]_ [log] _[ T]_

(16)
= _q_ ( _**y**_ ) _−_ _p_ ( _**y**_ _[′]_ _|_ _**y**_ ) _ψ_ ( _**y**_ _,_ _**y**_ _[′]_ ) _d_ _**y**_ _[′]_ + log _q_ ( _**y**_ _[′]_ ) exp[ _ψ_ ( _**y**_ _,_ _**y**_ _[′]_ )] _d_ _**y**_ _[′]_ _._
� � � � �


It was shown (Proposition 1, Schneider et al., 2023) that this loss function is convex in _ψ_ with the
unique minimizer

_ψ_ ( _**g**_ ( _**x**_ ) _,_ _**g**_ ( _**x**_ _[′]_ )) = log _[p]_ [(] _**[x]**_ _[′]_ _[|]_ _**[x]**_ [)] + _c_ ( _**x**_ ) _,_ (17)

_q_ ( _**x**_ _[′]_ )


where _c_ : R _[d]_ _�→_ R is an arbitrary scalar-valued function. Note that we also expressed _**y**_ = _**g**_ ( _**x**_ ) _,_ _**y**_ _[′]_ =
_**g**_ ( _**x**_ _[′]_ ) to continue the proof in terms of the relation between original and estimated latents. We insert
the definition of the model on the left hand side. Let us also denote _**h**_ _◦_ _**g**_ =: _**r**_, and the definition of
the ground-truth generating process on the right hand side to obtain


_ϕ_ ( _**f**_ [ˆ] ( _**r**_ ( _**x**_ )) _,_ _**r**_ ( _**x**_ _[′]_ )) _−_ _α_ ( _**x**_ _[′]_ ) = log _p_ ( _**x**_ _[′]_ _|_ _**f**_ ( _**x**_ )) _−_ log _q_ ( _**x**_ _[′]_ ) + _c_ ( _**x**_ ) _._ (18)


Inserting the potential [2] as _α_ ( _**x**_ ) = log _q_ ( _**x**_ ) simplifies the equation to


_ϕ_ ( _**f**_ [ˆ] ( _**r**_ ( _**x**_ )) _,_ _**r**_ ( _**x**_ _[′]_ )) = log _p_ ( _**x**_ _[′]_ _|_ _**f**_ ( _**x**_ )) + _c_ ( _**x**_ ) _._ (19)


16


Published as a conference paper at ICLR 2025


From here onwards, we will use that _ϕ_ is the negative squared Euclidean norm (A.3) and correspondingly, the positive conditional is a normal distribution with concentration **Λ** = **Σ** _[−]_ _u_ [1] (A.2),


_−∥_ _**f**_ [ˆ] ( _**r**_ ( _**x**_ )) _−_ _**r**_ ( _**x**_ _[′]_ ) _∥_ 2 [2] [=] _[ −]_ [(] _**[f]**_ [(] _**[x]**_ [)] _[ −]_ _**[x]**_ _[′]_ [)] _[⊤]_ **[Λ]** [(] _**[f]**_ [(] _**[x]**_ [)] _[ −]_ _**[x]**_ _[′]_ [) +] _[ c]_ _[′]_ [(] _**[x]**_ [)] (20)


where we pulled the normalization constant of _p_ into the function _c_ _[′]_ for brevity.


**Step 2: Properties of the feature encoder.** Starting from the last equation, we compute the derivative
with respect to _**x**_ and _**x**_ _[′]_ on both sides and obtain


_**J**_ _**r**_ _[⊤]_ [(] _**[x]**_ _[′]_ [)] _**[J]**_ [ ˆ] _**f**_ [(] _**[r]**_ [(] _**[x]**_ [))] _**[J]**_ _**[r]**_ [(] _**[x]**_ [) =] **[ Λ]** _**[J]**_ _**[f]**_ [(] _**[x]**_ [)] _[.]_ (21)


Because this equation holds for any _**x**_ _[′]_ _∈_ supp _q_ independently of _**x**_, we can conclude that the
Jacobian matrix of _**r**_ needs to be constant. From there it follows that _**r**_ is affine. Let us write
_**r**_ ( _**x**_ ) = _**Lx**_ + _**b**_ . Then, the Jacobian matrix is _**J**_ _**r**_ = _**L**_ . Inserting this yields


_**L**_ _[⊤]_ _**J**_ ˆ _**f**_ [(] _**[Lx]**_ [ +] _**[ b]**_ [)] _**[L]**_ [ =] **[ Λ]** _**[J]**_ _**[f]**_ [(] _**[x]**_ [)] _[.]_ (22)


We next establish that _**L**_ has full rank: because the dynamics function _**f**_ is bijective by assumption,
_**J**_ _**f**_ ( _**x**_ ) has full rank _d_ . **Λ** has full rank by assumption about the distribution _p_ _**ε**_ ( _**ε**_ ) . All matrices on
the LHS are square and need to have full rank as well for any point _**x**_ . Hence, we can conclude that
_**L**_ has full rank, and likewise _**J**_ ˆ _**f**_ [. From there, we conclude that] _**[ r]**_ [ and][ ˆ] _**[f]**_ [ are bijective.]


Next, we derive additional constraints on the matrix _**L**_ . We insert the solution for _**r**_ obtained so far in
Eq. 20,


_−∥_ _**f**_ [ˆ] ( _**Lx**_ + _**b**_ ) _−_ _**Lx**_ _[′]_ _−_ _**b**_ _∥_ [2] = _−_ ( _**f**_ ( _**x**_ ) _−_ _**x**_ _[′]_ ) _[⊤]_ **Λ** ( _**f**_ ( _**x**_ ) _−_ _**x**_ _[′]_ ) + _c_ _[′]_ ( _**x**_ ) (23)


and take the derivative twice with respect to _**x**_ _[′]_, to obtain


_**L**_ _[⊤]_ _**L**_ = **Λ** _⇔_ _**L**_ _[⊤]_ = **Λ** _**L**_ _[−]_ [1] _._ (24)


Without loss of generality, we introduce _**Q**_ _∈_ R _[d][×][d]_ to write _**L**_ in terms of **Λ** as _**L**_ = _**Q**_ **Λ** [1] _[/]_ [2] .
Inserting into the previous equation lets us conclude


_**L**_ _[⊤]_ _**L**_ = **Λ** [1] _[/]_ [2] _**Q**_ _[⊤]_ _**Q**_ **Λ** [1] _[/]_ [2] = **Λ** (25)

_**Q**_ _[⊤]_ _**Q**_ = **Λ** _[−]_ [1] _[/]_ [2] **ΛΛ** _[−]_ [1] _[/]_ [2] = _**I**_ _,_ (26)


from which follows that _**Q**_ is an orthogonal matrix. Hence, _**L**_ is a composition of an orthogonal
transform and **Σ** _[−]_ _**ε**_ [1] _[/]_ [2], concluding the first part of the proof for statement (a).


**Step 3: Dynamics.** To derive part (b), we start at the condition Eq. 20 again to determine the value of
_c_ _[′]_ ( _**x**_ ). We can consider two special cases:


_**f**_ ( _**x**_ ) = _**x**_ _[′]_ : _c_ _[′]_ ( _**x**_ ) = _−∥_ _**f**_ [ˆ] ( _**r**_ ( _**x**_ )) _−_ _**r**_ ( _**x**_ _[′]_ ) _∥_ [2] _≤_ 0 _,_ (27)

ˆ
_**f**_ ( _**r**_ ( _**x**_ )) = _**r**_ ( _**x**_ _[′]_ ) : _c_ _[′]_ ( _**x**_ ) = ( _**f**_ ( _**x**_ ) _−_ _**x**_ _[′]_ ) _[⊤]_ **Λ** ( _**f**_ ( _**x**_ ) _−_ _**x**_ _[′]_ ) _≥_ 0 _,_ (28)


where we use that the concentration matrix **Λ** of a Normal distribution is positive semi-definite. When
combining both conditions for points where _**f**_ ( _**x**_ ) = _**x**_ _[′]_ and _**f**_ [ˆ] ( _**r**_ ( _**x**_ )) = _**r**_ ( _**x**_ _[′]_ ) the only admissible
solution is _c_ _[′]_ ( _**x**_ ) = 0 for points with _**f**_ [ˆ] ( _**r**_ ( _**x**_ )) = _**r**_ ( _**f**_ ( _**x**_ )), i.e. _**f**_ [ˆ] ( _**x**_ ) = _**r**_ ( _**f**_ ( _**r**_ _[−]_ [1] ( _**x**_ ))), hinting at the
final solution. However, we have not shown yet that this solution is unique.


To show uniqueness, without loss of generality, we use the ansatz (with a residual _**v**_ )


_**f**_ ( _**x**_ ) = _**A**_ 1 _**f**_ [ˆ] ( _**Lx**_ + _**b**_ ) + _**d**_ 1 + _**v**_ ( _**x**_ ) _,_ (29)


and computing the derivative with respect to _**x**_ yields


_**J**_ _**f**_ ( _**x**_ ) = _**A**_ 1 _**J**_ ˆ _**f**_ [(] _**[Lx]**_ [ +] _**[ b]**_ [)] _**[L]**_ [ +] _**[ J]**_ _**[v]**_ [(] _**[x]**_ [)] _[.]_ (30)


2
This is feasible in practice by parameterizing _α_ ( _**x**_ ) as a kernel density estimate, but empirically often not
required. See Appendix B for additional technical details.


17


Published as a conference paper at ICLR 2025


We insert this into Eq. 22 and obtain


_**L**_ _[⊤]_ _**J**_ ˆ _**f**_ [(] _**[Lx]**_ [ +] _**[ b]**_ [)] _**[L]**_ [ =] **[ Λ]** _**[A]**_ [1] _**[J]**_ [ ˆ] _**f**_ [(] _**[Lx]**_ [ +] _**[ b]**_ [)] _**[L]**_ [ +] **[ Λ]** _**[J]**_ _**[v]**_ [(] _**[x]**_ [)] (31)

( _**L**_ _[⊤]_ _−_ **Λ** _**A**_ 1 ) _**J**_ ˆ _**f**_ [(] _**[Lx]**_ [ +] _**[ b]**_ [)] _**[L]**_ [ =] **[ Λ]** _**[J]**_ _**[v]**_ [(] _**[x]**_ [)] (32)

( _**L**_ _[⊤]_ _−_ **Λ** _**A**_ 1 ) = **Λ** _**J**_ _**v**_ ( _**x**_ ) _**L**_ _[−]_ [1] _**J**_ _**f**_ _[−]_ ˆ [1] [(] _**[Lx]**_ [ +] _**[ b]**_ [)] (33)


The left hand side is a constant, hence the same needs to hold true for the right hand side. Without
loss of generality, let us introduce an arbitrary matrix _**A**_ 2 we set as this constant,

_**J**_ _**v**_ ( _**x**_ ) _**L**_ _[−]_ [1] _**J**_ _**f**_ _[−]_ ˆ [1] [(] _**[Lx]**_ [ +] _**[ b]**_ [) =] _**[ A]**_ [2] (34)

_**J**_ _**v**_ ( _**x**_ ) = _**A**_ 2 _**J**_ ˆ _**f**_ [(] _**[Lx]**_ [ +] _**[ b]**_ [)] _**[L]**_ (35)


which only admits the solution


_**v**_ ( _**x**_ ) = _**A**_ 2 _**f**_ [ˆ] ( _**Lx**_ + _**b**_ ) + _**d**_ 2 _,_ (36)


where we introduced an additional integration constant _**d**_ 2 . Inserting this into the ansatz in Eq. 29
gives


_**f**_ ( _**x**_ ) = _**A**_ 1 _**f**_ [ˆ] ( _**Lx**_ + _**b**_ ) + _**d**_ 1 + _**v**_ ( _**x**_ ) _,_ (37)

_**f**_ ( _**x**_ ) = ( _**A**_ 1 + _**A**_ 2 ) _**f**_ [ˆ] ( _**Lx**_ + _**b**_ ) + ( _**d**_ 1 + _**d**_ 2 ) _._ (38)


Using the shorthand _**A**_ = _**A**_ 1 + _**A**_ 2, _**d**_ = _**d**_ 1 + _**d**_ 2 we can repeat the steps in Eqs. 30–33 to arrive at
the condition


( _**L**_ _[⊤]_ _−_ **Λ** _**A**_ ) _**J**_ ˆ _**f**_ [(] _**[Lx]**_ [ +] _**[ b]**_ [)] _**[L]**_ [ = 0] _[.]_ (39)


Since all matrices have full rank, the only valid solution is _**A**_ = **Λ** _[−]_ [1] _**L**_ _[⊤]_ = _**L**_ _[−]_ [1] . Inserting back into
the ansatz yields the refined solution


_**f**_ ( _**x**_ ) = _**L**_ _[−]_ [1] [ ˆ] _**f**_ ( _**Lx**_ + _**b**_ ) + _**d**_ _,_ (40)


and for brevity, we let _**ξ**_ = _**f**_ [ˆ] ( _**r**_ ( _**x**_ )) = _**f**_ [ˆ] ( _**Lx**_ + _**b**_ ):


_**f**_ ( _**x**_ ) = _**L**_ _[−]_ [1] _**ξ**_ + _**d**_ _._ (41)

We then insert the current solution into Eq. 20 and input _**r**_ which gives

_∥_ _**ξ**_ _−_ _**Lx**_ _[′]_ _−_ _**b**_ _∥_ [2] = ( _**L**_ _[−]_ [1] _**ξ**_ + _**d**_ _−_ _**x**_ _[′]_ ) _[⊤]_ **Λ** ( _**L**_ _[−]_ [1] _**ξ**_ + _**d**_ _−_ _**x**_ _[′]_ ) + _c_ _[′]_ ( _**x**_ ) (42)

= ( _**L**_ _[−]_ [1] _**ξ**_ + _**d**_ _−_ _**x**_ _[′]_ ) _[⊤]_ _**L**_ _[⊤]_ _**L**_ ( _**L**_ _[−]_ [1] _**ξ**_ + _**d**_ _−_ _**x**_ _[′]_ ) + _c_ _[′]_ ( _**x**_ ) (43)

= _∥_ _**ξ**_ + _**Ld**_ _−_ _**Lx**_ _[′]_ _∥_ [2] + _c_ _[′]_ ( _**x**_ ) (44)

_c_ _[′]_ ( _**x**_ ) = _∥_ _**ξ**_ _−_ _**Lx**_ _[′]_ _−_ _**b**_ _∥_ [2] _−∥_ _**ξ**_ _−_ _**Lx**_ _[′]_ + _**Ld**_ _∥_ [2] (45)


Let us denote _**v**_ = _**ξ**_ _−_ _**Lx**_ _[′]_ and note that _**v**_ and _**x**_ remain independent variables. We then get


_c_ _[′]_ ( _**x**_ ) = _∥_ _**v**_ _−_ _**b**_ _∥_ [2] _−∥_ _**v**_ + _**Ld**_ _∥_ [2] (46)

= _−_ 2 _**v**_ _[⊤]_ ( _**b**_ + _**Ld**_ ) + _∥_ _**b**_ _∥_ [2] _−∥_ _**Ld**_ _∥_ [2] (47)


Because _**v**_ and _**x**_ vary independently and the equation is true for any pair of these points, both sides
of the equation need to be independent of their respective variables. This is true only if _**b**_ = _−_ _**Ld**_ .
Hence, it follows that


_c_ _[′]_ ( _**x**_ ) = 0 and _**d**_ = _−_ _**L**_ _[−]_ [1] _**b**_ _._ (48)
Inserting _**d**_ into Eq. 40 gives the final solution,

_**f**_ ( _**x**_ ) = _**L**_ _[−]_ [1] [ ˆ] _**f**_ ( _**Lx**_ + _**b**_ ) _−_ _**L**_ _[−]_ [1] _**b**_ _._ (49)


Solving for _**f**_ [ˆ] gives us


_**f**_ ˆ( _**Lx**_ + _**b**_ ) = _**Lf**_ ( _**x**_ ) + _**b**_ (50)
_**f**_ ˆ( _**x**_ ) = _**Lf**_ ( _**L**_ _[−]_ [1] ( _**x**_ _−_ _**b**_ )) + _**b**_ = ( _**r**_ _◦_ _**f**_ _◦_ _**r**_ _[−]_ [1] )( _**x**_ ) (51)

which concludes the proof.


18


Published as a conference paper at ICLR 2025


B K ERNEL DENSITY ESTIMATE CORRECTION


Theorem 1 requires to include a “potential function” _α_ into our model. In this section, we discuss
how this function can be approximated by a kernel density estimate (KDE) in practice. The KDE
intuitively corrects for the case of non-uniform marginal distributions. Correcting with _α_ overcomes
the limitation of requiring a uniform marginal distribution discussed before (Zimmermann et al.,
2021). While other solutions have been discussed, such as training a separate MLP (Matthes et al.,
2023), the KDE solution discussed below is conceptually simpler and non-parametric.


For the models considered in the main paper, we considered representation learning in Euclidean
space, while Appendix D contains some additional experiments for the very common case of training
embeddings on the hypersphere. For both cases, we can parameterize appropriate KDEs.


For the Euclidean case, we use the KDE based on the squared Euclidean norm,



_ϵ_



_q_ ˆ( _**x**_ ) = 1
_ϵM_



_M_
�



_−_ _[∥]_ _**[x]**_ _[ −]_ _**[x]**_ _[i]_ _[∥]_ [2]

� _i_ =1 exp � _ϵ_



_,_ _**x**_ _i_ _∼_ _q_ ( _**x**_ ) _._ (52)
�



We note that in the limit _ϵ →_ 0, _M →∞_, this estimate converges to the correct distribution,
_q_ ˆ( _**x**_ ) _→_ _q_ ( _**x**_ ) . This is also the case used in Theorem 1. However, this estimate depends on the ground
truth latents _**x**_ _i_, which are not accessible during training. Hence, we need to find an expression that
depends on the observable data. We leverage the feature encoder _**h**_ to express the estimator as



_ϵ_



ˆ 1
_q_ _**h**_ ( _**y**_ ) =
_ϵM_



_M_
�



_−_ _[∥]_ _**[h]**_ [(] _**[y]**_ [)] _[ −]_ _**[h]**_ [(] _**[y]**_ _[i]_ [)] _[∥]_ [2]

� _i_ =1 exp � _ϵ_



_,_ _**y**_ _i_ _∼_ _q_ ( _**y**_ ) _._ (53)
�



We can express this estimator in terms of the final solution, _**r**_ ( _**x**_ ) = _**h**_ ( _**g**_ ( _**x**_ )) = _**Q**_ **Σ** _[−]_ _u_ [1] _[/]_ [2] _**x**_ + _**b**_ in the
theorem. If we express the solution in terms of the ground truth latents again, the orthogonal matrix
_**Q**_ vanishes and we obtain



�



_ϵ_



�



_._ _**y**_ _i_ _∼_ _q_ ( _**y**_ ) (54)



_q_ ˆ _**h**_ ( _**x**_ ) = 1
_ϵM_



_M_
� exp


_i_ =1



_M_
�



_−_ _[∥]_ **[Σ]** _u_ _[−]_ [1] _[/]_ [2] ( _**x**_ _−_ _**x**_ _i_ ) _∥_ [2]



This corresponds to a KDE using a Mahalanobis distance with covariance matrix **Σ** _u_, which is a valid
KDE of _q_ .


We can derive a similar argument when computing embeddings and dynamics on the hypersphere.
When, a von Mises-Fisher distribution is suitable to express the KDE, and we obtain



ˆ
_q_ ( _**x**_ ) = _[C]_ _[p]_ [(] _[κ]_ [)]

_M_



_M_
� exp( _κ_ _**x**_ _[⊤]_ _**x**_ _i_ ) _,_ _**x**_ _i_ _∼_ _q_ ( _**x**_ ) (55)


_i_ =1



where _C_ _p_ ( _κ_ ) is the normalization constant of the von Mises-Fisher distribution. This again approaches
the correct data distribution for ˆ _q →_ _q_ as _M, κ →∞_ . Following the same arguments above, but


κ p = 16 κ p = 64 κ p = 128








|Col1|Col2|Col3|
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


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
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
||||||
||||||
||||||
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
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
||||||
||||||
||||||
||||||
||||||
||||||
||||||



Figure 8: Introducing KDE into the loss allows to compensate for non-uniform marginal distribution. We
show performance in terms of _R_ [2] across datasets with increasingly non-uniform marginal. We replicate the
data-generating process and experimental setup performed by Zimmermann et al. (2021, Figure 2).


19


Published as a conference paper at ICLR 2025


using _**r**_ ( _**x**_ ) = _**h**_ ( _**g**_ ( _**x**_ )) = _**Qx**_ as the indeterminacy on the hypersphere, we can express this in terms
of the ground truth latents,



ˆ
_q_ _**h**_ ( _**x**_ ) = _[C]_ _[p]_ [(] _[κ]_ [)]

_N_


= _[C]_ _[p]_ [(] _[κ]_ [)]

_N_



_M_
� exp � _κ_ _**r**_ ( _**x**_ ) _[⊤]_ _**r**_ ( _**x**_ _i_ )� _,_ (56)


_i_ =1


_M_
� exp � _κ_ _**x**_ _[⊤]_ _**x**_ _i_ � _,_ (57)


_i_ =1



which is again a valid KDE.


It is interesting to consider the effect of the KDE on the loss function. Inserting _ψ ←_ _ψ −_ log ˆ _q_ into
the loss function yields



_−_ log _p_ _ψ_ ( _**x**_ _|_ _**x**_ [+] _, N_ ) = _−_ ( _ψ_ ( _**x**_ _i_ _,_ _**x**_ [+] _i_ [)] _[ −]_ [log ˆ] _[q]_ [(] _**[x]**_ [+] _i_ [)) + log]



_N_
� _e_ _[ψ]_ [(] _**[x]**_ _[i]_ _[,]_ _**[x]**_ _j_ _[−]_ [)] _[−]_ [log ˆ] _[q]_ [(] _**[x]**_ _[−]_ _j_ [)] _,_ (58)


_i_ =1



= _−ψ_ ( _**x**_ _i_ _,_ _**x**_ [+] _i_ [) + log ˆ] _[q]_ [(] _**[x]**_ [+] _i_ [) + log]



_N_
�


_i_ =1



1
_q_ ˆ( _**x**_ _[−]_ _j_ [)] _[e]_ _[ψ]_ [(] _**[x]**_ _[i]_ _[,]_ _**[x]**_ _j_ _[−]_ [)] _,_ (59)



= _−ψ_ ( _**x**_ _i_ _,_ _**x**_ [+] _i_ [) + log]


= _−ψ_ ( _**x**_ _i_ _,_ _**x**_ [+] _i_ [) + log]



_N_
�


_i_ =1



_N_
� _w_ _h_ ( _**x**_ [+] _i_ _[,]_ _**[ x]**_ _[−]_ _j_ [)] _[e]_ _[ψ]_ [(] _**[x]**_ _[i]_ _[,]_ _**[x]**_ _j_ _[−]_ [)] (61)


_i_ =1



_qq_ ˆˆ _hh_ (( _**xx**_ _[−]_ _j_ [+] _i_ [)][)] _[e]_ _[ψ]_ [(] _**[x]**_ _[i]_ _[,]_ _**[x]**_ _j_ _[−]_ [)] _,_ (60)



with the importance weights _w_ _h_ ( _**x**_ [+] _i_ _[,]_ _**[ x]**_ _[−]_ _j_ [) =] _qq_ ˆˆ _hh_ (( _**xx**_ _[−]_ _j_ [+] _i_ [)][)] [. Intuitively, this means that the negative]

examples are re-weighted according to the density ratio between the current positive and each
negative sample.


**Empirical motivation.** Figure 8 shows preliminary results on applying this KDE correction to
contrastive learning models. We followed the setting from Zimmermann et al. (2021) and re-produced
the experiment reported in Fig. 2 in their paper. We use 3D latents, a 4-layer MLP as non-linear
mixing function with a final projection layer to 50D observed data. The reference, positive and
negative distributions are all vMFs parameterized according to _κ_ (x-axis) in the case of the reference
and negative distribution and _κ_ _p_ for the positive distribution.


The grey curve shows the decline in empirical identifiability ( _R_ [2] ) as the uniformity assumption is
violated by an increasing concentration _κ_ (x-axis). Applying a KDE correction to the data resulted in
substantially improved performance (red lines).


However, when testing the method directly on the dynamical systems considered in the paper, we did
not found a substantial improvement in performance. One hypothesis for this is that the distribution
of points on the data manifold (not necessarily the whole R _[d]_ is already sufficiently uniform. Hence,
while the theory requires inclusion of the KDE term (and it did not degrade results), we suggest to
drop this computationally expensive term when applying the method on real-world datasets that are
approximately uniform.


20


Published as a conference paper at ICLR 2025


C A DDITIONAL R ELATED W ORK


**Contrastive learning.** An influential and conceptual motivation for our work is Contrastive Predictive
Coding (CPC) (Oord et al., 2018) which uses the InfoNCE loss with an additional non-linear
projection head implemented as an RNN to aggregate information from multiple time steps. Then, an
affine projection is used for multiple forward prediction steps. However, contrary to our approach, the
“dynamics model” is not explicitly parameterized, limiting its interpretability. Similar frameworks
have been successfully applied across various domains, including audio, vision, and language, giving
rise to applications such as wav2vec (Schneider et al., 2019), time contrastive networks for video
(TCN; Sermanet et al., 2018) or CPCv2 (Henaff, 2020). An interesting parallel to DLC+LDS in
image representation learning is AnInfoNCE (Rusak et al., 2024) which extends a SimCLR-like
(Chen et al., 2020b) training setup by an additional learnable diagonal matrix in its similarity function.


**Non-Contrastive learning.** Models such as data2vec (Baevski et al., 2022) and JEPA (Assran et al.,
2023) learns a representation by trying to predict missing information in latent space, using an MSE
loss. JEPA uses asymmetric encoders, and on top a predictor model in latent space parameterized by
a neural net. However, these approaches do not provide any identifiability guarantees.


**System identification.** In system identification, a problem closely related to the one addressed in this
work is known as ”nonlinear system identification. Widely used algorithms for this problem include
Extended Kalman Filter (EKM) (McGee & Schmidt, 1985) and Nonlinear Autoregressive Moving
Average with Exogenous inputs (NARMAX) (Chen & Billings, 1989). EKF is based on linearizing
_**g**_ and _**f**_ using a first-order Taylor-series approximation and then apply the Kalman Filter (KF) to
the linearized functions. NARMAX, on the other hand, typically employs a power-form polynomial
representation to model the non-linearities. In neuroscience, practical (generative algorithms) include
systems modeling linear dynamics (fLDS; Gao et al., 2016) or non-linear dynamics modelled by
RNNs (LFADS; Pandarinath et al., 2018). Hurwitz et al. (2021) provide a detailed summary of
additional algorithms.


**Nonlinear ICA.** The field of Nonlinear ICA has recently provided identifiability results for identifying
latent variables, usually employing auxiliary variables such as class labels or time information
(Hyvarinen & Morioka, 2016; 2017; Hyvarinen et al., 2019; Khemakhem et al., 2020; Sorrenson
et al., 2020). In the case of time series data, Time Contrastive Learning (TCL) (Hyvarinen & Morioka,
2016) uses a contrastive loss to predict the segment-ID of multivariate time-series which was shown
to perform Non-linear ICA. Permutation Contrastive Learning (PCL) (Hyvarinen & Morioka, 2017)
permutes the time series and aims to distinguish positive and negative pairs.


**Temporal causal representation learning.** In Nonlinear ICA, the factors are assumed to be _in-_
_dependent_, subject to some indeterminacy in the original latent variables. However, this approach
encounters challenges when the latent variables have time-delayed causal relationships. Approaches
like LEAP (Sorrenson et al., 2020) and TDLR (Yao et al., 2021) address these challenges in both
stationary and non-stationary environments, even when the transition function’s parametric form
is unknown. CaRING (Yao et al., 2022) extends these results to cases where the mixing function
is non-invertible. Lastly, CITRIS (Lippe et al., 2022) introduces intervention target information to
enhance the identification of latent causal factors. In this work, we do not aim to estimate the temporal
causal graph. Instead, we focus on estimating the dynamics model _f_ using an interpretable and
explicitly parameterized dynamics model (e.g. _∇_ -SLDS) which can later be analyzed for applications
such as scientific discovery.


**Switching Linear Dynamical Systems.** Several papers propose methods to infer SLDSs (Ackerson
& Fu, 1970; Chang & Athans, 1978; Ghahramani & Hinton, 2000), leading to a variety of extensions
and variants. For example, Recurrent SLDSs (Linderman et al., 2017; Dai et al., 2022) address statedependent switching by changing the switch transition distribution to _p_ ( _y_ _t_ _|y_ _t−_ 1 _, x_ _t−_ 1 ), allowing for
more flexible dependencies on previous states. Another extension, Explicit duration SLDS introduces
additional latent variables to model the distribution of switch durations explicitly (Chiappa et al.,
2014). Some approaches relax the assumption of linear dynamics, such as in the case of SNLDS
and RSSSM, where the dynamics model is assumed to be nonlinear (Dong et al., 2020; Chow &
Zhang, 2013). In the context of Nonlinear Independent Component Analysis (ICA), recent extensions
include structured data generating processes (e.g., SNICA; Halv ¨ a et al., 2021) which were shown to ¨
be useful for the inference of switching dynamics. In this vein, Balsells-Rodas et al. (2023) proposed
additional identifiability theory for the switching case. Other approaches, based on Neural Ordinary
Differential Equations (Neural ODEs; Chen et al., 2020a; Shi & Morris, 2021), or methods aimed at


21


Published as a conference paper at ICLR 2025


discovering switching dynamics within recurrent neural networks (Smith et al., 2021), also present
interesting avenues for modeling switching dynamics.


**Deep state-space models.** Recently, (deep) state-space models (SSMs) such as S4 or Mamba (Gu
et al., 2021; Gu & Dao, 2023) have emerged as a promising architecture. These models are particularly
well-suited for capturing long-range dependencies, making them an attractive choice for sequence
modeling tasks.


**Symbolic Regression.** An alternative approach to modeling dynamical systems is the use of symbolic
regression, which aims to directly infer explicit symbolic mathematical expressions governing the
underlying dynamical laws. Examples include Sparse Identification of Nonlinear Dynamics (SINdy;
Brunton et al., 2016), as well as more recent transformer-based models (Kamienny et al., 2022;
d’Ascoli et al., 2023), which have demonstrated promise in discovering interpretable representations
of dynamical systems.


22


Published as a conference paper at ICLR 2025


D V ON M ISES –F ISHER ( V MF) CONDITIONAL DISTRIBUTIONS


In the main paper, we have shown experimental results that verify Theorem 1 in the case of Normal
distributed positive conditional distribution and using the Euclidean distance. This approach has
allowed for modeling latents and their dynamics in Euclidean space, which we argue is the most practical setting to apply DCL in. However, self-supervised learning methods and especially contrastive
learning have commonly been applied to produce representations on the hypersphere and using the
dot-product distance (Oord et al., 2018; Schneider et al., 2023; Wang & Isola, 2020; Zimmermann
et al., 2021; Chen et al., 2020b).


Here we validate empirically that Theorem 1 equally holds under the assumption of vMF conditional
distributions and using the dot-product distance _ϕ_ ( _**x**_ _,_ _**y**_ ) = _**x**_ _[⊤]_ _**y**_ as part of the loss. We run experiments as in Table 1 for the case where the true dynamics model _**f**_ is a linear dynamical system.
Additionally, we vary the setting similar to Figure 4 to show increasing ∆ _t_ (angular velocity).


We compare:


    - **DCL (ours)** – with linear dynamics: _**f**_ [ˆ] ( _**x**_ ) = _**Ax**_ [ˆ] .

    - **GTD** – the ground-truth dynamics model (LDS) _**f**_ [ˆ] ( _**x**_ ) = _**Ax**_ .

    - **No dynamics** – the baseline setting we use throughout the paper _**f**_ [ˆ] ( _**x**_ ) = _**x**_ .


    - **Asymmetric** – a variation on the baseline setting that uses asymmetric encoders (one for
reference, one for positive or negative) which would be a possible fix of Corollary 1. We can
obtain this setting by skipping the explicit dynamics modeling, and defining two encoders
_**h**_ 1 _,_ _**h**_ 2 which relate as follows: _**h**_ _◦_ _**f**_ := _**h**_ 1, _**h**_ := _**h**_ 2 .







Figure 9: Our findings from Table 1 hold equally under vMF noise distribution when using LDS ground truth
dynamics. We show empirical identifiability of the latents in terms of _R_ [2] under varying a) angles of the rotation
dynamics i.e. angular velocity ∆ _t_ (x-axis) and b) the magnitude of the dynamics noise _σ_ (panels, left: low noise,
right: high noise)


Similar to our results for the Euclidean case (Table 1), in Figure 9 we show results that experimentally
verify Theorem 1 for latent dynamics on hypersphere and using vMF as conditional distribution. Both
for low (left panel) and high (right panel) variance of the conditional distribution we can see that
DCL effectively identifies the ground truth latents on par with the oracle (GTD) model performance.
On the other hand, the baseline, standard time contrastive learning without dynamics, can not identify
the ground truth latents with underlying linear dynamics as predicted by Corollary 1. This prediction
is only violated in the case where the variance of the noise distribution is high enough, such that the
noise dominates the changes introduced by the actual dynamics. This is the case for dynamics with
rotations up to 4 degrees for _σ_ = 0 _._ 01 and angles up to 10 degrees for _σ_ = 0 _._ 1.


23


Published as a conference paper at ICLR 2025


E A DDITIONAL PLOTS FOR SLDS


σ = 0.0001 σ = 0.01


Figure 10: Visualizations of 6D linear dynamical systems at _σ_ = 0 _._ 0001 (left) and _σ_ = 0 _._ 01 for 10 degree
rotations. These systems are used in our SLDS experiments.


24


Published as a conference paper at ICLR 2025


F G ENERALIZATION – T RAIN - VS . T EST S ET


In the main paper, all metrics are evaluated using the full training dataset of the respective experiment.
We argue that this is sufficient for showing the efficacy of our model and verifying claims from the
theory in section 3 because a) in self-supervised learning, the model learns generalizable representations through pretext tasks, making overfitting less of a concern; b) the metrics we are interested in
are about uncovering the true underlying latent representation and dynamics of the available training
data, not of new data; and c) most importantly, we ensured that the training dataset is large enough to
approximate the full data distribution.


Nonetheless, here we show a series of control experiments to re-evaluate models on new and unseen
data. We do so by following the same data generating process of the given experiment (same dynamics
model and mixing function) and sample completely new trials (10% of the number of trials of the
training dataset). Every new trial starts at a random new starting point, with a randomly sampled new
mode sequence and regenerated with different seeds for the dynamics noise.


First, we re-evaluated every experiment of Figure 6 on the test dataset generated as described above
and show these results in Figure 11. Comparing those results to the train dataset version of Figure 6
shows that there is almost no difference in the performance (with regard to identifiability and systems
identification). The difference are so small, that visually comparing the results almost becomes
impractical, so we additionally provide the exact numbers of the first panel (variations on the number
of samples per trial) in Table 2.


Finally, to qualitatively and quantitatively show the difference between the train and test datasets
we provide a) depictions of the ground truth latents of 5 random test trials and their closest possible
matching trial from the training set in Figure 12 and b) a distribution of the distances (in terms of _R_ [2]
between the data from the test and train trials) between all test trials of one of a random test set and
their closest trial from the training dataset in Figure 13.


Figure 11: Same as Figure 6 (Variations and ablations for SLDS), but re-evaluated on a newly generated test
data with different starting points.


Table 2: A detailed view on the #samples panel from Figure 6 and 11 showing the difference between train and
test set evaluation.


DCL (ours) CL w/o dynamics CL w/ ground truth dynamics
_R_ [2] (train) _R_ [2] (test) _R_ [2] (train) _R_ [2] (test) _R_ [2] (train) _R_ [2] (test)
# samples


10 0.991 _±_ 0.00137 0.989 _±_ 0.00172 0.923 _±_ 0.03703 0.917 _±_ 0.04000 0.954 _±_ 0.00397 0.952 _±_ 0.00442

100 0.995 _±_ 0.00106 0.994 _±_ 0.00108 0.786 _±_ 0.06794 0.791 _±_ 0.06931 0.990 _±_ 0.00107 0.990 _±_ 0.00099

1000 0.995 _±_ 0.00074 0.995 _±_ 0.00078 0.765 _±_ 0.06671 0.770 _±_ 0.06973 0.990 _±_ 0.00507 0.990 _±_ 0.00511

10000 0.996 _±_ 0.00046 0.996 _±_ 0.00044 0.694 _±_ 0.06937 0.801 _±_ 0.10568 0.991 _±_ 0.00509 0.991 _±_ 0.00425


25


Published as a conference paper at ICLR 2025


Figure 12: Ground truth latents from five random trials of the testsets for Figure 11 and their closest match
within the corresponding trainset. The closest match is evaluated by computing the _R_ [2] -Score between a given
trial from the testset and every possible trial.


Figure 13: Histogram of all _R_ [2] -Scores between every trial from the testset and its closest possible match from
the trainset as shown in Figure 12.


26


Published as a conference paper at ICLR 2025


G V ARIATIONS AND ADDITIONAL BASELINES FOR THE D YN _R_ [2] M ETRIC


G.1 M ETHOD


As an addition to Table 1, we analyse the Dyn _R_ [2] in more detail. In Table 3 we show variants for the
metric. Firstly, we modify the number of forward prediction steps,



_**f**_ _[n]_ ( _**x**_ ) := ( _**f**_ _◦· · · ◦_ _**f**_
~~�~~ ~~�~~ � ~~�~~

_n_ times



)( _**x**_ ) (62)



and respectively for _**f**_ [ˆ] _[n]_ in relation to _**f**_ [ˆ] . We then consider two variants of Eq. 12. Firstly, we perform
multiple forward predictions ( _n >_ 1) and compare the resulting embeddings:


r2 ~~s~~ core( _**f**_ [ˆ] _[n]_ (ˆ _**x**_ ) _,_ _**Lf**_ _[n]_ ( _**L**_ _[′]_ _**x**_ ˆ + _**b**_ _[′]_ ) + _**b**_ ) _._ (63)


A rationale for this metric is that the prediction task becomes increasingly difficult with an increasing
number of time steps, and errors accumulate faster.


Secondly, as an additional control, we replace _**f**_ [ˆ] with the identity, and compute


r2 ~~s~~ core(ˆ _**x**_ _,_ _**Lf**_ _[n]_ ( _**L**_ _[′]_ _**x**_ ˆ + _**b**_ _[′]_ ) + _**b**_ ) _._ (64)


This metric can be regarded as a naiive baseline/control for comparing performance of the dynamics
model. If the dyn _R_ [2] is not significantly larger than this value, we cannot conclude to have obtained
meaningful dynamics.


For the lower part of Table 1, we report the resulting metrics in Table 3, setting the number of forward
steps _n_ to 1 or 10, and using either the original metric (Eq. 62), or the control (Eq. 63).


G.2 R ESULTS


For the SLDS system, we can corroborate our results further: the baseline model obtains a dyn _R_ [2]
of around 85% for single step prediction, both for the original and control metric. Our _∇_ -SLDS
model and the ground truth dynamical model obtain over 99.9% well above the level of the control
metric which remains at around 95%. The high value of the control metric is due to the small change
introduced by a single time step, and should be considered when using and interpreting the metric. If
more steps are performed, the performance of the _∇_ -SLDS model drops to about 95.5% vs. chance
level for the control metric, again highlighting the high performance of our model, but also the room
for improvement, as the oracle model stays at above 99% as expected.


For the Lorenz system, we do not see a substantial difference between original dyn _R_ [2] metric and
dyn _R_ [2] control for any of the considered algorithms. Yet, as noted in the main paper, _∇_ -SLDS is the
only dynamics model that gets a high _R_ [2] of 94.08%, vs. the lower 81.20% for a single LDS model, or
40.99% for the baseline model. In other words, while DCL with the _∇_ -SLDS dynamics model falls
short of identifying the true underlying dynamics for this non-linear chaotic system, without DCL we
wouldn’t even identify the latents. We leave optimizing the parameterization of the dynamics model
to identify non-linear chaotic systems for future work.


Table 3: Extended metrics for dynamics models including additional variations on the _dynR_ [2] metric where
_Control_ is replacing _**f**_ [ˆ] with the identity and _10 Steps_ is applying _**f**_ [ˆ] (and _**f**_ ) 10 times, i.e., predicting 10 steps
forward instead of only one step as is done in the _Original_ version.

|f p(ε) fˆ identifiable|Original Control Original Control|
|---|---|
|SLDS<br>Normal<br>identity<br>✗<br>SLDS<br>Normal<br>_∇_-SLDS<br>(✓)<br>SLDS<br>Normal<br>GT<br>(✓)|85.47_ ±_ 8.07<br>84.54_ ±_ 7.31<br>2.78_ ±_ 9.34<br>-58.62_ ±_ 7.22<br>99.93_ ±_ 0.01<br>95.15_ ±_ 0.68<br>95.53_ ±_ 0.47<br>-124.65_ ±_ 6.97<br>99.97_ ±_ 0.00<br>94.94_ ±_ 0.68<br>99.36_ ±_ 0.18<br>-129.32_ ±_ 6.95|
|Lorenz<br>Normal<br>identity<br>✗<br>Lorenz<br>Normal<br>LDS<br>✗<br>Lorenz<br>Normal<br>_∇_-SLDS<br>(✓)|27.02_ ±_ 8.72<br>27.17_ ±_ 8.74<br>22.87_ ±_ 7.13<br>24.85_ ±_ 7.14<br>80.30_ ±_ 14.13<br>82.98_ ±_ 12.64<br>-13.07_ ±_ 41.03<br>42.08_ ±_ 26.14<br>93.91_ ±_ 5.32<br>93.70_ ±_ 5.11<br>34.48_ ±_ 6.47<br>55.75_ ±_ 6.01|



27


Published as a conference paper at ICLR 2025


H N ON -I NJECTIVE M IXING F UNCTIONS


Our identifiability guarantees (Theorem 1, Def. 5) require an injective mixing function _**g**_ ( _**x**_ _t_ ) = _**y**_ _t_ .
On the first glance, this clashes with common requirements in system identification under _partial_
_observability_ . Specifically, let us consider a system of the form:


_**x**_ _t_ +1 = _**f**_ ( _**x**_ _t_ ) + _**ε**_ _t_ = _**Ax**_ _t_ + _**ε**_ _t_
(65)
_**y**_ _t_ = _**Cx**_ _t_ = _**g**_ ( _**x**_ _t_ ) _,_


where _**x**_ _t_ _∈_ R _[n]_, _**y**_ _t_ _∈_ R _[m]_, _**C**_ _∈_ R _[m][×][n]_ with _n > m_ is a non-square matrix that projects the
states of the dynamical system into a lower dimensional space. Similarly, we may have _n ≤_ _m_
with rank( _**C**_ ) _< n_ . In those cases, _**C**_ and hence _**g**_ is non-invertible, and naively, the injectivity
assumptions of Def. 5 would fail to hold.


However, in practice this issue can be tackled through the use of a time-delay embedding. Specifically,
we consider the following reformulation of the system:






=








_**C**_

 _**CA**_

 _**CA**_ [2]

...

 _**CA**_ _[τ]_










_**y**_ ˜ _t_ _[i]_ [:=]









_**y**_ _t−τ_

_**y**_ _t−τ_ +1

_**y**_ _t−τ_ +2

...

_**y**_ _t_



_**x**_ _t−τ_ +



**0**

 _**ν**_ _t_ [1]

_**ν**_ _t_ [2]

...

 _**ν**_ _t_ _[τ]_






(66)




~~�~~ � ~~�~~ �
_**O**_



where _**ν**_ _t_ _[τ]_ [:=] _**[ C]**_



_τ_ _−_ 1
� _**A**_ _[i]_ _**ε**_ _t−i_ _._ (67)


_i_ =0



For a sufficiently large time lag _τ_, the linear map ˜ _**g**_ ( _**x**_ _t−τ_ ) = _**Ox**_ _t−τ_ = ˜ _**y**_ _t_ _[i]_ [will become injective]
again. This is the case if _**O**_ is full rank which holds if _**A**_ is full rank, since _**f**_ is bijective and therefore
_**A**_ is a full rank square matrix. For example, if _**C**_ had rank _m_, then using at least _τ_ = _m_ _[n]_ [time steps]

would make ˜ _**g**_ injective and our theoretical guarantees from Theorem 1 would hold, up to the offset
introduced by the noise _**ν**_ .


In practice, the change in latent space between different time steps might be small (especially when
the time resolution of the system is very high). A practical way to avoid feeding increasingly large
inputs, is to not feed in all time-lags 0 _. . . τ_ into the construction of _**O**_, but to subselect _k_ time lags
_τ_ 1 _, . . ., τ_ _k_, with _τ_ 1 = 0 and _τ_ _k_ = _τ_, and instead consider the system










_**C**_

_**CA**_ _[τ]_ [2]

...

_**CA**_ _[τ]_ _[k][−]_ [1]

_**CA**_ _[τ]_

















=








_**y**_ ˜ _t_ _[i]_ [:=]









_**y**_ _t−τ_

_**y**_ _t−τ_ + _τ_ 2

...

_**y**_ _t−τ_ + _τ_ _k−_ 1

_**y**_ _t−τ_ _n_



_**x**_ _t−τ_ 1 +



**0**

 _**ν**_ _t_ [1]

_**ν**_ _t_ [2]

...

 _**ν**_ _t_ _[τ]_






(68)




~~�~~ � ~~�~~ �
_**O**_


This system allows to have a sufficiently large context window (from _t −_ _i_ 1 to _t_ ) to ensure injectivity,
while keeping the input dimensionality of the model fixed. Note, when we set _τ_ _i_ = _i −_ 1 for each _i_,
we recover Eq. 66.


Regarding the noise vector _**ν**_, Halv ¨ a et al. (2021) recently showed that noisy and noise-free demixing ¨
problems can be mapped onto each other. While a full rigorous proof in conjunction with our
Theorem 1 is beyond the scope of this work, invoking Theorem 1 of Halv ¨ a et al. (2021) to account ¨
for _**ν**_ is a promising avenue. Importantly, our empirical validation later on already shows that the
model is in practice indeed functioning even in the presence of the noise distribution.


H.1 E XPERIMENTAL V ALIDATION


We validate our theoretical considerations by two sets of experiments as closely related to the LDS
setting from Table 1 as possible. We use two types of mixing functions:


_**g**_ ( _**x**_ ) = _**C**_ 1 _**C**_ 2 _**x**_ _t_ (69)

_**g**_ ( _**x**_ ) = _**C**_ 2 _**g**_ _[′]_ ( _**C**_ 1 _**x**_ _t_ ) (70)


28


Published as a conference paper at ICLR 2025


a b linear c



non-linear





(𝜏) (𝜏)









linear non-linear



d e



f


Time steps (k) for fixed 𝜏=100 Time steps (k) for fixed 𝜏=100





Figure 14: Non-injective mixing functions can be successfully handled by a time-lag embedding. **a**, in the first
setting, we pass observations from _τ_ consecutive time steps into our feature encoder. **b**, empirical identifiability
of the latent space ( _R_ [2] ) for baseline (no dynamics) vs. DCL (linear dynamics) as we increase _n_ for a linear
and **c** non-linear mixing function. **d**, to achieve injective mixing functions through time-lag embeddings, we
here include the full 100-step length window, but only pass _k_ equidistantly spaced points within this window of
length _τ_ . **e**, empirical identifiability for baseline (no dynamics) vs. DCL (linear dynamics) as we increase the
number of points in the context for a fixed _τ_ = 100-step window and **f** for nonlinear mixing.


where _**C**_ 1 _∈_ R _[m][×][r]_, _**C**_ 2 _∈_ R _[r][×][n]_ are randomly sampled, _**g**_ : R _[r]_ _→_ R _[r]_ is a random injective and
nonlinear function and _m_ is the observable dimension, _n_ is the latent dimension and _r_ is the matrix
rank. In line with the LDS experiments from Table 1, we set _n_ = 6 and _m_ = 50 . The mixing
functions from Eq. 69 & 70 are then applied to the latents, including the parameter _r ∈{_ 1 _,_ 2 _,_ 3 _}_ to
restrict the rank and thereby dimensionality of the mixing functions to _r_ .


We run experiments for different numbers of time steps that get passed to the encoder _**h**_ . In the
first setting, we use _k_ consecutive time steps. In this setting, we expect that _more_ time steps than
the theoretical minimum _n/m_ are required, because the variation between consecutive time steps
is limited. To corroborate this hypothesis, we run a second variant where the length of the context
window is fixed, and _k_ equidistantly placed points are selected to be fed in the encoder.


As in Table 1, we repeat each experiment 9 times with different dataset and model seeds and report
the 95% confidence interval.


H.2 R ESULTS


Generally, we can see from Figure 14 that we can successfully verify our theoretical considerations
about weakening the injectivity constraint.


To begin with, we confirm that the default parametrization of DCL with only a single time step _i_ = 1
cannot solve the demixing problem. Next, for the theoretical minimum of time steps ( _i_ = 6 for rank
1, _i_ = 3 for rank 2, _i_ = 2 for rank 3) we can already observe a considerable improvement of the
empirical identifiability for DCL in terms of the _R_ [2] Score: 16% _→_ 66% for rank 1, 30% _→_ 78%
for rank 2, 43% _→_ 88% for rank 3 (Fig. 14b). As mentioned above, the minimal amount of time
steps required for the identifiability guarantees only hold for _**ν**_ _t_ = 0 which is not the case in these
experiments. As we increase the number of time points to 100, we can see that DCL approaches
close to perfect _R_ [2] Scores: For linear mixing, we get the best results for _i_ = 100 with 99% _R_ [2] for


29


Published as a conference paper at ICLR 2025


ranks 1, 2, and 3. While for nonlinear mixing, averaged across seeds, we get up to 86% for rank 1
and _i_ = 100, 97% for rank 2 and _i_ = 20, and 94% for _i_ = 20 . In contrast, the baseline without a
dynamics model, does not benefit at all from the additional time steps.


To further test our intuition about recovering injectivity, we include an experiment where the full
100-step length window is used, but only _n_ equidistantly spaced time steps are passed. In this setup,
we much quicker recover acceptable identifability scores for the linear (Fig. 14e) and non-linear
mixing settings (Fig. 14f).


30


Published as a conference paper at ICLR 2025


I D YNAMICAL S YSTEMS WITH C ONTROL S IGNAL


We have initially introduced the problem formulation of this paper (Equation 1) as identifying the
latent variables _**x**_ _t_ and the governing latent dynamics _**f**_ from the observations _**y**_ _t_ for:


_**x**_ _t_ +1 = _**f**_ ( _**x**_ _t_ ) + _**Bu**_ _t_ + _**ε**_ _t_
(71)
_**y**_ _t_ = _**g**_ ( _**x**_ _t_ ) + _**ν**_ _t_ _._


with control signal _**u**_, its control or actuator matrix _**B**_, system noise _**ε**_ and observation noise _**ν**_ .


However, so far we have only explicitly considered autonomous latent dynamics (see Def. 5), which
by definition did not include a control input _**u**_ _t_ . In this section, we show that the DCL framework
works equally well in the presence of a control input and verify this empirically.


Being able to include a control signal is crucial for many applications and common practice in
systems identification literature. Without adding _**u**_ _t_ to the model, the task of identifying dynamics
gets substantially harder as the effects of the control signal become entangled with the intrinsic
dynamics of the system. Additionally, only identifying the combined dynamics without factoring
out the effect of _**u**_ _t_ would make the framework less useful as it would not allow predictions in the
presence of new/different control inputs.


I.1 E MPIRICAL V ERIFICATION


We extend our existing experiments for linear dynamical systems (LDS) by including a control signal
_**u**_ _t_ in the data generating process:


_**x**_ _t_ +1 = _**Ax**_ _t_ + _**Bu**_ _t_ + _**ε**_ _t_ (72)


We train and evaluate four model variants:


    - **Baseline** : Identity dynamics model, using the control input with with a trainable _**B**_ . The
dynamics model is fit post-hoc,


    - **DCL** : with a LDS dynamics model where _**B**_ = 0,


    - **DCL w/ control:** with a LDS dynamics model and trainable _**B**_ .


We choose the control signal _**u**_ _t_ to be generated from:


    - **Step function** : A composition of a negative and positive step function, starting at random
time steps and random magnitudes.


    - **Linear Dynamical System** : _**u**_ _t_ +1 = _**A**_ _u_ _**u**_ _t_ + _**ε**_ _t_, similar to the LDS system used before for
latent dynamics.


I.2 E XPERIMENT D ETAILS


We generate three datasets with linear dynamics using a) no control, b) control following another
LDS, and c) control following a step function. Each dataset consists of 1000 trials, each trial is
1000 time steps long. The latent linear dynamics have intrinsic rotational dynamics with rotation
angles _θ_ _i_ _∼_ Uniform[0 _,_ 10] and control matrix _**B**_ _∼N_ ( _µ_ = 0 _._ 01 _,_ Σ = _**I**_ _·_ 0 _._ 01) . The system
noise _**ε**_ follows a standard normal with _σ_ _**ε**_ = 0 _._ 001 . The mixing function _**g**_ is the same nonlinear
mixing function with 4 layers as for Table 1. We use 5-dimensional latents and have 50-dimensional
observations. For the control following another LDS, we use the same sampling strategy for the
parameters as for the latent dynamics. For the control following a step function, we pick two points
_T_ 1, _T_ 2 such that the _T_ 2 _−_ _T_ 1 = 200 and that the step is centered within the trial including some
random offset. We use different controls _u_ _t_ for every trial. We extend DCL with a parametrization
of the linear dynamics model that follows Eq. 72 and includes the control _u_ _t_ and trainable control
matrix _**B**_ . We use the same dynamics model for the baseline. However there, this only affects the
post hoc dynamics fitting. We also train DCL with a LDS model that does not include the control
input. We train every model for 20k steps and besides that, we use the same hyperparameters as for
training the LDS models in Table 1.


31


Published as a conference paper at ICLR 2025


Table 4: Empirical identifiability results when including both system noise and a deterministic control signal _**u**_ .
The ground truth dynamics _**f**_ are chosen as an LDS, and the intrinsic dynamics model _**f**_ [ˆ] is either an LDS, or
the identity (baseline); _**Bu**_ indicates whether the dynamics model includes the control or not; _p_ _**ε**_ ( _**ε**_ ) is Normal
(small _σ_ ). Mean _±_ std. are across 3 datasets and 3 experiment repeats.


Data Model Results

ˆ ˆ
Control ( _**u**_ ) _**f**_ _**Bu**_ % _R_ [2] _↑_ %dyn _R_ [2] _↑_


Step (1D) identity ✓ 91.80 _±_ 1.15 87.27 _±_ 10.6
LDS ✗ 98.36 _±_ 0.66 99.16 _±_ 1.00
LDS ✓ 98.70 _±_ 0.44 99.53 _±_ 0.27


LDS (5D) identity ✓ 74.17 _±_ 13.8 76.47 _±_ 7.51
LDS ✗ 98.26 _±_ 0.17 99.87 _±_ 0.04
LDS ✓ 98.10 _±_ 0.35 99.85 _±_ 0.08


Figure 15: Visualization of dynamics inference in the presence of a control signal _**u**_ . **a** LDS dynamics are
complemented by a **b** 1D step function signal, which **c** is projected to 5D using a random matrix _**B**_ (not shown).
The ground truth dynamics then show signatures of the autonomous dynamics when _**u**_ = 0 (gray box), and
move through latent space as we apply the step function. **d**, DCL uses a dynamics model and the control input
_**u**_ ; **e**, the baseline uses only the control input _**u**_ . The three rows show different dynamical systems, each plot is
one trial with 1000 steps.


I.3 R ESULTS


As shown in Table 4, when applying a step function for the control signal (Step 1D), DCL is able
to identify both the latent space (98.7% _R_ [2] ) and the dynamics (99.5% dyn _R_ [2] ). This result holds
even when the control signal _**u**_ is not used for training the model. However, a baseline with identity
dynamics is not able to identify the dynamical system with a substantially lower dyn _R_ [2] of 87.3%.
Nevertheless, the latent space can be estimated reasonably well, although not perfect (91.8 % _R_ [2] ),
most likely because the availability of _**u**_ converts the de-mixing problem into a supervised learning
problem (we have access to ( _**u**_ _,_ _**y**_ ) pairs). This is reflected in the visualization in Figure 15: While
the baseline (identity dynamics, but using _**u**_ ) reasonably estimates the direction of _**u**_ provided during
training, the local dynamics cannot be fitted. We highlighted a gray box denoting the phase where
_**u**_ = 0 to facilitate easier comparison.


To test DCL for more complex control signals, we also apply a full 5D LDS as the control signal _**u**_
(Table 4, LDS 5D). Both latent space estimation and dynamics estimation performs on par with the
step setting ( _>_ 98%), while the baseline fails to estimate either the latent space or the dynamics with
_R_ [2] values below 80%.


32


Published as a conference paper at ICLR 2025



no smoothing



3s window



















LDS



LDS













t=0



total time (s) 270s



LDS



b LDSi


k=5 points i ii iii


identity identity


Figure 16: Application to real-world neural data. (a) We leverage the Allen visual coding dataset using calcium
imaging data (de Vries et al., 2020). We consider the subset of the data which is comprised of 10 movie repeats
with 900 samples each at 30Hz. We hold out the last trial for validation. Mouse icon from scidraw.io (Ann
Kennedy), panel adopted from Schneider et al. (2023). (b) We feed multiple time steps to the model as outlined in
Appendix H. All experiments concat five equidistant points with a maximum time lag of at the sample locations
( _t, t −_ 5 _, . . ., t −_ 20) . (c) Quantitative comparison of DCL with identity vs. linear dynamics and qualitative
overview of embeddings computed for the hyperparameters that (i) maximize consistency for LDS dynamics,
(ii) have high consistency for both models, (iii) maximize consistency for identity dynamics. Train embedding is
depicted as points (unsmoothed for LDS, smoothed for identity to show structure), test embedding as smoothed
black trajectory. (d) Effect of smoothing applied to the embedding, left column depicts unsmoothed, right
column smoothed embedding on train repeats. Upper panel is colored according to trial time, lower panel is
colored according to overall time (train portion of data, test only shown above). In all embedding plots, the
test-embedding smoothed with a 151-sample filter is overlayed as a black line.


Here we evaluate DCL using a real-world dataset obtained from the Allen Institute (de Vries et al.,
2020). The dataset contains recordings from awake, head-fixed mice as they viewed visual stimuli
including three movies on a continuous loop. The recordings were collected using 2-photon calcium
imaging. Our analysis focuses on paired data from 10 repetitions of “Natural Movie 1” comprised of
900 frames (Fig 16a). This exact dataset was also used by Schneider et al. (2023) and available as
allen-movie-one-ca-VISp-800-* in the CEBRA software package.


J.1 M ETHODS


We train DCL with an LDS dynamics model. As our baseline, we use the identical training setup
but with identity dynamics as in our synthetic experiments. We use an MLP with three layers.
The number of units for each layer scales with the embedding dimensionality _d_ . The last hidden
layer has 10 _d_ units and all previous layers have 30 _d_ units. We train on batches with 512 samples
(reference and positive) and use 2 [14] = 16384 negative samples. The model is optimized using the
Adam optimizer (Kingma, 2014) with learning rate 10 _[−]_ [4] . We train DCL for 10k steps using the
negative mean squared error as the similarity metric. For both models we also make use of the
time-lag embeddings introduced in Appendix H (Figure 14), setting _k_ = 5 and _τ_ = 20 (Figure 16b).
Additionally, we vary the dimensionality _d ∈{_ 8 _,_ 16 _,_ 32 _,_ 64 _}_ of the embeddings. We also vary the
time offset _i_ that determines the time step of the positive sample _t_ pos = _t_ ref + _i_ relative to the time step
of the reference sample and run experiments for _i ∈{_ 1 _,_ 5 _,_ 10 _,_ 15 _,_ 20 _,_ 25 _,_ 30 _}_ . We train each model 5
times with different initializations. For a shuffle control, we randomize the time dimension within
each individual movie repeat (900 frames) and conduct the same training as previously discussed.
We train on the neural activity of the first 9 repetitions (8100 samples, 270s) and use the 10th (900
samples, 30s) for evaluation. To evaluate the models, we compute the consistency metric – the _R_ [2]
metric between the embeddings – used by Schneider et al. (2023) between the five runs of the same
model and training setup as a proxy for empirical identifiability. For the consistency calculation, we


33


Published as a conference paper at ICLR 2025


a **b** c













identity LDS dynamics



Consistency, identity dynamics (R²)



Consistency, identity dynamics (R²)



Figure 17: Comparison of consistency with a shuffle control on (a) the train split and (b) the validation split. (c),
example embeddings for both identity and LDS dynamics, in shuffle and no shuffle conditions.


fit a linear regression model between all pairs of embeddings across training runs, and quantify the
resulting 5 _×_ 4 comparisons using the _R_ [2] . We report the mean _R_ [2] on the validation trial across all
comparisons.


J.2 R ESULTS


We compare the consistency scores between multiple runs of the same model training for DCL
with LDS and our baseline (Figure 16c). Overall, DCL with a linear dynamics model has higher
consistency scores compared to training without a dynamics model for all settings except for time
offset equal to 1, where results are on par. The positive effect of leveraging dynamics learning
increases as we increase the time offset. Across all variations of the time offset, we additionally
observe an increase in consistency with an increase in the dimensionality up to 32 dimensions,
followed by a drop in consistency for 64-dimensional embeddings.


In addition to the quantitative performance improvements of DCL with LDS, we can also observe
more structured embedding spaces for DCL with LDS in Figures 16c and 16d compared to DCL
without a dynamics model. The embeddings of DCL with LDS exhibit a clear manifold and follow
relatively smooth trajectories, while the embeddings and trajectories of DCL without a dynamics
model are considerably more entangled. Considering the color code showing the relative time of each
trial, we can also see that DCL with LDS recovers an embedding space in which each trial follows
roughly the same circular motion as the other trials.


When removing temporal structure by shuffling (Fig. 17), neither embedding shows non-trivial
structure and the consistency metric is low on both the train (panel a) and validation set (panel b).


J.3 D ISCUSSION


We observe the strongest improvements in consistency for those settings in which the time difference
between the reference and positive sample is the largest (time offset 30). Under our assumed dynamics
model, predicting further ahead in time results in dynamics which differ increasingly from the identity
dynamics. In our synthetic data experiments, we already observed a similar effect. The identity
dynamics baseline was able to estimate an embedding space when the noise dominated the system
dynamics, but failed to estimate the embedding space as the dynamics became more prominent
(Fig 5).


Note that our baseline model is similar – but technically not identical – to the CEBRA-time models
considered by Schneider et al. (2023) on the same dataset. Our results demonstrate potential
improvements through the introduction of the dynamics model: First, we demonstrate that fitting
embeddings in Euclidean space using a negative mean squared error as the loss function, vs. the
cosine similarity used by Schneider et al. (2023) benefits from the introduction of a dynamics model.
Second, the circular, repetitive structure of movie repeats also emerges in this Euclidean space,
most clearly in the presence of a dynamics model (Fig. 16c, i, LDS). Third, we found it crucial to
include the positive sample in the denominator which stabilizes training in the absence of normalized
embeddings.


34


Published as a conference paper at ICLR 2025


K C OMPUTATIONAL R EQUIREMENTS


As stated in the main paper, we required 120 GPU days of compute for the experiments we ran for
the paper. This does not include the initial period of prototyping and exploration that preceeded the
final sweep of experiments. To provide more transparency and more detailed breakdown, we list the
number of experiments (= model trainings) run for each table and figure in the paper.


Result Number of Experiments


Table 1 171
Figure 4 (SLDS) 162
Figure 5 & 7 (Lorenz) 675
Figure 6 (Ablation) 648
Figure 8 (KDE) 1,125
Figure 9 (SLDS with vMF) 432


Total 3,213


Table 5: Number of experiments per table/figure.


For 120 GPU days this comes out at an average of 53 minutes per experiment that made it into the
final paper. However, the actual runtime of an average DCL training is 15-20 minutes for settings
equivalent to the experiments in Table 1. This difference to what we report as the overall average
compute time per experiment can be attributed to several factors: (1) approximately one-third of
experiments involved KDE estimation which required 4-6x longer training times, (2) extensive
evaluation and metric computation for debugging and reporting purposes added additional overhead,
and (3) additional experimental iterations that did not make it into the final paper but contributed to
the total compute time of the final sweep.


35


Published as a conference paper at ICLR 2025


L O N COMPONENT - WISE VS . LINEAR IDENTIFIABILITY


In the context of non-linear ICA, the mean correlation coefficient (MCC) is frequently reported
as a measure of _component-wise identifiability_ . In non-linear ICA, it is typically assumed that a
set of independent sources _s_ 1 ( _t_ ) _, . . ., s_ _n_ ( _t_ ) is passed through a mixing function to arrived at the
observable signal (cf. Hyvarinen & Morioka, 2017). In contrast, in our work the sources are not
independent, but are conditioned on the previous time step and the passed through a dynamics model.
This is conceptually similar to the conditional independent assumption in Hyvarinen et al. (2019)
with auxiliary variable _**f**_ ( _**x**_ ), but with the distinction that at training time, we do not have _**u**_ available,
only _**x**_ which requires the use of a dynamics model.


Because of these distinctions in the generation of the latent space, we can generally not expect
component-wise identifiability (Theorem 1) but will instead obtain linear identifiability. Related
work in non-linear ICA likewise can only provide linear identifiability for a comparable Gaussian
case (Hyvarinen et al., 2019; Zimmermann et al., 2021; Schneider et al., 2023). However, if we
assume access to the dynamics model (but not the actual latent space), it is possible to reduce
ambiguity in the latent space, and assuring component-wise identifiability.


In Table 6 we compare MCC and % _R_ [2] across two datasets (SLDS and Lorenz) to extend Table 1
of the main paper. As expected, we observe a non-perfect MCC score for all models except for the
SLDS model which is provided with the ground-truth dynamics. Due to Eq 29 this strenghten the
guarantee to component-wise identifiabilily, yielding an MCC of close to 100%.


Data _**f**_ Model _**f**_ [ˆ] MCC % _R_ [2]


SLDS identity 0 _._ 59 _±_ 0 _._ 06 76 _._ 80 _±_ 7 _._ 40
SLDS SLDS 0 _._ 70 _±_ 0 _._ 05 99 _._ 52 _±_ 0 _._ 05

SLDS GT 1 _._ 00 _±_ 0 _._ 00 99 _._ 20 _±_ 0 _._ 10


Lorenz identity 0 _._ 34 _±_ 0 _._ 07 41 _._ 00 _±_ 8 _._ 57
Lorenz LDS 0 _._ 68 _±_ 0 _._ 14 81 _._ 20 _±_ 16 _._ 9

Lorenz SLDS 0 _._ 78 _±_ 0 _._ 13 94 _._ 08 _±_ 2 _._ 75


Table 6: SLDS and Lorenz dataset from Table 1 with the addition of the MCC metric.


36


Published as a conference paper at ICLR 2025


M C OMPARISON TO ADDITIONAL TIME - SERIES MODELS


The baseline – DCL without a dynamics model – employed in all experiments in the main paper is
technically a CEBRA-time (Schneider et al., 2023) model which was originally designed for timeseries inference. Note that we use the negative mean squared error as the similarity measure in almost
all experiments, and include general improvements for estimation of Euclidean embeddings also in
the baseline model (positive sample in the denominator of the InfoNCE loss, additional negative
examples). Other modes of CEBRA-time (using a cosine similarity, etc.) were not prominently
considered in this work. Other popular contrastive learning methods include time-contrastive learning
(TCL; Hyvarinen & Morioka, 2016), permutation contrastive learning (PCL; Hyvarinen & Morioka,
2017). More recently, VAE models designed for time-series analysis and dynamics learning were
proposed, such as Temporally Disentangled Representation Learning (TDRL; Yao et al., 2022).


Naturally, these methods propose different data generating processes, and the empirical performance
of DCL as well as these comparison methods will strongly depend on whether these assumptions
are met. For instance, TCL requires non-stationary independent sources, PCL requires stationary
independent sources, and TDRL uses non-parametric transition models with non-Gaussian noise.
Hence, in general, it is not possible to fairly compare all these methods as they have different regimes
of operation. We still include a comparison of these methods and DCL for dynamics inference.


M.1 V ERIFYING B ASELINES


Yao et al. (2022) published a benchmarking setup for these algorithms [3] . We adapted the TDRL
codebase minimally and provide a reference implementation for our experiments in our official code
release. Our implementation can be diffed against the original TDRL code to highlight the minimal
code changes performed for running the benchmarking suite. We leverage this codebase to run
verified baseline algorithms on our SLDS dataset. First, we ensure that we can reproduce the results
from Yao et al. (2022). We report the results in Table 7 for the “changing” experimental setting.


We perform 5 runs with different seeds for the exact hyperparameter configurations reported in the
TDRL codebase. Remaining differences in the in numbers might be attributed to discrepancies
between the paper and the code distribution or the choice of different random seeds, as we observed
large variances in some of the models. We label these models with ”-B” to indicate the _base_
configuration provided by the public code base. We report the full results in Table 7.


MCC
Model Reproduced Reported


PCL-B 0 _._ 535 _±_ 0 _._ 030 0 _._ 599 _±_ 0 _._ 041

TCL-B 0 _._ 367 _±_ 0 _._ 018 0 _._ 399 _±_ 0 _._ 021

TDRL-B 0 _._ 910 _±_ 0 _._ 067 0 _._ 958 _±_ 0 _._ 017


Table 7: Verification on “changing” experiment Setting from Yao et al. (2022)


M.2 E XPERIMENT D ETAILS


Both TCL and TDRL make use of a categorical context variable indicating changes in the distributions
of the true latents. To make the comparison to our framework as fair as possible, we choose the SLDS
datasets to allow for a similar context variable in form of the mode/state sequence that modulates the
switching between linear dynamics. Our dataset is comprised of 5 modes, which corresponds to the
same number of categories the models from Table 7 already use.


**Base models.** To be able to apply the baseline models to our SLDS setting, we only change their
base configuration in two required ways: We increase the input dimension from 8 to 50 to match the
observation produced by the SLDS and reduce the latent dimension from 8 to 6.


**Large models.** Because PCL is the closest match to our existing baseline (CEBRA-time) and our
encoder architecture is equal to the baseline architecture, we introduce an additional variant “PCL-L”
(L=Large) to match the number of parameters as close as possible. We do so by increasing the hidden
dimension of the PCL encoder model from 50 to 160 and reduce the number of layers from 4 to 3,


3 [Code: https://github.com/weirayao/tdrl (MIT License)](https://github.com/weirayao/tdrl)


37


Published as a conference paper at ICLR 2025


low noise high noise
Model MCC (%) _R_ [2] (%) MCC (%) _R_ [2] (%)


TCL-B 36 _._ 07 _±_ 2 _._ 27 66 _._ 56 _±_ 3 _._ 87 37 _._ 21 _±_ 3 _._ 66 61 _._ 75 _±_ 12 _._ 56

PCL-B 68 _._ 28 _±_ 2 _._ 40 91 _._ 33 _±_ 0 _._ 85 66 _._ 86 _±_ 3 _._ 16 77 _._ 99 _±_ 3 _._ 93

PCL-L 68 _._ 02 _±_ 2 _._ 77 90 _._ 92 _±_ 1 _._ 16 69 _._ 67 _±_ 3 _._ 86 80 _._ 88 _±_ 1 _._ 38

TDRL-B 64 _._ 34 _±_ 6 _._ 01 83 _._ 85 _±_ 7 _._ 78 62 _._ 93 _±_ 5 _._ 37 80 _._ 90 _±_ 7 _._ 82

TDRL-L 63 _._ 51 _±_ 4 _._ 87 84 _._ 01 _±_ 7 _._ 98 62 _._ 65 _±_ 6 _._ 29 81 _._ 40 _±_ 6 _._ 42


CEBRA-time 59 _._ 46 _±_ 5 _._ 84 76 _._ 80 _±_ 7 _._ 40 65 _._ 71 _±_ 4 _._ 99 98 _._ 66 _±_ 0 _._ 19

DCL+SLDS 69 _._ 62 _±_ 4 _._ 78 99 _._ 52 _±_ 0 _._ 05 68 _._ 76 _±_ 5 _._ 05 98 _._ 95 _±_ 0 _._ 08


DCL+GT SLDS 99 _._ 51 _±_ 0 _._ 06 99 _._ 20 _±_ 0 _._ 10 98 _._ 57 _±_ 0 _._ 16 97 _._ 82 _±_ 0 _._ 17


Table 8: Baseline results for TCL, PCL, and TDRL models on switching linear dynamics datasets. The low
noise setting is equivalent to Table 1. For the high noise setting (low ∆ _t_ ), we use larger noise and lower rotation
angles, setting _σ_ _**ε**_ = 0 _._ 001, max( _θ_ _i_ ) = 5.


effectively increasing the number of parameters by factor 5. Because TDRL can be considered the
most promising baseline candidate (beside CEBRA-time) based on the results from table 7, we also
double its encoder size from using hidden dimension 128 to 256, resulting in the ”TDRL-L” baseline
model.


**Dataset.** Leverage two versions of the SLDS dataset used in the main paper. First, we apply it to the
exact setting of the SLDS in our Table 1 to compare against our default setting with dynamics noise
_σ_ _**ε**_ = 0 _._ 0001 and rotation angles max( _θ_ _i_ ) = 10 . Additionally, since our main baseline (CEBRA-time)
performed best on datasets with lower ∆ _t_ where the noise dominates over the dynamics, we also
compare against an SLDS dataset generated with larger dynamics noise _σ_ _**ε**_ = 0 _._ 001 and smaller
rotation angles max( _θ_ _i_ ) = 5 (see Figure 4b). We generate 3 different versions of each dataset using
different random seeds and on each dataset we train 3 models with different seeds, resulting in 9
models for each baseline and for each of the two settings. We train every baseline model for 50
epochs or until the training time reaches 8 hours.


**Metrics.** We compute the Mean Correlation Coefficient (MCC) as well as the _R_ [2] metric used for
Table 1 during training. For the baselines run with the public code base from Yao et al. (2022), we
follow their reporting strategy and report the best MCC complemented by the the best _R_ [2] achieved at
any point during training to make the baseline appear even stronger. For our DCL models and our
default baseline, we report the MCC and _R_ [2] based on the last model checkpoint as in the main paper.


M.3 R ESULTS


We outline results for all benchmarked algorithms in Table 8.


In the low noise setting, DCL with the SLDS dynamics model achieves an _R_ [2] of 99.5%. The next
best baseline algorithm is the PCL base model, with a maximum _R_ [2] of 91.3%. While both DCL and
PCL are learning by contrasting samples across time in the time series, only DCL with the SLDS
dynamics model can fully model the ground truth dynamical process. In contrast, the score function
in PCL is setup to model variations along _independent_ latent dimensions.


Interestingly, PCL outperforms CEBRA-time, which is equivalent to running DCL without a dynamics model (76.8%). This indicates that the score method in PCL (component-wise linear transformations) outperforms the score method in CEBRA-time on this dataset (negative squared Euclidean
distance). It would be interesting to combine the scoring method in PCL with the dynamics model in
DCL for further improvements on non-linear dynamics settings.


In the high noise setting, DCL with SLDS dynamics achieves comparable performance as CEBRAtime, as outlined in the main paper (99.0% vs. 98.7%). In this setting, PCL performs worse,
potentially because the score function in CEBRA-time is better matched to the dominating Gaussian
system noise.


In all cases, MCC is not a meaningful metric, with highest scores ranging around 60–70%. An
exception is training DCL with the underlying ground truth dynamics system, which achieves
component-wise identifiability and an MCC of 99.51%.


38


