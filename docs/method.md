# Mathematical specification

Source: Zahraoui et al., *Enhancing evidence estimation through informed
probability density approximation*, [arXiv:2512.10283v2](https://arxiv.org/abs/2512.10283v2),
Section III, Algorithm 1, and Section VII; published
[DOI:10.1103/1554-y6ns](https://doi.org/10.1103/1554-y6ns).
The accessible public v2 was used because the supplied attachment could not be
downloaded in the development environment. Its equations were inspected directly.

## Morph and total correlation

For dimension d and order L, choose m=floor(d/L) non-overlapping length-L blocks.
Dimensions outside these blocks receive independent singleton factors (Eqs. 3-5):

$$
M(\theta)=\prod_{b\in B_L}\widehat p_b(\theta_b)
\prod_{s\in S(B_L)}\widehat p_s(\theta_s).
$$

Each factor is normalized, and every dimension appears once, so M is normalized.
Its log density is the sum of factor log densities. A joint draw uses independent
keys and independent draws for every factor.

Equation 6 defines block total correlation. We estimate it on fitting samples:

$$
\widehat C_b=\sum_i w_i\left[
\log\widehat p_b(\theta_{i,b})-\sum_{k\in b}\log\widehat p_k(\theta_{i,k})
\right],\qquad \sum_iw_i=1.
$$

This is an in-sample KDE plug-in estimate, not an exact KL divergence. Negative
finite-sample values are retained in diagnostics and clipped to zero for SGM's
nonnegative scores. Marginal log densities are cached across candidates. The
weighted extension uses the same normalized weights for KDEs and expectations.

## KDE convention

For block dimension q, effective sample size is n_eff=1/sum(w_i^2). Define
mu=sum(w_i*x_i) and the unbiased weighted covariance

$$
\Sigma=\frac{\sum_iw_i(x_i-\mu)(x_i-\mu)^T}{1-\sum_iw_i^2},\qquad
h=\left[n_{eff}(q+2)/4\right]^{-1/(q+4)}.
$$

Each mixture component has covariance h^2 Sigma and weight w_i. This is the
multivariate Silverman convention in SciPy/JAX, not the univariate robust
IQR-based bandwidth variant. `bandwidth_scale` multiplies h. The opt-in ridge
changes Sigma to Sigma+regularization*diag(diag(Sigma)); its default is zero.

## SGM

Algorithm 1 is implemented directly: stable-sort all candidate blocks by score;
start one construction from each of the top K seeds; scan candidates in ranked
order and accept disjoint blocks until m are selected; retain the highest-scoring
construction. On tied scores, retain input candidate order and the first winning
seed. Candidate enumeration uses lexicographic order. The complete candidate set
ensures each construction can be filled. SGM does not guarantee a global maximum.

## MorphZ

Fit the proposal using one posterior batch; use a separate batch for bridge
sampling. With q(theta)=L(theta)*pi(theta), r(theta)=q(theta)/M(theta), and
f1=Nposterior/(Nposterior+Nproposal), f2=1-f1, Eqs. 9-10 imply the iteration

$$
z_{t+1}=\frac{\operatorname{mean}_{M}[r/(f_1r+f_2z_t)]}
{\operatorname{mean}_{posterior}[1/(f_1r+f_2z_t)]}.
$$

The solver uses log-sum-exp/log-add-exp and an importance estimate for its initial
value. Convergence is |delta log z| <= log(1+rtol), which is invariant to target
normalization offsets. It reports iteration history and explicit nonconvergence.
If every proposal has zero target density, the wrapper raises a no-overlap error.

Relative error follows [Gronau et al. (2017), Eq. 17](https://arxiv.org/abs/1703.05984).
For p=q/z, a=p/(f1*p+f2*M), b=M/(f1*p+f2*M), it is

$$
RE^2=\frac{\operatorname{var}_M(a)}{N_{proposal}\operatorname{mean}_M(a)^2}
+\frac{\tau_b\operatorname{var}_{posterior}(b)}
{N_{posterior}\operatorname{mean}_{posterior}(b)^2}.
$$

We estimate tau_b with an FFT initial-positive-sequence estimator, rather than
the tutorial's autoregressive spectral-density fit. Alternatively supply tau_b.
This uncertainty is conditional on fitted M and posterior coverage; it does not
include KDE fitting, block-selection uncertainty, or missed posterior modes.
It is returned as NaN when the bridge iteration fails to converge.
