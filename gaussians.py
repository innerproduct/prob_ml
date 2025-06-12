import dataclasses
from typing import Self
import functools

from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

@dataclasses.dataclass
class Gaussian:
    # Gaussian distribution with mean mu and covariance Sigma
    mu: Float[Array, "D "]
    Sigma: Float[Array, "D D"]

    @functools.cached_property
    def cov_SVD(self):
        """square root of the covariance matrix, vis SVD"""
        if jnp.isscalar(self.mu):
            return jnp.eye(1), jnp.sqrt(self.Sigma).reshape(1, 1)
        else:
            Q, S, _ = jnp.linalg.svd(self.Sigma, full_matrices=True, hermitian=True)
            return Q, jnp.sqrt(S)
    
    @functools.cached_property
    def logdet(self):
        """log-determinant of the covariance matrix
        e.g. for computing the log-pdf
        """
        _, S = self.cov_SVD
        return 2 * jnp.sum(jnp.log(S))

    @functools.cached_property
    def precision(self):
        """precision matrix.
        you probably don't want to use this directly but rather prec_mult
        """
        Q, S = self.cov_SVD
        return Q @ jnp.diag(1 / S) ** 2 @ Q.T

    def prec_mult(self, x: Float[Array, "D "]) -> Float[Array, "D "]:
        """precision matrix multiplication
        implements Sigma^{-1} @ x. For numerical stability, we use Cholesky factorization.
        """
        Q, S = self.cov_SVD
        return Q @ jnp.diag(1 / S**2) @ Q.T @ x
    
    def log_pdf(self, x: Float[Array, "D "]) -> float:
        """log N(x;mu,Sigma)"""
        return (
            -0.5 * (x - self.mu) @ jnp.linalg.solve(self.Sigma, x - self.mu)
            - 0.5 * jnp.linalg.slogdet(self.Sigma)[1]
            - 0.5 * len(self.mu) * jnp.log(2 * jnp.pi)
        )
    
    def pdf(self, x: Float[Array, "D "]) -> float:
        """N(x;mu,Sigma)"""
        return jnp.exp(self.log_pdf(x))
        
    def cdf(self, x):
        if jnp.isscalar(self.mu):
            return 0.5 * (
                1 + jax.scipy.special.erf((x - self.mu) / jnp.sqrt(2 * self.Sigma)) # type: ignore
            )
        else:
            raise NotImplementedError("CDF for multivariate Gaussian not implemented")
            
    def precision_using_inv(self):
        """precision matrix.
        you probably don't want to use this directly
        """
        return jnp.linalg.inv(self.Sigma)

    def mp(self):
        """precision-adjusted mean"""
        return self.prec_mult(self.mu)
        
    def mp_v0(self):
        """precision-adjusted mean"""
        return self.precision @ self.mu
        
    def __mul__(self, other: Self) -> Self:
        """Products of Gaussian pdfs are Gaussian pdfs! (Up to a normalization constant.)
        Multiplication of two Gaussian PDFs (not RVs!)
        other: Gaussian RV
        """
        Sigma = jnp.linalg.inv(self.precision + other.precision)
        mu = Sigma @ (self.mp + other.mp)
        return Gaussian(mu-mu, Sigma=Sigma)
    
    @functools.singledispatchmethod    
    def __add__(self, other: Float[Array, "D "] | float) -> Self:
        """Affine maps of Gaussian RVs are Gaussian RVs
        shift of a Gaussian RV by a constant.
        We implement this as a singledispatchedmethod, jnp.ndarrays cannot be dispatched on,
        and register the addition of two RVs below
        """
        other = jnp.asarray(other)
        return Gaussian(mu=self.mu + other, Sigma=self.Sigma)
        
    def __rmatmul__(self, A: Float[Array, "N D"]) -> Self:
        """
        Linear maps of Gaussian RVs are Gaussian RVs
        returns 
        p(A @ x) = N(A @ x; A @ mu, A @ Sigma @ A.T)
        """
        return Gaussian(mu=A @ self.mu, Sigma=A @ self.Sigma @ A.T)
        
        
        
    def __getitem__(self, i) -> Self:
        """marginals. Anyone know a type-hint for slicing strings?"""
        return Gaussian(
            mu=jnp.atleast_1d(self.mu[i]), Sigma=jnp.atleast_ed(self.Sigma[i, i])
        )
        
    def condition(self, A: Float[Array, "N D"], y: Float[Array, "N"],
                      Lambda: Float[Array, "N N"]) -> Self:
         """Linear conditionals of Gaussian RVs are Gaussian RVs
         returns p(self | y) = N(y; A @ self, Lambda) * self / p(y)
         """
         Gram = A @ self.Sigma @ A.T + Lambda
         L = jax.scipy.linalg.cho_factor(Gram, lower=True) # type: ignore
         mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu) # type: ignore
         Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve( # type: ignore
                 L, A @ self.Sigma
         )
         return Gaussian(mu=mu, Sigma=Sigma)
         
    def condition_pls(
        self,
        A: Float[Array, "N D"],
        y: Float[Array, "N"],
        Lambda: Float[Array, "N N"],
        max_steps=None,
        policy=None,
        atol=1e-6,
        rtol=1e-6,        
    ) -> Self:
        """
        Condition, using probabilistic linear solver
        """
        N, M = A.shape
        assert y.shape == (N,)
        assert Lambda.shape == (N, N)
        assert self.mu.shape == (M,)
        
        # the terms that show in the computation of the posterior mean / cov:
        Gram = A @ self.Sigma @ A.T + Lambda # shape (N, N)
        cov_pred_obs = self.Sigma @ A.T # covariance of the prior with the observations
        b = y - A @ self.mu
        
        # solving with a probabilistic linear solver:
        solver_prior = Gaussian(mu=jnp.zeros((N,)), Sigma=jnp.eye(N))
        # solve, _, _, _ = GEQRF(Gram, solver_prior, max_steps=max_steps)
        solve, _, _, _, _ = GEPNF(
            Gram,
            solver_prior,
            b=b,
            max_steps=max_steps,
            policy=policy,
            atol=atol,
            rtol=rtol,
        )
        
        posterior_mu = self.mu + cov_pred_obs @ solve(b).mu
        cov_correction = jnp.array(
            [solve(cov_pred_obs[i, :].mu for i in range(cov_pred_obs.shape[0]))]        
        )
        posterior_Sigma = self.Sigma - cov_pred_obs @ cov_correction.T
        return Gaussian(mu=posterior_mu, Sigma=posterior_Sigma)
    
    def sample(
           self,
           key: PRNGKeyArray,
           num_samples: int = 1,
    ) -> Float[Array, "N D"]:
        """sample from the Gaussian RV"""
        if jnp.isscalar(self.mu):           
            return jax.random.normal(key, (num_samples,)) * self.std + self.mu  # type: ignore #DOUBLE CHECK THIS
        else:
            Q, S = self.cov_SVD
            z = S[..., :] * jax.random.normal(key, (num_samples, len(self.mu))) # type: ignore
            return z @ Q.T + self.mu
            




@Gaussian.__add__.register
def _add_gaussians(self, other: Self) -> Self:
    # sum of two Gaussian RVs (whose JOINT distribution is Gaussian)
    return Gaussian(mu=self.mu + other.mu, Sigma=self.Sigma + other.Sigma)

### Plotting Tools ###
def gp_shading(yy, g: Gaussian) -> Float[Array, "N M"]:
    return jnp.exp(
        -((yy - g.mu) ** 2) / (2 * g.std**2)
    ) # / (std * jnp.sqrt(2 * jnp.pi))



def plot_gaussian(
    ax,
    g: Gaussian,
    xplot: Float[Array, "N"],
    color="C0",
    yy=None,
    cmap="viridis",
    key=jax.random.PRNGKey(0), # type: ignore
    **kwargs
) -> None:
    
    ax.plot(xplot, g.mu, color=color, lw=1, **kwargs)
    ax.plot(xplot, g.mu + 2 * g.std, color=color, **kwargs, lw=0.5, ls="--")
    ax.plot(xplot, g.mu - 2 * g.std, color=color, **kwargs, lw=0.5, ls="--")
    ax.plot(xplot, g.sample(key, num_samples=3).T, color=color, **kwargs, lw=0.5)
    
    if yy is not None:
        shading = gp_shading(yy, g)
        ax.imshow(
            shading,
            extent=(xplot[0, 0], xplot[-1, 0], jnp.min(yy), jnp.max(yy)),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            alpha=0.8,  # shading / jnp.max(shading))
        )
        ax.set_ylim(jnp.min(yy), jnp.max(yy))
        
### Probabilistic Linear Algebra

class CGPolicy:
    def __call__(self, *, r, **kwargs):
        return -r
        
class CholeskyPolicy:
    def __init__(self, pivoted=True):
        self._pivoted = pivoted
        
    def __call__(self, *, i, r, **kwargs):
        s = jnp.zeros_like(r)
        return s.at[jnp.abs(r).argmax() if self._pivoted else i].set(1)

def GEQRF(
     A: Float[Array, "M N"], prior: Gaussian
) -> (Callable, Float[Array, "M M"], Float[Array, "M N"], Float[Array, "M M"]): # type: ignore
    M, N = A.shape # dimensions
    U, R, nu = jnp.zeros((M, N)), jnp.zeros((N, N)), jnp.zeros(N) # storage
    Sigma = prior.Sigma
    for n in range(N): # matrix decomposition
        an = A[:, n]
        un = Sigma @ an
        un = un / jnp.sqrt(jnp.dot(an, un))
        U = U.at[:, n].set(un)
        R = R.at[: n + 1, n].set(an.T @ U[:, : n + 1])
        nu = nu.at[n].set(jnp.dot(an, prior.mu))
        Sigma = Sigma - jnp.outer(un, un)
    def solve(b: Float[Array, "N"]) -> Gaussian: # solver routine
        alpha = jnp.zeros((N,))
        alpha = alpha.at[0].set((b[0] - nu[0]) / R[0, 0])
        for n in range(1, N):
            alpha = alpha.at[n].set(
                    (b[n] - nu[n] - jnp.dot(alpha[:n], R[:n, n])) / R[n, n]            
            )
        return Gaussian(prior.mu + U @ alpha, Sigma)
        
    return solve, R, U, Sigma


def GEPNF(
    A: Float[Array, "M N"],
    prior: Gaussian,
    b: Float[Array, "M"],
    max_steps=None,
    policy=None,
    atol=1e-6,
    rtol=1e-6,
) -> (Callable, Float[Array, "M M"], Float[Array, "M N"], Float[Array, "M M"]): # type: ignore
    """
    Probabilistic linear solver
    Generalized version of GEQRF, for general matrix decompositions, identified by a policy
        
    Solves the linear system A.T x = b for x, where
    A: matrix of size (M, N)
    prior: Gaussian prior over the solution x of size (N,)
    b: right-hand side of size (M,). Used only for the solve policy (can be set toi a random vector if no rhs is given. Not used by all policies))    
    """
    pass