---
title: "보조정리 4 and 정리 5"
author: "장태영"
date: 2022-07-27
weight: 5
---

$\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}$
$\newcommand{\abs}[1]{\left\lvert #1 \right\rvert}$
$\newcommand{\Uc}{\mathcal{U}}$
$\newcommand{\Cc}{\mathcal{C}}$
$\newcommand{\eps}{\epsilon}$
$\newcommand{\Real}{\mathbb{R}}$

<div class="lemma">[Lemma 3 in the paper]

If $\Sigma_0 \in \Uc(s_0, \zeta_0)$ and $\Sigma\in \Uc(\zeta)$ then we have 

1. $K(f_{\Sigma_0}, f_\Sigma)\leq \zeta^4\zeta_0^2 \norm{\Sigma-\Sigma_0}_F^2 $
2. $V(f_{\Sigma_0}, f_\Sigma)\leq \frac 32 \zeta^4\zeta_0^2 \norm{\Sigma-\Sigma_0}_F^2 $

</div>

<div class="lemma">[Lemma 4 in the paper]

If $a=b=1/2$ , $x>1$, and $\tau/x >0$ is sufficiently small then $\pi_{ij}^u(x)\geq \sqrt{\frac{1}{2\pi}} \frac{\tau}{x^2}$ where $\pi_{ij}^u(\sigma_{ij})$ is the unconstrained marginal prior density of $\sigma_{ij}$

</div>


<div class="theorem">[Theorem 5 in the paper ; The lower bound for $\pi(B_{\eps_n})$]
Here are the conditions we need for this theorem.

1. $\Sigma_0 \in \Uc(s_0, \zeta_0)$ with $\zeta_0 <\zeta$
1. $p \asymp n^\beta$ for some $0<\beta<1$
1. $\zeta^4\leq p$
1. $\zeta^2\zeta_0^2 \leq s_0\log p$
1. $n \geq \max \\{1/\zeta_0^4,s_0/ (1-\zeta_0/\zeta)^2 \\} \log p/\zeta^4$
1. $p^{-1}<\lambda < \log p/\zeta_0$
1. $a=b=1/2$
1. $(p^2\sqrt{n})^{-1}\lesssim \tau \lesssim (p^2\sqrt{n})^{-1}\sqrt{s_0\log p}$ 
1. (Additional, From Thm 1 at page 5 of the paper) $(p+s_0)\log p = o(n)$ i.e. $\eps_n^2 \rightarrow 0$
1. (Additional, From page 4 of the paper) $p = O(s_0)$

If the conditions above hold, then we have $\pi(B_{\eps_n})\geq \exp\Big\\\{-(5+\frac 1\beta )n\eps_n^2 \Big\\\}$
</div>

<div class="proof">[Proof of Lemma 4]

Because we have $a=b=1/2$,
$$\sigma_{ij}|\rho_{ij} \sim N(0, \frac{\rho_{ij}}{1-\rho_{ij}}\tau^2) \\;,\\; \rho_{ij} \sim \text{Beta}(a,b) $$ is equivalent to $$\sigma_{ij}| \lambda_{ij} \sim N(0, \lambda_{ij}^2\tau^2)\\;,\\; \lambda_{ij} \sim \text{C}^+(0, 1)$$ where $\text{C}^+(0, s)$ denotes the standard half-Cauchy distribution on positive real with a scale parameter $s$. 

$$\begin{equation*}
\begin{aligned}
p(\sigma, \rho) &= p(\sigma | \rho) p(\rho) = \frac{1}{\sqrt{2\pi \frac{\rho}{1-\rho}\tau^2}}\exp\Big(-\frac{1}{2\frac{\rho}{1-\rho}\tau^2}\sigma^2 \Big) \frac{1}{\pi}\rho^{-1/2}(1-\rho)^{-1/2} \quad \because \\; \Gamma(1/2) = \sqrt{\pi}
\end{aligned}
\end{equation*}$$

$$\begin{equation*}\begin{aligned}
    \lambda &= \sqrt{\frac{\rho}{1-\rho}} \quad (\lambda>0)\quad \text{and} \quad \rho = \frac{\lambda^2}{\lambda^2+1} \\\\
    \text{Jacobian} &= \abs{\frac{d\rho}{d\lambda}} = \frac{2\lambda}{(\lambda^2+1)^2}
\end{aligned}\end{equation*}$$

$$\begin{equation*}\begin{aligned}
    p(\sigma, \lambda) &= \frac{1}{\sqrt{2\pi \lambda^2 \tau^2}}\exp\Big(-\frac{1}{2\lambda^2 \tau^2}\sigma^2 \Big) \frac{1}{\pi} \sqrt{\frac{\lambda^2}{\lambda^2+1} \frac{1}{\lambda^2+1}}^{-1} \frac{2\lambda}{(\lambda^2+1)^2} \\\\
    &= \frac{1}{\sqrt{2\pi \lambda^2 \tau^2}}\exp\Big(-\frac{1}{2\lambda^2 \tau^2}\sigma^2 \Big) \frac{2}{\pi} \frac{\lambda^2+1}{\lambda} \frac{\lambda}{(\lambda^2+1)^2} \\\\
    &= \frac{1}{\sqrt{2\pi \lambda^2 \tau^2}}\exp\Big(-\frac{1}{2\lambda^2 \tau^2}\sigma^2 \Big) \frac{2}{\pi} \frac{1}{(\lambda^2+1)} \\\\
    &= p(\sigma|\lambda)p(\lambda)
\end{aligned}\end{equation*}$$

Hence we can conclude that $$\sigma | \rho \sim N(0, \frac{\rho}{1-\rho}\tau^2) \\;, \\; \rho \sim \text{Beta}(1/2, 1/2)$$ is equivalent to $$\sigma |\lambda \sim N(0, \lambda^2 \tau^2)\\;, \\; \lambda\sim C^+(0,1)$$

Now we shall derive tight bound for marginal prior density of $\sigma$
$$\begin{equation*}\begin{aligned}
    \pi^u(\sigma) &= \int_0^\infty p(\sigma, \lambda) \, d\lambda \\\\
    &= \int_0^\infty \frac{1}{\sqrt{2\pi \lambda^2 \tau^2}}\exp\Big(-\frac{1}{2\lambda^2 \tau^2}\sigma^2 \Big) \frac{2}{\pi} \frac{1}{(\lambda^2+1)} \, d\lambda \\\\
    & \text{change of variable} : u = 1 / \lambda^2 \Leftrightarrow \lambda = u^{-1/2} \quad d\lambda = -\frac{1}{2} u^{-3/2}\, du \\\\
    &= \int_0^\infty \frac{1}{\sqrt{2\pi \tau^2}} u^{1/2}\exp\Big(-\frac{1}{2\tau^2}\sigma^2 u \Big) \frac{2}{\pi}\frac{u}{1+u} \frac{1}{2}u^{-3/2}\, du \\\\
    &= \int_0^\infty \frac{1}{\sqrt{2\pi^3 \tau^2}} \exp\Big(-\frac{\sigma^2}{2\tau^2} u \Big) \frac{1}{1+u} \, du \\\\
    & \text{change of variable} : z = 1+u \Leftrightarrow u = z-1 \\\\
    &= \frac{1}{\tau \sqrt{2\pi^3}} \exp\Big(\frac{\sigma^2}{2\tau^2} \Big) \int_1^\infty \frac{1}{z} \exp\Big(-\frac{\sigma^2}{2\tau^2}z \Big)\, dz
\end{aligned}\end{equation*}$$

Define exponential integral $E_1$ as the following : $$E_1(x) = \int_1^\infty \frac{1}{z}\exp(-zx)\, dz \quad \forall \\; x>0  $$ Then it has tight bound given as $$\frac{1}{2}\exp(-x)\log\Big( 1+ \frac{2}{x}\Big) < E_1(x) < \exp(-x)\log\Big(1+\frac{1}{x} \Big) \quad x>0 $$
Note that this tight bound is mentioned in [Wikipedia : Exponential integral](https://en.wikipedia.org/wiki/Exponential_integral)

Thus, we have 
$$\begin{equation*}\begin{aligned}
    \pi^u(\sigma) &= \frac{1}{\tau \sqrt{2\pi^3}} \exp\Big(\frac{\sigma^2}{2\tau^2} \Big) E_1\Big(\frac{\sigma^2}{2\tau^2} \Big)
\end{aligned}\end{equation*}$$
Using the tight bound of $E_1$ given above, we get
$$\begin{equation*}\begin{aligned}
    \pi^u(\sigma) &< \frac{1}{\tau \sqrt{2\pi^3}}\log\Big(1+\frac{2\tau^2}{\sigma^2} \Big) \\\\
    \pi^u(\sigma) &> \frac{1}{2\tau \sqrt{2\pi^3}}\log\Big(1+\frac{4\tau^2}{\sigma^2} \Big) 
\end{aligned}\end{equation*}$$
From now on, we will call these two inequalities as upper and lower bound of marginal prior density of $\sigma_{ij}$ respectively. 

Then, using lower bound of marginal prior density of $\sigma_{ij}$, we have the following :
$$\begin{equation*}\begin{aligned}
    \pi_{ij}^u(x) &\geq \frac{1}{2\tau}\sqrt{\frac{1}{2\pi^3}}\log\Big(1+\frac{4\tau^2}{x^2}\Big) \\\\ 
    &\geq \frac{1}{4\tau}\sqrt{\frac{1}{2\pi^3}}\frac{4\tau^2}{x^2} \quad \because \log(1+x)\geq \frac 12 x \quad \text{when} \\; 0\leq x \leq 1 \quad \text{and} \quad \tau/x \\; \text{is suff. small} \\\\
    &= \sqrt{\frac{1}{2\pi^3}}\frac{\tau}{x^2}
\end{aligned}\end{equation*}$$
</div>

<div class="proof">[Proof of Theorem 5]

Note that $B_{\eps}$ is defined as $B_\eps = \\{f_\Sigma : \Sigma \in \Cc_p , \\; K(f_{\Sigma_0}, f_\Sigma)< \eps^2 , \\; V(f_{\Sigma_0}, f_\Sigma)< \eps^2 \\}$

By Lemma 3, it suffices to show that $\pi\Big(\norm{\Sigma- \Sigma_0}_F^2 \leq \frac{2}{3\zeta^4\zeta_0^2}\eps_n^2\Big)\geq \exp(-Cn\eps_n^2)$ 

This is because 

$$
\begin{equation*}
\begin{aligned}
&\norm{\Sigma- \Sigma _ 0} _ F^2 \leq \frac{2}{3\zeta^4\zeta _ 0^2}\eps _ n^2 \\\\ 
&\Rightarrow K(f _ {\Sigma _ 0, f _ \Sigma}) \leq \zeta^4\zeta _ 0^2 \norm{\Sigma - \Sigma _ 0} _ F^2 \leq \frac 23 \eps _ n^2 < \eps _ n^2 \quad \text{and} \quad V(f _ {\Sigma _ 0}, f _ \Sigma) \leq \frac 32 \zeta^4\zeta _ 0^2 \norm{\Sigma - \Sigma _ 0} _ F^2 \leq \eps _ n^2 \\\\
&\Rightarrow f _ \Sigma \in B _ {\eps _ n}
\end{aligned}
\end{equation*}
$$

so that $\pi(B_{\eps_n}) \geq \pi\Big(\norm{\Sigma- \Sigma_0}_F^2 \leq \frac{2}{3\zeta^4\zeta_0^2}\eps_n^2\Big)$

Note that 
$$
\begin{equation*}
\begin{aligned}
&\pi\Big(\norm{\Sigma- \Sigma_0} _ F^2 \leq \frac{2}{3\zeta^4\zeta _ 0^2}\eps _ n^2\Big) = \pi\Big(\norm{\Sigma- \Sigma _ 0} _ F^2 \leq \frac{2}{3\zeta^4\zeta _ 0^2} \frac{(p+s _ 0)\log p}{n} \Big) \\\\
&\geq \pi \left( \sum _ {i\neq j}(\sigma _ {ij} - \sigma _ {ij} ^ \ast)^2 \leq \frac{2}{3\zeta^4\zeta _ 0^2}\frac{s _ 0\log p}{n} \\;, \\; \sum _ {j=1}^p(\sigma _ {jj} - \sigma _ {jj} ^ \ast)^2 \leq \frac{2}{3\zeta^4\zeta _ 0^2}\frac{p\log p}{n} \right) \quad \because \\; x\leq \alpha, y \leq \gamma \Rightarrow x+y \leq \alpha + \gamma \\\\ 
&\geq \pi\Big(\max _ {i\neq j}(\sigma _ {ij}- \sigma _ {ij} ^ \ast)^2\leq \frac{2}{3\zeta^4\zeta _ 0^2}\frac{s _ 0\log p}{p(p-1)n} \\;,\\; \max _ {1\leq j\leq p}(\sigma _ {jj}- \sigma _ {jj} ^ \ast)^2 \leq \frac{2}{3\zeta^4 \zeta _ 0^2}\frac{\log p}{n} \Big) := \pi(A _ {n, \Sigma_0})
\end{aligned}
\end{equation*}
$$
where $\Sigma_0 = (\sigma_{ij}^\ast)$

We will introduce Weyl's theorem here. (Source : [Wikipedia : Weyl's inequality](https://en.wikipedia.org/wiki/Weyl%27s_inequality))

If $A, B$ are $n\times n$ symmetric (or Hermitian) matrices then $\lambda_k(A) + \lambda_n(B) \leq \lambda_k(A+B) \leq \lambda_k(A) +\lambda_1(B)\quad \forall\\; k=1, \cdots, n$ where $\lambda_1(M)\geq \cdots \geq \lambda_n(M)$ are eigenvalues of symmetric matrix $M\in \Real^{n\times n}$

Here, we will plug in $A= \Sigma_0$ , $B= \Sigma - \Sigma_0$ so that $A+B = \Sigma$ 

Also, we will use two more properties about matrix norm. 

The first one is that for symmetric A, we have $-\norm{A}_2 \leq \lambda(A)\leq \norm{A}_2$ where $\lambda(A)$ is any eigenvalue of $A$. This is because $\lambda(A)^2 = \lambda(A^2) = \lambda(A^T A) \Rightarrow \abs{\lambda(A)} = \sqrt{\lambda(A^T A)}\leq \norm{A}_2 $

The second one is the special case of the Hölder inequality
$$\norm{A} _ 2 \leq \sqrt{\norm{A} _ 1\norm{A} _ \infty}$$
(Source : [Wikipedia : Matrix Norm](https://en.wikipedia.org/wiki/Matrix_norm)) Also, if $A$ is symmetric, then
$$\norm{A} _ 1 = \norm{A} _ \infty$$
since the former is maximum absolute column sum and the latter is maximum absolute row sum. Thus we get $\norm{A}_2 \leq \norm{A}_1$ given $A$ is symmetric.

We want to show that $\Sigma \in A_{n, \Sigma_0} \Rightarrow \Sigma \in \Uc(\zeta)$

Suppose $\Sigma \in A_{n, \Sigma_0}$.
Then we have
$$\norm{\Sigma- \Sigma_0} _ 1 \leq (p-1)\max_{i\neq j}\abs{\sigma _ {ij}- \sigma _ {ij} ^ \ast} + \max_{1\leq j\leq p}\abs{\sigma_{jj} - \sigma_{jj}^\ast}$$

$$\begin{equation*}\begin{aligned}
    \lambda_{min}(\Sigma) &\geq \lambda_{min}(\Sigma _ 0) + \lambda _ {min}(\Sigma-\Sigma _ 0) \quad \because \\; \text{Weyl's thm} \\\\ 
    &\geq \lambda _ {min}(\Sigma _ 0) - \norm{\Sigma- \Sigma _ 0}  _  2 \quad \because \\; -\norm{A}  _  2 \leq \lambda(A) \leq \norm{A} _ 2 \\\\
    &\geq \lambda _ {min}(\Sigma _ 0) - \norm{\Sigma-\Sigma _ 0} _ 1 \quad \because \\; \norm{A} _ 2 \leq \norm{A} _ 1 \\; \text{given } A \text{ is symmetric } \\\\
    &\geq \zeta _ 0^{-1} - \Big\\{ (p-1)\sqrt{ \frac{2}{3\zeta^4\zeta _ 0^2}\frac{s _ 0\log p}{p(p-1)n}}  + \sqrt{\frac{2}{3\zeta^4 \zeta _ 0^2}\frac{\log p}{n} } \Big\\} \quad \because \\; \Sigma _ 0 \in \Uc(s _ 0, \zeta _ 0)\\;,\\; \Sigma \in A _ {n, \Sigma _ 0} \\\\
    &:= \zeta _ 0^{-1} - \star \\\\
    & \\\\
    \lambda _ {max}(\Sigma) &\leq \lambda _ {max}(\Sigma _ 0) + \lambda _ {max}(\Sigma- \Sigma _ 0) \\\\
    &\leq \lambda _ {max}(\Sigma _ 0) + \norm{\Sigma - \Sigma _ 0} _ 2 \\\\
    &\leq \lambda _ {max}(\Sigma _ 0) + \norm{\Sigma - \Sigma _ 0} _ 1 \\\\
    &\leq \zeta _ 0 + \Big\\{ (p-1)\sqrt{ \frac{2}{3\zeta^4\zeta _ 0^2}\frac{s _ 0\log p}{p(p-1)n}}  + \sqrt{\frac{2}{3\zeta^4 \zeta _ 0^2}\frac{\log p}{n} } \Big\\} \\\\
    &= \zeta _ 0 + \star 
\end{aligned}\end{equation*}$$


We shall claim that $\star \rightarrow 0$ as $n\rightarrow \infty$

$$\star \leq \sqrt{\frac{2}{3\zeta^4\zeta_0^2}} \sqrt{\frac{(s_0+1)\log p}{n}} \leq \sqrt{\frac{2}{3\zeta^4\zeta_0^2}} \sqrt{\frac{(s_0+p)\log p}{n}} \rightarrow 0 \quad \because \\; \eps_n \rightarrow 0 $$

Thus, combining the fact that $\zeta_0 < \zeta$ and $\star \rightarrow 0$, we get $$\lambda_{min}(\Sigma)\geq \zeta_0^{-1} - \star > \zeta^{-1} \quad \text{and} \quad \lambda_{max}(\Sigma)\leq \zeta_0 + \star < \zeta \quad \text{for all suff. large } n$$

Hence, we have shown that $\Sigma \in A_{n, \Sigma_0} \Rightarrow \Sigma \in \Uc(\zeta)$ as desired.

Using above, we get $\pi(A_{n, \Sigma_0})\geq \pi^u(A_{n, \Sigma_0})$ since $$\pi(A_{n, \Sigma_0}) = \frac{\pi^u(A_{n, \Sigma_0})\text{I}(\Sigma\in \Uc(\zeta))}{\pi^u(\Sigma\in \Uc(\zeta))} = \frac{\pi^u(A_{n, \Sigma_0})}{\pi^u(\Sigma\in \Uc(\zeta))} \geq \pi^u(A_{n, \Sigma_0}) \quad \because \\; \pi^u(\Sigma \in \Uc(\zeta))\leq 1$$

Here, we shall briefly check what we have already shown. 
$$\pi(B _ {\eps _ n}) \geq \pi\Big(\norm{\Sigma- \Sigma _ 0} _ F^2 \leq \frac{2}{3\zeta^4\zeta _ 0^2}\eps _ n^2\Big) \geq \pi(A _ {n, \Sigma _ 0})\geq \pi^u(A _ {n, \Sigma _ 0}) $$

Hence, from now on, our goal is to prove that $\pi^u(A_{n, \Sigma_0})\geq \exp(-Cn\eps_n^2)$

Note that
$$
\begin{equation*}\begin{aligned}
\pi ^ u(A _ {n, \Sigma _ 0}) &= \pi ^ u\Big(\max _ {i\neq j}(\sigma _ {ij}- \sigma _ {ij} ^ \ast) ^ 2\leq \frac{2}{3\zeta ^ 4\zeta _ 0 ^ 2}\frac{s _ 0\log p}{p(p-1)n} \\;,\\; \max _ {1\leq j\leq p}(\sigma _ {jj}- \sigma _ {jj} ^ \ast) ^ 2 \leq \frac{2}{3\zeta ^ 4 \zeta _ 0 ^ 2}\frac{\log p}{n}\Big) \\\\
&= \pi ^ u\Big(\max _ {i\neq j}(\sigma _ {ij}- \sigma _ {ij} ^ \ast) ^ 2\leq \frac{2}{3\zeta ^ 4\zeta _ 0 ^ 2}\frac{s _ 0\log p}{p(p-1)n}\Big) \times \pi ^ u\Big(\max _ {1\leq j\leq p}(\sigma _ {jj}- \sigma _ {jj} ^ \ast) ^ 2 \leq \frac{2}{3\zeta ^ 4 \zeta _ 0 ^ 2}\frac{\log p}{n} \Big) \\\\
&= \prod _ {i < j} \pi ^ u\Big((\sigma _ {ij}- \sigma _ {ij} ^ \ast) ^ 2\leq \frac{2}{3\zeta ^ 4\zeta _ 0 ^ 2}\frac{s _ 0\log p}{p(p-1)n} \Big) \times \prod _ {j=1} ^ p \pi ^ u\Big((\sigma _ {jj}- \sigma _ {jj} ^ \ast) ^ 2 \leq \frac{2}{3\zeta ^ 4 \zeta _ 0 ^ 2}\frac{\log p}{n} \Big)
\end{aligned}\end{equation*}
$$

This is because all elements of $\Sigma$ are independent to each other given unconstrained setting. 

Observe $$
\prod _ {j=1}^p \pi^u\Big((\sigma_{jj}- \sigma_{jj}^\ast)^2 \leq \frac{2}{3\zeta^4 \zeta_0^2}\frac{\log p}{n} \Big)
$$ first. We want to find a lower bound of this term.

$$\begin{equation*}\begin{aligned}
&\prod _ {j=1} ^ p \pi ^ u\Big((\sigma _ {jj}- \sigma _ {jj} ^ \ast) ^ 2 \leq \frac{2}{3\zeta ^ 4 \zeta _ 0 ^ 2}\frac{\log p}{n} \Big) = \prod _ {j=1} ^ p \pi ^ u\Big(\abs{\sigma _ {jj}- \sigma _ {jj} ^ \ast} \leq \sqrt\psi \Big) \quad \text{where} \\; \psi:= \frac{2}{3\zeta ^ 4 \zeta _ 0 ^ 2}\frac{\log p}{n} \\\\
&= \prod _ {j=1} ^ p \pi ^ u(\sigma _ {jj} ^ \ast -\sqrt \psi \leq \sigma _ {jj}\leq \sigma _ {jj} ^ \ast + \sqrt \psi ) \quad \because \\; \psi \rightarrow 0 \\; \text{so that}\\; \sigma _ {jj} ^ \ast - \sqrt \psi \geq 0 \\; \text{for all suff. large } n \\\\ 
&\text{Note that } \sigma _ {jj}\sim \Gamma(1, \lambda/2)\\; \text{and}\\; \pi ^ u(\sigma _ {jj}) = \frac{\lambda}{2}\exp(-\frac{\lambda}{2}\sigma _ {jj}) \\; \text{is decreasing function } \\\\
&\geq \prod _ {j=1} ^ p 2\sqrt\psi \,  \pi ^ u(\sigma _ {jj} ^ \ast + \sqrt \psi) = \prod _ {j=1} ^ p 2\sqrt\psi \frac \lambda 2 \exp(-\frac \lambda 2 (\sigma _ {jj} ^ \ast + \sqrt \psi )) = \prod _ {j=1} ^ p \sqrt\psi \lambda  \exp(-\frac \lambda 2 (\sigma _ {jj} ^ \ast + \sqrt \psi )) \\\\
&\geq \prod _ {j=1} ^ p \sqrt\psi \lambda  \exp(-\frac \lambda 2 (\zeta _ 0 + \sqrt \psi )) \quad \because \\; \sigma _ {jj} ^ \ast \leq \lambda _ {max}(\Sigma _ 0)\leq \zeta _ 0 \quad \text{due to energy boundedness} \\\\
&= \Big\\{\sqrt\psi \lambda  \exp(-\frac \lambda 2 (\zeta _ 0 + \sqrt \psi ))  \Big\\} ^ p
\end{aligned}\end{equation*}$$

Using a condition $\log p/\zeta ^ 4 \zeta _ 0 ^ 4 \leq n$ , we have $\lambda\sqrt \psi \leq \lambda \zeta _ 0$ since
$$\lambda \sqrt{\psi} = \lambda \sqrt{\frac{2}{3\zeta ^ 4 \zeta_0^2}\frac{\log p}{n}} = \lambda \zeta_0 \sqrt{\frac{2}{3\zeta^4 \zeta_0^4}\frac{\log p}{n}} \leq \lambda \zeta_0 \sqrt{\frac 23}\leq \lambda \zeta_0$$

Hence, we can proceed the above inequality as the following :
$$\begin{equation*}\begin{aligned}
    &\prod_{j=1}^p \pi^u\Big((\sigma_{jj}- \sigma_{jj}^\ast)^2 \leq \frac{2}{3\zeta^4 \zeta_0^2}\frac{\log p}{n} \Big) \geq \Big\\{\sqrt\psi \lambda  \exp(-\frac \lambda 2 (\zeta_0 + \sqrt \psi ))  \Big\\}^p \\\\
    &= \exp(p\log \lambda \sqrt \psi)\exp\Big(- \frac \lambda 2 p \zeta_0 - \frac \lambda 2 p \sqrt \psi \Big) \\\\
    &\geq \exp(p\log \lambda \sqrt \psi)\exp\Big(- \frac \lambda 2 p \zeta_0 - \frac \lambda 2 p \zeta_0 \Big)  \quad \because \\; \lambda\sqrt \psi \leq \lambda \zeta_0 \\\\
    &= \exp\Big(- p\lambda \zeta_0 - p \log \frac{1}{\lambda\sqrt{\psi}}\Big) \\\\
    &\geq \exp\Big(-p \log p - p \log \frac{1}{\lambda\sqrt{\psi}} \Big) \quad \because \\; \lambda < \log p / \zeta_0 \\; \text{by assumption}
\end{aligned}\end{equation*}$$

Here, we shall claim that $1/\sqrt \psi \leq \zeta^3 p^{1/2\beta}$ for all sufficiently large $n$

$$\begin{equation*}\begin{aligned}
    1/\sqrt \psi &= \sqrt{\frac 32}\zeta_0 \zeta^2 \sqrt{\frac{n}{\log p}} \\\\
    &< \sqrt{\frac 32}\zeta^3 \sqrt{\frac{n}{\log p}} \quad \because \zeta_0 < \zeta \\; \text{by assumption} \\\\ 
    &\leq \sqrt{\frac 32}\zeta^3 Cp^{1/2\beta} \frac{1}{\sqrt{\log p}} \quad \because \\; p \asymp n^\beta \,,\, n^\beta\leq Cp \text{ for some } C>0 \text{ by assumption } \\\\
    &\leq \zeta^3p^{1/2\beta} \quad \because \\; p \\; \text{gets large enough to attain} \\; \sqrt{3/2}C/\sqrt{\log p}<1
\end{aligned}\end{equation*}$$

We will complete our process of finding lower bound of $$\prod_{j=1}^p \pi^u\Big((\sigma_{jj}- \sigma_{jj}^\ast)^2 \leq \frac{2}{3\zeta^4 \zeta_0^2}\frac{\log p}{n} \Big)$$ as the below. 

$$\begin{equation*}\begin{aligned}
    &\prod_{j=1}^p \pi^u\Big((\sigma_{jj}- \sigma_{jj}^\ast)^2 \leq \frac{2}{3\zeta^4 \zeta_0^2}\frac{\log p}{n} \Big) \geq \exp\Big(-p \log p - p \log \frac{1}{\lambda\sqrt{\psi}} \Big) \\\\
    &\geq \exp\Big(-p\log p - p\log \frac{\zeta^3 p^{1/2\beta}}{\lambda} \Big) \quad \because \\; 1/\sqrt \psi \leq \zeta^3 p^{1/2\beta} \\; \text{for all sufficiently large } n \\\\
    &\geq \exp\Big(-p \log p - p (1 + \frac 34 + \frac{1}{2\beta})\log p\Big) \quad \because \\; p^{-1}<\lambda \\; \text{and} \\; \zeta^4\leq p \quad \text{by assumption} \\\\
    &\geq \exp\Big(-(3+\frac{1}{2\beta})p \log p\Big)
\end{aligned}\end{equation*}$$

Hence we have
$$
\begin{equation} \label{diagonal ineq}
    \prod_{j=1}^p \pi^u\Big((\sigma_{jj}- \sigma_{jj}^\ast)^2 \leq \frac{2}{3\zeta^4 \zeta_0^2}\frac{\log p}{n} \Big) \geq \exp\Big(-(3+\frac{1}{2\beta})p \log p\Big)
\end{equation}
$$
for sufficiently large $n$.

Next, we shall find a lower bound of
$$\prod_{i < j} \pi^u\Big((\sigma_{ij}- \sigma_{ij}^\ast)^2\leq \frac{2}{3\zeta^4\zeta_0^2}\frac{s_0\log p}{p(p-1)n} \Big).$$
Note that it can be decomposed as the following.

$$\begin{equation*}\begin{aligned}
    &\prod_{i < j} \pi^u\Big((\sigma_{ij}- \sigma_{ij}^\ast)^2\leq \frac{2}{3\zeta^4\zeta_0^2}\frac{s_0\log p}{p(p-1)n} \Big) = \prod_{i<j} \pi^u(\abs{\sigma_{ij} - \sigma_{ij}^\ast}\leq \sqrt \phi)  \quad \text{where} \\; \phi = \frac{2}{3\zeta^4\zeta_0^2}\frac{s_0\log p}{p(p-1)n} \\\\
    &= \prod_{(i,j)\in s(\Sigma_0)}\pi^u(\abs{\sigma_{ij} - \sigma_{ij}^\ast}\leq \sqrt \phi) \times \prod_{(i,j) \notin s(\Sigma_0)\, ,\, i<j}\pi^u(\abs{\sigma_{ij}}\leq \sqrt \phi) 
\end{aligned}\end{equation*}$$

Before finding the lower bound of those two terms above, recall the tight bound of marginal prior density of off diagonal $\sigma_{ij}$ of covariance matrix.
$$\begin{equation*}\begin{aligned}
    \pi^u(\sigma_{ij}) &< \frac{1}{\tau \sqrt{2\pi^3}}\log\Big(1+\frac{2\tau^2}{\sigma_{ij}^2} \Big) \\\\
    \pi^u(\sigma_{ij}) &> \frac{1}{2\tau \sqrt{2\pi^3}}\log\Big(1+\frac{4\tau^2}{\sigma_{ij}^2} \Big) 
\end{aligned}\end{equation*}$$

We will deal with
$$\prod_{(i,j) \notin s(\Sigma_0)\, ,\, i<j}\pi^u(\abs{\sigma_{ij}}\leq \sqrt \phi)$$
first. 

$$\begin{equation*}\begin{aligned}
    &\pi^u(\abs{\sigma_{ij}}>\sqrt \phi) = 2\pi^u(\sigma_{ij}>\sqrt \phi) = \int_{\sqrt\phi}^\infty \pi^u(\sigma_{ij})\, d\sigma_{ij}\\\\
    &\leq 2\int_{\sqrt\phi}^\infty \frac{1}{\tau \sqrt{2\pi^3}}\log\Big(1+\frac{2\tau^2}{\sigma_{ij}^2} \Big) \,d\sigma_{ij} \quad \because \\; \text{upper bound of marginal prior density of $\sigma_{ij}$} \\\\
    &\leq 2\int_{\sqrt \phi}^\infty \frac{1}{\tau \sqrt{2\pi^3}} \frac{2\tau^2}{\sigma_{ij}^2} \, d\sigma_{ij} \quad \because \\; \log(1+x)\leq x \quad \forall \\; x>-1 \quad \text{by supporting line lemma}\\
    &= 2\tau \sqrt{\frac{2}{\pi^3}} \int_{\sqrt{\phi}}^\infty \frac{1}{\sigma_{ij}^2}\, d\sigma_{ij} = 2\tau \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}} \\\\
    &\\\\
    &\prod_{(i,j) \notin s(\Sigma_0)\, ,\, i<j}\pi^u(\abs{\sigma_{ij}}\leq \sqrt \phi) = \prod_{(i,j) \notin s(\Sigma_0)\, ,\, i<j} \big(1 - \pi^u(\abs{\sigma_{ij}}>\sqrt{\phi})\big) \\\\
    &\geq \prod_{(i,j) \notin s(\Sigma_0)\, ,\, i<j} \Big(1-  2\tau \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}} \Big) \quad \because \\; \pi^u(\abs{\sigma_{ij}}>\sqrt \phi) \leq 2\tau \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}}  \\\\
    &\geq \Big(1-  2\tau \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}} \Big)^{p^2} \\\\
    &\geq \exp\Big(-4\tau \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}} \Big)^{p^2} \quad \because \\; \log(1-x)\geq -2x \quad \text{when}\\; 0\leq x\leq 1/2 \\\\
    &\text{Note that} \\; \tau/\sqrt{\phi} \\; \text{is small enough when $n$ is sufficiently large} \quad \because\\; (p^2\sqrt{n})^{-1}\lesssim \tau \lesssim (p^2\sqrt{n})^{-1}\sqrt{s_0\log p} \\\\
    &\tau/\sqrt{\phi} \leq C \frac{1}{p^2}\sqrt{\frac{s_0\log p}{n}}\sqrt{\frac{p(p-1)n}{s_0\log p}}\frac{2}{3\zeta_0^2\zeta^4} \leq \tilde C \frac{1}{p} \rightarrow 0
\end{aligned}\end{equation*}$$

We can proceed the inequality as the following.
$$\begin{equation*}\begin{aligned}
    &\prod_{(i,j) \notin s(\Sigma_0)\, ,\, i<j}\pi^u(\abs{\sigma_{ij}}\leq \sqrt \phi) \geq \exp\Big(-4\tau \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}} \Big)^{p^2} \\\\
    &= \exp\Big(-4\tau p^2 \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}} \Big) \\\\
    &\geq \exp\Big(-4\sqrt{\frac{2}{\pi^3}}\tilde C p^2\frac 1p \Big) \quad \because \\; \tau/\sqrt{\phi}\leq \tilde C \frac 1p \quad \text{by above} \\\\
    &= \exp(-Cp) \quad \text{for some } C>0
\end{aligned}\end{equation*}$$

Thus we have
$$
\begin{equation} \label{offdiagonal zeros ineq}
\prod _ {(i,j) \notin s(\Sigma _ 0)\\, ,\\, i < j } \pi ^ u(\abs{\sigma _ {ij}}\leq \sqrt \phi) \geq \exp\Big(-4\tau p ^ 2 \sqrt{\frac{2}{\pi^3}} \frac{1}{\sqrt{\phi}} \Big) \geq \exp(- Cp)
\end{equation}
$$
for sufficiently large $n$.

Finally, we shall find the lower bound of 
$$\prod _ {(i,j)\in s(\Sigma _ 0)}\pi^u(\abs{\sigma_{ij} - \sigma_{ij}^\ast}\leq \sqrt \phi)$$

Recall that marginal prior density of off diagonal $\sigma_{ij}$ given as $\pi^u(\sigma_{ij})$ is decreasing function with respect to $\abs{\sigma_{ij}}$ since 
$$ \pi^u(\sigma_{ij})= \int_0^\infty \frac{1}{\sqrt{2\pi^3 \tau^2}} \exp\Big(-\frac{\sigma_{ij}^2}{2\tau^2} u \Big) \frac{1}{1+u} \, du $$
Also, note that since $\phi = \frac{2}{3\zeta^4\zeta_0^2}\frac{s_0\log p}{p(p-1)n}\rightarrow 0 $ as $n$ tends to sufficiently large, we can write 
$$\begin{equation*}
    (\abs{\sigma_{ij}- \sigma_{ij}^\ast}\leq \sqrt\phi) = (\sigma_{ij}^\ast - \sqrt\phi \leq \sigma_{ij} \leq \sigma_{ij}^\ast + \sqrt \phi) \begin{cases}\subset (0, \infty) & \\; \text{if} \quad \sigma_{ij}^\ast >0 \\\\
    \subset (-\infty, 0) &\\; \text{if} \quad \sigma_{ij}^\ast <0
    \end{cases}
\end{equation*}$$
for sufficiently large $n$, since $\sigma_{ij}^\ast \neq 0 \\; \text{due to} \\;  (i, j)\in s(\Sigma_0)$

Therefore, we have the following inequality.
$$\begin{equation*}\begin{aligned}
    \pi^u(\abs{\sigma_{ij} - \sigma_{ij}^\ast}\leq \sqrt \phi) \begin{cases}
    \geq 2\sqrt{\phi} \, \pi^u(\sigma_{ij}^\ast + \sqrt{\phi}) & \quad \text{if} \quad \sigma_{ij}^\ast >0 \\\\
    \geq 2\sqrt{\phi} \, \pi^u(\sigma_{ij}^\ast - \sqrt{\phi}) & \quad \text{if} \quad \sigma_{ij}^\ast <0 
    \end{cases}
\end{aligned}\end{equation*}$$

Combining three facts, we can yield $\abs{\sigma_{ij}^\ast} \leq \zeta_0$ for all $i\neq j$. Those facts are given as the following.

1. The largest entry in magnitude of positive definite matrix lies on the diagonal (Source : Gockenbagh Linear Algebra Lemma 386)
1. Energy boundedness : $\lambda_n \norm{x}_2^2 \leq x^T A x \leq \lambda_1 \norm{x}_2^2$ if symmetric $A$ has $\text{spec}(A) = \\{\lambda_1\geq \cdots \geq \lambda_n\\}$
1. $\lambda_{max}(\Sigma_0)\leq \zeta_0$ by $\Sigma_0 \in \Uc(s_0, \zeta_0)$ assumption

Hence we have
$$\begin{equation*}\begin{aligned}
    \abs{\sigma_{ij}^\ast} \leq \max_k{\abs{\sigma_{kk}^\ast}} = \max_k \sigma_{kk}^\ast \leq \lambda_{max}(\Sigma_0) \leq \zeta_0
\end{aligned}\end{equation*}$$
and what follow this are 
$$\begin{equation*}\begin{aligned}
    &\sigma_{ij}^\ast + \sqrt{\phi} \leq 2\zeta_0 \quad \text{if} \quad \sigma_{ij}^\ast >0 \\\\
    &\abs{\sigma_{ij}^\ast - \sqrt{\phi}} \leq 2\zeta_0 \quad \text{if} \quad \sigma_{ij}^\ast < 0
\end{aligned}\end{equation*}$$

Using this, we get $\pi^u(\abs{\sigma_{ij} - \sigma_{ij}^\ast}\leq \sqrt \phi) \geq 2\sqrt{\phi}\, \pi^u(2\zeta_0)$

$$\begin{equation*}\begin{aligned}
    &\prod_{(i,j)\in s(\Sigma_0)}\pi^u(\abs{\sigma_{ij} - \sigma_{ij}^\ast}\leq \sqrt \phi) \geq \Big(2\sqrt{\phi}\, \pi^u(2\zeta_0) \Big)^{s_0} \geq \Big(\sqrt{\phi}\, \pi^u(2\zeta_0) \Big)^{s_0} \geq \Bigg(\pi^u(2\zeta_0)\sqrt{\frac{2s_0\log p}{3\zeta^4 \zeta_0^2 p^2 n}} \Bigg)^{s_0} \\\\
    &= \exp\Bigg(s_0\log \pi^u(2\zeta_0) + \frac 12 s_0 \log \frac{2s_0\log p}{3\zeta^4\zeta_0^2p^2 n} \Bigg) \\\\
    &\geq \exp\Big(s_0 \log \pi^u(2\zeta_0) + \frac 12 s_0 \log \frac 23\frac{1}{\zeta^2p^2n} \Big) \quad \because\\; \zeta^2 \zeta_0^2 \leq s_0\log p \quad \text{by assumption}
\end{aligned}\end{equation*}$$

Note that by taking advantage of Lemma 4, we can write
$\pi^u(2\zeta_0)\geq \frac{1}{\sqrt{2\pi^3}}\frac{\tau}{4\zeta_0^2} $
Of course, we should show that $\tau/2\zeta_0$ is sufficiently small to justify the use of Lemma 4. Since $\zeta_0$ is fixed, we shall show that $\tau \rightarrow 0$ as $n\rightarrow \infty$

$$\begin{equation*}\begin{aligned}
    \tau & \lesssim \frac{\sqrt{s_0\log p}}{p^2\sqrt{n}}  \quad \text{by assumption} \\\\
    & \lesssim \frac{\sqrt{s_0\log s_0}}{p^2\sqrt{n}} \quad \because p \lesssim s_0 \\\\ 
    &\leq  \frac{s_0}{p^2\sqrt{n}} \leq \frac{1}{\sqrt{n}} \quad \because \\; \log s_0 \leq s_0 \leq p^2 
\end{aligned}\end{equation*}$$

Thus $\tau \lesssim \frac{1}{\sqrt{n}}$ so that $\tau / \zeta_0$ is sufficiently small as $n$ gets sufficiently large.

Combining with the condition $\tau \gtrsim 1/\sqrt{n}p^2$ , we get 
$\pi^u(2\zeta_0)\geq \frac{1}{\sqrt{2\pi^3}}\frac{\tau}{4\zeta_0^2} \gtrsim \frac{1}{\sqrt{n}p^2} $

Now we shall proceed our target inequality.

$$\begin{equation*}\begin{aligned}
    &\prod_{(i,j)\in s(\Sigma_0)}\pi^u(\abs{\sigma_{ij} - \sigma_{ij}^\ast}\leq \sqrt \phi)\\
    &\geq \exp\Big(s_0 \log \pi^u(2\zeta_0) + \frac 12 s_0 \log \frac 23\frac{1}{\zeta^2p^2n} \Big)  \\\\
    &\geq \exp\Big(s_0 \log \Big(\frac{\tilde C}{\sqrt{n}p^2} \Big) + \frac 12 s_0 \log \frac 23\frac{1}{\zeta^2p^2n} \Big) \\\\
    &= \exp\Big(\frac 12 s_0 \log \Big(\frac{\tilde C^2}{np^4} \Big) + \frac 12 s_0 \log \frac 23\frac{1}{\zeta^2p^2n} \Big) \\\\
    &= \exp \Big(\frac 12 s_0 \log \big(\frac{2\tilde{C}^2}{3\zeta^2p^6n^2} \big)\Big)
\end{aligned}\end{equation*}$$

Here, we gonna use two inequalities

1. $C^\ast p^{-2/\beta} \leq n^{-2}$ for some $C^\ast>0$
1. $1/\zeta^2 \geq 1/p$

The first one comes from $p\asymp n^\beta$ so that $n^\beta \leq \tilde C^{\ast} p$ and the second one comes from $\zeta^4\leq p$ and $1<\zeta_0 < \zeta$. Using those inequalities, we get

$$\begin{equation*}\begin{aligned}
&\prod _ {(i,j)\in s(\Sigma _ 0)}\pi^u(\abs{\sigma _ {ij} - \sigma _ {ij}^\ast}\leq \sqrt \phi)\\\\
&\geq \exp \Big(\frac 12 s _ 0 \log \big(\frac{2\tilde{C}^2}{3\zeta^2p^6n^2} \big)\Big) \\\\
&\geq \exp\Big(\frac 12 s _ 0 \log\big(\frac 23 \tilde C^2 p^{-1}p^{-6} C^\ast p^{-2/\beta} \big) \Big) \\\\
&= \exp\Big(\frac 12 s _ 0 \log (Cp^{-(7+\frac{2}{\beta})}) \Big) \\\\
&= \exp\Big(-\frac12(7+\frac 2\beta)s _ 0 \log p + \frac 12 s _ 0 \log C \Big) \\\\
&\geq \exp\Big(-\frac12(7+\frac 2\beta)s _ 0 \log p - \frac 12 s _ 0 \log p \Big) \quad \text{for suff. large } n \\\\
&= \exp\Big(-(4+\frac 1\beta)s _ 0\log p \Big)
\end{aligned}\end{equation*}$$

Hence we have

$$
\begin{equation}\label{offdiagonal nonzeros ineq}
\prod _ { (i,j) \in s( \Sigma _ 0 ) } \pi ^ u( \abs{ \sigma  _  {ij} -  \sigma _ {ij}^\ast} \leq  \sqrt  \phi )  \geq  \exp \left (- (4+ \frac 1 \beta ) s _ 0 \log p \right )
\end{equation}
$$

At last, combining \eqref{diagonal ineq}, \eqref{offdiagonal zeros ineq}, and \eqref{offdiagonal nonzeros ineq}, we have

$$\begin{equation*}\begin{aligned}
    &\pi^u(A_{n, \Sigma_0}) \geq \exp\Big(-(3+\frac{1}{2\beta})p \log p\Big) \times \exp(-Cp) \times \exp\Big(-(4+\frac 1\beta)s_0\log p \Big)  \\\\
    &\geq \exp\Big(-(3+\frac{1}{2\beta})p \log p\Big) \times \exp(-Cp) \times \exp\Big(-(4+\frac 1\beta)s_0\log p \Big) \times \exp(Cp) \times \exp(-p\log p) \\\\
    &= \exp\Big(-(4+\frac{1}{2\beta})p \log p\Big) \times \exp\Big(-(4+\frac 1\beta)s_0\log p \Big) \geq \exp\Big(-(4+\frac 1\beta)(p+s_0)\log p\Big) \\\\
    &= \exp\Big(-(4+\frac 1\beta)n\eps_n^2 \Big)
\end{aligned}\end{equation*}$$

Since we have already shown that 

$$\pi(B _ {\eps _ n}) \geq \pi\Big(\norm{\Sigma- \Sigma _ 0} _ F^2 \leq \frac{2}{3\zeta^4\zeta _ 0^2}\eps _ n^2\Big) \geq \pi(A _ {n, \Sigma _ 0})\geq \pi^u(A _ {n, \Sigma _ 0})$$

we can conclude that 

$$\pi(B_{\eps_n}) \geq \exp\Big(-(4+\frac 1\beta) n\eps_n^2 \Big) $$
</div>