---
title: "Theorem 2 "
author: "이경원, 정진욱"
date: 2022-07-27
weight: 6
---

# 모형

다음의 모형을 생각하자. 

$$
\begin{equation}\label{eqn-model}
    X_1, \cdots, X_n | \Sigma \sim N(0, \Sigma)
\end{equation}
$$

양의 정수 $s_0$와 실수 $\zeta_0 > 1$에 대해 다음과 같은 모수공간을 생각한다. 
\begin{equation}
U(s_0, \zeta_0) = \\{ \Sigma \in C_p: s(\Sigma) \leq s_0,~ \zeta_0^{-1} \leq \lambda_{\min}(\Sigma) \leq \lambda_{\max}(\Sigma) \leq \zeta_0 \\}
\end{equation}
여기서 $s(\Sigma)$는 행렬 $\Sigma$의 0이 아닌 비대각성분의 개수를 의미한다. 

# 정리 

<div class="theorem">
모형 <a href="#eqn-model">(1)</a>과 양의 정수 $s_0$ 와 실수 $\zeta_0 > 1$에 대해 $\Sigma_0 \in \mathcal{U}(s_0, \zeta_0)$이라 하자. $s_0^2 (\log p)^3 = O(p^2n)$이면 작은 상수 $\epsilon > 0$에 대해 다음이 성립한다.

\begin{equation}
    \inf_{\hat{\Sigma}}\sup_{\Sigma_0 \in \mathcal{U}(s_0, \zeta_0)} \mathbb{E}_0 \lVert\hat{\Sigma} - \Sigma_0 \rVert_F^2 \gtrsim \frac{(p+s_0) \log p}{n} I\left(3p < s_0 < p^{3/2 - \epsilon/2}\right) + \frac{p+s_0}{n}
\end{equation}
</div>

<div class="remark">

이 정리는 공분산 추정의 minimax lower bound를 알려준다. 

논문의 Theorem 1에서는 사후수렴속도가 $\dfrac{(p+s_0) \log p}{n}$임을 보였는데, 이는 $3p < s_0 < p^{3/2 - \epsilon/2}$일 때 베이즈 추론이 minimax 이고, 그렇지 않은 경우에도 nearly minimax $(\log p)$ 임을 의미한다.

</div>

# 증명

<div class="proof">
다음의 두 항목을 증명하면 된다. 

* $3p < s_0 < p^{3/2 - \epsilon/2}$인 경우,
    $$\begin{equation}\label{eqn-13} \inf_{\hat{\Sigma}}\sup_{\Sigma_0 \in B_1} \mathbb{E}_0 \lVert\hat{\Sigma} - \Sigma_0 \rVert_F^2 \gtrsim \frac{s_0 \log p}{n} \end{equation}$$
    이 성립하는 $B_1 \subset \mathcal{U}(s_0, \zeta_0)$가 존재함을 보인다. 
* 나머지 경우,
    $$\begin{equation}\label{eqn-14} \inf_{\hat{\Sigma}}\sup_{\Sigma_0 \in B_2} \mathbb{E}_0 \lVert\hat{\Sigma} - \Sigma_0 \rVert_F^2 \gtrsim \frac{s_0 + p}{n} \end{equation}$$
    이 성립하는 $B_2 \subset \mathcal{U}(s_0, \zeta_0)$가 존재함을 보인다. 

먼저 첫 항목을 보이자. $\nu = \sqrt{\epsilon/4}$에 대해 $r = \lfloor p/2 \rfloor,~\epsilon_{np} = \nu \sqrt{\log p / n}$이라 하자. $A_m(u)$를 $m$번째 행과 열이 $u$의 값을 갖고, 나머지에서 모두 0인 대칭행렬이라 하자. 즉, 
\begin{equation}
    (A_m(u))_{ij} = \begin{cases} u & i = m \text{ or} j = m \\\\ 0 & \text{otherwise} \end{cases}
\end{equation}

이라 하자. 이제, 다음과 같이 $B_1$을 정의한다.

\begin{equation}
    B_1 := \left\\{ \Sigma(\theta) : \Sigma(\theta) = I_p + \epsilon_{np} \sum_{m=1}^r \gamma_m A_m(\lambda_m),~\theta = (\gamma, \lambda) \in \Theta \right\\}
\end{equation}

여기서 $\gamma = (\gamma_1,\cdots, \gamma_r) \in \Gamma = [0, 1]^r$, $\lambda = (\lambda_1, \cdots, \lambda_r)^T \in \Lambda \subset \mathbb{R}^{r \times p}$,
\begin{equation}
\begin{aligned}
\Lambda= \bigg\\{ \lambda = (\lambda_{ij}) : &\lambda_{mi} \in \\{0, 1\\},~ \lVert \lambda_m \rVert_0 = k,~\sum_{i=1}^{p-r} \lambda_{mi} = 0, \\\\
&m \in \\{ 1, \cdots, r \\},~ \text{ satisfying } \max_{1 \leq i \leq p} \sum_{m=1}^r \lambda_{mi} \leq 2k
\bigg\\},
\end{aligned}
\end{equation}
$k = \lceil c_{np} / 2 \rceil - 1,~ c_{np} = \lceil s_0 / p \rceil$, $\Theta = \Gamma \times \Lambda$이다. 

이제 $B_1 \subset \mathcal{U}(s_0, \zeta_0)$과 <a href="#eqn-13">논문의 식 (13)</a>이 성립함을 보이면 된다. 

먼저, 임의의 $\zeta_0 >1$과 충분히 큰 $n$에 대해 $\Sigma(\theta) \in B_1$의 가장 큰 고유치가 $\zeta_0$보다 작다는 것은 다음과 같이 보일 수 있다.
\begin{equation}
    \lVert \Sigma(\theta) \rVert \leq \lVert \Sigma(\theta) \rVert_1 \leq 1 + 2 k \epsilon_{np} \leq 1 + c_{np} \nu \sqrt{\log p / n} \leq \zeta_0
\end{equation}
마지막 부등호는 가정 $s_0^2 (\log p)^3 = O(p^2 n)$에 의해 성립한다.

다음으로, 임의의 $\zeta_0 >1$과 충분히 큰 $n$에 대해
\begin{equation}
    2k\epsilon_{np} \leq c_{np} \nu \sqrt{\log p / n}\leq \left( 1 + \frac{s_0}{p} \right) \nu \sqrt{\log p / n} \leq 1 - \zeta_0^{-1}
\end{equation}
가 성립하므로 $\Sigma(\theta) - \zeta_0^{-1} I_p$는 대각지배(diagonally dominant)행렬이고, 대칭이며 모든 성분이 0보다 크거나 같아 양의 준정부호 행렬이다[^diagdom]. 따라서, $\Sigma(\theta)$의 가장 작은 고유치는 $\zeta_0^{-1}$보다 크다. 

마지막으로 $\Sigma(\theta)$의 비대각성분은 모두 $A_m$들에 의해서만 나타나므로
\begin{equation}
s(\Sigma(\theta)) \leq 2 kp \leq s_0
\end{equation}
에서 $B_1 \subset \mathcal{U}(s_0, \zeta_0)$를 얻는다. 

이제 <a href="#eqn-13">논문의 식 (13)</a>이 성립함을 보이자. 이를 위해, 다음의 보조정리를 소개한다. 

<div class="lemma" style="border: solid; padding: 30px; margin: 10px">(Lemma 3 of T. Tony Cai, Harrison H. Zhou (2012))

> OPTIMAL RATES OF CONVERGENCE FOR SPARSE COVARIANCE MATRIX ESTIMATION

For any $s > 0$ and any estimator $T$ of $\psi(\theta)$ based on an observation from the experiment $\\{ P_\theta,~\theta \in \Theta \\}$,

\begin{equation}
\max_{\theta \in \Theta} 2^s \mathbb{E}_\theta d^s(T, \psi(\theta)) \geq  \alpha \frac{r}{2} \min _ {1 \leq i \leq r} \lVert \overline{\mathbb{P}} _ {i, 0} \wedge \overline{\mathbb{P}} _ {i, 1} \rVert,
\end{equation}

where $\overline{\mathbb{P}} _ {i, a}$ is the mixture distribution over all $P_\theta$ with $\gamma_i(\theta)$ ﬁxed to be a while all other components of $\theta$ vary over all possible values, i.e.,
\begin{equation}
\overline{\mathbb{P}} _ {i, a} = \frac{1}{2^{r-1} |\Lambda|} \sum_{\theta \in \Theta_{i, a}} P_\theta,
\end{equation}

for $\Theta_{i, a} = \\{ \theta \in \Theta: \gamma_i(\theta) = a \\}$,

\begin{equation}
\lVert \mathbb{P} \wedge \mathbb{Q} \rVert = \int (p \wedge q) d \mu,
\end{equation}
for probability measures $\mathbb{P}$ and $\mathbb{Q}$ which have densities $p$ and $q$ respectively, $\mathbb{E} _ \theta$ is expectation with respect to $[X_1, \cdots, X_n | \theta]$, $H(x, y)$ is the Hamming distance defined as
$$H(x, y) = \sum_{j=1}^r |x_i - y_i|, \quad x, y \in \\{ 0, 1 \\}^r.$$

\begin{equation}
\alpha = \min_{(\theta, \theta^\prime) : H(\gamma(\theta), \gamma(\theta^\prime)) \geq 1} d^s(\psi(\theta), \psi(\theta^\prime)) / H(\gamma(\theta), \gamma(\theta^\prime))
\end{equation}

<div class="proof">
최댓값은 평균보다 크거나 같으므로 

\begin{equation}
\begin{aligned}
\max_{\theta \in \Theta} 2^s \mathbb{E}_\theta d^s(T, \psi(\theta)) 
&\geq \frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta} 2^s \mathbb{E} _ \theta d^s(T, \psi(\theta)) \\\\
&= \frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta} \mathbb{E} _ \theta (2 d(T, \psi(\theta)))^s
\end{aligned}
\end{equation}

$\hat{\theta} := \arg\min d^s(T, \psi(\theta))$라 하면 (유일하지 않다면 적당히 하나를 잡으면 된다.)

\begin{equation}
\begin{aligned}
\mathbb{E} _ \theta (2 d(T, \psi(\theta)))^s
&\geq \mathbb{E} _ \theta (d(T, \psi(\theta)) + d(T, \psi(\hat{\theta})) )^s \\\\
&\geq \mathbb{E} _ \theta (d(\psi(\hat{\theta}), \psi(\theta)))^s
\end{aligned}
\end{equation}
를 얻는다. 마지막 부등식에서 삼각부등식을 사용하였다. 

정리하면 다음을 얻는다. 

\begin{equation}
\begin{aligned}
\max_{\theta \in \Theta} 2^s \mathbb{E}_\theta d^s(T, \psi(\theta)) 
&\geq \frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta}  \mathbb{E} _ \theta (d(\psi(\hat{\theta}), \psi(\theta)))^s \\\\
&\geq \frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta} \mathbb{E} _ \theta \left[\frac{(d(\psi(\hat{\theta}), \psi(\theta)))^s}{H(\gamma(\theta), \gamma(\theta^\prime)) \vee 1 } H(\gamma(\theta), \gamma(\theta^\prime))\right] \\\\
&\geq \alpha \frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta} \mathbb{E} _ \theta \left[  H(\gamma(\theta), \gamma(\theta^\prime))\right]
\end{aligned}
\end{equation}

이제 
\begin{equation}
\frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta} \mathbb{E} _ \theta \left[  H(\gamma(\theta), \gamma(\theta^\prime))\right] \geq \frac{r}{2} \min _ {1 \leq i \leq r} \lVert \overline{\mathbb{P}} _ {i, 0} \wedge \overline{\mathbb{P}} _ {i, 1} \rVert
\end{equation}
을 보이면 원하는 결과를 얻는다. 

\begin{equation}
\begin{aligned}
&\frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta} \mathbb{E} _ \theta \left[  H(\gamma(\theta), \gamma(\theta^\prime))\right] \\\\
&= \frac{1}{2^r |\Lambda| } \sum _ {\theta \in \Theta} \sum_{i=1}^r \mathbb{E} _ \theta \left[  |\gamma_i(\theta) - \gamma_i(\theta^\prime))| \right] \\\\ 
&= \sum_{i=1}^r \frac{1}{2^r |\Lambda| } \sum_{\rho \in \Gamma} \sum _ {\theta : \gamma(\theta) = \rho} \mathbb{E} _ \theta \left[  |\gamma_i(\theta) - \gamma_i(\theta^\prime))| \right] \\\\ 
&= \frac{1}{2} \sum_{i=1}^r \left\\{ \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 0} \sum _ {\theta : \gamma(\theta) = \rho} \mathbb{E} _ \theta \left[  |\gamma_i(\theta) - \gamma_i(\theta^\prime))| \right] + \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 1} \sum _ {\theta : \gamma(\theta) = \rho} \mathbb{E} _ \theta \left[  |\gamma_i(\theta) - \gamma_i(\theta^\prime))| \right] \right\\} \\\\ 
&= \frac{1}{2} \sum_{i=1}^r \left\\{ \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 0} \sum _ {\theta : \gamma(\theta) = \rho} \mathbb{E} _ \theta \left[  |\gamma_i(\theta^\prime))| \right] + \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 1} \sum _ {\theta : \gamma(\theta) = \rho} \mathbb{E} _ \theta \left[  |1 - \gamma_i(\theta^\prime))| \right] \right\\} \\\\ 
&= \frac{1}{2} \sum_{i=1}^r \left\\{ \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 0} \sum _ {\theta : \gamma(\theta) = \rho} \int _ \theta^\prime  \gamma_i(\theta^\prime)) d\mathbb{P}_ {\theta^\prime} + \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 1} \sum _ {\theta : \gamma(\theta) = \rho} \int _ \theta^\prime  1 - \gamma_i(\theta^\prime)) d\mathbb{P}_ {\theta^\prime} \right\\} \\\\ 
&= \frac{1}{2} \sum_{i=1}^r \left\\{  \int _ \theta^\prime  \gamma_i(\theta^\prime)) \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 0} \sum _ {\theta : \gamma(\theta) = \rho} d\mathbb{P}_ {\theta^\prime} + \int _ \theta^\prime  \left( 1 - \gamma_i(\theta^\prime)) \right) \frac{1}{2^{r-1} |\Lambda| } \sum_{\rho_i = 1} \sum _ {\theta : \gamma(\theta) = \rho} d\mathbb{P}_ {\theta^\prime} \right\\} \\\\ 
&= \frac{1}{2} \sum_{i=1}^r \left\\{  \int _ \theta^\prime  \gamma_i(\theta^\prime)) d \overline{\mathbb{P}} _ {i, 0} + \int _ \theta^\prime  \left( 1 - \gamma_i(\theta^\prime)) \right) d \overline{\mathbb{P}} _ {i, 0} \right\\} \\\\ 
&\geq \frac{1}{2} \sum_{i=1}^r \int d \left[ \overline{\mathbb{P}} _ {i, 0} \wedge \overline{\mathbb{P}} _ {i, 1} \right] \\\\
&\geq \frac{r}{2} \min _ {1 \leq i \leq r} \lVert \overline{\mathbb{P}} _ {i, 0} \wedge \overline{\mathbb{P}} _ {i, 1} \rVert
\end{aligned}
\end{equation}

증명이 끝났다. 
</div>

</div>

위의 보조정리에 $s=2$를 대입하면 다음의 부등식을 얻는다. 
\begin{equation}
    \inf_{\hat{\Sigma}} \max_{\theta \in \Theta} 2^2 \mathbb{E}_\theta \lVert \hat{\Sigma}- \Sigma(\theta) \rVert_F^2 \geq \alpha \frac{r}{2} \min _ {1 \leq i \leq r} \lVert \overline{\mathbb{P}} _ {i, 0} \wedge \overline{\mathbb{P}} _ {i, 1} \rVert
\end{equation} 

여기서 
\begin{equation}
\alpha = \min_{(\theta, \theta^\prime) : H(\gamma(\theta), \gamma(\theta^\prime)) \geq 1} \lVert \Sigma(\theta) - \Sigma(\theta^\prime) \rVert_F^2 / H(\gamma(\theta), \gamma(\theta^\prime))
\end{equation}
이다. 

이때 임의의 $\theta,~\theta^\prime \in \Theta$에 대해 
\begin{equation}
\begin{aligned}
\lVert \Sigma(\theta) - \Sigma(\theta^\prime) \rVert_F^2 
&= \epsilon_{np}^2 \left\lVert  \sum_{m=1}^r \gamma_m(\theta) A_m(\lambda_m(\theta)) -  \sum_{m=1}^r \gamma_m(\theta^\prime) A_m(\lambda_m(\theta^\prime)) \right\rVert_F^2 \\\\
& \geq 2k\epsilon_{np}^2 H(\gamma(\theta), \gamma(\theta^\prime))
\end{aligned}
\end{equation}
이므로 $k$와 $r$의 정의($r = \lfloor p/2 \rfloor,~k = \lceil c_{np} / 2 \rceil - 1,~ c_{np} = \lceil s_0 / p \rceil$)로부터 다음을 얻는다. 

\begin{equation}
\alpha r \geq 2k\epsilon_{np}^2 r \geq \nu^2 \left( \frac{1}{2} - \frac{p}{s_0} \right) \frac{s_0 \log p}{n} \asymp \frac{s_0 \log p}{n}
\end{equation}
<!-- 두번째 부등식에서 $s_0 > 3p$임을 이용하였다.  -->

이제, 다음을 만족하는 적당한 상수 $c_1 > 0$이 존재함을 보이면 증명이 끝난다. 
\begin{equation}
\min _ {1 \leq i \leq r} \lVert \overline{\mathbb{P}} _ {i, 0} \wedge \overline{\mathbb{P}} _ {i, 1} \rVert \geq c_1
\end{equation}

[^diagdom]: A Hermitian diagonally dominant matrix $A$ with real non-negative diagonal entries is positive semidefinite. From https://en.wikipedia.org/wiki/Diagonally_dominant_matrix#Applications_and_properties

</div>

TBA