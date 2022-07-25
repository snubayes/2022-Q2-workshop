---
title: "공분산 행렬 추정 문제"
author: "이재용 교수님"
date: 2022-07-25
weight: 1
---

# 1. 모형 

## 1.1. 다변량 정규 모형 

$\mu \in \mathbb{R}^k,~\Sigma \in \mathbb{R}^{k \times k},~\Sigma > 0$에 대해 다음과 같은 모형을 생각하자[^dim]. 

$$
\begin{equation}
X_1, \cdots, X_n | \mu, \Sigma \stackrel{i.i.d.}{\sim} N(\mu, \Sigma)
\end{equation}
$$

[^dim]: 많은 곳에서는 모수의 차원으로 $p$를 사용하나 여기서는 $k$를 사용한다. 

정밀도 행렬은 $\Omega  = \Sigma^{-1}$로 정의된다.

여기서 $\mu$와 $\Sigma$를 추정하는 것이 문제이다. 

일반적으로는, 평균의 추정보다 공분산 추정이 어려운데, 이는 '양의 정부호 행렬'이라는 제약이 있기 때문이다.

## 1.2. 가능도 

$$
\begin{equation}
\begin{aligned}
    L(\mu, \Sigma) 
    &= \prod_{i=1}^n N(x_i, \mu, \Sigma) \\\\
    &= \prod_{i=1}^n |2\pi \Sigma|^{-1/2} e^{-\frac{1}{2}(x_i -\mu)' \Sigma^{-1} (x_i -\mu)} \\\\
    &\propto  | \Sigma|^{-n/2}  \prod_{i=1}^n e^{-\frac{1}{2}(x_i -\mu)' \Sigma^{-1} (x_i -\mu)} \\\\
    &\propto  | \Sigma|^{-n/2}  \prod_{i=1}^n e^{-\frac{1}{2} tr(\Sigma^{-1}  (x_i -\mu) (x_i -\mu)'} \\\\
    &\propto  | \Sigma|^{-n/2}  \prod_{i=1}^n e^{-\frac{n}{2} tr(\Sigma^{-1}  [S_n + (\bar{x} - \mu)(\bar{x} - \mu)'])}
    \end{aligned}
\end{equation}
$$

여기서 $nS_n + n(\bar{x} - \mu)(\bar{x} - \mu)' = (x_i -\mu) (x_i -\mu)'$이다.

## 1.3. 로그 가능도

$$
\begin{equation}
l(\mu, \Sigma) = C - \frac{n}{2} \log |\Sigma| -\frac{n}{2}tr\left(\Sigma^{-1}  [S_n + (\bar{x} - \mu)(\bar{x} - \mu)']\right)
\end{equation}
$$

# 2. 빈도론 추정 

## 2.1. 최대가능도 추정량 

빈도론자의 추정량은 다음과 같이 주어진다.

$$\hat{\mu}^{MLE} = \bar{x},~\hat{\Sigma} = \frac{1}{n} \sum (x_i - \bar{x})(x_i - \bar{x})' = S_n$$

$\mu =0$임이 알려져 있으면, $\hat{\Sigma}^{MLE} = \frac{1}{n} \sum x_i x_i'$이다. 

# 3. 베이즈 추정

## 3.1. 베이즈 모형

### 3.1.1. 켤레 사전분포 

다음과 같은 켤레사전분포를 생각한다. 

$$
\begin{equation}
\begin{gathered}
\Omega \sim W(\nu_0, B_0^{-1}) \\\\
\mu|\Omega \sim N(\mu_0, \Sigma/\kappa_0)
\end{gathered}
\end{equation}
$$

여기서 $W$는 위사트(Wishart) 분포로 공분산 행렬 $\Sigma$에 대한 사전분포를 고려한다면, 역-위샤트(inverse-Wishart) 사전분포를 고려하면 된다. 

### 3.1.2. 사후분포 

사후분포는 

$\nu_n = \nu_0 + n,~\kappa_n = \kappa_0 + n,~\mu_n = \frac{1}{\kappa_0 +n} (\kappa_0 \mu_0 + n \bar{x}),$
$$B_n = B_0 + n S_n + \frac{n \kappa_0}{n+\kappa_0} (\mu_0 - \bar{x}) (\mu_0 - \bar{x})'$$
를 모수로 갖는 위샤트 분포로 주어진다.

## 3.2. 베이즈 추정량 

위의 사전분포로부터 

$$
\begin{equation}
\begin{gathered}
\hat{\mu}^B = \mu_n \\\\
\hat{\Sigma}^B = \frac{1}{\nu_n - k - 1} B_n
\end{gathered}
\end{equation}
$$

으로 주어진다. 

## 3.3. 제프리스 사전분포 

### 3.3.1. 사전분포 

$$
\begin{equation}\pi(\mu, \Sigma) d\mu d\Sigma \propto |\Sigma|^{-\frac{k+1}{2}} d \mu d\Sigma
\end{equation}$$

### 3.3.2. 사후분포 

$$\mu|\Sigma, \mathbb{X} \sim N\left(\bar{x},~\frac{1}{n}\Sigma\right)$$
$$\Sigma|\mathbb{X} \sim IW_k(k + n,~(n-1)S_n)$$

### 3.3.3. 베이즈 추정량 

$$\hat{\mu}^B = \bar{x}$$
$$\hat{\Sigma}^B = \frac{n-1}{n-k-2} S_n$$ 

## 3.4. 위샤트 분포 

$nu > k-1,~B>0$에 대해 양의 정부호 행렬 $W$가 위샤트 분포 $W_k(\nu, B)$를 따른다는 것은, 다음을 의미한다. 

$$f(w)dw = \frac{1}{2^{\nu k / 2} |B| \Gamma_k(\nu/2)} |w|^\frac{\nu - k -1}{2} e^{-\frac{1}{2}tr(B^{-1}w)}$$

여기서 $dw = \prod_{i \leq j} dw_{ij}$를 의미한다. 

그러면, $\mathbb{E}[W] = \nu B$이다. 

## 3.5. 역-위샤트 분포

$\Omega \sim IW_k (\nu, A), ~\nu > k-1,~ A >0$이라는 것은 다음을 의미한다. 

$$f(\omega) d\omega = \frac{|A|^\frac{\nu - k -1}{2}}{2^\frac{k(\nu-k-1)}{2} \Gamma_k(\nu/2)} |\omega|^{-\frac{\nu}{2}} e^{-\frac{1}{2} tr(\Omega^{-1} A)}$$ 

다음이 성립한다. 

* $W \sim W_k(\nu, B) \Longleftrightarrow W^{-1} \sim IW_k(\nu+k+1, B^{-1})$.
* $\Omega \sim IW_k(\nu, A) \Rightarrow \mathbb{E}[\Omega] = \frac{1}{\nu - 2k - 2} A, ~ \nu - 2k -2 > 0$.

## 3.6. $\mu = 0$인 정규모형의 예 

모형
$$X_1, \cdots, X_n | \Sigma \stackrel{i.i.d.}{\sim} N_k(0, \Sigma)$$
$$\Omega \sim W_k(\nu_0, B_0^{-1})$$
의 사후분포는
$$
\begin{equation}
\begin{gathered}
    \Omega|\mathbb{X} \sim W_k(\nu_0 + n, (B_0 + nS)^{-1}) \\\\
    \Sigma|\mathbb{X} \sim IW_k(\nu_0 + n, B_0 + nS)
\end{gathered}
\end{equation}
$$

이 모형은 빈도론자들의 공분산 행렬 추정 모형을 그대로 옮긴 것인데, 베이즈주의자들 사이에서도 논란이 있다. 

고정된 $k$에 대해서는 사후분포, 베이즈 추정량들이 좋은 성질을 가짐이 알려져 있다.

우리는 $k$가 변하는 경우를 함께 고려해보고자 한다. 

# 4. 공분산의 사용처 

다음과 같은 분야에서 공분산 추론은 중요한 위상을 갖는다. 

* 주성분 분석(PCA)
* 판별 분석
* 변수들간의 독립성, 조건부 독립성 검정
* 정준상관분석 

# 5. 고차원 모형 

2000년대에 들어서, 고차원 모형에 대한 관심이 급증하였다. 고차원 모형이란, 모수의 차원 $k$가 자료의 크기 $n$과 함께 커지는 경우를 생각한다. 심지어, 다음과 같은 상황을 고려하기도 한다. 
$$ k \stackrel{n \rightarrow \infty}{\longrightarrow} \infty.$$

과거에는 자료의 크기와 관계 없이 고정된 차원을 갖는 모형들을 고려하였다.

20세기 후반, 사람들은 '데이터 많으니 더 큰 모형을 고려할 수 있지 않을까' 하는 생각을 하기 시작했다. 즉, 자료가 커질 때, 모형의 복잡도도 함께 커지는 문제를 고려하였다. 이러한 상황에서는 기존에 알려진 모형의 점근적 성질들이 성립하지 않는 문제들이 발생하였고, 현대의 통계학은 이러한 문제를 해결하는 데 관심을 가지고 있다. 

## 5.1. 고차원 공분산 추정의 어려움 

$n$과 $k$가 동시에 커지면서 다음과 같은 문제가 발생한다. 

1. $\dfrac{k}n$이 클수록, $\lambda_{\max} (S_n) >> \lambda_{\max}(\Sigma)$이고 $\lambda_{\min}(S_n) << \lambda_{\min}(\Sigma)$이다. 
2. (Johnstone & Lu 2009) $S_n$의 고유벡터는 $\Sigma$의 고유벡터로 수렴하지 않는다. 

1번의 문제는 과거에도 널리 알려져 있었으며, 이를 피하기 위한 다양한 가정들이 시도되었다. 최근에는 성김(sparse)가정을 주로 한다. 

# 6. 공분산의 분해 

공분산 행렬의 추정이 어려운 이유는 양의 정부호라는 제약조건 때문이다. 이를 피하기 위해 다음과 같이 공분산을 분해하여 생각하는 방법들이 제안되었다. 

## 6.1. 촐레스키 분해 

촐레스키 분해(Cholesky decomposition)은 공분산 행렬을 다음과 같이 분해한다. 

$$\Sigma =  CC',~ C = (c_{ij})$$

여기서 $C$는 $c_{{ii}} > 0$인 하삼각행렬(lower triangular matrix)이다.

증명:

수학적 귀납법을 사용한다. $k=1$일 때는 자명하다. 
$$\Sigma = \begin{bmatrix} \Sigma_{11} & \sigma_{12}' \\\\ \sigma_{12} & \sigma_{22} \end{bmatrix} = \begin{bmatrix} C_1 & 0 \\\\ x' & y \end{bmatrix} \begin{bmatrix} C_1 & x \\\\ 0' & y \end{bmatrix} $$
을 만족하는 $x,~y$가 존재함을 보이면 된다.

위의 식을 계산해보면,
$$\Sigma = \begin{bmatrix} \Sigma_{11} & \sigma_{12}' \\\\ \sigma_{12} & \sigma_{22} \end{bmatrix} = \begin{bmatrix} C_1 C_1' & C_1 x \\\\ x' C & x'x + y^2  \end{bmatrix}$$
에서 $x = C_1^{-1} \sigma_{12},~ y = \sqrt{\sigma_{22} - x'x}$이다. $\blacksquare$

촐레스키 분해는 간단하지만 직관적으로 통계적인 의미를 갖지 않아 잘 사용되지 않는다.

참고: $\Sigma$가 위샤트 분포를 따르면 $C$의 분포는 발렛 분포를 따른다는 것이 알려져 있다. 

## 6.2. 대각화 정리 (주성분 분석)

$\Sigma = PDP'$와 같이 분해한다. 여기서 $P = [u_1, \cdots, u_k]$인 직교행렬, $D= diag(\lambda_1, \cdots, \lambda_k)$인 대각행렬이다. 

이 분해는 다른 문제(주성분 분석 등)에서는 유용하게 사용되나, 공분산 추정의 문제에서는 잘 사용되지 않는다. 

## 6.3. 수정된 촐레스키 분해 

수정된 촐레스키 분해(modified Cholesky decomposition)는 다음과 같다. 

### 6.3.1. 동기 

모형 
$$
\begin{equation}
X = \begin{pmatrix} X_1 \\\\ \vdots \\\\ X_k \end{pmatrix} \sim N(0, \Sigma)
\end{equation}
$$
에서, 각 성분의 분포를 다음과 같이 나타낼 수 있다.

$$
\begin{equation}
\begin{aligned}
X_1 &= \epsilon_1, \\\\
X_2 &= a_{21} X_1 + \epsilon_2, \\\\
X_3 &= a_{31} X_1 + a_{32} X_2 + \epsilon_3, \\\\ 
&\vdots \\\\
X_k &= a_{k1} X_1 + \cdots + a_{k, k-1} X_{k-1} + \epsilon_k
\end{aligned}
\end{equation}
$$

즉, $X = AX + \epsilon, ~ \epsilon \sim N(0, D)$와 같은 형태로 나타낼 수 있다. 여기서 
$$
\begin{equation}
A = \begin{bmatrix} 0 & 0 & \cdots  & 0 \\\\ a_{21} & 0 & \cdots & 0 \\\\ a_{31} & a_{32} & \cdots & 0 \\\\ \vdots & \vdots & \ddots & \vdots \\\\ a_{k1} & a_{k2} & \cdots & 0  \end{bmatrix}
\end{equation}
$$
이고, 이러한 $A$를 촐레스키 인자라 부른다. 

그러면, 
$$
\begin{equation}
\begin{gathered}
(I - A) X = \epsilon, \\\\
Var((I-A)X) = Var(\epsilon), \\\\
(I-A) \Sigma (I-A)' = D, \\\\
\Sigma = (I-A)^{-1} D (I-A)'^{-1}, \\\\
\Omega = (I-A) D^{-1} (I-A)' 
\end{gathered}
\end{equation}
$$

즉, $\Sigma$를 추정하는 공분산 추정 문제를, $A$와 $D$를 추정하는 선형회귀문제로 바꿀 수 있다. 

수정된 촐레스키 분해는 주로 사용된다. 

# 7. 빈도론 추정 

## 7.1. 벌점함수 방법들 

$$
\begin{equation}
\begin{aligned}
\hat{\Sigma} 
&= \arg\min_{\Sigma} \left[ - l(\Sigma) + \lambda Pen(\Sigma) \right] \\\\ 
&= \arg\min_{\Sigma} \log |\Sigma| + tr(\Sigma^{-1} S_n) + \sum_{i < j} P_\lambda( \sigma_{ij})
\end{aligned}
\end{equation}
$$

주로 사용하는 벌점함수로는 다음이 있다. 

* $P_\lambda(\theta) = \lambda |\theta|$ ($L_1$-penalty, LASSO penalty)
* $P_\lambda(\theta) = \lambda^2 - (|\theta| - \lambda)^2 I(|\theta| < \lambda)$, (hard thresholding)
* $P_\lambda'(\theta) \lambda I(|\theta| \leq \lambda) + \dfrac{(a \lambda - \theta)_+}{a - 1} I(|\theta| > \lambda),~ a>2$

공분산 행렬의 역행렬을 계산하는 것이 비싸기 때문에, 다음과 같은 손실함수를 고려하기도 한다. 

$$
\begin{equation}
\sum_{i, j} (s_{ij} - \sigma_{ij})^2 + \sum_{i < j} P_\lambda( \sigma_{ij})
\end{equation}
$$

혹은, 다음과 같이 정밀도 행렬을 추정하는 문제를 고려하기도 한다. 

$$
\begin{equation}
\hat{\Omega} = \arg\min_{\Omega} - \log |\Omega| + tr(\Omega S_n) + \sum_{i < j} P_\lambda( \omega_{ij})
\end{equation}
$$

## 7.2. Lam & Fan (2009)

빈도론의 대표적인 연구 결과를 소개한다. 

Lam & Fan (2009)은 적당한 벌점함수에 대해 
$$\|\hat{\Sigma} - \Sigma_0\|_F^2 = O_p\left(  \frac{(p_n + s_n) \log p_n}{n} \right)$$
을 보였다. 여기서 $s_n$은 $\Sigma$에서 0이 아닌 비대각원소의 개수, $p_n$은 차원을 의미한다. 

보통, 공분산 행렬의 추정 분제에서는 최적의 수렴속도가 다음과 같이 주어진다. 

$$\frac{\text{0이 아닌 모수의 개수} \times \log(\text{차원})}{n}$$

빈도론자들은 이러한 $\hat{\Sigma}$를 찾는 구체적인 방법들에 대해 관심을 갖는다. 

## 7.3. 성김 가정이 없는 추정 방법들 

최근에는 주로 성김 가정을 하나, 이전에는 어떤 추정 방법들을 제안했나 살펴본다. 

### 7.3.1. Stein (1975)

$S = PDP'$와 같이 나타내자. 적당한 고유치들의 변환 $\Lambda$에 대해 공분산 행렬의 추정량으로 $\hat{\Sigma} = P \Lambda(D) P'$로 제안한다. 

Johnstone의 문제에서 알 수 있듯, 고차원 행렬 문제에서는 고유벡터를 찾는 것도 어렵기 때문에 $P$를 제대로 추정하기 어렵다. 

### 7.3.2. Ledoit & Wolf (2004)

공분산 행렬의 추정량으로 축소 추정량 $\hat{\Sigma}  = \rho_1 S + \rho_2 I$를 제안하였다. 


## 7.4. 성김 가정 하에서의 빈도론 추정 방법들

> Bickel & Levina (2008)
> Thresholidng, Tapering, Banding 

* Thresholding estimator는 다음과 같이 주어진다.

$$\begin{equation}
\begin{aligned}
\hat{\Sigma}
&= (\hat{\sigma_{ij}}) \\\\ \hat{\sigma}_{ij} \\\\
&= \begin{cases} s_{ij} I\left(|s_{ij}| > c \sqrt{ \frac{\log p}{n}}\right) & (i \neq j) \\\\ s_{ij} & (i=j) \end{cases} 
\end{aligned}
\end{equation}$$

* banding estimator는 공분산 행렬이 대각성분 근처에서만 0이 아닌 성분을 갖는 추정량을 제안한다. 

$$\begin{equation} \hat{\Sigma} = B_k(S) = (s_{ij} I(|i-j| \leq k)) \end{equation}$$

* tapering estimator는 공분산 행렬이 대각성분에서 멀어질수록 0에 가까워지는 추정량을 제안한다. 

\begin{equation}
\hat{\Sigma} = T_k(S) = ( w_{ij}^{(k)} s_{ij}), \quad w_{ij}^{(k)} = \begin{cases} 1, & |i-j| \leq \frac{k}{2} \\\\ 2 - \frac{|i-j|}{k/2}, & \frac{k}{2} < |i-j| \leq k \\\\ 0, & \text{o.w.}  \end{cases}
\end{equation}

# 8. 베이즈 방법 

모수공간에 제약이 있을 때, 사전분포를 부여하는 것이 어렵다. 

## 8.1. 그래프 모형 

### 8.1.1. 그래프 

$G = (V, E)$라 하자. $\Omega = (\omega_{ij})$와 같이 나타낼 때, $V = \{ 1,2, \cdots, k \}$,
$$E \subset V \times V = \{ (i, j) : Cov(X_i, X_j) \neq 0 \text{ or } w_{ij} \neq 0 \}$$
으로 정의한다. 

### 8.1.2. G-Wishart 분포

$\Omega \sim W_G(b, D), b > 2, D > 0$는 다음을 의미한다. 

$$\pi(\Omega | G) = \frac{1}{I_G(b, D)} | \Omega|^{\frac{b-2}{2}} e^{-\frac{1}{2} tr(D \Omega)} I(\Omega \in M_G^+)$$

여기서 $M_G^+ = \{ \Omega : \Omega > 0,~ \omega_{ij} \neq 0 \Leftrightarrow (i, j)\in E \}$이다. 

사후분포는 $\Omega | \mathbb{X}, G \sim W_G(b+n, D+S)$로 주어진다. 

이 분포는 단순히 위샤트 분포에 제약조건을 추가한 것이라 직관적이나, 정규화 상수 $I_G(b, D)$의 계산이 사실상 불가능하다.

이러한 문제로 분해가능(decomposible)이라는 가정을 추가한다. 분해가능하지 않을 때는 수치적으로 정규화 상수를 계산하나 차원이 커질 때 계산이 거의 불가능하다. 

### 8.1.3. 그래프 모형 

이와 같은 모형을 그래프 모형(graphical model)이라 한다. 

그래프 모형에서 $w_{ij} = 0$은 $X_i \perp X_j|X_{~(i,j)}$, 즉, 조건부 독립성을 의미한다. 

참고: $\sigma_{ij} = 0$은 $X_i \perp X_j$, 즉, 주변 독립성을 의미한다. 

### 8.1.4. G-inverse-Wishart 분포 

$\Sigma \sim IW_G(\delta, U)$는 다음과 같은 밀도함수를 갖는다. 

$$\pi(\Sigma | G) = \frac{1}{I_G(\delta, U)} | \Sigma|^{-\frac{\delta+2}{2}} e^{-\frac{1}{2} tr(\Sigma^{-1} U)} I(\Sigma \in M_G^+)$$

### 8.1.5. 성질 

그래프 모형은 사전분포와 사후분포가 잘 정의된다는 장점을 갖는다. 

## 8.2. 축소 사전분포 

### 8.2.1. Wang (2015)

공분산 행렬의 각 성분에 다음과 같은 분포를 가정하는 모형도 있다. 

$$
\begin{equation}
\begin{aligned}
\sigma_{ij} &\sim w \delta_0 + (1-w) Normal, \\\\
\sigma_{ii} & \sim Exp 
\end{aligned}
\end{equation}
$$

우리가 이번에 볼 논문은 이를 연속형 분포로 확장한 것이다. 

## 8.3. 사후처리 사후분포 (이광민)

post-processed posterior

> 사이비 베이즈(?)

전체 모수 공간을 $\Theta^\ast$, 원하는 모수 공간을 $\Theta \subset \Theta^\ast$라 하자. 

사전분포 $\pi^\ast$가 계산이 쉬운 사후분포 $\pi^\ast(\cdot | \mathbb{X})$를 갖는다고 하자. 

사후처리 사후분포는 다음과 같은 요소들로 구성된다. 

* 사후 처리 함수 $f: \Theta^\ast \rightarrow \Theta$
* 사후처리 사후분포 $[f(\theta^\ast)|\theta^\ast \sim \pi^\ast(\cdot | \mathbb{X}_n)] = \pi(\cdot | \mathbf{X})$ 

이 방법은 이론적 정당성을 더 확보해야 한다. 

## 8.4. Berger, Sun & Song (2020)

다음과 같은 사전분포를 고려한다. 

$$
\begin{equation}
\pi(\Sigma|a, b, H) \propto \frac{1}{|\Sigma|^a \left[ \prod_{i < j} (\lambda_i - \lambda_j) \right]^b } e^{-\frac{1}{2} tr(\Sigma^{-1} H)}
\end{equation}
$$

여기서 $\lambda_1 > \lambda_2 > \cdots > \lambda_k$는 $\Sigma$의 고유치이다.

사후분포의 성질은 아직 규명되지 않았다.