---
title: "공분산 행렬의 베이즈 추론"
author: "조성일 교수님"
date: 2022-07-25
weight: 2
---

# 모형 

$$X_1, \cdots, X_n | \Sigma \sim N(0, \Sigma)$$ 
$$\Sigma = (\sigma_{ij})_{i,j=1}^p > 0$$ 

## 사전분포 

### 비대각원소

비대각원소에는 정규혼합 사전분포를 가정한다. 

$$\pi(\sigma_{ij}) = (1 - \pi) N(\sigma_{ij}; 0, v_0^2) + \pi N(\sigma_{ij}; 0, v_1^2), \quad i \neq j$$

### 대각원소

대각원소에는 축소 사전분포를 가정한다. 

$$\pi(\sigma_{ii}) = Exp(\lambda/2)$$

### 공분산 행렬 

이를 정리하면 다음과 같다. 

$$
\begin{equation}
\begin{aligned}
    \pi(\Sigma) 
    &= \left[ c(\theta) \right]^{-1} \prod_{i < j} \left[ (1 - \pi) N(\sigma_{ij}; 0, v_0^2) + \pi N(\sigma_{ij}; 0, v_1^2) \right]\\\\
    &\qquad \times \prod_{i=1}^p \text{Exp}\left(\sigma_{ii}; \frac{\lambda}{2}\right) I(\Sigma \in M^+)
\end{aligned}
\end{equation}
$$

여기서 $c(\theta)$는 정규화 상수이며 모수 $\theta = \\{v_0, v_1, \pi, \lambda \\}$이다. 

## 변수 

$Z = (z_{ij})_{i < j} \in  \\{0, 1 \\}^{\frac{p(p-1)}{2}}$

## 계층모형

이 모형은 다음과 같은 계층모형으로 나타낼 수 있다. 

\begin{equation}
\begin{aligned}
\pi(\Sigma|Z, \theta)
&= \left[ c(\theta) \right]^{-1} \prod_{i < j} N(\sigma_{ij}; 0, v_{z_{ij}}^2 ) \\\\
    &\qquad \times \prod_{i=1}^p \text{Exp}\left(\sigma_{ii}; \frac{\lambda}{2}\right) I(\Sigma \in M^+) \\\\
\pi(Z|\theta) 
&= \left[ c(\theta) \right]^{-1}c(z, v_0, v_1, \lambda) \prod_{i < j } \pi^{z_{ij}} (1-\pi)^{1-z_{ij}}.
\end{aligned}
\end{equation}

# Block-Giibs Sampler 

위의 계층모형에서 표본을 추출하는 깁스 샘플러는 다음과 같다. 

\begin{equation}
\begin{aligned}
\pi(\Sigma, Z|X_1, \cdots, X_n)
&\propto \prod_{i=1}^n N_p(X_i; 0, \Sigma) \\\\
    &\quad \times \prod_{i < j} N(\sigma_{ij}; 0, v_{z_{ij}}^2 )  \pi^{z_{ij}} (1-\pi)^{1-z_{ij}} \\\\
    &\quad \times \prod_{i=1}^p \text{Exp}\left(\sigma_{ii}; \frac{\lambda}{2}\right) I(\Sigma \in M^+) \\\\
    &\propto |\Sigma|^{-\frac{n}{2}} \exp\left[ -\frac{1}{2} tr(S\Sigma^{-1}) \right] \\\\
    &\quad \times \prod _ {i < j} \left\\{ \exp\left( - \frac{\sigma_{ij}^2}{2v_{z_{ij}}^2} \right) \right \\} \pi^{z_{ij}} (1-\pi)^{1-z_{ij}} \\\\
    &\quad \times \prod_{i=1}^p \exp\left( -\frac{\lambda}{2} \sigma_{ii} \right)
\end{aligned}
\end{equation}

$$P(z_{ij} = 1|\Sigma, X_1, \cdots, X_n) = \frac{\pi N(\sigma_{ij}; 0, v_1^2)}{\pi N(\sigma_{ij} 0, v_1^2) + (1- \pi) N(\sigma_{ij}; 0, v_0^2)}$$

$V = (v_{z_{ij}}^2)$은 $p \times p$ 대칭 행렬, $v_{z_{ij}}^2 = 0$ for $i = j$. 

$$\Sigma = \begin{pmatrix} \Sigma_{11} & \sigma_{12} \\\\ \sigma_{12}^\prime & \sigma_{22} \end{pmatrix}$$

$$ S = X^\prime X = \begin{pmatrix} S_{11} & s_{12} \\\\ s_{12}^\prime & s_{22} \end{pmatrix}$$

$$V = \begin{pmatrix} V_{11} & v_{12} \\\\ v_{12}^\prime & 0 \end{pmatrix}$$

이제 다음과 같은 변환을 생각하자. 
$$(\sigma_{12},~\sigma_{22}) \mapsto (u=\sigma_{12},~v=\sigma_{22} - \sigma_{12}^\prime \Sigma_{11}^{-1} \sigma_{12})$$

이 변환의 야코비안은 다음과 같이 계산된다. 

\begin{equation}
    |J| = \left|\begin{pmatrix} 1 & 0 \\\\ -2\Sigma_{11}^{-1} \sigma_{12} & 1 \end{pmatrix}\right| = 1
\end{equation}

그러면, 공분산의 역행렬은
\begin{equation}
\begin{aligned}
    \Sigma^{-1}
    &= \begin{pmatrix} \Sigma_{11}^{-1} + \Sigma_{11}^{-1} \sigma_{12} \left( \sigma_{22} - \sigma_{21}^\prime \Sigma_{11}^{-1} \sigma_{12}\right) \sigma_{12}^\prime \Sigma_{11}^{-1} & ... \\\\ -(\sigma_{22} - \sigma_{12}^\prime \Sigma_{11}^{-1}\sigma_{12})^{-1}\sigma_{12}^\prime \Sigma_{11}^{-1} & \left(\sigma_{22} - \sigma_{12}^\prime \Sigma_{11}^{-1} \sigma_{12}\right)^{-1} \end{pmatrix} \\\\
    &= \begin{pmatrix} \Sigma_{11}^{-1} + \Sigma_{11}^{-1} u u^\prime \Sigma_{11}^{-1} v^{-1} & -\Sigma_{11}^{-1} uv^{-1} \\\\ -u^\prime \Sigma_{11}^{-1} v^{-1} & v^{-1} \end{pmatrix}
\end{aligned}
\end{equation}

따라서, 
\begin{equation}
\begin{aligned}
    |\Sigma| 
    &= |\Sigma_{11}| |\sigma_{22} - \sigma_{12}^\prime \Sigma_{11}^{-1} \sigma_{12} | \\\\
    &= |\Sigma_{11}| (\sigma_{22} - \sigma_{12}^\prime \Sigma_{11}^{-1}\sigma_{12}) \\\\
    &\propto v, \\\\ 
    tr(S\Sigma^{-1}) 
    &= tr\left[ \begin{pmatrix} S_{11} & s_{12} \\\\ s_{21}^\prime & s_{22} \end{pmatrix} \begin{pmatrix} \Sigma_{11}^{-1} + \Sigma_{11}^{-1} u u^\prime \Sigma_{11}^{-1} v^{-1} & -\Sigma_{11}^{-1} uv^{-1} \\\\ -u^\prime \Sigma_{11}^{-1} v^{-1} & v^{-1} \end{pmatrix} \right] \\\\
    &\propto u^\prime \Sigma_{11}^{-1} S_{11} \Sigma_{11}^{-1} u v^{-1} -2 s_{12}^\prime \Sigma_{11}^{-1} u v^{-1} + s_22 v^{-1}
\end{aligned}
\end{equation}

또한, 
\begin{equation}
    \prod_{i < j} \exp\left( - \frac{\sigma_j^2}{2 v_{z_{ij}}^2} \right) \propto \exp\left( - \frac{1}{2} u^\prime D^{-1} v \right),
\end{equation}
여기서 $D = diag(v_{12}),~ v = \sigma_{22} - \sigma_{12}^\prime \Sigma_{11}^{-1} \sigma_{12}$이다. 
\begin{equation}
    \prod_{i=1}^p \exp\left(  - \frac{\lambda}{2} \sigma_{ii} \right) \propto \exp\left( - \frac{\lambda}{2} \left( u^\prime \Sigma_{11}^{-1} u + v \right) \right)
\end{equation}
에서, 

\begin{equation}
    \log \pi(u, v | \cdot ) \propto -\frac{1}{2} \left \\{  n \log v + u^\prime \Sigma_{11}^{-1} S_{11} \Sigma_{11}^{-1} u v^{-1} - 2 s_{12}^\prime \Sigma_{11}^{-1} u v^{-1} + s_{22} v^{-1} + u^\prime D^{-1} u + \lambda u^\prime \Sigma_{11}^{-1} u + \lambda v  \right \\},
\end{equation}

\begin{equation}
\begin{gathered}
    \pi(u|v, z, ...) = N \left(  (B+D^{-1})^{-1} w, (B+D^{-1})^{-1} \right), \\\\
    B = \Sigma_{11}^{-1} S_{11} \Sigma_{11}^{-1} v^{-1} + \lambda \Sigma_{11}^{-1}, \\\\
    w = \Sigma_{11}^{-1} s_{12} v^{-1}.
\end{gathered}
\end{equation}

이는 Generalized inverse Gaussian, $GIG(q, a, b)$로, 그 확률밀도함수는 
\begin{equation}
    f(x) = \frac{(a/b)^{q/2}}{2K_g (\sqrt{ab})} \lambda^{p-1}e^{-(ax + b/x)/2}
\end{equation}
로 주어진다. 

즉, $\pi(v|u, z, ...) = GIG\left(1 - n/2, \lambda, u^\prime \Sigma_{11}^{-1} S_{11} \Sigma_{11}^{-1} u  - 2 s_{12}^\prime \Sigma_{11}^{-1} u + s_{22}\right)$이다. 

이 분포들에서 순서대로 표본을 추출하면 된다. 

> 참고: 실제로 이를 구현하면 수치적 오류로 인해 알고리듬이 잘 돌아가지 않는다. 

# 논문의 사전분포 

Armigan?

* $\Sigma = (\sigma_{ij})$
* $\rho = (\rho_{ij})$
* $\pi(\Sigma, \rho) = \prod_{i > j} N\left( \sigma_{ij}; 0, \frac{\rho_{ij}}{1-\rho_{ij}} \tau^2 \right) Beta(\rho_{ij}; a, b) \times \prod_{i=1}^p \text{Exp}\left(\sigma_{ii}; \frac{\lambda}{2} \right)$
* $v = (v_{ij}^2) = \begin{pmatrix} V_{11} & v_{12} \\\\ v_{12}^\prime & 0 \end{pmatrix},~ v_{ij}^2 = \dfrac{\rho_{ij}}{1 - \rho_{ij}} \tau^2$

$\phi_{ij} = \dfrac{\rho_{ij}}{1 - \rho_{ij}}$라 하자. 그러면, 
\begin{equation}
\begin{aligned}
    \sigma_{ij}|\phi_{ij} &\sim N(0, \phi_{ij} \tau^2), \\\\
    \phi_{ij}|\psi_{ij} &\sim \text{Gamma}(a, \psi_{ij}), \\\\
    \psi_{ij} &\sim \text{Gamma}(b, 1)
\end{aligned}
\end{equation}
에서 
\begin{equation}
\begin{aligned}
\pi(\psi_{ij}|\phi_{ij}, ...)
&\propto \psi_{ij}^{b-1}e^{-\psi_{ij}} \phi_{ij}^{a-1} e^{-\psi_{ij} \phi_{ij}} \psi_{ij}^a \\\\
&= \psi_{ij}^{a+b-1} e^{- (\phi_{ij} + 1)\psi_{ij}} \\\\
&= \text{Gamma}(\cdot, a+b, \phi_{ij}+ 1).
\end{aligned}
\end{equation}
\begin{equation}
\begin{aligned}
\pi(\phi_{ij}|...)
&\propto \phi_{ij}^{-\frac{1}{2}}e^{-\frac{\sigma_{ij}^2}{2\phi_{ij} \tau^2}} \phi_{ij}^{a-1} e^{-\psi_{ij} \phi_{ij}} \\\\
&= \phi_{ij}^{a-\frac{1}{2}-1}e^{-\frac{\sigma_{ij}^2}{2\phi_{ij} \tau^2} - \psi_{ij} \phi_{ij}} \\\
&= GIG\left(a - \frac{1}{2},~ 2\psi_{ij},~ \frac{\sigma_{ij}^2}{\tau^2}\right)
\end{aligned}
\end{equation}

## 논문의 의의 

* 축소 사전분포의 사용 
* 이론적 성질의 규명