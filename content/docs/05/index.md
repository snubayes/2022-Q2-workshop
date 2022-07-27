---
title: "Lemma 4 and Theorem 5"
author: "장태영"
date: 2022-07-27
weight: 5
---

<style>
body {
  counter-set: theorem 4;
}
</style>

<div class="theorem">(Lower bound for $\pi(B_n)$)
만일 다음의 조건들이 만족되면

1. $\Sigma_0 \in \mathcal{U}(s_0, \zeta_0)$ with $\zeta_0 < \zeta$
2. $p \asymp n^\beta$ with $0 < \beta< 1$
3. $\zeta^\ast \leq p$
4. $\zeta^2 \zeta_0^2 \leq s_0 \log p$
5. $n \geq \max\\{ 1/ \zeta_0^\ast,~ s_0 / (1 - \zeta_0 / \zeta)^2 \\} \log p / \zeta^\ast$
6. $p^{-1} < \lambda < \log p / \zeta_0$ 
7. $a=b=1/2$
8. $(p^2 \sqrt{n})^{-1} \lesssim \tau \lesssim (p^2 \sqrt{n})^{-1} \sqrt{s_0 \log p}$
9. (From page 5, Theorem 1) $(p + s_0) \log p = o(n)$
10. (From page 5) $p = O(s_n)$

다음이 성립한다

\begin{equation}
\pi(B _ {\epsilon _ n})  \geq \exp \left\\{ - \left( s + \frac{1}{\beta} \right) n \epsilon _ n^2 \right \\}.
\end{equation}

여기서 $\epsilon_n = \sqrt{\frac{(p+s_0) \log p}{n}},$
$$B_{\epsilon} = \\{ f_{\Sigma} : \Sigma \in \mathcal{C} _ p, ~ K(f_{\Sigma_0}, f_{\Sigma}) < \epsilon^2,~ V(f_{\Sigma_0}, f_{\Sigma}) < \epsilon^2 \\}$$
이다. 
</div>

이 정리의 증명을 위해서 Lemma 3와 Lemma 4가 필요하다. 