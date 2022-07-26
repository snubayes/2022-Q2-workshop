---
title: "정리 3 and 보조정리 2"
author: "신창수"
date: 2022-07-26
weight: 3
---

# 정리 3

<div class="theorem">
The upper bound of the packing number

If ${\zeta}^4 \leq p, \ p \asymp n^{\beta} \ for\ some\ 0 \leq \beta \leq 1,\ s _ n = c _ 1n \epsilon _ n^2/lnp,\ L _ n = c _ 2n \epsilon _ n^2\ \ and\ \delta _ n=\epsilon _ n/\zeta^3\ for\ some\ constants\ c _ 1>1\ and c _ 2>0, we\ have$

$$
\begin{align*}
    lnD(\epsilon _ n,P _ n,d) \ \leq \ (12+1/\beta)c _ 1n\epsilon _ n^2
\end{align*}
$$

</div>

$\newcommand{\defeq}{\overset{\text{\tiny def}}{=}}$

<div class="proof">

이 정리는 Lemma1의 세 가지 조건 중 첫 번째 조건이 성립함을 보이는 정리이다. 증명에 앞서 각각의 용어들에 대해 정의하자. $D(\epsilon _ n,P _ n,d)$ 은 $P _ n$ 안에서, 각각의 쌍이 이루는 거리들의 거리가 $\epsilon _ n$보다 크거나 같은 점들의 최대 개수로 정의한다. 또한,
$$P _ n=\\{f _ \Sigma \ :\ |s(\Sigma,\delta _ n)|\leq s _ n, \ \zeta^{-1} \leq \lambda _ {min}(\Sigma) \leq \lambda _ {max}(\Sigma) \leq \zeta,||\Sigma|| _ {max} \leq L _ n  \\}$$

$$\mathcal{U}(\delta _ n,s _ n,L _ n,\zeta)=\\{\Sigma \in \mathcal{c} _ p : |s(\Sigma,\delta _ n)|\leq s _ n,\zeta^{-1} \leq \lambda _ {min}(\Sigma) \leq \lambda _ {max} (\Sigma) \leq \zeta,||\Sigma|| _ {max} \leq L _ n \\}$$

$s _ 0$ : 공분산 행렬의 비대각 원소 중 0이 아닌 원소들의 상한
$$\epsilon _ n \defeq \\{\dfrac{(p+s _ 0)lnp}{n}\\}^{\dfrac{1}{2}}$$

라고 정의하자. 최종적으로는, $lnD(\epsilon _ n,\mathcal{P} _ n,d)\ \leq \ (12+1/\beta)c _ 1n\epsilon _ n^2$ 임을 보일 건데, 그 전에 먼저 lemma3 소개하자.

<div class="lemma" style="border: solid; padding: 30px; margin: 10px">[Lemma3](Lemma A.1 in [6])

If  $P _ {\Omega _ {k}}$ is the density of $N _ p(0,\Omega _ {k}^{-1})$, k=1,2, then for all $\Omega _ k \in \mu _ {0}^+$, k=1,2, and $d _ i$, i=1,2,…,p, eigenvalues of $A=\Sigma _ 1^{-1/2}\Sigma _ 2\Sigma _ 1^{-1/2}$, we have that for some $\delta > 0$ and constant $c _ o>0$,

1. $c _ 0^{-1}||\Sigma _ 1-\Sigma _ 2|| _ 2^2\leq \sum _ {i=1}^{p}|d _ i-1|^2\leq c _ o||\Sigma _ 1-\Sigma _ 2|| _ 2^2$
2. $h(p _ {\Omega _ {1}},p _ {\Omega _ {2}})<\delta$ implies $\max\limits _ i|d _ i-1|<1$ and $||\Omega _ 1-\Omega _ 2|| _ 2 \leq c _ oh^2(p _ {\Omega _ {1}},p _ {\Omega _ {2}})$
3. $h^2(p _ {\Omega _ {1}},p _ {\Omega _ {2}}) \leq c _ o||\Omega _ 1-\Omega _ 2|| _ 2^2$

여기서, $h()$는 hellinger distance 로, 우리 논문에서 $d()$와 같음
</div>

따라서, lemma3-(3)에 의해,
$d(f _ {\Sigma _ 1},f _ {\Sigma _ 2}) \leq c\zeta||\Omega _ 1-\Omega _ 2|| _ F$ 임을 알 수 있고, 여기서 $\Omega$는 $\Sigma^{-1}$ 이다.

위 lemma3-(3)과 우리 논문의 lemma5를 통해, $\Omega _ 1=\Sigma _ 1^{-1}\Sigma _ 1\Sigma _ 1^{-1}$ 임을 이용하면,
$d(f _ {\Sigma _ 1},f _ {\Sigma _ 2}) \leq C\zeta^3||\Sigma _ 1-\Sigma _ 2|| _ F$ (1) 식을 얻을 수 있다.

$\epsilon$-packing의 정의에 의해, $d(f _ {\Sigma _ {i}},f _ {\Sigma _ {j}})\geq \epsilon _ n$ 으로부터, (1) 식과 결합하여, $||\Sigma _ 1-\Sigma _ 2|| _ F \geq \dfrac{\epsilon _ n}{C\zeta^3}$ 의 결과를 얻을 수 있다. 따라서, 집합을 $\mathcal{P} _ n$ 에서, $\mathcal{U}(\delta _ n,s _ n,L _ n,\zeta)$ 으로 바꾸고, 거리를 Frobenius norm 으로 바꾸어서 위의 결과를 적용하자. 여기서 격자 개념을 생각해보면, 격자의 대각선 부분들까지 고려해주어야 한다. 따라서, 
$$lnD(\epsilon _ n,P _ n,d) \leq lnD(\dfrac{\epsilon _ n}{C\zeta^3},\mathcal{U}(\delta _ n,s _ n,L _ n,\zeta),||\cdot|| _ F)
\leq ln\left\\{\left(\dfrac{L _ n\sqrt{p+j}}{\dfrac{\epsilon _ n}{C\zeta^3}}\right)^p \displaystyle \sum\limits _ {j=1}^{s _ n}\left(\dfrac{\sqrt{p+j}\dfrac{1}{\sqrt{2}}L _ n}{\dfrac{\epsilon _ n}{C\zeta^3}}\right)^{j}{\dfrac{p}{2} \displaystyle \choose j} \right\\}$$
여기서, $\sqrt{p+j}$는 격자의 대각선을 고려해준 항이고, $\dfrac{1}{\sqrt{2}}$는 Frobenius norm에서, symmetric term들의 중복을 고려해 준 값이다.
한편, $\left(\dfrac{L _ n\sqrt{p+j}}{\dfrac{\epsilon _ n}{C\zeta^3}}\right)$에서 $j\leq s _ n \leq p^2$임을 통해, j를 p에 대한 부등식으로 적절히 바꾸어주면,
$\dfrac{L _ n\sqrt{p+j}}{\dfrac{\epsilon _ n}{C\zeta^3}}\leq \dfrac{2p\zeta^3L _ n}{\epsilon _ n}$ 을 얻을 수 있다.
따라서, 
$$
\begin{equation*}
\begin{gathered}
ln\left\\{\left(\dfrac{L _ n\sqrt{p+j}}{\dfrac{\epsilon _ n}{C\zeta^3}}\right)^p \displaystyle \sum\limits _ {j=1}^{s _ n}\left(\dfrac{\sqrt{p+j}\dfrac{1}{\sqrt{2}}L _ n}{\dfrac{\epsilon _ n}{C\zeta^3}}\right)^{j}{\dfrac{p}{2} \choose j} \right \\} \\\\
=ln\left\\{\left(\dfrac{2pC\zeta^3L _ n}{\epsilon _ n}\right)^p \displaystyle \sum\limits _ {j=1}^{s _ n}\left(\dfrac{\sqrt{2}C\zeta^3L _ np}{\epsilon _ n}\right)^j{\dfrac{p}{2} \choose j}\right\\} \\\\
= ln\left[((2p)^p(\sqrt{2}p)^{s _ n})\left(\dfrac{C\zeta^3L _ n}{\epsilon _ n}\right)^p \displaystyle \sum\limits _ {j=1}^{s _ n} \left(\dfrac{C\zeta^3L _ n}{\epsilon _ n}\right)^j{\dfrac{p}{2} \choose j}\right] \\\\
= pln2+plnp+s _ n(\dfrac{1}{2}ln2+lnp)+pln\left(\dfrac{CL _ n\zeta^3}{\epsilon _ n}\right)+ln\left( \displaystyle \sum\limits _ {j=1}^{s _ {n}}\left(\dfrac{2CL _ n\zeta^3}{\epsilon _ n}\right)(\dfrac{p^2}{2})^j\right) \\\\
\leq pln2+plnp+s _ n(\dfrac{1}{2}ln2+lnp)+pln\left(\dfrac{CL _ n\zeta^3}{\epsilon _ n}\right)+s _ {n}ln\left(\dfrac{2CL _ n\zeta^3p^2}{\epsilon _ n}\right) \\\\
\leq pln2+plnp+\dfrac{1}{2}s _ nln2+s _ nlnp+(p+s _ n)ln(2CL _ n)+(p+s _ n)ln\zeta^3+(p+s _ n)ln\dfrac{1}{\epsilon _ n}+2s _ nlnp 
\end{gathered}
\end{equation*}
$$
에서 적절한 상수를 곱해주면,
$$\leq 2(p+s _ n)lnp+(p+s _ n)ln(2CL _ n)+(p+s _ n)ln\zeta^3+(p+s _ n)ln\dfrac{1}{\epsilon _ n}+2s _ nlnp $$
이다. 먼저, $(p+s _ n)ln(2CL _ n)\leq 6s _ nlnp$ 임을 보일건데,
위에서 정의한 $s _ n,\epsilon _ n,L _ n$을 통해, $s _ n=c _ 1(p+s _ 0)$ 이므로, $p+s _ n=(1+c _ 1)p+c _ 1s _ 0$임을 알 수 있다. 또한, $2CL _ n=2c _ 2n\epsilon _ n^2$ 이다.
이를 좌변에 대입하면,
$$
\begin{equation}
\begin{gathered}
(p+s _ n)ln(2CL _ n)=((c _ 1+1)+c _ 1s _ 0)ln2c _ 2n\epsilon _ n^2 \\\\
=((c _ 1+1)p+c _ 1s _ 0)ln2c _ 2(p+s _ 0)lnp
\end{gathered}
\end{equation}
$$
이다. $c _ 1 > 1$ 가정에 의해,
$$((c _ 1+1)+c _ 1s _ 0)ln2c _ 2n\epsilon _ n^2\leq 2c _ 1(p+s _ 0)ln(2c _ 2(p+s _ 0)lnp)$$
이다. 한편, $s _ 0$는 비대각 원소 중 0이 아닌 것들의 개수의 상한이므로 $s _ o\leq p^2$ 임을 알 수 있다.
따라서, 적절한 차수 $p^3$을 통해 $2c _ 2(p+s _ 0)lnp<p^3$ 을 얻을 수 있다. 이를 통해,
$$(p+s _ n)ln(2CL _ n)\leq 2c _ 1(p+s _ 0)lnp^3=6s _ nlnp$$
부등식을 얻을 수 있다.
이를 정리하면,
$$
\begin{equation}
\begin{gathered}
2(p+s _ n)lnp+(p+s _ n)ln(2CL _ n)+(p+s _ n)ln\zeta^3+(p+s _ n)ln\dfrac{1}{\epsilon _ n}+2s _ nlnp \\\\
\leq 2(p+s _ n)lnp+6s _ nlnp+(p+s _ 0)ln\zeta^3+(p+s _ n)ln(\dfrac{1}{\epsilon _ n})+2s _ nlnp
\end{gathered}
\end{equation}
$$
이다. 이제 $(p+s _ 0)ln\zeta^3\leq \dfrac{3}{4}(p+s _ n)lnp$ 임을 보이자.
$$(p+s _ n)ln\zeta^3=\dfrac{3}{4}(p+s _ n)ln\zeta^4$$
인데, 가정에 의해, $\zeta^4\leq p$이므로,
$$(p+s _ n)ln\zeta^3\leq \dfrac{3}{4}(p+s _ n)lnp$$
임을 알 수 있다. 이를 대입하여 정리하면,
$$
\begin{equation}
\begin{gathered}
\leq 2(p+s _ n)lnp+6s _ nlnp+(p+s _ 0)ln\zeta^3+(p+s _ n)ln(\dfrac{1}{\epsilon _ n})+2s _ nlnp \\\\
\leq 2(p+s _ n)lnp+6s _ nlnp+\dfrac{3}{4}(p+s _ n)lnp+(p+s _ n)ln(\dfrac{1}{\epsilon _ n})+2s _ nlnp
\end{gathered}
\end{equation}
$$
이다.
세번째 항에, 앞에서 정의한 $\epsilon _ n$을 대입하여 정리하면,
$$(p+s _ n)ln(\dfrac{1}{\epsilon _ n})=\dfrac{1}{2}(p+s _ n)ln\dfrac{n}{(p+s _ o)lnp}$$
임을 알 수 있고,
우변
$$
\begin{equation}
\begin{gathered}
= \dfrac{1}{2\beta}(p+s _ n)ln\left(\dfrac{n}{(p+s _ o)lnp}\right)^\beta \\\\
=\dfrac{1}{2\beta}(p+s _ n)lnn^\beta-\dfrac{1}{2}(p+s _ n)ln(p+s _ 0)lnp \\\\
\leq \dfrac{1}{2\beta}(p+s _ n)lnn^\beta=\dfrac{1}{2\beta}(p+s _ n)lnp
\end{gathered}
\end{equation}
$$
이다.($\because p\asymp n^\beta$)
이를 다시 처음 부등식에서 정리하면,
$$\leq 2(p+s _ n)lnp+6s _ nlnp+\dfrac{3}{4}(p+s _ n)lnp+\dfrac{1}{2\beta}(p+s _ n)lnp+2s _ nlnp $$이다.
이 부등식에 적절한 상수배를 해주면,
$$
\begin{equation}
\begin{gathered}
\leq 6s _ nlnp+\dfrac{11}{4}(p+s _ n)lnp+\dfrac{1}{2\beta}(p+s _ n)lnp+2s _ nlnp \\\\
(6+\dfrac{1}{2\beta})(\dfrac{s _ n}{c _ 1}+s _ n)lnp< \dfrac{1}{2}(12+\dfrac{1}{\beta})(c _ 1+c _ 1)n\epsilon _ n^2 \quad (\because c _ 1>1) \\\\
=(12+\dfrac{1}{2\beta})c _ 1n\epsilon _ n^2
\end{gathered}
\end{equation}
$$
이다. 따라서,
$$lnD(\epsilon _ n,P _ n,d)\leq(12+\dfrac{1}{\beta})c _ 1n\epsilon _ n^2$$이 성립해서,
이를 통해 Lemma1의 첫 번째 조건이 성립함을 알 수 있다.

</div>

# 보조정리 2

<div class="lemma">

If $a=b=\dfrac{1}{2},\tau=O(\dfrac{1}{p^2}\sqrt{\dfrac{s _ 0lnp}{n}}),\ s _ 0lnp=O(n)\ and\ \zeta >3, we\ have, for\ some\ constant\\ C>0,$

$$
\begin{equation*}
    \pi^u(\Sigma \in \mathcal{U}(\zeta))>\left\\{\dfrac{\lambda \zeta}{8}exp(-\dfrac{\lambda \zeta}{4}-C) \right\\}^p
\end{equation*}
$$

</div>

<div class="proof">
이 Lemma는 논문의 Theorem4 의 증명에서 활용되는 Lemma이다.

Gershgorin circle Thm에 의해, covariance matrix의 eigenvalue 들은 적어도 $[\sigma _ {jj}-\sum\limits _ {k\neq j}|\sigma _ {kj}|,\sigma _ {jj}+\sum\limits _ {k\neq j}|\sigma _ {kj}|], j \in \\{1,2,\cdots,p\\}$ 안에 있다는 것을 알 수 있다. 따라서,
$$
\begin{equation}
\begin{gathered}
\pi^u(\Sigma \in \mathcal{U}(\zeta)) \\\\
\geq \pi^u(min _ j(\sigma _ {jj}-\sum _ {k\neq j}|\sigma _ {kj}|)>0,\zeta^{-1} \leq \lambda _ {min}(\Sigma)\leq \lambda _ {max}(\Sigma)\leq \zeta)
\end{gathered}
\end{equation}
$$
임을 보이면 된다. 먼저 $\min\limits _ {j}(\sigma _ {jj}-\sum\limits _ {k\neq j}|\sigma _ {kj}|)>0$을 살펴보면, 1-norm의 정의에 의해,
$||\Sigma|| _ 1 = \max\limits _ {1\leq j\leq n} \sum\limits _ {i=1}^{m}|a _ {ij}|$\ 이므로,
$$\lambda _ {max}(\Sigma) \leq ||\Sigma|| _ 1 = \max\limits _ {j}(\sigma _ {jj}+\sum\limits _ {k\neq j}|\sigma _ {kj}|)\leq \max\limits _ {j}2\sigma _ {jj}$$
로 표현할 수 있다. 또한, G.C Thm에 의해,
$$\lambda _ {min}(\Sigma)\geq \min\limits _ {j}(\sigma _ {jj}-\sum\limits _ {k\neq j}|\sigma _ {kj}|)$$
로 표현할 수 있다. 따라서,
이를 위의 식 $\zeta^{-1} \leq \lambda _ {min}(\Sigma)\leq \lambda _ {max}(\Sigma)\leq \zeta$에 적용해서 다시 표현하면,
$$\pi^u(\Sigma \in \mathcal{U}(\zeta)) \geq \pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\sum\limits _ {k\neq j}|\sigma _ {kj}|)\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta)$$
이고, $P(A)\geq P(A \cap B)=P(A|B)P(B)$ 성질을 이용하면,
$$
\begin{equation}
\begin{gathered}
\pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\sum\limits _ {k\neq j}|\sigma _ {kj}|)\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta) \\\\
\geq \pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\sum\limits _ {k\neq j}|\sigma _ {kj}|)\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta \ | \max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1} )\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1}) 
\end{gathered}
\end{equation}
$$
인데, 조건부에 의해 다음 부등식이 되고,
$$\geq \pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\zeta^{-1})\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta \ | \max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1} )\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1})$$
에서, $\sigma _ {jj}$ 항과 $\sigma _ {ij}$는 독립이므로 조건부 항을 없앨 수 있다. 따라서,
$$=\pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\zeta^{-1})\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta)\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1})$$
이다.
다시 정리해보면, 다음과 같은 부등식을 얻을 수 있다.
$$\pi^u(\Sigma \in \mathcal{U}(\zeta)) \geq \underline{\pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\zeta^{-1})\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta)} \ * \ \underline{\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1})}$$
먼저 첫 번째 밑줄 확률을 계산해보면,
$$
\begin{equation}
\begin{gathered}
\pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\zeta^{-1})\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta) \\\\
\geq \pi^u(2\zeta^{-1} \leq \sigma _ {jj} \leq \dfrac{\zeta}{2}, \forall j) \\\\
\geq \prod\limits _ {j=1}^p \pi^u(2\zeta^{-1}\leq \sigma _ {jj} \leq \dfrac{\zeta}{2})
\end{gathered}
\end{equation}
$$
에서 $\sigma _ {jj}$가 $\Gamma(1,\dfrac{\lambda}{2})$를 따른다는 가정에 의해, $f(\sigma _ {jj})=\dfrac{\lambda}{2}exp(-\dfrac{\lambda}{2}\sigma _ {jj})$의 pdf 를 갖고, 가로 길이가 $\left(\dfrac{2}{\zeta},\dfrac{\zeta}{2}\right)$이고 세로 길이가 $\left(0,f\left(\dfrac{2}{\zeta}\right)\right)$ 인 직사각형을 생각하면,
이는 pdf 의 전체 넓이 보다는 작으므로, 이를 통해
$$
\begin{equation}
\begin{gathered}
\prod\limits _ {j=1}^p\pi^u(2\zeta^{-1}\leq \sigma _ {jj} \leq \dfrac{\zeta}{2}) \\\\
=\left\\{\left(\dfrac{\zeta}{2}-\dfrac{2}{\zeta}\right)\dfrac{\lambda}{2}exp(-\dfrac{\lambda \zeta}{4})\right\\}^p \geq \left\\{ \dfrac{\lambda \zeta}{8} exp(-\dfrac{\lambda \zeta}{4})\right\\}^p
\end{gathered}
\end{equation}
$$
임을 알 수 있다. 이제 두 번째 밑줄 확률인 $\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1})$를 계산할 건데, 먼저 이를 위한 lemma 하나를 소개하자.

<div class="lemma" style="border: solid; padding: 30px; margin: 10px">[lemma 1 in [12]]

The univariate horseshoe density $p(\theta)$ satisfies the following:
$$
\begin{align*}
    &(a) \lim\limits _ {\theta \rightarrow 0}p(\theta) = \infty\\
    &(b) \ For\ \theta \neq 0, \dfrac{K}{2}log\left(1+\dfrac{4}{\theta^2}\right)<p(\theta)<Klog\left(1+\dfrac{2}{\theta^2}\right), \ where\ K=\dfrac{1}{\sqrt{2\pi^3}}
\end{align*} 
$$

</div>

따라서, 위의 lemma\ (b) 를 통해, $\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1})$의 계산을 위한 부등식인
$$\pi^u(\sigma _ {kj})\leq \dfrac{1}{\tau \sqrt{2\pi^3}}ln\left(1+\dfrac{2\tau^2}{\sigma _ {kj}^2}\right)$$
를 알 수 있다. 한편, $|\sigma _ {kj}|$ 는 이대일 변환이므로, 2가 곱해져서,
$$\pi^u(|\sigma _ {kj}|\geq(\zeta p)^{-1})\leq \dfrac{1}{\zeta}\sqrt{\dfrac{2}{\pi^3}} \displaystyle \int _ {(\zeta p)^{-1}}^{\infty}ln\left(1+\dfrac{2\tau^2}{x^2}\right) dx$$
가 되고,
$$
\begin{equation}
\begin{gathered}
\leq \sqrt{\dfrac{2}{\pi^3}} \displaystyle \int _ {(\zeta p)^{-1}}^{\infty}\dfrac{2\tau^2}{x^2} dx \quad ( \because ln(1+x) \leq x, \ when\ x\geq 0) \\\\
= \sqrt{\dfrac{2}{\pi^3}} \displaystyle \int _ {(\zeta p)^{-1}}^{\infty}\dfrac{2\tau}{x^2} dx \\\\
=\dfrac{2\sqrt{2}}{\sqrt{\pi^3}}\tau\zeta p \\\\
\end{gathered}
\end{equation}
$$
임을 알 수 있다. 이를 통해 이제 $\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1})$ 를 계산해보면,
$$
\begin{equation}\label{eqn-1}
\begin{gathered}
\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1}) = \prod\limits _ {k\neq j} \left\\{    1-\pi^u(|\sigma _ {kj}|\geq (\zeta p)^{-1})   \right\\} \\\\
= \left(1-\dfrac{2\sqrt{2}}{\sqrt{\pi^3}}\tau\zeta p\right)^{p(p-1)} \\\\
\geq \left(1-\dfrac{2\sqrt{2}}{\sqrt{\pi^3}}\tau\zeta p\right)^{p^2} \quad (\because \text{괄호안의 값은확률로 1보다 작으므로}) \\\\
\geq exp\left(-\dfrac{4\sqrt{2}}{\sqrt{\pi^3}}\tau \zeta p^3\right) \ (\because log(1-x) \geq -2x,\ when\ x\leq \dfrac{1}{2})
\end{gathered}
\end{equation}
$$
이다. 한편, 주어진 조건에서 $\tau=O\left(\dfrac{1}{p^2}\sqrt{\dfrac{s _ 0lnp}{n}}\right), s _ 0lnp=O(n)$ 이라 했으므로,
$$\dfrac{\tau}{\dfrac{1}{p^2}\sqrt{\dfrac{s _ olnp}{n}}}\leq c _ 1, \dfrac{s _ olnp}{n}\leq c _ 2,\ c _ 1,c _ 2>0$$
임을 알 수 있다. 이들을 조합하면
$$\Rightarrow \ \tau p^2 \leq \sqrt{c _ 2}c _ 1 \defeq c _ 3 \ \Rightarrow \tau p^3 \leq c _ 3p\ \Rightarrow \ -\tau p^3 \geq -c _ 3p$$
이다. 이를 <a href="#eqn-1">위 식</a>에서 활용하면,
$$exp\left(-\dfrac{4\sqrt{2}}{\sqrt{\pi^3}}\tau \zeta p^3\right) \geq exp\left(-c _ 3 \dfrac{4\sqrt{2}}{\sqrt{\pi^3}} \zeta p\right)=exp(-Cp)$$
이다. 따라서, 두번째 밑줄 확률의 부등식을 다음과 같이 구할 수 있다.
$$\pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1}) \geq exp(-Cp)$$
이제 첫 번째 밑줄 식과 두 번째 밑줄 식의 결과를 종합하면,
$$
\begin{equation}
\begin{gathered}
\pi^u(\Sigma \in \mathcal{U}(\zeta)) \geq \pi^u(\zeta^{-1}\leq\min\limits _ {j}(\sigma _ {jj}-\zeta^{-1})\leq 2\max\limits _ {j}\sigma _ {jj}\leq\zeta) \ * \ \pi^u(\max\limits _ {k \neq j}|\sigma _ {kj}|< (\zeta p)^{-1}) \\\\
\geq \left\\{ \dfrac{\lambda \zeta}{8} exp(-\dfrac{\lambda \zeta}{4})\right\\}^p exp(-Cp)=\left\\{ \dfrac{\lambda \zeta}{8} exp(-\dfrac{\lambda \zeta}{4}-C)\right\\}^p
\end{gathered}
\end{equation}
$$
임을 알 수 있다.

</div>
