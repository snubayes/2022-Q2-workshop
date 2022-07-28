---
title: "다양체"
author: "이재용 교수님"
date: 2022-07-27
weight: 10
---

> Reference: Muirhead (1982) - Aspects of Multivariate Analysis.

> 앞 부분 추가 필요 

$$I = \int_A f(x_1, \cdots, x_m) dx_1 \cdots dx_m = \int_{A^\prime} f(x(y)) \det \begin{pmatrix} \frac{\partial x_i}{\partial y_i} \end{pmatrix} dy_1 \cdots dy_m$$

$$
\begin{equation*}
\begin{aligned}
A^\prime &= y(A) \\\\
x_1 &= x_1(y_1, \cdots, y_m) \\\\
x_2 &= x_2(y_1, \cdots, y_m) \\\\
\vdots & \\\\
x_m &= x_m(y_1, \cdots, y_m) \\\\
\end{aligned}
\end{equation*}
$$

$x \mapsto y(x)$

각 $i=1,2, \cdots, m$에 대해 
$$dx_i = \frac{\partial x_i}{\partial y_1} dy_1 + \cdots \frac{\partial x_i}{\partial y_m} dy_m$$

<div class="example">[2차원의 예]

$$
\begin{equation*}
\begin{aligned}
I &= \int_A f(x_1, x_2) dx_1 dx_2 \\\\
&= \int_{A^\prime} f(x(y)) \left( \frac{\partial x_1}{\partial y_1} dy_1 +  \frac{\partial x_1}{\partial y_2} dy_2   \right) \left( \frac{\partial x_2}{\partial y_1} dy_1 +  \frac{\partial x_2}{\partial y_2} dy_2   \right)
\end{aligned}
\end{equation*}
$$

이때, 
$$\begin{equation*}
\begin{aligned}
 &\left( \frac{\partial x_1}{\partial y_1} dy_1 +  \frac{\partial x_1}{\partial y_2} dy_2   \right) \left( \frac{\partial x_2}{\partial y_1} dy_1 +  \frac{\partial x_2}{\partial y_2} dy_2   \right) \\\\
 &= \frac{\partial x_1}{\partial y_1} \frac{\partial x_2}{\partial y_1} dy_1 dy_1 + \frac{\partial x_1}{\partial y_1} \frac{\partial x_2}{\partial y_2} dy_1 dy_2  + \frac{\partial x_1}{\partial y_2} \frac{\partial x_2}{\partial y_1} dy_2 dy_1  + \frac{\partial x_1}{\partial y_2} \frac{\partial x_2}{\partial y_2} dy_2 dy_2 \\\\
 &= \left( \frac{\partial x_1}{\partial y_1} \frac{\partial x_2}{\partial y_2} - \frac{\partial x_1}{\partial y_2} \frac{\partial x_2}{\partial y_1}  \right) dy_1 dy_2
 \end{aligned}
\end{equation*}$$
가 성립한다. 

이러한 관점에서, 

$$\begin{equation*}
\begin{aligned}
dy_1 \wedge dy_2 &= - dy_2 \wedge dy_1 \\\\
dy_i \wedge dy_i &= 0
\end{aligned}
\end{equation*}$$

이 되도록 $dy_1 \wedge dy_2$를 정의할 수 있다. (exteerior product)

</div>

<div class="definition">[기호]

$$
\begin{equation}
    y = \begin{bmatrix} y_1 \\\\ \vdots \\\\ y_m \end{bmatrix}
\end{equation}
$$

$$
\begin{equation}
    dy = \begin{bmatrix} dy_1 \\\\ \vdots \\\\ dy_m \end{bmatrix}
\end{equation}
$$

</div>

<div class="theorem">

$dy$가 $m \times 1$, $dx =  Bdy$, $B$는 정칙(non-singular)이면 다음이 성립한다. 
\begin{equation}
    \bigwedge_{i=1}^m dx_i = \det B \bigwedge_{i=1}^m dy_i 
\end{equation}

</div>

exterior product를 사용하면 복잡한 야코비안의 계산을 곱셈으로 바꿀 수 있다.

<div class="definition">[미분 형식]

다음을 $r$차 미분 형식(differential form of degree $r$)이라 한다.

$$\sum_{i_1 < \cdots < i_r} h_{i_1 \cdots i_r}(x) dx_{i_1} \wedge \cdots \wedge dx_{i_r}$$

</div>

<div class="example">[이차 미분 형식]

> 놓쳤음,,,

</div>

<div class="theorem">

\begin{equation}
\begin{aligned}
x_1 &= r \sin \theta_1 \sin \theta_2 \cdots \sin \theta_{m-2} \sin \theta_{m-1}, \\\\
x_2 &= r \sin \theta_1 \sin \theta_2 \cdots \sin \theta_{m-2} \cos \theta_{m-1}, \\\\
x_3 &= r \sin \theta_1 \sin \theta_2 \cdots \cos \theta_{m-2}, \\\\
& \vdots \\\\
x_{m-1} &= r \sin \theta_1 \cos \theta_2, \\\\
x_m &= r \cos \theta_1,
\end{aligned}
\end{equation}
$r > 0,~ 0< \theta_i \leq \pi,~i=1,2,\cdots, m-2,~ 0< \theta_{m-1} < 2\pi$
라 하자. 

그러면, 다음이 성립한다. 
\begin{equation}
\bigwedge_{i=1}^m dx_i = r^{m-1} \sin^{m-2} \theta_2 \sin^{m-3} \theta_3 \cdots \sin \theta_{m-2} \bigwedge_{i=1}^{m-1} d \theta_i \wedge dr
\end{equation}

<div class="proof">

모든 변수를 제곱하면서 더해나가면
\begin{equation}
\begin{aligned}
x_1^2 &= r^2 \sin^2 \theta_1 \sin \theta_2 \cdots \sin^2 \theta_{m-2} \sin^2 \theta_{m-1}, \\\\
x_1^2 + x_2^2 &= r^2 \sin^2 \theta_1 \sin^2 \theta_2 \cdots \sin^2 \theta_{m-2}, \\\\
& \vdots \\\\
x_1^2 + \cdots + x_{m-1}^2 &= r^2 \sin^2 \theta_1,
\end{aligned}
\end{equation}

모두 더하면 
\begin{equation}
x_1^2 + \cdots + x_m^2 = r^2
\end{equation}
이 된다. 

미분 형식으로부터
$$
\begin{equation*}
\begin{aligned}
2x_1 dx_1 &= r^2 \sin^2 \theta_1 \cdots \sin^2 \theta_{m-2} ( 2 \sin \theta_{m-1}) \cos \theta_{m-1} d\theta_{m-1}, \\\\
2x_1 dx_1 + 2x_2 dx_2 &= r^2 \sin^2 \theta_1 \cdots \sin^2 \theta_{m-3} ( 2 \sin \theta_{m-2}) \cos \theta_{m-2} \cos \theta_{m-1} d\theta_{m-2}, \\\\
&\vdots \\\\
2 x_1 dx_1 + \cdots + 2x_m dx_m &= 2r dr
\end{aligned}
\end{equation*}
$$

양 변을 곱하면 좌변은 
$$\begin{equation*}
2^m x_1 \cdots x_m \bigwedge dx_i 
\end{equation*}$$

우변은 
$$\begin{equation*}
2^m r^{2m-1} \sin^{2m-3} \theta_1 \sin^{2m-5} \theta_2 \cdots \sin \theta_{m-1} \cos \theta_1 \cdots \cos \theta_{m-1} \bigwedge{i=1}^{m-1} d\theta_i \wedge dr 
\end{equation*}$$

정리하면, 원하는 결과를 얻는다. 

</div>
</div>

<div class="remark">
야코비안을 계산하는 새롭고 유용[^dk]한 방법이라는 의의가 있다.
</div>

## $m$차원 공간에서 $n$-form의 적분 $(m \geq n)$

> 다양체(manifold)의 엄밀한 정의는 여기서 다루지 않는다. diffeomorphism[^diffeomorphism]에 의해 변환된 유클리드 공간의 부분집합을  다양체라 생각한다.

[^diffeomorphism]: 미분 가능하고 1-1, 역함수도 미분 가능한 함수

다양체 $M$이 diffeomorphism $\phi: \mathbb{R}^n \rightarrow M$에 의해 정의된다고 하자. 

다양체 $M$위의 미분 형식 $w$는 다음과 같이 정의한다. 

\begin{equation}
\int_M w = \pm \int_{\mathbb{R}^n} w_{\phi(x_1, \cdots, x_m)} \left( \frac{\partial \phi}{\partial x_1} (x_1 \cdots x_n) \cdots \frac{\partial \phi}{\partial x_n} (x_1 \cdots x_n)   \right) dx_1 \cdots dx_n
\end{equation}

> $w$의 의미...?[^dk]

여기서 
$$
\begin{equation}
(w_1 \wedge \cdots \wedge w_n )(v_1, \cdots, v_n) = 
\begin{vmatrix}
w_1(v_1),~ w_2(v_1),~\cdots,~w_n(v_1) \\\\
w_1(v_2),~ w_2(v_2),~\cdots,~w_n(v_2) \\\\
\vdots \\\\
w_1(v_n),~ w_2(v_n),~\cdots,~w_n(v_n)
\end{vmatrix}
\end{equation}
$$

<div class="example">

$w = \frac{1}{x} dy \wedge dz - \frac{1}{y} dx \wedge dz$라 하자. $\mathbb{R}^3$ 위의 단위구의 상반구에서 적분을 해보자.

diffeomorphism $$\phi: R= \\{ (s, t): s^2 + t^2 \leq 1 \\} \rightarrow M ,~(s, t) \mapsto (s, t, \sqrt{1-s^2-t^2})$$을 생각하면, 

$$\begin{equation*}
\int_M w = \int_R w_{\phi(s, t)} \left( \frac{\partial \phi}{\partial s},~\frac{\partial \phi}{\partial t} \right) ds \wedge dt 
\end{equation*}$$

여기서 
$$\begin{equation*}
\begin{aligned}
\frac{\partial \phi}{\partial s} &= \left(1,~0,~ - \frac{s}{\sqrt{1-s^2-t^2}} \right) \\\\
\frac{\partial \phi}{\partial t} &= \left(0,~1,~ - \frac{t}{\sqrt{1-s^2-t^2}} \right)
\end{aligned}
\end{equation*}$$

이므로 
$$\begin{equation*}
\begin{aligned}
\int_M w 
&= \int_M \frac{1}{x} dy \wedge dz - \int_K \frac{1}{y} dx \wedge dz \\\\
&= \int_R \frac{1}{s}(dy \wedge dz) \left( \frac{\partial \phi}{\partial s}, \frac{\partial \phi}{\partial t} \right) ds dt - \cdots \\\\
&= \int_R \frac{1}{s} \begin{vmatrix} 0 & - \frac{s}{\sqrt{1-s^2-t^2}} \\\\ 1 & - \frac{t}{\sqrt{1-s^2-t^2}} \end{vmatrix} ds dt - \cdots \\\\
&= \int_R \frac{1}{s} \frac{s}{\sqrt{1-s^2-t^2}} ds dt \cdots 
\end{aligned}
\end{equation*}$$

</div>

> 그래서 미분 형식이 뭐냐? 그냥 measure다... 
> integrator다,, 혹은,, signed measure다,,[^dk]

[^dk]: 사실 여기도 이해 못했음

## 행렬의 미분 형식 

<div calss="definition">
$n \times m$ 행렬 $X = [x_{ij}]$를 생각하자. 행렬 $X$의 미분형식은 다음과 같이 정의된다. 

$$
\begin{equation*}
\begin{aligned}
    dX &= [dx_{ij}] \\\\
    (dX) &= \bigwedge_{i=1}^n \bigwedge_{i=1}^m dx_{ij}
\end{aligned}
\end{equation*}
$$

$X$에 제약이 있으면 자유로운 성분들에 대해서만 미분형식 성분을 생각한다. 

가령, 만일 $X$가 $m \times m$ 대칭행렬이면 
$$\begin{equation*}
    (dX) = \bigwedge_{1 \leq i \leq j \leq m} dx_{ij}
\end{equation*}$$

$X$가 상삼각행렬이어도 
$$\begin{equation*}
    (dX) = \bigwedge_{1 \leq i \leq j \leq m} dx_{ij}
\end{equation*}$$

</div>

<div class="theorem">

$X \in \mathbb{R}^{n \times m},~B \in \mathbb{R}^{n \times n},~ Y \in \mathbb{R}^{n \times m}$에 대해 $B$가 정칙행렬이고 
$X = BY$라 하자. 

그러면, 다음이 성립한다
\begin{equation}
(dX) = (\det B)^{m} (dY)
\end{equation}

<div class="proof">

$$\begin{equation*}
\begin{aligned}
dx
&= [dx_1 \cdots dx_m] \\\\
&= B dY \\\\
&= B [dy_1 \cdots dy_m], \\\\
d\underline{x_i} &= B d\underline{y_i},~ i=1,2,\cdots, m, \\\\
(d\underline{x_i}) &= (\det B) (d\underline{y_i}),~ i=1,2,\cdots, m, \\\\
(dX) &= \bigwedge_{i=1}^m (d \underline{x_i}) = \bigwedge_{i=1}^m (\det B) (d \underline{y_i}) = (\det B)^m (dY).
\end{aligned}
\end{equation*}$$

</div>

</div>

<div class="theorem">
$n \geq m$에 대해 $n \times m$ 행렬 $Z$에서 $rank(Z) = m$이라 하자. 

$$Z = H_1 T, \quad H_1^\prime H_1 = I_m$$
인 $n \times m$ 행렬 $H_1$와 $m \times m$ 상삼각행렬 $T$로 분해할 수 있다. (QR 분해)

$$H = [H_1; H_2] = [h_1 \cdots h_n] \in \mathbb{R}^{n \times n}$$
과 같이 나타낸다고 하자. 

그러면 다음이 성립한다. 
$$\begin{equation*}
    (dZ) = \prod_{i=1}^m t_{ii}^{n-i} (dT)  (H_1^\prime dH_1)
\end{equation*}$$
여기서 $$(H_1 dH_1) = \bigwedge_{i=1}^m \bigwedge_{j=i+1}^n h_j^\prime dh_i$$
이다.

</div>

<div class="theorem">

$A = Z Z^\prime$일때,,, 놓쳤음,,

</div>

<div class="definition">[Stiefel manifold]

다음과 같이 정의되는 공간을 Stiefel 다양체라 한다. 

$$V_{m, n}:= \\{ H_1 \in \mathbb{R}^{n \times m} : H_1 ^\prime H_1 = I_m \\}$$

</div>

<div class="remark">

Stiefel 다양체 $V_{n,m}$는 다음과 같이 생각할 수도 있다.

* $nm - \frac{m (m+1)}{2}$ 차원의 유클리드 공간 
* 혹은, $R^{n m}$의 반지름이 $\sqrt{m}$인 구의 부분집합

</div>

<div class="remark">

$m=n$인 경우, 
$$V_{n,m} = O(m) = \left \\{ H \in \mathbb{R}^{m \times m} : H^\prime H = I_m \right \\}$$
은 행렬곱에 대한 군을 이룬다. 

</div>

<div class="remark">[$O(m)$ 상의 미분 형식]

$H^\prime dH$: skew-symmetric 

$(H^\prime dH)$는 군 변환 중 좌변환, 우변환에 불변이다.[^dk]

좌변환, 우변환에 불변인 측도는 유일하다.[^dk][^dk-2]

$$M(D) = \int_D (H^\prime dH)$$

</div>

[^dk-2]: 알았다... 나는 오늘 이해한 것이 없구나...

재미있는[^dk] 정리를 하나 소개한다.

<div class="theorem">
    
$$\int_{V_{m, n}} (H_1^\prime d H_1) = \frac{2^m \pi^{mn/2}}{\Gamma_m(n/2)}$$

<div class="proof">[증명의 스케치]

$Z = (z_{ij}) \in \mathbb{R}^{n \times m}$에서 $z_{ij} \stackrel{i.i.d.}{\sim} N(0, 1)$이라 하자. 

$$\int e^{-\frac{1}{2} tr(Z^\prime Z)} (dZ) = (2 \pi)^{\frac{nm}{2}}$$

을 이용하면 된다. 

</div>

</div>