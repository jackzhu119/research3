## 2. Evidence-Retaining Functional Calibration

### 2.1. A frozen predictor and a support-only calibration interface

Evidence-Retaining Functional Calibration (ERFC) asks whether the information in a small calibration set supports changing a particular future prediction. Its adaptation object is a residual prediction function. The EEGNet encoder, original classifier, batch-normalization statistics and dropout state remain frozen [REF:EEGNET]. The principal anchor applies Euclidean alignment using a covariance transform estimated from the available calibration inputs [REF:EA]. Each target episode supplies a support set \(S=\{(x_i,y_i)\}_{i=1}^{n}\), with binary labels and \(n\leq20\). The chronological acquisition budget and quality-control rules are specified in the experimental protocol.

Write the anchor margin as \(m_0(x)\) and its class-1 probability as \(p_0(x)=\sigma(m_0(x))\), where \(\sigma(t)=(1+e^{-t})^{-1}\). A common probability floor of \(10^{-12}\) defines the numerical anchor for the proposed method, matched controls and fallbacks. ERFC uses a logistic link, as in established probability-calibration approaches [REF:PLATT], but conditions its residual function on frozen features rather than fitting a global sigmoid of the anchor margin alone.

The fitting interface receives support features, anchor logits, labels and a fixed configuration. Prediction receives the resulting state, query features and query anchor logits. It has no query-label argument. No query-batch statistic is estimated: conditional on the supplied frozen features and logits, a query's decision depends on its support set and its own feature vector.

### 2.2. A fixed residual function class

Let \(\phi(x)\in\mathbb R^d\) denote a frozen feature vector. A reference mean \(\mu_{\mathrm{ref}}\) and whitener \(W_{\mathrm{ref}}\), established from training-side reference data, define

\[
z(x)=(\phi(x)-\mu_{\mathrm{ref}})W_{\mathrm{ref}},\qquad
k(x,x')=\tau^2\left[d^{-1}z(x)^\top z(x')+1\right].
\tag{1}
\]

This reference transform is reused during calibration and prediction. The linear kernel and intercept define the residual reproducing-kernel Hilbert space \(\mathcal H_k\) [REF:RKHS]. They are fixed modeling components; ERFC's decision mechanism concerns which support-compatible residuals justify a query correction.

For numerical definiteness, the implemented kernel includes an explicit event-identity nugget,

\[
k_\epsilon(x,x')=k(x,x')+\epsilon\,
\mathbf1\{\operatorname{UID}(x)=\operatorname{UID}(x')\}.
\tag{2}
\]

Support identifiers are unique and disjoint from future-query identifiers. Thus the support Gram matrix has \(\epsilon I\), support–query cross-kernels have no identity increment, and the query diagonal includes \(\epsilon\). All Cholesky directions are retained. The nugget is part of the declared function class, not additional observations.

### 2.3. Retaining an observed calibration gain

Let \(n_c\) be the number of support samples with class \(c\). When both classes are represented, weights \(w_i=n/(2n_{y_i})\) satisfy \(\sum_iw_i=n\). ERFC minimizes the class-balanced logistic objective

\[
J(f)=\sum_{i=1}^{n}w_i\left[
\operatorname{softplus}(m_{0i}+f(x_i))
-y_i(m_{0i}+f(x_i))\right]
+\frac{\lambda}{2}\|f\|_{\mathcal H_{k_\epsilon}}^2.
\tag{3}
\]

The implementation obtains a finite feasible estimate \(\widehat f\) with \(\widehat J=J(\widehat f)\leq J_0=J(0)\). Define its observed regularized support gain as \(G=J_0-\widehat J\). Exact global convergence is not required to define this gain. With a fixed allowance \(c\in[0,1]\), the retained-function set is

\[
T=\widehat J+cG,\qquad
\mathcal C_T=\{f\in\mathcal H_{k_\epsilon}:J(f)\leq T\}.
\tag{4}
\]

Every member retains at least \((1-c)G\) of the observed gain. The analyzed configuration uses \(c=0.1\), retaining at least 90% of that gain. The set expresses compatibility with the measured support objective; it is a deterministic model set rather than a posterior credible region.

### 2.4. Completing a prediction beyond the support span

The retained set includes functions that agree on all support points but differ at a query. This distinction is essential when a small support set leaves feature directions unresolved. Let \(K_{SS}=BB^\top\) be the Cholesky factorization of the nugget-augmented support Gram matrix. For a query \(q\), define

\[
a_q=B^{-1}k_{Sq},\qquad
s_q=k_\epsilon(q,q)-\|a_q\|_2^2.
\tag{5}
\]

The Schur innovation \(s_q\geq0\) is the squared norm of the query representer orthogonal to the support span. The standard RKHS decomposition [REF:RKHS] reduces the extremal query problem to

\[
f_S=Bv,\quad f(q)=a_q^\top v+\sqrt{s_q}\,t,\quad
\|f\|^2=\|v\|_2^2+t^2.
\tag{6}
\]

Only the orthogonal component aligned with the query representer is needed at an extremum. If \(F(v)\) denotes Equation (3) with support residuals \(Bv\), the query range follows from minimizing or maximizing \(a_q^\top v+\sqrt{s_q}t\) subject to \(F(v)+\lambda t^2/2\leq T\). The extra scalar \(t\) represents support-invisible completion, not a parameter learned from query labels.

A direct witness can settle the decision before endpoint computation. Let \(\widehat v\) be the feasible fitted coefficient and \(\mu_q=a_q^\top\widehat v\). For \(s_q>0\), choosing \(t=-\mu_q/\sqrt{s_q}\) preserves every fitted support residual while making the query correction zero. Its additional objective cost is \(\lambda\mu_q^2/(2s_q)\). Therefore,

\[
\frac{\lambda\mu_q^2}{2s_q}\leq cG
\quad\Longrightarrow\quad
0\in\{f(q):f\in\mathcal C_T\}.
\tag{7}
\]

This is a sufficient querywise no-correction certificate. A positive innovation need not always prevent adaptation: its completion cost is assessed against the retained objective allowance. Conversely, \(s_q=0\) removes this particular orthogonal completion direction but does not make every retained coefficient vector identical. Ambiguity among support-compatible values of \(v\) is still assessed by the objective profile. The no-completion control sets the innovation term to zero while preserving the fitted support state, separating this mechanism from a change in the residual fit.

### 2.5. Conservative outer bounds from finite optimization

For a direction \(h\in\{a_q,-a_q\}\) and tilt \(\zeta>0\), consider \(\psi_{h,\zeta}(v)=F(v)-\zeta h^\top v\). Its Hessian is

\[
\nabla^2F(v)=B^\top\operatorname{diag}
\{w_i p_i(v)[1-p_i(v)]\}B+\lambda I
\succeq\lambda I.
\tag{8}
\]

Strong convexity converts any finite Newton iterate \(\widetilde v\) into a lower bound on the tilted minimum. With \(g=\nabla\psi_{h,\zeta}(\widetilde v)\),

\[
M_{h,\zeta}=\psi_{h,\zeta}(\widetilde v)
-\frac{\|g\|_2^2}{2\lambda}
\leq\inf_v\psi_{h,\zeta}(v).
\tag{9}
\]

Weak duality then gives

\[
\widehat U_h(\zeta)=\frac{T-M_{h,\zeta}}{\zeta}
+\frac{\zeta s_q}{2\lambda}
\geq\sup_{F(v)+\lambda t^2/2\leq T}
\{h^\top v+\sqrt{s_q}t\}.
\tag{10}
\]

The residual term preserves the real-arithmetic outer-bound direction without treating a convergence flag as an exact optimum. Taking the smallest valid bound over evaluated positive tilts remains conservative. Doubling and bisection locate informative tilts within fixed computational budgets; their role is to tighten a valid bound.

The deployed decision needs one direction. Set \(d_q=\operatorname{sign}(\mu_q)\), evaluate \(h=-d_qa_q\), and form

\[
e_q=-\min_{\zeta\in\mathcal Z}\widehat U_{-d_qa_q}(\zeta),
\qquad
\delta_q=d_q\max(0,e_q).
\tag{11}
\]

A positive \(e_q\) establishes that every retained function corrects in the fitted direction. A nonpositive bound leaves the anchor unchanged. Because \(\widehat f\) is feasible, a bound demanding a correction larger than \(|\mu_q|\) is inconsistent and triggers an anchor return. Computation stays in support-sized systems; the encoder is not optimized per query. This construction also separates decision conservatism from uncertainty magnitude alone. A large completion direction can veto a correction through Equation (7), whereas a tighter objective profile can support a nonzero update in the same function class. The deciding quantity is the nearest permitted query correction relative to zero, not a standalone variance score or the size of the fitted MAP correction.

### 2.6. Projecting the anchor probability

For the ideal residual range \([\ell_q,u_q]\), define

\[
I_q=[L_q,U_q]=[\sigma(m_0(q)+\ell_q),
\sigma(m_0(q)+u_q)].
\tag{12}
\]

The anchor-relative minimax cross-entropy decision has the closed form

\[
\arg\min_p\sup_{\eta\in I_q}
\{\operatorname{CE}(\eta,p)-\operatorname{CE}(\eta,p_0(q))\}
=\operatorname{clip}(p_0(q),L_q,U_q).
\tag{13}
\]

To see this, regret is affine in \(\eta\). When \(p>p_0\), its worst case occurs at \(L_q\); when \(p<p_0\), it occurs at \(U_q\). Minimizing each branch gives projection onto the interval. If the anchor lies inside, zero regret is attainable by retaining it. Outside, the worst regret at the projected endpoint equals the negative Bernoulli Kullback–Leibler divergence from that endpoint to the anchor and is nonpositive.

The one-sided implementation uses an outer interval with an unbounded far endpoint, yielding Equation (11) followed by \(p(q)=\sigma(m_0(q)+\delta_q)\). Thus a loose valid bound yields a smaller correction. This conditional loss statement concerns probabilities represented by the retained set. Membership of the true future conditional probability is an empirical question; the construction does not supply future-accuracy or coverage guarantees.

### 2.7. Fixed configuration and anchor recovery

The recorded candidate `a0.3-c0.1` uses a linear-plus-intercept kernel with \(\tau=0.3\), \(\lambda=1\), \(c=0.1\) and \(\epsilon=10^{-10}\). CPU float64 solves use at most 60 Newton iterations, 32 bracket steps and 12 bisection steps; gradient and root tolerances are \(10^{-10}\) and \(10^{-7}\). Fixed roundoff padding is \(10^{-10}\). This implements the real-arithmetic bounds numerically, without a formal floating-point interval-arithmetic proof.

Empty, single-class or fewer-than-two-sample supports return the common anchor. Zero kernel amplitude, negligible support gain, zero query cross-information, the witness in Equation (7), invalid joint-kernel geometry or unavailable finite bounds also preserve it. At \(c=1\), the zero residual belongs to \(\mathcal C_T\), so all queries recover the anchor. The matched MAP control uses the same kernel and fitted state but directly applies \(\mu_q\); it removes the query-functional retention decision. The support factor requires \(O(n^2)\) storage. Each Newton step solves an \(n\)-dimensional positive-definite system, and query profiling reuses the support features and factorization. Tilt iterations are temporary computations of an extremum; they do not update the stored fitted state. Numerical diagnostics record the support gain, innovation, witness cost, solver residuals, selected tilt and applied correction for each query. These controls isolate the mechanism that distinguishes ERFC: adaptation requires support evidence that survives the permitted completion of the queried prediction.
