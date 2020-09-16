# `physics.py`

## `flux_expected_r86(jm, freq, y_max, y_min=None)`

Equation 8 of Reynolds (1986) states that the total flux of a monopolar jet can be calculated by the following equation:

$${\displaystyle S_\nu=\int^{y_{\rm max}}_{y_{\rm 0}} {\frac{2 w(y)}{d^2} \frac{a_{\rm j}}{a_{\rm k}} T(y) \nu^2 \left( 1 - e^{-\tau(y)} \right) dy}}$$

or,

${\displaystyle S_\nu=\frac{2 a_{\rm j} w_0 T_0 \nu^2}{a_{\rm k} d^2}\int^{y_{\rm max}}_{y_{\rm 0}} {\rho^{\epsilon + q_{\rm T}}\left( 1 - e^{-\tau_0 \rho^{q_\tau}}\right) dy}}$

where $\rho = y / y_0$, $y = r\sin(i)$ and $y_0 = r_0 \sin(i)$.

This evaluates to the following definite integral:

${\displaystyle S_\nu=\frac{2 a_{\rm j} w_0 T_0 \nu^2}{a_{\rm k} d^2}  \left[ \frac{y}{q_\tau c } \, \rho^{\epsilon + q_{\rm T}} \, \tau(y)^{-\frac{c}{q_\tau}}\left( q_\tau \, \tau(y)^{\frac{c}{q_\tau}}+c \, \Gamma \left( \frac{c}{q_\tau},\tau(y) \right) \right) \right]^{y_{\rm max}}_{y_{\rm 0}}}$

where $c=1+\epsilon + q_{\rm T}$.

For disc wind models employed by `RaJePy`, $\rho=y/y^\prime_0$ where $y^\prime_0=r^\prime_0 \sin(i)$, $r^\prime_0=\frac{\epsilon \, w_0}{\tan\left(\theta/{2}\right)}$ and $\theta$ is the full opening angle at the base of the jet. Consequently, $y_{\rm max}=\sin(i) \left( l + r^\prime_0 - r_0\right)$ where $l$ is the length of the monopolar jet to integrate over and $r_0$ is the defined launching radius of the disc-wind. The lower limit of the integral is set to $y^\prime_0$.