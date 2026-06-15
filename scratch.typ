#let inner(l, r) = $chevron.l #l, #r chevron.r$
#let zpos(t) = $z_(#t) (p)$
#let pos = $z_(t-1) (p)$
#let ept = $epsilon_t$
#let ep = $epsilon_t (p)$
#let tbar = $ norm(Pi_(R_t \/ V_t) (theta))_2 $
update rule:
$ z_t = z_(t-1) (p) + delta_t u_t + ept $
$ zpos(t) = pos + delta_t u_t (p) + ept (p) = pos + delta_t + ept (p) $
martingale differences:
we have update rule
#let c4 = $c_4$
$ X_t = delta_t inner(B u_t, theta) - c4 (zpos(t)^2 - zpos(t-1)^2)tbar^2 \
= delta_t inner(B u_t, theta) - c4 ((pos^2 + delta_t^2 + ep^2 + 2 delta_t pos + 2 delta_t ep + 2 pos ep) - pos^2)tbar^2 \
= delta_t inner(B u_t, theta) - c4 (delta_t^2 + ep^2 + 2 delta_t pos + 2 delta_t ep + 2 pos ep)tbar^2 $

// NOTE: From here I won't use the above variables again except for c4, and I'm redefining tbar, ep

#let tbar = $ overline(theta)_t $
#let dt = $ delta_t $
#let ep = $ epsilon $
#let eiter = $epsilon_"iter"$
#let c100 = $c_100$
#let c8 = $c_8$
Following the conventions of (citation here of the blues paper) we make use of the following notation for clarity, and in addition we write $epsilon$ for $epsilon_t (p)$
// TODO: this spacing is not good there's probably a good way to do this
$ x := pos "  " overline(theta)_t = norm(Pi_(R_t \/ V_t) (theta))_2 "  " theta_t = inner(B u_t, theta) $

we can therefore rerwite the above expression as
$
  X_t = delta_t theta_t - c4 (dt^2 + 2 dt x + epsilon^2 + 2 dt epsilon + 2 x epsilon) tbar^2\ 
  = dt theta_t - c4(dt^2 + 2 dt x) tbar^2 - c4 (ep^2 + 2 dt ep + 2 x ep) tbar^2
$

#let ee(expr) = $EE_(t-1) [#expr]$
where the final term is the contribution of the additional $epsilon$ in the update rule, and the first two terms match the expression of the original algorithm. 
We first establish that $abs(X_t) <= 1$, and to that end note the following bounds from (cite blues paper) that we will use throughout this section.

We have $x in [-1,1]$ and $x + dt = z_(t-1)(p) + delta_t u_t (p) in [-1,1]$, with the second inequality holding in this setting
as $z_(t-1)(p) + delta_t u_t (p)$ is the ideal choice of step before adding in noise. 
We therefore have $abs(dt + 2 x) = abs((dt + x) + x) <= 2$.
We also note that $abs(dt) <= 2$ as mentioned in the previous section.
Furthermore we observe $abs(ep + 2 dt + 2 x) <= abs(ep) + 2 abs(dt + x) <= 1 + 2= 3$ 
where we used $abs(ep) <= eiter <= 1$. Finally, we will use the fact that $theta_t <= tbar <= 1/c8$. 
The first inequality follows from the fact that $theta_t = inner(B u_t, theta)$ where $B u_t in R_t slash V_t$ and $norm(B u_t)_2 <= 1$ and so
$inner(B u_t, theta) = inner(B u_t, Pi_(R_t \/ V_t) (theta)) <= 1 dot norm(Pi_(R_t \/ V_t) (theta))_2 = tbar$.
The latter inequality follows from the fact that in good times we assume $norm(V_(f_t - 1 \/ V_t) (theta)) <= 1 slash c8$, and $R_t in V_(f_t - 1)$
as the pivot and any alive units at time $t$ must have been alive at time $t-1$, before the start of the current pivot phase.


We can therefore bound $abs(X_t)$ using the established bounds:
$
  abs(X_t) = abs(dt theta_t - c4(dt^2 + 2 dt x) tbar^2 - c4 (ep^2 + 2 dt ep + 2 x ep) tbar^2) \
  <= abs(dt) tbar + c4 abs(dt) abs(dt + x) tbar^2 + c4 abs(ep)abs(ep + 2 dt + 2x)tbar^2\
  <= 2/c8 + c4 dot 2 dot 1 dot 1/c8^2 + 4 dot 1 dot 3 dot 1/c8^2 <= 1
$
by our choice of $c4, c8$.




Our bounds on the conditional moments proceed as in (cite the blues paper here), the main choice in adapting
the proof to this setting is the assumption $eiter <= ee(dt^2)$ for all iterations $t$ as previously mentioned.
We use $ee(dot)$ to denote the expectation conditioned on the randomness of iterations $1,...,t-1$. 
In particular, under this conditioning $theta_t, tbar, x, u_t$ are decided. 
$
  ee(X_t) &= ee(dt theta_t) - ee(c4(dt^2 + 2 dt x) tbar^2) - ee(c4 (ep^2 + 2 dt ep + 2 x ep) tbar^2) &"Defn of "X_t  \
  &= theta_t ee(dt^2) - c4 tbar^2 ee(dt^2 + 2 dt x) - c4 tbar^2 ee(ep^2 + 2 dt ep + 2 x ep) &"Properties of "ee(dot) \
  &= 0 - c4 tbar^2 ee(dt^2) - c4 tbar^2 ee(ep(ep + 2 dt + 2x)) &dt "is mean zero" \
  &<= - c4 tbar^2 ee(dt^2) + c4 tbar^2 abs(ee(ep(ep + 2 dt + 2x))) &x <= abs(x) \
  &<= - c4 tbar^2 ee(dt^2) + c4 tbar^2 ee(abs(ep(ep + 2 dt + 2x))) &"Jensen's inequality" \

  &<= - c4 tbar^2 ee(dt^2) + c4 tbar^2 ee(abs(ep dot 3 )) &abs(ep) <= 1, dt + x in [-1,1] \
  &<= - c4 tbar^2 ee(dt^2) + c4 tbar^2 eiter dot 3 &"Expectation of a bounded RV" \
  &<= - c4 tbar^2 ee(dt^2) + (c4 dot 3)/(c100) tbar^2 ee(dt^2) & "we assume" eiter <= ee(dt^2), forall t\
  &= (-c4 + (c4 dot 3)/c100) tbar^2 ee(dt^2)
$

Proceeding to bound the conditional second moment, we proceed as follows:


$
  X_t^2 &<= 2 dt^2 theta_t^2 + 2 c4^2 (underbrace(dt ^2 + 2 dt x, a) + underbrace(ep^2 + 2 x ep + 2 dt ep, "b"))^2 tbar^4 &"define" a,b "for notational clarity" \
  &= 2 dt^2 theta_t^2 + 2 c4^2 (a^2 + b^2 + 2 a b) tbar^4 &"expanding" \
  &= 2 dt^2 theta_t^2 + 2 c4^2 a^2 tbar^4 + 2 c4^2 (b^2 + 2 a b) tbar^4 &"splitting off terms depending on "ep \
  &<= 2 dt^2 tbar^2 + 2 c4^2 a^2 tbar^4 + 2 c4^2 (b^2 + 2 a b) tbar^4 & abs(theta_t) <= tbar \
  &<= 2 dt^2 tbar^2 + 2 c4^2/c8^2 a^2 tbar^2 + 2 c4^2/c8^2 (b^2 + 2 a b) tbar^2 &abs(tbar) <= 1/c8 "in good times" \
$
Note that $dt + 2x = (dt + x) + x$ where $dt + x in [-1,1], x in [-1,1]$ and so $abs(dt + 2x) <= abs(dt + x) + abs(x) <= 2$. This implies
$ a^2 = (dt^2 + 2 dt x)^2 =(dt underbrace((dt + 2x), <= 2))^2 <= 4 dt ^2  $
Now to bound the $b^2 + 2 a b$ term we notice that
$ abs(b) = abs(ep^2 + 2 x ep + 2 dt ep) <= abs(ep (ep + 2x + 2dt)) <= 3 eiter "as before" $
$ abs(2 a b) <= 2 abs(dt^2 + 2 dt x) dot abs(b) <= 2 abs(dt) abs(dt + 2 x) dot 3 eiter <= 2 dot 2 dot 2 dot 3 eiter <= 24 eiter $
which gives $b^2 + 2 a b <= 9 eiter^2 + 24 eiter <= 9 eiter + 24 eiter = 33 eiter$. Plugging this in and taking the conditional expectation, we have
$ 
  ee(X_t^2) <= 2 ee(dt^2) tbar^2 + 2 c4^2/c8^2 (4 ee(dt^2)) tbar^2 + 2 c4^2/c8^2 (33 eiter) tbar^2 \
  <= 2 ee(dt^2) tbar^2 + 2 c4^2/c8^2 (4 ee(dt^2)) tbar^2 + 2 c4^2/c8^2 (33 ee(dt^2) / c100) tbar^2 \
  = tbar^2 ee(dt^2) (2 + 8 c4^2 slash c8^2 + 66c4^2 slash (c8^2 c100))
$

Combining our bounds on the first and second moments gives
$ ee(X_t + X_t^2) <= tbar^2 ee(dt^2) (-c4 + 3c4/c100 + 2 + 8c4^2/c8^2 + 66c4^2/(c8^2 c100)) $
and so we have $ee(X_t + X_t^2)$ for choices of $c4, c8, c100$ such that $-c4 + 3c4/c100 + 2 + 8c4^2/c8^2 + 66c4^2/(c8^2 c100) <= 0$.
We note that in the original proof the terms involving $c100$ which fall out of terms depending on $ep$ are not present,
and choosing $c4 = 4, c8 = 8$ recovers the original argument to give $-c4 + 2 + 8 c4^2 slash c8^2 = -4 + 2 + 2 = 0$.

