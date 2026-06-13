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
$ X_t = delta_t inner(B u_t, theta) - 4 (zpos(t)^2 - zpos(t-1)^2)tbar^2 \
= delta_t inner(B u_t, theta) - 4 ((pos^2 + delta_t^2 + ep^2 + 2 delta_t pos + 2 delta_t ep + 2 pos ep) - pos^2)tbar^2 \
= delta_t inner(B u_t, theta) - 4 (delta_t^2 + ep^2 + 2 delta_t pos + 2 delta_t ep + 2 pos ep)tbar^2 $

// NOTE: From here I won't use the above variables again, and I'm redefining tbar, ep

#let tbar = $ overline(theta)_t $
#let dt = $ delta_t $
#let ep = $ epsilon $
#let eiter = $epsilon_"iter"$
#let c100 = $c_100$
#let c4 = $c_4$
#let c8 = $c_8$
Following the conventions of (citation here of the blues paper) we make use of the following notation for clarity, and in addition write $epsilon$ for $epsilon_t (p)$
// TODO: this spacing is not good there's probably a good way to do this
$ x := pos "  " overline(theta)_t = tbar "  " theta_t = inner(B u_t, theta) $

we can therefore rerwite the above expression as
$
  X_t = delta_t theta_t - c4 (dt^2 + 2 dt x + epsilon^2 + 2 dt epsilon + 2 x epsilon) tbar^2\ 
  = dt theta_t - c4(dt^2 + 2 dt x) tbar^2 - c4 (ep^2 + 2 dt ep + 2 x ep) tbar^2
$

#let ee(expr) = $EE_(t-1) [#expr]$
where the final term is the contribution of the additional $epsilon$ in the update rule, and the first two terms match the expression of the original algorithm. We use $ee(dot)$ to denote the expectation conditioned on the randomness of iterations $1,...,t-1$. In particular, under this conditioning $theta_t, tbar, x, u_t$ are constants.
$
  ee(X_t) = ee(dt theta_t) - ee(c4(dt^2 + 2 dt x) tbar^2) - ee(c4 (ep^2 + 2 dt ep + 2 x ep) tbar^2)  \
  = theta_t ee(dt) - c4 tbar^2 ee(dt^2 + 2 dt x) - c4 tbar^2 ee(ep^2 + 2 dt ep + 2 x ep) \
  = 0 - c4 tbar^2 ee(dt) - c4 tbar^2 ee(ep(ep + 2 dt + 2x)) \
  <= - c4 tbar^2 ee(dt) + c4 tbar^2 abs(ee(ep(ep + 2 dt + 2x))) \
  <= - c4 tbar^2 ee(dt) + c4 tbar^2 ee(abs(ep(ep + 2 dt + 2x))) \
$
// TODO: a) maybe give the necessary inequalites their own exposition
// TODO: we assume that epsilon <= 1. I wrote down the relevant assumptions somewhere in some picture on my phone.
note that $dt + x in [-1,1]$ and $abs(ep) <= 1$ so we have $abs(ep + 2 dt + 2x) <= 3$ implying
// FIXME: shoot when I did the optimization stuff I had a 12 instead of c_4 * 3 gotta rerun the solver
$
  <= - c4 tbar^2 ee(dt) + c4 tbar^2 eiter dot 3 \
  <= - c4 tbar^2 ee(dt) + (c4 dot 3)/(c100) tbar^2 ee(dt) \
  = (-c4 + (c4 dot 3)/c100) tbar^2 ee(dt)
$
// TODO: this is the big assumption it deserves its own exposition


// okay this second moment calc is just gonna be straight typesetting no exposition no justification but I will try to just make one logical step per line, to be combined later as I see fit
\
\

$
  X_t^2 <= 2 dt^2 theta_t^2 + 2 c4^2 (underbrace(dt ^2 + 2 dt x, a) + underbrace(ep^2 + 2 x ep + 2 dt ep, "b"))^2 tbar^4 \
  = 2 dt^2 theta_t^2 + 2 c4^2 (a^2 + b^2 + 2 a b) tbar^4 \
  = 2 dt^2 theta_t^2 + 2 c4^2 a^2 tbar^4 + 2 c4^2 b^2 + 2 a b tbar^4 \
$
Note that
// TODO: explain this with the other one, labeled a)
$ a^2 = (dt^2 + 2 dt x)^2 =(dt underbrace((dt + 2x), <= 2))^2 <= 4 dt ^2  $
// Q: why did I bound these by considering the absolute value again? b is getting squared but is the other one just convenience
// oh is it just to boudn the 2ab term
$ abs(b) = abs(ep^2 + 2 x ep + 2 dt e) <= abs(ep (ep + 2x + 2dt)) <= 3 eiter "as before" $
// here i'm using two of the inequalites that aren't obvious
$ abs(2 a b) <= 2 abs(dt^2 + 2 dt x) dot 3 eiter <= 2 abs(dt) abs(dt + 2 x) dot 3 eiter <= 2 dot 2 dot 2 dot 3 eiter <= 24 eiter $
which gives $b^2 + 2 a b <= 9 eiter^2 + 24 eiter <= 9 eiter + 24 eiter = 33 eiter$. Plugging this in, we have
$ 
  ee(X_t^2) <= 2 tbar^2 ee(dt^2) + (2 c4^2) slash c8^2 dot 4 tbar^2 ee(dt^2) + (2 c4^2)/c8^2 dot 33 eiter tbar^2 \
  <= ee(X_t^2) <= 2 tbar^2 ee(dt^2) + (2 c4^2) slash c8^2 dot 4 tbar^2 ee(dt^2) + (2 c4^2) slash c8^2 dot 33 (ee(dt^2) slash c100) tbar^2 \
  = tbar^2 ee(dt^2) (2 + 8 c4^2 slash c8^2 + 66c4^2 slash (c8^2 c100))
$