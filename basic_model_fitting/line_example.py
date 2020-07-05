# Originally cribbed from Jake VanderPlas' blog, because why not
# http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/

import numpy as np
import scipy.stats as st

np.random.seed(42)
true_alpha = 25.0
true_beta = 0.5
true_sigma = 10.0

xs = 100.0 * np.random.random(20)
ys = true_alpha + true_beta * xs

# J VdP then added more scatter in x but did not
# account for it in the model, which is silly.
# So let's not do that.
###xs = np.random.normal(xs, 10.0)
# And let's make the errors heteroscedastic,
# seeing how we will assert knowledge of them anyway.
sigmas = np.sqrt(st.chi2.rvs(15.0, size=len(xs)) / 15.0) * true_sigma # no special reason for using this distribution
ys = np.random.normal(ys, sigmas)




def ellipse(cov, center=np.zeros(2), level=0.683, ax=None, fmt='-', npts=100, **kwargs):
    """
    Useful function for plotting the error ellipse associated with a
    2-d Gaussian covariance matrix. Follows the post below, only
    with the size actually done correctly.
    https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    """
    theta = np.arange(npts) / (npts-1.0) * 2.0*np.pi
    sx = np.sqrt(cov[0,0])
    sy = np.sqrt(cov[1,1])
    rho = cov[0,1] / (sx*sy)
    x = np.cos(theta) * np.sqrt(1.0 + rho)
    y = np.sin(theta) * np.sqrt(1.0 - rho)
    pa = 0.25 * np.pi
    scl = np.sqrt(st.chi2.ppf(level, 2))
    newx = (x*np.cos(pa) - y*np.sin(pa)) * sx*scl + center[0]
    newy = (x*np.sin(pa) + y*np.cos(pa)) * sy*scl + center[1]
    if ax is None:
        return newx, newy
    ax.plot(newx, newy, fmt, **kwargs)
