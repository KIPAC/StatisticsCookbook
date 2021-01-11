import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats as st

def whist(x, smooth=True, kde_n=512, kde_range=None, bins='auto', plot=None,
              kde_kwargs={}, hist_kwargs={}, **kwargs):
    """
    Turn an array of samples, x, into an estimate of probability density at a
    discrete set of x values, possibly with some weights for the samples, and 
    possibly doing some smoothing.
    
    Return value is a dictionary with keys 'x' and 'density'.
    
    Calls scipy.stats.gaussian_kde if smooth=True, numpy.histogram if
    smooth=False. Additional options:
     kde_n (kde) - number of points to evaluate at (linearly covers kde_range)
     kde_range (kde) - range of kde evaluation (defaults to range of x)
     bins (histogram) - number of bins to use or array of bin edges
     plot (either) - if not None, plot the thing to this matplotlib device
     kde_kwargs - dictionary of options to gaussian_kde
     hist_kwargs - dictionary of options to histogram
     **kwargs - additional options valid for EITHER gaussian_kde or histogram,
       especially `weights`
    """
    if plot is not None:
        plot.hist(x, bins=bins, density=True, fill=False, **kwargs);
    if smooth:
        if kde_range is None:
            kde_range = (x.min(), x.max())
        h = {'x': np.linspace(kde_range[0], kde_range[1], kde_n)}
        h['density'] = st.gaussian_kde(x, **kde_kwargs, **kwargs)(h['x'])
    else:
        hi = np.histogram(x, bins=bins, density=True, **hist_kwargs, **kwargs)
        nb = len(hi[0])
        h = {'mids': 0.5*(hi[1][range(1,nb+1)]+hi[1][range(nb)]), 'density': hi[0]}
    if plot is not None:
        plot.plot(h['x'], h['density'], 'b-');
    return h

def whist_ci(h, sigmas=np.array([1.0,2.0]), prob=None, plot=None,
              plot_mode=True, plot_levels=True, plot_ci=True, fill_ci=True):
    """
    Accept a dictionary with keys 'x' and 'density' (e.g. from `whist`).
    Entries in 'x' must be equally spaced.
    
    Return the mode of the PDF, along with the endpoints of the HPD interval(s) 
    containing probabilities in `prob`, or number of "sigmas" given in `sigmas`.
    
    plot - if not None, plot the thing to this matplotlib thingy
    plot_mode, plot_levels, plot_ci, fill_ci - bells and whistles to include
    """
    if prob is None:
        prob = st.chi2.cdf(sigmas**2, df=1)
    mode = h['x'][np.argmax(h['density'])]
    imin = []
    imax = []
    iden = []
    ilev = []
    iprob = []
    theintervals = []
    o = np.argsort(-h['density']) # decreasing order
    k = -1
    for p in prob:
        k += 1
        for j in range(len(o)):
            if np.sum(h['density'][o[range(j)]]) / np.sum(h['density']) >= p: # NB failure if bins are not equal size
                reg = np.sort(o[range(j)])
                intervals = [[reg[0]]]
                if j > 0:
                    for i in range(1, len(reg)):
                        if reg[i] == reg[i-1]+1:
                            intervals[-1].append(reg[i])
                        else:
                            intervals.append([reg[i]])
                for i in range(len(intervals)):
                    imin.append(np.min(h['x'][intervals[i]]))
                    imax.append(np.max(h['x'][intervals[i]]))
                    iden.append(h['density'][o[j]])
                    ilev.append(p)
                    iprob.append( np.sum(h['density'][intervals[i]]) / np.sum(h['density']) )
                    theintervals.append(intervals[i])
                break
    imin = np.array(imin)
    imax = np.array(imax)
    ilev = np.array(ilev)
    iprob = np.array(iprob)
    iden = np.array(iden)
    ilow = imin - mode
    ihig = imax - mode
    icen = 0.5*(imin + imax)
    iwid = 0.5*(imax - imin)
    if plot is not None:
        if fill_ci:
            for i in range(len(ilow)-1, -1, -1):
                plot.fill(np.concatenate(([imin[i]],h['x'][theintervals[i]],[imax[i]])), 
                          np.concatenate(([0.0],h['density'][theintervals[i]],[0.0])), color=str(ilev[i]))
        plot.plot(h['x'], h['density'], 'k-');
        if plot_mode:
            plot.plot(mode, h['density'][o[0]], 'bo');
        for i in range(len(ilow)):
            if plot_ci:
                plot.plot([imin[i],imin[i]], [0.0,iden[i]], 'b-');
                plot.plot([imax[i],imax[i]], [0.0,iden[i]], 'b-');
            if plot_levels:
                plot.axhline(iden[i], color='g', ls='--')
    return {'mode':mode, 'level':ilev, 'prob':iprob, 'density':iden,
                'min':imin, 'max':imax, 'low':ilow, 'high':ihig, 'center':icen,
                'width':iwid}

def whist2d(x, y, smooth=None, plot=None, **kwargs):
    """
    Two-dimensional version of `whist`.

    Returns dictionary with entries 'x' and 'y' (1D arrays) and 'z' (2D array).
    
    **kwargs - options to pass on to numpy.histogram2d, e.g.
     weights
     bins
     (density=True is passed by fiat)
    
    Additional options:
     smooth - width of Gaussian smoothing (NOT True/False)
     plot - if not None, plot 2D histogram to this matplotlib thingy
    """
    h, xbreaks, ybreaks = np.histogram2d(x, y, density=True, **kwargs)
    if smooth is not None:
        h = gaussian_filter(h, smooth, mode='constant', cval=0.0)
    nx = len(xbreaks) - 1
    ny = len(ybreaks) - 1
    # transpose z so that x and y are actually x and y
    h = h.T
    if plot is not None:
        plt.imshow(h)
    return {'x': 0.5*(xbreaks[range(1,nx+1)]+xbreaks[range(nx)]),
            'y': 0.5*(ybreaks[range(1,ny+1)]+ybreaks[range(ny)]),
            'z': h
           }

def _get_contour_verts(cn):
    '''
    Get coordinates of the lines drawn by pyplot.contour.
    https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
    '''
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    return contours

def whist2d_ci(h, sigmas=np.array([1.0,2.0]), prob=None, plot=plt, plot_mode=True):
    """
    Two dimension version of whist_ci.

    Accepts a dictionary with keys 'x', 'y' and 'z' (1D, 1D, 2D arrays).
    'x' and 'y' entries must be equally spaced.

    Returns the mode of the PDF, along with the contours of the HPD regions(s)
    containing probabilities in `prob`.
     'mode' - array (length 2)
     'levels' - array (probabilities equivalent to sigmas, or reproduces prob)
     'contours' - list of (for each level) list of (for each contours) arrays
                  with shape (:,2) storing the contour vertices

    plot - if not None, plot the thing to this matplotlib device
           NB this MUST be a matplotlib thingy, or we can't find the contours
    """
    if plot is None:
        raise Exception("my_2D_hpd: `plot` argument may not be None, because python.")
    if prob is None:
        prob = st.chi2.cdf(sigmas**2, df=1)
    imode = np.unravel_index(np.argmax(h['z']), h['z'].shape)
    mode = np.array([h['x'][imode[1]], h['y'][imode[0]]])
    o = np.argsort(-h['z'], None) # decreasing order
    k = -1
    level = np.zeros(len(prob))
    j = 0
    for p in prob:
        k += 1
        for j in range(j, len(o)):
            if np.sum(h['z'][np.unravel_index(o[range(j)], h['z'].shape)]) / np.sum(h['z']) >= p:
                level[k] = h['z'][np.unravel_index(o[j], h['z'].shape)]
                break
    contours = []
    for lev in level:
        contours.append( _get_contour_verts(plot.contour(h['x'], h['y'], h['z'], levels=[lev], colors='k'))[0] )
    if plot_mode:
        plot.plot(mode[0], mode[1], 'bo')
    return {'mode':mode, 'levels':level, 'contours':contours}

def ci2D_plot(contours, plot, transpose=False, outline=True, fill=False,
                  Line2D_kwargs={}, fill_kwargs={}):
    '''
    `contours` argument should be an entry from the 'contours' element returned
    from whist2d_ci (a list of n*2 arrays)

    `plot` is a matplotlib thingy on which to plot

    Everything else is hopefully self explanatory
    '''
    if transpose:
        i = 1
        j = 0
    else:
        i = 0
        j = 1
    lkw = {'linestyle':'-', 'color':'k'}
    for k in Line2D_kwargs.keys():
        lkw[k] = Line2D_kwargs[k]
    fkw = {'color':'0.5'}
    for k in fill_kwargs.keys():
        fkw[k] = fill_kwargs[k]
    for con in contours:
        if fill:
            plot.fill(con[:,i], con[:,j], **fkw);
        if outline:
            plot.plot(con[:,i], con[:,j], **lkw);
