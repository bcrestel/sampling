import numpy as np

def AutoCorrelationFunction(samples, maxlag):
    """
    Compute autocorrelation from np.array of samples
    Ref: https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    Inputs:
        samples = np.array where each row is a samples (and col = index)
        maxlag = integer giving maximum lag
    """
    ACF = []
    MEAN = samples.mean(axis=0)
    VAR = samples.var(axis=0)
    n = len(samples)
    for ii in range(1, maxlag):
        Sm = samples - MEAN
        ACF.append( float((Sm[:-ii]*Sm[ii:]).sum(axis=0) / ((n-ii)*VAR)) )
    return np.array(ACF)


def PlotACF(ACF, ax, color=None):
    """ Plot the autocorrelation function in a clean format """
    maxlag = ACF.size + 1
    if color == None:
        ax.vlines(range(1,maxlag), np.minimum(ACF, 0.), np.maximum(ACF, 0.))
    else:
        ax.vlines(range(1,maxlag), np.minimum(ACF, 0.), np.maximum(ACF, 0.), color)
    ax.plot([0,maxlag], [0.,0.],'k')
    ax.plot([0,maxlag], [0.05,0.05],'k--')
    ax.plot([0,maxlag], [-0.05,-0.05],'k--')
    return ax


def PlotACFs(ACF, ax, color=None, labels=None):
    """ Plot several autocorrelation functions in a clean format """
    maxlag = ACF[0].size + 1
    if color == None:
        for acf in ACF:
            ax.vlines(range(1,maxlag), np.minimum(acf, 0.), np.maximum(acf, 0.))
    else:
        if labels == None:
            for acf, cc in zip(ACF, color):
                ax.vlines(range(1,maxlag), np.minimum(acf, 0.), \
                np.maximum(acf, 0.), cc)
        else:
            for acf, cc, ll in zip(ACF, color, labels):
                ax.vlines(range(1,maxlag), np.minimum(acf, 0.), \
                np.maximum(acf, 0.), cc, label=ll)
    ax.plot([0,maxlag], [0.,0.],'k')
    ax.plot([0,maxlag], [0.05,0.05],'k--')
    ax.plot([0,maxlag], [-0.05,-0.05],'k--')
    return ax
