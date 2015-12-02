import numpy as np
import matplotlib.pyplot as plt

def VisualizeBurninTimeSeries(samples, nbrows, nbcols, parameters,\
maxsample=500, nbrepeats=10, colname='p', rowname='t'):
    """ Plot a few time series to help identify the burn-in part
    Inputs:
        samples = np.array of dimensions (timesteps x parameters x samples)
        nbrows, nbcols = dimension of plots
        parameters = list containing indices of parameters to plot; must have
len(parameters) == nbrows*nbcols
        nbrepeats = nb of different plots you want """
    assert len(parameters) == nbrows*nbcols
    fig = plt.figure()
    txtsup = fig.suptitle(rowname+'=')
    AX = []
    for ii in range(nbrows*nbcols):
        AX.append(fig.add_subplot(nbrows, nbcols, ii+1))
    plt.show(block=False)
    # Repeat:
    for kk in range(nbrepeats):
        tt = np.random.randint(0, samples.shape[0]-1)
        for ax, pp in zip(AX, parameters):
            ax.clear()
            ax.plot(samples[tt,pp,:maxsample])
            ax.set_title(colname+' {}'.format(pp))
        txtsup.set_text(rowname+'={}'.format(tt))
        plt.draw()
        wait = raw_input('Press <ENTER> to continue')
