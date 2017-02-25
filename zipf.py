#
# Fitting Zipf distribution given frequency data using Probabilistic Programming
#
# See: http://stats.stackexchange.com/questions/6780/how-to-calculate-zipfs-law-coefficient-from-a-set-of-top-frequencies
#

import theano.tensor as tt
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

data = np.array( [26486, 12053, 5052, 3033, 2536, 2391, 1444, 1220, 1152, 1039] )

N = len( data )

print( "Number of data points: %d" % N )

f = data
#f = data / np.sum( data )

def build_model():
    with pm.Model() as model:
        # unsure about the prior...
        #s = pm.Normal( 's', mu=0.0, sd=100 )
        #s = pm.HalfNormal( 's', sd=10 )
        s = pm.Gamma('s', alpha=1, beta=10)

        def logp( f ):
            r = tt.arange( 1, N+1 )
            return -s * tt.sum( f * tt.log(r) ) - tt.sum( f ) * tt.log( tt.sum(tt.power(1.0/r,s)) )

        pm.DensityDist( 'obs', logp=logp, observed={'f': data} )

    return model


def run( n_samples=10000 ):
    model = build_model()
    with model:
        start = pm.find_MAP()
        step = pm.NUTS( scaling=start )
        trace = pm.sample( n_samples, step=step, start=start )

    pm.summary( trace )
    pm.traceplot( trace )
    pm.plot_posterior( trace, kde_plot=True )
    plt.show()

if __name__ == '__main__':
    run()
