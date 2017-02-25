#
# Checking correctness of likelihood...
#

import sys

import numpy as np


def loglike( f, s ):
    N = len( f )
    r = np.arange( 1, N + 1 )
    return -s * np.sum( f * np.log(r) ) - np.sum( f ) * np.log( np.sum( np.power(1.0/r,s) ) )


def main():

    data = np.array([26486, 12053, 5052, 3033, 2536, 2391, 1444, 1220, 1152, 1039])

    s = 1.45041

    print( loglike(data,s) )


if __name__ == '__main__':
    main()
