from __future__ import division

from functools import wraps

import numpy as np
from pandas import Series
from pandas import DataFrame
class Parabolic_sar():
    def __init__(self, s: DataFrame, params: dict={'af':0.02, 'amax':0.2}, high='high', Low='Low') -> None:
        self.values = self.calc_sar(s, params['af'], params['amax'], high, Low)

    def calc_sar(self, s, af, amax, high, Low) -> Series:
        high, Low = np.array(s[high]), np.array(s[Low])

        # Starting values
        sig0, xpt0, af0 = True, high[0], af
        sar = [Low[0] - (high - Low).std()]

        for i in range(1, len(s)):
            sig1, xpt1, af1 = sig0, xpt0, af0

            lmin = min(Low[i - 1], Low[i])
            lmax = max(high[i - 1], high[i])

            if sig1:
                sig0 = Low[i] > sar[-1]
                xpt0 = max(lmax, xpt1)
            else:
                sig0 = high[i] >= sar[-1]
                xpt0 = min(lmin, xpt1)

            if sig0 == sig1:
                sari = sar[-1] + (xpt1 - sar[-1])*af1
                af0 = min(amax, af1 + af)

                if sig0:
                    af0 = af0 if xpt0 > xpt1 else af1
                    sari = min(sari, lmin)
                else:
                    af0 = af0 if xpt0 < xpt1 else af1
                    sari = max(sari, lmax)
            else:
                af0 = af
                sari = xpt0

            sar.append(sari)

        # return Series(sar, index=s.index)
        return sar

    def get_values(self):
        return self.values