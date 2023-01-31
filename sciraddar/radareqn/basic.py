import math


"""

1.) radareqpow()
2.) radareqrng()
3.) radareqsnr()
4.) radareqsarsnr()
5.) radareqsarpow()
6.) radareqsarrng()
7.) radareqsearchpap()
8.) radareqsearchrng()
9.) radareqsearchsnr()
10.) radarmetricplot()

"""

def peak_power(G, Pt, F, L, R, RCS=1):
    """
    This function calculates an estimate of peak power using the radar equation.
    Parameters:
    - G: Antenna gain (dBi)
    - Pt: Transmit power (W)
    - F: Frequency (Hz)
    - L: System loss
    - R: Range (m)
    - RCS: Radar cross-section (m^2)
    """
    lambda_ = 3e8/F
    range_loss = (4*math.pi*R/lambda_)**2
    P_peak = (Pt*G*G*RCS*L)/range_loss
    return P_peak


if __name__ == "__main__":
    # Example usage
    G = 50
    Pt = 50
    F = 9.6e9
    L = 10
    R = 100e3

    P_peak = peak_power(G, Pt, F, L, R)
    print("Peak power: {:.50f} W".format(P_peak))