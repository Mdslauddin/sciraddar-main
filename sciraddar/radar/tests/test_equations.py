"""
__all__ = ['radareqpow','radareqrng','radareqsnr','radareqsarsnr','radareqsarpow','radareqsarrng',
'radareqsearchpap','radareqsearchrng','radareqsearchsnr','radarmetricplot']

"""
import math

def peak_power_estimate(c, k, T, RCS, losses, range_max, pulse_width, bandwidth, snr_threshold):
    """
    c: the speed of light in meters per second
    k: the Boltzmann constant in joules per Kelvin
    T: the temperature in Kelvin
    RCS: the radar cross section in square meters
    losses: the system losses in decibels
    range_max: the maximum range in meters
    pulse_width: the pulse width in seconds
    bandwidth: the bandwidth in Hertz
    snr_threshold: the minimum signal-to-noise ratio required for detection
    """
    # convert range to km
    range_max_km = range_max / 1000

    # calculate noise power
    noise_power = k * T * bandwidth

    # convert losses to noise figure
    noise_figure = 10 ** (losses / 10)

    # calculate signal-to-noise ratio
    snr = 10 ** (snr_threshold / 10)

    # calculate peak power
    peak_power = (4 * math.pi * (range_max_km ** 2) * noise_power * snr * noise_figure) / (RCS * c * pulse_width)

    return peak_power




def max_range_estimate(c, k, T, RCS, losses, peak_power, pulse_width, bandwidth, snr_threshold):
    """
    The function takes in the following parameters:

    c: the speed of light in meters per second
    k: the Boltzmann constant in joules per Kelvin
    T: the temperature in Kelvin
    RCS: the radar cross section in square meters
    losses: the system losses in decibels
    peak_power: the peak power in watts
    pulse_width: the pulse width in seconds
    bandwidth: the bandwidth in Hertz
    snr_threshold: the minimum signal-to-noise ratio required for detection
    """
    # convert peak power to dBW
    peak_power_db = 10 * math.log10(peak_power)

    # calculate noise power
    noise_power = k * T * bandwidth

    # convert losses to noise figure
    noise_figure = 10 ** (losses / 10)

    # calculate range
    range_max = math.sqrt((peak_power_db - noise_figure - 10 * math.log10(RCS) - 10 * math.log10(bandwidth) + 228.6) / (-20))

    # convert range to meters
    range_max *= 1000

    return range_max




def snr_estimate(c, k, T, RCS, losses, peak_power, range, pulse_width, bandwidth):
    """
    The function takes in the following parameters:

    c: the speed of light in meters per second
    k: the Boltzmann constant in joules per Kelvin
    T: the temperature in Kelvin
    RCS: the radar cross section in square meters
    losses: the system losses in decibels
    peak_power: the peak power in watts
    range: the range in meters
    pulse_width: the pulse width in seconds
    bandwidth: the bandwidth in Hertz
                The function first converts the range from meters to kilometers, and then calculates the noise power and noise figure based on the input values. It then uses the radar equation to calculate the signal power based on the input values. The signal-to-noise ratio is then calculated, converted to dB, and returned. You can call this function with different input parameters to get different SNR estimates.


    """
    # convert range to km
    range_km = range / 1000

    # calculate noise power
    noise_power = k * T * bandwidth

    # convert losses to noise figure
    noise_figure = 10 ** (losses / 10)

    # calculate signal power
    signal_power = (peak_power * (RCS ** 2) * (range_km ** 4)) / ((4 * math.pi) ** 3 * c ** 2 * noise_power * pulse_width * bandwidth)

    # calculate signal-to-noise ratio
    snr = signal_power / noise_power

    # convert to dB
    snr_db = 10 * math.log10(snr)

    return snr_db


import numpy as np

def sar_snr(image, background_mask=None):
    """
    image: a 2D numpy array representing the SAR image
    background_mask: an optional boolean mask indicating the background 
    Computes the signal-to-noise ratio (SNR) of a SAR image.
    :param image: a 2D numpy array representing the SAR image
    :param background_mask: a boolean mask indicating the background area of the image
    :return: the SNR of the image
    """

    if background_mask is None:
        # If no background mask is provided, assume all non-zero pixels are signal and the rest are noise
        signal_mask = image > 0
        noise_mask = np.logical_not(signal_mask)
    else:
        # Use the provided background mask to separate signal and noise
        signal_mask = np.logical_not(background_mask)
        noise_mask = background_mask

    # Compute the mean signal and noise values
    mean_signal = np.mean(image[signal_mask])
    mean_noise = np.mean(image[noise_mask])

    # Compute the standard deviation of the noise
    std_noise = np.std(image[noise_mask])

    # Compute the SNR
    snr = (mean_signal - mean_noise) / std_noise

    return snr



def sar_min_peak_power(c, range_resolution, pulse_bandwidth, antenna_area, snr, system_loss, target_rcs):
    """
    c: the speed of light in meters per second
    range_resolution: the desired range resolution in meters
    pulse_bandwidth: the SAR system's pulse bandwidth in Hertz
    antenna_area: the SAR system's antenna area in square meters
    snr: the desired signal-to-noise ratio
    system_loss: the SAR system's total losses in decibels
    target_rcs: the target radar cross section of the imaged object in square meters
    Computes the minimum peak transmit power required for a SAR system to achieve a desired resolution.
    :param c: the speed of light in m/s
    :param range_resolution: the desired range resolution in meters
    :param pulse_bandwidth: the SAR system's pulse bandwidth in Hz
    :param antenna_area: the SAR system's antenna area in square meters
    :param snr: the desired signal-to-noise ratio
    :param system_loss: the SAR system's total losses in dB
    :param target_rcs: the target radar cross section of the imaged object in square meters
    :return: the minimum peak transmit power in watts
    """

    # Compute the required signal power using the desired SNR and the noise power
    k = 1.38e-23  # Boltzmann constant in J/K
    T = 290  # Temperature in K (standard room temperature)
    noise_power = k * T * pulse_bandwidth  # Noise power in watts
    signal_power = noise_power * snr  # Signal power in watts

    # Compute the minimum detectable RCS at the desired range resolution and SNR
    min_rcs = (signal_power * (4 * math.pi) ** 3 * (range_resolution ** 4) * antenna_area ** 2) / (c ** 2 * pulse_bandwidth * noise_power)

    # Compute the required peak power based on the minimum detectable RCS and the target RCS
    min_peak_power = (4 * math.pi) ** 3 * c ** 2 * (min_rcs / target_rcs) * noise_power * pulse_bandwidth * antenna_area ** 2 / (range_resolution ** 4)

    # Adjust for system losses
    min_peak_power /= 10 ** (system_loss / 10)

    return min_peak_power


import math

def sar_max_range(c, range_resolution, pulse_bandwidth, antenna_area, snr, system_loss, peak_power, target_rcs):
    """
    c: the speed of light in meters per second
    range_resolution: the SAR system's range resolution in meters
    pulse_bandwidth: the SAR system's pulse bandwidth in Hertz
    antenna_area: the SAR system's antenna area in square meters
    snr: the desired signal-to-noise ratio
    system_loss: the SAR system's total losses in decibels
    peak_power: the SAR system's peak transmit power in watts
    target_rcs: the target radar cross section of the imaged object in square meters
    
    Computes the maximum detectable range for a SAR system.
    :param c: the speed of light in m/s
    :param range_resolution: the SAR system's range resolution in meters
    :param pulse_bandwidth: the SAR system's pulse bandwidth in Hz
    :param antenna_area: the SAR system's antenna area in square meters
    :param snr: the desired signal-to-noise ratio
    :param system_loss: the SAR system's total losses in dB
    :param peak_power: the SAR system's peak transmit power in watts
    :param target_rcs: the target radar cross section of the imaged object in square meters
    :return: the maximum detectable range in meters
    """

    # Compute the required minimum detectable RCS
    k = 1.38e-23  # Boltzmann constant in J/K
    T = 290  # Temperature in K (standard room temperature)
    noise_power = k * T * pulse_bandwidth  # Noise power in watts
    min_rcs = (snr * noise_power * (c ** 2) * (range_resolution ** 4)) / (4 * math.pi ** 3 * (antenna_area ** 2))

    # Compute the maximum detectable range
    max_range = ((peak_power * target_rcs) / (4 * math.pi) ** 3) ** 0.25 * (range_resolution ** 2) ** 0.5 / math.sqrt(min_rcs) / c

    # Adjust for system losses
    max_range *= 10 ** (system_loss / 20)

    return max_range



def search_radar_pap(max_range, radar_cross_section, noise_figure, system_loss, wavelength, pulse_duration, prf, peak_power):
    """
    max_range: the maximum detection range in meters
    radar_cross_section: the target radar cross section in square meters
    noise_figure: the radar system's noise figure in decibels
    system_loss: the radar system's total losses in decibels
    wavelength: the radar system's operating wavelength in meters
    pulse_duration: the radar system's pulse duration in seconds
    prf: the radar system's pulse repetition frequency in Hertz
    peak_power: the radar system's peak transmit power in watts
    
    Computes the power-aperture product (PAP) for a search radar system.
    :param max_range: the maximum detection range in meters
    :param radar_cross_section: the target radar cross section in square meters
    :param noise_figure: the radar system's noise figure in decibels
    :param system_loss: the radar system's total losses in decibels
    :param wavelength: the radar system's operating wavelength in meters
    :param pulse_duration: the radar system's pulse duration in seconds
    :param prf: the radar system's pulse repetition frequency in Hertz
    :param peak_power: the radar system's peak transmit power in watts
    :return: the power-aperture product in watts-square meters
    """

    # Compute the effective area of the antenna
    antenna_diameter = wavelength / math.pi
    antenna_area = math.pi * (antenna_diameter / 2) ** 2
    eff_area = antenna_area / 4

    # Compute the minimum detectable power
    k = 1.38e-23  # Boltzmann constant in J/K
    t = 290  # Temperature in K (standard room temperature)
    noise_power = k * t * eff_area * noise_figure * pulse_duration
    min_detectable_power = 4 * math.pi ** 3 * noise_power * (max_range ** 4) / (radar_cross_section * (wavelength ** 2) * prf)

    # Compute the power-aperture product
    pap = min_detectable_power / peak_power

    # Adjust for system losses
    pap *= 10 ** (-system_loss / 10)

    return pap



def search_radar_max_range(radar_cross_section, peak_power, min_detectable_power, wavelength, pulse_duration, prf, antenna_diameter, system_loss, noise_figure):
    """
    radar_cross_section: the target radar cross section in square meters
    peak_power: the radar system's peak transmit power in watts
    min_detectable_power: the minimum detectable power in watts
    wavelength: the radar system's operating wavelength in meters
    pulse_duration: the radar system's pulse duration in seconds
    prf: the radar system's pulse repetition frequency in Hertz
    antenna_diameter: the diameter of the antenna in meters
    system_loss: the radar system's total losses in decibels
    noise_figure: the radar system's noise figure in decibels
    
    Computes the maximum detectable range for a search radar system.
    :param radar_cross_section: the target radar cross section in square meters
    :param peak_power: the radar system's peak transmit power in watts
    :param min_detectable_power: the minimum detectable power in watts
    :param wavelength: the radar system's operating wavelength in meters
    :param pulse_duration: the radar system's pulse duration in seconds
    :param prf: the radar system's pulse repetition frequency in Hertz
    :param antenna_diameter: the diameter of the antenna in meters
    :param system_loss: the radar system's total losses in decibels
    :param noise_figure: the radar system's noise figure in decibels
    :return: the maximum detectable range in meters
    """

    # Compute the effective area of the antenna
    antenna_area = math.pi * (antenna_diameter / 2) ** 2
    eff_area = antenna_area / 4

    # Compute the maximum detectable range
    k = 1.38e-23  # Boltzmann constant in J/K
    t = 290  # Temperature in K (standard room temperature)
    noise_power = k * t * eff_area * noise_figure * pulse_duration
    max_range = (radar_cross_section * (wavelength ** 2) * prf * peak_power) / (4 * math.pi ** 3 * noise_power * eff_area * min_detectable_power)

    # Adjust for system losses
    max_range *= math.sqrt(10 ** (system_loss / 10))

    return max_range



import math

def search_radar_range_dependent_snr(radar_cross_section, peak_power, wavelength, pulse_duration, prf, antenna_diameter, system_loss, noise_figure, range_array):
    """
    radar_cross_section: the target radar cross section in square meters
    peak_power: the radar system's peak transmit power in watts
    wavelength: the radar system's operating wavelength in meters
    pulse_duration: the radar system's pulse duration in seconds
    prf: the radar system's pulse repetition frequency in Hertz
    antenna_diameter: the diameter of the antenna in meters
    system_loss: the radar system's total losses in decibels
    noise_figure: the radar system's noise figure in decibels
    range_array: an array of ranges in meters for which to compute the range-dependent SNR
    
    Computes the range-dependent signal-to-noise ratio (SNR) for a search radar system.
    :param radar_cross_section: the target radar cross section in square meters
    :param peak_power: the radar system's peak transmit power in watts
    :param wavelength: the radar system's operating wavelength in meters
    :param pulse_duration: the radar system's pulse duration in seconds
    :param prf: the radar system's pulse repetition frequency in Hertz
    :param antenna_diameter: the diameter of the antenna in meters
    :param system_loss: the radar system's total losses in decibels
    :param noise_figure: the radar system's noise figure in decibels
    :param range_array: an array of ranges in meters
    :return: an array of the range-dependent SNR values
    """

    # Compute the effective area of the antenna
    antenna_area = math.pi * (antenna_diameter / 2) ** 2
    eff_area = antenna_area / 4

    # Compute the noise power
    k = 1.38e-23  # Boltzmann constant in J/K
    t = 290  # Temperature in K (standard room temperature)
    noise_power = k * t * eff_area * noise_figure * pulse_duration

    # Compute the peak received power and range-dependent SNR
    peak_received_power = (radar_cross_section * peak_power * (wavelength ** 2)) / (4 * math.pi ** 3 * range_array ** 4)
    snr = (peak_received_power ** 2) / (noise_power * eff_area * prf)

    # Adjust for system losses
    snr *= 10 ** (-system_loss / 10)

    return snr



import matplotlib.pyplot as plt

def plot_radar_performance(radar_metric, range_array, xlabel, ylabel, title):
    """
    radar_metric: an array of the performance metric (e.g. SNR, maximum detectable range) for each range value in range_array
    range_array: an array of ranges in meters for which the performance metric was computed
    xlabel: the label for the x-axis
    ylabel: the label for the y-axis
    title: the title of the plot
    
    Plots a radar system's performance metric against target range.
    :param radar_metric: an array of the performance metric for each range value in range_array
    :param range_array: an array of ranges in meters
    :param xlabel: the label for the x-axis
    :param ylabel: the label for the y-axis
    :param title: the title of the plot
    :return: None
    """
    plt.plot(range_array, radar_metric)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
