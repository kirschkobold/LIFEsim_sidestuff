from astropy.modeling import models
from astropy import constants, units as u
#from jwst_backgrounds import jbt
import numpy as np
#from scipy.interpolate import interp1d
from astropy import constants as const
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

import numpy as np
from typing import Union

from lifesim.util import constants

h = const.h.value #Planck constant

def BB(T,w):
    bb = models.BlackBody(temperature=T*u.K, scale=1.0 * u.ph / (u.m ** 2 * u.micron * u.s * u.sr))
    return bb(w*u.micron)

def integrate_BB(T,w_min,w_max):
    waves = np.linspace(w_min,w_max,10000)
    BBs = BB(T,waves)
    ret = np.trapezoid(x=waves*u.micron, y=BBs).to('ph s-1 m-2 sr-1')
    return ret



def compare_zod(T,w,e,Z_T,Z_OD):
    return BB(T,w)*e/(BB(Z_T,w)*Z_OD)

def find_temp(target_factor,w,e,Z_T,Z_OD):
    def test_func(T):
        x = compare_zod(T,w,e,Z_T,Z_OD)-target_factor
        return np.abs(x)
    res = minimize_scalar(test_func,bounds=(1,60), method='bounded')
    return np.round(res.x,1)

def find_temp2(target_factor,w_min,w_max,zod):
    def test_func(T):
        x = integrate_BB(T,w_min,w_max)/zod-target_factor
        return np.abs(x)
    res = minimize_scalar(test_func,bounds=(1,60), method='bounded')
    return np.round(res.x,1)



def planck_law(x: np.ndarray,
               temp: Union[float, np.ndarray],
               mode: str):
    """
    Calculates the photon flux emitted from a black body according to Planck's law in the
    wavelength or frequency regime

    Parameters
    ----------
    x : np.ndarray
        The frequency of wavelength at which the photon fluxes are calculated in [Hz] or [m]
    temp : Union[float, np.ndarray]
        The temperature of the black body
    mode : str
        If ``x`` is given in [Hz], set ``mode = 'frequency'. If ``x`` is given in [m], set
        ``mode = 'wavelength'

    Raises
    ------
    ValueError
        If the mode is not recognized

    Returns
    -------
    fgamma : np.ndarray
        The photon flux at the respective wavelengths or frequencies
    """

    # select the correct mode
    if mode == 'wavelength':

        # account for the temperature being zero at some pixels
        with np.errstate(divide='ignore'):

            # the Planck law divided by the photon energy to obtain the photon flux
            fgamma = 2 * constants.c / (x**4) / \
               (np.exp(constants.h * constants.c / x / constants.k / temp) - 1)
    elif mode == 'frequency':

        # account for the temperature being zero at some pixels
        with np.errstate(divide='ignore'):

            # the Planck law divided by the photon energy to obtain the photon flux
            fgamma = np.where(temp == 0,
                              0,
                              2 * x**2 / (constants.c**2) /
                              (np.exp(constants.h * x / constants.k / temp))-1.)
    else:
        raise ValueError('Mode not recognised')

    return fgamma

def black_body(mode: str,
               bins: np.ndarray,
               width: np.ndarray,
               temp: Union[float, np.ndarray],
               radius: float = None,
               distance: float = None):
    """
    Calculates the black body photon flux in wavelength or frequency as well as for planetary or
    stellar sources

    Parameters
    ----------
    mode : str
        Defines the mode of the ``black_body`` function.
            - ``mode = 'wavelength'`` : Clean photon flux black body spectrum over wavelength is
              returned. Parameters used are ``bins``, ``width`` and ``temp``
            - ``mode = 'frequency'`` : Clean photon flux black body spectrum over frequency is
              returned. Parameters used are ``bins``, ``width`` and ``temp``
            - ``mode = 'star'`` : Photon flux black body spectrum received from a star of specified
              radius from the specified distance. All parameters are used. In this mode, the
              parameter ``bins`` needs to be in wavelength
            - ``mode = 'planet'`` : Photon flux black body spectrum received from a planet of
              specified radius from the specified distance. All parameters are used. In this mode,
              the parameter ``bins`` needs to be in wavelength
    bins : np.ndarray
        The wavelength or frequency bins at which the black body is evaluated in [m] or [Hz]
        respectively
    width : np.ndarray
        The width of the wavelength or frequency bins to integrate over the black body spectrum in
        [m] or [Hz] respectively
    temp : Union[float, np.ndarray]
        The temperature of the black body
    radius : float
        The radius of the spherical black body object. For ``mode = 'star'`` in [sun_radii], for
        ``mode = 'planet'`` in [earth_radii]
    distance : float
        The distance between the instrument and the observed object in [pc]

    Raises
    ------
    ValueError
        If the mode is not recognized

    Returns
    -------
    fgamma : np.ndarray
        The photon flux at the respective wavelengths or frequencies
    """

    if mode == 'star':
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='wavelength') * width \
                 * np.pi * ((radius * constants.radius_sun) / (distance * constants.m_per_pc)) ** 2
    elif mode == 'planet':
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='wavelength') * width \
                 * np.pi * ((radius * constants.radius_earth) / (distance * constants.m_per_pc)) ** 2
    elif mode == 'wavelength':
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='wavelength') * width
    elif mode == 'frequency':
        # TODO remove hardcoded np.newaxis solution. The redim is needed for the PhotonNoiseExozodi
        #   class
        fgamma = planck_law(x=bins,
                            temp=temp,
                            mode='frequency') * width[:, np.newaxis, np.newaxis]
    else:
        raise ValueError('Mode not recognised')

    return fgamma

def get_wl_bins_const_spec_res():
        """
        Create the wavelength bins for the given spectral resolution and wavelength limits.
        """
        wl_edge = 4
        wl_bins = []
        wl_bin_widths = []
        wl_bin_edges = [wl_edge]

        while wl_edge < 18.5:

            # set the wavelength bin width according to the spectral resolution
            wl_bin_width = wl_edge / 20 / \
                           (1 - 1 / 20 / 2)

            # make the last bin shorter when it hits the wavelength limit
            if wl_edge + wl_bin_width > 18.5:
                wl_bin_width = 18.5 - wl_edge

            # calculate the center and edges of the bins
            wl_center = wl_edge + wl_bin_width / 2
            wl_edge += wl_bin_width

            wl_bins.append(wl_center)
            wl_bin_widths.append(wl_bin_width)
            wl_bin_edges.append(wl_edge)

        # convert everything to [m]
        wl_bins = np.array(wl_bins) * 1e-6  # in m
        wl_bin_widths = np.array(wl_bin_widths) * 1e-6  # in m
        wl_bin_edges = np.array(wl_bin_edges) * 1e-6  # in m

        return wl_bins, wl_bin_widths, wl_bin_edges

if __name__ == "__main__":
    
    Z_temp = 286
    Z_OD = 1e-7
    short_wave = 4 #microns
    long_wave = 18.5 #microns

    resolution = 100
    L_detector = 18.5
    NA = 0.33

    mirror_emissivity = 0.025
    bc_emissivity = 1


    better_than_factor = 1/3

    print(f"Zodiacal background is at {Z_temp} K with an optical depth of {Z_OD}")
    print("")

    T_mirror = find_temp(better_than_factor,long_wave,mirror_emissivity,Z_temp,Z_OD)

    print(f"Mirrors should be at a temperature where the emissivity is less than the zodiacal background at longest wavelength")
    print(f"Mirror should be at a temperature of {T_mirror} K")
    print(f"")

    T_scattering = find_temp(better_than_factor,long_wave,bc_emissivity,Z_temp,Z_OD)

    print(f"Fibre sees more thermal background than just the mirror, as BC has ~100% emissivity")
    print(f"Scattering from the BC leads to a temperature of {T_scattering} K")
    print(f"")

    Zod_radiance_at_shortest_wavelength = Z_OD*integrate_BB(Z_temp,short_wave,short_wave+short_wave/resolution)
    T_spectrograph = find_temp2(better_than_factor*NA**2,short_wave,L_detector,Zod_radiance_at_shortest_wavelength)

    print(f"Spectrograph has a resolution of {resolution}, and short channels are sensitive to long wave background")
    print(f"The spectrograph should be at a temperature of {T_spectrograph} K")
    print("")

    
    #T_detector = find_temp(better_than_factor/resolution/2/np.pi,L_detector,1,Z_temp,Z_OD)
    T_detector = find_temp2(better_than_factor*NA**2/2/np.pi,short_wave,L_detector,Zod_radiance_at_shortest_wavelength)

    print(f"The detector sees the hemisphere of a very cold stop, and also has a different cutoff wavelength of {L_detector} um")
    print(f"The temperature of the very cold stop must be {T_detector} K")
    print("")

    print(f"Finally the detector must also be cooled down (likely on the level of <10 K)")
    
    wl_bins, wl_bin_widths, wl_bin_edges = get_wl_bins_const_spec_res()

    mirror_temp = 40
    mirror_area = 4. * np.pi * (2 / 2.) ** 2
    emissivity = 0.9

    # calculate the black body radiation emitted by the mirror
    # emissivity per wavelength bin?
    mirror_bb = emissivity * black_body(mode='wavelength',
                                            bins=wl_bins,
                                            width=wl_bin_widths,
                                            temp=mirror_temp)

    # integrate over area and solid angle
    tm_leak = mirror_bb * mirror_area * np.pi
    print(f"Thermal mirror leakage is {tm_leak} photon s-1 per wavelength bin")