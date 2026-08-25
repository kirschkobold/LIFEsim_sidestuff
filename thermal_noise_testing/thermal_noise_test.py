def pn_dark_current(self) -> None:
    if self.detector_dark_current == 'MIRI':
        self.dark_current_pix = 0.2
        self.photon_rates_nchop['pn_dc'] = (np.sqrt(self.dark_current_pix * self.pix_per_wl)
                                                   * np.ones((self.wl_bins.shape[0])))
    elif self.detector_dark_current == 'manual':
        if self.dark_current_pix == None:
            raise ValueError('Dark current per pixel needs to be specified in manual mode')
        self.photon_rates_nchop['pn_dc'] = (np.sqrt(self.dark_current_pix * self.pix_per_wl)
                                                   * np.ones((self.wl_bins.shape[0])))
    else:
        raise ValueError('Unkown detector type')

def pn_thermal_background_detector(self) -> None:
    h = 6.62607e-34
    k = 1.380649e-23
    c = 2.99792e+8
    if self.detector_thermal == 'MIRI':
        # pitch - gap
        area_pixel = ((25 - 2) * 1e-6) ** 2
        det_wl_min = 5e-6
        det_wl_max = 28e-6
    else:
        raise ValueError('Unkown detector type')
    wl_bins = np.linspace(start=det_wl_min, stop=det_wl_max, num=self.wl_resolution, endpoint=True)
    B_photon = 2 * c / wl_bins ** 4 / (np.exp(h * c / (wl_bins * k * self.det_temp)) - 1)
    B_photon_int = np.trapz(y=B_photon, x=wl_bins)
    thermal_emission_det = 2 * np.pi * area_pixel * B_photon_int

    self.photon_rates_nchop['pn_tbd'] = (np.sqrt(thermal_emission_det * self.pix_per_wl)
                                                * np.ones((self.wl_bins.shape[0])))

def pn_thermal_primary_mirror(self) -> None:
    prefactor = 4 * self.primary_emmisivity * (
            np.pi * self.diameter_ap * self.magnification / self.f_number
            * self.secondary_primary_ratio / (1 - self.secondary_primary_ratio)
    ) ** 2
    thermal_emission_primary = prefactor * black_body(mode='wavelength',
                                                      bins=self.wl_bins,
                                                      width=self.wl_bin_widths,
                                                      temp=self.primary_temp)
    self.photon_rates_nchop['pn_tbpm'] = np.sqrt(thermal_emission_primary)
    
def pn_agnostic(self) -> None:
    if (self.eps_white is None) or (self.eps_cold is None) or (self.eps_hot is None):
        raise ValueError('Agnostic scaling variables need to be specified in agnostic mode')

    self.photon_rates_nchop['pn_ag_ht'] = (self.eps_hot * 0.342 * self.diameter_ap ** 2
                                                  * black_body(mode='wavelength',
                                                               bins=self.wl_bins,
                                                               width=self.wl_bin_widths,
                                                               temp=self.temp_star)
                                                  / 4 / black_body(mode='wavelength',
                                                                   bins=np.array((0.5e-6)),
                                                                   width=np.array((0.05e-6)),
                                                                   temp=self.temp_star)
                                                  )

    self.photon_rates_nchop['pn_ag_cld'] = (self.eps_cold * 0.947 * self.diameter_ap ** 2
                                                   * black_body(mode='wavelength',
                                                                bins=self.wl_bins,
                                                                width=self.wl_bin_widths,
                                                                temp=self.agnostic_spacecraft_temp)
                                                   / 4 / black_body(mode='wavelength',
                                                                    bins=np.array((58e-6)),
                                                                    width=np.array((0.05e-6)),
                                                                    temp=50.)
                                                   )

    self.photon_rates_nchop['pn_ag_wht'] = self.eps_white * np.ones_like(self.wl_bins)

def fundamental_collect(self):
    self.photon_rates_nchop['fundamental'] = np.sqrt(self.photon_rates_nchop['pn_sgl'] ** 2
                                                             + self.photon_rates_nchop['pn_ez'] ** 2
                                                             + self.photon_rates_nchop['pn_lz'] ** 2)

    # because of the incoherent combination of the final outputs, see Mugnier 2006
    if self.simultaneous_chopping:
        self.photon_rates_chop['fundamental'] *= np.sqrt(2)

    self.photon_rates_nchop['snr'] = (self.photon_rates_nchop['signal']
                                             / self.photon_rates_nchop['fundamental'])

    self.photon_rates_chop['pn_sgl'] = self.photon_rates_nchop['pn_sgl']
    self.photon_rates_chop['pn_ez'] = self.photon_rates_nchop['pn_ez']
    self.photon_rates_chop['pn_lz'] = self.photon_rates_nchop['pn_lz']
    self.photon_rates_chop['pn_dc'] = self.photon_rates_nchop['pn_dc']
    self.photon_rates_chop['pn_tbd'] = self.photon_rates_nchop['pn_tbd']
    self.photon_rates_chop['pn_tbpm'] = self.photon_rates_nchop['pn_tbpm']
    self.photon_rates_chop['pn_ag_ht'] = self.photon_rates_nchop['pn_ag_ht']
    self.photon_rates_chop['pn_ag_cld'] = self.photon_rates_nchop['pn_ag_cld']
    self.photon_rates_chop['pn_ag_wht'] = self.photon_rates_nchop['pn_ag_wht']
    self.photon_rates_chop['fundamental'] = self.photon_rates_nchop['fundamental']
    self.photon_rates_chop['snr'] = self.photon_rates_nchop['snr']