# Copyright (C) 2018  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
This modules provides models that have analytic solutions for the
log likelihood.
"""

import copy
import logging

import pycbc.types

from .base import BaseModel

import pycbc.psd

from pycbc.waveform.pre_merger_waveform import (
    pre_process_data_lisa_pre_merger,
    generate_waveform_lisa_pre_merger,
)
from pycbc.psd.lisa_pre_merger import generate_pre_merger_psds
from pycbc.waveform.waveform import parse_mode_array
from .tools import marginalize_likelihood


# As per https://stackoverflow.com/questions/715417 this is
# distutils.util.strtobool, but it's being removed in python3.12 so recommend
# to copy source code as we've done here.
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))



class LISAPreMergerModel(BaseModel):
    r"""Model for pre-merger inference in LISA.

    Parameters
    ----------
    variable_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    static_params: Dict[str: Any]
        Dictionary of static parameters used for waveform generation.
    psd_file : str
        Path to the PSD file. Uses the same PSD file for LISA_A and LISA_E
        channels.
    **kwargs :
        All other keyword arguments are passed to ``BaseModel``.
    """
    name = "lisa_pre_merger"

    def __init__(
        self,
        variable_params,
        static_params=None,
        psd_file=None,
        **kwargs
    ):
        # Pop relevant values from kwargs
        cutoff_time = int(kwargs.pop('cutoff_time'))
        seed = int(kwargs.pop('seed'))
        kernel_length = int(kwargs.pop('kernel_length'))
        psd_kernel_length = int(kwargs.pop('psd_kernel_length'))
        window_length = int(kwargs.pop('window_length'))
        extra_forward_zeroes = int(kwargs.pop('extra_forward_zeroes'))
        tlen = int(kwargs.pop('tlen'))
        sample_rate = float(kwargs.pop('sample_rate'))
        data_file = kwargs.pop('data_file')
        inj_keys = [item for item in kwargs.keys() if item.startswith('injparam')]
        inj_params = {}
        for key in inj_keys:
            value = kwargs.pop(key)
            # Type conversion needed ... Ugly!!
            if key in ['injparam_run_phenomd']:
                value = strtobool(value)
            elif key in ['injparam_approximant']:
                pass  # Convert to string, so do nothing
            elif key in ['injparam_t_obs_start']:
                value = int(value)
            elif key in ['injparam_mode_array']:
                # If value is
                if "(" in value:
                    raise NotImplementedError
                elif "[" in value:
                    raise NotImplementedError
                else:
                    # e.g '22 33 44'
                    pass
            else:
                value = float(value)
            inj_params[key.replace('injparam_', '')] = value

        inj_params = parse_mode_array(inj_params)
        logging.info(f"Pre-merger injection parameters:\n {inj_params}")
        
        # set up base likelihood parameters
        super().__init__(variable_params, **kwargs)

        self.static_params = parse_mode_array(static_params)

        if psd_file is None:
            raise ValueError("Must specify a PSD file!")

        # Zero phase PSDs for whitening
        # Only store the frequency-domain PSDs
        logging.info("Generating pre-merger PSDs")
        self.whitening_psds = {}
        self.whitening_psds['LISA_A'] = generate_pre_merger_psds(
            psd_file,
            sample_rate=sample_rate,
            duration=tlen,
            kernel_length=psd_kernel_length,
        )["FD"]
        self.whitening_psds['LISA_E'] = generate_pre_merger_psds(
            psd_file,
            sample_rate=sample_rate,
            duration=tlen,
            kernel_length=psd_kernel_length,
        )["FD"]

        # Store data for doing likelihoods.
        curr_params = inj_params
        self.kernel_length = kernel_length
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.cutoff_time = cutoff_time
        self.extra_forward_zeroes = extra_forward_zeroes

        # Load the data from the file
        data = {}
        for channel in ["LISA_A", "LISA_E"]:
            data[channel] = pycbc.types.timeseries.load_timeseries(
                data_file,
                group=f"/{channel}",
            )

        # Want to remove this!
        # Need time from end of data (this includes the time after tc)
        cutoff_time = self.compute_cutoff_time(curr_params)
        # Pre-process the pre-merger data
        # Returns time-domain data
        # Uses UIDs: 4235(0), 4236(0)
        logging.info("Pre-processing pre-merger data")
        pre_merger_data = pre_process_data_lisa_pre_merger(
            data,
            sample_rate=sample_rate,
            psds_for_whitening=self.whitening_psds,
            window_length=self.window_length, 
            cutoff_time=cutoff_time,
            extra_forward_zeroes=self.extra_forward_zeroes,
        )

        self.lisa_a_strain = pre_merger_data["LISA_A"]
        self.lisa_e_strain = pre_merger_data["LISA_E"]

        # Frequency-domain data for computing log-likelihood
        self.lisa_a_strain_fd = pycbc.strain.strain.execute_cached_fft(
            self.lisa_a_strain,
            copy_output=True,
            uid=3223965
        )
        self.lisa_e_strain_fd = pycbc.strain.strain.execute_cached_fft(
            self.lisa_e_strain,
            copy_output=True,
            uid=3223967
        )

    def compute_cutoff_time(self, cparams):
        """Cutoff time including post merger data."""
        return (
            self.cutoff_time
            + (
                cparams['start_gps_time']
                + cparams['t_obs_start']
                - cparams['tc']
            )
        )

    def _loglikelihood(self):
        """Compute the pre-merger log-likelihood."""
        cparams = copy.deepcopy(self.static_params)
        cparams.update(self.current_params)
        # Compute cutoff times that includes
        cutoff_time = self.compute_cutoff_time(cparams)
        # Generate the pre-merger waveform
        # These waveforms are whitened
        # Uses UIDs: 1234(0), 1235(0), 1236(0), 1237(0)
        ws = generate_waveform_lisa_pre_merger(
            cparams,
            psds_for_whitening=self.whitening_psds,
            window_length=self.window_length,
            sample_rate=self.sample_rate,
            cutoff_time=cutoff_time,
            extra_forward_zeroes=self.extra_forward_zeroes,
        )
        wform_lisa_a = ws['LISA_A']
        wform_lisa_e = ws['LISA_E']

        # Compute <h|d> for each channel
        snr_A = pycbc.filter.overlap_cplx(
            wform_lisa_a,
            self.lisa_a_strain_fd,
            normalized=False,
        )
        snr_E = pycbc.filter.overlap_cplx(
            wform_lisa_e,
            self.lisa_e_strain_fd,
            normalized=False,
        )
        # Compute <h|h> for each channel
        a_norm = pycbc.filter.sigmasq(wform_lisa_a)
        e_norm = pycbc.filter.sigmasq(wform_lisa_e)

        hs = snr_A + snr_E
        hh = (a_norm + e_norm)

        return marginalize_likelihood(hs, hh, phase=False)
