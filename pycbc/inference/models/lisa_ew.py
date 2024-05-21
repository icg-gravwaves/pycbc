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
import os
import numpy
import numpy.random

from .base import BaseModel

import pycbc.psd

from pycbc.waveform.pre_merger_waveform import (
    generate_data_lisa_pre_merger,
    generate_waveform_lisa_pre_merger,
)
from pycbc.psd.lisa_pre_merger import generate_pre_merger_psds
from pycbc.waveform.waveform import parse_mode_array
from .tools import marginalize_likelihood

global COUNTS
COUNTS = {}


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



class LISAEarlyWarningModel(BaseModel):
    r"""Ian is messing around

    Parameters
    ----------
    variable_params : (tuple of) string(s)
        A tuple of parameter names that will be varied.
    **kwargs :
        All other keyword arguments are passed to ``BaseModel``.


    """
    name = "lisa_ew"

    def __init__(
            self,
            variable_params,
            static_params=None,
            phase_marginalization=False,
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
        psd_duration = int(kwargs.pop('psd_duration'))
        inj_keys = [item for item in kwargs.keys() if item.startswith('injparam')]
        inj_params = {}
        for key in inj_keys:
            value = kwargs.pop(key)
            # Type conversion needed ... Ugly!!
            if key in ['injparam_run_phenomd', 'injparam_zero_noise']:
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
        
        if isinstance(phase_marginalization, str):
            phase_marginalization = strtobool(phase_marginalization)
        self.phase_marginalization = phase_marginalization
        if self.phase_marginalization:
            logging.warning(
                "Phase marginalization is enabled! "
                "This may leaded to incorrect results!"
            )

        # set up base likelihood parameters
        super().__init__(variable_params, **kwargs)

        self.static_params = parse_mode_array(static_params)

        length = int(tlen * sample_rate)
        flen = length // 2 + 1

        if psd_file is None:
            raise ValueError("Must specify a PSD file!")

        # Assume A & E PSDs are the same
        psd = pycbc.psd.from_txt(
            psd_file, flen, 1./tlen, 1./tlen, is_asd_file=False
        )
        self.psds_for_datagen = {}
        self.psds_for_datagen['LISA_A'] = psd
        self.psds_for_datagen['LISA_E'] = psd.copy()

        # Zero phase PSDs for whitening
        # Only store the frequency-domain PSDs
        self.whitening_psds = {}
        self.whitening_psds['LISA_A'] = generate_pre_merger_psds(
            psd_file,
            sample_rate=sample_rate,
            duration=psd_duration,
            kernel_length=psd_kernel_length,
        )["FD"]
        self.whitening_psds['LISA_E'] = generate_pre_merger_psds(
            psd_file,
            sample_rate=sample_rate,
            duration=psd_duration,
            kernel_length=psd_kernel_length,
        )["FD"]

        # Store data for doing likelihoods.
        curr_params = inj_params
        self.kernel_length = kernel_length
        self.window_length = window_length
        self.cutoff_time = cutoff_time
        self.extra_forward_zeroes = extra_forward_zeroes

        # Want to remove this!
        # FIXME: think this might be wrong now
        cutoff_time = self.cutoff_time + (curr_params['t_obs_start'] - curr_params['tc'])
        # Generate the pre-merger data
        # Returns time-domain data
        # Uses UIDs: 4235(0), 4236(0)
        data = generate_data_lisa_pre_merger(
            curr_params,
            psds_for_datagen=self.psds_for_datagen,
            psds_for_whitening=self.whitening_psds,
            seed=seed,
            window_length=self.window_length, 
            cutoff_time=cutoff_time,
            sample_rate=sample_rate,
            extra_forward_zeroes=self.extra_forward_zeroes,
        )
        self.lisa_a_strain = data["LISA_A"]
        self.lisa_e_strain = data["LISA_E"]

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

    def _loglikelihood(self):
        """Compute the pre-merger log-likelihood."""
        cparams = copy.deepcopy(self.static_params)
        cparams.update(self.current_params)
        # FIXME: think this is wrong
        cutoff_time = self.cutoff_time + (cparams['t_obs_start'] - cparams['tc'])
        # Generate the pre-merger waveform
        # These waveforms are whitened
        # Uses UIDs: 1234(0), 1235(0), 1236(0), 1237(0)
        ws = generate_waveform_lisa_pre_merger(
            cparams,
            psds_for_whitening=self.whitening_psds,
            window_length=self.window_length,
            sample_rate=self.sample_rate,
            cutoff_time=cutoff_time,
            kernel_length=self.kernel_length,
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

        return marginalize_likelihood(hs, hh, phase=self.phase_marginalization)
