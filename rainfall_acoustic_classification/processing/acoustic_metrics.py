#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: Acoustic Metrics
=======================
Exaustive acoustic metric extraction for one sample.
Uses a specialized scalar extractor to bypass all NumPy ambiguity errors.

Metadata
--------
:Author: Giovanni G. R. Milan
:Date: 2026-03-26
:Version: 4.0.0
:Status: Development
"""
from typing import Optional, Dict, Any
import numpy as np
import scipy.stats as stats
from librosa import feature as libfeat
from librosa import power_to_db
from maad import sound as maddsd 
from maad import features as maadfeat
from pywt import wavedec as pywtwavedec
from dataclasses import dataclass, fields
from rainfall_acoustic_classification.utils import get_standard_logger


@dataclass(frozen=True)
class AcousticMetricsConfig:
    """
    Configuration DTO for the Acoustic Metrics Extraction process.

    Implements strict Separation of Concerns (Normalization vs Validation).

    Parameters
    ----------
    sample_rate : int, default=24000
        Target sample rate for the audio analysis. Must be strictly positive.
    fft_window_size : int, default=1024
        Size of the Fast Fourier Transform (FFT) window. Must be strictly positive.
    """
    sample_rate: int = 24000
    fft_window_size: int = 1024

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'AcousticMetricsConfig':
        """
        Instantiates the configuration by filtering only valid fields from kwargs.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments containing potential configuration parameters.

        Returns
        -------
        AcousticMetricsConfig
            An immutable instance of the configuration object.
        """
        valid_fields = {f.name for f in fields(cls)}
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered_args)

    def __post_init__(self) -> None:
        """
        Post-initialization hook to trigger data normalization and validation.
        """
        self._normalize()
        self._validate()
    
    def _normalize(self) -> None:
        """
        Normalizes internal attributes by casting them to their expected types.
        Fails silently to allow the `_validate` method to catch persistent errors.
        """
        try:
            if not isinstance(self.sample_rate, int):
                object.__setattr__(self, 'sample_rate', int(self.sample_rate))
            if not isinstance(self.fft_window_size, int):
                object.__setattr__(self, 'fft_window_size', int(self.fft_window_size))
        except (ValueError, TypeError):
            pass

    def _validate(self) -> None:
        """
        Validates the logical boundaries of the configuration attributes.

        Raises
        ------
        ValueError
            If `sample_rate` or `fft_window_size` are not strictly positive integers.
        """
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive integer. Got: {self.sample_rate}")
        
        if not isinstance(self.fft_window_size, int) or self.fft_window_size <= 0:
            raise ValueError(f"fft_window_size must be a positive integer. Got: {self.fft_window_size}")


class AcousticMetrics:
    """
    Worker class dedicated to the exhaustive extraction of acoustic metrics.

    Computes 91 distinct acoustic features including alpha indices, temporal moments, 
    spectral properties, Mel-frequency cepstral coefficients (MFCCs), and wavelet transforms.

    Parameters
    ----------
    config : AcousticMetricsConfig, optional
        The immutable configuration object. If None, default parameters are used.
    **kwargs : dict
        Additional arguments to pass to the configuration factory if `config` is None.

    Attributes
    ----------
    config : AcousticMetricsConfig
        Read-only property that returns the active configuration.
    """
    def __init__(self, config: Optional[AcousticMetricsConfig] = None, **kwargs):
        if config is not None:
            if not isinstance(config, AcousticMetricsConfig):
                raise TypeError(f"Config must be AcousticMetricsConfig. Got: {type(config)}")
            else:
                self._metrics_config = config
        else:
            self._metrics_config = AcousticMetricsConfig.from_kwargs(**kwargs)

    @property
    def config(self) -> AcousticMetricsConfig:
        """
        Retrieves the metrics configuration.

        Returns
        -------
        AcousticMetricsConfig
            The immutable configuration object in use.
        """
        return self._metrics_config

    def calculate(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extracts all acoustic metrics from a given audio signal.

        Parameters
        ----------
        y : numpy.ndarray
            The 1D audio time-series array to be analyzed.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping the name of each extracted metric to its computed scalar value.
            Missing or failed computations default to 0.0.
        """
        res = {}
        sr = self.config.sample_rate
        n_fft = self.config.fft_window_size
        hop = n_fft // 2
        
        # 1. Transformations (Pre-calculus)
        # Using epsilon to prevent math domain errors (log of 0)
        Sxx_power, tn, fn, _ = maddsd.spectrogram(y, sr, nperseg=n_fft, noverlap=hop, mode='psd')
        Sxx_power = np.maximum(Sxx_power, 1e-12)
        Sxx_mag = np.sqrt(Sxx_power)
        psd_avg = np.mean(Sxx_power, axis=1)
        
        
        # --- Group: Alpha Acoustic Indices ---
        try:
            res['temp_entropy'] = self._get_scalar(maadfeat.temporal_entropy(y))
            
            res['ACI'] = self._get_scalar(maadfeat.acoustic_complexity_index(Sxx_mag)[2])
            
            dt = tn[1] - tn[0]
            res['AGI'] = self._get_scalar(maadfeat.acoustic_gradient_index(Sxx_power, dt)[3])

            spectral_entropy = maadfeat.spectral_entropy(Sxx_power, fn)
            res['EAS'] = self._get_scalar(spectral_entropy[0])
            res['ECU'] = self._get_scalar(spectral_entropy[1])
            res['ECV'] = self._get_scalar(spectral_entropy[2])

            res['num_peaks'] = self._get_scalar(maadfeat.number_of_peaks(psd_avg, fn, mbins=None, threshold=None))

            ndsi_res = maadfeat.soundscape_index(Sxx_power, fn)
            res['NDSI'] = self._get_scalar(ndsi_res[0])
            res['ratioBA'] = self._get_scalar(ndsi_res[1])
            res['anthro_energy'] = self._get_scalar(ndsi_res[2])
            res['bio_energy'] = self._get_scalar(ndsi_res[3])

            res['BI'] = self._get_scalar(maadfeat.bioacoustics_index(Sxx_power, fn))

            res['ADI'] = self._get_scalar(maadfeat.acoustic_diversity_index(Sxx_power, fn))
            res['AEI'] = self._get_scalar(maadfeat.acoustic_eveness_index(Sxx_power, fn))
            
            res['roughness'] = self._get_scalar(maadfeat.surface_roughness(Sxx_power, fn))

            res['tfsd'] = self._get_scalar(maadfeat.tfsd(Sxx_power, fn, tn))

        except Exception: 
            pass
        
        # --- Group: Temporal ---
        t_moments = maadfeat.temporal_moments(y)
        res['temp_skew'] = self._get_scalar(t_moments[2])
        res['temp_kurtosis'] = self._get_scalar(t_moments[3])

        rms = libfeat.rms(y=y, frame_length=n_fft, hop_length=hop)
        res['rms_mean'] = self._get_scalar(np.mean(rms))
        res['rms_std'] = self._get_scalar(np.std(rms))
        res['rms_var'] = self._get_scalar(np.square(np.std(rms)))


        zcr = libfeat.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
        res['zcr_mean'] = self._get_scalar(np.mean(zcr))
        res['zcr_std'] = self._get_scalar(np.std(zcr))
        res['zcr_var'] = self._get_scalar(np.square(np.std(zcr)))

        res['mae'] = self._get_scalar(np.mean(np.abs(y)))

        # --- Group: Spectral ---
        res['psd_mean'] = self._get_scalar(np.mean(psd_avg))
        res['psd_std'] = self._get_scalar(np.std(psd_avg))
        res['psd_var'] = self._get_scalar(np.square(np.std(psd_avg)))

        res['spec_slope'] = self._get_scalar(stats.linregress(fn, psd_avg)[0])

        s_moments = maadfeat.spectral_moments(psd_avg)
        res['spec_mean'] = self._get_scalar(s_moments[0])
        res['spec_std'] = self._get_scalar(s_moments[1])
        res['spec_var'] = self._get_scalar(np.square(s_moments[1]))
        res['spec_skew'] = self._get_scalar(s_moments[2])
        res['spec_kurtosis'] = self._get_scalar(s_moments[3])

        res['spec_centroid'] = self._get_scalar(np.mean(libfeat.spectral_centroid(S=Sxx_mag, sr=sr)))
        
        res['spec_flatness'] = self._get_scalar(np.mean(libfeat.spectral_flatness(S=Sxx_mag)))
        res['spec_rolloff'] = self._get_scalar(np.mean(libfeat.spectral_rolloff(S=Sxx_mag, sr=sr)))
        
        try:
            bw_50, bw_90 = maadfeat.spectral_bandwidth(y, sr, nperseg=n_fft)
            res['spec_bandwidth_50%'] = self._get_scalar(bw_50)
            res['spec_bandwidth_90%'] = self._get_scalar(bw_90)
        except Exception:
            pass

        try:
            spec_bdw = libfeat.spectral_bandwidth(S=Sxx_mag, sr=sr)
            res['spec_bandwidth_mean'] = self._get_scalar(np.mean(spec_bdw))
        except Exception:
            pass


        mel = libfeat.melspectrogram(S=Sxx_power, sr=sr)
        mfccs = libfeat.mfcc(S=power_to_db(mel), n_mfcc=20)
        mfccs_delta = libfeat.delta(mfccs)
        
        mfcc_m = np.mean(mfccs, axis=1)
        mfcc_d_m = np.mean(mfccs_delta, axis=1)
        for i in range(20): 
            res[f'mfcc_{i+1}'] = self._get_scalar(mfcc_m[i])
            res[f'mfcc_delta_{i+1}'] = self._get_scalar(mfcc_d_m[i])

        res['mfcc_mean'] = self._get_scalar(np.mean(mfcc_m))
        res['mfcc_std'] = self._get_scalar(np.std(mfcc_m))
        res['mfcc_var'] = self._get_scalar(np.square(np.std(mfcc_m)))

        # --- Group: Multi-scale ---
        try: 
            coeffs = pywtwavedec(y, 'db4', level=5)
            
            ca5 = coeffs[0]
            ca5_energy = np.sum(ca5**2) / len(ca5)
            res['wav_approx_energy'] = self._get_scalar(ca5_energy)
            res['wav_approx_std'] = self._get_scalar(np.std(ca5))
            res['wav_approx_var'] = self._get_scalar(np.square(np.std(ca5)))

            details = coeffs[1:]
            
            wav_energys = [ca5_energy]

            for i, detail in enumerate(details):
                level = 5 - i
            
                detail_energy = np.sum(detail**2) / len(detail)
                wav_energys.append(detail_energy)

                res[f'wav_detail_lvl{level}_energy'] = self._get_scalar(detail_energy)
                res[f'wav_detail_lvl{level}_std'] = self._get_scalar(np.std(detail))

            res['wav_energy_mean'] = self._get_scalar(np.mean(wav_energys))
        except:
            pass
        
        expected_metrics = [
            'temp_entropy', 'ACI', 'AGI', 'EAS', 'ECU', 'ECV', 'num_peaks', 
            'NDSI', 'ratioBA', 'anthro_energy', 'bio_energy', 'BI', 'ADI', 'AEI', 
            'roughness', 'tfsd', 'temp_skew', 'temp_kurtosis', 
            'rms_mean', 'rms_std', 'rms_var', 'zcr_mean', 'zcr_std', 'zcr_var', 'mae',
            'psd_mean', 'psd_std', 'psd_var', 'spec_slope', 'spec_mean', 'spec_std', 'spec_var', 'spec_skew', 'spec_kurtosis',
            'spec_centroid', 'spec_flatness', 'spec_rolloff', 'spec_bandwidth_mean', 'spec_bandwidth_50%', 'spec_bandwidth_90%',
            'mfcc_mean', 'mfcc_std', 'mfcc_var', 'wav_energy_mean', 'wav_approx_energy', 'wav_approx_std', 'wav_approx_var'
        ]
        for i in range(1, 21):
            expected_metrics.append(f'mfcc_{i}')
            expected_metrics.append(f'mfcc_delta_{i}')
        for lvl in range(1, 6):
            expected_metrics.append(f'wav_detail_lvl{lvl}_energy')
            expected_metrics.append(f'wav_detail_lvl{lvl}_std')
            expected_metrics.append(f'wav_detail_lvl{lvl}_var')

        for metric in expected_metrics:
            if metric not in res:
                res[metric] = np.nan

        return res
    
    def _get_scalar(self, val: Any) -> float:
        """
        THE TRUTH EXTRACTOR: Guarantees a float return without triggering 
        any 'ambiguous array' boolean checks.

        Converts to a NumPy representation first, then uses non-boolean 
        statistical reductions to return a safe scalar.

        Parameters
        ----------
        val : Any
            The raw value returned by a metric extraction function, which could be
            a scalar, a boolean array, a NaN, or a multidimensional array.

        Returns
        -------
        """
        try:
            if isinstance(val, np.ndarray):
                if val.size == 0: 
                    return np.nan
            
                if val.dtype == bool:
                    if np.mean(val) > 5.0:
                        return 1.0
                    else:
                        return 0.0
            
                return float(np.nanmean(val))
            else:
                number = float(val)
            
                if np.isinf(number) or np.isnan(number):
                    return np.nan
                
                return number
        except:
            return np.nan

if __name__ == '__main__':
    pass