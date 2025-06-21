# -*- coding: utf-8 -*-
from collections import Counter
from os import path
import re
import numpy as np
from MS2Tools.structure import UnitFloat
def abs_mz_tol(mz, tolerance, res_define_at = 0):
    """
    Obtain absolute m/z tolerance.

    Parameters
    ----------
    mz:
        target m/z
    tolerance:
        m/z tolerance for finding mz rois
        it will be considered as in unit "ppm" if tolerance is float > 1
    res_define_at:
        m/z values to define the ppm resolution, using 0 if not set

    Returns
    -------
        absolute m/z tolerance
    """
    if not isinstance(tolerance, UnitFloat):
        tolerance = UnitFloat(tolerance, "ppm" if tolerance > 1 else "Dalton")
    if tolerance.unit_info.lower() == "dalton":
        return np.full(mz.shape, tolerance)

    if isinstance(mz, np.ndarray):
        mz = mz.copy()
        mz[mz < res_define_at] = res_define_at
    elif isinstance(mz, pd.Series):
        mz = mz.copy()
        mz.loc[mz < res_define_at] = res_define_at
    else:
        mz = max(mz, res_define_at)
    return tolerance * mz * 1e-6

class SpectraData(object):
    def __init__(self, metaData, mz, intensity, annotation: np.ndarray = None, not_noise: np.ndarray = None,
                 normalized: bool = False):
        self.metaData = metaData
        if len(mz) != len(intensity):
            raise ValueError("mz and intensity arrays must have the same length")
        if annotation is not None and len(mz) != len(annotation):
            raise ValueError("mz, intensity and annotation arrays must have the same length")
        self._mz = mz
        self._intensity = intensity
        self._annotation = annotation
        self._not_noise = np.ones_like(mz, dtype=bool) if not_noise is None else not_noise
        self._normalized = normalized


    def __len__(self):
        return sum(self._not_noise)

    def __getitem__(self, item):
        return np.asarray([self._mz[item], self._intensity[item]])

    def __copy__(self):
        return self.__class__(self._mz.copy(), self._intensity.copy(), self._not_noise.copy())

    @property
    def normalized(self):
        return self._normalized

    @property
    def not_noise(self):
        return self._not_noise

    @property
    def sorted(self):
        return np.all(self._mz[:-1] <= self._mz[1:])

    @property
    def mz(self):
        return self._mz[self._not_noise]

    @property
    def intensity(self):
        return self._intensity[self._not_noise]

    @property
    def annotation(self):
        return self._annotation[self._not_noise] if self._annotation is not None else None

    def denoise(
        self,
        precursor_mz: float = None,
        threshold_rel: float = 0.0,
        threshold_abs: float = None,
        mz_tol = 20,
        include_precursor: bool = True,
        consider_precursor: bool = True,
        mz_range=None,
        auto_threshold_counts: int = None,
        ms2_noise: float = 3.0,
        sn_threshold: float = 3.0,
        clear_noise: bool = False,
        res_define_at: float = 0.0,
    ):
        """
        Purifying MSMS spectrum by removing noisy fragment and satellites

        Args:
            precursor_mz: precursor m/z
            threshold_rel: relative threshold for MSMS intensity filtering
            threshold_abs: absolute threshold for MSMS intensity filtering
            mz_tol: m/z tolerance for detecting precursor fragment
            include_precursor: indicating if the precursor fragment should be included
            consider_precursor: indicating if the precursor fragment should be considered
            mz_range: m/z range of fragment ions
            auto_threshold_counts: minimal number of fragments with the same low intensity
            used for calculating threshold (will be ignored when `threshold_abs` is set)
            ms2_noise: noise level for fragment ion intensities (will be ignored when `threshold_abs` is set)
            sn_threshold: signal-to-noise ratio for ms2 fragments (for calculating `threshold_abs`,
            will be ignored when `threshold_abs` is set)
            clear_noise: if the noise fragments should be cleared
            res_define_at: m/z value (Da) for defining resolution

        Returns:
            purified Spectrum object
        """
        # precursor filtering
        mz_bound_min = 0
        if precursor_mz is not None:
            mz_abs = abs_mz_tol(precursor_mz, mz_tol, res_define_at)
            mz_bound_max = precursor_mz + mz_abs if include_precursor else precursor_mz - mz_abs
        else:
            mz_bound_max = np.inf
        if mz_range is not None:
            mz_bound_max = min(mz_bound_max, mz_range[1])
            mz_bound_min = mz_range[0]

        self._not_noise = (self._mz >= mz_bound_min) & (self._mz <= mz_bound_max)

        # intensity filtering
        if np.any(self._not_noise):
            int_max = (
                max(self._intensity[self._not_noise & (self._mz <= precursor_mz - mz_tol)])
                if not consider_precursor and include_precursor
                else max(self._intensity[self._not_noise])
            )
            if not self._normalized and auto_threshold_counts is not None:
                threshold_abs = max(
                    threshold_abs,
                    self._auto_threshold(auto_threshold_counts) if auto_threshold_counts else ms2_noise * sn_threshold,
                )
            threshold_abs = int_max * threshold_rel if self._normalized else max(int_max * threshold_rel, threshold_abs)
            self._not_noise = self._not_noise & (self._intensity >= threshold_abs)
            self._remove_satellites()
        if clear_noise:
            self._mz = self._mz[self._not_noise]
            self._intensity = self._intensity[self._not_noise]
            if self._annotation is not None:
                self._annotation = self._annotation[self._not_noise]
            self._not_noise = self._not_noise[self._not_noise]

    def normalize(
        self,
        normalize_to: float = 1,
        consider_precursor: bool = True,
        precursor_mz: float = None,
        mz_tol = 20,
        res_define_at: float = 0.0,
    ):
        if not consider_precursor:
            precursor_mz = precursor_mz or self._mz[self._not_noise].max()
            mz_abs = abs_mz_tol(precursor_mz, mz_tol, res_define_at)
            max_intensity = max(self._intensity[self._not_noise & (self._mz <= precursor_mz - mz_abs)])
        else:
            max_intensity = max(self._intensity[self._not_noise])
        self._intensity = self._intensity / max_intensity * normalize_to
        self._normalized = True

    def _remove_satellites(self, mz_tolerance: float = 0.3, intensity_tolerance: float = 0.2):
        mz = self.mz
        intensity = self.intensity
        if len(mz) == 0:
            return
        # satellite fragment removing
        tmp = np.nonzero(np.diff(mz) <= mz_tolerance)[0]
        mz_group = []
        tmp_group = []
        for index in tmp:
            if index + 1 in tmp:
                tmp_group.append(index)
            else:
                if tmp_group:
                    tmp_group.append(index)
                    mz_group.append(tmp_group)
                else:
                    mz_group.append([index])
                tmp_group = []

        satellites_index = []
        for index in mz_group:
            index.append(index[-1] + 1)
            intensity_cad = intensity[index]
            mz_cad = mz[index]
            if len(cad_index := self._find_satellite_cads(mz_cad, intensity_cad, mz_tolerance, intensity_tolerance)):
                satellites_index = satellites_index + [index[x] for x in cad_index]
        if satellites_index:
            self._not_noise[np.nonzero(self.not_noise)[0][satellites_index]] = False

    @staticmethod
    def _find_satellite_cads(
        mz_cad: np.ndarray,
        intensity_cad: np.ndarray,
        mz_tolerance: float = 0.3,
        intensity_tolerance: float = 0.2,
    ):
        cad_index = []
        while len(mz_cad) > 1:
            max_index = intensity_cad.argmax()
            intensity_thr = intensity_cad[max_index] * intensity_tolerance
            if all(intensity_cad > intensity_thr):
                break
            if len(
                cad_index_ := np.nonzero(
                    (intensity_cad <= intensity_thr) & (abs(mz_cad - mz_cad[max_index]) <= mz_tolerance)
                )[0]
            ):
                mz_cad = np.delete(mz_cad, cad_index_)
                intensity_cad = np.delete(intensity_cad, cad_index_)
                cad_index += cad_index_.tolist()
            else:
                mz_cad = np.delete(mz_cad, max_index)
                intensity_cad = np.delete(intensity_cad, max_index)
        return cad_index

    def _auto_threshold(self, auto_threshold_counts):
        counts = pd.Series(Counter(self._intensity)).sort_index()
        res = 0.0
        if any(is_above := (counts >= auto_threshold_counts)):
            res = counts[is_above].index[-1] + 1
        if (len_int := len(self._intensity)) < auto_threshold_counts:
            if len_int > 1 and len(np.unique(self._intensity)) == 1:
                res = np.unique(self._intensity)[0] + 1
        return res

class SpectrumFileReader(object):
    def __init__(self, file, file_type=None):
        self._file = file
        self._file_type = file_type or path.splitext(file)[-1][1:].lower()
        self._parser = self._get_parser()

    def read_file(self):
        spectra = []
        for spectrum in self._parser:
            spectra.append(spectrum)
        return spectra

    def _get_parser(self):
        if self._file_type in ['mgf', 'msp', 'cef']:
            return eval("self._parse_" + self._file_type.lower())()
        else:
            raise RuntimeError(f'Not supported file types: {self._file_type}')

    def _parse_mgf(self):
        metadata = {}
        mz = []
        intensity = []
        started = False


        with open(self._file, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                if '=' in line:
                    split_line = line.split("=", 1)
                    if split_line[0].lower() == 'name':
                        # Obtaining the parameters inside the title index
                        name = re.search(r'(?<=NAME=)[^.]+', line)
                        if name:
                            metadata['name'] = self._convert_value(name.group(0))
                            continue

                if line == 'BEGIN IONS':
                    started = True
                    continue

                if started:
                    if '=' in line:
                        # Obtaining the params
                        split_line = line.split("=", 1)
                        value = split_line[1].strip()
                        if split_line[0].lower() == 'pepmass':
                            # New for mgf with precursor intensity
                            if " " in value:
                                value = value.split(" ", 1)[0]
                            else:
                                value = re.compile(r'\s|\t').sub('', split_line[1])
                        metadata[split_line[0].lower()] = self._convert_value(value)
                    elif line != 'END IONS':
                        # Obtaining the mz and intensity
                        split_line = re.split(r'\s|\t', line)
                        mz.append(float(split_line[0]))
                        intensity.append(float(split_line[1]))
                    else:
                        metadata['mz'] = metadata.pop('pepmass')
                        if 'rtinseconds' in metadata:
                            metadata['rt'] = metadata.pop('rtinseconds')
                        elif 'rtinminutes' in metadata:
                            metadata['rt'] = metadata.pop('rtinminutes')
                        yield SpectraData(metadata, np.array(mz), np.array(intensity))
                        started = False
                        metadata = {}
                        mz = []
                        intensity = []

    def _parse_msp(self):
        metadata = {}
        mz = []
        intensity = []

        # Peaks counter. Used to track and count the number of peaks
        peak_count = 0

        with open(self._file, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()

                if len(line) == 0:
                    continue

                if ':' in line:
                    # Obtaining the params
                    split_line = line.split(":", 1)
                    split_line[0] = split_line[0].lower()
                    split_line[1] = split_line[1].strip()
                    if split_line[0] == 'comments':
                        if ' ' in split_line[1]:
                            # Obtaining the parameters inside the comments index
                            for s in split_line[1].split('" "'):
                                split_line = s.split("=", 1)
                                if split_line[0] in metadata.keys() and split_line[0] == 'smiles':
                                    metadata[split_line[0] + "_2"] = self._convert_value(split_line[1])
                                else:
                                    metadata[split_line[0]] = self._convert_value(split_line[1])
                        else:
                            metadata[split_line[0]] = split_line[1]
                    else:
                        metadata[split_line[0]] = self._convert_value(split_line[1])
                else:
                    # Obtaining the mz and intensity
                    peak_count += 1

                    split_line = re.split(r'[\s\t]', line, maxsplit=2)

                    mz.append(float(split_line[0]))
                    intensity.append(float(split_line[1]))

                    # Obtaining the mz and intensity
                    if int(metadata['num peaks']) == peak_count:
                        metadata['mz'] = metadata.pop('precursormz')
                        if 'retentiontime' in metadata:
                            metadata['rt'] = metadata.pop('retentiontime')
                        yield SpectraData(metadata, np.array(mz), np.array(intensity))
                        peak_count = 0
                        metadata = {}
                        mz = []
                        intensity = []


    @staticmethod
    def _convert_value(value: str):
        res = value
        try:
            res = int(value)
        except ValueError:
            res = float(value)
        finally:
            return res

    @staticmethod
    def _rename_key(dict_data: dict, old_keys: list, new_keys: list, convert_to: type = None):
        for old_key, new_key in zip(old_keys, new_keys):
            if old_key in dict_data:
                dict_data.update(
                    {new_key: convert_to(dict_data.pop(old_key)) if convert_to else dict_data.pop(old_key)})
        return dict_data


import pandas as pd
def export_mgf(SpecData, key_map: dict = None, file: str = None, append: bool = False):
    msp_text = []
    key_map = key_map or {'mz': 'PEPMASS', 'rt': 'RTINSECONDS', 'charge': 'CHARGE'}
    spec_text = ['BEGIN IONS']
    spec_text += [f'{key_map.get(k, k)}= {SpecData.metaData.get(k)}' for k in SpecData.metaData.keys() if
                  k.lower() != 'num peaks']
    # spec_text.append(f'Num Peaks: {len(SpecData.mz)}')
    spec = pd.DataFrame({"mz": SpecData.mz, "intensity": SpecData.intensity})
    spec_text += list(spec.apply(lambda x: ' '.join(x.astype(str)), axis=1))
    spec_text.append('END IONS')
    msp_text.append('\n'.join(spec_text))
    msp_text.append('\n')
    res = '\n'.join(msp_text)
    if file:
        with open(file, 'a' if append else 'w') as f:
            f.write(res)
        return True
    else:
        return res

def export_combine_mgf(SpecData, key_map: dict = None, file: str = None, append: bool = False):
    msp_text = []
    key_map = key_map or {'mz': 'PEPMASS', 'rt': 'RTINSECONDS', 'charge': 'CHARGE'}
    for tgt_spec in SpecData:
        spec_text = ['BEGIN IONS']
        spec_text += [f'{key_map.get(k, k)}={tgt_spec.metaData.get(k)}' for k in tgt_spec.metaData.keys() if
                      k.lower() != 'num peaks']
        # spec_text.append(f'Num Peaks: {len(SpecData.mz)}')
        spec = pd.DataFrame({"mz": tgt_spec.mz, "intensity": tgt_spec.intensity})
        spec_text += list(spec.apply(lambda x: ' '.join(x.astype(str)), axis=1))
        spec_text.append('END IONS')
        msp_text.append('\n'.join(spec_text))
        msp_text.append('\n')
    res = '\n'.join(msp_text)
    if file:
        with open(file, 'a' if append else 'w') as f:
            f.write(res)
        return True
    else:
        return res
