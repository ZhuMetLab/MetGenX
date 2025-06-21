import numpy as np

class SpecEntityData(object):
    def __init__(self, spec, precursormz):
        """
        :param spec: a list of (m/z,intensity), representing a MS2 spectrum.
        :param precursormz: m/z of precursor ion for MS2 spectrum
        """
        super(SpecEntityData, self).__init__()
        self.spec = spec
        self.precursormz = precursormz
        self.NL = None

    def CleanSpec(self, TopN=20, min_int=0.01):
        """
        Removes low intensity peaks from the MS2 spectrum and keeps only the top N peaks.
        :param TopN: Number of remained peaks default 20.
        :param min_int: minimum intensity cutoff, default 0.01.
        """
        spec_cleaned = self.spec
        spec_cleaned.sort(key=lambda x: x[1], reverse=True)
        spec_cleaned = [i for i in spec_cleaned if i[1] >= min_int]
        if len(spec_cleaned) > TopN:
            spec_cleaned = spec_cleaned[0:TopN]
        spec_cleaned = list(np.around(spec_cleaned, 2))
        spec_cleaned = [tuple(i) for i in spec_cleaned]
        spec_cleaned.sort(key=lambda x: x[0], reverse=False)
        self.spec = spec_cleaned

    def CalNLSpec(self, NL_range=[5,200]):
        """
        Calculates the neutral loss (NL) spectrum by subtracting each peak's m/z value from the precursor m/z value
        and rounding to 2 decimal places. Peaks with m/z equal to 0 are assigned an intensity of 0 in the NL spectrum.
        """
        spec = self.spec
        NL = []
        for peak in spec:
            if peak[0] != 0:
                NL_mz = round(self.precursormz - peak[0], 2)
                if NL_mz >= NL_range[0] and NL_mz<=NL_range[1]:
                    NL.extend([tuple([NL_mz, peak[1]])])
            else:
                NL.extend([tuple([0, 0])])
        self.NL = NL

    def Generate(self):
        """
        Returns a dictionary with the cleaned MS2 spectrum (Spec), precursor m/z value (precursor_mz),
        and NL spectrum (SpecNL).
        """
        return dict({"Spec": self.spec, "precursor_mz": self.precursormz, "SpecNL": self.NL})

    def Normalization(self):
        max_int = max([peak[1] for peak in self.spec])
        normalized_peak = self.spec
        normalized_peak = [(peak[0], peak[1]/max_int) for peak in normalized_peak]
        self.spec = normalized_peak

    def denoise(self):
        if self.precursormz is not None:
            spec_denoised = [peak for peak in self.spec if peak[0] <= self.precursormz+0.1]
            self.spec = spec_denoised
    def tokenize(self):
        spec_token = [("<bos>",1)]
        if self.spec is not None:
            spec_token.append(("<spe>",1))
            spec_token = spec_token + self.spec
        if self.NL is not None:
            spec_token.append(("<nl>",1))
            spec_token = spec_token+self.NL
        if self.precursormz is not None:
            spec_token.append(("<pre>",1))
            spec_token.append((self.precursormz, 2))
        spec_token.append(("<eos>", 1))
        return spec_token

    def tokenize_spec2vec(self):
        spec_token, spec_intensity = [], []
        if self.spec is not None:
            spec_token = spec_token + ["peak@" + str(peak[0]) for peak in self.spec]
            spec_intensity=spec_intensity+[peak[1] for peak in self.spec]
        # if self.precursormz is not None:
        #     spec_token.append(("peak@"+str(self.precursormz)))
        #     spec_intensity.append((2))
        if self.NL is not None:
            spec_token = spec_token+["nl@" + str(peak[0]) for peak in self.NL]
            spec_intensity = spec_intensity+[peak[1] for peak in self.NL]
        return spec_token, spec_intensity
    def cal_segment(self):
        segment_vec = [1]*(len(self.spec)+2)+[2]*(len(self.spec)+1)+[3]*3
        return segment_vec

import math

def GenerateSpec2vec(mz, intensity, precursor_mz, TopN=100, TopN_mz=0.5, min_int=0.01,
                              min_frag=10, NL_range=[0.5, 200]):
    spec = list(zip(mz, intensity))
    if len(spec) >= min_frag:
        SpecData = SpecEntityData(spec=spec, precursormz=round(precursor_mz, 2))
        SpecData.Normalization()
        SpecData.denoise()
        if TopN_mz is not None:
            if TopN is not None:
                TopN = min(math.ceil(TopN_mz * precursor_mz), TopN)
            else:
                TopN = math.ceil(TopN_mz * precursor_mz)
        SpecData.CleanSpec(TopN=TopN, min_int=min_int)
        SpecData.CalNLSpec(NL_range=NL_range)
        spec_token, spec_intensity = SpecData.tokenize_spec2vec()
    else:
        spec_token, spec_intensity = None, None
    return spec_token, spec_intensity


