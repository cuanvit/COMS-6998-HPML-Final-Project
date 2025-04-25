import numpy as np

class Quantizer:
    def __init__(self, num_bins=256, clip=0.05):
        self.num_bins = num_bins
        self.clip = clip
        edges = np.linspace(-clip, clip, num_bins + 1)
        self.centers = (edges[:-1] + edges[1:]) / 2
        self.edges   = edges

    def encode(self, rets):
        r = np.clip(rets, -self.clip, self.clip)
        return np.digitize(r, self.edges) - 1

    def decode(self, ids):
        return self.centers[ids]