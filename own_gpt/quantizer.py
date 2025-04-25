import numpy as np

class Quantizer:
    def __init__(self, num_bins=64, clip=0.05):
        self.num_bins = num_bins
        self.clip = clip
        edges = np.linspace(-clip, clip, num_bins + 1)
        self.centers = (edges[:-1] + edges[1:]) / 2
        self.edges   = edges

    def encode(self, rets):
        # Want stuff within a specific range, so clipping
        r = np.clip(rets, -self.clip, self.clip)
        ids = np.digitize(r, self.edges) - 1
        ids = np.clip(ids, 0, self.num_bins - 1)
        return ids

    def decode(self, ids):
        return self.centers[ids]