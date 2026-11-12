"""Stream complete float64 radius edges with conservative matrix screening."""

import numpy as np
import torch


class GpuNeighbors:
    """Matrix lower bounds screen; NumPy directly verifies returned edges.

    The bound subtracts 64 * eps * (dimension + 4) times the sum of squared
    norms, dominating float64 dot-product/norm error for these bounded rows.
    No nearest-neighbour truncation or approximate radius query is used.
    """

    def __init__(self, points, device, signature_norms):
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.points = torch.tensor(points, device=self.device, dtype=torch.float64)
        self.norms = self.points.square().sum(1)
        self.signature_norms = torch.tensor(signature_norms, device=self.device)
        self.count = len(points)
        self.threshold = 0.0
        self.error_factor = 64 * np.finfo(np.float64).eps * (points.shape[1] + 4)

    def iter_radius_edges(self, points, query_ids, radius):
        queries = torch.tensor(points, device=self.device, dtype=torch.float64)
        identity = torch.tensor(query_ids, device=self.device)
        query_norms = queries.square().sum(1)[:, None]
        for start in range(0, self.count, 131072):
            stop = min(start + 131072, self.count)
            norm_sum = query_norms + self.norms[start:stop][None]
            lower = (
                (1 - self.error_factor) * norm_sum
                - 2 * (queries @ self.points[start:stop].T)
                - 1e-300
            )
            inside = (identity >= start) & (identity < stop)
            local_rows = torch.nonzero(inside).flatten()
            lower[local_rows, identity[inside] - start] = torch.inf
            possible = lower <= radius**2
            possible_other = possible.any(1).cpu().numpy()
            possible &= self.signature_norms[start:stop][None] > self.threshold
            # Keep materialization bounded even at a large radius. The caller
            # may reject from any partial edge set and stop this generator.
            pending = [(0, len(queries))]
            while pending:
                begin, end = pending.pop()
                mask = possible[begin:end]
                if end - begin > 1 and int(mask.sum()) > 1000000:
                    middle = (begin + end) // 2
                    pending.extend(((middle, end), (begin, middle)))
                    continue
                rows, columns = torch.nonzero(mask, as_tuple=True)
                edges = torch.stack((rows + begin, columns + start), 1).cpu().numpy()
                yield edges[:, 0], edges[:, 1], possible_other
