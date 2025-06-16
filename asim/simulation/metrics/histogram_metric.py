from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class HistogramIntersectionMetric:
    def __init__(self, min_val: float, max_val: float, n_bins: int, weight: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.n_bins = n_bins
        self.bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        self.weight = weight

        self.aggregate_objects: bool = False
        self.independent_timesteps: bool = True

    def _create_histogram(self, data: List[float], normalize: bool = True) -> np.ndarray:
        hist, _ = np.histogram(data, bins=self.bin_edges)

        if normalize:
            # Normalize to create probability distribution
            hist = hist.astype(float)
            if hist.sum() > 0:
                hist = hist / hist.sum()

        return hist

    def _calculate_intersection(self, dist1: npt.NDArray[np.float64], dist2: npt.NDArray[np.float64]) -> float:
        hist1 = self._create_histogram(dist1, normalize=True)
        hist2 = self._create_histogram(dist2, normalize=True)
        intersection = np.sum(np.minimum(hist1, hist2))
        return intersection

    def _calculate_bhattacharyya(self, dist1: npt.NDArray[np.int_], dist2: npt.NDArray[np.int_]) -> float:
        hist1 = self._create_histogram(dist1, normalize=True)
        hist2 = self._create_histogram(dist2, normalize=True)
        bhattacharyya_coeff = np.sum(np.sqrt(hist1 * hist2))
        return bhattacharyya_coeff

    def _calculate_wasserstein(self, dist1: npt.NDArray[np.int_], dist2: npt.NDArray[np.int_]) -> float:
        hist1 = self._create_histogram(dist1, normalize=True)
        hist2 = self._create_histogram(dist2, normalize=True)
        # Calculate the Wasserstein distance (Earth Mover's Distance)
        # This is a simple implementation using the cumulative distribution function (CDF)
        cdf1 = np.cumsum(hist1)
        cdf2 = np.cumsum(hist2)
        # Calculate the Wasserstein distance
        wasserstein_distance = np.sum(np.abs(cdf1 - cdf2))
        return wasserstein_distance

    def calculate_intersection(
        self, dist1: npt.NDArray[np.float64], dist2: npt.NDArray[np.float64], log_mask: npt.NDArray[np.bool_]
    ) -> Dict[str, float]:
        assert dist1.shape[0] == dist2.shape[0], "Distributions must have the same number of objects"
        assert dist1.ndim == 2
        assert dist2.ndim == 2
        assert log_mask.ndim == 2

        intersection = 0.0
        bhattacharyya = 0.0
        wasserstein = 0.0

        if self.independent_timesteps:
            # (n_objects, n_rollouts * n_steps)
            for obj_dist1, obj_dist2, obj_mask in zip(dist1, dist2, log_mask):
                intersection += self._calculate_intersection(obj_dist1[obj_mask], obj_dist2[obj_mask])
                bhattacharyya += self._calculate_bhattacharyya(obj_dist1[obj_mask], obj_dist2[obj_mask])
                wasserstein += self._calculate_wasserstein(obj_dist1[obj_mask], obj_dist2[obj_mask])
            intersection /= dist1.shape[0]  # Average intersection over all objects
            bhattacharyya /= dist1.shape[0]  # Average Bhattacharyya coefficient over all objects
            wasserstein /= dist1.shape[0]  # Average Wasserstein distance over all objects

        else:
            raise NotImplementedError

        return {
            "intersection": float(intersection),
            "bhattacharyya": float(bhattacharyya),
            "wasserstein": float(wasserstein),
        }

    def plot_histograms(
        self,
        dist1: npt.NDArray[np.float64],
        dist2: npt.NDArray[np.float64],
        mask: Optional[npt.NDArray[np.bool_]] = None,
        labels: Optional[Tuple[str, str]] = None,
        title: str = "Histogram Comparison",
    ) -> None:
        def _apply_mask(
            data: npt.NDArray[np.float64], mask: Optional[npt.NDArray[np.bool_]]
        ) -> npt.NDArray[np.float64]:
            flat_data = []
            for obj_data, obj_mask in zip(data, mask):
                if mask is not None:
                    flat_data.extend(obj_data[obj_mask].tolist())
                else:
                    flat_data.extend(obj_data.tolist())
            return np.array(flat_data)

        hist1 = self._create_histogram(_apply_mask(dist1, mask), normalize=True)
        hist2 = self._create_histogram(_apply_mask(dist2, mask), normalize=True)

        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        width = (self.max_val - self.min_val) / self.n_bins

        plt.figure(figsize=(10, 6))

        if labels is None:
            labels = ("Distribution 1", "Distribution 2")

        plt.bar(bin_centers, hist1, width, alpha=0.5, label=labels[0], color="blue")
        plt.bar(bin_centers, hist2, width, alpha=0.5, label=labels[1], color="red")

        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def detailed_analysis(self, dist1: List[float], dist2: List[float]) -> dict:
        hist1 = self._create_histogram(dist1, normalize=True)
        hist2 = self._create_histogram(dist2, normalize=True)

        intersection = self.calculate_intersection(dist1, dist2)

        # Calculate additional metrics
        kl_div_1_2 = np.sum(hist1 * np.log(hist1 / (hist2 + 1e-10) + 1e-10))
        kl_div_2_1 = np.sum(hist2 * np.log(hist2 / (hist1 + 1e-10) + 1e-10))

        # Bhattacharyya distance
        bhattacharyya = -np.log(np.sum(np.sqrt(hist1 * hist2)))
        bhattacharyya_coeff = np.sum(np.sqrt(hist1 * hist2))
        return {
            "histogram_1": hist1,
            "histogram_2": hist2,
            "intersection": intersection,
            "kl_divergence_1_to_2": kl_div_1_2,
            "kl_divergence_2_to_1": kl_div_2_1,
            "bhattacharyya_distance": bhattacharyya,
            "bhattacharyya_coeff": bhattacharyya_coeff,
            "bin_edges": self.bin_edges,
        }


class BinaryHistogramIntersectionMetric:
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.aggregate_objects: bool = False
        self.independent_timesteps: bool = True

    def _create_histogram(self, data: List[int], normalize: bool = True) -> np.ndarray:
        # Binary histogram: bins are [0, 1]
        hist = np.zeros(2, dtype=float)
        data = np.asarray(data)
        hist[0] = np.sum(data == 0)
        hist[1] = np.sum(data == 1)
        if normalize and hist.sum() > 0:
            hist = hist / hist.sum()
        return hist

    def _calculate_intersection(self, dist1: npt.NDArray[np.int_], dist2: npt.NDArray[np.int_]) -> float:
        hist1 = self._create_histogram(dist1, normalize=True)
        hist2 = self._create_histogram(dist2, normalize=True)
        intersection = np.sum(np.minimum(hist1, hist2))
        return intersection

    def _calculate_bhattacharyya(self, dist1: npt.NDArray[np.int_], dist2: npt.NDArray[np.int_]) -> float:
        hist1 = self._create_histogram(dist1, normalize=True)
        hist2 = self._create_histogram(dist2, normalize=True)
        bhattacharyya_coeff = np.sum(np.sqrt(hist1 * hist2))
        return bhattacharyya_coeff

    def calculate_intersection(
        self, dist1: npt.NDArray[np.int_], dist2: npt.NDArray[np.int_], log_mask: npt.NDArray[np.bool_]
    ) -> Dict[str, float]:
        assert dist1.shape[0] == dist2.shape[0], "Distributions must have the same number of objects"
        assert dist1.ndim == 2
        assert dist2.ndim == 2
        assert log_mask.ndim == 2

        intersection = 0.0
        bhattacharyya = 0.0

        if self.independent_timesteps:
            for obj_dist1, obj_dist2, obj_mask in zip(dist1, dist2, log_mask):
                intersection += self._calculate_intersection(obj_dist1[obj_mask], obj_dist2[obj_mask])
                bhattacharyya += self._calculate_bhattacharyya(obj_dist1[obj_mask], obj_dist2[obj_mask])
            intersection /= dist1.shape[0]
            bhattacharyya /= dist1.shape[0]
        else:
            raise NotImplementedError

        return {"intersection": intersection, "bhattacharyya": bhattacharyya}

    def plot_histograms(
        self,
        dist1: npt.NDArray[np.int_],
        dist2: npt.NDArray[np.int_],
        mask: Optional[npt.NDArray[np.bool_]] = None,
        labels: Optional[Tuple[str, str]] = None,
        title: str = "Binary Histogram Comparison",
    ) -> None:
        def _apply_mask(data: npt.NDArray[np.int_], mask: Optional[npt.NDArray[np.bool_]]) -> npt.NDArray[np.int_]:
            flat_data = []
            for obj_data, obj_mask in zip(data, mask):
                if mask is not None:
                    flat_data.extend(obj_data[obj_mask].tolist())
                else:
                    flat_data.extend(obj_data.tolist())
            return np.array(flat_data)

        hist1 = self._create_histogram(_apply_mask(dist1, mask), normalize=True)
        hist2 = self._create_histogram(_apply_mask(dist2, mask), normalize=True)

        bin_centers = np.array([0, 1])
        width = 0.4

        plt.figure(figsize=(6, 4))

        if labels is None:
            labels = ("Distribution 1", "Distribution 2")

        plt.bar(bin_centers - width / 2, hist1, width, alpha=0.5, label=labels[0], color="blue")
        plt.bar(bin_centers + width / 2, hist2, width, alpha=0.5, label=labels[1], color="red")

        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.title(title)
        plt.xticks([0, 1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def detailed_analysis(self, dist1: List[int], dist2: List[int]) -> dict:
        hist1 = self._create_histogram(dist1, normalize=True)
        hist2 = self._create_histogram(dist2, normalize=True)

        intersection = np.sum(np.minimum(hist1, hist2))

        # Calculate additional metrics
        kl_div_1_2 = np.sum(hist1 * np.log((hist1 + 1e-10) / (hist2 + 1e-10)))
        kl_div_2_1 = np.sum(hist2 * np.log((hist2 + 1e-10) / (hist1 + 1e-10)))

        # Bhattacharyya distance
        bhattacharyya = -np.log(np.sum(np.sqrt(hist1 * hist2)) + 1e-10)
        bhattacharyya_coeff = np.sum(np.sqrt(hist1 * hist2))
        return {
            "histogram_1": hist1,
            "histogram_2": hist2,
            "intersection": intersection,
            "kl_divergence_1_to_2": kl_div_1_2,
            "kl_divergence_2_to_1": kl_div_2_1,
            "bhattacharyya_distance": bhattacharyya,
            "bhattacharyya_coeff": bhattacharyya_coeff,
            "bin_edges": np.array([0, 1, 2]),
        }
