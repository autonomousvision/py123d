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

    def calculate_intersection(
        self, dist1: npt.NDArray[np.float64], dist2: npt.NDArray[np.float64], log_mask: npt.NDArray[np.bool_]
    ) -> Dict[str, float]:
        assert dist1.shape[0] == dist2.shape[0], "Distributions must have the same number of objects"
        assert dist1.ndim == 2
        assert dist2.ndim == 2
        assert log_mask.ndim == 2

        intersection = 0.0

        if self.independent_timesteps:
            # (n_objects, n_rollouts * n_steps)
            for obj_dist1, obj_dist2, obj_mask in zip(dist1, dist2, log_mask):
                intersection += self._calculate_intersection(obj_dist1[obj_mask], obj_dist2[obj_mask])
            intersection /= dist1.shape[0]  # Average intersection over all objects

        else:
            raise NotImplementedError

        return intersection

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


# # Example usage
# if __name__ == "__main__":
#     # Generate sample data
#     np.random.seed(42)

#     # Two normal distributions with different parameters
#     dist1 = np.random.normal(5, 2, 1000).tolist()
#     dist2 = np.random.normal(7, 1.5, 1000).tolist()

#     # Create metric with specified range and bins
#     metric = HistogramIntersectionMetric(min_val=0, max_val=12, n_bins=20)

#     # Calculate intersection
#     intersection = metric.calculate_intersection(dist1, dist2)
#     print(f"Histogram Intersection: {intersection:.4f}")

#     # Plot histograms
#     metric.plot_histograms(dist1, dist2, labels=("Normal(5,2)", "Normal(7,1.5)"))

#     # Detailed analysis
#     analysis = metric.detailed_analysis(dist1, dist2)
#     print(f"KL Divergence (1→2): {analysis['kl_divergence_1_to_2']:.4f}")
#     print(f"KL Divergence (2→1): {analysis['kl_divergence_2_to_1']:.4f}")
#     print(f"Bhattacharyya Distance: {analysis['bhattacharyya_distance']:.4f}")

#     # Example with more similar distributions
#     print("\n" + "=" * 50)
#     print("Example with more similar distributions:")

#     dist3 = np.random.normal(6, 1.8, 1000).tolist()
#     dist4 = np.random.normal(6.5, 1.6, 1000).tolist()

#     intersection2 = metric.calculate_intersection(dist3, dist4)
#     print(f"Histogram Intersection: {intersection2:.4f}")

#     metric.plot_histograms(
#         dist3, dist4, labels=("Normal(6,1.8)", "Normal(6.5,1.6)"), title="More Similar Distributions"
#     )
