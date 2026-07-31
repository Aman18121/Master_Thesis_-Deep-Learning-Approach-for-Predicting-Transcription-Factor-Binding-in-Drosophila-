'''This python script belongs to 
author = [AMAN YADAV]
script 6/6
as part of Masterthesis [A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila]
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr



# random permutation and input perturbation tests for the trained multimodal model
# refer to the permutation-control and in-silico perturbation analysis in the main thesis report
class GenomicPermuter:
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        self.X_Atac = None
        self.X_Dna = None
        self.Y_chip = None
        self.Y_chip_permuted = None
        self.shuffled_indices = None

        self.multi_true_results = {}
        self.multi_perm_results = {}

        print(f"GenomicPermuter is active with seed {seed}.")



    # check that ATAC, DNA and ChIP arrays contain matching genomic windows
    @staticmethod
    def _validate_multimodal_inputs(multimodal_inputs, Y_chip):
        if not isinstance(multimodal_inputs, (list, tuple)) or len(multimodal_inputs) != 2:
            raise ValueError("multimodal_inputs must be [X_Atac, X_Dna].")

        X_Atac, X_Dna = multimodal_inputs

        if X_Atac is None or X_Dna is None or Y_chip is None:
            raise ValueError("ATAC, DNA and ChIP arrays cannot be None.")
        if len(X_Atac) != len(Y_chip) or len(X_Dna) != len(Y_chip):
            raise ValueError("ATAC, DNA and ChIP arrays must contain the same number of windows.")
        if np.asarray(X_Atac).ndim != 3 or np.asarray(X_Dna).ndim != 3 or np.asarray(Y_chip).ndim != 3:
            raise ValueError("ATAC, DNA and ChIP arrays must be three-dimensional.")
        if np.asarray(X_Atac).shape[:2] != np.asarray(X_Dna).shape[:2] or np.asarray(X_Dna).shape[:2] != np.asarray(Y_chip).shape[:2]:
            raise ValueError("ATAC, DNA and ChIP window dimensions do not align.")



    # randomly disconnect ChIP targets from their matching ATAC and DNA windows
    # this preserves the target distribution but removes window-level biological pairing
    def fit_transformer_random(self, multimodal_inputs, Y_chip):
        self._validate_multimodal_inputs(multimodal_inputs, Y_chip)

        self.X_Atac = np.asarray(multimodal_inputs[0])
        self.X_Dna = np.asarray(multimodal_inputs[1])
        self.Y_chip = np.asarray(Y_chip)

        total_windows = len(self.Y_chip)
        self.shuffled_indices = self.rng.permutation(total_windows)
        self.Y_chip_permuted = self.Y_chip[self.shuffled_indices]

        print(f"Biological pairing removed across {total_windows} genomic windows.")
        return [self.X_Atac, self.X_Dna], self.Y_chip_permuted



    # compare normal test pairing against the shuffled-target control
    # a performance drop after shuffling shows dependence on correct genomic pairing
    # refer to the shuffled-target control results in the main thesis report
    def evaluate_models(self, trained_model, batch_size=16, max_correlation_windows=None):
        if self.Y_chip_permuted is None:
            raise ValueError("No permuted targets found. Run fit_transformer_random() first.")
        if trained_model is None:
            raise ValueError("No trained multimodal model was provided.")

        multimodal_inputs = [self.X_Atac, self.X_Dna]

        print("Computing performance on correctly paired test data...")
        self.multi_true_results = trained_model.evaluate(
            x=multimodal_inputs, y=self.Y_chip,
            batch_size=batch_size, verbose=0, return_dict=True
        )

        print("Computing performance on shuffled-target test data...")
        self.multi_perm_results = trained_model.evaluate(
            x=multimodal_inputs, y=self.Y_chip_permuted,
            batch_size=batch_size, verbose=0, return_dict=True
        )

        print("Calculating pooled and track-wise correlations...")
        multi_predictions = trained_model.predict(
            multimodal_inputs, batch_size=batch_size, verbose=0
        )

        if max_correlation_windows is None:
            selected_indices = np.arange(len(self.Y_chip))
        else:
            total_selected = min(int(max_correlation_windows), len(self.Y_chip))
            if total_selected < 1:
                raise ValueError("max_correlation_windows must select at least one window.")
            selected_indices = self.rng.choice(
                len(self.Y_chip), size=total_selected, replace=False
            )

        selected_true_chip = self.Y_chip[selected_indices]
        selected_permuted_chip = self.Y_chip_permuted[selected_indices]
        selected_predictions = multi_predictions[selected_indices]

        true_chip_values = selected_true_chip.ravel()
        permuted_chip_values = selected_permuted_chip.ravel()
        pred_chip_values = selected_predictions.ravel()

        true_mask = np.isfinite(true_chip_values) & np.isfinite(pred_chip_values)
        perm_mask = np.isfinite(permuted_chip_values) & np.isfinite(pred_chip_values)

        true_chip_values = true_chip_values[true_mask]
        true_pred_values = pred_chip_values[true_mask]
        permuted_chip_values = permuted_chip_values[perm_mask]
        perm_pred_values = pred_chip_values[perm_mask]

        if true_chip_values.size > 1 and np.std(true_chip_values) > 1e-6 and np.std(true_pred_values) > 1e-6:
            self.multi_true_results["pearson_r"] = pearsonr(true_chip_values, true_pred_values)[0]
            self.multi_true_results["spearman_rho"] = spearmanr(true_chip_values, true_pred_values)[0]
        else:
            self.multi_true_results["pearson_r"] = 0.0
            self.multi_true_results["spearman_rho"] = 0.0

        if permuted_chip_values.size > 1 and np.std(permuted_chip_values) > 1e-6 and np.std(perm_pred_values) > 1e-6:
            self.multi_perm_results["pearson_r"] = pearsonr(permuted_chip_values, perm_pred_values)[0]
            self.multi_perm_results["spearman_rho"] = spearmanr(permuted_chip_values, perm_pred_values)[0]
        else:
            self.multi_perm_results["pearson_r"] = 0.0
            self.multi_perm_results["spearman_rho"] = 0.0

        true_track_pearson = []
        perm_track_pearson = []
        total_channels = selected_true_chip.shape[-1]

        for channel in range(total_channels):
            true_channel = selected_true_chip[:, :, channel].ravel()
            perm_channel = selected_permuted_chip[:, :, channel].ravel()
            pred_channel = selected_predictions[:, :, channel].ravel()

            true_mask = np.isfinite(true_channel) & np.isfinite(pred_channel)
            perm_mask = np.isfinite(perm_channel) & np.isfinite(pred_channel)

            true_channel = true_channel[true_mask]
            true_pred_channel = pred_channel[true_mask]
            perm_channel = perm_channel[perm_mask]
            perm_pred_channel = pred_channel[perm_mask]

            if true_channel.size > 1 and np.std(true_channel) > 1e-6 and np.std(true_pred_channel) > 1e-6:
                true_track_pearson.append(pearsonr(true_channel, true_pred_channel)[0])
            else:
                true_track_pearson.append(np.nan)

            if perm_channel.size > 1 and np.std(perm_channel) > 1e-6 and np.std(perm_pred_channel) > 1e-6:
                perm_track_pearson.append(pearsonr(perm_channel, perm_pred_channel)[0])
            else:
                perm_track_pearson.append(np.nan)

        self.multi_true_results["mean_track_pearson"] = float(np.nanmean(true_track_pearson))
        self.multi_true_results["track_pearson_sd"] = float(np.nanstd(true_track_pearson))
        self.multi_perm_results["mean_track_pearson"] = float(np.nanmean(perm_track_pearson))
        self.multi_perm_results["track_pearson_sd"] = float(np.nanstd(perm_track_pearson))

        print("Permutation-control summary:")
        for metric in self.multi_true_results:
            true_value = self.multi_true_results[metric]
            perm_value = self.multi_perm_results[metric]
            print(f"{metric}: true={true_value:.5f} | shuffled={perm_value:.5f}")

        return self.multi_true_results, self.multi_perm_results



    # build an empirical null distribution by repeatedly shuffling ChIP windows
    # the empirical p-value tests whether observed mean track correlation exceeds random pairing
    # the plus-one correction prevents a zero p-value from a finite permutation sample
    def permutation_correlation_test(
        self,
        trained_model,
        n_permutations=100,
        batch_size=16,
        max_windows=None
    ):
        if self.X_Atac is None or self.X_Dna is None or self.Y_chip is None:
            raise ValueError("Run fit_transformer_random() before the permutation test.")
        if n_permutations < 1:
            raise ValueError("n_permutations must be at least 1.")

        multi_predictions = trained_model.predict(
            [self.X_Atac, self.X_Dna],
            batch_size=batch_size,
            verbose=0
        )

        if max_windows is None:
            selected_indices = np.arange(len(self.Y_chip))
        else:
            total_selected = min(int(max_windows), len(self.Y_chip))
            if total_selected < 1:
                raise ValueError("max_windows must select at least one window.")
            selected_indices = self.rng.choice(
                len(self.Y_chip), size=total_selected, replace=False
            )

        selected_chip = self.Y_chip[selected_indices]
        selected_predictions = multi_predictions[selected_indices]

        observed_track_pearson = []
        total_channels = selected_chip.shape[-1]

        for channel in range(total_channels):
            true_values = selected_chip[:, :, channel].ravel()
            pred_values = selected_predictions[:, :, channel].ravel()
            valid_mask = np.isfinite(true_values) & np.isfinite(pred_values)
            true_values = true_values[valid_mask]
            pred_values = pred_values[valid_mask]

            if true_values.size > 1 and np.std(true_values) > 1e-6 and np.std(pred_values) > 1e-6:
                observed_track_pearson.append(pearsonr(true_values, pred_values)[0])
            else:
                observed_track_pearson.append(np.nan)

        observed_mean = float(np.nanmean(observed_track_pearson))
        null_mean_correlations = []

        for _ in range(n_permutations):
            shuffled_chip = selected_chip[self.rng.permutation(len(selected_chip))]
            shuffled_track_pearson = []

            for channel in range(total_channels):
                shuffled_values = shuffled_chip[:, :, channel].ravel()
                pred_values = selected_predictions[:, :, channel].ravel()
                valid_mask = np.isfinite(shuffled_values) & np.isfinite(pred_values)
                shuffled_values = shuffled_values[valid_mask]
                pred_values = pred_values[valid_mask]

                if shuffled_values.size > 1 and np.std(shuffled_values) > 1e-6 and np.std(pred_values) > 1e-6:
                    shuffled_track_pearson.append(pearsonr(shuffled_values, pred_values)[0])
                else:
                    shuffled_track_pearson.append(np.nan)

            null_mean_correlations.append(float(np.nanmean(shuffled_track_pearson)))

        null_mean_correlations = np.asarray(null_mean_correlations, dtype=float)
        empirical_p_value = float(
            # plus-one correction described in the permutation analysis of the thesis report
            (1 + np.sum(null_mean_correlations >= observed_mean))
            / (n_permutations + 1)
        )

        return {
            "observed_mean_track_pearson": observed_mean,
            "observed_track_pearson": observed_track_pearson,
            "null_mean_track_pearson": null_mean_correlations,
            "null_mean": float(np.nanmean(null_mean_correlations)),
            "null_sd": float(np.nanstd(null_mean_correlations, ddof=1))
            if n_permutations > 1 else 0.0,
            "empirical_p_value": empirical_p_value,
            "n_permutations": int(n_permutations)
        }



    # mutate every DNA position while keeping the matching chromatin input unchanged
    # this is an in-silico sensitivity test and should not be interpreted as causal binding evidence
    # refer to the DNA perturbation analysis and limitations in the main thesis report
    def DNA_mutation_test(self, trained_model, start_idx=0, end_idx=5, plot=True):
        if self.X_Atac is None or self.X_Dna is None:
            raise ValueError("Run fit_transformer_random() before DNA mutation testing.")

        _, sequence_length, total_bases = self.X_Dna.shape

        if start_idx < 0:
            raise ValueError("start_idx cannot be negative.")
        if end_idx > len(self.X_Dna):
            raise ValueError("end_idx exceeds the number of available test windows.")
        if start_idx >= end_idx:
            raise ValueError("start_idx must be smaller than end_idx.")

        total_selected = end_idx - start_idx
        impact_matrix = np.zeros((sequence_length, total_bases), dtype=np.float32)

        print(f"Mutating all {sequence_length} DNA positions from window {start_idx} to {end_idx}...")

        for target_idx in range(start_idx, end_idx):
            ATAC_feed = self.X_Atac[target_idx:target_idx + 1]
            Dna_feed = self.X_Dna[target_idx:target_idx + 1]
            reference_prediction = trained_model.predict([ATAC_feed, Dna_feed], verbose=0)[0]

            for position in range(sequence_length):
                original_base = int(np.argmax(Dna_feed[0, position]))
                mutated_bases = []

                for base in range(total_bases):
                    if base != original_base:
                        mutated_bases.append(base)

                mutation_batch_size = len(mutated_bases)
                batch_Atac = np.repeat(ATAC_feed, mutation_batch_size, axis=0)
                batch_Dna = np.repeat(Dna_feed, mutation_batch_size, axis=0)

                for mutation_index, mutated_base in enumerate(mutated_bases):
                    batch_Dna[mutation_index, position, original_base] = 0.0
                    batch_Dna[mutation_index, position, mutated_base] = 1.0

                mutated_predictions = trained_model.predict([batch_Atac, batch_Dna], verbose=0)

                for mutation_index, mutated_base in enumerate(mutated_bases):
                    impact_matrix[position, mutated_base] += np.mean(
                        np.abs(reference_prediction - mutated_predictions[mutation_index])
                    )

        impact_matrix = impact_matrix / total_selected
        print("DNA mutation test completed.")

        if plot:
            plt.figure(figsize=(12, 4))
            plt.imshow(impact_matrix.T, aspect="auto", cmap="viridis")
            plt.colorbar(label="Mean prediction change")
            plt.yticks(range(4), ["A", "C", "G", "T"])
            plt.xlabel("DNA position")
            plt.ylabel("Alternative base")
            plt.title("DNA Mutation Impact")
            plt.tight_layout()
            plt.show()

        return impact_matrix



    # scale ATAC and TN5 features around the strongest chromatin position
    # X_Atac contains ATAC plus TN5 channels, so this tests their joint chromatin contribution
    # refer to the chromatin perturbation analysis in the main thesis report
    def ATAC_mutation_test(self, trained_model, multipliers=None, radii=None, target_idx=0):
        if self.X_Atac is None or self.X_Dna is None:
            raise ValueError("Run fit_transformer_random() before ATAC mutation testing.")
        if target_idx < 0 or target_idx >= len(self.X_Atac):
            raise ValueError("target_idx is outside the available test-window range.")

        if multipliers is None:
            multipliers = [0.0, 0.5, 1.5, 2.0]
        if radii is None:
            radii = [50, 250, 500]

        ATAC_feed = np.copy(self.X_Atac[target_idx:target_idx + 1])
        Dna_feed = self.X_Dna[target_idx:target_idx + 1]

        sequence_length = ATAC_feed.shape[1]
        signal_per_position = np.mean(np.abs(ATAC_feed[0]), axis=-1)
        center_position = int(np.argmax(signal_per_position))

        print(f"Strongest chromatin signal found at position {center_position}.")
        print(f"Peak signal value: {signal_per_position[center_position]:.6f}")

        ATAC_curve_data = {}

        for radius in radii:
            ATAC_curve_data[radius] = {}
            start_position = max(0, center_position - radius)
            end_position = min(sequence_length, center_position + radius)

            print(
                f"radius ±{radius}: positions [{start_position}:{end_position}], "
                f"mean signal={np.mean(signal_per_position[start_position:end_position]):.6f}"
            )

            for multiplier in multipliers:
                perturbed_Atac = np.copy(ATAC_feed)
                perturbed_Atac[0, start_position:end_position, :] *= multiplier
                multi_prediction = trained_model.predict(
                    [perturbed_Atac, Dna_feed], verbose=0
                )[0]
                ATAC_curve_data[radius][multiplier] = multi_prediction

        print("ATAC perturbation grid completed.")
        return ATAC_curve_data



    # plot the change in model output after local chromatin scaling
    # positive or negative changes show sensitivity of predicted ChIP signal to chromatin scaling
    def plot_dimmer_curves(self, trained_model, curve_data, target_idx=0, save_path="dimmer_curves.png"):
        if self.X_Atac is None or self.X_Dna is None:
            raise ValueError("Run fit_transformer_random() before plotting perturbation curves.")
        if target_idx < 0 or target_idx >= len(self.X_Atac):
            raise ValueError("target_idx is outside the available test-window range.")

        ATAC_feed = self.X_Atac[target_idx:target_idx + 1]
        Dna_feed = self.X_Dna[target_idx:target_idx + 1]
        reference_prediction = trained_model.predict([ATAC_feed, Dna_feed], verbose=0)[0]

        sequence_length = ATAC_feed.shape[1]
        signal_per_position = np.mean(np.abs(ATAC_feed[0]), axis=-1)
        center_position = int(np.argmax(signal_per_position))

        fig, axis = plt.subplots(figsize=(8, 5))

        for radius in curve_data:
            multipliers = sorted(curve_data[radius])
            prediction_changes = []
            start_position = max(0, center_position - radius)
            end_position = min(sequence_length, center_position + radius)

            for multiplier in multipliers:
                multi_prediction = curve_data[radius][multiplier]
                mean_change = np.mean(
                    multi_prediction[start_position:end_position]
                    - reference_prediction[start_position:end_position]
                )
                prediction_changes.append(mean_change)

            axis.plot(
                multipliers, prediction_changes,
                marker="o", linewidth=2,
                label=f"Window radius: ±{radius} positions"
            )

        axis.axhline(0, color="black", linestyle="--", alpha=0.5)
        axis.set_title("ATAC Accessibility Sensitivity", fontsize=11, fontweight="bold", pad=12)
        axis.set_xlabel("ATAC scaling factor\n(0 = removed | 1 = original | 2 = doubled)", fontsize=10, labelpad=8)
        axis.set_ylabel("Mean change in predicted ChIP-seq signal", fontsize=10)
        axis.legend(loc="best")
        axis.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"ATAC sensitivity curve saved to: {save_path}")



    # compare one metric between correct and shuffled target pairing
    # this figure provides the visual summary used for the permutation-control comparison in the report
    def plot_comparison(self, target_metric="mae", save_path="permutation_control.png"):
        if not self.multi_true_results or not self.multi_perm_results:
            raise ValueError("Run evaluate_models() first.")

        target_metric = target_metric.lower()
        if target_metric not in self.multi_true_results:
            raise ValueError(
                f"Metric '{target_metric}' is unavailable. "
                f"Available metrics: {list(self.multi_true_results)}"
            )

        true_value = self.multi_true_results[target_metric]
        perm_value = self.multi_perm_results[target_metric]

        fig, axis = plt.subplots(figsize=(6, 5))
        lower_is_better = "mae" in target_metric or "loss" in target_metric
        colors = ["#2ca02c", "#d62728"] if lower_is_better else ["#1f77b4", "#ff7f0e"]

        bars = axis.bar(
            ["Correct pairing", "Shuffled-target control"],
            [true_value, perm_value],
            color=colors, edgecolor="black", width=0.4
        )

        maximum_value = max(abs(true_value), abs(perm_value), 1e-7)

        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + maximum_value * 0.02,
                f"{height:.4f}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )

        if lower_is_better:
            y_label = f"{target_metric.upper()} (lower is better)"
            title = "Prediction Error: Correct vs Shuffled Pairing"
        else:
            y_label = f"{target_metric.upper()} (higher is better)"
            title = "Correlation: Correct vs Shuffled Pairing"

        axis.set_title(title, fontsize=10, fontweight="bold", pad=15)
        axis.set_ylabel(y_label, fontsize=10)
        axis.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return fig
