'''This python script belongs to 
author = [AMAN YADAV]
script 2/6

as part of Masterthesis [A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila]
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

class Visualizer:
   
    def __init__(self, concatenated_data, window_chromosomes, track_names=None):
       
        self.concatenated_data = concatenated_data
        self.window_chroms = np.array(window_chromosomes) if window_chromosomes is not None else None   #<- shortened from window_chromosomes
        self.names = track_names

    #    fetching preloaded data from genomic loader=
    @classmethod
    def from_loader(cls, loader_instance):                                          #<- taking instaces directly from loader class 
        if not hasattr(loader_instance, 'concatenated_data') or loader_instance.concatenated_data is None:
            raise ValueError("The provided data loader instance hasn't merged metrics yet. Run add_chrmos() first!")
            
        return cls(
            concatenated_data=loader_instance.concatenated_data,
            window_chromosomes=getattr(loader_instance, 'window_chromosomes', None),
            track_names=getattr(loader_instance, 'names', None)
        )

    # =signal distribution of chip and atac (signal comaprision)
    
    def ATAC_vs_CHIP(self, window_idx=952, limit=True):                             
        #  fails if the chip is not already concatenated 
        if self.concatenated_data is None:
            raise ValueError("Master data array is uninitialized.")
        

        n_windows = self.concatenated_data.shape[0]
        n_channels = self.concatenated_data.shape[2]  
        win_size = self.concatenated_data.shape[1]      

        if window_idx >= n_windows or window_idx < 0:                              #<- gate kepper so the index dosnt go out of the limit 
            raise IndexError(f"window_idx {window_idx} is out of bounds for {n_windows} total windows.")
        #============limit handles number of plots===========================
        #============put limit if too many chip datasets=====================
        if limit is False or limit is True:
            n_tracks = n_channels
        elif isinstance(limit, int):
            if limit <= 0 or limit > n_channels:
                raise ValueError(f"Limit {limit} must be between 1 and {n_channels}.")
            n_tracks = limit
        else:
            n_tracks = n_channels

        n_tracks = int(n_tracks)

        #==========default names if not provided by the user=================================

        if self.names and len(self.names) >= n_tracks:
            track_names = self.names[:n_tracks]
        else:
            track_names = ["ATAC"] + [f"TF{i}" for i in range(1, n_tracks)]

        fig, axes = plt.subplots(n_tracks, 1, figsize=(12, max(4, n_tracks * 2.0)), sharex=True)

        if n_tracks == 1:
            axes = [axes]

        for i in range(n_tracks):
            sig = self.concatenated_data[window_idx, :, i]
            color = 'teal' if i == 0 else 'darkred'

            axes[i].plot(sig, color=color, linewidth=1.5)
            axes[i].fill_between(range(int(win_size)), sig, color=color, alpha=0.2)
            axes[i].set_ylabel(track_names[i], rotation=0, labelpad=40, fontweight='bold', va='center')

            max_val = np.max(sig)
            axes[i].set_ylim(0, max_val * 1.2 if max_val > 0 else 1)

        plt.xlabel("Position in Window (bp)", fontsize=12)
        loc_str = f" Location: {self.window_chroms[window_idx]}" if self.window_chroms is not None else ""      #<- checks for chrms in the plotted window
        plt.suptitle(f"Genomic Signals for Window {window_idx} ({n_tracks} Channels){loc_str}", fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("window_tracks.png", dpi=300)
        plt.show()


    # active windows across the training chromosomes 
    def active_windows(self, threshold_chip=0.2, threshold_atac=0.2):              #<- checking active windows if they go over the threshold 
        max_sigs = np.max(self.concatenated_data, axis=1)
        
        #user decided thresholds for active windows 
        atac_mask = max_sigs[:, 0] > threshold_atac
        chip_mask = np.all(max_sigs[:, 1:] > threshold_chip, axis=1)
        
        active_window = atac_mask & chip_mask
        active_index = np.where(active_window)[0]
        inactive_index = np.where(~active_window)[0]

        print(f"Filter Complete: Active: {len(active_index)} | Inactive: {len(inactive_index)}")
        return active_index, inactive_index



   
    #=======ATAC vs Chip tracks corelation profiles ===================
    #=================regression plots=================================
    def atac_chip_window_correlation(
        self,
        limit=True,
        summary_stat="max",
        method="pearson",
        save_csv=True,
        save_plot=True
    ):
        

        if self.concatenated_data is None:
            raise ValueError("Master data array is uninitialized.")

        if self.concatenated_data.shape[2] < 2:
            raise ValueError("Need at least one ATAC channel and one ChIP-seq channel.")

        from scipy.stats import pearsonr, spearmanr

        n_channels = self.concatenated_data.shape[2]

        if self.names and len(self.names) >= n_channels:
            track_names = self.names
        else:
            track_names = ["ATAC"] + [f"ChIP_track_{i}" for i in range(1, n_channels)]
        #============set limit to control plotting==========================
        
        if limit is True or limit is False:
            print("limit is active")
            selected_channels = list(range(1, n_channels))
            
        elif isinstance(limit, int):
            
            selected_channels = list(range(1, min(n_channels, limit + 1)))
        elif isinstance(limit, (list, tuple)):
            selected_channels = list(limit)
        else:
            raise ValueError("limit must be True, False, int, list, or tuple.")

        invalid_channels = [ch for ch in selected_channels if ch <= 0 or ch >= n_channels]
        if invalid_channels:
            raise ValueError(
                f"Invalid ChIP channel(s): {invalid_channels}. "
                f"Channel 0 is ATAC. Valid ChIP channels are 1 to {n_channels - 1}."
            )

        def summarize(signal_matrix):
            if summary_stat == "max":
                return np.max(signal_matrix, axis=1)
            elif summary_stat == "mean":
                return np.mean(signal_matrix, axis=1)
            elif summary_stat == "sum":
                return np.sum(signal_matrix, axis=1)
            else:
                raise ValueError("summary_stat must be 'max', 'mean', or 'sum'.")

        atac_window = summarize(self.concatenated_data[:, :, 0])

        results = []

        for ch in selected_channels:
            chip_window = summarize(self.concatenated_data[:, :, ch])

            valid_mask = np.isfinite(atac_window) & np.isfinite(chip_window)
            atac_valid = atac_window[valid_mask]
            chip_valid = chip_window[valid_mask]

            if np.std(atac_valid) < 1e-8 or np.std(chip_valid) < 1e-8:
                corr_value = np.nan
                p_value = np.nan
            else:
                if method == "pearson":
                    corr_value, p_value = pearsonr(atac_valid, chip_valid)
                elif method == "spearman":
                    corr_value, p_value = spearmanr(atac_valid, chip_valid)
                else:
                    raise ValueError("method must be either 'pearson' or 'spearman'.")

            results.append({
                "ChIP_channel": ch,
                "ChIP_track": track_names[ch],
                "summary_stat": summary_stat,
                f"{method}_correlation_with_ATAC": corr_value,
                "p_value": p_value,
                "n_windows_used": len(atac_valid)
            })

            plt.figure(figsize=(6, 5))
            plt.scatter(atac_valid, chip_valid, alpha=0.25, s=8)

            if len(atac_valid) > 2 and np.std(atac_valid) > 1e-8 and np.std(chip_valid) > 1e-8:
                slope, intercept = np.polyfit(atac_valid, chip_valid, 1)
                x_line = np.linspace(np.min(atac_valid), np.max(atac_valid), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, linestyle="--", linewidth=2, label="Linear fit")

            plt.xlabel(f"ATAC-seq window {summary_stat} signal")
            plt.ylabel(f"{track_names[ch]} ChIP-seq window {summary_stat} signal")
            plt.title(
                f"ATAC vs {track_names[ch]} ChIP-seq window signal\n"
                f"{method.capitalize()} r = {corr_value:.4f}"
            )
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()

            if save_plot:
                safe_name = str(track_names[ch]).replace(" ", "_").replace("/", "_")
                out_png = f"atac_vs_{safe_name}_{summary_stat}_{method}_window_scatter.png"
                plt.savefig(out_png, dpi=300)
                print(f"Saved window-level scatter plot to: {out_png}")

            plt.show()

        corr_df = pd.DataFrame(results)

        print("\nWindow-level ATAC vs selected ChIP-seq correlation:")
        print(corr_df)

        if save_csv:
            out_csv = f"atac_chip_window_{summary_stat}_{method}_correlation.csv"
            corr_df.to_csv(out_csv, index=False)
            print(f"\nSaved correlation table to: {out_csv}")

        return corr_df


    


    def plot_chromosomal_distribution(self, threshold_chip=0.2, threshold_atac=0.2):            #<- chromosomal distribution in pie chart 
        if self.window_chroms is None:
            raise ValueError("Chromosome labels array uninitialized. Run add_chrmos() first.")

        active_index, _ = self.active_windows(threshold_chip, threshold_atac)

        if len(active_index) == 0:
            print("No active windows found with these thresholds.")
            return

        chrom_labels = self.window_chroms[active_index]
        counts = Counter(chrom_labels)
        chroms = sorted(counts.keys())
        sizes = [counts[c] for c in chroms]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(chroms)))

        wedges, texts, autotexts = ax.pie(
            sizes, labels=chroms, autopct='%1.1f%%', startangle=140, colors=colors,
            pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1.5)
        )

        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=11)

        ax.text(0, 0, f"Total Active\nWindows\n\n{len(active_index)}",          #<- center label
                ha='center', va='center', fontsize=12, fontweight='bold')

        ax.legend(wedges, [f"{chroms[i]} (n={sizes[i]})" for i in range(len(chroms))],
                  title="Chromosomes", loc="center left", bbox_to_anchor=(1.1, 0, 0.5, 1))

        ax.set_title(f"Genomic Distribution of Active Windows\n(ATAC > {threshold_atac}, TFs > {threshold_chip})",
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.show()
        return fig




        
    def ATAC_vs_CHIP_multiple_windows(self, window_list=None, limit=True, smooth_atac=True, sigma=10): #<- multiple windows and chip tracks in one plot
        from scipy.ndimage import gaussian_filter1d
        
        if window_list is None:
            window_list = []

        n_wins = len(window_list)
        if n_wins == 0:
            print("Error: The windows list cannot be empty.")
            return

        n_channels = self.concatenated_data.shape[2]
        win_size = self.concatenated_data.shape[1]

        if limit is False or limit is True:
            n_tracks = n_channels
        elif isinstance(limit, int):
            if limit <= 0 or limit > n_channels:
                raise ValueError(f"Limit {limit} must be between 1 and {n_channels}.")
            n_tracks = limit
        else:
            n_tracks = n_channels

        n_tracks = int(n_tracks)
        n_wins = int(n_wins)

        if self.names and len(self.names) >= n_tracks:
            track_names = self.names[:n_tracks]
        else:
            track_names = ["ATAC"] + [f"TF{i}" for i in range(1, n_tracks)]

        colors = ['#1f77b4'] + ['#d62728'] * (n_tracks - 1)
        
        fig, axes = plt.subplots(n_tracks, n_wins, 
                                 figsize=(4 * n_wins, max(5, n_tracks * 2.5)), 
                                 sharex='col', sharey='row')
        
        if n_tracks == 1 or n_wins == 1:
            axes = np.atleast_2d(axes)
            if n_wins == 1 and n_tracks > 1:
                axes = axes.T

        for col_idx, win_val in enumerate(window_list):
            chrom_label = self.window_chroms[win_val] if self.window_chroms is not None else "Unknown"        #<- checks for the chrms in the window 
            
            for row_idx in range(n_tracks): 
                ax = axes[row_idx, col_idx]
                sig = self.concatenated_data[win_val, :, row_idx]
                
                if row_idx == 0 and smooth_atac:                                    #<- Conditionally applies smoothing
                    sig = gaussian_filter1d(sig, sigma=sigma)
                    
                bp_pos = np.arange(win_size)
                
                ax.plot(bp_pos, sig, color=colors[row_idx], linewidth=1.4)
                ax.fill_between(bp_pos, sig, color=colors[row_idx], alpha=0.18)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelsize=9)
                
                if col_idx == 0:
                    ax.set_ylabel(track_names[row_idx], fontsize=11, fontweight='bold', rotation=0, labelpad=30, ha='right')
                if row_idx == 0:
                    ax.set_title(f"Window: {win_val}\n({chrom_label})", fontsize=12, fontweight='bold', pad=10)
                if row_idx == (n_tracks - 1):
                    ax.set_xlabel("Relative position (bp)", fontsize=10)

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.12, top=0.88)                                  #<- Reserves structural buffer zones
        
        fig.suptitle(f"Master Thesis: Comparison of {n_tracks} Genomic Signals Across {n_wins} Windows", 
                     fontsize=16, fontweight='bold', y=0.96)
        
        plt.show()
        return fig


            #================whole-genome ATAC and selected ChIP-seq tracks================
    def whole_genome_view(
        self, chip_tracks=(0,), summary_stat="mean", max_points=5000
    ):
        """
        Plot ATAC at the top and selected ChIP tracks below it.
        ChIP numbering starts from 0 and excludes the ATAC channel.
        """
        if self.concatenated_data is None or self.window_chroms is None:
            raise ValueError("Merged signals or chromosome labels are missing.")

        data = np.asarray(self.concatenated_data)
        chrom_labels = np.asarray(self.window_chroms).astype(str)
        n_windows, _, n_channels = data.shape
        n_chip = n_channels - 1

        if isinstance(chip_tracks, (int, np.integer)):
            chip_tracks = [int(chip_tracks)]
        else:
            chip_tracks = [int(track) for track in chip_tracks]

        if not chip_tracks:
            raise ValueError("Select at least one ChIP track.")
        if len(set(chip_tracks)) != len(chip_tracks):
            raise ValueError("chip_tracks contains duplicate tracks.")
        if any(track < 0 or track >= n_chip for track in chip_tracks):
            raise ValueError(f"ChIP tracks must be between 0 and {n_chip - 1}.")
        if len(chrom_labels) != n_windows:
            raise ValueError("Chromosome labels do not match the windows.")
        if summary_stat not in {"mean", "max"}:
            raise ValueError("summary_stat must be 'mean' or 'max'.")
        if max_points < 2:
            raise ValueError("max_points must be at least 2.")

        #<- summarize the signal within every genomic window
        summarize = np.mean if summary_stat == "mean" else np.max
        window_signals = summarize(data, axis=1)

        chromosomes = list(dict.fromkeys(chrom_labels))
        grouped_indices, x_parts = [], []
        boundaries, midpoints, labels = [], [], []
        cursor = 0

        #<- divide each chromosome into a readable number of plotting bins
        for chromosome in chromosomes:
            indices = np.flatnonzero(chrom_labels == chromosome)
            n_bins = min(
                len(indices),
                max(1, round(max_points * len(indices) / n_windows))
            )
            groups = np.array_split(indices, n_bins)

            grouped_indices.extend(groups)
            x_parts.append(np.arange(cursor, cursor + n_bins))
            midpoints.append(cursor + (n_bins - 1) / 2)
            labels.append(chromosome)

            cursor += n_bins
            boundaries.append(cursor - 0.5)

        x = np.concatenate(x_parts)

        #<- average window summaries inside each plotting bin
        def binned_signal(channel):
            return np.array([
                np.nanmean(window_signals[group, channel])
                for group in grouped_indices
            ])

        names = list(self.names) if self.names is not None else []

        def chip_name(track):
            channel = track + 1  #<- channel 0 is ATAC
            if len(names) >= n_channels:
                return names[channel]
            if len(names) == n_chip:
                return names[track]
            return f"ChIP track {track}"

        channels = [0] + [track + 1 for track in chip_tracks]
        plot_names = ["ATAC"] + [chip_name(track) for track in chip_tracks]
        colors = ["teal"] + list(
            plt.cm.tab10(np.linspace(0, 1, len(chip_tracks)))
        )

        n_plots = len(channels)
        fig, axes = plt.subplots(
            n_plots, 1,
            figsize=(24, max(4, 3 * n_plots)),
            sharex=True,
            squeeze=False
        )
        axes = axes[:, 0]

        for ax, channel, name, color in zip(
            axes, channels, plot_names, colors
        ):
            ax.plot(
                x, binned_signal(channel),
                color=color, linewidth=0.8, label=name
            )

            for boundary in boundaries[:-1]:
                ax.axvline(
                    boundary, color="grey",
                    linewidth=0.6, alpha=0.5
                )

            ax.set_ylabel(f"{name}\n{summary_stat} signal")
            ax.legend(loc="upper right")
            ax.grid(axis="y", alpha=0.2)
            ax.margins(x=0)

        axes[-1].set_xticks(midpoints)
        axes[-1].set_xticklabels(labels, rotation=45, ha="right")
        axes[-1].set_xlabel(
            "Chromosomes concatenated in genomic-window order"
        )

        fig.suptitle(
            "Whole-genome ATAC and ChIP-seq signal overview",
            fontsize=16,
            fontweight="bold"
        )
        fig.tight_layout()
        plt.show()

        return fig
