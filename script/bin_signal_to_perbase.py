import sys
import pandas as pd
import numpy as np

def bin_signal_to_perbase(bdg_path, windows_path, output_path, window_size=8192, stride=600):
    """
    Maps genomic signal tracks (bedGraph) onto fixed-size windows.
    Generates a NumPy matrix of shape (n_windows, window_size).
    """
    
    # Load the region definitions (BED file)
    windows = pd.read_csv(windows_path, sep="\t", header=None, names=["chrom", "start", "end"])
    
    # Load the signal track (bedGraph)
    # Using specific dtypes optimizes memory usage significantly
    bdg = pd.read_csv(bdg_path, sep="\t", header=None,
                      names=["chrom", "start", "end", "score"],
                      dtype={"chrom": str, "start": np.int32, "end": np.int32, "score": np.float32})

    all_window_arrays = []
    
    # Process genome chromosome by chromosome to keep memory footprint low
    for chrom in windows["chrom"].unique():
        chrom_windows = windows[windows["chrom"] == chrom]
        chrom_bdg = bdg[bdg["chrom"] == chrom]
        
        # Handle cases where a chromosome has windows but no signal
        if chrom_bdg.empty:
            all_window_arrays.append(np.zeros((len(chrom_windows), window_size), dtype=np.float32))
            continue

        # Create a signal array for the entire chromosome
        max_coord = max(chrom_windows["end"].max(), chrom_bdg["end"].max())
        chrom_signal = np.zeros(max_coord + 1, dtype=np.float32)

        # Fill the signal array based on bedGraph intervals
        for row in chrom_bdg.itertuples(index=False):
            chrom_signal[row.start : row.end] = row.score

        # Extract fixed-size slices for every window
        win_data = [chrom_signal[w.start : w.start + window_size] for w in chrom_windows.itertuples(index=False)]
        
        # Pad windows that fall off the end of a chromosome with zeros
        win_data = [np.pad(w, (0, window_size - len(w))) if len(w) < window_size else w for w in win_data]
        
        all_window_arrays.append(np.array(win_data))

    # Combine all windows into one final matrix
    final_matrix = np.vstack(all_window_arrays)

    # Add channel dimension for ATAC-seq data (N, window_size, 1)
    if "atac" in output_path.lower():
        final_matrix = final_matrix[..., np.newaxis]

    # Save the resulting matrix
    np.save(output_path, final_matrix)
    print(f"Processing complete. Output shape: {final_matrix.shape}")

if __name__ == "__main__":
    # Ensure all required arguments are provided
    if len(sys.argv) < 6:
        print("Error: Missing arguments.")
        print("Usage: python script.py <bdg> <bed> <output> <window_size> <stride>")
        sys.exit(1)

    # Parse arguments passed from Snakemake
    bdg_input = sys.argv[1]
    bed_input = sys.argv[2]
    npy_output = sys.argv[3]
    win_size = int(sys.argv[4])
    stride_val = int(sys.argv[5])
    
    # Execute the binning process
    bin_signal_to_perbase(bdg_input, bed_input, npy_output, window_size=win_size, stride=stride_val)
