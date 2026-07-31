import os
import sys

import numpy as np
from Bio import SeqIO


# Command line inputs
fasta_file = sys.argv[1]
output_file = sys.argv[2]
bin_size = int(sys.argv[3])
expected_windows = int(sys.argv[4])


if bin_size <= 0:
    raise ValueError(
        f"bin_size must be positive, but received {bin_size}"
    )

if expected_windows <= 0:
    raise ValueError(
        f"expected_windows must be positive, "
        f"but received {expected_windows}"
    )

if not os.path.isfile(fasta_file):
    raise FileNotFoundError(
        f"FASTA file not found: {fasta_file}"
    )


one_hot_list = []


# Parse each FASTA sequence
for record_index, record in enumerate(
    SeqIO.parse(fasta_file, "fasta"),
    start=1,
):
    sequence = str(record.seq).upper()

    # Every genomic sequence must represent one complete window
    if len(sequence) != bin_size:
        raise ValueError(
            f"Sequence length mismatch for {record.id}: "
            f"expected {bin_size} bases, "
            f"but found {len(sequence)} bases. "
            f"FASTA record number: {record_index}"
        )

    # Columns represent A, C, G and T
    encoded_sequence = np.zeros(
        (bin_size, 4),
        dtype=np.int8,
    )

    for position, base in enumerate(sequence):
        if base == "A":
            encoded_sequence[position, 0] = 1
        elif base == "C":
            encoded_sequence[position, 1] = 1
        elif base == "G":
            encoded_sequence[position, 2] = 1
        elif base == "T":
            encoded_sequence[position, 3] = 1

        # N and other ambiguous bases remain [0, 0, 0, 0]

    one_hot_list.append(encoded_sequence)


actual_windows = len(one_hot_list)


# Stop if the FASTA contains no sequences
if actual_windows == 0:
    raise ValueError(
        f"No FASTA sequences were found in {fasta_file}"
    )


# Stop instead of adding artificial zero windows
if actual_windows != expected_windows:
    raise ValueError(
        f"Window count mismatch: expected {expected_windows} windows, "
        f"but the FASTA contains {actual_windows}. "
        "Check the BED windows and FASTA extraction before continuing."
    )


# Shape: number of windows, sequence length, four nucleotides
one_hot_array = np.stack(
    one_hot_list,
    axis=0,
)


# Add the model channel dimension
# Final shape: number of windows, 1, sequence length, 4
one_hot_array = one_hot_array[:, np.newaxis, :, :]


# Validate the final shape
expected_shape = (
    expected_windows,
    1,
    bin_size,
    4,
)

if one_hot_array.shape != expected_shape:
    raise ValueError(
        f"Unexpected DNA array shape: "
        f"expected {expected_shape}, "
        f"but created {one_hot_array.shape}"
    )


# Create the output directory when running outside Snakemake
output_directory = os.path.dirname(output_file)

if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True,
    )


# Save the validated array
np.save(
    output_file,
    one_hot_array,
)


print(f"FASTA records encoded: {actual_windows}")
print(f"DNA array shape: {one_hot_array.shape}")
print(f"DNA array data type: {one_hot_array.dtype}")
print(f"Saved DNA array to: {output_file}")
