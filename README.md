A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila

This repository contains the preprocessing and machine-learning code used for the master's thesis:

A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila

Author: Aman Satyendra Yadav

The workflow processes public ATAC-seq, ChIP-seq and input-control datasets and converts them into model-ready genomic arrays. Three convolutional neural-network models are then used to predict continuous ChIP-seq signal profiles:

DNA-only model

ATAC plus Tn5 fragment model

Multimodal DNA plus ATAC and Tn5 model

Main files

File

Purpose

Snakefile

Runs the preprocessing workflow.

config.yaml

Contains preprocessing paths, reference files, metadata and window settings.

config_file_ml.yaml

Contains model paths, chromosome splits, seeds and training settings.

srr_metadata.csv

Lists ATAC, ChIP and input-control SRR accessions.

DataLoader3.py

Loads and aligns the processed genomic arrays.

Visualizer3.py

Visualizes ATAC and ChIP-seq relationships.

DNA_model3.py

Contains the DNA-only model.

ATAC_model3.py

Contains the ATAC plus Tn5 model.

ATAC_DNA_model3.py

Contains the multimodal model.

Insito_mutation3.py

Contains shuffled-pairing and in-silico perturbation analyses.

execution.ipynb

Notebook used to load data, train models and calculate results.

environment.yml

Conda environment specification.

script/

Helper scripts used by the Snakefile.

System requirements

The workflow is intended for Linux or an HPC cluster.

CPU: at least 8 cores; 16-32 cores are recommended for preprocessing

RAM: at least 32 GB; 64 GB or more is recommended

GPU: optional for preprocessing but recommended for model training

Storage: enough space for FASTQ, BAM, BedGraph, peak and NumPy files

Resource use depends on the number of sequencing datasets and the selected genomic-window size.

Do not run the complete workflow on a cluster login node. Request a compute node according to the rules of the local HPC system.

Installation

Create the Conda environment:

conda env create -n Testing -f environment.yml
conda activate Testing

If the environment already exists:

conda env update -n Testing -f environment.yml --prune

Verify Snakemake:

snakemake --version

Snakemake executor plug-ins do not install the main Snakemake package. If the command is missing, install it separately:

conda install -n Testing -c conda-forge -c bioconda snakemake

TensorFlow GPU support may also require a separate installation:

nvidia-smi
python -m pip install "tensorflow[and-cuda]"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

Many HPC compute nodes do not have internet access. If pip reports Network is unreachable, install TensorFlow from an internet-enabled node while the same Conda environment is active. Otherwise, use the cluster package mirror or offline packages.

Metadata file

The metadata CSV must contain these columns:

SRR,TYPE,INPUT_SRR

TYPE must be ATAC, CHIP or INPUT.

Each ChIP dataset must reference its matching input-control accession in INPUT_SRR.

INPUT_SRR remains empty for ATAC and INPUT rows.

The repository includes srr_metadata.csv with the accessions used in the thesis.

Preprocessing configuration

Edit config.yaml before running Snakemake. Check at least the following settings:

data_dir

results_dir

threads

srr_csv

windows.size

windows.stride

reference-genome and annotation URLs

The workflow uses the dm6 reference genome. Chromosome names must remain consistent, for example chr2L rather than 2L.

The output filename dm6_8kb_windows.bed is retained from an older version of the pipeline. The actual window size and stride are controlled by config.yaml.

Run preprocessing

Start with a dry run:

conda activate Testing
snakemake -n -p

Run the workflow:

snakemake --cores 32

The number of cores should match the resources assigned by the local system.

Main preprocessing outputs

Output

Location

Background-corrected ChIP arrays

results/perbase_chip_bgzero/{SRR}_perbase.npy

ATAC arrays

results/perbase_atac/{SRR}_perbase.npy

DNA one-hot array

results/dna_onehot.npy

Tn5 fragment files

results/ATAC_fragments/{SRR}_fragments.bed

Genomic windows

data/dm6_8kb_windows.bed

Before model training, confirm that the number and order of windows agree across the ChIP, ATAC, DNA, chromosome and Tn5 arrays.

Machine-learning configuration

Edit config_file_ml.yaml and check the following paths:

paths:
  chip_dir: "results/perbase_chip_bgzero"
  atac_dir: "results/perbase_atac"
  dna_path: "results/"
  chrms_path: "data/dm6_8kb_windows.bed"
  TN5_cutsite: "results/ATAC_fragments"

The configuration file also controls:

training, validation and test chromosomes

random seeds

epochs

batch sizes

early stopping

learning-rate reduction

peak weighting and background penalties

The chromosome groups must not overlap. Use the same chromosome split and target ChIP tracks when comparing the three models.

Load the project modules

from DataLoader3 import GenomicDataLoader
from Visualizer3 import Visualizer
from DNA_model3 import DNAPredictionModel, run_dna_seed_experiments
from ATAC_model3 import ATACPredictionModel, run_atac_seed_experiments
from ATAC_DNA_model3 import MultimodalGenomicModel, multimodal_seed_execution
from Insito_mutation3 import GenomicPermuter

The data loader must create these attributes before the model classes are initialized:

concatenated_data

window_chromosomes

dna_tracks

atac_tracks

names

The aligned Tn5 feature array must also be available as TN5_frags.

The supplied execution.ipynb notebook contains the loading, training, evaluation and visualization cells used for the thesis.

Run repeated-seed experiments

DNA-only model:

dna_results, dna_summary = run_dna_seed_experiments(
    loader,
    config_path="config_file_ml.yaml"
)

ATAC plus Tn5 model:

atac_results, atac_summary = run_atac_seed_experiments(
    loader,
    TN5_frags,
    config_path="config_file_ml.yaml"
)

Multimodal model:

multi_results, multi_summary = multimodal_seed_execution(
    loader,
    TN5_frags,
    config_path="config_file_ml.yaml"
)

The seed lists are read from config_file_ml.yaml. Results are written to results/seed_runs unless a different output directory is provided.

Evaluation

The models report:

weighted test loss

mean absolute error

pooled Pearson correlation

mean track-wise Pearson correlation

Wasserstein distance

Jensen-Shannon divergence

completed epochs

No single metric is sufficient. Sparse signals can produce a low MAE even when peak positions are predicted poorly. Compare all models using the same targets, preprocessing, chromosome split and random seeds.

Visualization and in-silico analysis

Visualizer3.py can be used to inspect:

ATAC and ChIP-seq profiles

active genomic windows

chromosome distributions

ATAC-ChIP correlations

genome-wide signal summaries

Insito_mutation3.py provides:

correct-versus-shuffled target comparison

repeated permutation testing

DNA substitution analysis

ATAC and Tn5 perturbation analysis

These perturbation results measure model sensitivity. They do not prove a causal biological relationship.

Common problems

snakemake: command not found

Activate the correct environment and install the main Snakemake package.

Network is unreachable

The selected node has no outbound internet connection. Install packages from an internet-enabled node or use the cluster package mirror.

TensorFlow detects no GPU

Confirm that a GPU was allocated, run nvidia-smi, and check that tensorflow[and-cuda] is installed in the active environment.

TN5_frags is not defined

Load or construct the aligned Tn5 fragment-feature array before running the ATAC or multimodal model.

Empty chromosome split

Check that chromosome names in config_file_ml.yaml match window_chromosomes and that each split contains windows.

Out-of-memory error

Reduce the relevant batch size or request additional RAM or GPU memory.

Reproducibility

Record the repository commit used for each analysis. Preserve the following files with the results:

environment.yml

config.yaml

config_file_ml.yaml

srr_metadata.csv

Snakemake logs

model checkpoints

per-seed result files

Package versions can be recorded with:

conda env export --from-history > environment-history.yml
conda list --explicit > environment-explicit.txt
conda list > package-versions.txt

Licence and data

The sequencing datasets are public data obtained from their original repositories. Users remain responsible for following the terms associated with each source dataset and software dependency.

Add a repository licence file before distributing or reusing the project code outside the thesis submission.
