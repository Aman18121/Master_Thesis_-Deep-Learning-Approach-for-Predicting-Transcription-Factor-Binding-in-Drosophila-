
# =============================================================================
# Drosophila ChIP-seq and ATAC-seq Preprocessing Workflow
# =============================================================================
#
# This Snakefile defines a configuration- and metadata-driven workflow for
# processing public ChIP-seq, input-control, and ATAC-seq datasets into
# genome-aligned, model-ready representations using the dm6 reference genome.
#
# The workflow performs read retrieval, quality control, adapter trimming,
# alignment, filtering, peak calling, signal normalization, genomic-window
# generation, signal binning, and DNA one-hot encoding. Dataset accessions and
# experiment types are supplied through the metadata CSV file, while paths,
# computational resources, reference files, adapters, and window parameters are
# specified in config.yaml.
#
# Primary outputs include normalized ChIP-seq target profiles, normalized ATAC-seq
# coverage, ATAC/Tn5-derived fragment representations, fixed genomic windows,
# and one-hot-encoded DNA sequence arrays for downstream machine-learning models.
#
# Requirements:
#   - A correctly configured config.yaml file
#   - A metadata CSV containing SRR, TYPE, and INPUT_SRR columns
#   - The required bioinformatics tools and Python dependencies
#   - Internet access for downloading SRA datasets and reference files
#
# Run from the workflow directory with:
#   snakemake --cores <number_of_cores>
#
# Author: [Aman S Yadav]
# Version: [part of master thesis (A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila)]
# Date: [2026-08-01]


# =====================================================================(!!!!!!  Imports   !!!!!!)==================================================================================================

import pandas as pd
import os



#---------------------------------------------------------------------------(control pannel) ------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
configfile: "config.yaml"    # Recommended: keep config.yaml in the same directory as this Snakefile.                                


# -------------------------------
# Config file access
# Do not change unless config.yaml structure changes
# -------------------------------

#---------------------------------Directory paths (chnage only through the config file --------
DATA_DIR = config["data_dir"]                   
RESULTS_DIR = config["results_dir"]
THREADS = config["threads"]





#-------------------------------------Window sizes and strides config---------------------------

WINDOW_SIZE = config["windows"]["size"]
WINDOW_STRIDE = config["windows"]["stride"]

REFERENCE = config["reference"]
GENOME_FASTA_URL = REFERENCE["genome_fasta_url"]
ANNOTATION_GTF_URL = REFERENCE["annotation_gtf_url"]




# -------------------------------
# CSV file access
# Do not change unless CSV structure changes (fragile ! ) ---------------------------------
# -------------------------------

df = pd.read_csv(config["srr_csv"])
SRR_IDS = list(df["SRR"])     
ATAC_IDS = list(df[df["TYPE"] == "ATAC"]["SRR"])               #<- Targeting Type ATAC
CHIP_IDS = list(df[df["TYPE"] == "CHIP"]["SRR"])               #<- Targeting Type CHIP
INPUT_IDS = list(df[df["TYPE"] == "INPUT"]["SRR"])             #<- Targeting Type Input Ids 
chip_to_input = dict(df[df.TYPE=="CHIP"][["SRR","INPUT_SRR"]].values)  #<- Targeting Input with their corresponding chip
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------rule all follows systematic executions (chnaging the positions can risk brekagae)-----------------------------


rule all:
    input:
    # R1 FastQ files (Targets downloading)----------------------------------
        expand(f"{DATA_DIR}/{{srr}}_1.fastq.gz", srr=SRR_IDS),


    # FastQC outputs for R1-------------------------------------------------
        expand(f"{RESULTS_DIR}/qc/{{srr}}_1_fastqc.html", srr=SRR_IDS),	



	# DM6 genome download and annotation--------------------------------------------
        f"{DATA_DIR}/dm6.fa",
        f"{DATA_DIR}/dm6.gtf",
        f"{DATA_DIR}/dm6_blacklist.bed",



    # user defined kb windows--------(The name 8kb follows older version of the pipline) ---------------------------------------------------
        f"{DATA_DIR}/dm6_8kb_windows.bed",
	f"{DATA_DIR}/dm6.fa.fai",



	# R1 trimmed FastQ files from fastp-------------------------------------
        *expand(f"{RESULTS_DIR}/trimmed_fastp/{{srr}}_1_trimmed.fastq.gz", srr=SRR_IDS),



    # fastp HTML reports----------------------------------------------------
        *expand(f"{RESULTS_DIR}/fastp_reports/{{srr}}_fastp.html", srr=SRR_IDS),
	*expand(f"{RESULTS_DIR}/trimmed_fastp/{{srr}}_1_trimmed.fastq.gz", srr=SRR_IDS),        



    # fastp JSON reports-----------------------------------------------------
        *expand(f"{RESULTS_DIR}/fastp_reports/{{srr}}_fastp.json", srr=SRR_IDS),




    #------------------------------------------------------------------------------------------------------------------------
	# Cutdapat_rule-------(older implementation  and currently not in use)---------------------------------------------------
	#*expand(f"{RESULTS_DIR}/trimmed/{{srr}}_1_trimmed.fastq.gz", srr=SRR_IDS),
        
	
	# trimmed qc-------------(older implementation and currently not in use) -------------------------------------------------
	#*expand(f"{RESULTS_DIR}/trimmed_qc/{{srr}}_1_trimmed_fastqc.html", srr=SRR_IDS),
    #-------------------------------------------------------------------------------------------------------------------------




	# multi_qc-------------(consolidated qc report)----------------------------------------------------
	f"{RESULTS_DIR}/multiqc_report/multiqc_report.html",
	



	# Bowtie2 index files------------(genome indexing)------------------------------------------
        *expand("{data}/bt2_index/dm6.{ext}", data=DATA_DIR, ext=["1.bt2","2.bt2","3.bt2","4.bt2","rev.1.bt2","rev.2.bt2"]),



	#Bowtie2_alingment-----------(for both ChIP and ATAC data)----------------------------------------------
	*expand(f"{RESULTS_DIR}/alignments/{{srr}}_sorted.bam", srr=SRR_IDS),
        *expand(f"{RESULTS_DIR}/alignments/{{srr}}_sorted.bam.bai", srr=SRR_IDS),


	#filtering_sorted_allingments ----------------(for both ChIP and ATAC data)-----------------------------
	*expand(f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam",srr=SRR_IDS),
        *expand(f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam.bai",srr=SRR_IDS),	

	
	#chipseq_coverage----------------------------------------------------------
	*expand(f"{RESULTS_DIR}/coverage/{{srr}}_chip_cov.bedGraph",srr=CHIP_IDS),


	#peak Genration-------------(only for chip)------------------------------------------------
	*expand(f"{RESULTS_DIR}/macs2_peaks/{{srr}}_peaks.narrowPeak", srr=CHIP_IDS),
	*expand(f"{RESULTS_DIR}/macs2_peaks/{{srr}}_summits.bed", srr=CHIP_IDS),
	

	# ATAC cut-site BEDs--------------------------------------------------------
        *expand(f"{RESULTS_DIR}/ATAC_cutsite/{{srr}}_cutsite.bed", srr=ATAC_IDS),
	
	# ATAC fragments BEDs-------------------------------------------------------
        *expand(f"{RESULTS_DIR}/ATAC_fragments/{{srr}}_fragments.bed", srr=ATAC_IDS),

	# ATAC_coverage--------------------------------------------------------------
	*expand(f"{RESULTS_DIR}/ATAC_coverage/{{srr}}_atac_cov.bedGraph", srr=ATAC_IDS),

	# ATAC-seq peaks-------------------------------------------------------------
        *expand(f"{RESULTS_DIR}/ATAC_peaks/{{srr}}_peaks.narrowPeak", srr=ATAC_IDS),
        *expand(f"{RESULTS_DIR}/ATAC_peaks/{{srr}}_summits.bed", srr=ATAC_IDS),

	# **Active regions from ATAC peaks**------------------------------------------
        f"{RESULTS_DIR}/ATAC_active_region/ATAC_active_regions.bed",	

	#ATAC Background region--------------------------------------------------------
	f"{RESULTS_DIR}/ATAC_background_region/ATAC_background_regions.bed",
	
	# Normalized ATAC tracks-------------------------------------------------------
        *expand(f"{RESULTS_DIR}/ATAC_normalized/{{srr}}_atac_norm.bedGraph", srr=ATAC_IDS),	

	#Normalized Chipseq--------------------------------------------------------------- 
	*expand(f"{RESULTS_DIR}/chip_normalized/{{srr}}_control_fixed.bdg", srr=CHIP_IDS),
	*expand(f"{RESULTS_DIR}/chip_normalized/{{srr}}_ip_fixed.bdg", srr=CHIP_IDS),
	*expand(f"{RESULTS_DIR}/chip_normalized/{{srr}}_FE.bdg", srr=CHIP_IDS),
	*expand(f"{RESULTS_DIR}/chip_normalized/{{srr}}_scale.txt", srr=CHIP_IDS),
	*expand(f"{RESULTS_DIR}/chip_normalized/{{srr}}_corrected_ready.bdg", srr=CHIP_IDS),


	#-----------------------------------sliding windows----(Dead)----------------------------------
	# *expand(f"{DATA_DIR}/dm6_8kb_windows_sliding.bed"),


 
	#chip binning-------------------(Utilization recommeded only for classification models)---------------------------------------------------
	*expand(f"{RESULTS_DIR}/binned_chip/{{srr}}_binned.bdg", srr = CHIP_IDS),

	#ATAC binning------------------------(Utilization recommeded only for classification models)-----------------------------------------------
	*expand(f"{RESULTS_DIR}/binned_ATAC/{{srr}}_binned.bdg", srr = ATAC_IDS),
	



    #chip binning and ATAC binning -------------------(Utilization recommeded only for 1bp  models)--------------------------------------------  
	*expand(f"{RESULTS_DIR}/perbase_chip/{{srr}}_perbase.npy" , srr = CHIP_IDS),
	*expand(f"{RESULTS_DIR}/perbase_atac/{{srr}}_perbase.npy", srr = ATAC_IDS ), 



	#One hot encoding--------------------------------------------------------------------

	f"{RESULTS_DIR}/dna_onehot.npy",
	

# rule for downloading Chipseq and ATAC reads ---------------------------------------------------------------------
# SRR IDS are directly accessd from the srr_list.csv---------------------------------------------------------------
# Requires Internet access !!!!!!!! -------------------------------------------------------------------------------

rule download_srr:
    output:
        f"{DATA_DIR}/{{srr}}_1.fastq.gz"
    #localrule: True
    resources:
        downloads=1
    shell:
        """
        set -euo pipefail

        # module load sra-toolkit

        echo "Checking SRA file for {wildcards.srr}..."

        if [ -f "{DATA_DIR}/{wildcards.srr}/{wildcards.srr}.sra" ]; then
            echo "Existing SRA file found for {wildcards.srr}. Skipping prefetch."
        else
            echo "SRA file not found. Starting prefetch..."
            prefetch {wildcards.srr} --output-directory {DATA_DIR}
        fi

        echo "Converting to FASTQ..."
        fastq-dump --split-files --gzip --outdir {DATA_DIR} {DATA_DIR}/{wildcards.srr}/{wildcards.srr}.sra
        """




# Downloading whole genome (can be changed if needed)-----------------------------------------------------------------------
# Use muste be sure to change the Variable names in whole Pipeline if the Genome is changed 
# direct download link is present in the config file ----------------------------------------------------------------------

rule download_dm6:
    output:
        fasta = f"{DATA_DIR}/dm6.fa",
        gtf   = f"{DATA_DIR}/dm6.gtf"
    params:
        genome_url = GENOME_FASTA_URL,
        gtf_url = ANNOTATION_GTF_URL
    message:
        "Downloading dm6 genome and annotation"
    shell:
        """
        set -euo pipefail
        mkdir -p {DATA_DIR}
        
        wget -O {DATA_DIR}/dm6.fa.gz {params.genome_url} 
        gunzip -c {DATA_DIR}/dm6.fa.gz > {output.fasta}
        rm -f {DATA_DIR}/dm6.fa.gz

        # genomic annotation 
        wget -O {DATA_DIR}/dm6.gtf.gz {params.gtf_url}
        gunzip -c {DATA_DIR}/dm6.gtf.gz > {output.gtf}
        rm -f {DATA_DIR}/dm6.gtf.gz
        """





# Quality check of chipseq and ATAC reads with basif fastqc --------------------------------------------------

rule fastqc:
    input:
        r1 = f"{DATA_DIR}/{{srr}}_1.fastq.gz"
        # kept only r1 to prevent errors in case of pe fastqc

    message: "------------ Intiating Quality checks ----------------"
    output:
        r1_html = f"{RESULTS_DIR}/qc/{{srr}}_1_fastqc.html"
        # incase of pairend it will be processed but dosent expect it as an output.
    threads: THREADS
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/qc

        # Always run FastQC on R1
        fastqc {input.r1} -o {RESULTS_DIR}/qc

        # If R2 exists, run FastQC on it (but Snakemake dosent expect this output)
        R2_FILE="{DATA_DIR}/{wildcards.srr}_2.fastq.gz"
        if [ -f "$R2_FILE" ]; then
            fastqc "$R2_FILE" -o {RESULTS_DIR}/qc
        fi
        """





# 2kb windows for gernome dm6 according to the reference(Chrom2Vec paper)--------------------------------------------- 
# The name 8kb is of old refrence and previous test experiments ------------------------------------------------------
# (!!! The name 8kb does not represent the binning size !!!)------------------------------------------------------------

rule make_8kb_windows:
    input:
        fai=f"{DATA_DIR}/dm6.fa.fai"
    output:
        bed=f"{DATA_DIR}/dm6_8kb_windows.bed"
    message:
        "Generating complete fixed genomic windows"
    shell:
        r"""
        awk 'BEGIN{{OFS="\t"}}
            $1 ~ /^(chr2L|chr2R|chr3L|chr3R|chr4|chrX|chrY)$/ {{
                for(i=0; i+{WINDOW_SIZE}<=$2; i+={WINDOW_SIZE})
                    print $1, i, i+{WINDOW_SIZE}
            }}' {input.fai} > {output.bed}
        """





# ---------------------------Dm6 indexing (primary requirment -> samtools)-------------------------------------------------
# samtools faidx manual (https://www.htslib.org/doc/samtools-faidx.html?utm_source)
# =============================================================================
# SAMTOOLS FASTA INDEX
# Creates dm6.fa.fai from the dm6 reference genome. The index stores chromosome
# names, lengths and file positions, allowing fast sequence retrieval. It is
# also used by downstream rules for generating windows and chromosome sizes.
# This index is not used for sequencing-read alignment.
# =============================================================================

rule faidx:
    input:
        fasta = f"{DATA_DIR}/dm6.fa"
    output:
        fai = f"{DATA_DIR}/dm6.fa.fai"
    message:
        "---------------Generating  genomic indexing from {input.fasta}------------------"
    shell:
        f"""
        samtools faidx {input.fasta}           #< dm6 indexing (fa file for downstreamig)
        """






# Trimming and quality check done in one rule using fastp------------------------------------------------------------
# with refrence to (Chrom2Vec paper) saved in diffrent dir-----------------------------------------------------------
# fastp comand line mnual -> (https://github.com/OpenGene/fastp#all-options)

rule fastp_trim:
    input:
        r1 = f"{DATA_DIR}/{{srr}}_1.fastq.gz"
    output:
        r1_trim = f"{RESULTS_DIR}/trimmed_fastp/{{srr}}_1_trimmed.fastq.gz",
        html = f"{RESULTS_DIR}/fastp_reports/{{srr}}_fastp.html",
        json = f"{RESULTS_DIR}/fastp_reports/{{srr}}_fastp.json"
    threads: THREADS
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/trimmed_fastp {RESULTS_DIR}/fastp_reports
        
        R2_IN="{DATA_DIR}/{wildcards.srr}_2.fastq.gz"
        R2_OUT="{RESULTS_DIR}/trimmed_fastp/{wildcards.srr}_2_trimmed.fastq.gz"

        # Checking pair and single end for seprate adpter trimming 

        if [ -f "$R2_IN" ]; then
            # PAIRED-END TRIM
            fastp -i {input.r1} -I "$R2_IN" \
                  -o {output.r1_trim} -O "$R2_OUT" \
                  -h {output.html} -j {output.json} \
                  -w {threads} --length_required 25
        else
            # SINGLE-END TRIM
            fastp -i {input.r1} \
                  -o {output.r1_trim} \
                  -h {output.html} -j {output.json} \
                  -w {threads} --length_required 25
        fi
        """




# Gatthered report for all the trimmed and untrimmed reads (for model training)----------------------------------------
# done according to the reffrence paper(Chrom2Vec paper)--------------------------------------------------------------
# multiqc user guide , reffrence taken from (https://docs.seqera.io/multiqc/getting_started/running_multiqc?utm_source)---

rule multiqc:
    input:
        expand(f"{RESULTS_DIR}/qc/{{srr}}_1_fastqc.html", srr=SRR_IDS),
        expand(f"{RESULTS_DIR}/trimmed_fastp/{{srr}}_1_trimmed.fastq.gz", srr=SRR_IDS),
        expand(f"{RESULTS_DIR}/fastp_reports/{{srr}}_fastp.html", srr=SRR_IDS)

    message: "------------Gathering QC report-(merging in progress)---------------"
    output:
        html = f"{RESULTS_DIR}/multiqc_report/multiqc_report.html"           #-> (html is recommended for viewing)
    threads: THREADS
    shell:
        """
        mkdir -p {RESULTS_DIR}/multiqc_report
        multiqc {RESULTS_DIR} -o {RESULTS_DIR}/multiqc_report -n multiqc_report.html --force          #<- combined qc report (both html and json)
        """






# Indexing the whole genome for easy alignment ---------------------------------------------------------------
# alignment of many queries to the genome is much faster, avoid scanning the genome repeatedly.-----------------
# =============================================================================
# BOWTIE 2 GENOME INDEX
# Builds the six Bowtie 2 index files required to align ChIP-seq, input-control
# and ATAC-seq reads to the dm6 reference genome. Downstream alignment rules
# access the complete index using the common prefix: data/bt2_index/dm6.
# This index is separate from the SAMtools .fai coordinate index above.
# reffered Manual for bowtie 2 ->(https://bowtie-bio.sourceforge.net/bowtie2/manual.shtml?)
# =============================================================================


rule bowtie2_index:
    input:
        genome = f"{DATA_DIR}/dm6.fa"
    message: "------------bowtie2-(indexing in progress)---------------"
    output:
        index_files = expand("{data}/bt2_index/dm6.{ext}", 
                             data=DATA_DIR, 
                             ext=["1.bt2", "2.bt2", "3.bt2", "4.bt2", "rev.1.bt2", "rev.2.bt2"])
    threads: THREADS
    shell:
        r"""
        mkdir -p {DATA_DIR}/bt2_index
        bowtie2-build {input.genome} {DATA_DIR}/bt2_index/dm6
        """






# Allinging trimmed chipseq and ATAC reads with whloe genome--------------------------------------------------
# sorting is also done inside the shell(this rule handels sorting aswell as allingment)-----------------------
# reffered Manual for bowtie 2 ->(https://bowtie-bio.sourceforge.net/bowtie2/manual.shtml?)

rule bowtie2_align:
    input:
        # Index files generated by bowtie2_index
        index = expand("{data}/bt2_index/dm6.{ext}",
                       data=DATA_DIR,
                       ext=["1.bt2","2.bt2","3.bt2","4.bt2","rev.1.bt2","rev.2.bt2"]),
        r1 = f"{RESULTS_DIR}/trimmed_fastp/{{srr}}_1_trimmed.fastq.gz"

    message: "------------bowtie2_alignment-(in progress)---------------"
    output:
        # BAM outputs as a list----------------------------
        bam_files = [
            f"{RESULTS_DIR}/alignments/{{srr}}_sorted.bam",
            f"{RESULTS_DIR}/alignments/{{srr}}_sorted.bam.bai"
        ]
    threads: THREADS
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/alignments

        # Optional paired-end
        R2_FILE="{RESULTS_DIR}/trimmed_fastp/{wildcards.srr}_2_trimmed.fastq.gz"


        #  (delet) old files if they exist  [Very importnat for correct data outflow in case of pipeline faliure midway]
        [ -f {output.bam_files[0]} ] && rm {output.bam_files[0]}
        [ -f {output.bam_files[1]} ] && rm {output.bam_files[1]}

        # condition for paired and single end
        if [ -f "$R2_FILE" ]; then
            bowtie2 -x {DATA_DIR}/bt2_index/dm6 \
                -1 {input.r1} -2 "$R2_FILE" -p {threads} \
                | samtools view -bS - \
                | samtools sort -o {output.bam_files[0]}
        else
            bowtie2 -x {DATA_DIR}/bt2_index/dm6 \
                -U {input.r1} -p {threads} \
                | samtools view -bS - \
                | samtools sort -o {output.bam_files[0]}
        fi

        # Index the sorted BAM
        samtools index {output.bam_files[0]}
        """






#filtering the alignments for pcr contamination , poor quality and ChrM--------------------------------
#refference manual -> (https://samtools.github.io/hts-specs/SAMv1.pdf) ------------ (page 7)---------

rule filter_alignments:
    input:
        bam = f"{RESULTS_DIR}/alignments/{{srr}}_sorted.bam"
    message: "------------Filtering allignment-(in progress)---------------"
    output:
        bam = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam",
        bai = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam.bai"
    threads: THREADS
    shell:
        """
        mkdir -p {RESULTS_DIR}/alignments
        
        # Keep high quality, remove duplicates/unmapped, and drop chrM
        samtools view -h -q 30 -F 1804 {input.bam} | \
        grep -v "chrM" | \
        samtools sort -@ {threads} -o {output.bam}

        samtools index {output.bam}
        """    






# The ChIP-seq coverage step computes a genome-wide signal track from aligned ChIP-seq reads-----------
# Generate a cleaned ChIP-seq coverage track from the filtered BAM file.
# The rule calculates genome-wide read coverage in bedGraph format, removes
# regions listed in the dm6 blacklist, validates the chromosome names and
# interval coordinates, and deletes temporary files to free disk space.
# Manual used fo this rule -> (https://bedtools.readthedocs.io/en/latest/content/tools/genomecov.html)

rule chipseq_coverage:
    input:
        bam = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam",
	blacklist = f"{DATA_DIR}/dm6_blacklist.bed",
	genome = f"{DATA_DIR}/dm6.fa.fai" 

    message: "------------Chip coverage-(in progress)---------------"
    output:
        bedgraph = f"{RESULTS_DIR}/coverage/{{srr}}_chip_cov.bedGraph"
 
    threads: THREADS
    shell:
        """
        mkdir -p {RESULTS_DIR}/coverage

        #  Generate coverage
        bedtools genomecov -ibam {input.bam} \
                           -g {input.genome} \
                           -bg \
        > {output.bedgraph}.tmp

        # Subtracting blacklist(from the input file)
        bedtools subtract -a {output.bedgraph}.tmp -b {input.blacklist} \
        > {output.bedgraph}.tmp2

        # Keep valid chromosomes and 4 columns
        awk 'BEGIN{{OFS="\t"}} $1 ~ /^chr/ && NF==4 && $3>$2 {{print $1,$2,$3,$4}}' {output.bedgraph}.tmp2 \
        > {output.bedgraph}

        # free up the memory 
        rm -f {output.bedgraph}.tmp {output.bedgraph}.tmp2
        """



#------------------Chipseq Preprocessing---(MACS2 signaling peaks)  --------------------------------------------------

# identifies regions of the genome with enriched signal (done according to the reffrence paper)
# peaks.narrowPeak -> Contains statistically significant enriched genomic regions.
# summits.bed      -> Contains the position of maximum enrichment within each peak.
# control_lambda.bdg -> Contains the local background signal estimated by MACS2.
# treat_pileup.bdg -> Contains the ChIP-seq treatment pileup signal.
# reffered manual -> (https://macs3-project.github.io/MACS/docs/callpeak)



rule macs2_peak:
    input:
        chip = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam",
        bai  = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam.bai",
        input_bam = lambda wildcards: (
            f"{RESULTS_DIR}/alignments/{chip_to_input[wildcards.srr]}_filtered.bam"
            if wildcards.srr in chip_to_input else []
        )

    message: "------------macs2_peak-(in progress)---------------"
    output:
        peaks   = f"{RESULTS_DIR}/macs2_peaks/{{srr}}_peaks.narrowPeak",
        summits = f"{RESULTS_DIR}/macs2_peaks/{{srr}}_summits.bed",
        control_lambda = f"{RESULTS_DIR}/macs2_peaks/{{srr}}_control_lambda.bdg",
        treat_pileup   = f"{RESULTS_DIR}/macs2_peaks/{{srr}}_treat_pileup.bdg"
    shell:
        r"""
        set -euo pipefail

        mkdir -p {RESULTS_DIR}/macs2_peaks

        # Detect whether the ChIP BAM is paired-end.
        CHIP_COUNT=$(samtools view -c -f 1 {input.chip})

        # Check whether an input-control BAM exists.
        if [ -n "{input.input_bam}" ] && [ -f "{input.input_bam}" ]; then

            # Detect whether the input-control BAM is paired-end.
            INPUT_COUNT=$(samtools view -c -f 1 {input.input_bam})

            # Use BAMPE only if both ChIP and input are paired-end.
            if [ "$CHIP_COUNT" -gt 0 ] && [ "$INPUT_COUNT" -gt 0 ]; then
                FORMAT="BAMPE"
            else
                FORMAT="BAM"
            fi

            echo "MACS2 format for {wildcards.srr}: $FORMAT"

            macs2 callpeak \
                -t {input.chip} \
                -c {input.input_bam} \
                -f "$FORMAT" \
                -g dm \
                -n {wildcards.srr} \
                --outdir {RESULTS_DIR}/macs2_peaks \
                --keep-dup 1 \
                --q 1e-10 \
                --bdg \
                --call-summits

        else
            # No input control was provided.
            if [ "$CHIP_COUNT" -gt 0 ]; then
                FORMAT="BAMPE"
            else
                FORMAT="BAM"
            fi

            echo "No input control found for {wildcards.srr}."
            echo "MACS2 format for {wildcards.srr}: $FORMAT"

            macs2 callpeak \
                -t {input.chip} \
                -f "$FORMAT" \
                -g dm \
                -n {wildcards.srr} \
                --outdir {RESULTS_DIR}/macs2_peaks \
                --keep-dup 1 \
                --q 1e-10 \
                --bdg \
                --call-summits
        fi
        """



#-----------------------------------------ATAC Preprocessing -------------------------------------------------
# specific to ATAC-seq and it processes the aligned reads to identify precise Tn5 transposase insertion sites.
# Generate Tn5 cut sites as one-base intervals in three-column BED format:
# chromosome, start and end. For paired-end reads, two cut sites are generated
# from each fragment by shifting the fragment start by +4 bp and its end by
# -5 bp. For single-end reads, the +4 bp or -5 bp shift is applied according
# to the alignment strand. The resulting cut sites are coordinate-sorted.
# manual reffred (https://bedtools.readthedocs.io/en/latest/content/tools/bamtobed.html?)
# (https://www.htslib.org/doc/samtools-sort.html?)

rule atac_cutsite:
    input:
        bam = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam"
    message: "------------Tn5 cutsite-(in progress)---------------"
    output:
        cutsite = f"{RESULTS_DIR}/ATAC_cutsite/{{srr}}_cutsite.bed"
    threads: 4
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/ATAC_cutsite

        PAIRED=$(samtools view -c -f 1 {input.bam})

        if [ $PAIRED -gt 0 ]; then
            echo "Paired-end detected: generating Tn5 cut sites"

            samtools sort -n -@ {threads} -m 2G {input.bam} \
            | bedtools bamtobed -i stdin -bedpe 2>/dev/null \
            | awk 'BEGIN{{OFS="\t"}} $1==$4 {{
                print $1, $2+4, $2+5;
                print $4, $6-5, $6-4;
            }}' \
            | awk '$2 >= 0 && $3 > $2' \
            | sort -k1,1 -k2,2n \
            > {output.cutsite}

        else
            echo "Single-end detected: generating Tn5 cut sites"

            bedtools bamtobed -i {input.bam} 2>/dev/null \
            | awk 'BEGIN{{OFS="\t"}} {{
                if($6=="+") print $1, $2+4, $2+5;
                else if($6=="-") print $1, $3-5, $3-4;
            }}' \
            | awk '$2 >= 0 && $3 > $2' \
            | sort -k1,1 -k2,2n \
            > {output.cutsite}
        fi
        """




# required cut site fragments for model input and downstreamprocessing-----------------------
# NOT USED IN DOWNSTREAM PROCESS (Directly accessed in Deepleaning model as an extra context foro ATAC data)
# Produces a four-column BED-like file: (chromosome    start    end    fragment_length)
# changable according to the requirement (Independent rule !! no breakage !!)
# (!!!!!! current code cant handle single end !!!!!!)
# reffered links 
# (https://bedtools.readthedocs.io/en/latest/content/tools/bamtobed.html?)
# (https://www.htslib.org/doc/samtools-sort.html?)
# (https://bedtools.readthedocs.io/en/latest/content/general-usage.html)

rule atac_fragments:
    input:
        bam=f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam"
    message: "------------atac_fragments-(in progress)---------------"
    output:
        fragments=f"{RESULTS_DIR}/ATAC_fragments/{{srr}}_fragments.bed"
    threads: 8
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/ATAC_fragments

        PAIRED=$(samtools view -c -f 1 {input.bam})

        if [ $PAIRED -gt 0 ]; then
            echo "Paired-end detected: generating true fragment intervals"

            samtools sort -n -@ {threads} -m 2G {input.bam} \
            | bedtools bamtobed -i stdin -bedpe 2>/dev/null \
            | awk 'BEGIN{{OFS="\t"}} $1==$4 {{
                start=$2+4;
                end=$6-5;
                size=end-start;
                if(size>0) print $1,start,end,size;
            }}' \
            | sort -k1,1 -k2,2n \
            > {output.fragments}

        else
            echo "ERROR: Single-end ATAC-seq cannot be used for reliable fragment-size classification." >&2
            exit 1
        fi
        """






# it scans the genome and counts how many Tn5 cuts fall into each position or bin.-----------------------------
# This rule converts the 1-bp Tn5 insertion sites into a genome-wide coverage track.


rule atac_coverage:
    input:
        cutsite = f"{RESULTS_DIR}/ATAC_cutsite/{{srr}}_cutsite.bed",
        genome = f"{DATA_DIR}/dm6.fa.fai",
        blacklist = f"{DATA_DIR}/dm6_blacklist.bed" # Add this
    message: "------------atac_coverage-(in progress)---------------"
    output:
        bedgraph = f"{RESULTS_DIR}/ATAC_coverage/{{srr}}_atac_cov.bedGraph"
    shell:
        """
        mkdir -p {RESULTS_DIR}/ATAC_coverage

        #  Filter negative coordinates and pipe directly into genomecov
        # Using 'stdin' tells bedtools to listen to the awk output
        awk '$2 >= 0 && $3 >= 0' {input.cutsite} | \
        bedtools genomecov -i stdin -g {input.genome} -bg > {output.bedgraph}.tmp

        #  Subtract blacklist
        bedtools intersect -v -a {output.bedgraph}.tmp -b {input.blacklist} > {output.bedgraph}

        # Clean storage
        rm {output.bedgraph}.tmp
        """





# Identifies regions of the genome with enriched Tn5 insertions,----------------------------------------------------------
# This rule identifies accessible chromatin regions by calling genomic areas with unusually high ATAC sequencing read or fragment accumulation.
# peaks are also filtered using blacklisted region of dm6 genome 
# user manual -> (https://macs3-project.github.io/MACS/docs/callpeak.html?)

rule atac_macs2_peak:
    input:
        bam = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam",
        bai = f"{RESULTS_DIR}/alignments/{{srr}}_filtered.bam.bai",
        blacklist = f"{DATA_DIR}/dm6_blacklist.bed"
    message: "------------atac_peak calling-(in progress)---------------"
    output:
        peaks = f"{RESULTS_DIR}/ATAC_peaks/{{srr}}_peaks.narrowPeak",
        summits = f"{RESULTS_DIR}/ATAC_peaks/{{srr}}_summits.bed"
        # summits are not used in downstreaming process for regression model but is kept for classification model outputs.
    threads: THREADS
    params:
        genome = "dm" 
    shell:
        r"""
        mkdir -p {RESULTS_DIR}/ATAC_peaks

        #  Determine if BAM file is paired-end or not
        IS_PAIRED=$(samtools view -c -f 1 {input.bam} | head -n 1)

        if [ "$IS_PAIRED" -gt 0 ]; then
            echo "--- {wildcards.srr}: PAIRED-END detected. Calling peaks with BAMPE ---"
            macs2 callpeak -t {input.bam} \
                -f BAMPE \
                -g {params.genome} \
                -n {wildcards.srr} \
                --outdir {RESULTS_DIR}/ATAC_peaks \
                --keep-dup all \
                --bdg --call-summits \
                --q 0.05
        else
            echo "--- {wildcards.srr}: SINGLE-END detected. Calling peaks with shift/ext ---"
            macs2 callpeak -t {input.bam} \
                -f BAM \
                -g {params.genome} \
                -n {wildcards.srr} \
                --outdir {RESULTS_DIR}/ATAC_peaks \
                --nomodel --shift -100 --extsize 200 \
                --keep-dup all \
                --bdg --call-summits \
                --q 0.05
        fi

        #  Blacklist Filtering
        # We filter the narrowPeak file to remove technical artifacts
        if [ -f "{output.peaks}" ]; then
            echo "--- Filtering {wildcards.srr} peaks against dm6 blacklist ---"
            bedtools intersect -v -a {output.peaks} -b {input.blacklist} > {output.peaks}.tmp
            mv {output.peaks}.tmp {output.peaks}
        fi
        """



# defining the union of accessible chromatin regions across all ATAC-seq samples----------
# This rule combines accessible chromatin peaks from all ATAC sequencing samples into one nonredundant set of active genomic regions.
# The output is one BED file containing the union of accessible regions detected across all ATAC samples.

rule atac_active_peaks:
    input:
        expand(f"{RESULTS_DIR}/ATAC_peaks/{{srr}}_peaks.narrowPeak", srr=ATAC_IDS)
    output:
        active_regions = f"{RESULTS_DIR}/ATAC_active_region/ATAC_active_regions.bed"
    shell:
        """
        mkdir -p {RESULTS_DIR}/ATAC_active_region
        cat {input} \
            | bedtools sort -g data/dm6.fa.fai \
            | bedtools merge \
            > {output.active_regions}
        """





# downloading blacklist dm6(used to exclude problematic genomic regions from your analysis.) ------------- 

rule download_dm6_blacklist:
    output:
        f"{DATA_DIR}/dm6_blacklist.bed"
    shell:
        """
        mkdir -p {DATA_DIR}

        # Download the dm6 blacklist file
        wget -O {DATA_DIR}/dm6_blacklist.bed.gz \
            https://github.com/Boyle-Lab/Blacklist/raw/main/lists/dm6-blacklist.v2.bed.gz

        # Unzip it
        gunzip -f {DATA_DIR}/dm6_blacklist.bed.gz
        """





# regions outside ATAC peaks.-------------------------------------------------
# Genrating background region ===============================================
# Background = Genome − ATAC active regions − Blacklisted regions
# manula used -> (https://bedtools.readthedocs.io/en/latest/content/tools/complement.html?)

rule atac_background:
    input:
        active = f"{RESULTS_DIR}/ATAC_active_region/ATAC_active_regions.bed",
        genome = f"{DATA_DIR}/dm6.fa.fai",
        blacklist = f"{DATA_DIR}/dm6_blacklist.bed"
    output:
        background = f"{RESULTS_DIR}/ATAC_background_region/ATAC_background_regions.bed"
    shell:
        """
        mkdir -p {RESULTS_DIR}/ATAC_background_region
        # Complement genome - active regions
        bedtools complement -i {input.active} -g {input.genome} \
        | bedtools subtract -a - -b {input.blacklist} \
        > {output.background}
        """





# ATAC normalization===================================================================================================================
# This rule converts raw ATAC cut site coverage into a background corrected, percentile scaled, and log transformed accessibility signal.
# 1. Calculate the mean ATAC signal in background regions.
# 2. Subtract the background mean and clip negative values to zero.
# 3. Extract corrected signals within MACS2 active regions. -> 
# 4. Calculate a scaling factor from the 99th to 99.9th percentile range.
# 5. Scale the corrected signal and apply log2(1 + signal).
# 6. Save the normalized BedGraph for downstream tensor construction.
# ===================================================================================================================================
# References:
# https://bedtools.readthedocs.io/en/latest/content/tools/intersect.html
# https://www.gnu.org/software/gawk/manual/gawk.html
# https://www.gnu.org/software/coreutils/manual/html_node/sort-invocation.html
# https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html
#======================================================================================================================================= 

rule atac_normalization:
    input:
        coverage = f"{RESULTS_DIR}/ATAC_coverage/{{srr}}_atac_cov.bedGraph",
        active_regions = f"{RESULTS_DIR}/ATAC_active_region/ATAC_active_regions.bed",
        background_regions = f"{RESULTS_DIR}/ATAC_background_region/ATAC_background_regions.bed"
    output:
        norm = f"{RESULTS_DIR}/ATAC_normalized/{{srr}}_atac_norm.bedGraph"
    shell:
        r"""
        set -euo pipefail
        # to makes the shell stop when an important error occurs

        mkdir -p {RESULTS_DIR}/ATAC_normalized

        #  Estimate background mean from background regions
        BG_MEAN=$(bedtools intersect -a {input.coverage} -b {input.background_regions} -wa \
            | awk '{{sum+=$4; count+=1}} END{{if(count>0) print sum/count; else print 0}}')

        echo "ATAC background mean: $BG_MEAN"

        #  Background subtract and clip negative values to zero
        awk -v bg=$BG_MEAN 'BEGIN{{OFS="\t"}} {{
            val=$4-bg;
            if(val<0) val=0;
            print $1,$2,$3,val;
        }}' {input.coverage} > {RESULTS_DIR}/tmp_{wildcards.srr}_atac_bg_corrected.bdg

        # Extract corrected ATAC signal values inside active regions
        bedtools intersect -a {RESULTS_DIR}/tmp_{wildcards.srr}_atac_bg_corrected.bdg \
            -b {input.active_regions} -wa \
            | awk '{{print $4}}' \
            | sort -n > {RESULTS_DIR}/tmp_{wildcards.srr}_atac_active_values.txt

        #  99th and 99.9th percentile values
        N=$(wc -l < {RESULTS_DIR}/tmp_{wildcards.srr}_atac_active_values.txt)

        P99=$(awk -v n=$N 'NR==int(n*0.99) {{print $1}}' \
            {RESULTS_DIR}/tmp_{wildcards.srr}_atac_active_values.txt)

        P999=$(awk -v n=$N 'NR==int(n*0.999) {{print $1}}' \
            {RESULTS_DIR}/tmp_{wildcards.srr}_atac_active_values.txt)

        # mean signal between P99 and P99.9 as scale factor
        SCALE=$(awk -v p99=$P99 -v p999=$P999 '
            {{
                if($1>=p99 && $1<=p999) {{
                    sum+=$1;
                    count+=1;
                }}
            }}
            END {{
                if(count>0 && sum>0) print sum/count;
                else print 1;
            }}' {RESULTS_DIR}/tmp_{wildcards.srr}_atac_active_values.txt)

        echo "ATAC percentile scale P99-P99.9 mean: $SCALE"

        # Scale corrected ATAC signal
        awk -v sc=$SCALE 'BEGIN{{OFS="\t"}} {{
            val=$4/sc;
            if(val<0) val=0;
            logval=log(val+1)/log(2);
            print $1,$2,$3,logval;
        }}' {RESULTS_DIR}/tmp_{wildcards.srr}_atac_bg_corrected.bdg > {output.norm}

        rm -f {RESULTS_DIR}/tmp_{wildcards.srr}_atac_bg_corrected.bdg
        rm -f {RESULTS_DIR}/tmp_{wildcards.srr}_atac_active_values.txt
        """
# ========================================================================================================================================================

# Chipseq Normalization ------------------------
# check if the bdgfiles only has 4 coloumns 
# Implemented during the process due to unknown bedgraph failures 
# rule checks coloumn orders and positioning 
#step 1 of normalizing 

rule fix_ip_bedgraph:
    input:
        bdg = f"{RESULTS_DIR}/macs2_peaks/{{srr}}_treat_pileup.bdg"

    output:
        fixed = f"{RESULTS_DIR}/chip_normalized/{{srr}}_ip_fixed.bdg"
    shell:
        """
	# maintanng tab spacing and removing invalid chrm names for preprocessing ....

        awk 'BEGIN{{OFS="\\t"}} $1 ~ /^chr[0-9XYMLR]+$/ {{print $1, int($2), int($3), $4}}' {input.bdg} > {output.fixed}
        """

rule fix_control_bedgraph:
    input:
        bdg = f"{RESULTS_DIR}/macs2_peaks/{{srr}}_control_lambda.bdg"
    output:
        fixed = f"{RESULTS_DIR}/chip_normalized/{{srr}}_control_fixed.bdg"
    shell:
        """
	# maintanng tab spacing and removing invalid chrm names for preprocessing ....(for control....)
        awk 'BEGIN{{OFS="\\t"}} $1 ~ /^chr[0-9XYMLR]+$/ {{print $1, int($2), int($3), $4}}' {input.bdg} > {output.fixed}
        """


rule get_fold_enrichment:
    input:
        t = f"{RESULTS_DIR}/chip_normalized/{{srr}}_ip_fixed.bdg",
        c = f"{RESULTS_DIR}/chip_normalized/{{srr}}_control_fixed.bdg"
    output:
        fe = f"{RESULTS_DIR}/chip_normalized/{{srr}}_FE.bdg"
    shell:
        "macs2 bdgcmp -t {input.t} -c {input.c} -m FE -o {output.fe}"

#=======================================================================================================================================



# ------------------------Step 2 Mean---------------------------------------------------
# kept just for user info (Experimental rule, Not used in downstream processing)
# (!!!! Should not be considered for actual tensor outputs , Implemented only for intial stages of information during devlopemnt  !!!!)

rule compute_mean:
    input:
        ip_fixed = f"{RESULTS_DIR}/chip_normalized/{{srr}}_ip_fixed.bdg",
        control_fixed = f"{RESULTS_DIR}/chip_normalized/{{srr}}_control_fixed.bdg"
    output:
        scale = f"{RESULTS_DIR}/chip_normalized/{{srr}}_scale.txt"
    shell:
        """
        # Compute mean signal of IP over control regions
        BG_IP=$(bedtools map -a {input.control_fixed} -b {input.ip_fixed} -c 4 -o mean \
                 | awk '{{sum += $4; count += 1}} END {{if(count>0) print sum/count; else print 0}}')

        # Compute mean signal of control
        BG_CTRL=$(awk '{{sum += $4; count += 1}} END {{if(count>0) print sum/count; else print 0}}' {input.control_fixed})

        # Compute scale factor
        SCALE=$(awk -v bg_ip=$BG_IP -v bg_ctrl=$BG_CTRL 'BEGIN{{if(bg_ctrl>0) print bg_ip/bg_ctrl; else print 1}}')

        # Save scale factor
        echo $SCALE > {output.scale}
        """



# ===================== Follow similar set of rules as ATAC percentile normalization ============================
# ===================================================================================================================================
# References:
# https://bedtools.readthedocs.io/en/latest/content/tools/intersect.html
# https://www.gnu.org/software/gawk/manual/gawk.html
# https://www.gnu.org/software/coreutils/manual/html_node/sort-invocation.html
# https://www.gnu.org/software/bash/manual/html_node/The-Set-Builtin.html
#======================================================================================================================================= 

rule chipseq_percentile_normalization:
    input:
        ip_fixed = f"{RESULTS_DIR}/chip_normalized/{{srr}}_FE.bdg",
        active = f"{RESULTS_DIR}/ATAC_active_region/ATAC_active_regions.bed",
        background = f"{RESULTS_DIR}/ATAC_background_region/ATAC_background_regions.bed"
    output:
        corrected = f"{RESULTS_DIR}/chip_normalized/{{srr}}_corrected_ready.bdg",
        bg_sd = f"{RESULTS_DIR}/chip_normalized/{{srr}}_background_sd.txt"
    shell:
        """
        # Intersecting IP with active regions and get 4th column
        bedtools intersect -a {input.ip_fixed} -b {input.active} -wa \
            | awk '{{print $4}}' | sort -n > {RESULTS_DIR}/tmp_{wildcards.srr}_active.txt

        # Computeing 99th and 99.9th percentile values
        N=$(wc -l < {RESULTS_DIR}/tmp_{wildcards.srr}_active.txt)
        P99=$(awk -v n=$N 'NR==int(n*0.99) {{print $1}}' {RESULTS_DIR}/tmp_{wildcards.srr}_active.txt)
        P999=$(awk -v n=$N 'NR==int(n*0.999) {{print $1}}' {RESULTS_DIR}/tmp_{wildcards.srr}_active.txt)

        # Mean of 99–99.9 percentile
        SCALE=$(awk -v p99=$P99 -v p999=$P999 '{{if($1>=p99 && $1<=p999) sum+=$1; count+=($1>=p99 && $1<=p999)}} END{{if(count>0) print sum/count; else print 1}}' {RESULTS_DIR}/tmp_{wildcards.srr}_active.txt)

        echo "Percentile scale: $SCALE"

        # Correct IP track using percentile scaling and log transformation
        awk -v scale=$SCALE 'BEGIN{{OFS="\\t"}} {{
            val=$4/scale;
            if(val<0) val=0;
            val=log(1+val);
            $4=val;
            print
        }}' {input.ip_fixed} > {output.corrected}

        # Compute SD over background regions
        bedtools intersect -a {input.ip_fixed} -b {input.background} -wa \
            | awk '{{sum+=$4; sumsq+=$4*$4; count+=1}} END {{if(count>1) print sqrt((sumsq-(sum*sum)/count)/(count-1)); else print 0}}' > {output.bg_sd}

        rm -f {RESULTS_DIR}/tmp_{wildcards.srr}_active.txt
        """








# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++(Dead rule)++++++++++++++++++++++++++++++++++++++++
#  Binning for model input ----------------------------------



rule generate_sliding_windows:
    input:
        genome = f"{DATA_DIR}/dm6.fa.fai"      # already exists from faidx
    output:
        bed = f"{DATA_DIR}/dm6_8kb_windows_sliding.bed"
    message:
        "---------------Generating 8192 bp sliding windows (stride 5120 bp)------------------"
    shell:
        r"""
        mkdir -p {DATA_DIR}
        bedtools makewindows \
            -g {input.genome} \
            -w {WINDOW_SIZE} \
            -s {WINDOW_STRIDE} \
        > {output.bed}
        """
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++(Dead rule)++++++++++++++++++++++++++++++++++++++++


# The bellow binning can only be used for calassification modles and not 1bp models ===================== 
# Procedure avg out evrysignal for 1 bin (1 signal per bin ) ==================================

rule CHIP_binning_fixed:
    input:
        # CHANGE: Point to your normalized bedGraph, not the raw peaks
        peaks = f"{RESULTS_DIR}/chip_normalized/{{srr}}_corrected_ready.bdg",
        windows = f"{DATA_DIR}/dm6_8kb_windows.bed" 
    output:
        binned = RESULTS_DIR + "/binned_chip/{srr}_binned.bdg"
    threads: 2
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.binned})
        
        # Use bedtools map on the normalized signal (column 4)
        # We use -o mean to get a smooth average signal for the 8kb window
        bedtools map \
            -a <(sort -k1,1 -k2,2n {input.windows}) \
            -b <(sort -k1,1 -k2,2n {input.peaks}) \
            -c 4 -o mean -null 0 \
            > {output.binned}
        """

rule ATAC_binning_fixed:
    input:
        peaks = f"{RESULTS_DIR}/ATAC_normalized/{{srr}}_atac_norm.bedGraph",
        windows = f"{DATA_DIR}/dm6_8kb_windows.bed"
    output:
        binned = RESULTS_DIR + "/binned_ATAC/{srr}_binned.bdg"
    threads: 2
    shell:
        r"""
        set -euo pipefail
        mkdir -p $(dirname {output.binned})

        # sorting and mapping (no temp file genration to avoid crash!!)
        bedtools map \
            -a <(sort -k1,1 -k2,2n {input.windows}) \
            -b <(sort -k1,1 -k2,2n {input.peaks}) \
            -c 4 -o mean -null 0 \
            > {output.binned}
        """




# Perbase inputs are required for 1bp CNN models 
# ====================== ChIP Per-base Binning ======================
rule bin_chip_signal:
    input:
        bdg="results/chip_normalized/{sample}_corrected_ready.bdg",
        bed="data/dm6_8kb_windows.bed"
    output:
        npy="results/perbase_chip/{sample}_perbase.npy"
    log:
        "logs/bin_chip/{sample}.log"
    params:
        win = config["windows"]["size"],
        strd = config["windows"]["stride"]
    shell:
        """
        mkdir -p results/perbase_chip
        python script/bin_signal_to_perbase.py {input.bdg} {input.bed} {output.npy} {params.win} {params.strd} > {log} 2>&1
        """


# ====================== ATAC Per-base Binning ======================
rule bin_atac_signal:
    input:
        signal  = f"{RESULTS_DIR}/ATAC_normalized/{{srr}}_atac_norm.bedGraph",
        windows = f"{DATA_DIR}/dm6_8kb_windows.bed"
    output:
        npy = f"{RESULTS_DIR}/perbase_atac/{{srr}}_perbase.npy"
    threads: 4
    params:
        win = config["windows"]["size"],
        strd = config["windows"]["stride"]
    shell:
        """
        mkdir -p {RESULTS_DIR}/perbase_atac
        python script/bin_signal_to_perbase.py {input.signal} {input.windows} {output.npy} {params.win} {params.strd}
        """








# DNA one hot encoding using same bins ans strides from the use defined config ===================================


rule one_hot_encode_dna:
    input:
        fasta="data/dm6.fa",
        windows="data/dm6_8kb_windows.bed"
    output:
        npy=f"{RESULTS_DIR}/dna_onehot.npy"
    params:
        bin_size=config["windows"]["size"],
        expected_windows=lambda wildcards: sum(1 for line in open("data/dm6_8kb_windows.bed"))
    shell:
        """
        mkdir -p processed

        # Extract sequences directly using the BED windows
        bedtools getfasta \
            -fi {input.fasta} \
            -bed {input.windows} \
            -fo processed/dm6_8kb.fa \
            -name

        # Run one-hot encode script
        python script/onehot_encode_fasta.py \
            processed/dm6_8kb.fa \
            {output.npy} \
            {params.bin_size} \
            {params.expected_windows}
        """

