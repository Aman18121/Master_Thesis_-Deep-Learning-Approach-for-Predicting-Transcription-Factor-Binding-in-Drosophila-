'''This python script belongs to 
author = [AMAN YADAV]
script 1/6

as part of Masterthesis [A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila]
'''


import yaml
import os
import numpy as np
import pandas as pd


''' The following script is responsible for loading ternsor ready data from the OS, for easy execution .
keeping config , snakefile and python scripts in the same directory is reccomended ''' 

class GenomicDataLoader:
    def __init__(self, config_path="config_file_ml.yaml"):

        #=======================config file loader============================
        # Load the configuration file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        
        #===================Extract paths and parameters from the YAML========
        paths = config.get("paths", {})
        params = config.get("params", {})


        # =========Directory paths (!!! no chnages required !!!) used config_file_ml.yaml============
        self.chip_dir = paths.get("chip_dir")
        self.atac_dir = paths.get("atac_dir")
        self.dna_dir = paths.get("dna_path")
        self.bed_path = paths.get("chrms_path")
        self.fragment_path = paths.get("TN5_cutsite")

        # ==================================== assinged names ========================================
        # ========= if not assigned (default names will be used E.g T1,T2.......)=====================
        
        self.chip_name_map = params.get("chip_names", {})
        self.atac_name_map = params.get("atac_names", {})
        
        # ======================Link directly to the YAML exclusion list=============================
        #==========================(Handles non satisfactory chip track exclussion)=================
        #==========Is recommended to be used after Visualization (if not sure)============================
        self.exclude_srr = params.get("exclude_srr", [])
        
        # =============== Initializing variables============================================
        self.chip_tracks = []
        self.atac_tracks = [] 
        self.dna_tracks = [] 
        self.fragment_tracks = [] 
        self.concatenated_data = None
        self.names = None
        self.chip_names = []
        self.atac_names = []          
        self.window_chromosomes = None

        #======================Intitialzation info ===========================================
        print(f"Genomic loader initialized using {config_path}")
        print(f"Excluding samples: {self.exclude_srr}")

    
    #====================== checking chip exclusions =================================
    #=================== will be used in class functions =============================
    def is_excluded(self, filename):
        """Checks if the filename contains any of the excluded SRRs defined in YAML."""
        return any(srr in filename for srr in self.exclude_srr)



    #========================================(loads the chip data from the chip directory)===========
    
    
    def read_chip_data(self):
        # ================== execution gate keeper)============================
        if not os.path.exists(self.chip_dir):
            raise FileNotFoundError(f" ChIP directory not found at {self.chip_dir}")
  
        chip_files = sorted(os.listdir(self.chip_dir))                                      # <- listing files inside chip directory 
        for file_name in chip_files:
            
            #=================(!!!!chip-seq data must be in mpy format!!!)====================
            #============(!!! Handles skiping the excluded chip "npy" file !!!)==============
            if file_name.endswith(".npy"):                                      #<- files are already converted to numpy array inside the preprocessing pieline..
                if self.is_excluded(file_name): # Updated check
                    print(f"Skipping excluded file: {file_name}")
                    continue
                #=================(path formation for loading data)===========================
                file_path = os.path.join(self.chip_dir, file_name)                  #<- joining making variables from config_file_ml.yaml into a path 
                data = np.load(file_path, mmap_mode='r')                       #<- laoding data 
                self.chip_tracks.append(data)                                   #<-storage in chip_tracks list

                srr_id = os.path.splitext(file_name)[0].replace("_perbase", "")                                    #<- real SRR ID for plotting and track labels
                display_name = self.chip_name_map.get(srr_id, srr_id)
                self.chip_names.append(display_name)

                print(f"Loaded ChIP Track: {file_name} as {display_name}")
            
        return self.chip_tracks




   #========================================(loads the atac data from the atac directory)===========     

    def read_atac_data(self):
        # ================== execution gate keeper)============================
        if not os.path.exists(self.atac_dir):
            raise FileNotFoundError(f" ChIP directory not found at {self.atac_dir}")
            
        atac_files = sorted(os.listdir(self.atac_dir))           #<- can handle only one ATAC at a time (this Ml pipline is not desined to process multiple / diffrent ATAC files) keep only one file in atac dir ....

        #=================(!!!! atacp-seq data must be in mpy format!!!)====================
            #============(!!! Handles skiping the excluded atac "mpy" file !!!)==============
        for file_name in atac_files: 
            if file_name.endswith(".npy"):
                #============(!!! Handles skiping the excluded atac "npy" file !!!)==============
                if self.is_excluded(file_name):
                    print(f"Skipping excluded file: {file_name}")
                    continue

                #=================(path formation for loading data)===========================
                file_path = os.path.join(self.atac_dir, file_name)          
                data = np.load(file_path, mmap_mode='r')             
                self.atac_tracks.append(data)

                srr_id = os.path.splitext(file_name)[0].replace("_perbase", "")                   #<- real SRR ID for plotting and track labels
                display_name = self.atac_name_map.get(srr_id, srr_id)
                self.atac_names.append(display_name)

                print(f"Loaded ATAC Track: {file_name} as {display_name}")

        return self.atac_tracks                                     #<- one file only in the list (ATAC)                              




    #========================================(loads the one hot encode  data from the atac directory)===========          
    
    def read_one_hot_encode(self):
        # ================== execution gate keeper)============================
        if not os.path.exists(self.atac_dir):
            raise FileNotFoundError(f" ChIP directory not found at {self.atac_dir}")
        

        
        dna_files = sorted(os.listdir(self.dna_dir))              #<- can handle only one DNA at a time (this Ml pipline is not desined to process multiple / diffrent DNA files) keep only one file in dna dir ....
        for file_name in dna_files:
            if file_name.endswith(".npy"):
                 
                if self.is_excluded(file_name):
                    print(f"Skipping excluded file: {file_name}")
                    continue

                #=================(path formation for loading data)===========================
                file_path = os.path.join(self.dna_dir, file_name)
                data = np.load(file_path, mmap_mode='r')
                
                # ============Squeezes channel for keeping it simmilar to chip and ATAC data (dont need 4 dim shape ) ============= 
                if data.ndim == 4 and data.shape[1] == 1:
                    data = np.squeeze(data, axis=1)
                    
                self.dna_tracks.append(data)              #<- sored in dna_tracks list 
                print(f"Loaded One-Hot DNA Matrix: {file_name}")
        return self.dna_tracks                                   #<- one file only in the list (DNA)



    #============================(Chip,ATAC and dna shape info )========================
    #=======should be executed just after loading all the data to ===========================

    def get_shape(self, chip=True, atac=True, one_hot=True):
        check_list = [
            (chip, "ChIP-seq Targets", self.chip_tracks),
            (atac, "ATAC-seq Signal", self.atac_tracks),
            (one_hot, "DNA Sequencings", self.dna_tracks)
        ]
        for should_check, label, data in check_list:             #shape inspector 
            if should_check:
                if isinstance(data, list) and len(data) > 0:
                    print(f"  {label} Shape: {data[0].shape} (List depth: {len(data)})")
                elif not isinstance(data, list) and data is not None:
                    print(f" {label} Shape: {data.shape}")
                else:
                    print(f" {label} data layer hasn't been instantiated yet!")



    
    #==========================combine all chip and atac data in one file ==============================
    #==================================(used for visualization)======================
    def chip_concantor(self):
        #========If either list is empty, the function stops.=======================
        if not self.chip_tracks or not self.atac_tracks:                                   #<- gate keeper of this function, make sure to run read_chip_data() and read_atac_data() before merging tensors                
            raise ValueError(" cant execute chip_concantor --------- Run read_chip_data() and read_atac_data() before merging tensors.")

            
        #===========removes dimensions of size one.===================================================
        # ===============number of genomic windows and the length of each window.==================
        atac_array = np.squeeze(self.atac_tracks[0])
        total_windows = atac_array.shape[0]   
        window_size = atac_array.shape[1]                                               #<- checking the shape of atac (expanding dims if its 2d), need one more dim for number of chips ....


        
        if atac_array.ndim == 2:                                                         
            atac_array = np.expand_dims(atac_array, axis=-1)

        # Removes unnecessary dimensions from each ChIP array and stacks all ChIP targets
        chip_stacked = np.stack([np.squeeze(arr) for arr in self.chip_tracks], axis=-1)     #<- stacking chips togather 
        
        total_required_bases = total_windows * window_size
        chip_trimmed = chip_stacked[:total_required_bases, :]
        chip_reshaped = chip_trimmed.reshape((total_windows, window_size, len(self.chip_tracks)))

        self.concatenated_data = np.concatenate([atac_array, chip_reshaped], axis=-1)

        #=============Check if name is given for all files ==============================
        atac_name = self.atac_names[0] if len(self.atac_names) > 0 else "ATAC"
        self.names = [atac_name] + self.chip_names

        print("Track names assigned:", self.names)

        return self.concatenated_data




    # ====================================add chrmosomes to the bins==========================
    # ====================impotant for chromosmes based sepration ===========================
    def add_chrmos(self): 
        if not self.bed_path or not os.path.exists(self.bed_path):
            raise FileNotFoundError(f"Provided path canvas invalid: {self.bed_path}")

        if self.concatenated_data is None:                            #<- gate kepper if concatenated data dosnt exit 
            self.concatenated_data = self.chip_concantor()            #<- automatic execution of chip_concantor()  

        bed_dataframe = pd.read_csv(self.bed_path, sep='\t', header=None, usecols=[0], names=['chrom'])
        bed_chromosomes = bed_dataframe['chrom'].values                           #<- adding chrmosome here..

        num_signal_windows = self.concatenated_data.shape[0]           #<-cheking number of signals 
        num_bed_windows = len(bed_chromosomes)                              #<-cheking number of chromosoems 

        # ----------- major problem if triggred ----------------------------------------------------
        # If this problem exits check the preprocessing pipline 
        if num_bed_windows != num_signal_windows:
            print(f" Alignment Warning: Array windows ({num_signal_windows}) differ from BED markers ({num_bed_windows})")
            min_windows = min(num_signal_windows, num_bed_windows)
            self.concatenated_data = self.concatenated_data[:min_windows, :, :]
            bed_chromosomes = bed_chromosomes[:min_windows]
        #---------------------------------------------------------------------------------------------
        self.window_chromosomes = bed_chromosomes
        return self.concatenated_data



    
    #=============================chrm info =============================================================
    def list_chromosomes(self):                                                 #<- menthod for listing chromosomes 
        if self.window_chromosomes is None:
            raise ValueError("Coordinate alignment arrays empty. Run add_chrmos() first.") 
        return list(np.unique(self.window_chromosomes))



                                                                         
    #=================loading atac fragments genrated during preprocessing==========================================================
    def read_all_atac_fragments_3d(self, blacklisted_chroms=None, window_size=None):                   #<- function for fragment frouping (ATAC context)
        #=====Handles black listed chrmsoems (if any chrms defined )====================================================================
        if blacklisted_chroms is None:
            blacklisted_chroms = []

        # ===========================fetch bed file from the fragment directory=========== 
        if os.path.isdir(self.fragment_path):
            bed_files = [f for f in os.listdir(self.fragment_path) if f.endswith('.bed')]
            if not bed_files:
                raise FileNotFoundError(f"No .bed files found in: {self.fragment_path}")
            self.fragment_path = os.path.join(self.fragment_path, bed_files[0])
            print(f"Auto-detected fragment file: {self.fragment_path}")
            
        # 
        if self.window_chromosomes is None or self.concatenated_data is None:                          #<- checks if chromose alreadyadded in the maine data
            raise ValueError("Coordinates are missing or unaligned. Run add_chrmos() first.")

        if window_size is None:                                                                        #<- Ensures dynamic configuration changes pass directly to the sub-fragment loops
            window_size = self.concatenated_data.shape[1]

        chrms_in_data = self.list_chromosomes()                                                        #<- check the list of chromosomes
        valid_chrms = [chrom for chrom in chrms_in_data if chrom not in blacklisted_chroms]

        bed_data = pd.read_csv(self.bed_path, sep='\t', header=None, usecols=[0, 1], names=['chrom', 'start'])
        total_windows = self.concatenated_data.shape[0]
        bed_data = bed_data.iloc[:total_windows]
        bed_starts = bed_data['start'].values                                                          #<- pulled out once as NumPy array, avoids slow .iloc lookups inside the loop

        atac_frags = np.zeros((total_windows, window_size, 3), dtype=np.float32)

        print(f"Streaming TN5 fragment details from: {self.fragment_path}")

        chrms_chunks = {chrom: [] for chrom in valid_chrms}
        for chunk in pd.read_csv(self.fragment_path, sep='\t', header=None,                            #<- reads file in chuncks so the process dont crash in between ..
                                  names=['chrom', 'start', 'end'],
                                  usecols=[0, 1, 2], chunksize=5000000, engine='c'):
            for chrom, sub_chunk in chunk.groupby('chrom'):                                            #<- groups chunk by chromosome in a single pass instead of re-filtering per chromosome
                if chrom in chrms_chunks:
                    chrms_chunks[chrom].append(sub_chunk)

        # Running totals across all chromosomes
        total_short = 0
        total_mono = 0
        total_di = 0

        for chrom in valid_chrms:
            if not chrms_chunks[chrom]:
                continue

            combined_df = pd.concat(chrms_chunks[chrom], ignore_index=True)
            combined_df['length'] = combined_df['end'] - combined_df['start']                          #<- chroms length for latter classifications

            # Sub-divide fragments by base-pair length cuts                                            #<- Slicing Data into 3 Biological Channels and tagging them on the basis of lengths
            short_mask = combined_df['length'] < 100
            mono_mask = (combined_df['length'] >= 180) & (combined_df['length'] <= 247)
            di_mask = (combined_df['length'] >= 315) & (combined_df['length'] <= 473)

            short_starts = np.sort(combined_df[short_mask]['start'].values)                            #<- sorted once per chromosome so window lookups can binary-search instead of full scan
            mono_starts = np.sort(combined_df[mono_mask]['start'].values)
            di_starts = np.sort(combined_df[di_mask]['start'].values)

            # Accumulate counts across chromosomes (instead of per-chromosome reporting)
            total_short += len(short_starts)
            total_mono += len(mono_starts)
            total_di += len(di_starts)

            matching_idx = np.where(self.window_chromosomes == chrom)[0]

                                                                                                          #<- Populating matrix values using direct indexing offsets
            for idx in matching_idx:
                w_start = bed_starts[idx]
                w_end = w_start + window_size

                # Channel 1: Short Fragments
                lo = np.searchsorted(short_starts, w_start, side='left')                                #<- binary search window bounds instead of boolean masking entire array
                hi = np.searchsorted(short_starts, w_end, side='left')
                short_frags = short_starts[lo:hi]
                if short_frags.size > 0:
                    offsets = (short_frags - w_start).astype(np.int64)
                    np.add.at(atac_frags[idx, :, 0], offsets, 1)                                        #<- vectorized increment, replaces per-fragment Python loop

                # Channel 2: Mononucleosomal Fragments
                lo = np.searchsorted(mono_starts, w_start, side='left')
                hi = np.searchsorted(mono_starts, w_end, side='left')
                mono_frags = mono_starts[lo:hi]
                if mono_frags.size > 0:
                    offsets = (mono_frags - w_start).astype(np.int64)
                    np.add.at(atac_frags[idx, :, 1], offsets, 1)

                # Channel 3: Dinucleosomal Fragments
                lo = np.searchsorted(di_starts, w_start, side='left')
                hi = np.searchsorted(di_starts, w_end, side='left')
                di_frags = di_starts[lo:hi]
                if di_frags.size > 0:
                    offsets = (di_frags - w_start).astype(np.int64)
                    np.add.at(atac_frags[idx, :, 2], offsets, 1)

        # Combined total across all chromosomes
        print(f"Total fragments across all chromosomes — Short: {total_short}, Mono: {total_mono}, "
              f"Di: {total_di}, Grand total: {total_short + total_mono + total_di}")

        print(f"Aligned fragment matrix setup complete. Matrix layout: {atac_frags.shape}")
        return atac_frags



    #=========(!!!! Imprtnat !!!!_ Verification of the chip and atac data)============================
    def verify_data_normalization(self):                                                 # <- normalization  check (very crucial _)
        if self.concatenated_data is None:                                                      #<- if files are correctly loaded or not 
            raise ValueError("concatenated_data is uninitialized. Cannot check normalization.")
            
        print("\nChecking Genomic Signal Scaling Status across All Data Channels...")
        num_channels = self.concatenated_data.shape[2]                                            #<- check num of channels in concanted file
        
                                                                                                   # Pull layout track labels if provided, otherwise create default indexes
        labels = self.names if (self.names and len(self.names) == num_channels) else [f"Channel_{i}" for i in range(num_channels)]
        
        passed_norm_test = True
        print("-" * 85),
        print(f"{'Track Name':<20} | {'Global Min':<12} | {'Global Max':<12} | {'Mean':<12} | {'Status':<15}")                              #<- tabular format
        print("-" * 85)
        
        for idx in range(num_channels):
            channel_slice = self.concatenated_data[:, :, idx]                                            #<-  checking chip signals min, max and mean
            c_min = float(np.min(channel_slice))
            c_max = float(np.max(channel_slice))
            c_mean = float(np.mean(channel_slice))
            
            if c_max > 50.0:                                                                                                                # < -  defiining Status 
                status = "UNNORMALIZED"
                passed_norm_test = False
            elif c_max <= 0.0:
                status = "SILENT/EMPTY"
            else:
                status = "NORMALIZED"
                
            print(f"{labels[idx]:<20} | {c_min:<12.4f} | {c_max:<12.4f} | {c_mean:<12.4f} | {status:<15}")
            
        print("-" * 85)
        if passed_norm_test:
            print("Success: All channels fall within stable boundaries for deep network gradients.")
        else:
            print("Warning: One or more tracks contain high raw values. Consider min-max or log2 scaling before training.")



  
