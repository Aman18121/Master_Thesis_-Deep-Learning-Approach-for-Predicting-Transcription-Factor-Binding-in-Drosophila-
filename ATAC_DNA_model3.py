'''This python script belongs to 
author = [AMAN YADAV]
script 5/6
as part of Masterthesis [A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila]
'''


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import random
from pathlib import Path
from scipy.stats import pearsonr, wasserstein_distance
import heapq
import yaml
from scipy.spatial.distance import jensenshannon



#(random seed initialization)
def set_global_seed(seed):
    random.seed(seed)                        
    np.random.seed(seed)                      
    tf.keras.utils.set_random_seed(seed)      
    try:
        tf.config.experimental.enable_op_determinism()  
    except Exception:
        pass                                  
    print(f"Global random seed: {seed}")



# multimodal prediction model data requirements
# requires the following data from the genomic data loader:
# combined_data (ATAC followed by all ChIP tracks)
# Dna_one_hot (one-hot encoded DNA sequence)
# normalized ATAC signal
# chromosome labels for leakage-safe splitting
# track names for ChIP identification
# TN5 fragment channels supplied during Train_Test_Split()

class MultimodalGenomicModel:
    def __init__(self, combined_data=None, window_chroms=None, Dna_one_hot=None, X_atac_list=None, track_names=None):       
        # Store foundational data from data loader
        self.combined_data = combined_data                                            
        self.window_chroms = np.array(window_chroms) if window_chroms is not None else None  
        self.Dna_one_hot = Dna_one_hot                                                
        self.X_atac_list = X_atac_list                                                
        self.names = track_names                                                     
        
        
        self.X_chrom_train = None 
        self.X_dna_train =None 
        self.Y_multi_train = None                                
        self.X_Atac_val = None 
        self.X_dna_val = None 
        self.Y_multi_val = None
        self.X_Atac_test = None 
        self.X_dna_test = None
        self.Y_multi_test = None 

        
        self.ATAC_input_shape = None
        self.dna_input_shape = None

        #meant to be initialized during model training=
        self.model = None
        self.history = None
        self.seed = None
        
        print("MultimodalGenomicModel is active.")

    #fetch data from the genomic data loader
    @classmethod
    def from_loader(cls, loader_instance):                              

        try:
            if not hasattr(loader_instance, 'dna_tracks') or loader_instance.dna_tracks is None:
                raise ValueError("one-hot DNA missing, run read_one_hot_encode() first.")
            if not hasattr(loader_instance, 'atac_tracks') or loader_instance.atac_tracks is None:
                raise ValueError("ATAC arrays missing, run read_atac_data() first.")
            if not hasattr(loader_instance, 'concatenated_data') or loader_instance.concatenated_data is None:
                raise ValueError("ChIP matrices not merged. Run add_chrmos() first.")
                
        except AttributeError as error:
            raise ValueError(f"Required loader attribute is missing: {error}") from error
        
        
        return cls(
            combined_data=loader_instance.concatenated_data,
            window_chroms=getattr(loader_instance, 'window_chromosomes', None),    
            Dna_one_hot=getattr(loader_instance, 'dna_tracks', None),                  
            X_atac_list=getattr(loader_instance, 'atac_tracks', None),             
            track_names=getattr(loader_instance, 'names', None)
        )




        
    # train/validation/test split using held-out chromosomes 
    # default path to config file if it exists in the same directory
        
    def Train_Test_Split(self, config_path="config_file_ml.yaml", TN5_frags=None):

        
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        
        # chrms 
        train_chrms = yaml_config["params"]["training_chrms"]
        test_chr = yaml_config["params"]["test_chrms"]
        val_chr = yaml_config["params"]["validation_chrms"]

        
        # for suporting both str and lists input
        train_chrms = [train_chrms] if isinstance(train_chrms, str) else list(train_chrms)
        test_chrms = [test_chr] if isinstance(test_chr, str) else list(test_chr)
        val_chrms = [val_chr] if isinstance(val_chr, str) else list(val_chr)


        
        #Critical error checking to verify all inputs are loaded
        try:
            if self.Dna_one_hot is None or len(self.Dna_one_hot) == 0:
                raise ValueError("Empty one-hot encoded array.")
                
            if self.X_atac_list is None or len(self.X_atac_list) == 0:
                raise ValueError("Empty ATAC track array list.")
                
            if self.window_chroms is None:
                raise ValueError("Chromosome alignment map is missing! Run loader.add_chroms() first.")
                
            if TN5_frags is None:
                raise ValueError("TN5 fragment features are missing. Provide TN5_frags.")
                
        except (AttributeError, TypeError) as error:
            raise ValueError(f"Unable to validate multimodal input data: {error}") from error

            

        #seprate extraction of  DNA, ATAC and TN5 feature matrices
        # DNA and chromatin tensors are the model features.
        
        X_Dna = self.Dna_one_hot[0] if isinstance(self.Dna_one_hot, (list, tuple)) else self.Dna_one_hot   
        X_Atac = self.X_atac_list[0] if isinstance(self.X_atac_list, (list, tuple)) else self.X_atac_list
        X_Dna = np.asarray(X_Dna)
        TN5_frags = np.asarray(TN5_frags)
        
        
        X_Atac = np.squeeze(X_Atac)     #<- flatten ATAC layout: (binsize, 1)
        
        if X_Atac.ndim == 2:
            print(f" ATAC is two dimensioned ")
            X_Atac = np.expand_dims(X_Atac, axis=-1) 
            
        if X_Atac.ndim != 3 or X_Dna.ndim != 3:
            raise ValueError(f"ATAC and DNA inputs must be three-dimensional: ATAC={X_Atac.shape}, DNA={X_Dna.shape}.")
            
        # standardize TN5 fragments to the same three-dimensional layout
        # TN5    
        
        if TN5_frags.ndim == 2:
            print("TN5_frags missing channel axis; expanding to 3D.")  
            TN5_frags = np.expand_dims(TN5_frags, axis=-1)
            
        if TN5_frags.ndim != 3:
            raise ValueError(f"TN5 fragment input must be three-dimensional: {TN5_frags.shape}.")
            
        if X_Atac.shape[:2] != TN5_frags.shape[:2]:
            raise ValueError(f"ATAC and TN5 fragment shapes do not align: {X_Atac.shape} vs {TN5_frags.shape}.")
        
        #combine ATAC and TN5 into the chromatin input
        #atac + tn5                                                                                                        
        X_TN5_frags = np.concatenate([X_Atac, TN5_frags], axis=-1) 
        
        # ChIP profiles are the prediction outputs.
        Y_chip = self.combined_data[:, :, 1:]                                
        
        self.dna_input_shape = X_Dna.shape[1:]                          
        self.ATAC_input_shape = X_TN5_frags.shape[1:]                 


        
        #confirm that every modality describes the same genomic windows
        chrms_array = self.window_chroms.astype(str)
        if X_TN5_frags.shape[0] != X_Dna.shape[0] or X_Dna.shape[0] != Y_chip.shape[0] or Y_chip.shape[0] != chrms_array.shape[0]:
            raise ValueError(
                f"Window counts do not align: chromatin={X_TN5_frags.shape[0]}, DNA={X_Dna.shape[0]}, targets={Y_chip.shape[0]}, chromosomes={chrms_array.shape[0]}."
            )
        # np.isin() checks which window chromosome belongs to each configured split.
        test_mask = np.isin(chrms_array, test_chrms)                      
        val_mask = np.isin(chrms_array, val_chrms)                        
        train_mask = np.isin(chrms_array, train_chrms)                  

        # prevent chromosome overlap and genomic data leakage
        try:
            if (
                np.any(train_mask & val_mask)
                or np.any(train_mask & test_mask)
                or np.any(val_mask & test_mask)
            ):
                raise ValueError(
                    "Training, validation, and test chromosome groups must not overlap."
                )

            if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
                raise ValueError(
                    "At least one chromosome split contains no windows. "
                    "Check the YAML chromosome configuration."
                )
        except (TypeError, ValueError) as error:
            raise ValueError(f"chromosome split is false / faulty: {error}") from error
            

        # DNA and chromatin are input features; ChIP profiles are prediction targets.
        self.X_chrom_train = X_TN5_frags[train_mask]
        self.X_dna_train   = X_Dna[train_mask]
        self.X_Atac_val   = X_TN5_frags[val_mask]
        self.X_dna_val     = X_Dna[val_mask]
        self.X_Atac_test  = X_TN5_frags[test_mask]
        self.X_dna_test    = X_Dna[test_mask]
        self.Y_multi_train = Y_chip[train_mask]
        self.Y_multi_val   = Y_chip[val_mask]
        self.Y_multi_test  = Y_chip[test_mask]

   
        
        print("\nMultimodal chromosome split done ")
        print(f"Train | chromatin {self.X_chrom_train.shape}, DNA {self.X_dna_train.shape}, ChIP {self.Y_multi_train.shape}")
        print(f"Validation: DNA={self.X_dna_val.shape}; chromatin={self.X_Atac_val.shape}; targets={self.Y_multi_val.shape} ({val_chrms})")
        print(f"Test chromosomes {test_chrms} -> chromatin {self.X_Atac_test.shape} / DNA {self.X_dna_test.shape} / labels {self.Y_multi_test.shape}")

        

        return ((self.X_chrom_train, self.X_dna_train), self.Y_multi_train), \
               ((self.X_Atac_val, self.X_dna_val), self.Y_multi_val), \
               ((self.X_Atac_test, self.X_dna_test), self.Y_multi_test)



 
    # multimodal sequential 1D-CNN model
    def ATAC_DNA_model_sequential(self, config_path="config_file_ml.yaml", seed=None):  
            
        try:
            if self.X_chrom_train is None or self.Y_multi_train is None:
                raise ValueError("splits missing, run Train_Test_Split first.")

            if self.X_Atac_val is None or self.Y_multi_val is None:
                raise ValueError("validation splits missing, run Train_Test_Split first.")

            if self.X_Atac_val.shape[0] == 0:
                raise ValueError("validation set is empty. Fix chromosome split before training.")

            if self.X_Atac_test is not None and self.X_Atac_test.shape[0] == 0:
                raise ValueError("test set is empty. Fix chromosome split before training.")

        except AttributeError as error:
            raise ValueError(f" invalid  multimodal splits: {error}") from error
            

        #infer flexible input and output dimensions 
        total_targets = self.Y_multi_train.shape[-1]                
        ATAC_frags_shape = self.ATAC_input_shape                           
        dna_shape   = self.dna_input_shape                            

        print(f" model seq started training : chromatin {ATAC_frags_shape}, DNA {dna_shape}, targets {total_targets}")

       
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        #at least one configured seed is required for reproducibility
        configured_seeds = yaml_config["params"]["model_seeds"]["multimodal"]
        if seed is None:
            print("using feist seed in the list")
            seed = configured_seeds[0]
        if seed not in configured_seeds:
            raise ValueError(f"Seed {seed} is not listed in config or model sequntial .")
            
        set_global_seed(seed)
        self.seed = seed

        # settings from config file
        epochs      = yaml_config["params"]["epochs_multimodal"]
        batch_size  = yaml_config["params"]["batch_size_multimodal"]
        multimodal_config = yaml_config["params"]["multimodal_callbacks"]
        earlystop_patience = multimodal_config["multimodal_early_stop_patience"]
        multi_modal_patience = multimodal_config["multimodal_reduce_lr_patience"]
        multi_lr_factor = multimodal_config["multimodal_lr_decay_factor"]
        multi_loss = yaml_config["params"]["multimodal_loss_tuning"]
        multi_peak_threshold = tf.constant(multi_loss["multimodal_peak_threshold"], dtype=tf.float32)
        multi_peak_multiplier = tf.constant(multi_loss["multimodal_peak_weight_multiplier"], dtype=tf.float32)
        multi_background_penalty = tf.constant(multi_loss["multimodal_background_penalty"], dtype=tf.float32)

     
        # Flexible input shapes are inferred from the loaded tensors.
        ATAC_frag_inputs = tf.keras.Input(shape=ATAC_frags_shape, name="chromatin_input")  #<- graph entry for ATAC plus TN5 features
        dna_inputs  = tf.keras.Input(shape=dna_shape,   name="dna_input")         

        # handles only chromatin features 
        chrms_features = tf.keras.layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(ATAC_frag_inputs)
        chrms_features = tf.keras.layers.BatchNormalization()(chrms_features)

        
        #  handles only dna feature (done according chromnitron paper ( reffer main thesis file )                                                                                                   
        dna_feat = tf.keras.layers.Conv1D(32, kernel_size=15, padding='same', activation='relu')(dna_inputs)
        dna_feat = tf.keras.layers.BatchNormalization()(dna_feat)

        
        # joins chromatin and DNA features along the channel dimension
        merged = tf.keras.layers.Concatenate(axis=-1)([chrms_features, dna_feat])

        # for gated model testing 
        '''
        # gate: learns importance of each channel at each base-pair
        # gate       = tf.keras.layers.Conv1D(merged.shape[-1], kernel_size=1, activation='sigmoid')(merged)
        # gated_feat = tf.keras.layers.Multiply()([merged, gate])

                                                                                                             
        # x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(gated_feat, gated_feat)
        # x = tf.keras.layers.LayerNormalization()(x)
        '''
        
       # taking merged input
        x = tf.keras.layers.Conv1D(64, kernel_size=15, padding='same')(merged)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Receptive-field scaling 
        # dilated convolution blocks d = 2
        x = tf.keras.layers.Conv1D(128, kernel_size=15, padding='same', dilation_rate=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Broad genomic-context  d =4
        x = tf.keras.layers.Conv1D(128, kernel_size=21, padding='same', dilation_rate=4)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Distant flanking context d = 8 
        x = tf.keras.layers.Conv1D(128, kernel_size=21, padding='same', dilation_rate=8)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        
        # stable per-base representation
        x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    
        outputs = tf.keras.layers.Conv1D(filters=total_targets, kernel_size=1, activation='linear', padding='same', dtype='float32')(x)  

        #both modalities are required for every prediction (dna & atac)
        model = tf.keras.Model(inputs=[ATAC_frag_inputs, dna_inputs], outputs=outputs)  

        # weighted MSE for continuous peak-profile prediction 
        def weighted_mse(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)                         
            y_pred = tf.cast(y_pred, tf.float32)
            peak_weights = tf.where(y_true >= multi_peak_threshold, multi_peak_multiplier, 1.0) # peak elevation 
            background_suppression = tf.where((y_true == 0.0) & (y_pred > 0.01), multi_background_penalty, 1.0)  # bg suppresion 
            return tf.reduce_mean(tf.square(y_true - y_pred) * peak_weights * background_suppression)
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss=weighted_mse,
            metrics=['mae']
        )

        model.summary()

        # settings are in config file 
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=earlystop_patience,
            restore_best_weights=True,                                
            verbose=1
        )

        checkpoint_path = Path("checkpoints") / "multimodal" / f"seed_{seed}" / "best_atac_dna_model.keras"  # independent best checkpoint for each seed
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        # when validation improvement stops
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=multi_lr_factor,                                        
            patience=multi_modal_patience,                                        
            min_lr=1e-6,
            verbose=1
        )

        print("Training simplified Multi-CNN Pipeline on Slurm...")

    
       
        train_ds = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "chromatin_input": self.X_chrom_train,
                    "dna_input": self.X_dna_train
                },
                self.Y_multi_train
            )
        )

        train_ds = (
            train_ds
            .shuffle(2048, seed=seed, reshuffle_each_iteration=True)     
            .batch(batch_size)
            .prefetch(1)                                                 
        )

        val_ds = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "chromatin_input": self.X_Atac_val,
                    "dna_input": self.X_dna_val
                },
                self.Y_multi_val
            )
        )

        val_ds = (
            val_ds
            .batch(batch_size)                                           
            .prefetch(1)
        )

        #train against validation chromosomes and preserve best weights
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop, checkpoint, reduce_lr]             
        )
        print(f"evaluation finished on {yaml_config['params']['test_chrms']}")
        self.model   = model
        self.history = history
        return model, history


    # Pearson correlation summary
    # pooled correlation
    # track-wise correlation for each selected ChIP target
    # mean Pearson correlation and standard deviation across tracks
    
    def pearson_corr(self, predictions=None, target_channels=None):  

        if self.model is None:
            raise ValueError("Train the model first.")

        if self.X_Atac_test is None or self.X_dna_test is None or self.Y_multi_test is None:
            raise ValueError("Test data is missing.")

        if predictions is None:
            predictions = self.model.predict(
                [self.X_Atac_test, self.X_dna_test]
            )

        total_channels = self.Y_multi_test.shape[-1]

        
        if target_channels is None:
            print(f"target channel is none, using all chip data")
            target_channels = list(range(total_channels))

        elif isinstance(target_channels, (int, np.integer)):
            print(f" target channel is provided")
            target_channels = [int(target_channels)]

        else:
            target_channels = [int(ch) for ch in target_channels]


        
        invalid_channels = []

        for ch in target_channels:
            if ch < 0 or ch >= total_channels:
                invalid_channels.append(ch)
        if invalid_channels:
            raise ValueError(
                f"Invalid tracks {invalid_channels}. Valid tracks are 0 to {total_channels - 1}."
            )

        
        if predictions.shape != self.Y_multi_test.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} does not match "
                f"target shape {self.Y_multi_test.shape}."
            )

        
        multi_pearson_scores = []
        multi_all_true = []
        multi_all_pred = []

        total_windows = self.Y_multi_test.shape[0] # use every held-out test window for reported correlations

        for ch in target_channels:

            #extract matching true and predicted values for one selected ChIP track
            true_chip_values = self.Y_multi_test[:total_windows, :, ch].ravel()
            pred_chip_values = predictions[:total_windows, :, ch].ravel()

            # to remove negative and inf values
            valid_mask = np.isfinite(true_chip_values) & np.isfinite(pred_chip_values)  

            true_chip_values = true_chip_values[valid_mask]
            pred_chip_values = pred_chip_values[valid_mask]

            # filtering zeroes 
            if (
                true_chip_values.size > 1
                and np.std(true_chip_values) > 1e-6
                and np.std(pred_chip_values) > 1e-6
            ):
                r_value, _ = pearsonr(true_chip_values, pred_chip_values)

            else:
                r_value = np.nan

            multi_pearson_scores.append(r_value)
            multi_all_true.append(true_chip_values)
            multi_all_pred.append(pred_chip_values)

       
        multi_pooled_true = np.concatenate(multi_all_true)                           
        multi_pooled_pred = np.concatenate(multi_all_pred)


        
        if (
            multi_pooled_true.size > 1
            and np.std(multi_pooled_true) > 1e-6
            and np.std(multi_pooled_pred) > 1e-6
        ):
            pooled_r, _ = pearsonr(multi_pooled_true, multi_pooled_pred)

        else:
            pooled_r = np.nan

        
        mean_pearson = np.nanmean(multi_pearson_scores)
        std_r = np.nanstd(multi_pearson_scores)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # pearson plot 
        ploting_point_limit = 10000
        stride = max(1, multi_pooled_true.size // ploting_point_limit)

        axes[0].scatter(
            multi_pooled_true[::stride],
            multi_pooled_pred[::stride],
            alpha=0.1,
            color="teal",
            s=1
        )

        axis_min, axis_max = 0, 1.5   #<- fixed display range, independent of outlier signal values

        axes[0].plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="darkorange",
            linestyle="--",
            lw=2,
            label="Identity Line"
        )

        axes[0].set_xlim(axis_min, axis_max)
        axes[0].set_ylim(axis_min, axis_max)

        axes[0].set_title(
            f"Multimodal Model: Pooled Pearson r = {pooled_r:.4f}", fontsize=10
        )

        axes[0].set_xlabel("True Experimental Track Intensities")
        axes[0].set_ylabel("Predicted Profile Intensities")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        #  track-wise Pearson correlations (bar graph )
        bars = axes[1].bar(
            range(len(target_channels)),
            multi_pearson_scores,
            edgecolor="black",
            alpha=0.8
        )

        for bar, score in zip(bars, multi_pearson_scores):

            if not np.isnan(score):

                text_position = score + 0.02 if score >= 0 else score - 0.06

                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    text_position,
                    f"{score:.3f}",
                    ha="center"
                )

        #  mean correlation across selected chip data 
        axes[1].axhline(
            mean_pearson,
            color="darkorange",
            linestyle="--",
            label=f"Mean r = {mean_pearson:.3f}"
        )

        axes[1].set_title(
            f"Multimodal Pearson per Track\n"
            f"Mean r = {mean_pearson:.4f} ± {std_r:.4f}", fontsize=10
        )

        axes[1].set_xlabel("ChIP-seq Track")
        axes[1].set_ylabel("Pearson r")
        axes[1].set_xticks(range(len(target_channels)))
        axes[1].set_xticklabels(target_channels)
        axes[1].set_ylim(-1, 1)
        axes[1].set_xlim(-0.5, len(target_channels) - 0.5)
        axes[1].legend()
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

        print("\nPearson correlation per track:")

        for channel, score in zip(target_channels, multi_pearson_scores):
            print(f"Track {channel}: r = {score:.4f}")

        print(f"\nPooled Pearson: {pooled_r:.4f}")
        print(f"Mean Pearson: {mean_pearson:.4f}")
        print(f"Standard deviation: {std_r:.4f}")

        return {
            "target_channels": target_channels,
            "pearson_per_track": multi_pearson_scores,
            "pooled_pearson": pooled_r,
            "mean_pearson": mean_pearson,
            "std_pearson": std_r
        }


    
    #  multimodal model evaluation 
    #  evaluate loss/MAE once and visualize selected true-versus-predicted profiles
    
    def ATAC_DNA_model_Evaluation(self, target_channel=None, window_indices=None):  

        try:
            if self.model is None or self.history is None:
                raise ValueError(
                    "Model or training history is uninitialized. "
                    "Run ATAC_DNA_model_sequential() first.")

            if (
                self.X_Atac_test is None
                or self.X_dna_test is None
                or self.Y_multi_test is None
            ):
                raise ValueError( "Test  data sets are missing. Run Train_Test_Split() first.")

        except AttributeError as error:
            raise ValueError(
                f"Unable to access the model or test partitions: {error}"
            ) from error

            
            
        # measure weighted loss and MAE on held-out chromosomes
        print(f"Evaluating the multimodal model on the held-out test set...")
        test_metrics = self.model.evaluate(x=[self.X_Atac_test, self.X_dna_test], y=self.Y_multi_test,
                                            batch_size=64, verbose=1, return_dict=True)  
        for metric_name, metric_value in test_metrics.items():
            print(f"Test {metric_name.upper()}: {metric_value:.5f}")

            

       
        print(f"Generating predictions for the test set...")
        preds = self.model.predict([self.X_Atac_test, self.X_dna_test], batch_size=16, verbose=1) 

        if preds.shape != self.Y_multi_test.shape:
            raise ValueError(f"Prediction shape {preds.shape} does not match target shape {self.Y_multi_test.shape}.")

        total_channels = self.Y_multi_test.shape[-1]
        

        # normalize user channel selection into a list of integers
        if target_channel is None:
            print("Target chanel is not mentioned , using all tracks ")
            target_channels = list(range(total_channels))
            
        elif isinstance(target_channel, (int, np.integer)):
            print(f"target channel is mentioned using {target_channel}")
            target_channels = [int(target_channel)]
            
        elif isinstance(target_channel, (list, tuple, range, np.ndarray)):
            target_channels = [int(channel) for channel in target_channel]
        else:
            raise ValueError("target_channel must be None, integer, list, tuple, range, or NumPy array.")
            

        
        invalid_channels = []

        for c in target_channels:
            if c < 0 or c >= total_channels:
                invalid_channels.append(c)
                
        if invalid_channels:
            raise ValueError(f"Invalid target channels: {invalid_channels}. Valid channel range is 0 to {total_channels - 1}.")

        #retrieve a display name for every ChIP track when available
        if hasattr(self, "names") and self.names is not None:
            raw_names = list(self.names[1:]) if len(self.names) > total_channels else list(self.names)
            channel_names = [raw_names[c] if c < len(raw_names) else f"Channel {c}" for c in range(total_channels)]
        else:
            channel_names = [f"Channel {c}" for c in range(total_channels)]

        #=training and validation loss curves
        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))

        axes1[0].plot(self.history.history["loss"], label="Training Loss", color="royalblue", linewidth=2)
        axes1[0].plot(self.history.history["val_loss"], label="Validation Loss", color="darkorange", linewidth=2)
        axes1[0].set_title("Multimodal Weighted MSE Loss")
        axes1[0].set_xlabel("Epoch"); axes1[0].set_ylabel("Loss")
        axes1[0].legend(); axes1[0].grid(True, linestyle="--", alpha=0.6)

        axes1[1].plot(self.history.history["mae"], label="Training MAE", color="royalblue", linewidth=2)
        axes1[1].plot(self.history.history["val_mae"], label="Validation MAE", color="darkorange", linewidth=2)
        axes1[1].set_title("Mean Absolute Error")
        axes1[1].set_xlabel("Epoch"); axes1[1].set_ylabel("MAE")
        axes1[1].legend(); axes1[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout(); plt.show(); plt.close(fig1)

        # convert window_indices into a valid list of test-window numbers
        if window_indices is None:
            print(f"windo indices is not specified using minimum 3 winows ")
            window_indices = list(range(min(3, self.X_Atac_test.shape[0])))
            
        elif isinstance(window_indices, (int, np.integer)):
            print(f"found window indices using {window_indices}")
            window_indices = [int(window_indices)]
        else:
            window_indices = [int(i) for i in window_indices]

            

        invalid_windows = []

        for i in window_indices:
            if i < 0 or i >= self.X_Atac_test.shape[0]:
                invalid_windows.append(i)

        try:
            if invalid_windows:
                raise ValueError(
                    f"Invalid window indices: {invalid_windows}. "
                    f"Valid range is 0 to {self.X_Atac_test.shape[0] - 1}."
                )

            if len(window_indices) == 0:
                raise ValueError("At least one test window must be selected.")

        except (AttributeError, TypeError) as error:
            raise ValueError(
                f"Unable to validate test-window indices: {error}"
            ) from error
            

        positions = np.arange(self.Y_multi_test.shape[1])

        # plot one figure per ChIP track and one row per selected genomic window
        for channel in target_channels:
            
            number_of_windows = len(window_indices)
            fig2, axes2 = plt.subplots(number_of_windows, 1, figsize=(14, 3 * number_of_windows), sharex=True)
            axes2 = np.atleast_1d(axes2)

            for axis, window_index in zip(axes2, window_indices):
                true_profile = self.Y_multi_test[window_index, :, channel]
                predicted_profile = preds[window_index, :, channel]

                axis.plot(positions, true_profile, label="True (Experimental)", color="royalblue", linewidth=1.5)
                axis.plot(positions, predicted_profile, label="Predicted", color="darkorange", linestyle="--", linewidth=1.5)
                axis.fill_between(positions, true_profile, color="royalblue", alpha=0.15)

                axis.set_title(f"Test Window {window_index}")
                axis.set_ylabel("Signal")
                axis.legend(loc="upper right")
                axis.grid(True, linestyle="--", alpha=0.3)

            axes2[-1].set_xlabel("Position in Window (bp)")
            fig2.suptitle(f"True and Predicted Profiles — {channel_names[channel]}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show(); plt.close(fig2)

        # Pearson correlation summary
        self.pearson_corr(predictions=preds, target_channels=target_channels)

        #MAE per selected ChIP-seq track
        # useful for identifying target-specific differences in absolute error
        mae_per_channel = [np.mean(np.abs(self.Y_multi_test[:, :, c] - preds[:, :, c])) for c in target_channels]

        fig3 = plt.figure(figsize=(max(8, len(target_channels) * 1.5), 5))
        plot_positions = np.arange(len(target_channels))
        bars = plt.bar(plot_positions, mae_per_channel, color="royalblue", edgecolor="black", alpha=0.8)

        maximum_mae = max(mae_per_channel)
        for bar, mae_value in zip(bars, mae_per_channel):
            plt.text(bar.get_x() + bar.get_width() / 2, mae_value + maximum_mae * 0.02,
                      f"{mae_value:.4f}", ha="center", va="bottom", fontsize=9)

        plt.title("MAE Across Selected ChIP-seq Tracks (Multimodal Model)")
        plt.xlabel("ChIP-seq Track"); plt.ylabel("Mean Absolute Error")
        plt.xticks(plot_positions, [channel_names[c] for c in target_channels], rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout(); plt.show(); plt.close(fig3)

        return {
            "model": self.model, "history": self.history, "predictions": preds,
            "test_metrics": test_metrics, "target_channels": target_channels,
            "mae_per_channel": mae_per_channel
        }

        

    
    # qualitative fitting visualization only; this is not an evaluation metric
    def plot_best_worst_profiles(self, predictions=None, target_channel=None, custom_names=None):  

        
        if self.model is None:
            raise ValueError(" No trained model found.")
            
        if self.X_Atac_test is None or self.Y_multi_test is None:
            raise ValueError("Test partitions are missing. Call Train_Test_Split first.")

        total_channels = self.Y_multi_test.shape[-1]
        sequence_length = self.Y_multi_test.shape[1]

        
        # use mean Pearson correlation across the selected channels for window ranking
        if target_channel is None:
            print("target channel not dound using all channels ")
            rank_channels = list(range(total_channels))
        elif isinstance(target_channel, (int, np.integer)):
            rank_channels = [int(target_channel)]
            
        elif isinstance(target_channel, (list, tuple, range, np.ndarray)):
            rank_channels = [int(c) for c in target_channel]     
            
        else:
            raise ValueError("target_channel must be None, an int, list, tuple, range, or NumPy array.")

            

        invalid_channels = []

        for c in rank_channels:
            if c < 0 or c >= total_channels:
                invalid_channels.append(c)
        if invalid_channels:
            raise ValueError(f"Invalid target channels {invalid_channels}. Valid range: 0 to {total_channels - 1}.")

        total_rows = len(rank_channels)                                  
        color_map = plt.get_cmap('tab20', total_rows)                         
        colors = [color_map(i) for i in range(total_rows)]

        if custom_names is not None:
            print("custom name not found ")
            base_names = list(custom_names)
            
        elif hasattr(self, 'names') and self.names is not None:
            base_names = list(self.names[1:]) if len(self.names) > total_channels else list(self.names)
        else:
            base_names = []

        display_names_all = [base_names[r] if r < len(base_names) else f"Track Channel {r}" for r in range(total_channels)]
        display_names = [display_names_all[c] for c in rank_channels]  #<- names remain aligned with the requested channel order

        #rank windows by Pearson correlation, not signal magnitude
        # A prediction pass over the full test set is required for correct ranking.
        if predictions is None:
            print(" Generating full test predictions for performance ranking.")
            predictions = self.model.predict([self.X_Atac_test, self.X_dna_test], verbose=1)

        scores = []
        for i in range(len(self.Y_multi_test)):
            r_per_channel = []
            for c in rank_channels:
                t = self.Y_multi_test[i, :, c]                         
                p = predictions[i, :, c]                               
                multi_r_val = pearsonr(t, p)[0] if np.std(t) > 1e-6 and np.std(p) > 1e-6 else -1.0
                r_per_channel.append(multi_r_val)
            scores.append((i, float(np.mean(r_per_channel))))         

        # r is calculated for every window.
        # Three best and two worst windows are retained only to visualize fitting quality.
       
        best_windows = heapq.nlargest(3, scores, key=lambda x: x[1])    
        worst_windows = heapq.nsmallest(2, scores, key=lambda x: x[1])  
        selected_windows = best_windows + worst_windows

        selected_indices = [item[0] for item in selected_windows]
        selected_corrs = [item[1] for item in selected_windows]

        multi_predictions_subset = predictions[selected_indices]

        total_columns = len(selected_windows)
        
        fig, axes = plt.subplots(total_rows, total_columns, figsize=(5 * total_columns, 3.3 * total_rows), sharex='col')
        if total_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if total_columns == 1:
            axes = np.expand_dims(axes, axis=1)

        for col_idx, win_val in enumerate(selected_indices):
            current_corr = selected_corrs[col_idx]
            is_best = col_idx < len(best_windows)
            coloumn_label = "BEST PERFORMANCE" if is_best else "WORST PERFORMANCE"
            coloumn_color = "forestgreen" if is_best else "firebrick"

            for row_idx, c in enumerate(rank_channels):
                ax = axes[row_idx, col_idx]
                true_sig = self.Y_multi_test[win_val, :, c]
                pred_sig = multi_predictions_subset[col_idx, :, c]


                ax.fill_between(range(sequence_length), true_sig, color='gray', alpha=0.1)
                ax.plot(true_sig, color='black', linewidth=1, linestyle='--', alpha=0.4, label='True')
                ax.plot(pred_sig, color=colors[row_idx % len(colors)], linewidth=1.8, label='Pred')

                
                if col_idx == 0:
                    ax.set_ylabel(display_names[row_idx], fontweight='bold', rotation=0,
                                  labelpad=45, va='center', fontsize=12)


                
                y_axis_maximum = max(np.max(true_sig), np.max(pred_sig), 0.1)
                
                ax.set_ylim(-0.02, y_axis_maximum * 1.4)
                
                ax.text(0.95, 0.05, f"r={current_corr:.2f}", transform=ax.transAxes,
                        ha='right', fontsize=10, fontweight='bold', color=coloumn_color)
                

                # graph asthetics 
                
                if row_idx == 0:
                    ax.set_title(f"{coloumn_label}\nWindow Index: {win_val}", fontsize=15, pad=15,
                                 fontweight='bold', color=coloumn_color)
                if row_idx < (total_rows - 1):
                    ax.set_xticks([])
                else:
                    ax.set_xlabel("Position (bp)", fontsize=11)
                if col_idx == 0 and row_idx == 0:
                    ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92 if total_rows > 3 else 0.82, hspace=0.25)
        plt.suptitle("Evaluation of Multimodal Profile Predictions: Best vs. Worst Outcomes",
                     fontsize=26, fontweight='bold')
        plt.show()

    # Wasserstein distance for peak location and shape
    # asks how far predicted ChIP signal mass must move along each 2,000-bp window
    # to match the experimental profile:
    # 0 bp = perfect spatial agreement; larger values = greater spatial mismatch

    def Multimodal_cal_wasserstein(self, predictions=None, eps=1e-7, track_numbers=None):

        
        try:
            if self.model is None:
                raise ValueError("no model found. Run ATAC_DNA_model_sequential() first.")
            if self.X_Atac_test is None or self.X_dna_test is None or self.Y_multi_test is None:
                raise ValueError("test splits missing. Run Train_Test_Split first.")
            if predictions is None:
                predictions = self.model.predict([self.X_Atac_test, self.X_dna_test], verbose=1)

        except AttributeError as error:
            raise ValueError(f"Unable to access model or test data: {error}") from error 

            


        total_channels = self.Y_multi_test.shape[-1]

        if track_numbers is not None:
            print("track number not given")
            start, end = track_numbers[0], track_numbers[1]

            if start < 0 or end >= total_channels or start > end:                                
                raise ValueError(f"track_numbers {track_numbers} is out of range. Valid range: [0, {total_channels - 1}]")
                
            channel_indices = list(range(start, end + 1))                                       
        else:
            channel_indices = list(range(total_channels))                                        
            
        base_names = list(self.names[1:]) if self.names is not None and len(self.names) > total_channels else \
                     (list(self.names) if self.names is not None else [])
        disp_names_all = [base_names[c] if c < len(base_names) else f"Track {c}" for c in range(total_channels)]

        
        multi_wasserstein_per_channel = []

        for c in channel_indices:                                                                 #
            wasserstein_vals = []

            for i in range(len(self.Y_multi_test)):                                  
                
                # Distribution weights cannot be negative, so linear-output negatives are clipped.
                mulit_true_profile = np.clip(self.Y_multi_test[i, :, c], 0, None)
                multi_pred_profile = np.clip(predictions[i, :, c], 0, None)

                true_sum = mulit_true_profile.sum()                          
                pred_sum = multi_pred_profile.sum()                          

                if true_sum < eps:
                    continue                                                         

                Total_positions = len(mulit_true_profile)
                positions = np.arange(Total_positions, dtype=np.float64)                

                # probability distributions of true and pred 
                true_prob = (mulit_true_profile + eps) / (true_sum + eps * Total_positions)
                pred_prob = (multi_pred_profile + eps) / (pred_sum + eps * Total_positions)

                wasserstein_value = wasserstein_distance(
                    positions,
                    positions,
                    u_weights=true_prob,
                    v_weights=pred_prob
                )                                                                     

                wasserstein_vals.append(wasserstein_value)

            multi_wasserstein_per_channel.append(np.nanmean(wasserstein_vals) if wasserstein_vals else np.nan)

        disp_chip_names = []

        for c in channel_indices:
            disp_chip_names.append(disp_names_all[c])                             
        plot_indices = list(range(len(channel_indices)))                                         

        # plot Wasserstein distance per track for cross-model comparison
        
        plt.figure(figsize=(8, 5))                                                    
        bars = plt.bar(plot_indices, multi_wasserstein_per_channel, color='steelblue', edgecolor='black')
        plt.title("Wasserstein Distance per Track (Multimodal: DNA + ATAC model)")
        plt.xlabel("Track")
        plt.ylabel("Mean Wasserstein Distance (bp)")
        plt.xticks(plot_indices, disp_chip_names, rotation=45, ha='right')
        
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.1f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"multimodal_wasserstein_per_track_seed_{self.seed}.png", dpi=300)
        plt.show()

        for name, val in zip(disp_chip_names, multi_wasserstein_per_channel):
            print(f"{name}: Wasserstein Distance = {val:.2f} bp")

        avg_wasserstein = np.nanmean(multi_wasserstein_per_channel)
        print(f"\nAverage Wasserstein Distance across all tracks: {avg_wasserstein:.2f} bp")           

        return {"wasserstein_per_channel": multi_wasserstein_per_channel, "names": disp_chip_names, "avg_wasserstein": avg_wasserstein}



    # Jensen-Shannon divergence
    # JSD measures how similar the predicted ChIP-seq signal distribution is to the
    # experimental distribution.
    # measures similarity between predicted and experimental ChIP signal distributions
    # after each profile is normalized to unit mass; lower values indicate better agreement
    
    def Multimodal_cal_jsd(self, predictions=None, eps=1e-7, track_numbers=None):         
        
        try:
            if self.model is None:
                raise ValueError("no model found. Run ATAC_DNA_model_sequential() first.")
            if self.X_Atac_test is None or self.X_dna_test is None or self.Y_multi_test is None:
                raise ValueError("test splits missing. Run Train_Test_Split first.")
            if predictions is None:
                predictions = self.model.predict([self.X_Atac_test, self.X_dna_test], verbose=1)
        except AttributeError as error:
            raise ValueError(f"Unable to access model or test data: {error}") from error

        
        total_channels = self.Y_multi_test.shape[-1]

        if track_numbers is not None:   
            print("")
            start, end = track_numbers[0], track_numbers[1]
            if start < 0 or end >= total_channels or start > end:                                
                raise ValueError(f"track_numbers {track_numbers} is out of range. Valid range: [0, {total_channels - 1}]")
                
            channel_indices = list(range(start, end + 1))                                        
        else:
            channel_indices = list(range(total_channels))                                       

        base_names = list(self.names[1:]) if self.names is not None and len(self.names) > total_channels else \
                     (list(self.names) if self.names is not None else [])
        disp_all_chip_names = [base_names[c] if c < len(base_names) else f"Track {c}" for c in range(total_channels)]

       
        mulit_jsd_per_channel = []

        for c in channel_indices:                                                                
            jsd_vals = []
            for i in range(len(self.Y_multi_test)):
                # Distribution weights cannot be negative, so linear-output negatives are clipped.
                true_profile = np.clip(self.Y_multi_test[i, :, c], 0, None)
                pred_profile = np.clip(predictions[i, :, c], 0, None)

                true_sum = true_profile.sum()                          
                pred_sum = pred_profile.sum()                          
                if true_sum < eps:
                    continue                                                          #

               
                true_prob = (true_profile + eps) / (true_sum + eps * len(true_profile))
                pred_prob = (pred_profile + eps) / (pred_sum + eps * len(pred_profile))

                multi_jsd_distance = jensenshannon(true_prob, pred_prob, base=2)
                jsd_vals.append(multi_jsd_distance ** 2)                                    

            mulit_jsd_per_channel.append(np.nanmean(jsd_vals) if jsd_vals else np.nan)

        # all chip names 
        disp_chip_names = []

        for c in channel_indices:
            disp_chip_names.append(disp_all_chip_names[c])                              
        plot_indices = list(range(len(channel_indices)))                                         

        plt.figure(figsize=(8, 5))                                                    
        bars = plt.bar(plot_indices, mulit_jsd_per_channel, color='mediumpurple', edgecolor='black')
        plt.title("JSD per Track (Multimodal: DNA + ATAC model)")
        plt.xlabel("Track")
        plt.ylabel("JSD")
        plt.ylim(0, 1)
        plt.xticks(plot_indices, disp_chip_names, rotation=45, ha='right')
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"multimodal_jsd_per_track_seed_{self.seed}.png", dpi=300)
        plt.show()

        for name, val in zip(disp_chip_names, mulit_jsd_per_channel):
            print(f"{name}: JSD = {val:.4f}")

        avg_jsd = np.nanmean(mulit_jsd_per_channel)                                        
        print(f"\nAverage JSD across all tracks: {avg_jsd:.4f}")

        return {"jsd_per_channel": mulit_jsd_per_channel, "names": disp_chip_names, "avg_jsd": avg_jsd}


# This function trains and evaluates the multimodal model independently with every
# random seed listed in the YAML file. It saves each run and calculates the mean and
# sample standard deviation across completed seeds.

def multimodal_seed_execution(loader, TN5_frags, config_path="config_file_ml.yaml",
                              output_dir="results/seed_runs"):  
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    seeds = yaml_config["params"]["model_seeds"]["multimodal"]       # seed list is controlled centrally by config file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "multimodal_seed_results.csv"
    txt_path = output_path / "multimodal_seed_results.txt"
    all_results = []

    # evaluation metrics=
    fieldnames = [
        "seed", "test_loss", "test_mae", "pooled_pearson", "mean_pearson",
        "std_pearson", "avg_wasserstein_bp", "avg_jsd", "epochs_completed"
    ]

    #independent seed loop
    for seed in seeds:
        print(f"\n{'=' * 70}\nRunning multimodal model with seed {seed}\n{'=' * 70}")
        tf.keras.backend.clear_session()                             
        set_global_seed(seed)                                        

        experiment = MultimodalGenomicModel.from_loader(loader)      #<- create a fresh model object while reusing loaded genomic arrays
        experiment.Train_Test_Split(config_path=config_path, TN5_frags=TN5_frags)
        experiment.ATAC_DNA_model_sequential(config_path=config_path, seed=seed)

        evaluation = experiment.ATAC_DNA_model_Evaluation()          #<- test metrics and predictions are generated once per seed
        predictions = evaluation["predictions"]                      #<- reuse identical predictions for every downstream metric
        pearson = experiment.pearson_corr(predictions=predictions)
        wasserstein = experiment.Multimodal_cal_wasserstein(predictions=predictions)
        jsd = experiment.Multimodal_cal_jsd(predictions=predictions)

        metrics = evaluation["test_metrics"]
        #=single structured result row for this completed seed
        result = {
            "seed": seed,
            "test_loss": float(metrics["loss"]),
            "test_mae": float(metrics["mae"]),
            "pooled_pearson": float(pearson["pooled_pearson"]),
            "mean_pearson": float(pearson["mean_pearson"]),
            "std_pearson": float(pearson["std_pearson"]),
            "avg_wasserstein_bp": float(wasserstein["avg_wasserstein"]),
            "avg_jsd": float(jsd["avg_jsd"]),
            "epochs_completed": len(experiment.history.history["loss"])
        }
        all_results.append(result)                                 

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)                            

        with open(txt_path, "w") as f:
            for row in all_results:
                f.write("\t".join(f"{key}: {row[key]}" for key in fieldnames) + "\n")

    # mean and sample standard deviation across completed seeds
    summary = {}
    metric_names = fieldnames[1:]
    for metric_name in metric_names:
        # aggregate only completed seed runs
        values = np.asarray([row[metric_name] for row in all_results], dtype=float)  
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0  # sample SD across independent seeds
        }

    with open(txt_path, "a") as f:
        f.write("\nOverall results (mean ± sample SD)\n")
        for metric_name, values in summary.items():
            f.write(f"{metric_name}: {values['mean']:.6f} ± {values['std']:.6f}\n")

    return all_results, summary
