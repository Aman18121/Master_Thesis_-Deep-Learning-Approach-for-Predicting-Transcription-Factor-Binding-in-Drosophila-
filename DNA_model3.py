'''This python script belongs to 
author = [AMAN YADAV]
script 3/6

as part of Masterthesis [A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila]
'''

import csv
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wasserstein_distance
import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from scipy.spatial.distance import jensenshannon



# The
def set_global_seed(seed):
    random.seed(seed)             
    np.random.seed(seed)          
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()  
    print(f"Global random seed: {seed}")




#(Dna prediction (main model). imported the script before execution 
# requires the following data from the genomic data loader 
# concatenated_data ( all chip data combined )
# one_hot encoded dna sequence 
# track names (for chip identification)
# chrms for splitting data 


class DNAPredictionModel:
    def __init__(self, concatenated_data, window_chroms, one_hot, track_names):           
        self.concatenated_data = concatenated_data
        self.window_chroms = np.array(window_chroms) if window_chroms is not None else None
        self.one_hot = one_hot
        self.names = track_names
        
        
        self.X_dna_train = None
        self.Y_dna_train = None                                                
        self.X_dna_val = None
        self.Y_dna_val = None 
        self.X_dna_test = None
        self.Y_dna_test = None

        # for model evaluation
        self.model = None                                                        
        self.history = None
        self.seed = None

        print("DNA model is active")


   
    @classmethod
    def from_loader(cls, loader_instance):          

        
        if not hasattr(loader_instance, 'dna_tracks') or loader_instance.dna_tracks is None:
            raise ValueError("one-hot DNA missing, run read_one_hot_encode() first.")
        if not hasattr(loader_instance, 'concatenated_data') or loader_instance.concatenated_data is None:
            raise ValueError("The chip data is missing. run read_concatenated_chip() first from genomic loader")

    
        return cls(
            concatenated_data=loader_instance.concatenated_data,
            window_chroms=getattr(loader_instance, 'window_chromosomes', None),    
            one_hot=getattr(loader_instance, 'dna_tracks', None),                  
            track_names=getattr(loader_instance, 'names', None)
        )


    # Train-test split of DNA and chip data
    # chnage chrms from config file 
    
    def Train_test_Split(self, config_path="config_file_ml.yaml"):

   
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        
        train_chr = yaml_config["params"]["training_chrms"]
        test_chr = yaml_config["params"]["test_chrms"]
        val_chr = yaml_config["params"]["validation_chrms"]

        # for both single and multiple chrms, convert to list for uniform processing
        if isinstance(train_chr, str): train_chr = [train_chr]
        if isinstance(test_chr, str): test_chr = [test_chr]
        if isinstance(val_chr, str): val_chr = [val_chr]

        
        if self.window_chroms is None or self.one_hot is None:                
            raise ValueError("window_chroms or one_hot data is missing. Ensure the genomic data loader has been executed correctly.")
            
    
        Dna_length = self.one_hot[0]
    
        # chip data (all ChIP-seq tracks) starts from second channel onward
        chip = self.concatenated_data[:, :, 1:]                                       
        chrom_array = self.window_chroms  

       
        test_mask = np.isin(chrom_array, test_chr)
        val_mask = np.isin(chrom_array, val_chr)
        train_mask = np.isin(chrom_array, train_chr)

        # check for overlapping chromosome groups and empty splits.
        try:
            if np.intersect1d(train_chr, test_chr).size > 0:
                raise ValueError("Training and testing chromosomes overlap. ")
            if np.intersect1d(train_chr, val_chr).size > 0:
                raise ValueError("Training and validation chromosomes overlap. ")
            if np.intersect1d(test_chr, val_chr).size > 0:
                raise ValueError("Testing and validation chromosomes overlap. ")  
            if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
                raise ValueError("At least one chromosome split contains no windows.") 
        except Exception as e:
            raise ValueError(f"Error checking chromosome overlaps: {e}")
        

        # DNA = features 
        # chip = prediction output 
        
        self.X_dna_train = Dna_length[train_mask]
        self.Y_dna_train = chip[train_mask]
        self.X_dna_val   = Dna_length[val_mask]
        self.Y_dna_val   = chip[val_mask]
        self.X_dna_test  = Dna_length[test_mask]
        self.Y_dna_test  = chip[test_mask]

        print("Available chromosomes:", sorted(set(chrom_array.astype(str))))
        print("Train windows:", np.sum(train_mask))
        print("Val windows:", np.sum(val_mask))
        print("Test windows:", np.sum(test_mask))

        print(f"Train Layout : Features {self.X_dna_train.shape} | Labels {self.Y_dna_train.shape}")
        return (self.X_dna_train, self.Y_dna_train), (self.X_dna_val, self.Y_dna_val), (self.X_dna_test, self.Y_dna_test)



    def _make_train_dataset(self, batch_size, seed=None):
        dataset = tf.data.Dataset.from_tensor_slices((self.X_dna_train, self.Y_dna_train))
        dataset = dataset.shuffle(256, seed=seed, reshuffle_each_iteration=True)
        return dataset.batch(batch_size).prefetch(1)


    def _make_val_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((self.X_dna_val, self.Y_dna_val))
        return dataset.batch(batch_size).prefetch(1)





    # 1d convolutional neural network for DNA sequence to predict ChIP-seq signal

    def DNA_model_sequential(self, input_shape=(None, 4) , config_path="config_file_ml.yaml", seed=None): 
        
        try:
            if self.X_dna_train is None or self.Y_dna_train is None:                                        
                raise ValueError("Splits are incomplete . Execute Train_test_Split() first.") 

        except Exception as e:
            raise ValueError(f"Error in training splits: {e}")


        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        
        epochs = yaml_config["params"]["epochs_DNA"]
        batch_size = yaml_config["params"]["batch_size_dna"]
        
        
        dna_callback_cfg = yaml_config["params"]["dna_callbacks"]
        dna_patience = dna_callback_cfg["dna_early_stop_patience"]
        dna_lr_patience = dna_callback_cfg["dna_reduce_lr_patience"]
        dna_lr_factor   = dna_callback_cfg["dna_lr_decay_factor"]

        dna_loss_cfg = yaml_config["params"]["dna_loss_tuning"]
        peak_threshold_dna  = dna_loss_cfg["dna_peak_threshold"]
        peak_multiplier_Dna = dna_loss_cfg["dna_peak_weight_multiplier"]
        background_penalty = dna_loss_cfg["dna_background_penalty"]

        configured_seeds = yaml_config["params"]["model_seeds"]["dna"]

        # (!!! important !!!)atleast 1 seed is neccesary for reporducibility

        if seed is None:
            print("No seed provided. Using default seed from configuration.")
            raise ValueError("A seed must be provided.")
        
        if seed not in configured_seeds:
            print(f"Seed {seed} is not configured for the DNA model.")
            raise ValueError(
                f"Seed {seed} is not configured for the DNA model. "
                f"Configured seeds: {configured_seeds}"
            )

        
        set_global_seed(seed)
        self.seed = seed

        # track count for input shape and output layer configuration
        Dna_length = self.X_dna_train.shape[1]

        total_chip_targets = self.Y_dna_train.shape[-1]  

        print(f" DNA Model Configuration: Input Shape = {input_shape} | Output Channels = {total_chip_targets}")

        # flexible  input shape for future chnages
        inputs = tf.keras.Input(shape=input_shape)   

        # Initial Convolution Block
        x = tf.keras.layers.Conv1D(64, kernel_size=15, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Level 2: Receptive Field Scaling (Rate = 2)
        # dialation increment 
        x = tf.keras.layers.Conv1D(128, kernel_size=15, padding='same', dilation_rate=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv1D(128, kernel_size=21, padding='same', dilation_rate=4)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

     
        x = tf.keras.layers.Conv1D(128, kernel_size=21, padding='same', dilation_rate=8)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        #  Resolution Normalization Layer (Re-aligns massive dilation fields down to base-pair scale)
        x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', dilation_rate=1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        outputs = tf.keras.layers.Conv1D(filters=total_chip_targets, kernel_size=1, activation='linear', padding='same')(x) 
        model = tf.keras.Model(inputs, outputs)

        Dna_graph_threshold = peak_threshold_dna
        Dna_graph_multiplier = peak_multiplier_Dna

        # custom loss function for genomic data, emphasizing peak regions and penalizing background noise
        # normal MSE was not sufficient for genomic data, as it treats all errors equally, leading to poor peak prediction. 
        def genomic_weighted_mse(y_true, y_pred):
           
            peak_weights = tf.where(y_true > Dna_graph_threshold, Dna_graph_multiplier, 1.0)
    
            bg_penalty_mask = tf.where((y_true < 0.01) & (y_pred > 0.05), background_penalty, 1.0)
            weights = peak_weights * bg_penalty_mask
    
            squared_errors = tf.square(y_true - y_pred)
            return tf.reduce_mean(squared_errors * weights)


        # settings are present in config files
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss=genomic_weighted_mse,
            metrics=['mae']
        )
            
    
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=dna_patience,
            restore_best_weights=True
        )

        Dna_checkpoint_dir = Path("checkpoints") / "dna" / f"seed_{seed}"
        Dna_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # to save the best model 
        checkpoint = ModelCheckpoint(
            filepath=str(Dna_checkpoint_dir / "best_dna_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1 )

        # learning rate scheduler 
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=dna_lr_factor,
            patience=dna_lr_patience,
            min_lr=1e-6,
            verbose=1 )

        # reffer config file for editing batch size and epochs
        train_data_Dna = self._make_train_dataset(batch_size, seed=seed)              
        val_data_Dna   = self._make_val_dataset(batch_size)

        print(f" Dispatching model to computing node execution clusters...")
        history = model.fit(
            train_data_Dna,
            validation_data=val_data_Dna,
            epochs=epochs,
            callbacks=[early_stop, checkpoint, reduce_lr])
        
        self.model = model
        self.history = history
        return model, history




    #pearson corelation summary 
    # pooled 
    # track wise (individual chip)
    # mean pearson 

    def pearson_cor_summary(self, predictions=None, target_tracks=None):

        try:
            if self.model is None:
                raise ValueError("Train the model first.")
            
            if self.X_dna_test is None or self.Y_dna_test is None:
                raise ValueError("Test data is missing.")

        except Exception as e:
            raise ValueError(f"model is not yet trained or test data is missing: {e}")


        if predictions is None:
            print(f"Running predictions on test set {self.seed}...")
            predictions = self.model.predict(self.X_dna_test)

        
        total_channels = self.Y_dna_test.shape[-1]

        # if not defined bu user target channel will take all the chip channels included in the experiment 
        # Its very memomry heavy 
        # its recommended by author to set a target  limit 
        if target_tracks is None:
            print("No target tracks specified. Evaluating all channels.")
            target_channels = list(range(total_channels))

        elif isinstance(target_tracks, (int, np.integer)):
            target_channels = [int(target_tracks)]
        else:
            target_channels = [int(ch) for ch in target_tracks]


        invalid_channels = []
        for ch in target_channels:
            if ch < 0 or ch >= total_channels:
                invalid_channels.append(ch)

        if invalid_channels:
            raise ValueError(
                f"Invalid tracks {invalid_channels}. Valid tracks are 0 to {total_channels - 1}."
            )

        Dna_pearson_scores = []
        Dna_all_true = []
        Dna_all_pred = []

        
        max_windows = self.Y_dna_test.shape[0]

        for ch in target_channels:
            
            true_chip_values = self.Y_dna_test[:max_windows, :, ch].flatten()
            pred_chip_values = predictions[:max_windows, :, ch].flatten()

          
            # values scould not be 0.0000 for a meaning full corelation
            if np.std(true_chip_values) > 1e-6 and np.std(pred_chip_values) > 1e-6:
                print(f"Calculating Pearson correlation for track {ch}...")
                r_value, _ = pearsonr(true_chip_values, pred_chip_values)
            else:
                # not a number 
                r_value = np.nan

            Dna_pearson_scores.append(r_value)
            Dna_all_true.append(true_chip_values)
            Dna_all_pred.append(pred_chip_values)

        #pooled corelation 
        Dna_pooled_true = np.concatenate(Dna_all_true)
        Dna_pooled_pred = np.concatenate(Dna_all_pred)

        if np.std(Dna_pooled_true) > 1e-6 and np.std(Dna_pooled_pred) > 1e-6:
            pooled_r, _ = pearsonr(Dna_pooled_true, Dna_pooled_pred)
        else:
            pooled_r = np.nan
            
        #mean r value 
        mean_r = np.nanmean(Dna_pearson_scores)
        std_r = np.nanstd(Dna_pearson_scores)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        stride = max(1, len(Dna_pooled_true) // 10000)

        axes[0].scatter(
            Dna_pooled_true[::stride],
            Dna_pooled_pred[::stride],
            s=1,
            alpha=0.1,
            color="teal"
        )

        axis_min, axis_max = 0, 0.7   
        
        # scatter plot of true vs predict

        axes[0].plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="darkorange",
            label="Identity line"
        )

        axes[0].set_xlim(axis_min, axis_max)
        axes[0].set_ylim(axis_min, axis_max)

        axes[0].set_title( f"DNA Model: Pooled Pearson r = {pooled_r:.4f}", fontsize=10)
        axes[0].set_xlabel("True Signal")
        axes[0].set_ylabel("Predicted Signal")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        
        #trach wise pearson corelation 
        bars = axes[1].bar(
            range(len(target_channels)),
            Dna_pearson_scores,
            edgecolor="black",
            alpha=0.8
        )

        for bar, score in zip(bars, Dna_pearson_scores):

            if not np.isnan(score):

                text_position = score + 0.02 if score >= 0 else score - 0.06

                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    text_position,
                    f"{score:.3f}",
                    ha="center"
                )

        # pearson line of tracks
        axes[1].axhline(
            mean_r,
            linestyle="--",
            color="darkorange",
            label=f"Mean r = {mean_r:.3f}"
        )

        axes[1].set_title(
            f"DNA Model: Pearson per Track\n"
            f"Mean r = {mean_r:.4f} ± {std_r:.4f}", fontsize=10
        )
        axes[1].set_xlabel("chip_seq Track")
        axes[1].set_ylabel("Pearson r")
        axes[1].set_xticks(range(len(target_channels)))
        axes[1].set_xticklabels(target_channels)
        axes[1].set_ylim(-0.1, 1)
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\nPearson correlation per track:")

        for ch, score in zip(target_channels, Dna_pearson_scores):
            print(f"Track {ch}: r = {score:.4f}")

        print(f"\nPooled Pearson: {pooled_r:.4f}")
        print(f"Mean Pearson: {mean_r:.4f}")
        print(f"Standard deviation: {std_r:.4f}")


        return {
            "pearson_per_track": Dna_pearson_scores,
            "pooled_pearson": pooled_r,
            "mean_pearson": mean_r,
            "std_pearson": std_r
        }

    
    #1d convolutional neural network for DNA sequence to predict ChIP-seq signal
    def DNA_model_Evaluation(self, target_channel=None, window_indices=None):

        try:
            if self.model is None:
                raise ValueError("No trained model found. Train the model first.")
            if self.X_dna_test is None or self.Y_dna_test is None:
                raise ValueError("Test data is missing. Run Train_test_Split() first.")
            if self.history is None:
                raise ValueError("Model history is uninitialized. Train the model first.")
        except Exception as e:
            raise ValueError(f"Model evaluation prerequisites not met or data is missing: {e}")
        
        print(f" Evaluating DNA model performance on test set...")

        
        test_DNA_metrics = self.model.evaluate(self.X_dna_test, self.Y_dna_test, batch_size=64, verbose=1)
        
        preds = self.model.predict(self.X_dna_test)

        
        total_channels = self.Y_dna_test.shape[-1]

        if target_channel is None:
            target_channels = list(range(total_channels))
        elif isinstance(target_channel, (int, np.integer)):
            target_channels = [int(target_channel)]
        elif isinstance(target_channel, (list, tuple, range, np.ndarray)):
            target_channels = [int(ch) for ch in target_channel]
        else:
            raise ValueError("target_channel must be None, an int, list, tuple, range, or NumPy array.")
        
        invalid_channels = []
        for chanels in target_channels:
            if chanels < 0 or chanels >= total_channels:
                invalid_channels.append(chanels) 



        if invalid_channels:
            raise ValueError(f"target_channel {invalid_channels} is out of range. Valid range: 0 to {total_channels - 1}")

        # fetching chip names 

        if hasattr(self, "names") and self.names is not None:
            print(f"Using provided track names for display.{self.names}")
            raw_names = self.names[1:] if len(self.names) > total_channels else self.names

            chanel_names = []
            for c in range(total_channels):
                name = raw_names[c] if c < len(raw_names) else f"Channel {c}"
                chanel_names.append(name)
                
            
        else:
            chanel_names = [f"Channel {c}" for c in range(total_channels)]

        for name, val in zip(self.model.metrics_names, test_DNA_metrics):
            print(f"  Test {name.upper()}: {val:.5f}")

        # Mae and loss plots 
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'],     label='Train Loss', color='royalblue', lw=2)
        plt.plot(self.history.history['val_loss'], label='Val Loss',   color='darkorange', lw=2)
        plt.title('Genomic Weighted MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'],     label='Train MAE', color='royalblue', lw=2)
        plt.plot(self.history.history['val_mae'], label='Val MAE',   color='darkorange', lw=2)
        plt.title('Mean Absolute Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        #converts window_indices into a valid list of test-window numbers.
        if window_indices is None:
            print(" window indices are not given . using minimum 3")
            window_indices = list(range(min(3, self.X_dna_test.shape[0])))
        elif isinstance(window_indices, (int, np.integer)):
            print()
            window_indices = [int(window_indices)]
        else:
            window_indices = [int(idx) for idx in window_indices]

        window_indices = [
            idx for idx in window_indices
            if 0 <= idx < self.X_dna_test.shape[0]
        ]

        if len(window_indices) == 0:
            raise ValueError("No valid window indices provided.")


        # 
        for window_idx in window_indices:
            n_tracks = len(target_channels)
            fig, axes = plt.subplots(n_tracks, 1, figsize=(14, 3 * n_tracks), sharex=True)
            axes = np.atleast_1d(axes)
            positions = np.arange(self.Y_dna_test.shape[1])

            for ax, ch in zip(axes, target_channels):
                true_profile = self.Y_dna_test[window_idx, :, ch]
                pred_profile = preds[window_idx, :, ch]

                ax.plot(true_profile, label="True", color="royalblue", alpha=0.8, lw=1.5)
                ax.plot(pred_profile, label="Predicted", color="darkorange", linestyle="--", lw=1.5)

                ax.fill_between(
                    positions,
                    true_profile,
                    color="royalblue",
                    alpha=0.15
                )
                ax.set_title(f"{chanel_names[ch]} | Test Window {window_idx}")
                ax.set_ylabel("Signal")
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Window Position (bp)")
            fig.suptitle(f"True and Predicted ChIP-seq Profiles | Test Window {window_idx}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

        # --- Pearson summary ------------------------
        self.pearson_cor_summary(
            predictions=preds,
            target_tracks=target_channels
        )

        # --- MAE per channel bar plot ---
        plt.figure()                                                      #  MAE per track (useful for excluding bad SRRs)
        mae_per_ch = [np.mean(np.abs(self.Y_dna_test[:, :, c] - preds[:, :, c])) for c in target_channels]
        bars = plt.bar(range(len(target_channels)), mae_per_ch, color='blue', edgecolor='black', alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + (max(mae_per_ch) * 0.02),
                     f"{h:.4f}", ha='center', va='bottom', fontsize=9)
        plt.title("MAE Across ChIP Tracks")
        plt.xlabel("Track")
        plt.ylabel("MAE")
        plt.xticks(range(len(target_channels)), [chanel_names[c] for c in target_channels], rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        
        return self.model, self.history
        
    #3 best and 2 worst fitting profiles of true and pred values (not a model evaluation metrics !!!)
    # non of the model in this thesis were evaluated on this profile ....
    
    def plot_best_worst_profiles(self, predictions=None, target_channel=None, custom_names=None):          
        try:
            if self.model is None:
                raise ValueError("no model found. run training model ")
            if self.X_dna_test is None or self.Y_dna_test is None:
                raise ValueError("test splits missing, run Train_test_Split first.")
        except AttributeError as error:
            raise ValueError(f"Model or test data attribute is missing: {error}") from error
            
        
        if predictions is None:
            print("running predictions on test set...")
            predictions = self.model.predict(self.X_dna_test, verbose=1)

        num_channels_total = self.Y_dna_test.shape[-1]

        # Normalize target_channel to a clean list of valid ints
        
        if target_channel is None:
            channels = list(range(num_channels_total))
        elif isinstance(target_channel, (int, np.integer)):
            channels = [int(target_channel)]
        elif isinstance(target_channel, (list, tuple, range, np.ndarray)):
            channels = [int(c) for c in target_channel]
        else:
            raise ValueError("target_channel must be None, int, list, tuple, range, or NumPy array.")

        invalid_channels = [c for c in channels if c < 0 or c >= num_channels_total]
        if invalid_channels:
            raise ValueError(f"Invalid channels {invalid_channels}. Valid range is 0 to {num_channels_total - 1}.")

        num_rows = len(channels)                                          #  only plot the requested channels, not all tracks
        cmap = plt.cm.get_cmap('tab20', num_rows)                         #  auto color per track
        colors = [cmap(i) for i in range(num_rows)]

        if custom_names is not None:
            base_names = list(custom_names)
        elif self.names is not None:
            base_names = list(self.names[1:]) if len(self.names) > num_channels_total else list(self.names)
        else:
            base_names = []

        disp_names_all = [base_names[c] if c < len(base_names) else f"Track {c}" for c in range(num_channels_total)]
        disp_names = [disp_names_all[c] for c in channels]                #  names for just the requested channels, in order

        # --- Ranking: average Pearson r across the requested channel(s) per window ---
        scores = []
        for i in range(len(self.Y_dna_test)):
            r_per_channel = []
            for c in channels:
                t = self.Y_dna_test[i, :, c]                              
                p = predictions[i, :, c]
                r = pearsonr(t, p)[0] if np.std(t) > 1e-6 and np.std(p) > 1e-6 else -1.0
                r_per_channel.append(r)
            scores.append((i, np.mean(r_per_channel)))                    

        scores.sort(key=lambda x: x[1], reverse=True)

        combined = scores[:3] + scores[-2:]                               #  top 3 + bottom 2
        indices  = [s[0] for s in combined]
        corrs    = [s[1] for s in combined]

        fig, axes = plt.subplots(num_rows, 5, figsize=(25, 3.3 * num_rows), sharex='col')
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        #r value for eevery indcies 
        # 3 best windows per track  
        # 2 worst windows  per track 
       # Ment for Visual comparision only 
        for col_idx, win_val in enumerate(indices):
            r       = corrs[col_idx]
            is_best = col_idx < 3
            col_lbl = "BEST" if is_best else "WORST"
            col_clr = "forestgreen" if is_best else "firebrick"

            for row_idx, c in enumerate(channels):                        #  iterate over requested channels, not range(num_rows) directly
                ax       = axes[row_idx, col_idx]
                true_sig = self.Y_dna_test[win_val, :, c]
                pred_sig = predictions[win_val, :, c]

                ax.fill_between(range(len(true_sig)), true_sig, color='gray', alpha=0.1)
                ax.plot(true_sig, color='black', linewidth=1, linestyle='--', alpha=0.4, label='True')
                ax.plot(pred_sig, color=colors[row_idx], linewidth=1.8, label='Pred')

                if col_idx == 0:
                    ax.set_ylabel(disp_names[row_idx], fontweight='bold', rotation=0, labelpad=45, va='center', fontsize=12)

                ymax = max(np.max(true_sig), np.max(pred_sig), 0.1)
                ax.set_ylim(-0.02, ymax * 1.4)
                ax.text(0.95, 0.05, f"r={r:.2f}", transform=ax.transAxes,
                        ha='right', fontsize=10, fontweight='bold', color=col_clr)

                if row_idx == 0:
                    ax.set_title(f"{col_lbl}\nwindow: {win_val}", fontsize=15, pad=15, fontweight='bold', color=col_clr)
                if row_idx < (num_rows - 1):
                    ax.set_xticks([])
                else:
                    ax.set_xlabel("Position (bp)", fontsize=11)
                if col_idx == 0 and row_idx == 0:
                    ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92 if num_rows > 3 else 0.82, hspace=0.25)
        plt.suptitle("best vs worst predicted windows", fontsize=26, fontweight='bold')
        plt.savefig("dna_best_worst.png", dpi=300)
        plt.show()
        print("saved to dna_best_worst.png")


    #================== calculating Wasserstein distance for peak shape/position accuracy of the DNA model==========================
    # how accurately the DNA-only model places the predicted ChIP-seq signal along each 2,000-bp test window.?
    #0 bp   = perfect spatial agreement
    #20 bp  = small spatial mismatch
    #150 bp = larger spatial mismatch
    # eps = epsilon: a very small positive number.
    
    def DNA_cal_wasserstein(self, predictions=None, eps=1e-7, track_numbers=None): 
        if self.model is None:
            raise ValueError("no model found. Run DNA_model_sequential() first.")
        if self.X_dna_test is None or self.Y_dna_test is None:
            raise ValueError("test splits missing, run Train_test_Split first.")
        if predictions is None:
            predictions = self.model.predict(self.X_dna_test, verbose=1)     #  DNA input passed to the trained DNA-only model

        #==============DNA channels=================================================
        num_channels = self.Y_dna_test.shape[-1]

        #============ if track_numbers is given, restrict evaluation to that inclusive range e.g. [1,7] = tracks 1 to 7===========
        if track_numbers is not None:              
            start, end = track_numbers[0], track_numbers[1]
            if start < 0 or end >= num_channels or start > end:  #  validate bounds to prevent silent out-of-range indexing
                raise ValueError(f"track_numbers {track_numbers} is out of range. Valid range: [0, {num_channels - 1}]")
            channel_indices = list(range(start, end + 1))     #  end+1 so the range is inclusive of the last track
        else:
            channel_indices = list(range(num_channels))    #  track_numbers=None: fall back to evaluating all channels as before

        base_names = list(self.names[1:]) if self.names is not None and len(self.names) > num_channels else \
                     (list(self.names) if self.names is not None else [])
        disp_names_all = [base_names[c] if c < len(base_names) else f"Track {c}" for c in range(num_channels)]

        # palce holder for all the chip wasserstien output
        wasserstein_per_channel = []
       
        for c in channel_indices:      
            wasserstein_vals = []

            for i in range(len(self.Y_dna_test)): 
                #distribution weights cannot be negative.
                true_profile = np.clip(self.Y_dna_test[i, :, c], 0, None)
                pred_profile = np.clip(predictions[i, :, c], 0, None)
                
                true_sum = true_profile.sum()  # total experimental signal
                pred_sum = pred_profile.sum()  # total pred signal

                if true_sum < eps:
                    continue               
                
                num_positions = len(true_profile)
                positions = np.arange(num_positions, dtype=np.float64)           

                #Converting the true profile into a probability distribution
                true_prob = (true_profile + eps) / (true_sum + eps * num_positions)
                pred_prob = (pred_profile + eps) / (pred_sum + eps * num_positions)

                wasserstein_value = wasserstein_distance(
                    positions,
                    positions,
                    u_weights=true_prob,
                    v_weights=pred_prob
                )                          #  distance in bp between the true and predicted signal distributions

                wasserstein_vals.append(wasserstein_value)

            wasserstein_per_channel.append(np.nanmean(wasserstein_vals) if wasserstein_vals else np.nan)

        disp_names = [disp_names_all[c] for c in channel_indices]                 
        plot_indices = list(range(len(channel_indices)))         
        # plotting Wasserstein distance per track to compare against multimodal/ATAC-only models

        plt.figure(figsize=(8, 5))     
        bars = plt.bar(plot_indices, wasserstein_per_channel, color='steelblue', edgecolor='black')
        plt.title("Wasserstein Distance per Track (DNA model)")
        plt.xlabel("Track")
        plt.ylabel("Mean Wasserstein Distance (bp)")
        plt.xticks(plot_indices, disp_names, rotation=45, ha='right')
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.1f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"dna_wasserstein_per_track_seed_{self.seed}.png", dpi=300)
        plt.show()

        for name, val in zip(disp_names, wasserstein_per_channel):
            print(f"{name}: Wasserstein Distance = {val:.2f} bp")

        avg_wasserstein = np.nanmean(wasserstein_per_channel)
        print(f"\nAverage Wasserstein Distance across all tracks: {avg_wasserstein:.2f} bp")  

        return {"wasserstein_per_channel": wasserstein_per_channel, "names": disp_names, "avg_wasserstein": avg_wasserstein}



    #JSD stands for Jensen–Shannon divergence. 
    
    def DNA_cal_jsd(self, predictions=None, eps=1e-7, track_numbers=None):                   
        if self.model is None:
            raise ValueError("no model found. Run DNA_model_sequential() first.")
        if self.X_dna_test is None or self.Y_dna_test is None:
            raise ValueError("test splits missing, run Train_test_Split first.")
        if predictions is None:
            predictions = self.model.predict(self.X_dna_test, verbose=1)

        num_channels = self.Y_dna_test.shape[-1]

        if track_numbers is not None:    #  if track_numbers is given, restrict evaluation to that inclusive range e.g. [1,7] = tracks 1 to 7
            
            start, end = track_numbers[0], track_numbers[1]
            if start < 0 or end >= num_channels or start > end:        
                raise ValueError(f"track_numbers {track_numbers} is out of range. Valid range: [0, {num_channels - 1}]")
            channel_indices = list(range(start, end + 1))      
        else:
            channel_indices = list(range(num_channels))     #  if track_numbers=None: fall back to evaluating all channels as before

        base_names = list(self.names[1:]) if self.names is not None and len(self.names) > num_channels else \
                     (list(self.names) if self.names is not None else [])
        disp_names_all = [base_names[c] if c < len(base_names) else f"Track {c}" for c in range(num_channels)]

        jsd_per_channel = []

        for c in channel_indices:   
            #  iterate only over selected channel indices, not necessarily all
            jsd_vals = []
            for i in range(len(self.Y_dna_test)):
                true_profile = np.clip(self.Y_dna_test[i, :, c], 0, None)
                pred_profile = np.clip(predictions[i, :, c], 0, None)

                true_sum = true_profile.sum()
                pred_sum = pred_profile.sum()
                if true_sum < eps:
                    continue   #     skip only windows with no experimental signal

                true_prob = (true_profile + eps) / (true_sum + eps * len(true_profile))
                pred_prob = (pred_profile + eps) / (pred_sum + eps * len(pred_profile))

                jsd_distance = jensenshannon(true_prob, pred_prob, base=2)
                jsd_vals.append(jsd_distance ** 2)                 #   square the distance to get the divergence

            jsd_per_channel.append(np.nanmean(jsd_vals) if jsd_vals else np.nan)

        disp_names = [disp_names_all[c] for c in channel_indices] #  pull only the names that correspond to the selected tracks
        plot_indices = list(range(len(channel_indices)))    #  remap to 0,1,2,... so bar positions are always clean regardless of which subset is shown

        plt.figure(figsize=(8, 5))
        bars = plt.bar(plot_indices, jsd_per_channel, color='mediumpurple', edgecolor='black')
        plt.title("JSD per Track (DNA model )")
        plt.xlabel("Track")
        plt.ylabel("JSD")
        plt.ylim(0, 1)
        plt.xticks(plot_indices, disp_names, rotation=45, ha='right')
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"dna_jsd_per_track_seed_{self.seed}.png", dpi=300)
        plt.show()

        for name, val in zip(disp_names, jsd_per_channel):
            print(f"{name}: JSD = {val:.4f}")

        avg_jsd = np.nanmean(jsd_per_channel)                                                      
        print(f"\nAverage JSD across all tracks: {avg_jsd:.4f}")

        return {"jsd_per_channel": jsd_per_channel, "names": disp_names, "avg_jsd": avg_jsd}

  

'''This function trains and evaluates the DNA-only model repeatedly using every random seed listed in the YAML file. 
It then saves the result for each seed and calculates the mean and standard deviation across seeds.'''

def run_dna_seed_experiments(
    loader,
    config_path="config_file_ml.yaml",
    output_dir="results/seed_runs"
):
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    seeds = yaml_config["params"]["model_seeds"]["dna"]
    batch_size = yaml_config["params"]["batch_size_dna"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    txt_path = output_path / "dna_seed_results.txt"
    csv_path = output_path / "dna_seed_results.csv"

    txt_path.write_text(
        "DNA-only repeated-seed results\n"
        "==============================\n\n",
        encoding="utf-8"
    )
    # evaluation metrics
    fieldnames = [
        "model",
        "seed",
        "weighted_test_loss",
        "test_mae",
        "pooled_pearson",
        "mean_track_pearson",
        "track_pearson_sd",
        "mean_wasserstein_bp",
        "mean_jsd",
        "epochs_completed"
    ]
    all_results = []
    
    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Training DNA-only model with seed {seed}")
        print("=" * 70)

        tf.keras.backend.clear_session()

        runner = DNAPredictionModel.from_loader(loader)
        runner.Train_test_Split(config_path=config_path)
        model, history = runner.DNA_model_sequential(
            input_shape=(None, 4),
            config_path=config_path,
            seed=seed
        )

        test_loss, test_mae = model.evaluate(
            runner.X_dna_test,
            runner.Y_dna_test,
            batch_size=batch_size,
            verbose=1
        )
        predictions = model.predict(
            runner.X_dna_test,
            batch_size=batch_size,
            verbose=1
        )
        pearson = runner.pearson_cor_summary(predictions=predictions)
        wasserstein = runner.DNA_cal_wasserstein(predictions=predictions)
        jsd = runner.DNA_cal_jsd(predictions=predictions)

        result = {
            "model": "DNA-only",
            "seed": int(seed),
            "weighted_test_loss": float(test_loss),
            "test_mae": float(test_mae),
            "pooled_pearson": float(pearson["pooled_pearson"]),
            "mean_track_pearson": float(pearson["mean_pearson"]),
            "track_pearson_sd": float(pearson["std_pearson"]),
            "mean_wasserstein_bp": float(wasserstein["avg_wasserstein"]),
            "mean_jsd": float(jsd["avg_jsd"]),
            "epochs_completed": len(history.history["loss"])
        }
        all_results.append(result)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        with txt_path.open("a", encoding="utf-8") as f:
            f.write(f"Seed: {seed}\n")
            f.write(f"Weighted test loss: {test_loss:.6f}\n")
            f.write(f"Test MAE: {test_mae:.6f}\n")
            f.write(f"Pooled Pearson: {pearson['pooled_pearson']:.6f}\n")
            f.write(f"Mean track Pearson: {pearson['mean_pearson']:.6f}\n")
            f.write(f"Track Pearson SD: {pearson['std_pearson']:.6f}\n")
            f.write(f"Mean Wasserstein: {wasserstein['avg_wasserstein']:.6f} bp\n")
            f.write(f"Mean JSD: {jsd['avg_jsd']:.6f}\n")
            f.write(f"Epochs completed: {len(history.history['loss'])}\n")
            f.write("-" * 40 + "\n")

    metric_names = [
        "weighted_test_loss",
        "test_mae",
        "pooled_pearson",
        "mean_track_pearson",
        "mean_wasserstein_bp",
        "mean_jsd"
    ]
    summary = {}
    for metric in metric_names:
        values = np.asarray([result[metric] for result in all_results], dtype=float)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        }

    with txt_path.open("a", encoding="utf-8") as f:
        f.write("\nOVERALL RESULTS ACROSS SEEDS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Seeds: {seeds}\n")
        for metric in metric_names:
            f.write(
                f"{metric}: {summary[metric]['mean']:.6f} "
                f"± {summary[metric]['sd']:.6f}\n"
            )

    return all_results, summary
