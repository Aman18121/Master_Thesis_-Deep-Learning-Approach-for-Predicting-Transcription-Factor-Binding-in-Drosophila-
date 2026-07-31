'''This python script belongs to 
author = [AMAN YADAV]
script 4/6
as part of Masterthesis [A Multimodal Deep Learning Approach for Predicting Transcription Factor Binding in Drosophila]
'''

import csv
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wasserstein_distance
import yaml
from scipy.spatial.distance import jensenshannon


# random seed function 
# It is done to make sure that the model produce exactly the sam eresult every time its run on a defined seed example = [11 ,22 33 .... etc]
# outside the class because it only handles systematic seed execution and not model training of this py script


def set_global_seed(seed):
    random.seed(seed)             
    np.random.seed(seed)          
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()   
    print(f"seed {seed} has been set for execution.")

    


#====================(ATAC prediction (main model) must be imported before execution ================================================
# requires the following data from the genomic data loader
# combined_data (all chip data combined), chromosome map (window_chroms), ATAC data (X_atac_list), and track names (track_names)...

class ATACPredictionModel:
    
    def __init__(self, combined_data, window_chroms, X_atac_list, track_names=None):  #. required inputs from genomic data loader
        
        self.combined_data = combined_data
        self.window_chroms = np.array(window_chroms) if window_chroms is not None else None
        self.X_atac_list = X_atac_list
        self.names = track_names

        # refer train_test_split function for more details on how the data is split into train, validation, and test sets
        self.X_train_set = None
        self.Y_train_set = None                        
        self.X_validation_set = None
        self.Y_validation_set = None
    
        self.X_test_set = None  
        self.Y_test_set =  None
        
        self.model   = None                                             
        self.history = None
        self.seed = None
        
        print("ATACPrediction model has been initialized with the provided data.")
        

    # geting the data from genomic data loader class and initializing the ATAC model with it
    # this script is designed as a continued pipeline where the genomic data loader prepares the data and this class uses that data for ATAC prediction 
    
    @classmethod
    def from_loader(cls, loader_instance):                            
        
        try:
            combined_data = loader_instance.concatenated_data
            window_chroms = loader_instance.window_chromosomes
            X_atac_list = loader_instance.atac_tracks
            track_names = getattr(loader_instance, "names", None)

            if combined_data is None or window_chroms is None or X_atac_list is None:
                raise ValueError("Loader instance is missing required data attributes. Run loader.read_atac_data() first.") 
        except AttributeError as e:
                raise ValueError(f"Error occurred while accessing loader instance attributes: {e}") from e

        return cls(
            combined_data=combined_data,
            window_chroms=window_chroms,
            X_atac_list=X_atac_list,
            track_names=track_names
        )

    
    # train test split using held out chrms 
    def Train_Test_Split(self, config_path="config_file_ml.yaml", TN5_frags=None):

        
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

            
    
        training_chrms = yaml_config["params"]["training_chrms"]
        testing_chrm = yaml_config["params"]["test_chrms"]
        validation_chrm = yaml_config["params"]["validation_chrms"]

        
        training_chrms = [training_chrms] if isinstance(training_chrms, str) else list(training_chrms)
        testing_chrm = [testing_chrm] if isinstance(testing_chrm, str) else list(testing_chrm)    #. handle both list and string from yaml
        validation_chrm  = [validation_chrm] if isinstance(validation_chrm, str) else list(validation_chrm)       #. handle both list and string from yaml
       
        if self.window_chroms is None :
            raise ValueError("Chromosome map missing. Check add_chrmos() in loader.")
            
        if self.X_atac_list is None or len(self.X_atac_list) == 0:
            raise ValueError("ATAC list is empty.")
            
        if TN5_frags is None:
            raise ValueError("TN5 fragment features are missing. Provide TN5_frags.")
        

        # Tn5 fragmentation data has 3 classification (mono, di and tri nucleosome) which is used as a feature for ATAC prediction model
        
        X_Atac = np.squeeze(self.X_atac_list[0]) # flatten the ATAC data to 2D if it has an extra dimension (
        TN5_frags = np.asarray(TN5_frags)
        
        if X_Atac.ndim == 2:
            X_Atac = np.expand_dims(X_Atac, axis=-1)
            
        if TN5_frags.ndim == 2:
            TN5_frags = np.expand_dims(TN5_frags, axis=-1)
            
        if X_Atac.ndim != 3 or TN5_frags.ndim != 3:
            raise ValueError(f"ATAC + TN5 fragment shapes are not compatible: {X_Atac.shape} vs {TN5_frags.shape}.")
            
        if X_Atac.shape[:2] != TN5_frags.shape[:2]:
            raise ValueError(f"ATAC and TN5 fragment shapes are not same  : {X_Atac.shape} vs {TN5_frags.shape}.")

            
      
        X_ATAC_Tn5_frags = np.concatenate([X_Atac, TN5_frags], axis=-1)   #. combining ATAC + TN5 fragments here  (Windows, 2000, 4)
        
  
        Y_chip = self.combined_data[:, :, 1:]                        
        chrms_array = self.window_chroms.astype(str)

        if X_ATAC_Tn5_frags.shape[0] != Y_chip.shape[0] or Y_chip.shape[0] != chrms_array.shape[0]:
            raise ValueError(
                f"Window counts are mismatched : inputs = {X_ATAC_Tn5_frags.shape[0]}, targets={Y_chip.shape[0]}, chromosomes={chrms_array.shape[0]}."
            )
            

            
        # using training, validation, and test chromosome lists to create boolean masks for splitting the data

        test_mask  = np.isin(chrms_array, testing_chrm)
        val_mask   = np.isin(chrms_array, validation_chrm)
        train_mask = np.isin(chrms_array, training_chrms)                 
        
        # ======== check if Is any genomic window assigned to two data splits? and if any of the splits are empty
        # constant problems gave the idea to add this check to avoid silent errors in training and evaluation.
        # A very important step , if not cheked can creater problems in accuracy or cause error which is hard to find 
        
        if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
            raise ValueError("Training, validation, and test chromosome groups are overlaping , check the data again .")

        if not np.any(train_mask):
            raise ValueError(f"No windows found for training chromosomes: {training_chrms}")
            
        if not np.any(test_mask):
            raise ValueError(f"No windows found for test chromosome: {testing_chrm}")
            
        if not np.any(val_mask):
            raise ValueError(f"No windows found for validation chromosome: {validation_chrm}")
            
        # ATAC + Tn5 frags = features
        # chip = prediction output
        
        self.X_train_set, self.Y_train_set = X_ATAC_Tn5_frags[train_mask], Y_chip[train_mask]
        self.X_validation_set,   self.Y_validation_set   = X_ATAC_Tn5_frags[val_mask],   Y_chip[val_mask]
        self.X_test_set,  self.Y_test_set  = X_ATAC_Tn5_frags[test_mask],  Y_chip[test_mask]
        
     
        # very importnat to check the shape compatibility before feeding it in the cnn model.
        print(f"\nSplit done:")
        print(f"  Train : X{self.X_train_set.shape} | Y{self.Y_train_set.shape}")
        print(f"  Val   : X{self.X_validation_set.shape}   | Y{self.Y_validation_set.shape}  [{validation_chrm}]")
        print(f"  Test  : X{self.X_test_set.shape}  | Y{self.Y_test_set.shape} [{testing_chrm}]")
        return (self.X_train_set, self.Y_train_set), (self.X_validation_set, self.Y_validation_set), (self.X_test_set, self.Y_test_set  )
    

    # model sequnctial (1d cnn)
    def model_sequential(self, input_shape=(None, 4) , config_path="config_file_ml.yaml", seed =None):
        
        if self.X_train_set is None or self.Y_train_set is None:              
            raise ValueError("spliting is peding , check the train , test and val data. run Train_Test_Split first.")
            
        if self.X_train_set.shape[-1] != input_shape[-1]:
            raise ValueError(f"Training data shape {self.X_train_set.shape[-1]} does not match input shape {input_shape[-1]}.")

       
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

            # hyperparameter tuning from config file

            epochs = yaml_config["params"]["epochs_ATAC"]
            batch_size = yaml_config["params"]["batch_size_atac"]
            atac_callback_cfg = yaml_config["params"]["atac_callbacks"]
            earlystop_patience = atac_callback_cfg["atac_early_stop_patience"]
            patience = atac_callback_cfg["atac_reduce_lr_patience"]
            lr_factor   = atac_callback_cfg["atac_lr_decay_factor"]
            atac_loss_cfg = yaml_config["params"]["atac_loss_tuning"]
            peak_threshold  = atac_loss_cfg["atac_peak_threshold"]
            peak_multiplier = atac_loss_cfg["atac_peak_weight_multiplier"]
            background_penalty  = atac_loss_cfg["atac_background_penalty"]
            configured_seeds = yaml_config["params"]["model_seeds"]["atac"]  
            

        # must provide a seed for reproducibility, either from the config or as an argument
        # its recommended to use config file for seed selection to maintain consistency across runs and experiments
        try:
            if seed is None:
                seed = configured_seeds[0]  # Use the first seed from the config if none provided
            elif seed not in configured_seeds:
                raise ValueError(f"Seed {seed} not in configured seeds: {configured_seeds}")
        except KeyError:
            raise ValueError("Configured seeds for ATAC model not found in the config file.")
        

        set_global_seed(seed)
        self.seed = seed

        # Flexible input shape so we can change window size or number of input channels later
        # its be done to mnatain flexibility across adding or removing data..
        
        total_targets = self.Y_train_set.shape[-1]                         
        
        
        inputs = tf.keras.Input(shape=input_shape)               

        atac_tn5_features = tf.keras.layers.Conv1D(64, kernel_size=15, padding='same')(inputs)
        atac_tn5_features = tf.keras.layers.BatchNormalization()(atac_tn5_features)
        atac_tn5_features = tf.keras.layers.Activation('relu')(atac_tn5_features)

        # Level 2: Receptive Field Scaling (Rate = 2)
        atac_tn5_features = tf.keras.layers.Conv1D(128, kernel_size=15, padding='same', dilation_rate=2)(atac_tn5_features)  #. expanding context window
        atac_tn5_features = tf.keras.layers.BatchNormalization()(atac_tn5_features)
        atac_tn5_features = tf.keras.layers.Activation('relu')(atac_tn5_features)

        # Level 3: Broad Context Capture (Rate = 4)
        atac_tn5_features = tf.keras.layers.Conv1D(128, kernel_size=21, padding='same', dilation_rate=4)(atac_tn5_features)  #. wider context
        atac_tn5_features = tf.keras.layers.BatchNormalization()(atac_tn5_features)
        atac_tn5_features = tf.keras.layers.Activation('relu')(atac_tn5_features)

        # Level 4: Distant Flanking Window (Rate = 8)
        atac_tn5_features = tf.keras.layers.Conv1D(128, kernel_size=21, padding='same', dilation_rate=8)(atac_tn5_features)
        atac_tn5_features = tf.keras.layers.BatchNormalization()(atac_tn5_features)
        atac_tn5_features = tf.keras.layers.Activation('relu')(atac_tn5_features)

        # Resolution Normalization Layer
        atac_tn5_features = tf.keras.layers.Conv1D(
            64, kernel_size=5, padding='same', dilation_rate=1)(atac_tn5_features)  #. smoothing before output
        atac_tn5_features = tf.keras.layers.BatchNormalization()(atac_tn5_features)
        atac_tn5_features = tf.keras.layers.Activation('relu')(atac_tn5_features)

        outputs = tf.keras.layers.Conv1D(filters=total_targets, kernel_size=1,
            activation='linear',
            padding='same')(atac_tn5_features)

        model = tf.keras.Model(inputs, outputs)

        

        # Weighted MSE – focuses on peaks and down-weights background (normal MSE performed poorly)
        # after trial and error, normal mse was not sufficient for accurate peak prediction, hence the weighted approach was implemented to improve model performance on biologically relevant regions.
        
        if peak_threshold is None or peak_multiplier is None or background_penalty is None:
            raise ValueError("ATAC loss configuration contains a missing value.")

        def ATAC_weighted_mse(y_true, y_pred):

            weights = tf.where(
                y_true >= peak_threshold, 
                peak_multiplier, 
                tf.where(y_true < 0.01, background_penalty, 1.0)
            )
            
            squared_err = tf.square(y_true - y_pred)
            return tf.reduce_mean(tf.reduce_mean(squared_err * weights, axis=[1, 2]))

       
        model.compile(optimizer='adam', loss=ATAC_weighted_mse, metrics=['mae'])

        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience= earlystop_patience, # if validation loss does not improve for this many epochs, training will stop early (overfitting prevention)
            restore_best_weights= True
        )

        checkpoint_dir = Path("checkpoints") / "atac_tn5" / f"seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        #saving the model with best scores
        checkpoint = tf.keras.callbacks.ModelCheckpoint(                   
            filepath=str(checkpoint_dir / "best_atac_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )


        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=patience,
            min_lr=1e-6,
            verbose=1
        )

        print("Training ATAC model... GPU acceleration is recommended for faster training. refer the user manual for this thesis to setup GPU for training.")
        print(f" Validation chromosome: {yaml_config['params']['validation_chrms']} | Test chromosome: {yaml_config['params']['test_chrms']}")
        history = model.fit(
            self.X_train_set, self.Y_train_set,
            validation_data=(self.X_validation_set, self.Y_validation_set),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr]
        )

        self.model   = model
        self.history = history
        return model, history

    
    # pearson corelation summary 
    # contaiss pooled pearson, mean pearson and std of pearson across all chip tracks
    # its execution take place in ATAC_Evaluation function after the model evaluation and prediction is done on test data
        
    def ATAC_pearson_corr(self, predictions=None, target_channels=None):

        try:
            if self.model is None or self.X_test_set is None or self.Y_test_set is None:
                raise ValueError("Model or test data not initialized. Run model_sequential and Train_Test_Split first.")
            if predictions is None:
                        print("Predictions not provided. Generating predictions using the model on the test set.")
                        predictions = self.model.predict(self.X_test_set)
        except AttributeError :
            raise ValueError(f"Error occurred while accessing model or test data attributes: {self.model}, {self.X_test_set}, {self.Y_test_set}")

        

        total_channels = self.Y_test_set.shape[-1]

        # if target channel == None then it will take all the chip tracks for pearson calculation
        # not recommended to take all tracks if there are too many chip tracks because it can cause memory issues and crash the kernel
        # faced by the author during the thesis
        if target_channels is None:
            target_channels = list(range(total_channels))
            
        elif isinstance(target_channels, (int, np.integer)):
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

        ATAC_pearson_scores = []
        ATAC_true = []
        ATAC_pred = []

        max_windows = self.Y_test_set.shape[0]

        for ch in target_channels:

            true_values = self.Y_test_set[:max_windows, :, ch].flatten()
            pred_values = predictions[:max_windows, :, ch].flatten()


            #====pearson calculation ================================
            
            if np.std(true_values) > 1e-6 and np.std(pred_values) > 1e-6:
                r_value, _ = pearsonr(true_values, pred_values)
            else:
                # not a number
                r_value = np.nan

            ATAC_pearson_scores.append(r_value)
            ATAC_true.append(true_values)
            ATAC_pred.append(pred_values)
            

        pooled_atac_true = np.concatenate(ATAC_true)
        pooled_atac_pred = np.concatenate(ATAC_pred)
        

        if np.std(pooled_atac_true) > 1e-6 and np.std(pooled_atac_pred) > 1e-6:
            pooled_r, _ = pearsonr(pooled_atac_true, pooled_atac_pred)
        else:
            pooled_r = np.nan

            
        mean_r = np.nanmean(ATAC_pearson_scores)
        std_r = np.nanstd(ATAC_pearson_scores)

        # scatter plot of pooled true vs predicted values 
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        stride = max(1, len(pooled_atac_true) // 10000)

        axes[0].scatter(
            pooled_atac_true[::stride],
            pooled_atac_pred[::stride],
            s=1,
            alpha=0.1,
            color="teal"
        )

        axis_min, axis_max = 0, 0.7   #. fixed display range, independent of outlier signal values

        axes[0].plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="orange",
            label="Identity line"
        )

        axes[0].set_xlim(axis_min, axis_max)
        axes[0].set_ylim(axis_min, axis_max)

        axes[0].set_title(
            f"ATAC Model: Pooled Pearson r = {pooled_r:.4f}", fontsize=10
        )
        axes[0].set_xlabel("True Signal")
        axes[0].set_ylabel("Predicted Signal")
        axes[0].legend()
        axes[0].alpha = 0.3
        axes[0].grid(alpha=0.3)

        # track wise pearson bar plot
        # recommended to use limit or target channels if there are too many chip tracks because it can cause memory issues and crash the kernel
        # its sufficient for this thesis to use only 5-10 chip tracks for pearson calculation and visualization

        bars = axes[1].bar(range(len(target_channels)), ATAC_pearson_scores,
            edgecolor="black",
            alpha=0.8
        )

        for bar, score in zip(bars, ATAC_pearson_scores):

            if not np.isnan(score):

                text_position = score + 0.02 if score >= 0 else score - 0.06

                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    text_position,
                    f"{score:.3f}",
                    ha="center"
                )
        
        axes[1].axhline(
            mean_r,
            linestyle="--",
            color="cyan",
            label=f"Mean r = {mean_r:.3f}"
        )
        

        axes[1].set_title(
            f"ATAC Model: Pearson per Track\n"
            f"Mean r = {mean_r:.4f} ± {std_r:.4f}", fontsize=10
        )
        
        axes[1].set_xlabel("Track")
        axes[1].set_ylabel("Pearson r")
        axes[1].set_xticks(range(len(target_channels)))
        axes[1].set_xticklabels(target_channels)
        axes[1].set_ylim(-0.1, 1)
        axes[1].set_xlim(-0.5, len(target_channels) - 0.5)   # This was kept by default, it can be chnaged accordingly to a diff data
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\nPearson correlation per track:")

        for ch, score in zip(target_channels, ATAC_pearson_scores):
            print(f"Track {ch}: r = {score:.4f}")

        print(f"\nPooled Pearson: {pooled_r:.4f}")
        print(f"Mean Pearson: {mean_r:.4f}")
        print(f"Standard deviation: {std_r:.4f}")

        return {
            "pearson_per_track": ATAC_pearson_scores,
            "pooled_pearson": pooled_r,
            "mean_pearson": mean_r,
            "std_pearson": std_r
        }


    #full trained  model evaluation 
    # This function bellow wil evaluate metrics like MAE , MSE , pearson summary and visual comparisions between chip  and atac true vs pred datas 
    #  it is programed to take target channels in case there are lot of chip data which can crash the kernel because of oom error ...

    
    def ATAC_Evaluation(self, target_channel=None, window_number=None):
        
        try:
            if self.model is None or self.X_test_set is None or self.Y_test_set is None:
                raise ValueError("Model or test data not initialized. Run model_sequential and Train_Test_Split first.")
        except AttributeError:
            raise ValueError(f"Error occurred while accessing model or test data attributes: {self.model}, {self.X_test_set}, {self.Y_test_set}")

        print("Evaluating ATAC model on the held-out test chromosomes...")
        test_metrics = self.model.evaluate(self.X_test_set, self.Y_test_set, batch_size=64, verbose=1)

        for name, val in zip(self.model.metrics_names, test_metrics):
            print(f"  Test {name.upper()}: {val:.5f}")

        #==============predict on held out chrms========================
        preds = self.model.predict(self.X_test_set)                    #. actual evaluation of predicted vs real data
        total_channels = self.Y_test_set.shape[-1]

        if target_channel is None:
            target_channels = list(range(total_channels))
            
        elif isinstance(target_channel, (int, np.integer)):
            target_channels = [int(target_channel)]
            
        elif isinstance(target_channel, (list, tuple, range, np.ndarray)):
            target_channels = [int(ch) for ch in target_channel]
            
        else:
            raise ValueError("target_channel must be None, an int, list, tuple, range, or NumPy array.")

        # channel validation
        invalid_channels = [ch for ch in target_channels if ch < 0 or ch >= total_channels]
        
        if invalid_channels:
            raise ValueError(f"Invalid target channels {invalid_channels}. Valid range is 0 to {total_channels - 1}.")
            

        #===========get names for evry track rom config if exist ======================================
        if hasattr(self, "names") and self.names is not None:
            
            raw_names = self.names[1:] if len(self.names) > total_channels else self.names
            
            chan_names = [
                raw_names[c] if c < len(raw_names) else f"Channel {c}"
                for c in range(total_channels)
            ]
            
        else:
            chan_names = [f"Channel {c}" for c in range(total_channels)]

        # . loss curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'],     label='Train Loss', color='royalblue', lw=2)
        plt.plot(self.history.history['val_loss'], label='Val Loss',   color='darkorange', lw=2)
        plt.title('Genomic Weighted MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)


        
        plt.subplot(1, 2, 2)                                            #. MAE evaluator across epochs
        plt.plot(self.history.history['mae'],     label='Train MAE', color='royalblue', lw=2)
        plt.plot(self.history.history['val_mae'], label='Val MAE',   color='darkorange', lw=2)
        plt.title('Mean Absolute Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        

        # converts window_indices into a valid list of test-window numbers.
        
        if window_number is None:
            print("No specific window number provided. Defaulting to the first 3 test windows.")
            window_indices = list(range(min(3, self.X_test_set.shape[0])))
            
        elif isinstance(window_number, (int, np.integer)):
            print(f" plotting for window number {window_number} only.")
            window_indices = [int(window_number)]
            
        else:
            print(f"Plotting for multiple window numbers: {window_number}")
            window_indices = [int(idx) for idx in window_number]

        window_indices = [
            idx for idx in window_indices
            if 0 <= idx < self.X_test_set.shape[0]
        ]

        try:
            if not window_indices:
                raise ValueError("No valid window indices found. Check the provided window_number(s).")
        except ValueError as e:
            print(f"Error: {e}")
            return  

            

        for windows in window_indices:
            total_channels = len(target_channels)
            
            fig, axes = plt.subplots(total_channels, 1, figsize=(14, 3 * total_channels), sharex=True)
            axes = np.atleast_1d(axes)
            
            positions = np.arange(self.Y_test_set.shape[1])

            for ax, ch in zip(axes, target_channels):
                true_profile = self.Y_test_set[windows, :, ch]
                pred_profile = preds[windows, :, ch]

                ax.plot(true_profile, label='True', color='royalblue', alpha=0.8, lw=1.5)
                ax.plot(pred_profile, label='Predicted', color='darkorange', linestyle='--', lw=1.5)
                ax.fill_between(positions, true_profile, color='royalblue', alpha=0.15)
                ax.set_title(f"{chan_names[ch]} | Test Window {windows}")
                ax.set_ylabel("Signal")
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Window Position (bp)")
            fig.suptitle(f"True and Predicted ChIP-seq Profiles | Test Window {windows}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

        self.ATAC_pearson_corr(
            predictions=preds,
            target_channels=target_channels
        )

        # --- MAE per channel bar plot ---
        plt.figure()             #. MAE per chip track (useful for excluding bad SRRs)
        
        mae_per_chip = [np.mean(np.abs(self.Y_test_set[:, :, c] - preds[:, :, c])) for c in target_channels]
        
        bars = plt.bar(range(len(target_channels)), mae_per_chip, 
                       color='blue', 
                       edgecolor='black', alpha=0.8)

        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, h + (max(mae_per_chip) * 0.02),
                     f"{h:.4f}", ha='center', va='bottom', fontsize=9)

        plt.title("MAE Across ChIP Tracks")
        plt.xlabel("ChIP Track")
        plt.ylabel("MAE")
        plt.xticks(range(len(target_channels)), [chan_names[c] for c in target_channels], rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        return self.model, self.history


        
    # 3 best and 2 worst fitting profiles of true and pred values =====(not a model evaluation metrics)!!!
        
    def plot_best_worst_profiles(self, predictions=None, target_channel=None , custom_names=None):   

        try:
            if self.model is None or self.X_test_set is None or self.Y_test_set is None:
                raise ValueError("Model or test data not initialized. Run model_sequential and Train_Test_Split first.")    
        except AttributeError:
            raise ValueError(f"Error occurred while accessing model or test data attributes: {self.model}, {self.X_test_set}, {self.Y_test_set}")
        
            

        if predictions is None:
            print("Predictions not provided. Generating predictions using the model on the test set.")
            predictions = self.model.predict(self.X_test_set, verbose=1)

        total_channels = self.Y_test_set.shape[-1]

        # =======Normalize target_channel to a clean list of valid ints, regardless of what's passed in=================
        # very heavy on memory (authors recommendation) to use only 5-10 chip tracks for visualization, otherwise it can cause memory issues and crash the kernel.
        if target_channel is None:
            print("No specific target provided. using  all channels.")
            rank_channels = list(range(total_channels))
            
        elif isinstance(target_channel, (int, np.integer)):
            print(f"target identified , plotting for channel {target_channel} only.")
            rank_channels = [int(target_channel)]
            
        elif isinstance(target_channel, (list, tuple, range, np.ndarray)):
            print(f"multiple target channels identified , plotting for channels {target_channel}.")
            rank_channels = [int(c) for c in target_channel]
            
        else:
            raise ValueError("target_channel must be None, an int, list, tuple, range, or NumPy array.")

        invalid_number = [c for c in rank_channels if c < 0 or c >= total_channels]
        
        if invalid_number:
            raise ValueError(f"Invalid channel {invalid_number}. Valid range is 0 to {total_channels - 1}.")


        total_rows = len(rank_channels)                                 
        color_map = plt.cm.get_cmap("tab20", total_rows)
        colors = [color_map(i) for i in range(total_rows)]

        if custom_names is not None:
            print("identfied custom names in config file. Using custom names for channels.")
            base_names = list(custom_names)
        elif hasattr(self, 'names') and self.names is not None:
            print("Using names from loader instance for channels.")
            base_names = list(self.names[1:]) if len(self.names) > total_channels else list(self.names)
        else:
            print("No custom names or loader names found. Using default channel names.")
            base_names = []

        # create display names for channels, using base_names if available, otherwise defaulting to "Track {index}"
        disp_names_all = [base_names[r] if r < len(base_names) else f"Track {r}" for r in range(total_channels)]
        disp_names = [disp_names_all[c] for c in rank_channels]         

        # --- Ranking: average Pearson r across the requested channel(s) per window ---

        scores = []                                                      
        for i in range(len(self.Y_test_set)):
            r_per_channel = []
            for c in rank_channels:
                t = self.Y_test_set[i, :, c]    # true signal for window i, channel c
                p = predictions[i, :, c]     # predicted signal for window i, channel c
                r_val = pearsonr(t, p)[0] if np.std(t) > 1e-6 and np.std(p) > 1e-6 else -1.0
                r_per_channel.append(r_val)
            scores.append((i, np.mean(r_per_channel)))                 

        print(f"Total windows evaluated: {len(scores)}")
        scores.sort(key=lambda x: x[1], reverse=True)

        
        best_windows  = scores[:min(3, len(scores))]
        worst_windows = scores[-min(2, len(scores)):]
        selected_windows  = best_windows + worst_windows                                 

        selected_indices = [s[0] for s in selected_windows]
        selected_corrs   = [s[1] for s in selected_windows]

        number_columns = len(selected_windows)
        fig, axes = plt.subplots(total_rows, number_columns, figsize=(5 * number_columns, 3.3 * total_rows), sharex='col')  
        if total_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if number_columns == 1:
            axes = np.expand_dims(axes, axis=1)

        
        # Visual comparison only (best/worst windows) – not a formal evaluation metric
        # becuase it is only based on a few windows and not the entire test set. 
        for col_idx, win_val in enumerate(selected_indices):
            r        = selected_corrs[col_idx]
            best_index  = col_idx < len(best_windows)
            col_lbl  = "BEST" if best_index else "WORST"
            col_clr  = "forestgreen" if best_index else "firebrick"


            for row_idx, c in enumerate(rank_channels):                 
                axis        = axes[row_idx, col_idx]
                true_signals = self.Y_test_set[win_val, :, c]
                pred_signals = predictions[win_val, :, c]

                axis.fill_between(range(len(true_signals)), true_signals, color='gray', alpha=0.1)
                axis.plot(true_signals, color='black', linewidth=1, linestyle='--', alpha=0.4, label='True')
                axis.plot(pred_signals, color=colors[row_idx % len(colors)], linewidth=1.8, label='Pred')

                if col_idx == 0:
                    axis.set_ylabel(disp_names[row_idx], fontweight='bold', rotation=0, labelpad=45, va='center', fontsize=12)

                ymax = max(np.max(true_signals), np.max(pred_signals), 0.1)
                axis.set_ylim(-0.02, ymax * 1.4)
                axis.text(0.95, 0.05, f"r={r:.2f}", transform=axis.transAxes,
                          ha='right', fontsize=10, fontweight='bold', color=col_clr)

                if row_idx == 0:
                    axis.set_title(f"{col_lbl}\nWindow: {win_val}", fontsize=15, pad=15, fontweight='bold', color=col_clr)
                if row_idx < (total_rows - 1):
                    axis.set_xticks([])
                else:
                    axis.set_xlabel("Position (bp)", fontsize=11)
                if col_idx == 0 and row_idx == 0:
                    axis.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92 if total_rows > 3 else 0.82, hspace=0.25)
        plt.suptitle("ATAC Profile Predictions: Best vs Worst Windows", fontsize=26, fontweight='bold')
        plt.show()


    # Wasserstein distance calculation for ATAC + Tn5 model evaluation 
    # how accurately the ATAC + Tn5 model places the predicted ChIP-seq signal along each 2,000-bp test window.?
    #0 bp   = perfect spatial agreement
    #20 bp  = small spatial mismatch
    #150 bp = larger spatial mismatch
  
    def ATAC_cal_wasserstein(self, predictions=None, eps=1e-7, track_numbers=None):         

        if self.model is None:
            raise ValueError("no model found. Run model_sequential() first.")
        if self.X_test_set is None or self.Y_test_set is None:
            raise ValueError("test splits missing, run Train_Test_Split first.")
        if predictions is None:
            print("Predictions not provided. Generating predictions using the model on the test set.")
            predictions = self.model.predict(self.X_test_set)   

       
        total_channels = self.Y_test_set.shape[-1]

        if track_numbers is not None:                                                            
            start, end = track_numbers[0], track_numbers[1]
            if start < 0 or end >= total_channels or start > end:                                  
                raise ValueError(
                    f"track_numbers {track_numbers} is out of range. "
                    f"Valid range: [0, {total_channels - 1}]"
                )
            channel_indices = list(range(start, end + 1))                                        
        else:
            channel_indices = list(range(total_channels))                                          


        base_names = list(self.names[1:]) if self.names is not None and len(self.names) > total_channels else \
                     (list(self.names) if self.names is not None else [])
        
        all_names = [base_names[c] if c < len(base_names) else f"Track {c}" for c in range(total_channels)]

        # palce holder for all the chip wasserstien output 
        wasserstein_per_channel = []
         # iterate only over selected channel indices, not necessarily all
        for c in channel_indices:       
            wasserstein_vals = []

            for i in range(len(self.Y_test_set)):                                                     
                # removes negative values, and calculates the total signal in each profile.
                #distribution weights cannot be negative.
                true_profile = np.clip(self.Y_test_set[i, :, c], 0, None)
                pred_profile = np.clip(predictions[i, :, c], 0, None)

                true_sum = true_profile.sum()  #.total experimental signal
                pred_sum = pred_profile.sum()  #.total pred signal

                if true_sum < eps:
                    continue   # skip only windows with no experimental signal

                # genomic position axis
                num_positions = len(true_profile)
                positions = np.arange(num_positions, dtype=np.float64) # 1-bp genomic positions within the window

                #====Converting the true profile into a probability distribution
                true_prob = (true_profile + eps) / (
                    true_sum + eps * num_positions
                )
                pred_prob = (pred_profile + eps) / (
                    pred_sum + eps * num_positions
                )

                wasserstein_value = wasserstein_distance(
                    positions,
                    positions,
                    u_weights=true_prob,
                    v_weights=pred_prob
                )       
                
                #. distance in bp between the true and predicted signal distributions

                wasserstein_vals.append(wasserstein_value)

            wasserstein_per_channel.append(
                np.nanmean(wasserstein_vals) if wasserstein_vals else np.nan
            )

        disp_names = [all_names[c] for c in channel_indices]                                
        plot_indices = list(range(len(channel_indices)))       

        # plotting Wasserstein distance per track to compare against multimodal/DNA-only models
        
        plt.figure(figsize=(8, 5))                                                             
        bars = plt.bar(
            plot_indices,
            wasserstein_per_channel,
            color="steelblue",
            edgecolor="black"
        )
        plt.title("Wasserstein Distance per Track (ATAC model)")
        plt.xlabel("Track")
        plt.ylabel("Wasserstein Distance (bp)")
        plt.xticks(plot_indices, disp_names, rotation=45, ha="right")

        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        plt.tight_layout()
        plt.savefig(f"atac_wasserstein_per_track_seed_{self.seed}.png", dpi=300)
        plt.show()

        for name, val in zip(disp_names, wasserstein_per_channel):
            print(f"{name}: Wasserstein Distance = {val:.2f} bp")

        avg_wasserstein = np.nanmean(wasserstein_per_channel)

        print(
            f"Avg Wasserstein Distance across all chip tracks: "
            f"{avg_wasserstein:.2f} bp"
        )                                                                                         

        return {
            "wasserstein_per_channel": wasserstein_per_channel,
            "names": disp_names,
            "avg_wasserstein": avg_wasserstein
        }



    #JSD stands for Jensen–Shannon divergence. It measures how similar the predicted ChIP-seq signal distribution is to the experimental distribution.
    # JSD evaluation is important because it quantifies how well the model captures the true signal patterns, 
    # which is critical for understanding the biological relevance of the predictions.

    def ATAC_cal_jsd(self, predictions=None, eps=1e-7, track_numbers=None):                           

        if self.model is None:
            raise ValueError("no model found. Run model_sequential() first.")
        if self.X_test_set is None or self.Y_test_set is None:
            raise ValueError("test splits missing, run Train_Test_Split first.")
        if predictions is None:
            predictions = self.model.predict(self.X_test_set, verbose=1)

        total_channels = self.Y_test_set.shape[-1]

        if track_numbers is not None:                                                           
            start, end = track_numbers[0], track_numbers[1]
            print(f"Calculating JSD for tracks {start} to {end}...")

            if start < 0 or end >= total_channels or start > end:                                  
                raise ValueError(f"track_numbers {track_numbers} is out of range. Valid range: [0, {total_channels - 1}]")
            channel_indices = list(range(start, end + 1))     

        else:
            print("No specific track range provided. Calculating JSD for all tracks.")
            channel_indices = list(range(total_channels))                                          

        base_names = list(self.names[1:]) if self.names is not None and len(self.names) > total_channels else \
                     (list(self.names) if self.names is not None else [])
        
        all_names = [base_names[c] if c < len(base_names) else f"Track {c}" for c in range(total_channels)]

        jsd_per_channel = []

        for c in channel_indices:                                                                 
            jsd_vals = []
            for i in range(len(self.Y_test_set)):
                true_profile = np.clip(self.Y_test_set[i, :, c], 0, None)
                pred_profile = np.clip(predictions[i, :, c], 0, None)

                true_chip_sum = true_profile.sum()
                pred_sum = pred_profile.sum()
                if true_chip_sum < eps:
                    continue                                                      

                #probability distributions for true and predicted profiles, adding a small epsilon to avoid division by zero
                true_prob = (true_profile + eps) / (true_chip_sum + eps * len(true_profile))
                pred_prob = (pred_profile + eps) / (pred_sum + eps * len(pred_profile))

                
                jsd_distance = jensenshannon(true_prob, pred_prob, base=2)
                jsd_vals.append(jsd_distance ** 2)                                

            jsd_per_channel.append(np.nanmean(jsd_vals) if jsd_vals else np.nan)

        disp_names = [all_names[c] for c in channel_indices]                               
        plot_indices = list(range(len(channel_indices)))                                        



        plt.figure(figsize=(8, 5))
        bars = plt.bar(plot_indices, jsd_per_channel, color='mediumpurple', edgecolor='black')
        plt.title("JSD per Track (ATAC model)")
        plt.xlabel("Track")
        plt.ylabel("JSD")
        plt.ylim(0, 1)
        plt.xticks(plot_indices, disp_names, rotation=45, ha='right')
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(f"atac_jsd_per_track_seed_{self.seed}.png", dpi=300)
        plt.show()

        for name, val in zip(disp_names, jsd_per_channel):
            print(f"{name}: JSD = {val:.4f}")

        avg_jsd = np.nanmean(jsd_per_channel)                                     
        print(f"\nAverage JSD across all tracks: {avg_jsd:.4f}")

        return {"jsd_per_channel": jsd_per_channel, "names": disp_names, "avg_jsd": avg_jsd}

    

# seed experiments for ATAC + Tn5 model 
# modifiable for different seeds and batch sizes in the config file.
def run_atac_seed_experiments(
    loader,
    TN5_frags,
    config_path="config_file_ml.yaml",
    output_dir="results/seed_runs"
):
    
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    seeds = yaml_config["params"]["model_seeds"]["atac"]
    batch_size = yaml_config["params"]["batch_size_atac"]

    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    txt_path = output_path / "atac_tn5_seed_results.txt"
    csv_path = output_path / "atac_tn5_seed_results.csv"

    txt_path.write_text(
        "ATAC + Tn5 repeated-seed results\n"
        "================================\n\n",
        encoding="utf-8"
    )

    # summary CSV file for all seeds, with columns for each metric
    # Csv file will be rewritten if the script is interrupted and restarted, but all completed seeds will be preserved.
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
        print(f"Traning ATAC started on {seed}")
        print("=" * 70)

        #clear trash from previous runs to avoid memory issues
        tf.keras.backend.clear_session()

        
        runner = ATACPredictionModel.from_loader(loader)
        runner.Train_Test_Split(
            config_path=config_path,
            TN5_frags=TN5_frags
        )
        model, history = runner.model_sequential(
            input_shape=(None, 4),
            config_path=config_path,
            seed=seed
        )

        
        test_loss, test_mae = model.evaluate(
            runner.X_test_set,
            runner.Y_test_set,
            batch_size=batch_size,
            verbose=1
        )


        # predict on the test set and calculate evaluation metrics
        predictions = model.predict(
            runner.X_test_set,
            batch_size=batch_size,
            verbose=1
        )
        pearson = runner.ATAC_pearson_corr(predictions=predictions)
        wasserstein = runner.ATAC_cal_wasserstein(predictions=predictions)
        jsd = runner.ATAC_cal_jsd(predictions=predictions)

        # result dictionary for the current seed, to be appended to the CSV and TXT files
        # will be displayed in the console and saved to the output directory for later analysis (author - a path must be provided in the code itself )
        result = {
            "model": "ATAC + Tn5",
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

        #
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        # report results for the current seed to the TXT file
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

    #overall summary across all seeds, including mean and standard deviation for each metric
    metric_names = [
        "weighted_test_loss",
        "test_mae",
        "pooled_pearson",
        "mean_track_pearson",
        "mean_wasserstein_bp",
        "mean_jsd"
    ]

    # seprate summary dictionary to hold mean and standard deviation for each metric across all seeds
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
