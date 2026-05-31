"""
04_train_baseline_per_basin.py
Entraînement LSTM baseline - UN MODÈLE PAR BASSIN
Avec NSE affiché, checkpoint pour reprise, et visualisations
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l2
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
import gc
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

current_dir = Path(__file__).parent
project_root = current_dir.parent

baseline_dir = project_root / "baseline_data_48_per_basin"
models_dir = project_root / "models_48_per_basin"
results_dir = project_root / "results_48_per_basin"
plots_dir = results_dir / "plots"

for d in [models_dir, results_dir, plots_dir]:
    d.mkdir(exist_ok=True, parents=True)

# Fichier de progression
PROGRESS_FILE = results_dir / "progress.json"

BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
RANDOM_SEED = 42

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = [32, 16]
DROPOUT_RATES = [0.3, 0.3, 0.2]
LEARNING_RATE = 0.001
L2_REGULARIZATION = 0.001

print("="*70)
print("ENTRAÎNEMENT LSTM BASELINE - UN MODÈLE PAR BASSIN")
print("="*70)

# ============================================
# FONCTIONS
# ============================================

def calculate_nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 2 or np.std(o) < 1e-8:
        return -9999.0
    return 1 - np.sum((o - s)**2) / np.sum((o - np.mean(o))**2)

def calculate_kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[mask], sim[mask]
    if len(o) < 2:
        return -9999.0
    r = np.corrcoef(o, s)[0, 1] if np.std(o) > 0 and np.std(s) > 0 else 0
    alpha = np.std(s) / (np.std(o) + 1e-8)
    beta = np.mean(s) / (np.mean(o) + 1e-8)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return []

def save_progress(basin_id):
    completed = load_progress()
    if basin_id not in completed:
        completed.append(basin_id)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(completed, f)

def build_model(input_shape):
    model = Sequential(name="LSTM_Baseline")
    model.add(LSTM(LSTM_UNITS_1, return_sequences=True,
                   input_shape=input_shape,
                   kernel_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_RATES[0]))
    model.add(LSTM(LSTM_UNITS_2, return_sequences=False,
                   kernel_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_RATES[1]))
    for units in DENSE_UNITS:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    return model

# ============================================
# CHARGEMENT DES BASSINS
# ============================================

with open(baseline_dir / "basin_list.txt") as f:
    all_basin_ids = [line.strip() for line in f]

completed = load_progress()
remaining = [b for b in all_basin_ids if b not in completed]

print(f"Total: {len(all_basin_ids)} | Déjà traités: {len(completed)} | Restants: {len(remaining)}")

if not remaining:
    print("Tous les bassins sont déjà traités.")
    exit(0)

# ============================================
# BOUCLE PAR BASSIN
# ============================================

all_results = []

for i, basin_id in enumerate(remaining):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(remaining)}] Bassin {basin_id}")
    print(f"{'='*60}")
    
    try:
        # Chargement
        X_basin = np.load(baseline_dir / f"basin_{basin_id}" / "X.npy")
        y_basin = np.load(baseline_dir / f"basin_{basin_id}" / "y.npy")
        
        n_total = len(X_basin)
        if n_total < 500:
            print(f"Trop peu de séquences ({n_total}), skip")
            save_progress(basin_id)
            continue
        
        # Split temporel
        train_end = int(n_total * (1 - VALIDATION_SPLIT - TEST_SPLIT))
        val_end = int(n_total * (1 - TEST_SPLIT))
        
        X_train, y_train = X_basin[:train_end], y_basin[:train_end]
        X_val, y_val = X_basin[train_end:val_end], y_basin[train_end:val_end]
        X_test, y_test = X_basin[val_end:], y_basin[val_end:]
        
        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        # Modèle
        model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        basin_model_dir = models_dir / f"basin_{basin_id}"
        basin_model_dir.mkdir(exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
            ModelCheckpoint(
                filepath=str(basin_model_dir / 'best_model.keras'),
                monitor='val_loss', save_best_only=True, verbose=0
            ),
            ModelCheckpoint(
                filepath=str(basin_model_dir / 'checkpoint.weights.h5'),
                monitor='val_loss', save_best_only=False, save_weights_only=True, verbose=0
            ),
            CSVLogger(filename=str(basin_model_dir / 'training_log.csv'), append=False)
        ]
        
        # Reprise depuis checkpoint
        checkpoint_path = basin_model_dir / 'checkpoint.weights.h5'
        if checkpoint_path.exists():
            print("  Reprise depuis checkpoint...")
            model.load_weights(checkpoint_path)
        
        # Entraînement
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=callbacks, verbose=1, shuffle=True
        )
        
        # Évaluation
        best_model = tf.keras.models.load_model(basin_model_dir / 'best_model.keras')
        y_pred = best_model.predict(X_test, verbose=0).flatten()
        
        nse_val = calculate_nse(y_test, y_pred)
        kge_val = calculate_kge(y_test, y_pred)
        mae_val = np.mean(np.abs(y_pred - y_test))
        rmse_val = np.sqrt(np.mean((y_pred - y_test)**2))
        
        print(f"  NSE: {nse_val:.4f} | KGE: {kge_val:.4f} | MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f}")
        
        # Visualisation
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Courbes d'apprentissage
            if history and hasattr(history, 'history') and 'loss' in history.history:
                axes[0].plot(history.history['loss'], label='Train')
                axes[0].plot(history.history['val_loss'], label='Validation')
                axes[0].set_title(f'Loss - Bassin {basin_id}')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('MSE')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Observations vs Prédictions
            n_plot = min(200, len(y_test))
            axes[1].plot(y_test[:n_plot], 'k-', label='Observé', alpha=0.7)
            axes[1].plot(y_pred[:n_plot], 'r-', label='Prédit', alpha=0.7)
            axes[1].set_title(f'Test - Bassin {basin_id} (NSE={nse_val:.3f})')
            axes[1].set_xlabel('Pas de temps')
            axes[1].set_ylabel('Débit')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(basin_model_dir / f'training_curves_{basin_id}.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"  ⚠ Visualisation: {e}")
        
        # Sauvegarde résultats
        all_results.append({
            'basin_id': basin_id,
            'nse': nse_val, 'kge': kge_val,
            'mae': mae_val, 'rmse': rmse_val,
            'n_train': len(X_train), 'n_test': len(X_test)
        })
        
        # Progression
        save_progress(basin_id)
        
        # Sauvegarde intermédiaire
        pd.DataFrame(all_results).to_csv(results_dir / "baseline_per_basin_summary.csv", index=False)
        
        del model, best_model, X_basin, y_basin
        gc.collect()
        
    except Exception as e:
        print(f"   Erreur: {e}")
        save_progress(basin_id)

# ============================================
# SYNTHÈSE FINALE
# ============================================

if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / "baseline_per_basin_summary.csv", index=False)
    
    valid_nse = [r['nse'] for r in all_results if r['nse'] > -9000]
    valid_kge = [r['kge'] for r in all_results if r['kge'] > -9000]
    
    print(f"\n{'='*70}")
    print("SYNTHÈSE FINALE")
    print(f"{'='*70}")
    print(f"Bassins traités : {len(all_results)}")
    print(f"NSE médian : {np.median(valid_nse):.4f}")
    print(f"NSE moyen  : {np.mean(valid_nse):.4f} ± {np.std(valid_nse):.4f}")
    print(f"KGE médian : {np.median(valid_kge):.4f}")
    print(f"\nRésultats : {results_dir / 'baseline_per_basin_summary.csv'}")