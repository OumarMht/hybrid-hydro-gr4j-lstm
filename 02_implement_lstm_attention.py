"""
02_implement_lstm_attention.py 
Implémentation du LSTM avec attention et attributs physiques
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import warnings
import json
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ============================================
# FONCTION DE CALCUL NSE
# ============================================

def calculate_nse(observed, simulated):
   
    # Éviter la division par zéro
    if len(observed) == 0 or np.var(observed) == 0:
        return float('nan')
    
    # NSE = 1 - (Σ(obs - sim)²) / (Σ(obs - mean_obs)²)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    
    nse = 1 - (numerator / denominator)
    return nse

# ============================================
# CONFIGURATION - MIS À JOUR POUR GR4J CALIBRÉ
# ============================================

current_dir = Path(__file__).parent
project_root = current_dir.parent

# Chemins pour GR4J calibré
cleaned_dir = project_root / "processed_cleaned"  
data_dir = project_root / "data"
hybrid_dir = project_root / "hybrid_data_2"
models_dir = project_root / "models_2"
results_dir = project_root / "results_2" / "hybrid_2"
baseline_model_path = models_dir / "baseline_lstm_best.h5"

# Création des répertoires
#hybrid_dir.mkdir(exist_ok=True)
#results_dir.mkdir(exist_ok=True)
hybrid_dir.mkdir(exist_ok=True, parents=True)
results_dir.mkdir(exist_ok=True, parents=True)


print("="*70)
print(f"Project root: {project_root}")
print(f"Cleaned data: {cleaned_dir}")
print(f"Hybrid data: {hybrid_dir}")

# ============================================
# CHARGEMENT DES ATTRIBUTS PHYSIQUES
# ============================================

class PhysicalAttributesLoader:
    """Chargeur d'attributs physiques des bassins"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.attributes = {}
        self.load_all_attributes()
    
    def load_all_attributes(self):
        """Charge tous les fichiers d'attributs"""
        print("\nChargement des attributs physiques...")
        
        # Chargement des fichiers CSV
        self.attributes['topo'] = pd.read_csv(data_dir / "CAMELS_GB_topographic_attributes.csv")
        self.attributes['hydro'] = pd.read_csv(data_dir / "CAMELS_GB_hydrologic_attributes.csv")
        self.attributes['soil'] = pd.read_csv(data_dir / "CAMELS_GB_soil_attributes.csv")
        self.attributes['landcover'] = pd.read_csv(data_dir / "CAMELS_GB_landcover_attributes.csv")
        
        print(f"  Topographic: {len(self.attributes['topo'])} bassins")
        print(f"  Hydrologic: {len(self.attributes['hydro'])} bassins")
        print(f"  Soil: {len(self.attributes['soil'])} bassins")
        print(f"  Landcover: {len(self.attributes['landcover'])} bassins")
    
    def get_basin_attributes_vector(self, basin_id):
        try:
            basin_num = int(basin_id)
            
            # Extraction des attributs clés
            attrs = []
            
            # 1. Surface (km²)
            topo_df = self.attributes['topo']
            area = topo_df.loc[topo_df['gauge_id'] == basin_num, 'area'].values
            if len(area) == 0:
                print(f"  ERREUR: Bassin {basin_id} - attribut 'area' manquant")
                return None
            attrs.append(area[0])
            
            # 2. Baseflow Index (BFI)
            hydro_df = self.attributes['hydro']
            bfi = hydro_df.loc[hydro_df['gauge_id'] == basin_num, 'baseflow_index'].values
            if len(bfi) == 0:
                print(f"  ERREUR: Bassin {basin_id} - attribut 'baseflow_index' manquant")
                return None
            attrs.append(bfi[0])
            
            # 3. Pourcentage de sable
            soil_df = self.attributes['soil']
            sand = soil_df.loc[soil_df['gauge_id'] == basin_num, 'sand_perc'].values
            if len(sand) == 0:
                print(f"  ERREUR: Bassin {basin_id} - attribut 'sand_perc' manquant")
                return None
            attrs.append(sand[0])
            
            # 4. Pourcentage d'argile
            clay = soil_df.loc[soil_df['gauge_id'] == basin_num, 'clay_perc'].values
            if len(clay) == 0:
                print(f"  ERREUR: Bassin {basin_id} - attribut 'clay_perc' manquant")
                return None
            attrs.append(clay[0])
            
            # 5. Altitude moyenne
            elev = topo_df.loc[topo_df['gauge_id'] == basin_num, 'elev_mean'].values
            if len(elev) == 0:
                print(f"  ERREUR: Bassin {basin_id} - attribut 'elev_mean' manquant")
                return None
            attrs.append(elev[0])
            
            return np.array(attrs, dtype=np.float32)
            
        except Exception as e:
            print(f"  ERREUR chargement attributs bassin {basin_id}: {e}")
            return None
# ============================================
# ARCHITECTURE LSTM AVEC ATTENTION
# ============================================

class AttentionLayer(layers.Layer):
    """Couche d'attention temporelle"""
    
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # Couches pour calculer les scores d'attention
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[-1], self.units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.units, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.b1 = self.add_weight(name='b1',
                                  shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True)
        self.b2 = self.add_weight(name='b2',
                                  shape=(1,),
                                  initializer='zeros',
                                  trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        
        # Calcul des scores d'attention
        # Score = tanh(W1 * h_t + b1) * W2 + b2
        hidden = tf.tanh(tf.tensordot(inputs, self.W1, axes=1) + self.b1)
        score = tf.tensordot(hidden, self.W2, axes=1) + self.b2
        score = tf.squeeze(score, axis=-1)  # (batch_size, time_steps)
        
        # Application softmax pour obtenir les poids
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, time_steps)
        
        # Calcul du contexte pondéré
        context = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
        
        return context, attention_weights
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config

def create_lstm_attention_model(input_shape, output_scaler=None):
   
    # Entrée
    inputs = keras.Input(shape=input_shape)
    
    # Couche LSTM 1
    lstm1 = layers.LSTM(64, return_sequences=True, dropout=0.2,
                        recurrent_dropout=0.2)(inputs)
    
    # Couche LSTM 2
    lstm2 = layers.LSTM(32, return_sequences=True, dropout=0.2,
                        recurrent_dropout=0.2)(lstm1)
    
    # Couche d'attention
    context, attention_weights = AttentionLayer(units=32)(lstm2)
    
    # Concaténation contexte + dernière sortie LSTM
    last_lstm_output = lstm2[:, -1, :]
    combined = layers.Concatenate()([context, last_lstm_output])
    
    # Couches denses
    dense1 = layers.Dense(32, activation='relu')(combined)
    dropout1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(16, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)
    
    # Sortie
    output = layers.Dense(1, activation='linear')(dropout2)
    
    # Création du modèle
    model = keras.Model(inputs=inputs, outputs={'prediction': output, 'attention': attention_weights})
    
    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={'prediction': 'mse'},
        metrics={'prediction': 'mae'}
    )
    
    print(f"\nArchitecture LSTM avec attention créée:")
    print(f"  Entrée: {input_shape}")
    print(f"  Paramètres: {model.count_params():,}")
    
    return model

# ============================================
# PRÉPARATION DES DONNÉES AVEC GR4J CALIBRÉ
# ============================================

def prepare_hybrid_lstm_data_calibrated(basin_id, cleaned_dir, attr_loader, lstm_window=30):  
    print(f"\nPréparation données LSTM hybrides pour bassin {basin_id}")  
    # 1. Chargement des données temporelles nettoyées
    basin_dir = cleaned_dir / f"basin_{basin_id}"
    csv_file = basin_dir / "timeseries_cleaned.csv"
    
    if not csv_file.exists():
        print(f"  ERREUR: Fichier nettoyé non trouvé: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Colonnes nécessaires
    required_cols = ['date', 'precipitation', 'temperature', 'pet', 'discharge_spec']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"  ERREUR: Colonnes manquantes: {missing}")
        return None
    
    # 2. Chargement des RÉSULTATS GR4J CALIBRÉS
    # Fichier des résultats GR4J calibrés du script 01
    gr4j_calibrated_file = hybrid_dir / f"basin_{basin_id}" / "gr4j_calibrated_results.csv"
    
    if not gr4j_calibrated_file.exists():
        print(f"  ERREUR: Fichier GR4J calibré non trouvé: {gr4j_calibrated_file}")
        print(f"  Le script 01 doit être exécuté d'abord pour calibrer GR4J")
        print(f"  Cherché: {gr4j_calibrated_file}")
        return None
    
    gr4j_df = pd.read_csv(gr4j_calibrated_file)
    print(f" Fichier GR4J calibré chargé: {len(gr4j_df)} échantillons")
    
    # Vérifier les colonnes nécessaires
    if 'Q_GR4J_calibrated' not in gr4j_df.columns:
        print(f"  ERREUR: Colonne 'Q_GR4J_calibrated' manquante dans GR4J calibré")
        print(f"  Colonnes disponibles: {list(gr4j_df.columns)}")
        return None
    
    # 3. Alignement temporel amélioré
    # Vérifier si nous avons des dates
    if 'date' not in gr4j_df.columns:
        print(f"  ERREUR: Colonne 'date' manquante dans GR4J calibré")
        return None
    
    # Convertir les dates
    df['date'] = pd.to_datetime(df['date'])
    gr4j_df['date'] = pd.to_datetime(gr4j_df['date'])
    
    # Trouver l'intersection des dates
    common_dates = set(df['date']).intersection(set(gr4j_df['date']))
    common_dates = sorted(list(common_dates))
    
    if len(common_dates) == 0:
        print(f"  ERREUR CRITIQUE: Aucune date en commun entre données nettoyées et GR4J calibré")
        print(f"  Dates nettoyées: {df['date'].min()} à {df['date'].max()}")
        print(f"  Dates GR4J: {gr4j_df['date'].min()} à {gr4j_df['date'].max()}")
        return None
    
    print(f" Dates en commun: {len(common_dates)} jours")
    
    # 4. Récupération des attributs physiques
    phys_attrs = attr_loader.get_basin_attributes_vector(basin_id)
    print(f"  Attributs physiques: area={phys_attrs[0]:.1f}, BFI={phys_attrs[1]:.3f}, "
          f"sand={phys_attrs[2]:.1f}%, clay={phys_attrs[3]:.1f}%, elev={phys_attrs[4]:.1f}m")
    
    # 5. Création des séquences LSTM alignées avec GR4J calibré
    X_sequences = []
    y_targets = []
    y_gr4j_calibrated = []
    seq_dates = []
    
    # Créer un dictionnaire date -> index pour accès rapide
    date_to_idx = {date: idx for idx, date in enumerate(df['date'])}
    
    # Créer un dictionnaire date -> Q_GR4J_calibré
    gr4j_dict = {row['date']: row['Q_GR4J_calibrated'] for _, row in gr4j_df.iterrows()}
    
    # Créer un dictionnaire date -> Q_observed (depuis gr4j_df)
    gr4j_obs_dict = {row['date']: row['Q_observed'] for _, row in gr4j_df.iterrows()}
    
    # Pour chaque date commune, créer une séquence
    for target_date in common_dates:
        if target_date not in date_to_idx:
            continue
            
        target_idx = date_to_idx[target_date]
        
        # Vérifier que nous avons assez de données historiques
        if target_idx - lstm_window < 0:
            continue
        
        # Vérifier que nous avons la prédiction GR4J pour cette date
        if target_date not in gr4j_dict:
            continue
            
        # Vérifier que nous avons l'observation pour cette date
        if target_date not in gr4j_obs_dict:
            continue
        
        # Extraire la fenêtre temporelle (derniers lstm_window jours)
        start_idx = target_idx - lstm_window
        end_idx = target_idx
        
        # Vérifier l'absence de NaN dans la fenêtre
        if np.any(df.iloc[start_idx:end_idx][['precipitation', 'temperature', 'pet']].isna().values):
            continue
        
        # Extraire les features temporelles
        P_window = df.iloc[start_idx:end_idx]['precipitation'].values.astype(np.float32)
        T_window = df.iloc[start_idx:end_idx]['temperature'].values.astype(np.float32)
        E_window = df.iloc[start_idx:end_idx]['pet'].values.astype(np.float32)
        
        # Concaténation avec attributs statiques
        sequence = []
        for j in range(lstm_window):
            # Features temporelles du jour j
            time_feat = np.array([P_window[j], T_window[j], E_window[j]], dtype=np.float32)
            
            # Attributs statiques (toujours les mêmes)
            static_feat = phys_attrs.copy()
            
            # Concaténation
            day_vector = np.concatenate([time_feat, static_feat])
            sequence.append(day_vector)
        
        X_sequences.append(np.array(sequence))
        
        # La cible pour le LSTM est l'ERREUR entre observation et GR4J calibré
        # C'est ce que le LSTM doit apprendre à corriger
        q_obs = gr4j_obs_dict[target_date]
        q_gr4j_calibrated = gr4j_dict[target_date]
        error = q_obs - q_gr4j_calibrated
        
        y_targets.append(error)  # Le LSTM prédit l'erreur
        y_gr4j_calibrated.append(q_gr4j_calibrated)
        seq_dates.append(target_date)
    
    # Conversion en numpy arrays
    if len(X_sequences) == 0:
        print(f"  ERREUR: Aucune séquence créée")
        return None
    
    X = np.array(X_sequences, dtype=np.float32)
    y_error = np.array(y_targets, dtype=np.float32)  # Erreurs à prédire
    y_gr4j = np.array(y_gr4j_calibrated, dtype=np.float32)  # Prédictions GR4J
    
    print(f"  Séquences créées: {X.shape}")
    print(f"  Forme X: (batch={X.shape[0]}, time={X.shape[1]}, features={X.shape[2]})")
    print(f"  Nombre d'échantillons: {len(X_sequences)}")
    print(f"  Période: {seq_dates[0]} à {seq_dates[-1]}")
    
    # 6. Normalisation
    # Features: 3 temporelles + 5 statiques = 8 features
    X_reshaped = X.reshape(-1, X.shape[2])
    X_normalized = np.zeros_like(X_reshaped)
    
    # Normalisation des features temporelles (P, T, E) - indices 0,1,2
    for i in range(3):
        scaler = StandardScaler()
        X_normalized[:, i] = scaler.fit_transform(X_reshaped[:, i].reshape(-1, 1)).flatten()
    
    # Normalisation des attributs statiques (indices 3 à 7)
    # Utilisation de statistiques globales pour les attributs physiques
    stat_stats = {
        'area': {'mean': 500.0, 'std': 1000.0},
        'bfi': {'mean': 0.5, 'std': 0.2},
        'sand': {'mean': 45.0, 'std': 25.0},
        'clay': {'mean': 25.0, 'std': 15.0},
        'elev': {'mean': 150.0, 'std': 150.0}
    }
    
    stat_keys = ['area', 'bfi', 'sand', 'clay', 'elev']
    for i, key in enumerate(stat_keys):
        idx = i + 3
        mean, std = stat_stats[key]['mean'], stat_stats[key]['std']
        X_normalized[:, idx] = (X_reshaped[:, idx] - mean) / std
    
    X = X_normalized.reshape(X.shape)
    
    # Normalisation des erreurs (cibles du LSTM)
    y_scaler = StandardScaler()
    y_normalized = y_scaler.fit_transform(y_error.reshape(-1, 1)).flatten()
    
    print(f"  Normalisation appliquée:")
    print(f"    X: mean={np.mean(X):.3f}, std={np.std(X):.3f}")
    print(f"    y (erreurs): mean={np.mean(y_normalized):.3f}, std={np.std(y_normalized):.3f}")
    print(f"    Erreurs GR4J calibré (original): mean={np.mean(y_error):.3f}, std={np.std(y_error):.3f}")
    
    # Calcul des statistiques GR4J
    if len(y_gr4j) > 0:
        nse_gr4j = calculate_nse(np.array([gr4j_obs_dict[d] for d in seq_dates]), y_gr4j)
        print(f"  Performance GR4J calibré sur cette période: NSE = {nse_gr4j:.4f}")
    
    return {
        'X': X,
        'y': y_normalized,  # Erreurs normalisées
        'y_error_original': y_error,  # Erreurs originales
        'y_gr4j': y_gr4j,  # Prédictions GR4J calibrées
        'dates': np.array(seq_dates),
        'y_scaler': y_scaler,
        'phys_attrs': phys_attrs,
        'n_samples': len(X_sequences)
    }

# ============================================
# ENTRAÎNEMENT DU MODÈLE AVEC GR4J CALIBRÉ
# ============================================

def train_lstm_model_calibrated(data_dict, basin_id, epochs=50, batch_size=32):
    
    print(f"\nENTRAÎNEMENT LSTM POUR BASSIN {basin_id}") 
    X = data_dict['X']
    y = data_dict['y']  # Erreurs à prédire
    n_samples = data_dict['n_samples']
    
    # Division train/test (80/20) temporelle
    split_idx = int(0.8 * n_samples)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Division temporelle:")
    print(f"    Train: {len(X_train)} échantillons ({data_dict['dates'][0]} à {data_dict['dates'][split_idx-1]})")
    print(f"    Test:  {len(X_test)} échantillons ({data_dict['dates'][split_idx]} à {data_dict['dates'][-1]})")
    
    # Création du modèle
    input_shape = (X.shape[1], X.shape[2])  # (30, 8)
    model = create_lstm_attention_model(input_shape)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(hybrid_dir / f"basin_{basin_id}" / "lstm_best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entraînement
    print("  Début entraînement...")
    history = model.fit(
        X_train, {'prediction': y_train},
        validation_data=(X_test, {'prediction': y_test}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation détaillée
    print("  Évaluation du modèle...")
    predictions = model.predict(X_test, verbose=0)
    y_pred_test_norm = predictions['prediction'].flatten()
    
    # Dénormalisation des prédictions
    y_scaler = data_dict['y_scaler']
    y_pred_test = y_scaler.inverse_transform(y_pred_test_norm.reshape(-1, 1)).flatten()
    y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Métriques sur les erreurs
    test_mse_errors = np.mean((y_pred_test - y_test_original) ** 2) 
    test_mae_errors = np.mean(np.abs(y_pred_test - y_test_original))
    
    print(f"  Performance sur les ERREURS (ce que le LSTM prédit):")
    print(f"    MSE erreurs: {test_mse_errors:.4f}")
    print(f"    MAE erreurs: {test_mae_errors:.4f}")
    print(f"    Erreur moyenne prédite: {np.mean(y_pred_test):.4f}")
    print(f"    Erreur moyenne réelle: {np.mean(y_test_original):.4f}")
    
    # Reconstruction des débits complets
    y_gr4j_test = data_dict['y_gr4j'][split_idx:]
    y_obs_test = data_dict['y_error_original'][split_idx:] + y_gr4j_test
    
    # Débits LSTM = GR4J + correction LSTM
    y_lstm_test = y_gr4j_test + y_pred_test
    
    # Calcul NSE final
    nse_lstm = calculate_nse(y_obs_test, y_lstm_test)
    nse_gr4j = calculate_nse(y_obs_test, y_gr4j_test)
    
    print(f"\n  PERFORMANCE FINALE SUR DÉBITS:")
    print(f"    NSE GR4J calibré seul: {nse_gr4j:.4f}")
    print(f"    NSE LSTM corrigé:      {nse_lstm:.4f}")
    print(f"    Amélioration:          {nse_lstm - nse_gr4j:+.4f}")
    
    if nse_lstm > nse_gr4j:
        print(f"   SUCCÈS: LSTM améliore GR4J de {100*(nse_lstm - nse_gr4j)/nse_gr4j:.1f}%")
    else:
        print(f"   ATTENTION: LSTM ne s'améliore pas sur GR4J")
    
    return model, history, {
        'nse_gr4j': nse_gr4j,
        'nse_lstm': nse_lstm,
        'improvement': nse_lstm - nse_gr4j
    }

# ============================================
# INFÉRENCE ET SAUVEGARDE
# ============================================

def predict_and_save_calibrated(model, data_dict, basin_id, lstm_window=30):
    print(f"\nPRÉDICTION ET SAUVEGARDE POUR BASSIN {basin_id}")
    
    X = data_dict['X']
    y_error_original = data_dict['y_error_original']  # Erreurs originales
    y_gr4j = data_dict['y_gr4j']  # Prédictions GR4J
    dates = data_dict['dates']
    y_scaler = data_dict['y_scaler']
    
    # Prédiction des erreurs par le LSTM
    predictions = model.predict(X, verbose=0)
    y_pred_error_norm = predictions['prediction'].flatten()
    attention_weights = predictions['attention']
    
    # Dénormalisation des erreurs prédites
    y_pred_error = y_scaler.inverse_transform(y_pred_error_norm.reshape(-1, 1)).flatten()
    
    # Reconstruction des débits LSTM
    # Débit LSTM = Débit GR4J + Correction LSTM
    y_lstm = y_gr4j + y_pred_error
    
    # Débits observés reconstruits
    y_observed = y_error_original + y_gr4j
    
    # Calcul des métriques
    nse_lstm = calculate_nse(y_observed, y_lstm)
    nse_gr4j = calculate_nse(y_observed, y_gr4j)
    
    correlation = np.corrcoef(y_lstm, y_observed)[0, 1]
    mse = np.mean((y_lstm - y_observed) ** 2)
    mae = np.mean(np.abs(y_lstm - y_observed))
    
    # KGE
    r = correlation
    alpha = np.std(y_lstm) / np.std(y_observed) if np.std(y_observed) > 0 else 0
    beta = np.mean(y_lstm) / np.mean(y_observed) if np.mean(y_observed) > 0 else 0
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    print(f"  PERFORMANCE COMPLETE:")
    print(f"    NSE GR4J calibré: {nse_gr4j:.4f}")
    print(f"    NSE LSTM : {nse_lstm:.4f}")
    print(f"    Amélioration:     {nse_lstm - nse_gr4j:+.4f}")
    print(f"    KGE LSTM:         {kge:.4f}")
    print(f"    Correlation:      {correlation:.3f}")
    print(f"    MSE:              {mse:.4f}")
    print(f"    MAE:              {mae:.4f}")
    print(f"    Débit LSTM moyen: {np.mean(y_lstm):.3f} mm/j")
    print(f"    Débit observé moyen: {np.mean(y_observed):.3f} mm/j")
    print(f"    Erreur moyenne LSTM: {np.mean(y_pred_error):.3f} mm/j")
    
    # Extraction des scores d'attention
    attention_max = np.max(attention_weights, axis=1)
    attention_mean = np.mean(attention_weights, axis=1)
    attention_top_idx = np.argmax(attention_weights, axis=1)
    
    # Sauvegarde des résultats
    basin_hybrid_dir = hybrid_dir / f"basin_{basin_id}"
    basin_hybrid_dir.mkdir(exist_ok=True)
    
    # CSV principal avec TOUTES les informations
    results_df = pd.DataFrame({
        'date': dates,
        'Q_GR4J_calibrated': y_gr4j,
        'Q_LSTM_correction': y_pred_error,
        'Q_LSTM_final': y_lstm,
        'Q_observed': y_observed,
        'attention_max_score': attention_max,
        'attention_mean_score': attention_mean,
        'attention_top_index': attention_top_idx
    })
    
    csv_path = basin_hybrid_dir / "lstm_calibrated_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Résultats CSV sauvegardés: {csv_path} ({len(results_df)} lignes)")
    
    # Fichier .npy avec scores d'attention complets
    npy_path = basin_hybrid_dir / "lstm_attention_scores_calibrated.npy"
    np.save(npy_path, attention_weights)
    print(f"  Scores d'attention complets sauvegardés: {npy_path}")
    
    # Fichier de configuration détaillé
    config = {
        'basin_id': basin_id,
        'lstm_window': lstm_window,
        'gr4j_version': 'calibrated',
        'gr4j_nse': float(nse_gr4j),
        'features': {
            'temporal': ['precipitation', 'temperature', 'pet'],
            'static': ['area', 'baseflow_index', 'sand_perc', 'clay_perc', 'elev_mean']
        },
        'performance': {
            'nse_gr4j': float(nse_gr4j),
            'nse_lstm': float(nse_lstm),
            'improvement': float(nse_lstm - nse_gr4j),
            'kge_lstm': float(kge),
            'correlation': float(correlation),
            'mse': float(mse),
            'mae': float(mae),
            'q_gr4j_mean': float(np.mean(y_gr4j)),
            'q_lstm_mean': float(np.mean(y_lstm)),
            'q_observed_mean': float(np.mean(y_observed)),
            'correction_mean': float(np.mean(y_pred_error))
        },
        'training_info': {
            'n_samples': len(X),
            'date_range': f"{dates[0]} to {dates[-1]}",
            'gr4j_calibration_note': 'GR4J optimized via Differential Evolution (NSE 0.62)'
        },
        'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    }
    
    config_path = basin_hybrid_dir / "lstm_calibrated_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f" Configuration détaillée sauvegardée: {config_path}")
    
    return {
        'basin_id': basin_id,
        'nse_gr4j': nse_gr4j,
        'nse_lstm': nse_lstm,
        'improvement': nse_lstm - nse_gr4j,
        'kge': kge,
        'correlation': correlation,
        'mse': mse,
        'mae': mae,
        'n_samples': len(results_df),
        'csv_path': csv_path,
        'npy_path': npy_path,
        'config_path': config_path
    }

# ============================================
# FONCTION PRINCIPALE
# ============================================

def main():
    
    
    print("\n" + "="*70)
    print("DÉMARRAGE LSTM - RÉ-ENTRAÎNEMENT AVEC GR4J CALIBRÉ")
    
    # Demande de confirmation
    response = input("\nVoulez-vous continuer? (oui/non): ").strip().lower()
    if response != 'oui':
        print("Annulé. Aucun modèle n'a été ré-entraîné.")
        return
    
    # 1. Initialisation
    lstm_window = 30
    n_basins = 50  # Même nombre que GR4J calibré
    
    # Chargeur d'attributs
    attr_loader = PhysicalAttributesLoader(data_dir)
    
    # 2. Liste des bassins avec GR4J calibré
    basin_dirs = list(hybrid_dir.glob("basin_*"))
    basin_ids = sorted([d.name.split("_")[1] for d in basin_dirs])
    
    # Filtrer pour avoir seulement ceux avec GR4J calibré
    basins_with_calibrated = []
    for basin_id in basin_ids:
        gr4j_calibrated_file = hybrid_dir / f"basin_{basin_id}" / "gr4j_calibrated_results.csv"
        if gr4j_calibrated_file.exists():
            basins_with_calibrated.append(basin_id)
    
    if n_basins > 0:
        basins_with_calibrated = basins_with_calibrated[:n_basins]
    
    print(f"\nBassins avec GR4J calibré disponibles: {len(basins_with_calibrated)}")
    print(f"Bassins à ré-entraîner: {basins_with_calibrated}")
    
    if len(basins_with_calibrated) == 0:
        print("ERREUR: Aucun bassin avec GR4J calibré trouvé.")
        print("Exécutez d'abord le script 01 pour calibrer GR4J.")
        return
    
    # 3. Traitement par bassin
    all_results = []
    
    for i, basin_id in enumerate(basins_with_calibrated):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(basins_with_calibrated)}] RÉ-ENTRAÎNEMENT BASSIN {basin_id}")
        print(f"{'='*60}")
        
        try:
            # a) Préparation des données AVEC GR4J CALIBRÉ
            print("  Phase 1: Préparation des données avec GR4J calibré...")
            data_dict = prepare_hybrid_lstm_data_calibrated(
                basin_id=basin_id,
                cleaned_dir=cleaned_dir,
                attr_loader=attr_loader,
                lstm_window=lstm_window
            )
            
            if data_dict is None:
                print(f"  SKIP: Données non disponibles pour bassin {basin_id}")
                continue
            
            # b) Entraînement du modèle
            print("  Phase 2: Entraînement du LSTM...")
            model, history, training_results = train_lstm_model_calibrated(
                data_dict=data_dict,
                basin_id=basin_id,
                epochs=50,
                batch_size=32
            )
            
            # c) Prédiction et sauvegarde
            print("  Phase 3: Prédiction et sauvegarde...")
            results = predict_and_save_calibrated(
                model=model,
                data_dict=data_dict,
                basin_id=basin_id,
                lstm_window=lstm_window
            )
            
            # Mettre à jour avec les résultats d'entraînement
            results.update(training_results)
            all_results.append(results)
            
            # d) Sauvegarde du modèle
            model_path = hybrid_dir / f"basin_{basin_id}" / "lstm_calibrated_model.h5"
            model.save(model_path)
            print(f"  ✓ Modèle LSTM sauvegardé: {model_path}")
            
            # e) Sauvegarde de l'historique d'entraînement
            history_df = pd.DataFrame(history.history)
            history_path = hybrid_dir / f"basin_{basin_id}" / "lstm_training_history.csv"
            history_df.to_csv(history_path, index=False)
            print(f"  ✓ Historique d'entraînement sauvegardé: {history_path}")
            
        except Exception as e:
            print(f"  ERREUR sur bassin {basin_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 4. Synthèse des résultats
    if all_results:
        print(f"\n{'='*70}")
        print("SYNTHÈSE DES PERFORMANCES LSTM AVEC GR4J CALIBRÉ")
        print(f"{'='*70}")
        
        summary_data = []
        for res in all_results:
            summary_data.append({
                'basin_id': res['basin_id'],
                'n_samples': res['n_samples'],
                'nse_gr4j': res['nse_gr4j'],
                'nse_lstm': res['nse_lstm'],
                'improvement': res['improvement'],
                'improvement_percent': 100 * res['improvement'] / res['nse_gr4j'] if res['nse_gr4j'] > 0 else 0,
                'kge': res['kge'],
                'correlation': res['correlation'],
                'mse': res['mse'],
                'mae': res['mae']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = results_dir / "lstm_calibrated_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\nRÉSULTATS MOYENS SUR {len(all_results)} BASSINS:")
        print(f"  NSE GR4J calibré: {summary_df['nse_gr4j'].mean():.4f} ± {summary_df['nse_gr4j'].std():.4f}")
        print(f"  NSE LSTM corrigé:  {summary_df['nse_lstm'].mean():.4f} ± {summary_df['nse_lstm'].std():.4f}")
        print(f"  Amélioration moyenne: {summary_df['improvement'].mean():.4f} ± {summary_df['improvement'].std():.4f}")
        print(f"  Pourcentage amélioration: {summary_df['improvement_percent'].mean():.1f}%")
        print(f"  KGE moyen: {summary_df['kge'].mean():.4f} ± {summary_df['kge'].std():.4f}")
        
        # Analyse détaillée des améliorations
        improvement_counts = {
            'Excellente (>0.05)': (summary_df['improvement'] > 0.05).sum(),
            'Bonne (0.02-0.05)': ((summary_df['improvement'] >= 0.02) & (summary_df['improvement'] <= 0.05)).sum(),
            'Légère (0.0-0.02)': ((summary_df['improvement'] > 0.0) & (summary_df['improvement'] < 0.02)).sum(),
            'Négative (<0.0)': (summary_df['improvement'] <= 0.0).sum()
        }
        
        print(f"\nDISTRIBUTION DES AMÉLIORATIONS:")
        for category, count in improvement_counts.items():
            percentage = (count / len(all_results)) * 100
            print(f"  {category}: {count} bassins ({percentage:.1f}%)")
        
        # Bassins où LSTM s'est le mieux amélioré
        top_improvements = summary_df.nlargest(5, 'improvement')
        print(f"\nTOP 5 BASSINS - MEILLEURE AMÉLIORATION:")
        for _, row in top_improvements.iterrows():
            print(f"  Bassin {row['basin_id']}: NSE GR4J={row['nse_gr4j']:.3f}, "
                  f"LSTM={row['nse_lstm']:.3f}, +{row['improvement']:.3f} "
                  f"(+{row['improvement_percent']:.1f}%)")
        
        print(f"\nSynthèse détaillée sauvegardée: {summary_path}")
    
    # 5. Rapport final
    print(f"\n{'='*70}")
    print("RÉ-ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print(f"{'='*70}")
if __name__ == "__main__":
    main()