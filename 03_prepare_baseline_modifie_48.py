"""
03_prepare_baseline_per_basin.py
Prépare les données baseline - UN DOSSIER PAR BASSIN
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

current_dir = Path(__file__).parent
project_root = current_dir.parent

cleaned_dir = project_root / "processed_cleaned"
baseline_dir = project_root / "baseline_data_48_per_basin"
data_dir = project_root / "data"

print(f"Racine du projet : {project_root}")
print(f"Dossier nettoyé : {cleaned_dir}")
print(f"Dossier baseline : {baseline_dir}")

if not cleaned_dir.exists():
    print(f"ERREUR : Dossier nettoyé non trouvé")
    print("Exécutez 02_data_cleaning.py d'abord !")
    exit(1)

# Paramètres
SEQ_LENGTH = 30
baseline_dir.mkdir(exist_ok=True)

# Variables
input_features = [
    'precipitation', 'temperature', 'pet', 
    'humidity', 'shortwave_rad', 'longwave_rad'
]
static_features = ['area', 'baseflow_index', 'sand_perc', 'clay_perc', 'elev_mean']
target_feature = 'discharge_spec'

# ============================================
# CHARGEMENT DES ATTRIBUTS STATIQUES
# ============================================

print("\nChargement des attributs statiques...")
try:
    topo_df = pd.read_csv(data_dir / "CAMELS_GB_topographic_attributes.csv")
    hydro_df = pd.read_csv(data_dir / "CAMELS_GB_hydrologic_attributes.csv")
    soil_df = pd.read_csv(data_dir / "CAMELS_GB_soil_attributes.csv")
    
    static_dict = {}
    for _, row in topo_df.iterrows():
        bid = str(row['gauge_id'])
        static_dict[bid] = {
            'area': row.get('area', 100.0),
            'elev_mean': row.get('elev_mean', 200.0)
        }
    for _, row in hydro_df.iterrows():
        bid = str(row['gauge_id'])
        if bid in static_dict:
            static_dict[bid]['baseflow_index'] = row.get('baseflow_index', 0.5)
    for _, row in soil_df.iterrows():
        bid = str(row['gauge_id'])
        if bid in static_dict:
            static_dict[bid]['sand_perc'] = row.get('sand_perc', 50.0)
            static_dict[bid]['clay_perc'] = row.get('clay_perc', 20.0)
    
    # Vérifier les attributs manquants et exclure les bassins incomplets
    missing_attrs_basins = []
    for bid in static_dict:
        missing = [key for key in static_features if key not in static_dict[bid]]
        if missing:
           missing_attrs_basins.append((bid, missing))
           del static_dict[bid]  # Exclure le bassin
    print(f"  Exclus : {len(missing_attrs_basins)} bassins avec attributs manquants")    
except Exception as e:
    print(f"  Erreur chargement attributs : {e}")
    static_dict = {}

# ============================================
# TRAITEMENT DES BASSINS
# ============================================

basin_dirs = list(cleaned_dir.glob("basin_*"))
gr4j_chap4_dir = project_root / "hybrid_data_chap4"
valid_basins = []
for d in basin_dirs:
    bid = d.name.split("_")[1]
    if (gr4j_chap4_dir / f"basin_{bid}" / "gr4j_chap4_results.csv").exists():
        valid_basins.append(d)

basin_dirs = valid_basins
print(f"\nFiltré à {len(basin_dirs)} bassins avec résultats GR4J Chap4")

skipped_basins = []
all_basin_ids = []
basin_stats = []

for i, basin_dir in enumerate(basin_dirs):
    basin_id = basin_dir.name.split("_")[1]
    print(f"\n[{i+1}/{len(basin_dirs)}] Bassin {basin_id}")
    
    try:
        # Lecture du fichier nettoyé
        csv_file = basin_dir / "timeseries_cleaned.csv"
        if not csv_file.exists():
            print(f"  Fichier manquant")
            skipped_basins.append((basin_id, "file_missing"))
            continue
            
        ts_df = pd.read_csv(csv_file)
        if ts_df.empty:
            print(f"  Fichier vide")
            skipped_basins.append((basin_id, "empty"))
            continue
        
        # Identification de la colonne date
        date_column = None
        for col in ts_df.columns:
            if 'date' in col.lower():
                date_column = col
                break
        
        if date_column is None:
            if 'index' in ts_df.columns:
                date_column = 'index'
            else:
                for col in ts_df.columns:
                    try:
                        pd.to_datetime(ts_df[col].head(5))
                        date_column = col
                        break
                    except:
                        continue
        
        if date_column is None:
            skipped_basins.append((basin_id, "no_date"))
            continue
        
        # Conversion et indexation par date
        ts_df['date'] = pd.to_datetime(ts_df[date_column], errors='coerce')
        ts_df.set_index('date', inplace=True)
        ts_df = ts_df[ts_df.index.notna()]
        
        if len(ts_df) == 0:
            skipped_basins.append((basin_id, "invalid_dates"))
            continue
        
        # Vérification des colonnes requises
        missing = [f for f in input_features + [target_feature] if f not in ts_df.columns]
        if missing:
            skipped_basins.append((basin_id, f"missing_{len(missing)}"))
            continue
        
        if len(ts_df) < SEQ_LENGTH + 1:
            skipped_basins.append((basin_id, "insufficient_data"))
            continue
        
        print(f"   Dates : {ts_df.index.min().date()} à {ts_df.index.max().date()}")
        print(f"   Lignes : {len(ts_df)}")
        
        # Vérification des attributs statiques
        if basin_id not in static_dict:
           print(f"   Bassin {basin_id} : attributs statiques manquants - ignoré")
           skipped_basins.append((basin_id, "missing_static_attrs"))
           continue
        basin_static = static_dict[basin_id]

        for key in static_features:
            ts_df[key] = basin_static.get(key, 0.0)
        
        # Préparation des features pour normalisation
        scaler = StandardScaler()
        all_features = input_features + static_features
        
        # Comptage des valeurs manquantes par colonne
        nan_counts = ts_df[all_features].isna().sum()
        if (nan_counts > len(ts_df) * 0.1).any():
            print(f"   ATTENTION : >10% NaN dans : {nan_counts[nan_counts > len(ts_df)*0.1].index.tolist()}")
            skipped_basins.append((basin_id, "too_many_nans"))
            continue

        # Remplissage des trous par interpolation linéaire
        features_data = ts_df[all_features].interpolate(method='linear', limit=5)

        # Remplissage des bords (début et fin) avec les valeurs valides les plus proches
        features_data = features_data.fillna(method='bfill').fillna(method='ffill')

        # Exclusion si des valeurs manquantes persistent
        if features_data.isna().any().any():
            print(f"   Impossible de remplir tous les NaN - ignoré")
            skipped_basins.append((basin_id, "persistent_nans"))
            continue
        
        # Normalisation des features
        features_scaled = scaler.fit_transform(features_data)
        
        # Création du dataframe normalisé
        ts_normalized = ts_df.copy()
        ts_normalized[all_features] = features_scaled
        
        # Suppression des lignes sans débit (on n'interpole pas les débits)
        n_nan = ts_normalized[target_feature].isna().sum()
        if n_nan > 0:
            print(f"   {n_nan} lignes sans débit - supprimées")
            ts_normalized = ts_normalized.dropna(subset=[target_feature])
            
            if len(ts_normalized) < SEQ_LENGTH + 1:
                print(f"   Données insuffisantes après suppression des débits manquants")
                skipped_basins.append((basin_id, "insufficient_discharge"))
                continue
        
        # Création des séquences pour le LSTM
        data = ts_normalized[all_features + [target_feature]].values
        
        X_basin = []
        y_basin = []
        
        for j in range(len(data) - SEQ_LENGTH):
            X_basin.append(data[j:j+SEQ_LENGTH, :-1])
            y_basin.append(data[j+SEQ_LENGTH, -1])
        
        if len(X_basin) == 0:
            skipped_basins.append((basin_id, "no_sequences"))
            continue
        
        X_basin = np.array(X_basin)
        y_basin = np.array(y_basin)
        
        # Sauvegarde par bassin
        basin_out_dir = baseline_dir / f"basin_{basin_id}"
        basin_out_dir.mkdir(exist_ok=True, parents=True)
        
        np.save(basin_out_dir / "X.npy", X_basin)
        np.save(basin_out_dir / "y.npy", y_basin)
        
        all_basin_ids.append(basin_id)
        basin_stats.append({
            'basin_id': basin_id,
            'n_sequences': len(X_basin),
            'y_mean': y_basin.mean(),
            'y_std': y_basin.std(),
            'start_date': ts_df.index.min(),
            'end_date': ts_df.index.max()
        })
        
        print(f"   Séquences : {len(X_basin)} | Features : {len(all_features)}")
        print(f"   Sauvegardé dans : {basin_out_dir}")
        
    except Exception as e:
        print(f"   Erreur : {str(e)[:80]}")
        skipped_basins.append((basin_id, f"error: {str(e)[:30]}"))

# ============================================
# SAUVEGARDE GLOBALE
# ============================================

print(f"\n{'='*60}")
print("PRÉPARATION TERMINÉE")
print(f"{'='*60}")
print(f"Bassins : {len(all_basin_ids)}/{len(basin_dirs)} réussis")
print(f"Ignorés : {len(skipped_basins)}")

# Liste des bassins
with open(baseline_dir / "basin_list.txt", "w") as f:
    for bid in all_basin_ids:
        f.write(f"{bid}\n")

if basin_stats:
    pd.DataFrame(basin_stats).to_csv(baseline_dir / "basin_stats.csv", index=False)

if skipped_basins:
    pd.DataFrame(skipped_basins, columns=['basin_id', 'reason']).to_csv(
        baseline_dir / "skipped_basins.csv", index=False
    )

print(f"\n {len(all_basin_ids)} bassins sauvegardés dans {baseline_dir}/")
print(f"basin_list.txt, basin_stats.csv")
print("\n" + "="*60)
print("PRÊT POUR L'ENTRAÎNEMENT PAR BASSIN")
print("="*60)