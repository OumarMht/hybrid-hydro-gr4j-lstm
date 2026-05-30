import pandas as pd
import os
from pathlib import Path
import sys

# ============================================
# CORRECTION DES CHEMINS - IMPORTANT
# ============================================

# Déterminer le chemin racine du projet
current_dir = Path(__file__).parent  
project_root = current_dir.parent    # Camels_GB/

# Dossiers utiles
data_dir = project_root / "data"
timeseries_dir = data_dir / "timeseries"
output_dir = project_root / "processed"

print(f"Project root: {project_root}")
print(f"Data dir: {data_dir}")
print(f"Timeseries dir: {timeseries_dir}")
print(f"Output dir: {output_dir}")

# Vérifier si le dossier timeseries existe
if not timeseries_dir.exists():
    print(f"ERREUR : Dossier timeseries introuvable à l'emplacement {timeseries_dir}")
    sys.exit(1)

# 1. Lister tous les fichiers CSV dans timeseries
basin_files = list(timeseries_dir.glob("*.csv"))
print(f"\nSearching in: {timeseries_dir}")
print(f"Found {len(basin_files)} basin files")

# Afficher les 5 premiers fichiers pour vérification
for i, f in enumerate(basin_files[:5]):
    print(f"  {i+1}. {f.name}")

# Si aucun fichier CSV n'est trouvé
if len(basin_files) == 0:
    print("\nERREUR : Aucun fichier CSV trouvé !")
    print("Veuillez vérifier :")
    print(f"  1. Le chemin : {timeseries_dir}")
    print(f"  2. L'extension des fichiers : doit être .csv")
    sys.exit(1)

# 2. Lire tous les attributs statiques
attributes = {}
print("\nLoading attribute files...")
for attr_file in data_dir.glob("CAMELS_GB_*_attributes.csv"):
    attr_name = attr_file.stem.replace("CAMELS_GB_", "").replace("_attributes", "")
    try:
        attributes[attr_name] = pd.read_csv(attr_file)
        print(f"  ✓ Loaded {attr_name} attributes: {attr_file.name}")
    except Exception as e:
        print(f"  ✗ Error loading {attr_file.name}: {e}")

# 3. Fusionner pour chaque bassin
print(f"\nProcessing {len(basin_files)} basins...")
for i, basin_file in enumerate(basin_files):
    # Extraire l'ID du bassin depuis le nom de fichier
    filename_parts = basin_file.stem.split("_")
    # Format: CAMELS_GB_hydromet_timeseries_10002_19701001-20150930
    if len(filename_parts) >= 5:
        basin_id = filename_parts[4]  # Le 5ème élément est l'ID
    else:
        # Le nom du fichier ne correspond pas au format attendu
        print(f"ERREUR: Format de nom de fichier inattendu : {basin_file.name}")
        print(f"Format attendu : CAMELS_GB_*_timeseries_ID_datedébut-datefin.csv")
        sys.exit(1) 
    
    print(f"\n[{i+1}/{len(basin_files)}] Processing basin {basin_id} from {basin_file.name}")
    
    try:
        # Lire les séries temporelles
        ts_df = pd.read_csv(basin_file)
        
        # Vérifier les colonnes
        print(f"  Colonnes trouvées: {list(ts_df.columns)}")
        
        # Convertir la date si la colonne existe
        if 'date' in ts_df.columns:
            ts_df['date'] = pd.to_datetime(ts_df['date'])
            ts_df.set_index('date', inplace=True)
            print(f"  Date range: {ts_df.index.min()} to {ts_df.index.max()}")
            print(f"  Rows: {len(ts_df)}")
        else:
           print(f"Aucune colonne 'date' trouvée dans {basin_file.name}")
           print(f"Colonnes disponibles: {list(ts_df.columns)}")
           sys.exit(1)        
        # Ajouter l'ID du bassin
        ts_df['basin_id'] = basin_id       
        # Fusionner avec les attributs statiques
        merged_attributes = 0
        for attr_name, attr_df in attributes.items():
            # Chercher la colonne d'ID 
            id_columns = [col for col in attr_df.columns if 'id' in col.lower() or 'gauge' in col.lower()]
            
            if id_columns:
                id_col = id_columns[0]
                # Convertir basin_id au même type
                try:
                    basin_id_num = int(basin_id)
                    if basin_id_num in attr_df[id_col].values:
                        basin_attrs = attr_df[attr_df[id_col] == basin_id_num].iloc[0].to_dict()
                        for key, value in basin_attrs.items():
                            if key != id_col:  # Ne pas dupliquer l'ID
                                ts_df[f"{attr_name}_{key}"] = value
                        merged_attributes += 1
                except:
                    # Si conversion échoue, essayer comme string
                    if str(basin_id) in attr_df[id_col].astype(str).values:
                        basin_attrs = attr_df[attr_df[id_col].astype(str) == str(basin_id)].iloc[0].to_dict()
                        for key, value in basin_attrs.items():
                            if key != id_col:
                                ts_df[f"{attr_name}_{key}"] = value
                        merged_attributes += 1
        
        print(f"  Merged {merged_attributes} attribute files")
        
        # Sauvegarder par bassin
        basin_output_dir = output_dir / f"basin_{basin_id}"
        basin_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = basin_output_dir / "timeseries_raw.csv"
        ts_df.to_csv(output_path)
        print(f"  enregistré dans  {output_path.relative_to(project_root)}")
        
    except Exception as e:
        print(f" ERROR processing {basin_file.name}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print(f"IMPORT TERMINÉ !")
print(f"Bassins traités sauvegardés dans : {output_dir.relative_to(project_root)}/")
print("="*50)