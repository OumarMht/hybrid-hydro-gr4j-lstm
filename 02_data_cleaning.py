import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys  
warnings.filterwarnings('ignore')

# ============================================
# CHEMINS ABSOLUS
# ============================================

current_dir = Path(__file__).parent
project_root = current_dir.parent

processed_dir = project_root / "processed"
cleaned_dir = project_root / "processed_cleaned_v"

print(f"Project root: {project_root}")
print(f"Processed dir: {processed_dir}")
print(f"Cleaned dir: {cleaned_dir}")

if not processed_dir.exists():
    print(f"ERROR: Processed directory not found at {processed_dir}")
    print("Run 01_data_import.py first!")
    exit(1)

cleaned_dir.mkdir(exist_ok=True)

# Compteurs
total_original = 0
total_cleaned = 0
total_removed = 0
basin_dirs = list(processed_dir.glob("basin_*"))
print(f"\nFound {len(basin_dirs)} basins to clean")

# ============================================
# TRAITEMENT DE CHAQUE BASSIN
# ============================================

for i, basin_dir in enumerate(basin_dirs):
    basin_id = basin_dir.name.split("_")[1]
    print(f"\n[{i+1}/{len(basin_dirs)}] Cleaning basin {basin_id}")
    
    try:
        # 1. LECTURE DU FICHIER BRUT
        ts_df = pd.read_csv(basin_dir / "timeseries_raw.csv")
        
        # CORRECTION : 'Unnamed: 0' → 'date'
        if 'Unnamed: 0' in ts_df.columns:
            ts_df = ts_df.rename(columns={'Unnamed: 0': 'date'})
            print(f"   Fixed: 'Unnamed: 0' → 'date'")
        
        # 2. VÉRIFICATION DE LA COLONNE DATE
        if 'date' not in ts_df.columns:
            # Chercher d'autres noms possibles
            date_candidates = [col for col in ts_df.columns 
                             if any(word in col.lower() for word in ['date', 'time', 'day', 'timestamp'])]
            if date_candidates:
                ts_df = ts_df.rename(columns={date_candidates[0]: 'date'})
                print(f"  Renamed '{date_candidates[0]}' → 'date'")
            else:
                print(f"  ERROR: No 'date' column found in basin {basin_id}")
                print(f"  Available columns: {list(ts_df.columns)}")
                continue        
        # 3. CONVERSION DE LA DATE
        ts_df['date'] = pd.to_datetime(ts_df['date'], errors='coerce')
        ts_df.set_index('date', inplace=True)
        
        # Vérifier les dates
        if ts_df.index.isna().any():
            print(f"  WARNING: {ts_df.index.isna().sum()} invalid dates removed")
            ts_df = ts_df[ts_df.index.notna()]
        
        print(f"  Date range: {ts_df.index.min().date()} to {ts_df.index.max().date()}")
        
        # 4. NETTOYAGE : SUPPRIMER LES LIGNES SANS discharge_spec
        rows_before = len(ts_df)
        if 'discharge_spec' not in ts_df.columns:
            print(f"   'discharge_spec' not found. Skipping.")
            continue
            
        ts_clean = ts_df.dropna(subset=['discharge_spec'])
        rows_after = len(ts_clean)
        removed = rows_before - rows_after
        
        if len(ts_clean) == 0:
            print(f"   No data after cleaning. Skipping.")
            continue
        
        # 5. CRÉER UNE PLAGE DE DATES CONTINUE
        full_date_range = pd.date_range(
            start=ts_clean.index.min(),
            end=ts_clean.index.max(),
            freq='D'
        )
        ts_clean = ts_clean.reindex(full_date_range)
        
        # 6. SAUVEGARDE CORRIGÉE
        basin_cleaned_dir = cleaned_dir / f"basin_{basin_id}"
        basin_cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        # Réinitialiser l'index et renommer explicitement
        ts_for_csv = ts_clean.reset_index()
        
        # CORRECTION  : La colonne s'appellera 'date' (pas 'index')
        if 'index' in ts_for_csv.columns:
            ts_for_csv = ts_for_csv.rename(columns={'index': 'date'})
        # Si déjà 'date', on garde
        
        # Sauvegarder le fichier principal
        ts_for_csv.to_csv(basin_cleaned_dir / "timeseries_cleaned.csv", index=False)
        
        # 7. RAPPORT
        cleaning_report = {
            'basin_id': basin_id,
            'original_rows': rows_before,
            'cleaned_rows': len(ts_clean),
            'missing_discharge': removed,
            'date_range_start': ts_clean.index.min(),
            'date_range_end': ts_clean.index.max(),
            'total_days': len(ts_clean)
        }
        
        report_df = pd.DataFrame([cleaning_report])
        report_df.to_csv(basin_cleaned_dir / "cleaning_report.csv", index=False)
        
        # Mise à jour des totaux
        total_original += rows_before
        total_cleaned += len(ts_clean)
        total_removed += removed
        
        print(f"   Rows: {rows_before} → {len(ts_clean)} (removed: {removed})")
        print(f"  Saved with 'date' column")
        
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")

# ============================================
# RAPPORT FINAL
# ============================================

print("\n" + "="*60)
print("CLEANING COMPLETE")
print("="*60)
print(f"Total basins processed: {len(basin_dirs)}")
print(f"Total rows: {total_original:,} → {total_cleaned:,}")
print(f"Rows removed (no discharge): {total_removed:,}")
print(f"Retention rate: {(total_cleaned/total_original*100):.1f}%")
print(f"\nOutput: {cleaned_dir}/")
print("Each file has a 'date' column (not 'index')")
print("="*60)