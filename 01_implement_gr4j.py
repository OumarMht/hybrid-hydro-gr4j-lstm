"""
01_implement_gr4j.py 
- VERSION AVEC CALIBRATION AUTOMATIQUE OPTIMALE
-par algorithme d'Évolution Différentielle pour optimisation globale
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import json
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

current_dir = Path(__file__).parent
project_root = current_dir.parent

# Chemins
cleaned_dir = project_root / "processed_cleaned"
data_dir = project_root / "data"
hybrid_dir = project_root / "hybrid_data_2"
diagnostics_dir = project_root / "diagnostics_2" / "gr4j_2"
results_dir = project_root / "results_2" / "hybrid_2"
 
# Création des répertoires
##hybrid_dir.mkdir(exist_ok=True)
##diagnostics_dir.mkdir(exist_ok=True, parents=True)
##results_dir.mkdir(exist_ok=True)
hybrid_dir.mkdir(exist_ok=True, parents=True)
diagnostics_dir.mkdir(exist_ok=True, parents=True)
results_dir.mkdir(exist_ok=True, parents=True)

print("="*70)
print("IMPLEMENTATION GR4J AVEC CALIBRATION AUTOMATIQUE (ÉVOLUTION DIFFÉRENTIELLE)")
print("="*70)

# ============================================
# MODÈLE GR4J AVEC CALIBRATION
# ============================================

class GR4JModel:
    """
    Implémentation du modèle hydrologique GR4J avec optimisation
    """
    
    def __init__(self, params=None):
        if params is None:
            # Valeurs initiales raisonnables pour l'optimisation
            self.params = {'X1': 350.0, 'X2': 0.0, 'X3': 90.0, 'X4': 1.9}
        else:
            self.params = params
        
        self.S = 0.0
        self.R = 0.0
        self.history_S = []
        self.history_R = []
        self.history_Q = []
    
    def reset_states(self):
        self.S = 0.0
        self.R = 0.0
        self.history_S = []
        self.history_R = []
        self.history_Q = []
    
    def production_store(self, P, E):
        if P >= E:
            Es = E
            net_rain = P - E
            S_ratio = self.S / self.params['X1']
            Ps = (self.params['X1'] * (1 - S_ratio**2) * np.tanh(net_rain / self.params['X1']) /
                  (1 + S_ratio * np.tanh(net_rain / self.params['X1'])))
        else:
            Es = P
            net_evap = E - P
            S_ratio = self.S / self.params['X1']
            Ps = 0
            self.S = self.S - (self.params['X1'] * (2 - S_ratio) * np.tanh(net_evap / self.params['X1']) /
                              (1 + (1 - S_ratio) * np.tanh(net_evap / self.params['X1'])))
            self.S = max(0, self.S)
        
        self.S = self.S + Ps
        Perc = self.S * (1 - (1 + (self.S / (self.params['X1'] * 4/9))**4)**(-0.25))
        self.S = self.S - Perc
        
        return Ps, Es, Perc
    
    def exchange_function(self, Ps, Perc):
        exchange = self.params['X2'] * (Ps + Perc) / 2
        return exchange
    
    def routing_store(self, Perc, exchange):
        self.R = self.R + Perc + exchange
        Qr = self.R * (1 - (1 + (self.R / self.params['X3'])**4)**(-0.25))
        self.R = self.R - Qr
        Qd = 0.9 * Qr
        Qb = 0.1 * Qr
        return Qd + Qb
    
    def simulate(self, P_series, E_series, warm_up_days=365):
        """
        Simulation GR4J complète
        """
        self.reset_states()
        
        # Pré-chauffage
        for i in range(min(warm_up_days, len(P_series))):
            Ps, Es, Perc = self.production_store(P_series[i], E_series[i])
            exchange = self.exchange_function(Ps, Perc)
            self.routing_store(Perc, exchange)
        
        # Simulation principale
        Q_simulated = []
        start_idx = min(warm_up_days, len(P_series))
        
        for i in range(start_idx, len(P_series)):
            Ps, Es, Perc = self.production_store(P_series[i], E_series[i])
            exchange = self.exchange_function(Ps, Perc)
            Q = self.routing_store(Perc, exchange)
            Q_simulated.append(Q)
        
        return np.array(Q_simulated)
    
    def simulate_for_optimization(self, P_series, E_series, Q_obs, warm_up_days=365):
        """
        Simulation optimisée pour la calibration
        """
        Q_sim = self.simulate(P_series, E_series, warm_up_days)
        
        # Aligner les longueurs
        min_len = min(len(Q_sim), len(Q_obs[warm_up_days:]))
        return Q_sim[:min_len], Q_obs[warm_up_days:warm_up_days+min_len]

# ============================================
# FONCTIONS D'OPTIMISATION
# ============================================

def calculate_kge(q_obs, q_sim):
    """
    Calcule le Kling-Gupta Efficiency (KGE)
    """
    if len(q_obs) < 2:
        return -999
    
    # Supprimer les NaN
    mask = ~np.isnan(q_obs) & ~np.isnan(q_sim)
    q_obs = q_obs[mask]
    q_sim = q_sim[mask]
    
    if len(q_obs) < 2:
        return -999
    
    # Statistiques
    r = np.corrcoef(q_obs, q_sim)[0, 1]
    alpha = np.std(q_sim) / np.std(q_obs) if np.std(q_obs) > 0 else 0
    beta = np.mean(q_sim) / np.mean(q_obs) if np.mean(q_obs) > 0 else 0
    
    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

def calculate_nse(q_obs, q_sim):
    """
    Calcule le Nash-Sutcliffe Efficiency (NSE)
    """
    if len(q_obs) < 2:
        return -999
    
    mask = ~np.isnan(q_obs) & ~np.isnan(q_sim)
    q_obs = q_obs[mask]
    q_sim = q_sim[mask]
    
    if len(q_obs) < 2:
        return -999
    
    numerator = np.sum((q_obs - q_sim) ** 2)
    denominator = np.sum((q_obs - np.mean(q_obs)) ** 2)
    
    if denominator == 0:
        return -999
    
    nse = 1 - numerator / denominator
    return nse

def objective_function(params, P_series, E_series, Q_obs, warm_up_days=365):
    """
    Fonction objectif pour l'optimisation : maximiser le NSE
    """
    # Extraire les paramètres
    X1, X2, X3, X4 = params
    
    # Vérifier les contraintes physiques
    if X1 <= 0 or X3 <= 0 or X4 <= 0:
        return 999  # Pénalité élevée
    
    # Créer le modèle avec les paramètres testés
    gr4j = GR4JModel(params={'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})
    
    # Simuler
    try:
        Q_sim = gr4j.simulate(P_series, E_series, warm_up_days)
        Q_obs_trimmed = Q_obs[warm_up_days:warm_up_days+len(Q_sim)]
        
        # Calculer NSE
        nse = calculate_nse(Q_obs_trimmed, Q_sim)
        
        # On maximise NSE, donc on minimise (1 - NSE)
        return 1 - nse
        
    except:
        return 999  # En cas d'erreur

def calibrate_gr4j(P_series, E_series, Q_obs, bounds=None, maxiter=50, popsize=15):
    """
    Calibration automatique de GR4J par Évolution Différentielle
    
    Args:
        P_series: Précipitations (mm/j)
        E_series: Évapotranspiration potentielle (mm/j)
        Q_obs: Débits observés (mm/j)
        bounds: Bornes des paramètres [(min, max) pour X1, X2, X3, X4]
        maxiter: Nombre maximum d'itérations
        popsize: Taille de la population
        
    Returns:
        best_params: Meilleurs paramètres trouvés
        best_nse: Meilleur NSE atteint
        optimization_history: Historique de l'optimisation
    """
    print("  Début de la calibration par Évolution Différentielle...")
    
    # Bornes par défaut si non spécifiées
    if bounds is None:
        # Bornes réalistes pour GR4J (basées sur la littérature)
        bounds = [(100.0, 1500.0),    # X1: Production store capacity [mm]
                  (-5.0, 5.0),        # X2: Groundwater exchange coefficient
                  (10.0, 500.0),      # X3: Routing store capacity [mm]
                  (0.5, 4.0)]         # X4: Time base [days]
    
    # Historique de l'optimisation
    history = {
        'params': [],
        'fitness': [],
        'nse': []
    }
    
    # Fonction de rappel pour enregistrer l'historique
    def callback(xk, convergence):
        # Calculer le NSE pour ce jeu de paramètres
        X1, X2, X3, X4 = xk
        gr4j = GR4JModel(params={'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})
        Q_sim = gr4j.simulate(P_series, E_series, warm_up_days=365)
        Q_obs_trimmed = Q_obs[365:365+len(Q_sim)]
        nse = calculate_nse(Q_obs_trimmed, Q_sim)
        
        history['params'].append(xk.copy())
        history['fitness'].append(1 - nse)
        history['nse'].append(nse)
        
        if len(history['nse']) % 10 == 0:
            print(f"    Itération {len(history['nse'])} - Meilleur NSE: {max(history['nse']):.4f}")
    
    # Optimisation par Évolution Différentielle
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(P_series, E_series, Q_obs),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        callback=callback,
        disp=False
    )
    
    # Extraire les résultats
    best_params = result.x
    best_fitness = result.fun
    best_nse = 1 - best_fitness
    
    # Dernière simulation avec les meilleurs paramètres
    X1, X2, X3, X4 = best_params
    gr4j = GR4JModel(params={'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})
    Q_sim_best = gr4j.simulate(P_series, E_series, warm_up_days=365)
    
    print(f"  Calibration terminée!")
    print(f"  Meilleurs paramètres: X1={X1:.1f}, X2={X2:.3f}, X3={X3:.1f}, X4={X4:.3f}")
    print(f"  Meilleur NSE atteint: {best_nse:.4f}")
    
    return best_params, best_nse, history, Q_sim_best

# ============================================
# DIAGNOSTICS AVANCÉS GR4J (inchangé)
# ============================================

class GR4JDiagnostics:
    """
    Classe pour analyser les erreurs structurelles de GR4J
    """
    
    def __init__(self, basin_id, Q_obs, Q_sim, dates=None):
        self.basin_id = basin_id
        self.Q_obs = np.array(Q_obs)
        self.Q_sim = np.array(Q_sim)
        self.dates = dates
        self.results = {}
        
        # Nettoyage des données
        mask = (~np.isnan(self.Q_obs)) & (~np.isnan(self.Q_sim))
        self.Q_obs = self.Q_obs[mask]
        self.Q_sim = self.Q_sim[mask]
    
    def decompose_nse(self):
        """
        Décompose le NSE en trois composantes
        """
        # Calcul des statistiques de base
        mean_obs = np.mean(self.Q_obs)
        mean_sim = np.mean(self.Q_sim)
        std_obs = np.std(self.Q_obs)
        std_sim = np.std(self.Q_sim)
        
        # Corrélation
        if std_obs > 0 and std_sim > 0 and len(self.Q_obs) > 1:
            corr = np.corrcoef(self.Q_obs, self.Q_sim)[0, 1]
        else:
            corr = 0
        
        # Calcul NSE décomposé
        if std_obs > 0:
            alpha = (mean_sim - mean_obs) / std_obs
            beta = (std_sim / std_obs) - 1
            gamma = 2 * (std_sim / std_obs) * (1 - corr)
        else:
            alpha = beta = gamma = 0
        
        # NSE total
        nse = 1 - (alpha**2 + beta**2 + gamma)
        
        # Contribution relative de chaque composante
        total_error = alpha**2 + beta**2 + gamma
        if total_error > 0:
            contrib_alpha = alpha**2 / total_error * 100
            contrib_beta = beta**2 / total_error * 100
            contrib_gamma = gamma / total_error * 100
        else:
            contrib_alpha = contrib_beta = contrib_gamma = 0
        
        decomposition = {
            'nse_total': nse,
            'mean_obs': mean_obs,
            'mean_sim': mean_sim,
            'std_obs': std_obs,
            'std_sim': std_sim,
            'correlation': corr,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'contrib_bias': contrib_alpha,
            'contrib_variability': contrib_beta,
            'contrib_correlation': contrib_gamma
        }
        
        self.results['nse_decomposition'] = decomposition
        return decomposition
    
    def calculate_volume_bias(self):
        """
        Calcul du biais volumique total
        """
        if len(self.Q_obs) == 0:
            return {}
        
        total_volume_obs = np.sum(self.Q_obs)
        total_volume_sim = np.sum(self.Q_sim)
        
        volume_bias = {
            'total_volume_obs': float(total_volume_obs),
            'total_volume_sim': float(total_volume_sim),
            'absolute_bias': float(total_volume_sim - total_volume_obs),
            'relative_bias': float((total_volume_sim - total_volume_obs) / total_volume_obs * 100 if total_volume_obs > 0 else 0)
        }
        
        self.results['volume_bias'] = volume_bias
        return volume_bias
    
    def calculate_peak_errors(self):
        """
        Analyse spécifique des erreurs sur les pics de crue
        """
        if len(self.Q_obs) < 100:
            return {}
        
        # Identifier les pics (max locaux)
        peaks_mask = np.zeros_like(self.Q_obs, dtype=bool)
        
        for i in range(1, len(self.Q_obs)-1):
            if self.Q_obs[i] > self.Q_obs[i-1] and self.Q_obs[i] > self.Q_obs[i+1]:
                peaks_mask[i] = True
        
        q_obs_peaks = self.Q_obs[peaks_mask]
        q_sim_peaks = self.Q_sim[peaks_mask]
        
        if len(q_obs_peaks) < 5:
            return {}
        
        # Calcul des erreurs sur les pics
        peak_errors = {
            'n_peaks': int(len(q_obs_peaks)),
            'mean_peak_obs': float(np.mean(q_obs_peaks)),
            'mean_peak_sim': float(np.mean(q_sim_peaks)),
            'peak_bias': float(np.mean(q_sim_peaks - q_obs_peaks)),
            'peak_mae': float(np.mean(np.abs(q_sim_peaks - q_obs_peaks))),
            'peak_mse': float(np.mean((q_sim_peaks - q_obs_peaks)**2)),
            'relative_peak_error': float(np.mean(np.abs(q_sim_peaks - q_obs_peaks) / (q_obs_peaks + 0.01)))
        }
        
        self.results['peak_errors'] = peak_errors
        return peak_errors
    
    def create_diagnostic_report(self, params_gr4j=None):
        """
        Génère un rapport complet de diagnostic
        """
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC COMPLET GR4J - Bassin {self.basin_id}")
        print(f"{'='*70}")
        
        # 1. NSE décomposé
        nse_decomp = self.decompose_nse()
        print(f"\n1. DECOMPOSITION NSE:")
        print(f"   NSE total: {nse_decomp['nse_total']:.4f}")
        print(f"   Corrélation: {nse_decomp['correlation']:.4f}")
        print(f"   Moyenne observée: {nse_decomp['mean_obs']:.3f} mm/j")
        print(f"   Moyenne simulée: {nse_decomp['mean_sim']:.3f} mm/j")
        print(f"   Biais (α): {nse_decomp['alpha']:.3f} σ")
        print(f"   Erreur variabilité (β): {nse_decomp['beta']:.3f}")
        
        # 2. Biais volumique
        vol_bias = self.calculate_volume_bias()
        print(f"\n2. BIAIS VOLUMIQUE:")
        print(f"   Volume total observé: {vol_bias['total_volume_obs']:.1f} mm")
        print(f"   Volume total simulé: {vol_bias['total_volume_sim']:.1f} mm")
        print(f"   Biais absolu: {vol_bias['absolute_bias']:.1f} mm")
        print(f"   Biais relatif: {vol_bias['relative_bias']:.1f}%")
        
        # 3. Erreurs sur les pics
        peak_err = self.calculate_peak_errors()
        if peak_err:
            print(f"\n3. ERREURS SUR LES PICS:")
            print(f"   Nombre de pics analysés: {peak_err['n_peaks']}")
            print(f"   Pic moyen observé: {peak_err['mean_peak_obs']:.3f} mm/j")
            print(f"   Pic moyen simulé: {peak_err['mean_peak_sim']:.3f} mm/j")
            print(f"   Biais sur pics: {peak_err['peak_bias']:.3f} mm/j")
            print(f"   MAE sur pics: {peak_err['peak_mae']:.3f} mm/j")
            print(f"   Erreur relative pics: {peak_err['relative_peak_error']:.2%}")
        
        # 4. Paramètres GR4J
        if params_gr4j:
            print(f"\n4. PARAMETRES GR4J CALIBRES:")
            print(f"   X1 (production): {params_gr4j.get('X1', 'N/A'):.1f} mm")
            print(f"   X2 (échange): {params_gr4j.get('X2', 'N/A'):.3f} mm")
            print(f"   X3 (routage): {params_gr4j.get('X3', 'N/A'):.1f} mm")
            print(f"   X4 (temps base): {params_gr4j.get('X4', 'N/A'):.3f} jours")
        
        # 5. Recommandations
        print(f"\n5. DIAGNOSTIC ET RECOMMANDATIONS:")
        
        # Diagnostic basé sur la décomposition NSE
        if nse_decomp['contrib_bias'] > 50:
            print(f"   → PROBLEME PRINCIPAL: BIAIS SYSTEMATIQUE")
            if nse_decomp['mean_sim'] > nse_decomp['mean_obs']:
                print(f"     ACTION: Réduire X1 (rétention) ou ajuster X2 (échange négatif)")
            else:
                print(f"     ACTION: Augmenter X1 ou ajuster X2 (échange positif)")
        
        elif nse_decomp['contrib_variability'] > 50:
            print(f"   → PROBLEME PRINCIPAL: VARIABILITE")
            if nse_decomp['std_sim'] < nse_decomp['std_obs']:
                print(f"     ACTION: Réduire X3 (réponse plus rapide) ou X4 (temps base)")
            else:
                print(f"     ACTION: Augmenter X3 (réponse plus lente)")
        
        elif nse_decomp['contrib_correlation'] > 50:
            print(f"   → PROBLEME PRINCIPAL: TIMING")
            print(f"     ACTION: Ajuster X4 (temps de base) pour synchroniser")
        
        # Diagnostic pics
        if peak_err and peak_err['relative_peak_error'] > 0.5:
            print(f"   → PROBLEME MAJEUR: ERREUR SUR LES PICS")
            print(f"     ACTION: Réduire X4 pour réponse plus rapide aux crues")
        
        # Diagnostic biais volumique
        if abs(vol_bias['relative_bias']) > 20:
            print(f"   → PROBLEME: BIAIS VOLUMIQUE IMPORTANT")
            print(f"     ACTION: Ajuster X1 pour corriger le bilan hydrique")
        
        return self.results

# ============================================
# FONCTION PRINCIPALE AVEC CALIBRATION
# ============================================

def run_gr4j_with_calibration(basin_id, cleaned_dir, hybrid_dir, diagnostics_dir):
    """
    Exécute GR4J avec calibration automatique optimale
    """
    print(f"\n{'='*60}")
    print(f"GR4J AVEC CALIBRATION AUTOMATIQUE - Bassin {basin_id}")
    print(f"{'='*60}")
    
    # 1. Chargement des données
    basin_dir = cleaned_dir / f"basin_{basin_id}"
    csv_file = basin_dir / "timeseries_cleaned.csv"
    
    if not csv_file.exists():
        print(f"  ERREUR: Fichier non trouvé: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Vérification des colonnes
    required_cols = ['date', 'precipitation', 'pet', 'discharge_spec']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"  ERREUR: Colonnes manquantes: {missing}")
        return None
    
    # Préparation des séries
    P_series = df['precipitation'].fillna(0).values
    E_series = df['pet'].fillna(1.0).values
    Q_observed = df['discharge_spec'].fillna(0).values
    dates = pd.to_datetime(df['date']) if 'date' in df.columns else None
    
    print(f"  Données chargées: {len(df)} jours")
    print(f"  Pluie moyenne: {P_series.mean():.2f} mm/j")
    print(f"  PET moyenne: {E_series.mean():.2f} mm/j")
    print(f"  Débit moyen: {Q_observed.mean():.2f} mm/j")
    
    # 2. CALIBRATION AUTOMATIQUE par Évolution Différentielle
    print(f"\n  Lancement de la calibration optimale...")
    
    best_params, best_nse, opt_history, Q_simulated = calibrate_gr4j(
        P_series=P_series,
        E_series=E_series,
        Q_obs=Q_observed,
        maxiter=50,    # Suffisant pour convergence
        popsize=15     # Bon compromis vitesse/exploration
    )
    
    # 3. Diagnostic complet avec les paramètres calibrés
    X1, X2, X3, X4 = best_params
    calibrated_params = {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4}
    
    print(f"\n  Analyse des performances avec paramètres calibrés...")
    diagnostic = GR4JDiagnostics(
        basin_id=basin_id,
        Q_obs=Q_observed[365:365+len(Q_simulated)],
        Q_sim=Q_simulated,
        dates=dates[365:365+len(Q_simulated)] if dates is not None else None
    )
    
    results = diagnostic.create_diagnostic_report(params_gr4j=calibrated_params)
    
    # 4. Sauvegarde des résultats
    basin_diag_dir = diagnostics_dir / f"basin_{basin_id}"
    basin_diag_dir.mkdir(exist_ok=True)
    
    # a) Fichier JSON avec tous les diagnostics
    results['calibration'] = {
        'best_params': best_params.tolist(),
        'best_nse': float(best_nse),
        'optimization_history': {
            'nse_values': opt_history['nse'],
            'params_history': [p.tolist() for p in opt_history['params']]
        }
    }
    
    diag_path = basin_diag_dir / "gr4j_detailed_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # b) CSV avec métriques principales
    summary_data = {
        'basin_id': basin_id,
        'nse_total': results.get('nse_decomposition', {}).get('nse_total', np.nan),
        'correlation': results.get('nse_decomposition', {}).get('correlation', np.nan),
        'X1_calibrated': X1,
        'X2_calibrated': X2,
        'X3_calibrated': X3,
        'X4_calibrated': X4,
        'volume_bias_percent': results.get('volume_bias', {}).get('relative_bias', np.nan),
        'peak_error_relative': results.get('peak_errors', {}).get('relative_peak_error', np.nan),
        'calibration_success': True
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_path = basin_diag_dir / "gr4j_diagnostics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # 5. Résultats pour la fusion hybride
    basin_hybrid_dir = hybrid_dir / f"basin_{basin_id}"
    basin_hybrid_dir.mkdir(exist_ok=True)
    
    # Sauvegarde des sorties GR4J calibrées pour la fusion
    output_df = pd.DataFrame({
        'date': df['date'].iloc[365:365+len(Q_simulated)].values if len(df) > 365 else df['date'].iloc[len(Q_simulated):].values,
        'Q_GR4J_calibrated': Q_simulated,
        'Q_observed': Q_observed[365:365+len(Q_simulated)] if len(Q_observed) > 365 else Q_observed[len(Q_simulated):]
    })
    
    output_path = basin_hybrid_dir / "gr4j_calibrated_results.csv"
    output_df.to_csv(output_path, index=False)
    
    # 6. Graphique de convergence de l'optimisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Convergence NSE
    ax1.plot(opt_history['nse'], 'b-', linewidth=2)
    ax1.set_xlabel('Itération d\'optimisation')
    ax1.set_ylabel('NSE')
    ax1.set_title(f'Convergence de la calibration - Bassin {basin_id}')
    ax1.grid(True, alpha=0.3)
    
    # Meilleurs paramètres trouvés
    params_names = ['X1', 'X2', 'X3', 'X4']
    ax2.bar(params_names, best_params, color=['blue', 'green', 'red', 'orange'])
    ax2.set_ylabel('Valeur paramètre')
    ax2.set_title('Paramètres GR4J calibrés')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = basin_diag_dir / f"gr4j_calibration_basin_{basin_id}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Résultats sauvegardés:")
    print(f"    - Diagnostics détaillés: {diag_path}")
    print(f"    - Résumé métriques: {summary_path}")
    print(f"    - Sorties GR4J calibrées: {output_path}")
    print(f"    - Graphique de calibration: {plot_path}")
    
    return {
        'basin_id': basin_id,
        'diagnostics': results,
        'summary': summary_data,
        'best_params': best_params,
        'best_nse': best_nse,
        'Q_simulated': Q_simulated
    }

# ============================================
# FONCTION PRINCIPALE
# ============================================

def main():
    """
    Fonction principale exécutant GR4J avec calibration automatique
    """
    print("\n" + "="*70)
    print("GR4J AVEC CALIBRATION AUTOMATIQUE PAR ÉVOLUTION DIFFÉRENTIELLE")
    print("="*70)
    
    # Liste des bassins à analyser
    basin_dirs = list(cleaned_dir.glob("basin_*"))
    basin_ids = sorted([d.name.split("_")[1] for d in basin_dirs])
    
    # Limiter le nombre de bassins si nécessaire
    n_basins = min(50, len(basin_ids))  # Maximum 50 bassins
    if n_basins > 0:
        basin_ids = basin_ids[:n_basins]
    
    print(f"\nAnalyse de {len(basin_ids)} bassins:")
    print(f"  {basin_ids[:5]}..." if len(basin_ids) > 5 else f"  {basin_ids}")
    
    results_summary = []
    
    for i, basin_id in enumerate(basin_ids):
        print(f"\n[{i+1}/{len(basin_ids)}] Calibration bassin {basin_id}")
        
        try:
            result = run_gr4j_with_calibration(
                basin_id=basin_id,
                cleaned_dir=cleaned_dir,
                hybrid_dir=hybrid_dir,
                diagnostics_dir=diagnostics_dir
            )
            
            if result is not None:
                results_summary.append(result['summary'])
        
        except Exception as e:
            print(f"  ERREUR traitement bassin {basin_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # 7. Synthèse globale
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        global_summary_path = diagnostics_dir / "gr4j_global_calibration_summary.csv"
        summary_df.to_csv(global_summary_path, index=False)
        
        print(f"\n" + "="*70)
        print("SYNTHÈSE GLOBALE DE LA CALIBRATION GR4J")
        print("="*70)
        
        print(f"\nStatistiques sur {len(summary_df)} bassins:")
        
        if 'nse_total' in summary_df.columns:
            nse_mean = summary_df['nse_total'].mean()
            nse_std = summary_df['nse_total'].std()
            print(f"  NSE moyen après calibration : {nse_mean:.4f} ± {nse_std:.4f}")
            
            # Distribution des NSE
            excellent = (summary_df['nse_total'] > 0.75).sum()
            good = ((summary_df['nse_total'] > 0.5) & (summary_df['nse_total'] <= 0.75)).sum()
            acceptable = ((summary_df['nse_total'] > 0.0) & (summary_df['nse_total'] <= 0.5)).sum()
            poor = (summary_df['nse_total'] <= 0.0).sum()
            
            print(f"\n  Distribution des performances:")
            print(f"    Excellent (NSE > 0.75): {excellent} bassins ({excellent/len(summary_df)*100:.1f}%)")
            print(f"    Bon (0.50 < NSE ≤ 0.75): {good} bassins ({good/len(summary_df)*100:.1f}%)")
            print(f"    Acceptable (0.0 < NSE ≤ 0.50): {acceptable} bassins ({acceptable/len(summary_df)*100:.1f}%)")
            print(f"    Mauvais (NSE ≤ 0.0): {poor} bassins ({poor/len(summary_df)*100:.1f}%)")
        
        # Statistiques des paramètres calibrés
        param_cols = ['X1_calibrated', 'X2_calibrated', 'X3_calibrated', 'X4_calibrated']
        for col in param_cols:
            if col in summary_df.columns:
                mean_val = summary_df[col].mean()
                std_val = summary_df[col].std()
                param_name = col.replace('_calibrated', '')
                print(f"  {param_name:5} moyen : {mean_val:7.2f} ± {std_val:.2f}")
        
        print(f"\nSynthèse globale sauvegardée: {global_summary_path}")
        
    print(f"\n" + "="*70)
    print("CALIBRATION GR4J TERMINÉE")
    print("="*70)
    print(f"\nLes résultats de calibration sont disponibles dans:")
    print(f"  {diagnostics_dir}/")
   
if __name__ == "__main__":
    main()