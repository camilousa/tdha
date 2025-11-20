from src.utils.load_bold_data import load_multisite_data
from src.preprocessing.fc_builder import DynamicFcBuilder
from sklearn.linear_model import LogisticRegression
from src.utils.evaluation import validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.fc_builder import DynamicFcBuilder
from src.preprocessing.channel_builder import FcChannelBuilder, extract_channels
import mlflow

def build_features(train, labels, 
                   params={"fisher": True, "pearson": False, 
                          "spearman": False, "difussion": False, 
                          "spectral": False, "ddt": True, 
                           "win_len": 40, "step": 15}):
    
    
    builder = DynamicFcBuilder(
            win_len=params["win_len"],
            step=params["step"],
            norm="zscore",      # "robust" si sospechas outliers
            ddof=0,
            handle_nans="impute",  # usa "impute" o "none"; None no es válido
        )
        
    print('\n---fc generation---')
        # Para TODOS los sujetos (train es lista de (R,T)):
    fcs_list = builder.compute_dynamic_fc(train)   # -> list[(W_i, R, R)]
    print('Sujetos procesados:', len(fcs_list))
    print('Ejemplo shape primer sujeto:', fcs_list[0].shape)


    print('\n---channel extraction---\n')

        # 3) Builder de canales (C, R, R) por sujeto
    chan_builder = FcChannelBuilder(

            # ------------------ Familias uniformes ------------------
            # Fisher-z (convierte a z si llega en r)
            use_fisher=params["fisher"],
            # "mean", "std", "p10", "p90", "absmean"
            fisher_stats=("mean", ), 

            # Pearson r (convierte a r si llega en z)
            # "mean", "std", "p10", "p90", "absmean"
            use_pearson=params["pearson"],
            pearson_stats=("mean", ),

            # Spearman (aprox desde r/z vía cópula gaussiana)
            # "mean", "std", "p10", "p90", "absmean"
            use_spearman=params["spearman"],
            spearman_stats=("mean", ),

            # ------------------ Agregados de grafo ------------------
            use_diffusion=params["difussion"],     # Heat kernel: exp(-tL)
            t_list=[1.0],       # puedes dejar solo [1.0] al activar
            use_spec=params["spectral"] ,          # Proyección espectral: U U^T
            eig_k=4,                 # nº autovectores (3–5 suele ir bien)
            use_clustering=False,    # Diagonal con coef. de clustering por nodo
            use_entropy=False,       # Diagonal con entropía de distribución de pesos

            # Ajustes para grafo
            adj_mode="abs",          # {'abs','pos','neg_abs','raw'} — usar 'abs'/'pos' para L>=0
            adj_zero_diag=False,

            # ------------------ DDT temporal (Δ entre ventanas) ------------------
            use_ddt=params["ddt"],
            ddt_lags=(1,),         # empieza con (1, 2) cuando lo actives
            ddt_stats=("l1mean", "std"),
            ddt_ema_beta=0.85,       # None para desactivar EMA
            ddt_clip=(-3, 3),        # útil si trabajas en Fisher-z

            # ------------------ Miscelánea ------------------
            n_jobs=1,                # 0/1 => secuencial; evita error de joblib con 0
            verbose=False
        )



    X, y, s, names = extract_channels(fcs_list, labels, sites, chan_builder, verbose=True)

    print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
    return X, y


tracking_host = "http://54.82.7.123:5000" 
mlflow.set_tracking_uri(tracking_host)
mlflow.set_experiment("ablation_nyu_18_baseline")

_, _, train, labels, sites = load_multisite_data(n_rois=18)


"""
Canal : z_fisher
Canal : z_fisher + spearman
Canal : z_fisher + Spectral
Canal : z_fisher + Difussion
Canal : z_fisher + Spectral + Difussion
"""

C = 0.4
params={"fisher": True, "pearson": False, 
        "spearman": False, "difussion": False, 
        "spectral": True, "ddt": False, "win_len":  40, "step": 15}


with mlflow.start_run(run_name="dynamic: fisher + difussion + spectral + ddt"):
    X, y = build_features(train, labels, params=params)
    print("n fueatures", X.shape[1])
    model = SVC(C=C)
    results = validate(model, X, y, metrics=["accuracy", "precision", "recall", "f1"], cv=5)
    mlflow.log_param("model", "svc")
    mlflow.log_param("C", C)
    mlflow.log_params(params=params)
    mlflow.log_metrics(metrics=results)
