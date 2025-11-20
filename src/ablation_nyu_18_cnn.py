from src.utils.load_bold_data import load_multisite_data
from src.preprocessing.fc_builder import DynamicFcBuilder
from src.preprocessing.data_augmentation import AugmentedFCDataset, BoldAugmentorPro, FCSymmetricAugmentor
from src.preprocessing.data_augmentation import build_augmented_channels_for_indices
from src.preprocessing.channel_builder import FcChannelBuilder, extract_channels
from src.cnn.cnn import FCN2DCNN_GN
import numpy as np
import mlflow
from torch.utils.data import Dataset
import torch
from src.cnn.cv_train_validate import run_stratified_kfold


class NumpyFCDataset(Dataset):
    def __init__(self, X, y, selected_channels=None, dtype=torch.float32):
        """
        X: (N, C, 18, 18)
        y: (N,) etiquetas binarias 0/1 (se sanea si no)
        selected_channels: None o indices (list/np.ndarray/torch.Tensor)
        """
        X = np.asarray(X) if not torch.is_tensor(X) else X

        # Selección de canales robusta
        if selected_channels is not None:
            if torch.is_tensor(selected_channels):
                selected_channels = selected_channels.cpu().numpy()
            selected_channels = np.asarray(selected_channels)
            assert selected_channels.ndim == 1, "selected_channels debe ser 1D"
            assert selected_channels.min() >= 0 and selected_channels.max() < X.shape[1], \
                f"Índices de canal fuera de rango [0,{X.shape[1]-1}]"
            X = X[:, selected_channels, :, :]

        # A tensor float32
        self.X = torch.as_tensor(X, dtype=dtype).contiguous()
        assert self.X.ndim == 4, f"X debe ser (N,C,18,18), recibido {self.X.shape}"

        # Saneamos y -> enteros 0/1 long
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64, copy=False)
        y = (y != 0).astype(np.int64, copy=False)
        self.y = torch.as_tensor(y, dtype=torch.long)

        # Validación de tamaños
        assert self.X.shape[0] == self.y.shape[0], "Shapes incompatibles X vs y"

        # Para facilitar CV: expón y en numpy
        self.y_np = self.y.cpu().numpy()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # y ya viene saneada; no hace falta re-chequear por rendimiento
        return self.X[idx], self.y[idx]

    def __repr__(self):
        N, C, H, W = self.X.shape
        return f"NumpyFCDataset(N={N}, C={C}, H={H}, W={W}, dtype={self.X.dtype}, y_dtype={self.y.dtype})"



def make_create_model_fn(in_channels, reduce_mode="pool"):
    assert isinstance(in_channels, int) and in_channels > 0, "in_channels debe ser un entero > 0"
    def create_model_fn():
        return FCN2DCNN_GN(
            in_channels=in_channels,
            chans=(32, 64, 128),
            out_chan=96,
            num_classes=2,
            dropout_p=0.2,
            reduce_mode=reduce_mode
        )
    return create_model_fn



def initialize_fc():
    # 1) Cargar datos
    _, _, train, labels, sites = load_multisite_data(n_rois=18)
    print('\n---load bold---')
    print(np.unique(labels))

    print("Total de muestras:", len(train))
    print("Total con TDHA:", sum(labels), '-')
    print('total cn TDHA:', round(sum(labels)/len(train), 2) * 100, '%')

    # 2) Builder de dFC
    builder = DynamicFcBuilder(
        win_len=40,
        step=15,
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
        use_fisher=True,
        # "mean", "std", "p10", "p90", "absmean"
        fisher_stats=("mean", ), 

        # Pearson r (convierte a r si llega en z)
        # "mean", "std", "p10", "p90", "absmean"
        use_pearson=False,
        pearson_stats=("mean", ),

        # Spearman (aprox desde r/z vía cópula gaussiana)
        # "mean", "std", "p10", "p90", "absmean"
        use_spearman=False,
        spearman_stats=("mean", ),

        # ------------------ Agregados de grafo ------------------
        use_diffusion=True,     # Heat kernel: exp(-tL)
        t_list=[1.0],       # puedes dejar solo [1.0] al activar
        use_spec=True,          # Proyección espectral: U U^T
        eig_k=4,                 # nº autovectores (3–5 suele ir bien)
        use_clustering=False,    # Diagonal con coef. de clustering por nodo
        use_entropy=False,       # Diagonal con entropía de distribución de pesos

        # Ajustes para grafo
        adj_mode="abs",          # {'abs','pos','neg_abs','raw'} — usar 'abs'/'pos' para L>=0
        adj_zero_diag=True,

        # ------------------ DDT temporal (Δ entre ventanas) ------------------
        use_ddt=False,
        ddt_lags=(1,),         # empieza con (1, 2) cuando lo actives
        ddt_stats=("l1mean", "std"),
        ddt_ema_beta=0.85,       # None para desactivar EMA
        ddt_clip=(-3, 3),        # útil si trabajas en Fisher-z

        # ------------------ Miscelánea ------------------
        n_jobs=1,                # 0/1 => secuencial; evita error de joblib con 0
        verbose=False
    )



    X, y, s, names = extract_channels(fcs_list, labels, sites, chan_builder, verbose=True)
    return  X, y, s, names

def main():
    X, y, _, _ = initialize_fc()
    
    nyu_dataset = NumpyFCDataset(X, y)               # X: (177, 5, 18, 18), y: (177,)
    NUM_CANALES = X.shape[1]    

    # Dataset base SIN augment (tu clase actual)
    nyu_dataset = NumpyFCDataset(X, y)     # X:(N,C,18,18), y:(N,)

    # Augmentor en FC (ajusta sigma/alpha si quieres más/menos ruido)
    fc_augmentor = FCSymmetricAugmentor(sigma=0.02, alpha_shrink=0.05)
    
    # _----------------------------------------------

    # Builders por fold
    train_builder = lambda base, idx: AugmentedFCDataset(base, idx, augment=fc_augmentor, train=True)
    val_builder   = lambda base, idx: AugmentedFCDataset(base, idx, augment=None,        train=False)

    create_model_fn = make_create_model_fn(NUM_CANALES, reduce_mode="stride")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    USE_AUG = True  # o False para “sin augmentación”

    tdb = train_builder if USE_AUG else None
    vdb = val_builder   if USE_AUG else None
    

    fold_metrics_val, avg_val, std_val, fold_metrics_train, avg_tr, std_tr = run_stratified_kfold(
        dataset=nyu_dataset,
        create_model_fn=create_model_fn,      # tu lambda que crea FCN2DCNN_GN(in_channels=X.shape[1], ...)
        device=device,
        n_splits=5,
        batch_size=16,
        num_workers=0,
        seed=42,
        verbose_epochs=False,                 # <— desactiva prints por época si quieres
        verbose_cv=True,                      # <— deja los resúmenes por fold
        train_dataset_builder=tdb,   # <- ON/OFF según flag
        val_dataset_builder=vdb,     # <- ON/OFF según flag
        # kwargs del trainer:
        epochs=120,
        lr=1e-4,
        weight_decay=1e-4,
        es_patience=25,
        save_best_on='accuracy',
        early_stop_on='loss',
        final_selection='metric',
        class_weight_mode='balanced'
    )


    #print('\n----------------NYU-18-CV-5 FOLDS------------------')
    print('\nData Augmentation:', USE_AUG,'\n')
    #print('\n------AVG--------')
    #print(avg_metrics, )
    #print('\n------STD--------')
    #print(std_metrics, )

if __name__ == '__main__':
    main()