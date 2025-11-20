from src.utils.load_bold_data import load_multisite_data
from src.preprocessing.fc_builder import DynamicFcBuilder
from src.preprocessing.channel_builder import FcChannelBuilder, extract_channels
import numpy as np
from tqdm import tqdm 

# 1) Cargar datos
_, _, train, labels, sites = load_multisite_data(n_rois=18)
print('\n---load bold---')
print("Total de muestras:", len(train))
print("Total con TDHA:", sum(labels), '-', round(sum(labels)/len(train), 2) * 100, '%')

# 2) Builder de dFC
builder = DynamicFcBuilder(
    win_len=40,
    step=15,
    norm="zscore",      # "robust" si sospechas outliers
    ddof=0,
    fisher=True,        # desactiva si quieres r en [-1,1]
    diag_zero=True,
    handle_nans="impute",  # usa "impute" o "none"; None no es válido
)

print('\n---fc generation---')
# Para TODOS los sujetos (train es lista de (R,T)):
fcs_list = builder.compute_dynamic_fc(train)   # -> list[(W_i, R, R)]
print('Sujetos procesados:', len(fcs_list))
print('Ejemplo shape primer sujeto:', fcs_list[0].shape)


print('\n---channel extraction---')

# 3) Builder de canales (C, R, R) por sujeto
chan_builder = FcChannelBuilder(
    use_mean=True, use_abs=True, use_std=True,  # canales base típicos
    use_diffusion=False, t_list=[1.0],          # activa si quieres difusión
    use_spec=False, eig_k=4,                    # activa si quieres espectral
    use_clustering=False,
    use_entropy=False,
    use_ddt=False,
    adj_mode="abs", adj_zero_diag=True,
    n_jobs=0, verbose=False
)


X, y, s, names = extract_channels(fcs_list, labels, sites, chan_builder, verbose=True)

