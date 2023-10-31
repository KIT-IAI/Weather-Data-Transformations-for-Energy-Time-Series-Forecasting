from src.data_representations.calendar_features import calendar_features

# from src.data_representations.energy_differences import energy_differences
from src.data_representations.energy_naive import energy_naive
from src.data_representations.energy_pca import energy_pca, energy_pca_fit

from src.data_representations.dwd_naive import dwd_naive
from src.data_representations.dwd_stats import dwd_stats
from src.data_representations.dwd_pca import dwd_pca, dwd_pca_fit
from src.data_representations.dwd_pcacluster import dwd_pcacluster, dwd_pcacluster_fit
from src.data_representations.dwd_autoencoder import dwd_autoencoder, dwd_autoencoder_fit

from src.data_representations.nwp_autoencoder import nwp_autoencoder, nwp_autoencoder_fit
from src.data_representations.nwp_pca import nwp_pca, nwp_pca_fit
from src.data_representations.nwp_pcacluster import nwp_pcacluster, nwp_pcacluster_fit
from src.data_representations.nwp_naive import nwp_naive
from src.data_representations.nwp_rescale import nwp_rescale
from src.data_representations.nwp_stats import nwp_stats