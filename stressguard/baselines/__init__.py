from .base import ScoreBaseline, calibrate_scores, fit_static_threshold
from . import simple    # registers recon_ae, iforest, zscore, dlinear_recon
from . import reimpl    # registers anomaly_transformer, tranad, dcdetector, timesnet (compact reimplementations)
from . import catalog   # registers the 10 benchmark slots with verification status
