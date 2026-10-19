from .regimes import RegimeLabeler, market_return, hill_tail_index, apply_calendar, write_stress_split, load_stress_split
from .context import default_context, build_context
from .dataset import RegimeEventDataset, adverse_move, env_balanced_loader
from . import synthetic_regime   # registers "synthetic_regime"
from . import recipes            # registers the 9 real-data builders (recipes + loaders where data is public)
from . import real               # registers binance_um, us_equity (files already in data_raw/)
