from app.regressor.io_utils import load_ecomm_traffic as _load_ecomm_traffic


def load_ecomm_traffic():
    """Wrapper to load ecomm traffic data with standardized columns."""
    return _load_ecomm_traffic()

