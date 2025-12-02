import numpy as np
import OpenVisus as ov

BASE_URL = "https://nsdf-climate3-origin.nationalresearchplatform.org:50098/nasa/nsdf/climate3/dyamond/"

def open_llc2160(variable: str):
    """
    Open an LLC2160 DYAMOND variable using OpenVisus.

    Parameters
    ----------
    variable : str
        One of ["u", "v", "w", "theta", "salt", ...].

    Returns
    -------
    db : OpenVisus dataset handle
    """
    if variable in ("theta", "w"):
        base_dir = f"mit_output/llc2160_{variable}/llc2160_{variable}.idx"
    elif variable == "u":
        base_dir = "mit_output/llc2160_arco/visus.idx"
    else:
        # salt, v, etc.
        base_dir = f"mit_output/llc2160_{variable}/{variable}_llc2160_x_y_depth.idx"

    field = BASE_URL + base_dir
    return ov.LoadDataset(field)


_SALT_DB_CACHE = None

def get_salt_db():
    """
    Return a cached OpenVisus dataset for salinity.
    """
    global _SALT_DB_CACHE
    if _SALT_DB_CACHE is None:
        _SALT_DB_CACHE = open_llc2160("salt")
    return _SALT_DB_CACHE



def compute_land_ocean_mask3d(
    time: int = 0,
    quality: int = -18,
    eps: float = 0.0,
    use_surface_only: bool = False,
):
    """
    Compute a 3D land–ocean mask (z, y, x) from salinity.

    Convention:
        mask = 1  -> ocean (salinity > eps)
        mask = 0  -> land  (salinity <= eps)

    Parameters
    ----------
    time : int
        Time index for salinity. Land/sea generally doesn't change, so time=0 is fine.
    quality : int
        OpenVisus quality parameter (controls resolution).
        Use the same quality as for your temperature data.
    eps : float
        Threshold: values <= eps are land, > eps are ocean.
        For strictly 0=land, use eps=0.0.
    use_surface_only : bool
        If False (default):
            Read full 3D salinity and threshold directly.
        If True:
            Read only surface salinity (z=[0,1]) and broadcast mask
            to all depths. This is more efficient if you know
            land/sea doesn't change with depth.

    Returns
    -------
    mask3d : np.ndarray
        Array of shape (nz, ny, nx), dtype uint8, with 0/1 values.
    """
    db_salt = get_salt_db()

    if not use_surface_only:
        # Full 3D read
        salt3d = db_salt.read(time=time, quality=quality)  # (nz, ny, nx)
        mask3d = (salt3d > eps).astype(np.uint8)
        return mask3d

    salt_surface = db_salt.read(time=time, quality=quality, z=[0, 1])  # (1, ny, nx)
    mask2d = (salt_surface[0] > eps).astype(np.uint8)

    salt3d = db_salt.read(time=time, quality=quality)
    nz = salt3d.shape[0]

    mask3d = np.broadcast_to(mask2d, (nz, *mask2d.shape)).astype(np.uint8)
    return mask3d


def apply_land_ocean_mask(data: np.ndarray, mask3d: np.ndarray, land_value=np.nan):
    """
    Apply a 3D land–ocean mask to a variable on the same grid.

    Parameters
    ----------
    data : np.ndarray
        Variable data with shape compatible with mask3d.
        Typical shapes:
            (nz, ny, nx)
            (nt, nz, ny, nx)  -> will broadcast if mask has (nz, ny, nx)
    mask3d : np.ndarray
        3D mask, shape (nz, ny, nx), values 0/1.
    land_value : scalar
        Value to assign to land points (e.g., np.nan or 0.0).

    Returns
    -------
    masked_data : np.ndarray
        Data with land points replaced by land_value.
    """
    # Ensure boolean mask for broadcasting
    ocean = (mask3d == 1)

    masked_data = np.where(ocean, data, land_value)
    return masked_data



if __name__ == "__main__":
    db_theta = open_llc2160("theta")

    time = 0
    quality = -18
    theta3d = db_theta.read(time=time, quality=quality)  # (nz, ny, nx)
    mask3d = compute_land_ocean_mask3d(time=time, quality=quality, eps=0.0,
                                     use_surface_only=False)
    print("theta3d shape:", theta3d.shape)
    print("mask3d shape:", mask3d.shape)

    theta3d_ocean = apply_land_ocean_mask(theta3d, mask3d, land_value=np.nan)
    print("Masked theta3d (ocean only) computed.")
