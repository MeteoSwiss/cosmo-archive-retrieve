import xarray as xr
from cosmo_archive_retrieve import create_zarr_archive


def test_data_proce():
    """Test the processing of analysis and first guess data"""

    ds = create_zarr_archive.process_ana_file(
        "/project/s83c/cosuna/cosmo-archive-retrieve/input/laf2015112800"
    ).isel(time=0)
    ref = xr.open_zarr("/project/s83c/cosuna/cosmo-archive-retrieve/v1/data.zarr")

    for var in ds:
        xr.testing.assert_allclose(ds[var], ref[var])
    ds = create_zarr_archive.process_fg_file(
        "/project/s83c/cosuna/cosmo-archive-retrieve/input/lff2015112800"
    ).isel(time=0)
    for var in ds:
        xr.testing.assert_allclose(ds[var], ref[var])
