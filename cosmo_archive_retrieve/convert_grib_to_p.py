# Standard library
import argparse
import os
import re
import shutil

import numcodecs
import logging
import pickle
import multiprocessing as mp
import traceback
import time

# Third-party
import xarray as xr
from functools import partial
from tqdm import tqdm
import os
import multiprocessing as mp

import tempfile
from pathlib import Path
import math
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import idpi
from idpi.operators.destagger import destagger
from idpi.operators.relhum import relhum
from idpi.operators.vertical_interpolation import interpolate_k2p
from idpi import metadata, data_source, grib_decoder

logger = logging.getLogger(__name__)


def load_data(config: dict) -> None:
    """Load weather data from archive and store it in a Zarr archive.

    Parameters
    ----------
    config: dict
        main configuration of the application
    """

    regex = re.compile("laf.*")
    files_list = []
    for year in config["train_years"]:
        data_path = os.path.join(config["data_path"], "ANA" + year,'det')
        for root, _, files in os.walk(data_path):
            for filename in files:
                if regex.match(filename):
                    files_list.append(os.path.join(root, filename))
    process_ana_grib_file_c = partial(
            process_ana_grib_file,
            config=config,
        )   
 
    with mp.Pool(config["n_pool"]) as p:
        p.map(process_ana_grib_file_c, files_list)


def check_hypercube(dset: dict[xr.DataArray]) -> None:
    dims = {}
    for field, da in dset.items():
        for dim, length in zip(da.dims, da.shape):
            if dim in dims and dims[dim] != length:
                raise RuntimeError(
                    f"Dimension {dim} of hypercube (for {field}) not aligned between arrays. Field has size {length} while hypercube contains {dims[dim]}"
                )
            else:
                dims[dim] = length


def adapt_coords_to_zarr(
    name: str, var: xr.DataArray, full_path: str, file_type: Literal["ANA", "FG"]
) -> xr.DataArray:
    """Adapt the coordinates of the xarray loaded from grib in order to prepare it for writing the zarr dataset.
    A single time coordinate is generated with the datetime of the data (i.e. equivalent to valid_time),
    eliminating valid_time and ref_time. Also we check that this coordinate is consistent with the filename,
    since there are (rare) cases where the encoding of the step is wrong in the archive.
    Eliminate z dimension from 2d fields

    Parameters
    ----------
    name: str
        name of variable
    var: xr.DataArray
        variable
    full_path: str
        filename full path with grib data

    Returns
    -------
    xr.DataArray
        An array with coords adapted to write to zarr

    """

    # remove z dim for all 2d var in order to be able to create a dataset
    if "z" in var.sizes and var.sizes["z"] == 1:
        var = var.squeeze(dim="z")

    # The analysis files should all be generated a 'step' = 0. For unknown reasons,
    # laf2016031421 was generated with some variables on step 1. This generates variables (P & PP)
    # with unaligned time coordinates. We can not fix the archive data, so we hack it here.
    exp_step = 0 if file_type == "ANA" else 1
    if var.time.values != [exp_step]:
        var = var.assign_coords({"time": [exp_step]})
        logger.warning(f"Found a wrong 'step' (!={exp_step}) for {name} in {full_path}")

    # Due to the issue that step and ref_time is wrong (at last in one file of the archive)
    # we double check that the valid_time is as expected
    valid_time = var.coords["valid_time"].values
    filename = Path(full_path).name.replace("laf" if file_type == "ANA" else "lff", "")
    if (
        datetime.utcfromtimestamp(valid_time[0].astype(int) * 1e-9).strftime("%Y%m%d%H")
        != filename
    ):
        raise RuntimeError("Wrong valid time")

    # FG ref time is one hour before ref time of ANA
    # Therefore we remove these coordinates to avoid misalignment in zarr
    # We make valid_time the only (aligned) coordinate -> "time"
    var = var.drop_vars(["ref_time", "valid_time" ])#"time", "valid_time"])

#    var = var.assign_coords(
#        {
#            "time": xr.Variable(
#                "time",
#                valid_time,
#                encoding={"units": "hours since 1900-01-01"},
#                # xr.coding.times.decode_cf_datetime(valid_time),
#                # encoding={"dtype": "datetime64[ns]"},
#            )
#        }
#    )

    return var


def process_ana_grib_file(full_path: str, config):
    """Process the analysis file extracting and processing the require variables

    Parameters
    ----------
    full_path: str
        filename full path to analysis file.
    """

    out_params = {
        "T",
        "U_10M",
        "V_10M",
        "U",
        "V",
        "W",
        "T_2M",
        "QV",
        "TQV",
        "PMSL",
        "FI",
        "CLCT",
        "W_SNOW",
        "TD_2M",
    }

    logger.info(f"Processing analysis file: {full_path}")

    try:
        ds = idpi.grib_decoder.load(
            idpi.data_source.DataSource(datafiles=[full_path]),
            {
                "param": [
                    "T",
                    "U_10M",
                    "V_10M",
                    "U",
                    "V",
                    "W",
                    "PS",
                    "T_2M",
                    "QV",
                    "TQV",
                    "PMSL",
                    "HHL",
                    "HSURF",
                    "PP",
                    "P",
                    "CLCT",
                    "W_SNOW",
                    "TD_2M",
                ]
            },
        )

        idpi.metadata.set_origin_xy(ds, ref_param="T")

        pdset = {}

        for name, var in ds.items():
            for dim in ["x", "y"]:
                origind = "origin_" + dim
                if origind in var.attrs and var.attrs[origind] != 0.0:
                    old_coords = {}
                    for coord in ("valid_time", "ref_time"):
                        old_coords[coord] = var.coords[coord]
                    var = destagger(var, dim)
                    var = var.assign_coords(old_coords)

            if name == "HHL" or name == "W":
                # workaround until the typeOflevel used in the archive is supported (hybrid)
                # https://github.com/MeteoSwiss-APN/icon_data_processing_incubator/issues/131
#                var.attrs["origin_z"] = -0.5
#                var.attrs["vcoord_type"] = "model_level"
                old_coords = {}
                for coord in ("valid_time", "ref_time"):
                    old_coords[coord] = var.coords[coord]

                var = destagger(var, "z")
                var = var.assign_coords(old_coords)

            if name == "HHL":
                name = "HFL"

            var = adapt_coords_to_zarr(name, var, full_path, "ANA")
            pdset[name] = var

        pdset["FI"] = pdset["HFL"] * 9.80665
        pdset["FI"] = pdset["FI"].assign_attrs(pdset["P"].attrs)
        pdset["FI"] = pdset["FI"].assign_coords(pdset["P"].coords)
        

        pdset["FI"].attrs |= metadata.override(
                    pdset["FI"].message, shortName="FI"
        )



        for name in out_params:
            logger.info("Dumping "+ name)
            var = pdset[name]
            if 'z' in var.dims and 'time' in var.dims:
                var = interpolate_k2p(var, "linear_in_lnp", ds["P"], [50,100,150,200,250,300, 400,500,600,700,850,925,1000], "hPa")
                var.attrs |= metadata.override(var.message, edition="2")

                var.attrs |= metadata.override(
                    var.message, typeOfLevel="hybridPressure"
                )


            if 'time' in var.dims:
                with data_source.cosmo_grib_defs():
                    pathname=os.path.join(config["zarr_path"], os.path.basename(full_path))
                    logger.info("OOOOO"+ pathname)
                    with open(pathname, "a+b") as tmp:
                        grib_decoder.save(var, tmp)


        logger.info(f"Processed analysis file: {full_path}")

        check_hypercube(pdset)

        if not out_params.issubset(pdset):
            raise RuntimeError(
                f"Missing output parameter {out_params} in dataset {pdset.keys()}"
            )

    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error: {e}")


def process_fg_file(full_path: str) -> xr.Dataset:
    """Process the first guess file extracting and processing the require variables

    Parameters
    ----------
    full_path: str
        filename full path to first guess file.

    """
    try:
        ds = idpi.grib_decoder.load(
            idpi.data_source.DataSource(datafiles=[full_path]),
            {
                "param": [
                    "TOT_PREC",
                    # Net long wave radiation flux (m) (at the surface)
                    "ATHB_S",
                    # Latent Heat Net Flux (m)
                    "ALHFL_S",
                    # Sensible Heat Net Flux (m)
                    "ASHFL_S",
                    # Net short wave radiation flux (at the surface)
                    "ASOB_S",
                    "DURSUN",
                    "VMAX_10M",
                    "UMAX_10M",
                ]
            },
        )

        logger.info(f"Processed first guess file: {full_path}")

        pdset = {
            name: adapt_coords_to_zarr(name, var, full_path, "FG")
            for name, var in ds.items()
        }

        check_hypercube(pdset)
        return xr.Dataset(pdset)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Create a zarr archive.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument(
        "-t",
        "--tempdir",
        type=str,
        default=os.path.join("/scratch/", os.environ["USER"] + "/"),
    )
    parser.add_argument(
        "-o",
        type=str,
        default=os.path.join("/scratch/cosuna/mldata/pl/"),
    )

    args = parser.parse_args()

    data_config = {
        "data_path": "/store/mch/msopr/osm/KENDA-1/",
        "train_years": ["22"],       
        "zarr_path": args.o,
        "n_pool": args.n,
        "tempdir": args.tempdir,
        "compressor": numcodecs.Blosc(
            cname="lz4", clevel=7, shuffle=numcodecs.Blosc.SHUFFLE
        ),
    }

    load_data(data_config)
