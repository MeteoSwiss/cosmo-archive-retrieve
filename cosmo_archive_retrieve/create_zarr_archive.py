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

logger = logging.getLogger(__name__)


def append_or_create_zarr(
    data_out: xr.Dataset, config: dict, logger: logging.Logger
) -> None:
    """Append data to an existing Zarr archive or create a new one.

    Parameters
    ----------
    data_out: xr.Dataset
        dataset to be written in zarr.
    config: dict[str,str]
        configuration of application
    logger:
        logger
    """

    zarr_path = config["zarr_path"]

    logger.info(f"Archiving into zarr:{zarr_path}")

    if os.path.isdir(zarr_path):
        data_out.to_zarr(
            store=zarr_path,
            mode="a",
            consolidated=True,
            append_dim="time",
        )
    else:
        data_out.to_zarr(
            zarr_path,
            mode="w",
            consolidated=True,
        )


def find_last_checkpoint(data_path: str):
    """Find the last leadtime that was generated in the zarr store.

    Parameters
    ----------
    data_path: str
        path to zarr data store
    """

    try:
        ds = xr.open_zarr(data_path)
        checkpoint = ds.time.values[-1]

        return datetime.fromtimestamp(checkpoint.tolist() / 1e9, timezone.utc)
    except:
        return None


def collect_datasets(dir: str, start: int, end: int, config: dict[str, str]):
    """Collect pickled datasets in order and archives
    them into a single zarr store.

    Parameters
    ----------
    dir: str
        Directory where to find the pickle datasets
    start: int
        first leadtime generated.
    end: int
        last leadtime generated.
    config:
        configuration of the application.
    """

    logger = logging.getLogger("COLLECTOR")

    logger.info("Start process for collecting tmp files")
    for x in range(start, end + 1):
        filename = os.path.join(dir, str(x) + ".pickle")
        logger.info(f"waiting for file {filename}")
        while not os.path.exists(filename):
            time.sleep(0.2)
        with open(filename, "rb") as handle:
            logger.info(f"loading file {filename}")
            ds = pickle.load(handle)

        os.remove(filename)

        archive_dataset(ds, config, logger)


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def compute_first_date_avail(tar_file_paths: tuple[str]) -> datetime:
    """Compute the first date available in the list of tar files,
    extracted from filename pattern.

    Parameters
    ----------
    tar_file_paths: tuple[str]
        list of filenames of tar files

    Returns
    -------
    datetime
        first date found in the requested period of the archive.
    """

    return datetime.strptime(
        Path(tar_file_paths[0].replace(".tar", "")).name, "%Y%m%d"
    ).replace(tzinfo=timezone.utc)


def process_tar_file(
    first_leadtime: int,
    tar_file_paths: tuple[str],
    num_tot_leadtimes: int,
    hour_start: int,
    outdir: str,
    tmp_base_dir: str,
    file_type:str
):
    """Process an entire tar file from the archive, which contains 24 leadtimes.
    The output extracted dataset is stored in temporary pickle files.

    Parameters
    ----------
    first_leadtime: int
        first leadtime to process from the tar file. Typically a multiple of 24, but it might not
        be the case in case of a restart from a particular leadtime checkpoint.
    tar_file_paths: tuple[str]
        tuple of all tar files from the requested period.
    num_tot_leadtimes: int
        total number of leadtimes contained in the list of tar files.
    hour_start: int
        first hour where the processing start. Typically 0 unless starting from a checkpoint.
    outdir: str
        output directory where to store the processed dataset.
    tmp_base_dir: str
        base directory for creation of temporary directories.
    file_type: str
        whether the files to process are analysis or forecast files
    """

    if first_leadtime > num_tot_leadtimes:
        return

    with tempfile.TemporaryDirectory(prefix=tmp_base_dir) as tarextract_dir:

        ifile_start = math.floor(first_leadtime / 24)
        first_hour = first_leadtime % 24
        first_date_avail = compute_first_date_avail(tar_file_paths)
        first_date = first_date_avail + timedelta(days=ifile_start)
        logger.info(f"Extracting leadtime: {first_leadtime}, date: {first_date}")

        group_ana_files = []
        group_fg_files = []

        ana_file = tar_file_paths[ifile_start]
        logger.info(
            f"Copying from archive ({tarextract_dir}) to {os.path.join(tarextract_dir, Path(ana_file).name)}"
        )
        tmp_tar_file = os.path.join(tarextract_dir, Path(ana_file).name)
        shutil.copyfile(ana_file, tmp_tar_file)

        with tarfile.open(tmp_tar_file) as tar:
            tar.extractall(path=os.path.join(tarextract_dir, "ANA"))

        fg_file = re.sub(r"KENDA\/ANA(\d{2})", r"KENDA/FG\1", ana_file)
        shutil.copyfile(fg_file, tmp_tar_file)

        with tarfile.open(tmp_tar_file) as tar:
            tar.extractall(path=os.path.join(tarextract_dir, "FG"))

        first_leadtime_of_day = 0
        # on the first iteration we need to make sure we start on the actual hour after
        # the checkpoint, instead of the first hour of this day.
        if first_leadtime < hour_start:
            first_leadtime_of_day = hour_start % 24

        for hour in range(first_leadtime_of_day, 24):
            tdate = first_date + timedelta(hours=hour + first_hour)
            group_ana_files.append(
                os.path.join(
                    tarextract_dir,
                    "ANA",
                    f"{tdate.strftime('%Y%m%d')}/det/laf{tdate.strftime('%Y%m%d%H')}",
                )
            )
            group_fg_files.append(
                os.path.join(
                    tarextract_dir,
                    "FG",
                    f"{tdate.strftime('%Y%m%d')}/det/lff{tdate.strftime('%Y%m%d%H')}",
                )
            )


        for index, (ana_full_path, fg_full_path) in enumerate(
            zip(group_ana_files, group_fg_files)
        ):
            datasets = process_file(ana_full_path, file_type)
            first_guess_datasets = process_fg_file(fg_full_path)

            # And then here based on the decision, do either - right now let's just plug in and replace 
            serialize_dataset(
                first_guess_datasets.merge(datasets),
                first_leadtime + first_leadtime_of_day + index,
                outdir,
            )


def get_archive_filenames_list(config: dict) -> list[str]:
    """Get a list of (tar) filenames of the archive to be extracted.

    Parameters
    ----------
    config: dict
        main configuration of the application

    Returns
    -------
    list[str]
        list of filenames of the archive
    """
    tar_file_paths = []
    regex = re.compile(".*\.list")
    for year in config["train_years"]:
        data_path = os.path.join(config["data_path"], config["file_type"] + year)
        for root, _, files in os.walk(data_path):
            for file in files:
                if regex.match(file):
                    continue
                full_path = os.path.join(root, file)
                tar_file_paths.append(full_path)
    tar_file_paths.sort()
    return tar_file_paths


def load_data(config: dict) -> None:
    """Load weather data from archive and store it in a Zarr archive.

    Parameters
    ----------
    config: dict
        main configuration of the application
    """

    tar_file_paths = get_archive_filenames_list(config)
    num_tot_leadtimes = len(tar_file_paths) * 24
    first_date_avail = compute_first_date_avail(tar_file_paths)

    logger.info(f"Initial date available in archive: {first_date_avail}")
    logger.info(f"Total number of leadtimes: {num_tot_leadtimes}")

    zarr_checkpoint = find_last_checkpoint(config["zarr_path"])

    if zarr_checkpoint:
        checkpoint_hours_from_start = int(
            (zarr_checkpoint - first_date_avail).total_seconds() / 3600
        )
        hour_start = checkpoint_hours_from_start + 1
        logger.info(f"Found a checkpoint at time :{zarr_checkpoint}")

    else:
        hour_start = 0

    logger.info(f"Starting from hour :{hour_start}")


    # Ensure the temporary directory exists
    if not os.path.exists(config["tempdir"]):
        os.makedirs(config["tempdir"])

    with tempfile.TemporaryDirectory(prefix=config["tempdir"]) as tmpdir:
        data_collector = Process(
            target=collect_datasets,
            args=(tmpdir, hour_start, num_tot_leadtimes, config),
        )
        data_collector.start()

        tar_file_call = partial(
            process_tar_file,
            tar_file_paths=tar_file_paths,
            num_tot_leadtimes=num_tot_leadtimes,
            hour_start=hour_start,
            outdir=tmpdir,
            tmp_base_dir=config["tempdir"],
        )

        # iterate over all the leadtimes (hours) in the archive day by day
        n_pool = config["n_pool"]
        for x in tqdm(
            range(math.floor(hour_start / 24) * 24, num_tot_leadtimes + 1, 24 * n_pool)
        ):
            with mp.Pool(n_pool) as p:
                p.map(tar_file_call, [x + w * 24 for w in range(n_pool)])

            if data_collector.exception:
                error, traceback = data_collector.exception
                logger.error("Error on child collector:" + traceback)
                raise error
        data_collector.join()


def serialize_dataset(ds: xr.Dataset, x: int, outdir: str):
    """The (parallel) processed files from the archive are stored into temporary pickle files,
    which are later sequentially picked for generating the zarr storage.

    Parameters
    ----------
    ds: xr.Dataset
        dataset to be serialized
    x: int
        leadtime number
    outdir: str
        output directory
    """

    filename = os.path.join(outdir, str(x) + ".pickle")
    logger.info(f"Writing to tmp file: {filename}")
    with open(filename + ".lock", "wb") as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(filename + ".lock", filename)


def archive_dataset(ds: xr.Dataset, config: dict[str, str], logger: logging.Logger):
    """Archive the dataset into a zarr store"""

    for name, var in ds.items():
        if "time" in var.sizes:
            var = var.chunk(chunks={"time": 1})
        var.encoding = {"compressor": config["compressor"]}

        var.attrs.pop("message", None)
        ds[name] = var

    append_or_create_zarr(ds, config, logger)


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
    var = var.drop_vars(["ref_time", "time", "valid_time"])

    var = var.assign_coords(
        {
            "time": xr.Variable(
                "time",
                valid_time,
                encoding={"units": "hours since 1900-01-01"},
                # xr.coding.times.decode_cf_datetime(valid_time),
                # encoding={"dtype": "datetime64[ns]"},
            )
        }
    )

    return var

def process_file(full_path: str, file_type: str):
    """Process the file extracting and processing the required variables based on file type.

    Parameters
    ----------
    full_path: str
        Filename full path to the file.
    file_type: str
        Type of the file ('ANA' or 'FG').
    """

    if file_type == "ANA":
        out_params = constants.ANA_PARAMS
        param_list = list(out_params)
        log_message = "Processing analysis file"
    elif file_type == "FG":
        out_params = constants.FORECAST_PARAMS
        param_list = list(constants.FORECAST_PARAMS)
        log_message = "Processing forecast file"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    logger.info(f"{log_message}: {full_path}")

    try:
        ds = idpi.grib_decoder.load(
            idpi.data_source.DataSource(datafiles=[full_path]),
            {"param": param_list},
        )

        idpi.metadata.set_origin_xy(ds, ref_param="T")

        pdset = {}
        for name, var in ds.items():

            for dim in ["x", "y"]:
                origind = "origin_" + dim
                if origind in var.attrs and var.attrs[origind] != 0.0:
                    old_coords = {coord: var.coords[coord] for coord in ("valid_time", "ref_time")}
                    var = destagger(var, dim)
                    var = var.assign_coords(old_coords)

            if name == "HHL" or name == "W":
                var.attrs.update({"origin_z": -0.5, "vcoord_type": "model_level"})
                old_coords = {coord: var.coords[coord] for coord in ("valid_time", "ref_time")}
                var = destagger(var, "z")
                var = var.assign_coords(old_coords)

            if name == "HHL":
                name = "HFL"

            var = adapt_coords_to_zarr(name, var, full_path, file_type)

            if name in {"HFL", "HSURF"}:
                var = var.squeeze(dim="time")

            pdset[name] = var

        pdset["FI"] = pdset["HFL"] * 9.80665
        pdset["P0FL"] = (pdset["P"] - pdset["PP"]).squeeze(dim="time")
        pdset["RELHUM"] = relhum(pdset["QV"], pdset["T"], pdset["P"], clipping=True, phase="water")

        logger.info(f"Processed {file_type.lower()} file: {full_path}")

        check_hypercube(pdset)

        if not out_params.issubset(pdset):
            raise RuntimeError(f"Missing output parameter {out_params} in dataset {pdset.keys()}")
        
        dset = xr.Dataset({x: y for x, y in pdset.items() if x in out_params})

        return dset

    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error: {e}")



# def process_ana_file(full_path: str):
#     """Process the analysis file extracting and processing the require variables

#     Parameters
#     ----------
#     full_path: str
#         filename full path to analysis file.
#     """

#     out_params = {
#         "T",
#         "U_10M",
#         "V_10M",
#         "U",
#         "V",
#         "W",
#         "PS",
#         "T_2M",
#         "QV",
#         "TQV",
#         "PMSL",
#         "FI",
#         "HSURF",
#         "PP",
#         "P0FL",
#         "RELHUM",
#         "CLCT",
#         "W_SNOW",
#         "TD_2M",
#     }

#     logger.info(f"Processing analysis file: {full_path}")

#     try:
#         ds = idpi.grib_decoder.load(
#             idpi.data_source.DataSource(datafiles=[full_path]),
#             {
#                 "param": [
#                     "T",
#                     "U_10M",
#                     "V_10M",
#                     "U",
#                     "V",
#                     "W",
#                     "PS",
#                     "T_2M",
#                     "QV",
#                     "TQV",
#                     "PMSL",
#                     "HHL",
#                     "HSURF",
#                     "PP",
#                     "P",
#                     "CLCT",
#                     "W_SNOW",
#                     "TD_2M",
#                 ]
#             },
#         )

#         idpi.metadata.set_origin_xy(ds, ref_param="T")

#         pdset = {}
#         for name, var in ds.items():

#             for dim in ["x", "y"]:
#                 origind = "origin_" + dim
#                 if origind in var.attrs and var.attrs[origind] != 0.0:
#                     old_coords = {}
#                     for coord in ("valid_time", "ref_time"):
#                         old_coords[coord] = var.coords[coord]
#                     var = destagger(var, dim)
#                     var = var.assign_coords(old_coords)

#             if name == "HHL" or name == "W":
#                 # workaround until the typeOflevel used in the archive is supported (hybrid)
#                 # https://github.com/MeteoSwiss-APN/icon_data_processing_incubator/issues/131
#                 var.attrs["origin_z"] = -0.5
#                 var.attrs["vcoord_type"] = "model_level"
#                 for coord in ("valid_time", "ref_time"):
#                     old_coords[coord] = var.coords[coord]

#                 var = destagger(var, "z")
#                 var = var.assign_coords(old_coords)

#             if name == "HHL":
#                 name = "HFL"

#             var = adapt_coords_to_zarr(name, var, full_path, "ANA")

#             if name == "HFL" or name == "HSURF":
#                 var = var.squeeze(dim="time")

#             pdset[name] = var

#         pdset["FI"] = pdset["HFL"] * 9.80665
#         # P0FL should be a time invariant
#         pdset["P0FL"] = (pdset["P"] - pdset["PP"]).squeeze(dim="time")
#         pdset["RELHUM"] = relhum(
#             pdset["QV"], pdset["T"], pdset["P"], clipping=True, phase="water"
#         )

#         logger.info(f"Processed analysis file: {full_path}")

#         check_hypercube(pdset)

#         if not out_params.issubset(pdset):
#             raise RuntimeError(
#                 f"Missing output parameter {out_params} in dataset {pdset.keys()}"
#             )
#         # Return only the fields in out_params
#         dset = xr.Dataset({x: y for x, y in pdset.items() if x in out_params})

#         return dset

#     except (FileNotFoundError, OSError) as e:
#         logger.error(f"Error: {e}")


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


# def process_forecast_file(full_path: str):
#     """Process the forecast file extracting and processing the require variables

#     Parameters
#     ----------
#     full_path: str
#         filename full path to analysis file.
#     """


#     out_params = constants.FORECAST_PARAMS

#     logger.info(f"Processing forecast file: {full_path}")

#     try:
#         ds = idpi.grib_decoder.load(
#             idpi.data_source.DataSource(datafiles=[full_path]),
#             {
#                 "param": list(constants.FORECAST_PARAMS)
#             },
#         )

#         idpi.metadata.set_origin_xy(ds, ref_param="T")

#         pdset = {}
#         for name, var in ds.items():

#             for dim in ["x", "y"]:
#                 origind = "origin_" + dim
#                 if origind in var.attrs and var.attrs[origind] != 0.0:
#                     old_coords = {}
#                     for coord in ("valid_time", "ref_time"):
#                         old_coords[coord] = var.coords[coord]
#                     var = destagger(var, dim)
#                     var = var.assign_coords(old_coords)

#             if name == "HHL" or name == "W":
#                 # workaround until the typeOflevel used in the archive is supported (hybrid)
#                 # https://github.com/MeteoSwiss-APN/icon_data_processing_incubator/issues/131
#                 var.attrs["origin_z"] = -0.5
#                 var.attrs["vcoord_type"] = "model_level"
#                 for coord in ("valid_time", "ref_time"):
#                     old_coords[coord] = var.coords[coord]

#                 var = destagger(var, "z")
#                 var = var.assign_coords(old_coords)

#             if name == "HHL":
#                 name = "HFL"

#             var = adapt_coords_to_zarr(name, var, full_path, "FG")

#             if name == "HFL" or name == "HSURF":
#                 var = var.squeeze(dim="time")

#             pdset[name] = var

#         pdset["FI"] = pdset["HFL"] * 9.80665
#         # P0FL should be a time invariant
#         pdset["P0FL"] = (pdset["P"] - pdset["PP"]).squeeze(dim="time")
#         pdset["RELHUM"] = relhum(
#             pdset["QV"], pdset["T"], pdset["P"], clipping=True, phase="water"
#         )

#         logger.info(f"Processed analysis file: {full_path}")

#         check_hypercube(pdset)

#         if not out_params.issubset(pdset):
#             raise RuntimeError(
#                 f"Missing output parameter {out_params} in dataset {pdset.keys()}"
#             )
#         # Return only the fields in out_params
#         dset = xr.Dataset({x: y for x, y in pdset.items() if x in out_params})

#         return dset

#     except (FileNotFoundError, OSError) as e:
#         logger.error(f"Error: {e}")

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
        default= "/users/clechart/cosmo-archive-retrieve/cosmo_archive_retrieve",
    )
    parser.add_argument(
        "-f",
        "--file_type",
        choices=["ANA", "FG"],
        type=str,
        required=True,
        help="Specify the file type: ANA or FG"
    )
    args = parser.parse_args()

    data_config = {
        "data_path": "/users/clechart/cosmo-archive-retrieve/input",
        "train_years": ["15", "16", "17", "18", "19", "20"],
        "zarr_path": args.o,
        "file_type": args.file_type,
        "n_pool": args.n,
        "tempdir": args.tempdir,
        "compressor": numcodecs.Blosc(
            cname="lz4", clevel=7, shuffle=numcodecs.Blosc.SHUFFLE
        ),
    }

    load_data(data_config)

