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
from datetime import datetime, timedelta
from idpi.operators.destagger import destagger
from idpi.operators.relhum import relhum
from idpi.grib_decoder import GribReader

from idpi import metadata

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

    if os.path.exists(zarr_path):
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

        return datetime.utcfromtimestamp(checkpoint.tolist() / 1e9)
    except:
        return None


def collect_datasets(dir: str, start: int, end: int, config: dict[str, str]):
    """Collects pickle datasets in order and archives
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
        filename = os.path.join(dir, str(x) + ".pckl")
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


def compute_first_date_avail(tar_file_paths: tuple[str]):
    """Compute the first date available in the list of tar files,
    extracted from filename pattern.

    Parameters
    ----------
    tar_file_paths: tuple[str]
        list of filenames of tar files
    """

    return datetime.strptime(
        os.path.basename(tar_file_paths[0]).replace(".tar", ""), "%Y%m%d"
    )


def process_tar_file(
    first_leadtime: int,
    tar_file_paths: tuple[str],
    num_tot_leadtimes: int,
    hour_start: int,
    outdir: str,
    tmp_base_dir: str,
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
            anads = process_ana_file(ana_full_path)
            fgds = process_fg_file(fg_full_path)

            # FG ref time is one hour before ref time of ANA
            # Therefore we remove these coordinates to avoid misalignment in zarr
            # We make valid_time the only (aligned) coordinate -> "time"
            anads = anads.drop_vars(["ref_time", "time"])
            fgds = fgds.drop_vars(["ref_time", "time"])
            anads = anads.rename({"valid_time": "time"})
            fgds = fgds.rename({"valid_time": "time"})

            serialize_dataset(
                fgds.merge(anads),
                first_leadtime + first_leadtime_of_day + index,
                outdir,
            )


def load_data(config: dict) -> None:
    """Load weather data from archive and store it in a Zarr archive."""

    n_pool = config["n_pool"]
    tar_file_paths = []
    regex = re.compile(".*\.list")
    for year in data_config["train_years"]:
        data_path = os.path.join(data_config["data_path"], "ANA" + year)
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if regex.match(file):
                    continue
                full_path = os.path.join(root, file)
                tar_file_paths.append(full_path)
    tar_file_paths.sort()

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

    filename = os.path.join(outdir, str(x) + ".pckl")
    logger.info(f"Writing to tmp file: {filename}")
    with open(filename + ".lock", "wb") as handle:
        pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.rename(filename + ".lock", filename)


def archive_dataset(ds: xr.Dataset, config: dict[str, str], logger: logging.Logger):
    """ "Archive the dataset into a zarr store"""

    for name, var in ds.items():
        var = var.chunk(chunks={"time": 1})
        var.encoding = {"compressor": config["compressor"]}
        var.time.encoding = {"dtype": "float64"}

        if "z" in var.dims and var.sizes["z"] == 1:
            var = var.squeeze(dim="z")
            var = var.drop_vars("z")

        var.attrs.pop("message", None)
        ds[name] = var

    append_or_create_zarr(ds, config, logger)


def process_ana_file(full_path: str):
    """Process the analysis file extracting and processing the require variables

    Parameters
    ----------
    full_path: str
        filename full path to analysis file.
    """
    try:
        reader = GribReader.from_files([full_path])
        ds = reader.load_fieldnames(
            [
                "T",
                "U_10M",
                "V_10M",
                "U",
                "V",
                "PS",
                "T_2M",
                "P",
                "QV",
                "TQV",
                "PMSL",
                "HHL",
                "HSURF",
            ],
        )

        metadata.set_origin_xy(ds, ref_param="HHL")

        pdset = {}
        for name, var in ds.items():

            for dim in ["x", "y", "z"]:
                origind = "origin_" + dim
                if origind in var.attrs and var.attrs[origind] != 0.0:
                    var = destagger(var, dim)
                    var.attrs[origind] = 0.0
            if name == "HHL":
                name = "HFL"

            pdset[name] = var

        pdset["RELHUM"] = relhum(
            pdset["QV"], pdset["T"], pdset["P"], clipping=True, phase="water"
        )
        pdset["FI"] = pdset["HFL"] * 9.80665

        logger.info(f"Processed analysis file: {full_path}")
        return xr.Dataset(pdset)
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error: {e}")


def process_fg_file(full_path: str):
    """Process the first guess file extracting and processing the require variables

    Parameters
    ----------
    full_path: str
        filename full path to first guess file.

    """
    try:
        reader = GribReader.from_files([full_path])
        ds = reader.load_fieldnames(
            ["TOT_PREC"],
        )

        logger.info(f"Processed first guess file: {full_path}")
        return xr.Dataset(ds)
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
        default=os.path.join("scratch/", os.environ["USER"] + "/"),
    )
    parser.add_argument(
        "-o",
        type=str,
        default=os.path.join(
            "/scratch", os.environ["USER"], "/neural-lam/zarr/cosmo_ml_data.zarr"
        ),
    )

    args = parser.parse_args()

    data_config = {
        "data_path": "/store/archive/mch/msopr/owm/KENDA",
        "train_years": ["15", "16", "17", "18", "19", "20"],
        "zarr_path": args.o,
        "n_pool": args.n,
        "tempdir": args.tempdir,
        "compressor": numcodecs.Blosc(
            cname="lz4", clevel=7, shuffle=numcodecs.Blosc.SHUFFLE
        ),
    }

    load_data(data_config)
