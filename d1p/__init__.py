"""
d1p
===

This package implements a proof-of-concept of a Day-1 precipitation (D1P) product for the TROPICS
TMS sensors. The principal functionality provided by this package is conversion of TROPICS TMS
L1C files to ATMS L1C, so that they can be processed using the GPROF ATMS retrieval.
"""
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from pansat.time import to_datetime
import h5py
import numpy as np
import xarray as xr
from pathlib import Path


LOGGER = logging.getLogger(__name__)


ANGLE_BINS = np.array([
    -65.1314621 , -63.41350555, -61.72231293, -60.08193207,
    -58.48461151, -56.9211731 , -55.38699341, -53.87939453,
    -52.39673996, -50.93818665, -49.49941635, -48.07765198,
    -46.67004395, -45.27391052, -43.89008331, -42.51745987,
    -41.15645218, -39.8058548 , -38.46263885, -37.1254425 ,
    -35.79325485, -34.46810532, -33.15164185, -31.84277725,
    -30.53985405, -29.24048996, -27.94439697, -26.65211105,
    -25.36318588, -24.07855988, -22.79799271, -21.52083015,
    -20.24492645, -18.96933365, -17.69556046, -16.42426872,
    -15.15733814, -13.89472389, -12.63358688, -11.37255001,
    -10.11113453,  -8.85024261,  -7.59117746,  -6.33394861,
    -5.07801294,  -3.82180023,  -2.56380844,  -1.30583858,
    0.        ,   1.30583858,   2.56380844,   3.82180023,
    5.07801294,   6.33394861,   7.59117746,   8.85024261,
    10.11113453,  11.37255001,  12.63358688,  13.89472389,
    15.15733814,  16.42426872,  17.69556046,  18.96933365,
    20.24492645,  21.52083015,  22.79799271,  24.07855988,
    25.36318588,  26.65211105,  27.94439697,  29.24048996,
    30.53985405,  31.84277725,  33.15164185,  34.46810532,
    35.79325485,  37.1254425 ,  38.46263885,  39.8058548 ,
    41.15645218,  42.51745987,  43.89008331,  45.27391052,
    46.67004395,  48.07765198,  49.49941635,  50.93818665,
    52.39673996,  53.87939453,  55.38699341,  56.9211731 ,
    58.48461151,  60.08193207,  61.72231293,  63.41350555,
    65.1314621
])

ATMS_CHANNELS = {
    0: 1,
    1: 1,
    2: 1,
    3: 6
}

TROPICS_CHANNELS = [
    [None],
    [None],
    [(0, 0)],
    [(2, 0), (1, 2), None, (1, 1), None, (1, 0)]
]

FILE_HEADER = (
    "DOI=10.5067/GPM/ATMS/NPP/1C/07;\nDOIauthority=http://dx.doi.org/;\nDOIshortName=1CNPPATMS;"
    "\nAlgorithmID=1CATMS;\nAlgorithmVersion=2019-V;\n"
    "FileName={filename};"
    "\nSatelliteName={platform};\nInstrumentName={sensor};\nGenerationDateTime=2023-04-17T10:00:35.000Z;"
    "\nStartGranuleDateTime=2023-04-12T00:38:20.000Z;\nStopGranuleDateTime=2023-04-12T02:19:50.000Z;"
    "\nGranuleNumber=059355;\nNumberOfSwaths=4;\nNumberOfGrids=0;\nGranuleStart=SOUTHERNMOST_LATITUDE;"
    "\nTimeInterval=ORBIT;\nProcessingSystem=PPS;\nProductVersion=V07A;\nEmptyGranule=NOT_EMPTY;\nMissingData=36;\n"
)

SWATH_HEADER = (
    "NumberScansInSet=1;\nMaximumNumberScansTotal={n_scans};\nNumberScansBeforeGranule=0;\nNumberScansGranule={n_scans};"
    "\nNumberScansAfterGranule=0;\nNumberPixels=96;\nScanType=CROSSTRACK;\n"
)

INPUT_RECORD = (
    b'InputFileNames=1Base.TROPICS03.TMS.TB2023.20230609-S063649-E081213.000214.PAR702.HDF5,TROPICS03.BRTT.L1B.Orbit00213.V04-01.ST20230609-070039.ET20230609-083604.CT20231025-144703.nc;\nInputAlgorithmVersions=2023,04.01.01;\nInputGenerationDateTimes=2023-11-20T19:25:04.000Z,2023-Oct-25T14:47:03.726 UTC;\n'
)

NAVIGATION_RECORD = (
    b'LongitudeOnEquator=115.244881;\nUTCDateTimeOnEquator=2023-06-09T07:00:39.538Z;\nMeanSolarBetaAngle=40.616333;\nEphemerisFileName=;\nAttitudeFileName=;\nGeoControlFileName=;\nEphemerisSource=;\nAttitudeSource=;\nGeoToolkitVersion=;\nSensorAlignmentFirstRotationAngle=;\nSensorAlignmentSecondRotationAngle=;\nSensorAlignmentThirdRotationAngle=;\nSensorAlignmentFirstRotationAxis=;\nSensorAlignmentSecondRotationAxis=;\nSensorAlignmentThirdRotationAxis=;\n'
)

FILE_INFO = (
b'DataFormatVersion=7a;\nTKCodeBuildVersion=0;\nMetadataVersion=7a;\nFormatPackage=HDF5-1.10.9;\nBlueprintFilename=GPM.V7.1CTMS.blueprint.xml;\nBlueprintVersion=BV_69;\nTKIOVersion=3.100.1;\nMetadataStyle=PVL;\nEndianType=LITTLE_ENDIAN;\n'
)
XCAL_INFO = (
    b'CalibrationStandard=GPM GMI V07 Tb;\nCalibrationTable=1C.TROPICS03.TMS.XCAL2023-N.tbl;\nCalibrationLevel=N (None);\n'
)


def create_atms_file(
        path: Path,
        platform: str,
        sensor: str,
        data: xr.Dataset
) -> Path:
    """
    Create ATMS files from TMS L1C data.

    Args:
        Path: Path object pointing to the folder to which to write the ATMS
            L1C file.
        platform: The platform name to use in the filename and header.
        sensor: The sensor name to use in the filename and hedaer.
        data: An xarray.Dataset containing the L1C data.

    Return:
        The path of the create L1C file.
    """
    start_time = to_datetime(data["scan_time"][0].data)
    end_time = to_datetime(data["scan_time"][-1].data)
    granule = data.attrs["GranuleNumber"]

    filename = (
    f"1C.{platform}.{sensor}.XCAL2019-V.{start_time.strftime('%Y%m%d-S%h%m%s')}"
    f"-E{end_time.strftime('%H%M%S')}.{granule:06}.V07A.HDF5"
    )
    output_path = Path(path) / filename

    n_scans = data.scans.size

    n_pixels = 96
    with h5py.File(output_path, "w") as output:

        output.attrs["FileHeader"] = np.array(FILE_HEADER.format(
            filename=filename,
            platform=platform,
            sensor=sensor
        ).encode())
        output.attrs["InputRecord"] = np.array(INPUT_RECORD)
        output.attrs["NavigationRecord"] = np.array(NAVIGATION_RECORD)
        output.attrs["FileInfo"] = np.array(FILE_INFO)
        output.attrs["XCALinfo"] = np.array(XCAL_INFO)

        for atms_swath, inds in enumerate(TROPICS_CHANNELS):
            group = output.create_group(f"S{atms_swath + 1}")
            group.attrs[f"S{atms_swath + 1}_IncidenceAngleIndex"] = np.array(b'IncidenceAngleIndex=1;\n')
            angles = data["incidence_angle_s1"]
            signs = np.pad(np.sign(np.diff(angles)), ((0, 0), (1, 0)), mode="edge")
            angles *= signs

            pixel_inds = np.digitize(angles, ANGLE_BINS) - 1
            scan_inds = np.arange(angles.shape[0])
            scan_inds = np.broadcast_to(scan_inds[..., None], pixel_inds.shape)
            invalid = (pixel_inds < 0) | (pixel_inds >= ANGLE_BINS.size - 1)
            pixel_inds = np.minimum(np.maximum(0, pixel_inds), 95)

            tc = -9999 * np.ones((n_scans, 96, ATMS_CHANNELS[atms_swath]), dtype=np.float32)
            for ch_ind, tropics_inds in enumerate(inds):
                if tropics_inds is None:
                    continue
                tropics_swath, tropics_channel = tropics_inds
                tc_tropics = data[f"tbs_s{tropics_swath + 1}"].data[..., tropics_channel].ravel()
                tc[scan_inds.ravel(), pixel_inds.ravel(), ch_ind] = tc_tropics
                tc[scan_inds[invalid], pixel_inds[invalid], ch_ind] = -9999
            group.create_dataset("Tc", data=tc)

            group.attrs[f"S{atms_swath + 1}_SwathHeader"] = np.array(SWATH_HEADER.format(n_scans=n_scans).encode())

            lats = -9999 * np.ones((n_scans, 96), np.float32)
            lats[scan_inds.ravel(), pixel_inds.ravel()] = data["latitude_s1"].data.ravel()
            group.create_dataset("Latitude", data=lats)

            lons = -9999 * np.ones((n_scans, 96), np.float32)
            lons[scan_inds.ravel(), pixel_inds.ravel()] = data["longitude_s1"].data.ravel()
            group.create_dataset("Longitude", data=lons)

            lons = -9999 * np.ones((n_scans, 96), np.float32)
            lons[scan_inds.ravel(), pixel_inds.ravel()] = data["sun_local_time_s1"].data.ravel()
            group.create_dataset("sunLocalTime", data=lons)

            sga = -9999 * np.ones((n_scans, 96), np.float32)
            sga[scan_inds.ravel(), pixel_inds.ravel()] = data["sun_glint_angle_s1"].data.ravel().astype(np.int8)
            group.create_dataset("sunGlintAngle", data=sga)

            ia = -9999 * np.ones((n_scans, 96, 1), np.float32)
            ia[scan_inds.ravel(), pixel_inds.ravel(), 0] = data["incidence_angle_s1"].data.ravel()
            group.create_dataset("incidenceAngle", data=ia)

            qual = 3 * np.ones((n_scans, 96), np.int8)
            qual[scan_inds.ravel(), pixel_inds.ravel()] = data["quality_s1"].data.ravel()
            group.create_dataset("Quality", data=qual)

            iai = np.ones((n_scans, ATMS_CHANNELS[atms_swath]), np.int8)
            group.create_dataset("incidenceAngleIndex", data=iai)

            scan_time = group.create_group("ScanTime")
            year = data["scan_time"].dt.year.data.astype("int16")
            scan_time.create_dataset("Year", data=year)
            month = data["scan_time"].dt.month.data.astype("int8")
            scan_time.create_dataset("Month", data=month)
            day = data["scan_time"].dt.day.data.astype("int8")
            scan_time.create_dataset("DayOfMonth", data=day)
            scan_time.create_dataset("DayOfYear", data=day)
            hour = data["scan_time"].dt.hour.data.astype("int8")
            scan_time.create_dataset("Hour", data=hour)
            minute = data["scan_time"].dt.minute.data.astype("int8")
            scan_time.create_dataset("Minute", data=minute)
            second = data["scan_time"].dt.second.data.astype("int8")
            scan_time.create_dataset("Second", data=second)
            scan_time.create_dataset("SecondOfDay", data=second)
            millis = (data["scan_time"].dt.nanosecond.data / 1e6).astype("int8")
            scan_time.create_dataset("MilliSecond", data=second)

            sc_status = group.create_group("SCstatus")
            sc_status.create_dataset("SClatitude", data=data["spacecraft_latitude"].data)
            sc_status.create_dataset("SClongitude", data=data["spacecraft_longitude"].data)
            sc_status.create_dataset("SCaltitude", data=data["spacecraft_altitude"].data)
            sc_status.create_dataset("SCorientation", data=data["spacecraft_orientation"].data)
            sc_status.create_dataset("FractionalGranuleNumber", data=np.arange(n_scans))
    return output_path


def run_gprof(l1c_file: Path,) -> xr.Dataset:
    """
    Run GPROF on TROPICS L1C file.

    Args:
        l1c_file: A pathlib.Path object pointing to the TROPICS L1C file to process.

    Return:
        An xarray.Dataset containing the retrieval results.
    """
    try:
        from gprof_nn.data.preprocessor import run_preprocessor
        from gprof_nn import sensors
        from gprof_nn.legacy import run_gprof_standard
    except ImportError:
        raise RuntimeError(
            "The 'gprof_nn' package must be installed to run GPROF retrievals."
        )

    try:
        from pansat.products.satellite.gpm import l1c_tropics03_tms
    except ImportError:
        raise RuntimeError(
            "The 'pansat' package must be install to run GPROF TROPICS retrievals."
        )

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_tr = l1c_tropics03_tms.open(l1c_file)
        if data_tr.scans.size < 1:
            LOGGER.warning(
                "No scans in L1C file '%s'. Skipping.",
                l1c_file
            )
            return None
        input_file = create_atms_file(tmp, "TROPICS03", "ATMS", data_tr)
        try:
            res = run_gprof_standard(sensors.ATMS, "ERA5", input_file, "STANDARD", False)
        except Exception:
            LOGGER.exception(
                "The following error was encountered when processing file '%s'.",
                input_file
            )
            return None

        sp = res["surface_precip"]
        sp.data[sp.data < 0] = np.nan
        res["surface_precip"] = res.surface_precip.interpolate_na("scans", limit=4)
        return res
