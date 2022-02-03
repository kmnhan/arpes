"""Data loading for BL10, analyzer rotation"""
import numpy as np
import xarray as xr
from astropy.io import fits
import warnings
from pathlib import Path
import os.path
import copy
import arpes.config
from arpes.endstations.fits_utils import find_clean_coords
from arpes.provenance import provenance_from_file
from arpes.endstations import (
    EndstationBase,
    HemisphericalEndstation,
    SynchrotronEndstation,
)

__all__ = ('HERSEndstationERLab',)

class MyFITSEndstation(EndstationBase):
    """Loads data from the .fits format produced by the MAESTRO software and derivatives.
    This ends up being somewhat complicated, because the FITS export is written in LabView and
    does not conform to the standard specification for the FITS archive format.
    Many of the intricacies here are in fact those shared between MAESTRO's format
    and the Lanzara Lab's format. Conrad does not foresee this as an issue, because it is
    unlikely that many other ARPES labs will adopt this data format moving forward, in
    light of better options derivative of HDF like the NeXuS format.
    """

    PREPPED_COLUMN_NAMES = {
        "time": "time",
        "Delay": "delay-var",  # these are named thus to avoid conflicts with the
        "Sample-X": "cycle-var",  # underlying coordinates
        "Mira": "pump_power",
        # insert more as needed
    }

    SKIP_COLUMN_NAMES = {
        "Phi",
        "null",
        "X",
        "Y",
        "Z",
        "mono_eV",
        "Slit Defl",
        "Optics Stage",
        "Scan X",
        "Scan Y",
        "Scan Z",
        # insert more as needed
    }
    
    SKIP_COLUMN_FORMULAS = {
        lambda name: True if ("beamview" in name or "IMAQdx" in name) else False,
    }

    RENAME_KEYS = {
        # "Phi": "chi",
        # "Alpha": "beta",
        # "Beta": "beta",
        # "Azimuth": "chi",
        # "Pump_energy_uJcm2": "pump_fluence",
        # "T0_ps": "t0_nominal",
        # "W_func": "workfunction",
        # "Slit": "slit",
        # "LMOTOR0": "x",
        # "LMOTOR1": "y",
        # "LMOTOR2": "z",
        # "LMOTOR3": "theta",
        # "LMOTOR4": "beta",
        # "LMOTOR5": "chi",
        # "LMOTOR6": "alpha",
    }
    
    def resolve_frame_locations(self, scan_desc: dict = None):
        """These are stored as single files, so just use the one from the description."""
        if scan_desc is None:
            raise ValueError(
                "Must pass dictionary as file scan_desc to all endstation loading code."
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        p = Path(original_data_loc)
        if not p.exists():
            original_data_loc = os.path.join(arpes.config.DATA_PATH, original_data_loc)

        return [original_data_loc]

    def load_single_frame(self, frame_path: str = None, scan_desc: dict = None, **kwargs):
        """Loads a scan from a single .fits file.
        This assumes the DAQ storage convention set by E. Rotenberg (possibly earlier authors)
        for the storage of ARPES data in FITS tables.
        This involves several complications:
        1. Hydrating/extracting coordinates from start/delta/n formats
        2. Extracting multiple scan regions
        3. Gracefully handling missing values
        4. Unwinding different scan conventions to common formats
        5. Handling early scan termination
        """
        # Use dimension labels instead of
        self.trace("Opening FITS HDU list.")
        hdulist = fits.open(frame_path, ignore_missing_end=True)
        primary_dataset_name = None

        # Clean the header because sometimes out LabView produces improper FITS files
        for i in range(len(hdulist)):
            # This looks a little stupid, but because of confusing astropy internals actually works
            hdulist[i].header["UN_0_0"] = ""  # TODO This card is broken, this is not a good fix
            del hdulist[i].header["UN_0_0"]
            hdulist[i].header["UN_0_0"] = ""
            if "TTYPE2" in hdulist[i].header and hdulist[i].header["TTYPE2"] == "Delay":
                self.trace("Using ps delay units. This looks like an ALG main chamber scan.")
                hdulist[i].header["TUNIT2"] = ""
                del hdulist[i].header["TUNIT2"]
                hdulist[i].header["TUNIT2"] = "ps"

            self.trace(f"HDU {i}: Attempting to fix FITS errors.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdulist[i].verify("fix+warn")
                hdulist[i].header.update()
            # This actually requires substantially more work because it is lossy to information
            # on the unit that was encoded

        hdu = hdulist[1]

        scan_desc = copy.deepcopy(scan_desc)
        attrs = scan_desc.pop("note", scan_desc)
        attrs.update(dict(hdulist[0].header))

        drop_attrs = ["COMMENT", "HISTORY", "EXTEND", "SIMPLE", "SCANPAR", "SFKE_0"]
        for dropped_attr in drop_attrs:
            if dropped_attr in attrs:
                del attrs[dropped_attr]

        from arpes.utilities import rename_keys

        built_coords, dimensions, real_spectrum_shape = find_clean_coords(
            hdu, attrs, mode="MC", trace=self.trace
        )
        self.trace("Recovered coordinates from FITS file.")

        attrs = rename_keys(attrs, self.RENAME_KEYS)
        scan_desc = rename_keys(scan_desc, self.RENAME_KEYS)

        def clean_key_name(k: str) -> str:
            if "#" in k:
                k = k.replace("#", "num")

            return k

        attrs = {clean_key_name(k): v for k, v in attrs.items()}
        scan_desc = {clean_key_name(k): v for k, v in scan_desc.items()}

        # don't have phi because we need to convert pixels first
        deg_to_rad_coords = {"theta", "chi"}

        # convert angular attributes to radians
        for coord_name in deg_to_rad_coords:
            if coord_name in attrs:
                try:
                    attrs[coord_name] = float(attrs[coord_name]) * (np.pi / 180)
                except (TypeError, ValueError):
                    pass
            if coord_name in scan_desc:
                try:
                    scan_desc[coord_name] = float(scan_desc[coord_name]) * (np.pi / 180)
                except (TypeError, ValueError):
                    pass

        data_vars = {}

        all_names = hdu.columns.names
        n_spectra = len([n for n in all_names if "Fixed_Spectra" in n or "Swept_Spectra" in n])
        for column_name in hdu.columns.names:
            # we skip some fixed set of the columns, such as the one dimensional axes, as well as things that are too
            # tricky to load at the moment, like the microscope images from MAESTRO
            should_skip = False
            if column_name in self.SKIP_COLUMN_NAMES:
                should_skip = True

            for formula in self.SKIP_COLUMN_FORMULAS:
                if formula(column_name):
                    should_skip = True

            if should_skip:
                continue

            # the hemisphere axis is handled below
            dimension_for_column = dimensions[column_name]
            column_shape = real_spectrum_shape[column_name]

            column_display = self.PREPPED_COLUMN_NAMES.get(column_name, column_name)
            if "Fixed_Spectra" in column_display:
                if n_spectra == 1:
                    column_display = "spectrum"
                else:
                    column_display = "spectrum" + "-" + column_display.split("Fixed_Spectra")[1]

            if "Swept_Spectra" in column_display:
                if n_spectra == 1:
                    column_display = "spectrum"
                else:
                    column_display = "spectrum" + "-" + column_display.split("Swept_Spectra")[1]

            # sometimes if a scan is terminated early it can happen that the sizes do not match the expected value
            # as an example, if a beta map is supposed to have 401 slices, it might end up having only 260 if it were
            # terminated early
            # If we are confident in our parsing code above, we can handle this case and take a subset of the coords
            # so that the data matches
            try:
                resized_data = hdu.data.columns[column_name].array.reshape(column_shape)
            except ValueError:
                # if we could not resize appropriately, we will try to reify the shapes together
                rest_column_shape = column_shape[1:]
                n_per_slice = int(np.prod(rest_column_shape))
                total_shape = hdu.data.columns[column_name].array.shape
                total_n = np.prod(total_shape)

                n_slices = total_n // n_per_slice
                # if this isn't true, we can't recover
                data_for_resize = hdu.data.columns[column_name].array
                if total_n // n_per_slice != total_n / n_per_slice:
                    # the last slice was in the middle of writing when something hit the fan
                    # we need to infer how much of the data to read, and then repeat the above
                    # we need to cut the data

                    # This can happen when the labview crashes during data collection,
                    # we use column_shape[1] because of the row order that is used in the FITS file
                    data_for_resize = data_for_resize[
                        0 : (total_n // n_per_slice) * column_shape[1]
                    ]
                    warnings.warn(
                        "Column {} was in the middle of slice when DAQ stopped. Throwing out incomplete slice...".format(
                            column_name
                        )
                    )

                column_shape = list(column_shape)
                column_shape[0] = n_slices

                try:
                    resized_data = data_for_resize.reshape(column_shape)
                except Exception:
                    # sometimes for whatever reason FITS errors and cannot read the data
                    continue

                # we also need to adjust the coordinates
                altered_dimension = dimension_for_column[0]
                built_coords[altered_dimension] = built_coords[altered_dimension][:n_slices]

            data_vars[column_display] = xr.DataArray(
                resized_data,
                coords={k: c for k, c in built_coords.items() if k in dimension_for_column},
                dims=dimension_for_column,
                attrs=attrs,
            )

        def prep_spectrum(data: xr.DataArray):
            # don't do center pixel inference because the main chamber
            # at least consistently records the offset from the edge
            # of the recorded window
            if "pixel" in data.coords:
                phi_axis = (
                    data.coords["pixel"].values * (30/650) * (np.pi/180) #(30/650)
                )

                if "pixel" in data.coords:
                    data = data.rename(pixel="phi")

                data = data.assign_coords(phi=phi_axis)


            # Always attach provenance
            provenance_from_file(
                data, frame_path, {"what": "Loaded MC dataset from FITS.", "by": "load_MC"}
            )

            return data

        if "spectrum" in data_vars:
            data_vars["spectrum"] = prep_spectrum(data_vars["spectrum"])

        # adjust angular coordinates
        built_coords = {
            k: c * (np.pi / 180) if k in deg_to_rad_coords else c for k, c in built_coords.items()
        }

        self.trace("Stitching together xr.Dataset.")
        return xr.Dataset(
            {
                "safe-{}".format(name) if name in data_var.coords else name: data_var
                for name, data_var in data_vars.items()
            },
            attrs={**scan_desc, "name": primary_dataset_name},
        )

class HERSEndstationERLab(SynchrotronEndstation, HemisphericalEndstation, MyFITSEndstation):

    PRINCIPAL_NAME = "ALS-BL1001-ERLab"
    ALIASES = ["ALS-BL1001-ERLab", "HERS-ERLab", "ALS-HERS-ERLab", "BL1001-ERLab"]
    ANALYZER_INFORMATION = {
        "analyzer": "R4000",
        "analyzer_name": "Scienta R4000",
        # "parallel_deflectors": False,
        # "perpendicular_deflectors": True,
        # "analyzer_radius": None,
        "analyzer_type": "hemispherical",
    }
    RENAME_KEYS = {
        # "LMOTOR0": "x",
        # "LMOTOR1": "y",
        # "LMOTOR2": "z",
        # "Scan X": "scan_x",
        # "Scan Y": "scan_y",
        # "Scan Z": "scan_z",
        # "LMOTOR3": "theta",
        # "LMOTOR4": "beta",
        # "LMOTOR5": "chi",
        # "LMOTOR6": "alpha",
        # "LMOTOR9": "psi",
        # "MONOEV": "hv",
        "mono_eV": "hv",
        "SF_HV": "hv",
        "SS_HV": "hv",
        # "Slit Defl": "psi",
        # "S_Volts": "volts",
        # probably need something like an attribute list for extraction
        # "SFRGN0": "fixed_region_name",
        "SFE_0": "daq_center_energy",
        "SSLNM0": "lens_mode_name",
        "SSPE_0": "pass_energy",
        "UNDHARM": "undulator_harmonic",
        "UNDGAP": "undulator_gap",
        "RINGCURR": "beam_current",
        "SSFR_0": "frames_per_slice",
        # "SFBA_0": "phi_prebinning",
        # "SFBE0": "eV_prebinning",
        "LWLVNM": "daq_type",
        "Alpha":"beta",
    }

    RENAME_COORDS = {
        # "X": "x",
        # "Y": "y",
        # "Z": "z",
        # "Alpha":"beta",
    }


    ENSURE_COORDS_EXIST = ["alpha", "beta", "theta", "chi", "phi", "psi", "hv"]

    def load(self, scan_desc: dict = None, **kwargs):
        # in the future, can use a regex in order to handle the case where we postfix coordinates
        # for multiple spectra

        scan = super().load(scan_desc, **kwargs)

        coord_names = scan.coords.keys()
        will_rename = {}
        for coord_name in coord_names:
            if coord_name in self.RENAME_KEYS:
                will_rename[coord_name] = self.RENAME_KEYS.get(coord_name)

        for k, v in will_rename.items():
            if v in scan.coords:
                del scan.coords[v]

        renamed = scan.rename(will_rename)

        if "scan_x" in renamed.coords:
            for d in renamed.data_vars:
                if "spectrum" in d:
                    renamed[d].values = np.flip(
                        renamed[d].values, axis=renamed[d].dims.index("scan_x")
                    )


        # renamed["alpha"] = 0.0
        # if "pixel" in renamed.coords:
        #     phi_axis = (
        #         renamed.coords["pixel"].values * (3/65) * np.pi / 180
        #     )
        #     if "pixel" in renamed.coords:
        #         renamed = renamed.rename(pixel="phi")

        #     renamed = renamed.assign_coords(phi=phi_axis)

        return renamed

    def fix_prebinned_coordinates(self):
        pass

    def postprocess_final(self, data: xr.Dataset, scan_desc: dict = None):
        ls = [data] + data.S.spectra
        for l in ls:
            l.attrs.update(self.ANALYZER_INFORMATION)

            # if "GRATING" in l.attrs:
            #     l.attrs["grating_lines_per_mm"] = {
            #         "G201b": 600,
            #     }.get(l.attrs["GRATING"])

        # if "chi" not in data.coords:
        #     data.coords["chi"] = 0
        #     for s in data.S.spectra:
        #         s.coords["chi"] = 0

        # if "theta" not in data.coords:
        #     data.coords["theta"] = 0
        #     for s in data.S.spectra:
        #         s.coords["theta"] = 0

        if "Alpha" in data.coords:
            psi_axis = (
                data.coords["Alpha"].values * (np.pi / 180)
            )

            if "Alpha" in data.coords:
                data = data.rename(Alpha="psi")

            data = data.assign_coords(psi=psi_axis)

        coordlist = ["alpha", "beta", "chi", "theta", "psi"]

        for coord in coordlist:
            if coord not in data.coords:
                data.coords[coord] = 0
                for s in data.S.spectra:
                    s.coords[coord] = 0
    

        # data.coords['alpha'] = np.pi/2
        # for s in data.S.spectra:
        #     s.coords['alpha'] = np.pi/2

        data = super().postprocess_final(data, scan_desc)

        return data

    ATTR_TRANSFORMS = {
        "START_T": lambda l: {
            "time": " ".join(l.split(" ")[1:]).lower(),
            "date": l.split(" ")[0],
        },
        "SF_SLITN": lambda l: {
            "slit_number": int(l.split(" ")[0]),
            "slit_shape": l.split(" ")[-1].lower(),
            "slit_width": float(l.split(" ")[2]),
        },
    }

    MERGE_ATTRS = {
        # "mcp_voltage": None,
        # "repetition_rate": 5e8,
        # "undulator_type": "elliptically_polarized_undulator",
        # "undulator_gap": None,
        # "undulator_z": None,
        # "undulator_polarization": None,
    }