"""Provides broadcasted and process parallel curve fitting for PyARPES.

The core of this module is `broadcast_model` which is a serious workhorse in PyARPES for
analyses based on curve fitting. This allows simple multidimensional curve fitting by
iterative fitting across one or many axes. Currently basic strategies are implemented,
but in the future we would like to provide:

1. Passing xr.DataArray values to parameter guesses and bounds, which can be interpolated/selected
   to allow changing conditions throughout the curve fitting session.
2. A strategy allowing retries with initial guess taken from the previous fit. This is similar
   to some adaptive curve fitting routines that have been proposed in the literature.
"""

import contextlib
import os
import sys
from typing import Any, Callable, Dict, List, Tuple, Union

import arpes.fits.fit_models
import joblib
import lmfit
import numpy as np
import xarray as xr

# from arpes.provenance import update_provenance
# from arpes.trace import traceable
# from arpes.typing import DataType
# from arpes.utilities import normalize_to_spectrum
from joblib import Parallel, delayed

# from packaging import version
import tqdm

from . import mp_fits

__all__ = ("broadcast_model", "result_to_hints")


TypeIterable = Union[List[type], Tuple[type]]

# XARRAY_REQUIRES_VALUES_WRAPPING = version.parse(xr.__version__) > version.parse(
#     "0.10.0"
# )


# def wrap_for_xarray_values_unpacking(item):
#     """This is a shim for https://github.com/pydata/xarray/issues/2097."""
#     if XARRAY_REQUIRES_VALUES_WRAPPING:
#         return np.array(item, dtype=object)

#     return item


def result_to_hints(m: lmfit.model.ModelResult, defaults=None) -> Dict[str, Dict[str, Any]]:
    """Turns an `lmfit.model.ModelResult` into a dictionary with initial guesses.

    Args:
        m: The model result to extract parameters from
        defaults: Returned if `m` is None, useful for cell re-evaluation in Jupyter

    Returns:
        A dict containing parameter specifications in key-value rathan than `lmfit.Parameter`
        format, as you might pass as `params=` to PyARPES fitting code.
    """
    if m is None:
        return defaults
    return {k: {"value": m.params[k].value} for k in m.params}


def parse_model(model):
    """Takes a model string and turns it into a tokenized version.

    1. ModelClass -> ModelClass
    2. [ModelClass] -> [ModelClass]
    3. str -> [<ModelClass, operator as string>]

    i.e.

    A + (B + C) * D -> [A, '(', B, '+', C, ')', '*', D]

    Args:
        model: The model specification

    Returns:
        A tokenized specification of the model suitable for passing to the curve
        fitting routine.
    """
    if not isinstance(model, str):
        return model

    pad_all = ["+", "-", "*", "/", "(", ")"]

    for pad in pad_all:
        model = model.replace(pad, " {} ".format(pad))

    special = set(pad_all)

    def read_token(token):
        if token in special:
            return token
        try:
            token = float(token)
            return token
        except ValueError:
            try:
                return arpes.fits.fit_models.__dict__[token]
            except KeyError:
                raise ValueError("Could not find model: {}".format(token))

    return [read_token(token) for token in model.split()]


def is_notebook():
    # http://stackoverflow.com/questions/34091701/determine-if-were-in-an-ipython-notebook-session
    if "IPython" not in sys.modules:  # IPython hasn't been imported
        return False
    from IPython import get_ipython

    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), "kernel", None) is not None


@contextlib.contextmanager
def joblib_progress(file=None, notebook=None, dynamic_ncols=True, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar given as
    argument"""

    if file is None:
        file = sys.stdout

    if notebook is None:
        notebook = is_notebook()

    if notebook:
        tqdm_object = tqdm.tqdm_notebook(
            iterable=None, dynamic_ncols=dynamic_ncols, file=file, **kwargs
        )
    else:
        tqdm_object = tqdm.tqdm(iterable=None, dynamic_ncols=dynamic_ncols, file=file, **kwargs)

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


# @update_provenance("Broadcast a curve fit along several dimensions")
# @traceable
def broadcast_model(
    model_cls: Union[type, TypeIterable],
    data,
    broadcast_dims,
    params=None,
    model_params=None,
    progress=True,
    weights=None,
    safe=False,
    prefixes=None,
    window=None,
    parallelize=None,
    trace: Callable = None,
    parallel_kw=dict(),
    **kwargs,
):
    """Perform a fit across a number of dimensions.

    Allows composite models as well as models defined and compiled through strings.

    Args:
        model_cls: The model specification
        data: The data to curve fit
        broadcast_dims: Which dimensions of the input should be iterated across as opposed
          to fit across
        params: Parameter hints, consisting of plain values or arrays for interpolation
        progress: Whether to show a progress bar
        weights: Weights to apply when curve fitting. Should have the same shape as the input data
        safe: Whether to mask out nan values
        window: A specification of cuts/windows to apply to each curve fit
        parallelize: Whether to parallelize curve fits, defaults to True if unspecified and more
          than 20 fits were requested
        trace: Controls whether execution tracing/timestamping is used for performance investigation

    Returns:
        An `xr.Dataset` containing the curve fitting results. These are data vars:

        - "results": Containing an `xr.DataArray` of the `lmfit.model.ModelResult` instances
        - "residual": The residual array, with the same shape as the input
        - "data": The original data used for fitting
        - "norm_residual": The residual array normalized by the data, i.e. the fractional error
    """
    ori_data = data

    if params is None:
        params = {}

    if isinstance(broadcast_dims, str):
        broadcast_dims = [broadcast_dims]

    # data = normalize_to_spectrum(data)
    if isinstance(data, xr.Dataset):
        if "up" in data.data_vars:
            data = data.up
        data = data.S.spectrum

    cs = {}
    for dim in broadcast_dims:
        cs[dim] = data.coords[dim]

    other_axes = set(data.dims).difference(set(broadcast_dims))
    template = data.sum(list(other_axes))
    template.values = np.ndarray(template.shape, dtype=object)
    n_fits = np.prod(np.array(list(template.S.dshape.values())))

    if parallelize is None:
        parallelize = n_fits > 20

    # trace("Copying residual")
    residual = data.copy(deep=True)
    residual.values = np.zeros(residual.shape)

    # trace("Parsing model")
    model = parse_model(model_cls)

    fitter = mp_fits.MPWorker(
        data=data,
        uncompiled_model=model,
        prefixes=prefixes,
        params=params,
        model_params=model_params,
        safe=safe,
        weights=weights,
        window=window,
        **kwargs,
    )

    if parallelize:
        parallel_kw.setdefault("n_jobs", -1)
    else:
        parallel_kw.setdefault("n_jobs", 1)
        # trace(f"Running fits (nfits={n_fits}) in parallel (n_threads={os.cpu_count()})")

        # print("Running on multiprocessing pool... this may take a while the first time.")
        # from .hot_pool import hot_pool
        # pool = hot_pool.pool
        # exe_results = list(
        #     tqdm.tqdm(
        #         pool.imap(fitter, template.G.iter_coords()), total=n_fits, desc="Fitting on pool..."
        #     )
        # )

    if progress:
        with joblib_progress(desc="Fitting", total=n_fits) as _:
            exe_results = Parallel(**parallel_kw)(
                delayed(fitter)(c) for c in template.G.iter_coords()
            )
    else:
        exe_results = Parallel(**parallel_kw)(delayed(fitter)(c) for c in template.G.iter_coords())

    for fit_result, fit_residual, coords in exe_results:
        template.loc[coords] = np.array(fit_result, dtype=object)
        residual.loc[coords] = fit_residual

    return xr.Dataset(
        {
            "results": template,
            "data": ori_data,
            "residual": residual,
            "norm_residual": residual / data,
        },
        residual.coords,
    )
