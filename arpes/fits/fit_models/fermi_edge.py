"""Definitions of models involving Fermi edges."""

from lmfit.models import update_param_vals
from scipy import stats
import numpy as np
import lmfit as lf
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from arpes.constants import K_BOLTZMANN_EV_KELVIN
from .x_model_mixin import XModelMixin
from .functional_forms import (
    fermi_dirac,
    fermi_dirac_affine,
    gstep,
    gstep_stdev,
    gstepb,
    lorentzian,
    band_edge_bkg,
    g,
    twolorentzian,
)
import numba

__all__ = [
    "AffineBroadenedFD",
    "ExtendedAffineBroadenedFD",
    "FermiLorentzianModel",
    "FermiDiracModel",
    "GStepBModel",
    "TwoBandEdgeBModel",
    "BandEdgeBModel",
    "BandEdgeBGModel",
    "FermiDiracAffGaussModel",
    "GStepBStdevModel",
    "GStepBStandardModel",
    "TwoLorEdgeModel",
]


class AffineBroadenedFD(XModelMixin):
    """A model for fitting an affine density of states with resolution broadened Fermi-Dirac occupation."""

    @staticmethod
    def affine_broadened_fd(
        x,
        fd_center=0,
        fd_width=0.003,
        conv_width=0.02,
        const_bkg=1,
        lin_bkg=0,
        offset=0,
    ):
        """Fermi function convoled with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            fd_center: center of the step
            fd_width: width of the step
            conv_width: The convolution width
            const_bkg: constant background
            lin_bkg: linear background slope
            offset: constant background
        """
        dx = x - fd_center
        x_scaling = x[1] - x[0]
        fermi = 1 / (np.exp(dx / fd_width) + 1)
        return (
            gaussian_filter(
                (const_bkg + lin_bkg * dx) * fermi, sigma=conv_width / x_scaling
            )
            + offset
        )

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.affine_broadened_fd, **kwargs)

        self.set_param_hint("offset", min=0.0)
        self.set_param_hint("fd_width", min=0.0)
        self.set_param_hint("conv_width", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Make some heuristic guesses.

        We use the mean value to estimate the background parameters and physically
        reasonable ones to initialize the edge.
        """
        pars = self.make_params()

        pars["%sfd_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.mean().item() * 2)
        pars["%soffset" % self.prefix].set(value=data.min().item())

        pars["%sfd_width" % self.prefix].set(0.005)  # TODO we can do better than this
        pars["%sconv_width" % self.prefix].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


@numba.njit("f8[:,:](f8[:], i8)", cache=True)
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0], deg + 1), dtype=np.float64)
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_


@numba.njit("f8[:](f8[:,:], f8[:])", cache=True)
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_


@numba.njit("f8[:](f8[:], f8[:], i8)", cache=True)
def fit_poly_jit(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p


# adapted and improved from KWAN Igor procedures
@numba.njit(cache=True)
def AffBroadFD_F_G(x, center, temp, resolution, back0, back1, dos0, dos1):
    delta_x = x[1] - x[0]
    n_pad = int(resolution * 5.0 / delta_x)  # padding
    x_pad = n_pad * delta_x

    sigma = resolution / np.sqrt(8 * np.log(2))  # resolution given in FWHM
    x = np.linspace(x[0] - x_pad, x[-1] + x_pad, int(2 * n_pad + len(x)))

    affine_fd = (back0 + back1 * x) + (dos0 - back0 + (dos1 - back1) * x) / (
        1 + np.exp((x - center) / temp / 8.617333262145177e-5)
    )
    g_x = np.linspace(-x_pad, x_pad, 2 * n_pad + 1)
    # gauss = np.exp(-(g_x**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    gauss = (
        delta_x
        * np.exp(-(g_x**2) / (2 * sigma**2))
        / np.sqrt(2 * np.pi * sigma**2)
    )

    return affine_fd, gauss


class ExtendedAffineBroadenedFD(XModelMixin):
    """A model for fitting an affine density of states with resolution broadened Fermi-Dirac occupation."""

    @staticmethod
    def LinearBroadFermiDirac(
        x,
        center=0,
        temp=30,
        resolution=0.02,
        back0=1,
        back1=0,
        dos0=1,
        dos1=0,
    ):
        return np.convolve(
            *AffBroadFD_F_G(
                np.asarray(x, dtype=np.float64),
                center,
                temp,
                resolution,
                back0,
                back1,
                dos0,
                dos1,
            ),
            mode="valid",
        )

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.LinearBroadFermiDirac, **kwargs)
        self.set_param_hint("temp", min=0.0)
        self.set_param_hint("resolution", min=0.0)

    def guess(self, data, x, **kwargs):
        """Make some heuristic guesses.

        We use the mean value to estimate the background parameters and physically
        reasonable ones to initialize the edge.
        """
        pars = self.make_params()

        len_fit = max(round(len(x) * 0.05), 10)
        # len_fit = 10

        dos0, dos1 = fit_poly_jit(
            np.asarray(x[:len_fit], dtype=np.float64),
            np.asarray(data[:len_fit], dtype=np.float64),
            deg=1,
        )
        back0, back1 = fit_poly_jit(
            np.asarray(x[-len_fit:], dtype=np.float64),
            np.asarray(data[-len_fit:], dtype=np.float64),
            deg=1,
        )
        efermi = x[np.argmin(np.gradient(gaussian_filter1d(data, 0.2 * len(x))))]

        pars[f"{self.prefix}center"].set(value=efermi)
        pars[f"{self.prefix}back0"].set(value=back0)
        pars[f"{self.prefix}back1"].set(value=back1)
        pars[f"{self.prefix}dos0"].set(value=dos0)
        pars[f"{self.prefix}dos1"].set(value=dos1)
        pars[f"{self.prefix}temp"].set(30)
        pars[f"{self.prefix}resolution"].set(0.02)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiLorentzianModel(XModelMixin):
    """A Lorentzian multiplied by a gstepb background."""

    @staticmethod
    def gstepb_mult_lorentzian(
        x, center=0, width=1, erf_amp=1, lin_bkg=0, const_bkg=0, gamma=1, lorcenter=0
    ):
        """A Lorentzian multiplied by a gstepb background."""
        return gstepb(x, center, width, erf_amp, lin_bkg, const_bkg) * lorentzian(
            x, gamma, lorcenter, 1
        )

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.gstepb_mult_lorentzian, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)
        self.set_param_hint("gamma", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slorcenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiDiracModel(XModelMixin):
    """A model for the Fermi Dirac function."""

    def __init__(
        self, independent_vars=("x",), prefix="", missing="drop", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(fermi_dirac, **kwargs)

        self.set_param_hint("width", min=0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["{}center".format(self.prefix)].set(value=0)
        pars["{}width".format(self.prefix)].set(value=0.05)
        pars["{}scale".format(self.prefix)].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(gstepb, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("width", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        efermi = x[np.argmin(np.gradient(gaussian_filter1d(data, 0.1 * len(x))))]

        pars["%scenter" % self.prefix].set(value=efermi)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoBandEdgeBModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution.

    TODO, actually implement two_band_edge_bkg (find original author and their intent).
    """

    @staticmethod
    def two_band_edge_bkg():
        """Some missing model referenced in old Igor code retained for visibility here."""
        raise NotImplementedError

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {
                "prefix": prefix,
                "missing": missing,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(self.two_band_edge_bkg, **kwargs)

        self.set_param_hint("amplitude_1", min=0.0)
        self.set_param_hint("gamma_1", min=0.0)
        self.set_param_hint("amplitude_2", min=0.0)
        self.set_param_hint("gamma_2", min=0.0)

        self.set_param_hint("offset", min=-10)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        pars["%scenter" % self.prefix].set(value=0)
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution."""

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {
                "prefix": prefix,
                "missing": missing,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(band_edge_bkg, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("offset", min=-10)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        pars["%scenter" % self.prefix].set(value=0)
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class BandEdgeBGModel(XModelMixin):
    """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution."""

    @staticmethod
    def band_edge_bkg_gauss(
        x,
        center=0,
        width=0.05,
        amplitude=1,
        gamma=0.1,
        lor_center=0,
        offset=0,
        lin_bkg=0,
        const_bkg=0,
    ):
        """A model for fitting a Lorentzian and background multiplied into the fermi dirac distribution."""
        return np.convolve(
            band_edge_bkg(
                x, 0, width, amplitude, gamma, lor_center, offset, lin_bkg, const_bkg
            ),
            g(np.linspace(-6, 6, 800), 0, 0.01),
            mode="same",
        )

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {
                "prefix": prefix,
                "missing": missing,
                "independent_vars": independent_vars,
            }
        )
        super().__init__(self.band_edge_bkg_gauss, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("gamma", min=0.0)
        self.set_param_hint("offset", min=-10)
        self.set_param_hint("center", vary=False)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here.

        We should really do some peak fitting or edge detection to find
        okay values here.
        """
        pars = self.make_params()

        if x is not None:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
            pars["%slor_center" % self.prefix].set(value=x[np.argmax(data - slope * x)])
        else:
            pars["%slor_center" % self.prefix].set(value=-0.2)

        pars["%sgamma" % self.prefix].set(value=0.2)
        pars["%samplitude" % self.prefix].set(value=(data.mean() - data.min()) / 1.5)

        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%soffset" % self.prefix].set(value=data.min())

        # pars['%scenter' % self.prefix].set(value=0)
        pars["%swidth" % self.prefix].set(0.02)  # TODO we can do better than this

        return update_param_vals(pars, self.prefix, **kwargs)


class FermiDiracAffGaussModel(XModelMixin):
    """Fermi Dirac function with an affine background multiplied, then all convolved with a Gaussian."""

    @staticmethod
    def fermi_dirac_bkg_gauss(
        x, center=0, width=0.05, lin_bkg=0, const_bkg=0, scale=1, sigma=0.01
    ):
        """Fermi Dirac function with an affine background multiplied, then all convolved with a Gaussian."""
        return np.convolve(
            fermi_dirac_affine(x, center, width, lin_bkg, const_bkg, scale),
            g(x, (min(x) + max(x)) / 2, sigma),
            mode="same",
        )

    def __init__(
        self, independent_vars=("x",), prefix="", missing="drop", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.fermi_dirac_bkg_gauss, **kwargs)

        # self.set_param_hint('width', min=0)
        self.set_param_hint("width", vary=False)
        # self.set_param_hint('lin_bkg', max=10)
        # self.set_param_hint('scale', max=50000)
        self.set_param_hint("scale", min=0)
        self.set_param_hint("sigma", min=0, vary=True)
        self.set_param_hint("lin_bkg", vary=False)
        self.set_param_hint("const_bkg", vary=False)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["{}center".format(self.prefix)].set(value=0)
        # pars['{}width'.format(self.prefix)].set(value=0.05)
        pars["{}width".format(self.prefix)].set(value=0.0009264)
        pars["{}scale".format(self.prefix)].set(value=data.mean() - data.min())
        pars["{}lin_bkg".format(self.prefix)].set(value=0)
        pars["{}const_bkg".format(self.prefix)].set(value=0)
        pars["{}sigma".format(self.prefix)].set(value=0.023)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStdevModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_stdev(x, center=0, sigma=1, erf_amp=1, lin_bkg=0, const_bkg=0):
        """Fermi function convolved with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            center: center of the step
            sigma: width of the step
            erf_amp: height of the step
            lin_bkg: linear background slope
            const_bkg: constant background
        """
        dx = x - center
        return (
            const_bkg + lin_bkg * np.min(dx, 0) + gstep_stdev(x, center, sigma, erf_amp)
        )

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.gstepb_stdev, **kwargs)

        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%ssigma" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class GStepBStandardModel(XModelMixin):
    """A model for fitting Fermi functions with a linear background."""

    @staticmethod
    def gstepb_standard(x, center=0, sigma=1, amplitude=1, **kwargs):
        """Specializes paramters in gstepb."""
        return gstepb(x, center, width=sigma, erf_amp=amplitude, **kwargs)

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.gstepb_standard, **kwargs)

        self.set_param_hint("amplitude", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%ssigma" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%samplitude" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class TwoLorEdgeModel(XModelMixin):
    """A model for (two lorentzians with an affine background) multiplied by a gstepb."""

    def twolorentzian_gstep(
        x,
        gamma,
        t_gamma,
        center,
        t_center,
        amp,
        t_amp,
        lin_bkg,
        const_bkg,
        g_center,
        sigma,
        erf_amp,
    ):
        """Two Lorentzians, an affine background, and a gstepb edge."""
        TL = twolorentzian(
            x, gamma, t_gamma, center, t_center, amp, t_amp, lin_bkg, const_bkg
        )
        GS = gstep(x, g_center, sigma, erf_amp)
        return TL * GS

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.twolorentzian_gstep, **kwargs)

        self.set_param_hint("amp", min=0.0)
        self.set_param_hint("gamma", min=0)
        self.set_param_hint("t_amp", min=0.0)
        self.set_param_hint("t_gamma", min=0)
        self.set_param_hint("erf_amp", min=0.0)
        self.set_param_hint("sigma", min=0)
        self.set_param_hint("lin_bkg", min=-10, max=10)
        self.set_param_hint("const_bkg", min=-50, max=50)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for making better heuristic guesses here."""
        pars = self.make_params()

        pars["%scenter" % self.prefix].set(value=0)
        pars["%st_center" % self.prefix].set(value=0)
        pars["%sg_center" % self.prefix].set(value=0)
        pars["%slin_bkg" % self.prefix].set(value=0)
        pars["%sconst_bkg" % self.prefix].set(value=data.min())
        pars["%sgamma" % self.prefix].set(0.02)  # TODO we can do better than this
        pars["%st_gamma" % self.prefix].set(0.02)
        pars["%ssigma" % self.prefix].set(0.02)
        pars["%samp" % self.prefix].set(value=data.mean() - data.min())
        pars["%st_amp" % self.prefix].set(value=data.mean() - data.min())
        pars["%serf_amp" % self.prefix].set(value=data.mean() - data.min())

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
