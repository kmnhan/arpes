"""Some miscellaneous model definitions."""

from lmfit.models import update_param_vals
import lmfit as lf
import numpy as np
import xarray as xr

from .x_model_mixin import XModelMixin

__all__ = [
    "QuadraticModel",
    "PolynomialModel",
    "FermiVelocityRenormalizationModel",
    "LogRenormalizationModel",
]


class QuadraticModel(XModelMixin):
    """A model for fitting a quadratic function."""

    @staticmethod
    def quadratic(x, a=1, b=0, c=0):
        """Quadratic polynomial."""
        return a * x**2 + b * x + c

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Just defer to lmfit for initialization."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.quadratic, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for parameter guesses."""
        if x is None:
            pars = self.make_params()
            pars["%sa" % self.prefix].set(value=0)
            pars["%sb" % self.prefix].set(value=0)
            pars["%sc" % self.prefix].set(value=data.mean())
        else:
            a, b, c = np.polyfit(x, data, 2)
            pars = self.make_params(a=a, b=b, c=c)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class PolynomialModel(XModelMixin):
    """A polynomial model with up to 12 Parameters, specified by `degree`."""

    MAX_DEGREE = 12
    DEGREE_ERR = f"degree must be an integer equal to or smaller than {MAX_DEGREE}."

    valid_forms = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    @staticmethod
    def polynomial(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0, c8=0, c9=0, c10=0, c11=0, c12=0):
        if isinstance(x, np.ndarray):
            return np.polyval([c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0], x)
        else:
            coeffs = xr.DataArray(
                [c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0],
                coords={"degree": np.flip(np.arange(13))},
            )
            return xr.polyval(x, coeffs)

    def __init__(
        self, degree=9, independent_vars=("x",), prefix="", missing="raise", **kwargs
    ):
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        if "form" in kwargs:
            degree = int(kwargs.pop("form"))
        if not isinstance(degree, int) or degree > self.MAX_DEGREE:
            raise TypeError(self.DEGREE_ERR)

        self.poly_degree = degree
        pnames = [f"c{i}" for i in range(degree + 1)]
        kwargs["param_names"] = pnames
        kwargs.setdefault("name", f"Poly{self.poly_degree}")

        super().__init__(self.polynomial, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()
        if x is None:
            pars["c0"].set(value=data.mean())
            for i in range(1, self.poly_degree + 1):
                pars[f"{self.prefix}c{i}"].set(value=0.0)
        else:
            out = np.polyfit(x, data, self.poly_degree)
            for i, coef in enumerate(out[::-1]):
                pars[f"{self.prefix}c{i}"].set(value=coef)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class FermiVelocityRenormalizationModel(XModelMixin):
    """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials."""

    @staticmethod
    def fermi_velocity_renormalization_mfl(x, n0, v0, alpha, eps):
        """A model for Logarithmic Renormalization to Fermi Velocity in Dirac Materials.

        Args:
            x: value to evaluate fit at (carrier density)
            n0: Value of carrier density at cutoff energy for validity of Dirac fermions
            v0: Bare velocity
            alpha: Fine structure constant
            eps: Graphene Dielectric constant
        """
        #     y = v0 * (rs/np.pi)*(5/3 + np.log(rs))+(rs/4)*np.log(kc/np.abs(kF))
        fx = v0 * (1 + (alpha / (1 + eps)) * np.log(n0 / np.abs(x)))
        fx2 = v0 * (1 + (alpha / (1 + eps * np.abs(x))) * np.log(n0 / np.abs(x)))
        fx3 = v0 * (1 + (alpha / (1 + eps * x**2)) * np.log(n0 / np.abs(x)))
        # return v0 + v0*(alpha/(8*eps))*np.log(n0/x)
        return fx3

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """Sets physically reasonable constraints on parameter values."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.fermi_velocity_renormalization_mfl, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("n0", min=0.0)
        self.set_param_hint("eps", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for parameter estimation."""
        pars = self.make_params()

        # pars['%sn0' % self.prefix].set(value=10)
        # pars['%seps' % self.prefix].set(value=8)
        # pars['%svF' % self.prefix].set(value=(data.max()-data.min())/(kC-kD))

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC


class LogRenormalizationModel(XModelMixin):
    """A model for Logarithmic Renormalization to Linear Dispersion in Dirac Materials."""

    @staticmethod
    def log_renormalization(x, kF=1.6, kD=1.6, kC=1.7, alpha=0.4, vF=1e6):
        """Logarithmic correction to linear dispersion near charge neutrality in Dirac materials.

        As examples, this can be used to study the low energy physics in high quality ARPES spectra of graphene
        or topological Dirac semimetals.

        Args:
            x: The coorindates for the fit
            k: value to evaluate fit at
            kF: Fermi wavevector
            kD: Dirac point
            alpha: Fine structure constant
            vF: Bare Band Fermi Velocity
            kC: Cutoff Momentum
        """
        dk = x - kF
        dkD = x - kD
        return -vF * np.abs(dkD) + (alpha / 4) * vF * dk * np.log(np.abs(kC / dkD))

    def __init__(
        self, independent_vars=("x",), prefix="", missing="raise", name=None, **kwargs
    ):
        """The fine structure constant and velocity must be nonnegative, so we will constrain them here."""
        kwargs.update(
            {"prefix": prefix, "missing": missing, "independent_vars": independent_vars}
        )
        super().__init__(self.log_renormalization, **kwargs)

        self.set_param_hint("alpha", min=0.0)
        self.set_param_hint("vF", min=0.0)

    def guess(self, data, x=None, **kwargs):
        """Placeholder for actually making parameter estimates here."""
        pars = self.make_params()

        pars["%skC" % self.prefix].set(value=1.7)

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.doc = lf.models.COMMON_INIT_DOC
    guess.__doc__ = lf.models.COMMON_GUESS_DOC
