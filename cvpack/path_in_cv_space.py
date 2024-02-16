"""
.. class:: BasePathCollectiveVariable
   :platform: Linux, MacOS, Windows
   :synopsis: Base class for path collective variables.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from collections import OrderedDict

import openmm

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable
from .path import departure, progress, Measure
from .utils import convert_to_matrix


class PathInCVSpace(openmm.CustomCVForce, BaseCollectiveVariable):
    r"""
    A measure of the system's progress (:math:`s`) or deviation (:math:`z`) with
    respect to a path defined by a sequence of milestones in a collective variable
    space :cite:`Branduardi_2007`:

    .. math::

        s({\bf r}) = \frac{
            \dfrac{\sum_{i=1}^n i w_i({\bf r})}{\sum_{i=1}^n w_i({\bf r})} - 1
        }{n-1}
        \quad \text{or} \quad
        z({\bf r}) = - \frac{1}{\lambda} \ln \sum_{i=1}^n w_i({\bf r})

    where :math:`w_i({\bf r}) = e^{-\lambda \|{\bf c}({\bf r}) - \hat{\bf c}_i\|^2}` is
    a Gaussian kernel, :math:`{\bf c}({\bf r})` is a vector of collective variables,
    :math:`\hat{\bf c}_i` is a milestone located in the collective-variable space, and
    :math:`\lambda` is a parameter that controls the width of the kernels. The path is
    defined by the sequence of :math:`n` milestones.

    The squared norm of a vector :math:`{\bf x}` in the collective variable space is
    defined as

    .. math::

        \|{\bf x}\|^2 = {\bf x}^T {\bf D}^{-2} {\bf x}

    where :math:`{\bf D}` is a diagonal matrix with a characteristic scale for each
    collective variable as its diagonal elements. Appropriate boundary conditions are
    used for periodic collective variables.

    Parameters
    ----------
    measure
        The path-related measure to compute. Use `cvpack.path.progress` or
        `cvpack.path.departure`
    variables
        The collective variables that define the space
    milestones
        The milestones in the collective variable space. The number of rows must be
        equal to the number of milestones and the number of columns must be equal to
        the number of collective variables
    lambdaFactor
        The width of the Gaussian kernels
    scales
        The characteristic scales for the collective variables. If not provided, the
        scales are assumed to be 1 (in standard MD units) for each collective variable
    """

    yaml_tag = "!PathInCVSpace"

    @mmunit.convert_quantities
    def __init__(
        self,
        measure: Measure,
        variables: t.Iterable[BaseCollectiveVariable],
        milestones: mmunit.MatrixQuantity,
        lambdaFactor: mmunit.ScalarQuantity,
        scales: t.Optional[t.Iterable[mmunit.ScalarQuantity]] = None,
    ) -> None:
        if measure not in (progress, departure):
            raise ValueError(
                "Invalid measure. Use 'cvpack.path.progress' or 'cvpack.path.departure'."
            )
        variables = list(variables)
        scales = [1.0] * len(variables) if scales is None else list(scales)
        milestones, n, numvars = convert_to_matrix(milestones)
        if numvars != len(variables):
            raise ValueError("Wrong number of columns in the milestones matrix.")
        if n < 2:
            raise ValueError("At least two rows are required in the milestones matrix.")
        definitions = OrderedDict({"lambda": lambdaFactor})
        periods = {
            j: var.getPeriod().value_in_md_units()
            for j, var in enumerate(variables)
            if var.getPeriod() is not None
        }
        for i, values in enumerate(milestones):
            deltas = [f"{value}-cv{j}" for j, value in enumerate(values)]
            for j, period in periods.items():
                deltas[j] = f"min(abs({deltas[j]}),{period}-abs({deltas[j]})"
            definitions[f"x{i}"] = "+".join(
                f"({delta}/{scale})^2" for delta, scale in zip(deltas, scales)
            )
        definitions["xmin0"] = "min(x0,x1)"
        for i in range(n - 2):
            definitions[f"xmin{i+1}"] = f"min(xmin{i},x{i+2})"
        for i in range(n):
            definitions[f"w{i}"] = f"exp(lambda*(xmin{n - 2}-x{i}))"
        definitions["wsum"] = "+".join(f"w{i}" for i in range(n))
        expressions = [f"{key}={value}" for key, value in definitions.items()]
        if measure is progress:
            numerator = "+".join(f"{i}*w{i}" for i in range(1, n))
            expressions.append(f"{numerator}/({n - 1}*wsum)")
        else:
            expressions.append(f"xmin{n - 2} - log(wsum)/lambda")
        super().__init__("; ".join(reversed(expressions)))
        for i, variable in enumerate(variables):
            self.addCollectiveVariable(f"cv{i}", variable)
        self._registerCV(
            mmunit.dimensionless,
            measure,
            variables,
            milestones.tolist(),
            lambdaFactor,
            scales,
        )
