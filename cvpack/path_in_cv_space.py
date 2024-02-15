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
from .path import Departure, Progress
from .utils import convert_to_matrix


class PathInCVSpace(openmm.CustomCVForce, BaseCollectiveVariable):
    @mmunit.convert_quantities
    def __init__(
        self,
        mode: t.Union[Progress, Departure],
        variables: t.Iterable[BaseCollectiveVariable],
        milestones: mmunit.MatrixQuantity,
        lambdaFactor: mmunit.ScalarQuantity,
    ) -> None:
        if mode not in (Progress, Departure):
            raise ValueError(
                "Invalid mode. Use 'cvpack.path.Progress' or 'cvpack.path.Departure'."
            )
        variables = list(variables)
        milestones, n, numvars = convert_to_matrix(milestones)
        if numvars != len(variables):
            raise ValueError("Wrong number of columns in the milestones matrix.")
        if n < 2:
            raise ValueError("At least two rows are required in the milestones matrix.")
        definitions = OrderedDict({"lambda": lambdaFactor})
        for i, row in enumerate(milestones):
            definitions[f"x{i}"] = "+".join(
                f"({value}-cv{j})^2" for j, value in enumerate(row)
            )
        definitions["xmin0"] = "min(x0,x1)"
        for i in range(n - 2):
            definitions[f"xmin{i+1}"] = f"min(xmin{i},x{i+2})"
        for i in range(n):
            definitions[f"w{i}"] = f"exp(lambda*(xmin{n - 2}-x{i}))"
        definitions["wsum"] = "+".join(f"w{i}" for i in range(n))
        expressions = [f"{key}={value}" for key, value in definitions.items()]
        if mode is Progress:
            numerator = "+".join(f"{i}*w{i}" for i in range(1, n))
            expressions.append(f"{numerator}/({n - 1}*wsum)")
        else:
            expressions.append(f"xmin{n - 2} - log(wsum)/lambda")
        super().__init__("; ".join(reversed(expressions)))
        for i, variable in enumerate(variables):
            self.addCollectiveVariable(f"cv{i}", variable)
        self._registerCV(
            mmunit.dimensionless,
            variables,
            milestones.tolist(),
            lambdaFactor,
        )
