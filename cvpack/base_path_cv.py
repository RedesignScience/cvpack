"""
.. class:: BasePathCollectiveVariable
   :platform: Linux, MacOS, Windows
   :synopsis: Base class for path collective variables.

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t
from collections import OrderedDict

import openmm

from .cvpack import BaseCollectiveVariable
from .utils import evaluate_in_context

from cvpack import unit as mmunit


class BasePathCollectiveVariable(openmm.CustomCVForce, BaseCollectiveVariable):
    @mmunit.convert_quantities
    def __init__(
        self,
        variables: t.Iterable[BaseCollectiveVariable],
        milestones: t.Iterable[t.Union[openmm.Context, mmunit.VectorQuantity]],
        lambdaFactor: mmunit.ScalarQuantity,
        periods: t.Optional[t.Iterable[t.Optional[mmunit.ScalarQuantity]]] = None,
    ) -> None:
        variables = list(variables)
        periods = [None] * len(variables) if periods is None else list(periods)
        matrix = []
        for milestone in milestones:
            if isinstance(milestone, openmm.Context):
                matrix.append(evaluate_in_context(variables, milestone))
            else:
                matrix.append(tuple(milestone))
        n = len(matrix)
        if n < 2:
            raise ValueError("At least two milestones are required.")
        definitions = OrderedDict()
        definitions["lambda"] = lambdaFactor
        for i, row in enumerate(matrix):
            definitions[f"x{i}"] = "+".join(
                f"({value}-cv{j})^2" for j, value in enumerate(row)
            )
        definitions["xmin0"] = "min(x0,x1)"
        for i in range(n - 2):
            definitions[f"xmin{i+1}"] = f"min(xmin{i},x{i+2})"
        for i in range(n):
            definitions[f"w{i}"] = f"exp(lambda*(xmin{n - 2}-x{i}))"
        expressions = [self._getExpression(n)] + [
            f"{key}={value}" for key, value in reversed(definitions.items())
        ]
        super().__init__("; ".join(expressions))
        for i, variable in enumerate(variables):
            self.addCollectiveVariable(f"cv{i}", variable)
        self._registerCV(
            mmunit.dimensionless,
            variables,
            matrix,
            lambdaFactor,
        )


class PathProgress(BasePathCollectiveVariable):
    def _getExpression(self, n: int) -> str:
        scale = 1 / (n - 1)
        numerator = "+".join(f"{scale * i}*w{i}" for i in range(1, n))
        normalizing_constant = "+".join(f"w{i}" for i in range(n))
        return f"({numerator})/({normalizing_constant})"


class PathDeparture(BasePathCollectiveVariable):
    def _getExpression(self, n: int) -> str:
        normalizing_constant = "+".join(f"w{i}" for i in range(n))
        return f"xmin{n - 2} - log({normalizing_constant})/lambda"
