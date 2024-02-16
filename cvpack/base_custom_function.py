"""
.. class:: BaseCustomFunction
   :platform: Linux, MacOS, Windows
   :synopsis: Abstract class for collective variables defined by a custom function

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import itertools as it
import typing as t

import numpy as np

from cvpack import unit as mmunit

from .cvpack import BaseCollectiveVariable


class BaseCustomFunction(BaseCollectiveVariable):
    """
    Abstract class for collective variables defined by a custom function.
    """

    def _extractParameters(
        self,
        size: int,
        **parameters: t.Union[float, t.Iterable[float]],
    ) -> t.Tuple[t.Dict[str, float], t.Dict[str, t.List[float]]]:
        global_parameters = {}
        perbond_parameters = {}
        for name, data in parameters.items():
            if isinstance(data, t.Iterable):
                perbond_parameters[name] = list(it.islice(data, size))
            else:
                global_parameters[name] = data
        return global_parameters, perbond_parameters

    def _addParameters(  # pylint: disable=too-many-arguments
        self,
        overalls: t.Dict[str, float],
        perbonds: t.Dict[str, t.List[float]],
        groups: t.List[t.Tuple[int, ...]],
        pbc: bool = False,
        unit: t.Optional[mmunit.Unit] = None,
    ) -> None:
        # pylint: disable=no-member
        definitions = "; ".join(f"{name}={value}" for name, value in overalls.items())
        self.setEnergyFunction(f"{self.getEnergyFunction()}; {definitions}")
        for name in perbonds:
            self.addPerBondParameter(name)
        for group, *values in zip(groups, *perbonds.values()):
            self.addBond(group, values)
        self.setUsesPeriodicBoundaryConditions(pbc)
        if not np.isclose(
            mmunit.Quantity(1.0, unit).value_in_unit_system(mmunit.md_unit_system),
            1.0,
        ):
            raise ValueError(f"Unit {unit} is not compatible with the MD unit system.")
