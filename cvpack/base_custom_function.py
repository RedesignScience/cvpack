"""
.. class:: BaseCustomFunction
   :platform: Linux, MacOS, Windows
   :synopsis: Abstract class for collective variables defined by a custom function

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from openmm import unit as mmunit

from .cvpack import BaseCollectiveVariable
from .units import ScalarQuantity, VectorQuantity, value_in_md_units


class BaseCustomFunction(BaseCollectiveVariable):
    """
    Abstract class for collective variables defined by a custom function.
    """

    def _extractParameters(
        self,
        size: int,
        **parameters: t.Union[ScalarQuantity, VectorQuantity],
    ) -> t.Tuple[t.Dict[str, float], t.Dict[str, t.List[float]]]:
        global_parameters = {}
        perbond_parameters = {}
        for name, data in parameters.items():
            data = value_in_md_units(data)
            if isinstance(data, t.Iterable):
                perbond_parameters[name] = [data[i] for i in range(size)]
            else:
                global_parameters[name] = data
        return global_parameters, perbond_parameters

    def _addParameters(
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
        if (1 * unit).value_in_unit_system(mmunit.md_unit_system) != 1:
            raise ValueError(f"Unit {unit} is not compatible with the MD unit system.")
