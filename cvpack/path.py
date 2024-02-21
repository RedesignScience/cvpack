"""
.. module:: path
   :platform: Linux, MacOS, Windows
   :synopsis: Specifications for Path Collective Variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import yaml


class Metric(yaml.YAMLObject):
    """
    A measure of progress or deviation with respect to a path in CV space
    """

    yaml_tag = "!cvpack.path.Metric"

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Metric) and self.name == other.name


yaml.SafeDumper.add_representer(Metric, Metric.to_yaml)
yaml.SafeLoader.add_constructor(Metric.yaml_tag, Metric.from_yaml)


progress: Metric = Metric("progress")
deviation: Metric = Metric("deviation")
