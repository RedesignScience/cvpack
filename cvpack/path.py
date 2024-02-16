"""
.. module:: path
   :platform: Linux, MacOS, Windows
   :synopsis: Specifications for Path Collective Variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from typing import Literal

Measure = Literal["progress", "deviation"]

progress: Measure = "progress"
deviation: Measure = "deviation"
