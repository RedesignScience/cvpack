"""Useful Collective Variables for OpenMM"""

import yaml

# Add imports here
from ._version import __version__  # noqa: F401
from .angle import Angle  # noqa: F401
from .atomic_function import AtomicFunction  # noqa: F401
from .attraction_strength import AttractionStrength  # noqa: F401
from .centroid_function import CentroidFunction  # noqa: F401
from .composite_rmsd import CompositeRMSD  # noqa: F401
from .distance import Distance  # noqa: F401
from .generic import Generic  # noqa: F401
from .helix_angle_content import HelixAngleContent  # noqa: F401
from .helix_hbond_content import HelixHBondContent  # noqa: F401
from .helix_rmsd_content import HelixRMSDContent  # noqa: F401
from .helix_torsion_content import HelixTorsionContent  # noqa: F401
from .number_of_contacts import NumberOfContacts  # noqa: F401
from .path_in_cv_space import PathInCVSpace  # noqa: F401
from .radius_of_gyration import RadiusOfGyration  # noqa: F401
from .radius_of_gyration_sq import RadiusOfGyrationSq  # noqa: F401
from .residue_coordination import ResidueCoordination  # noqa: F401
from .rmsd import RMSD  # noqa: F401
from .sheet_rmsd_content import SheetRMSDContent  # noqa: F401
from .torsion import Torsion  # noqa: F401
from .torsion_similarity import TorsionSimilarity  # noqa: F401

for _cv in [
    Angle,
    AtomicFunction,
    AttractionStrength,
    CentroidFunction,
    CompositeRMSD,
    Distance,
    Generic,
    HelixAngleContent,
    HelixHBondContent,
    HelixRMSDContent,
    HelixTorsionContent,
    NumberOfContacts,
    PathInCVSpace,
    RadiusOfGyration,
    RadiusOfGyrationSq,
    ResidueCoordination,
    RMSD,
    SheetRMSDContent,
    Torsion,
    TorsionSimilarity,
]:
    yaml.SafeDumper.add_representer(_cv, _cv.to_yaml)
    yaml.SafeLoader.add_constructor(_cv.yaml_tag, _cv.from_yaml)
del _cv
