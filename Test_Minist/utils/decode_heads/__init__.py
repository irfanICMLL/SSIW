from .aspp_head import ASPPHead

from .fcn_head import FCNHead
from .fpn_head import FPNHead

from .psp_head import PSPHead

from .uper_head import UPerHead
from .segformer_head import SegFormerHead

__all__ = [
    'FCNHead', 'ASPPHead', 'FPNHead', 'PSPHead', 'SegFormerHead'
]
