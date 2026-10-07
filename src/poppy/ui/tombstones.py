import sys

from poppy import tombstones as _tombstones

sys.modules[__name__] = _tombstones
