"""The default channels' file name as common/shm_protocol.h spells it: kShmChannelPrefix,
SHM_PROTOCOL_VERSION and kShmChannelSuffix, for the tests to compare what the programs, the
launcher and the units name with."""
from pathlib import Path
import re

HEADER = Path(__file__).resolve().parent.parent / 'common/shm_protocol.h'


def channel_name():
    """The default channels' file name, such as shm-v31.bin."""
    text = HEADER.read_text()

    def define(name, value):
        return re.search(rf'^#define {name} {value}$', text, re.MULTILINE)[1]
    return define('kShmChannelPrefix', '"([^"]+)"') + define('SHM_PROTOCOL_VERSION', r'(\d+)') + \
        define('kShmChannelSuffix', '"([^"]+)"')
