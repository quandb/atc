# See: http://docs.python.org/library/pkgutil.html#pkgutil.extend_path
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)


import flags
import flag_parsers

FlagType = flag_parsers.FlagType
create = flags.Flags.create
reset_flags = flags.Flags.reset_all
parse_flag_for_test = flags.Flags.parse_flag_for_test
parse_flags = flags.Flags.parse_flags
print_flag_values = flags.Flags.print_flag_values
