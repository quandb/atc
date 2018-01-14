from enum import Enum

from exceptions import CantOpenFlagFileError, \
  CantParseArgumentError, IllegalFlagValueError

class FlagType(Enum):
  STRING = 1
  INT = 2
  FLOAT = 3
  BOOLEAN = 4
  MULTI_STRING = 5
  MULTI_INT = 6

def parse_string(flag_value):
    return flag_value

def parse_float(flag_value):
  try:
    return float(flag_value)
  except ValueError:
    raise IllegalFlagValueError(
      'Error flag parsing. Requires "float" value. Got "%s" instead.' %
      flag_value)

def parse_int(flag_value):
  try:
    return int(flag_value)
  except ValueError:
    raise IllegalFlagValueError(
      'Error flag parsing. Requires "int" value. Got "%s" instead.' %
      flag_value)

def parse_boolean(flag_value):
  lower_value = flag_value.lower()
  if not (lower_value == 'true' or lower_value == 'false'):
    raise IllegalFlagValueError(
      'Boolean flag value must be either "true" or "false"')
  return flag_value.lower() == 'true'

def parse_multi_string(flag_value):
  return flag_value.split(',')

def parse_multi_int(flag_value):
  try:
    return [int(str_number) for str_number in flag_value.split(',')]
  except ValueError:
    raise IllegalFlagValueError(
      "Error flag parsing. This flag requires a string of 'int' value"
    )

PARSER_MAP = {
  FlagType.STRING: parse_string,
  FlagType.FLOAT: parse_float,
  FlagType.INT: parse_int,
  FlagType.BOOLEAN: parse_boolean,
  FlagType.MULTI_STRING: parse_multi_string,
  FlagType.MULTI_INT: parse_multi_int,
}

FLAG_FILE_FLAG = 'flagfile'

def get_parser(parser_type):
  return PARSER_MAP[parser_type]

def parse_long_arg(arg):
  if arg.startswith('#') or len(arg.strip()) == 0:
    return None, None
  if not arg.startswith('--') or '=' not in arg:
    raise CantParseArgumentError(
      'Cannot parse argument "%s", must be in form --flag_name==flag_value' %
      arg)
  name, value = arg.lstrip('-').split('=', 1)
  return name, value

def parse_arguments(raw_arguments):
  arguments = dict([parse_long_arg(arg)
                    for arg in raw_arguments if arg is not None])
  if FLAG_FILE_FLAG in arguments:
    flagfile_args = parse_flag_file(
      arguments[FLAG_FILE_FLAG])
    # Arguments in the command line have higher priority
    flagfile_args.update(arguments)
    arguments = flagfile_args
  return arguments

def parse_flag_file(flagfile):
  try:
    args = [arg.strip() for arg in open(flagfile, 'r')]
  except IOError:
    raise CantOpenFlagFileError(
      'Cannot read flagfile %s' % flagfile)
  return dict([parse_long_arg(arg)
               for arg in args if arg is not None])
