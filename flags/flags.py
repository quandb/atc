import os.path
import sys
from collections import defaultdict

from io import StringIO
import six
from six import iteritems

import exceptions
import flag_parsers
from exceptions import IllegalFlagValueError

class Flags(object):
  def __init__(self):
    pass

  flag_registry = dict()
  raw_flags = dict()
  required_flags = dict()
  is_parsed = False

  @staticmethod
  def create(name, parser_type, description, default=None,
             required=False):
    """
    Create a Flag with its configurations.
    Args:
      name (str): The name of the flag. This must be unique.
      parser_type (flag_parsers.FlagType): The type of this flag
      description (str): The description of the flag.
      default: If not required, then it should have a default value.
      required (bool): Whether this flag is required or not.

    Returns:
      Flag: The object that contains flag value.

    """
    if name in Flags.flag_registry:
      raise exceptions.DuplicateFlagError(
        'Duplicated flag names. Flag "%s" already exists.' % name)

    parser = flag_parsers.get_parser(parser_type)
    flag = Flag(name, parser, default, description, required,
                module_name=Flags.get_calling_module_name())
    if name in Flags.raw_flags:
      flag_str_value = Flags.raw_flags.get(name)
      flag.parse_and_set(flag_str_value)
    else:
      if Flags.is_parsed and required:
        Flags.flag_registry[name] = flag
        Flags.print_help()
        raise exceptions.ValidationError(
          'Required flag "%s" has not been set.' % name)
      flag._value = default

    if required:
      Flags.required_flags[name] = flag
    Flags.flag_registry[name] = flag
    return flag

  @staticmethod
  def reset_all(reset_registry=True):
    if reset_registry:
      Flags.flag_registry = dict()
    Flags.raw_flags = dict()
    Flags.required_flags = dict()
    Flags.is_parsed = False

  @staticmethod
  def parse_flag_for_test(flag_name, flag_value):
    """
    Manual set a flag value for test.
    Args:
      flag_name (str): Flag name
      flag_value (str): Flag value
    """
    Flags.raw_flags[flag_name] = flag_value
    if flag_name in Flags.flag_registry:
      flag = Flags.flag_registry[flag_name]
      flag.parse_and_set(flag_value)

  @staticmethod
  def parse_flags(flag_values=None):
    """
    Parse flags from system args, a string or dict of (key, value).
    Args:
      flag_values: None if parsing from command line args; otherwise it could be
        string (full args) or a dict of (flag_name, flag_value)

    """
    if Flags.is_parsed:
      raise exceptions.DuplicateParsingError('Flags were already parsed')
    if flag_values is None:
      import sys
      flag_values = flag_parsers.parse_arguments(sys.argv[1:])
    elif isinstance(flag_values, six.string_types):
      flag_values = flag_parsers.parse_arguments(flag_values.split())
    for flag_name in flag_values:
      raw_value = flag_values[flag_name]
      if flag_name in Flags.flag_registry:
        flag = Flags.flag_registry[flag_name]
        flag.parse_and_set(raw_value)
        Flags.flag_registry[flag_name] = flag
      else:
        Flags.raw_flags[flag_name] = flag_values[flag_name]
    Flags.is_parsed = True
    Flags.validate_required()

  @staticmethod
  def validate_required():
    """
    Validate if all required flags exist.

    """
    for flag_name, flag in iteritems(Flags.required_flags):
      if flag.value() is None:
        Flags.print_help()
        raise exceptions.ValidationError(
          'Required flag "%s" has not been set.' % flag_name)

  @staticmethod
  def print_help():
    print('-------- Flags ----------')
    for name, flag in iteritems(Flags.flag_registry):
      print('--%s (%s%s): %s' % (
        name,
        'REQUIRED - ' if flag.required else '',
        flag.module_name,
        flag.description
      ))
    print('----------------------')

  disclaim_module_ids = set([id(sys.modules[__name__])])

  @staticmethod
  def get_calling_module_name():
    """Returns the module that's calling into this module.

    We generally use this function to get the name of the module calling a
    DEFINE_foo... function.

    Returns:
      The module object that called into this one.

    Raises:
      AssertionError: if no calling module could be identified.
    """
    for depth in range(1, sys.getrecursionlimit()):
      # sys._getframe is the right thing to use here, as it's the best
      # way to walk up the call stack.
      globals_for_frame = sys._getframe(depth).f_globals
      name = globals_for_frame.get('__name__', None)
      module = sys.modules.get(name, None)
      if id(module) not in Flags.disclaim_module_ids and name is not None:
        return sys.argv[0] if name == '__main__' else name

  @staticmethod
  def print_flag_values(output_flagfile=None):
    """
    Print flag values in form of flagfile.

    This is to output all configurations used in the program to later retrieve.
    All inputs should be the same when the program is run again with this
    output flagfile.

    Args:
      output_flagfile (str): location to write flag values to.
        If the location is a directory, flagfile will be written to
        $output_flagfile/flagfile

    """
    is_s3 = False
    if output_flagfile:
      is_s3 = output_flagfile.startswith('s3')
      if is_s3:
        handle = StringIO()
      elif os.path.isdir(output_flagfile):
        handle = open('%s/flagfile' % output_flagfile, 'w')
      else:
        handle = open(output_flagfile, 'w')
    else:
      handle = sys.stdout
    flag_by_module = defaultdict(list)
    # print flags registered by flags.create.
    for name, flag in iteritems(Flags.flag_registry):
      value_str = ''
      if isinstance(flag.value(), list):
        value_str = ','.join([str(x) for x in flag.value()])
      elif flag.value():
        value_str = str(flag.value())
      if value_str:
        flag_by_module[flag.module_name].append(
          '--%s=%s' % (name, value_str))
    for module_name, flag_values in iteritems(flag_by_module):
      handle.write('# %s\n' % module_name)
      for flag_value in flag_values:
        handle.write('%s\n' % flag_value)

    # other flags are passed in but not actually used.
    handle.write('# Others\n')
    for name, flag_value in iteritems(Flags.raw_flags):
      if (name not in Flags.flag_registry
          and name is not None
          and name != flag_parsers.FLAG_FILE_FLAG):
        handle.write('--%s=%s\n' % (name, flag_value))

    if is_s3:
      # S3Load.upload_text_data(
      #     handle.getvalue(), '%s/flagfile' % output_flagfile)
      pass

    if handle is not sys.stdout:
      handle.close()

class Flag(object):
  """
  Object for holding flag information.
  """
  def __init__(self, name, parser, default_value, description,
               required=False, module_name=None):
    self.name = name
    self.description = description
    self.default_value = default_value
    self.parser = parser
    self._value = None
    self.required = required
    self.module_name = module_name

  def __str__(self):
    return 'Flag(name=%s, default=%s, value=%s)' % (
      self.name, str(self.default_value), str(self._value))

  def parse_and_set(self, raw_value):
    if raw_value is not None:
      try:
        self._value = self.parser(raw_value)
      except Exception as e:
        print(e)
        raise IllegalFlagValueError(
          'Error parsing flag value "--%s=%s" (module: %s)' %
          (self.name, raw_value, self.module_name))

  def value(self):
    return self._value
