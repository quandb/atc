class Error(Exception):
  """The base class for all flags errors."""

class CantOpenFlagFileError(Error):
  """Raised if flagfile fails to open: doesn't exist, wrong permissions, etc."""

class DuplicateParsingError(Error):
  """Raised if the parsing function is called again without a reset."""

class CantParseArgumentError(Error):
  """Raised if the argument is failed to parse."""

class DuplicateFlagError(Error):
  """Raised if there is a flag naming conflict."""

class IllegalFlagValueError(Error):
  """Raised if the flag command line argument is illegal."""

class ValidationError(Error):
  """Raised if flag validator constraint is not satisfied."""
