"""
Examples of using Flags:
1. Error with required flag not set.
  $ python py/tenpoint7/flags/flags_example.py
  >> FATAL Flags parsing error: flag --name=None: Flag --name must be specified.
2. Only required flags.
  $ python py/tenpoint7/flags/flags_example.py --name=Chocolate
  >> Hello  Chocolate
3. Required & optional flags.
  $ python py/tenpoint7/flags/flags_example.py --name=Chocolate --title=Mr.
  >> Hello Mr. Chocolate
4. Use flagfile
  $ python py/tenpoint7/flags/flags_example.py --name=Chocolate --title=Mr. \
  --flagfile=py/tenpoint7/flags/flagfile_example
  >> Hello Mr. Chocolate
     How are you today?
5. Parameters in both flagfile and command line. Whatever appears later overrides.
  $ python py/tenpoint7/flags/flags_example.py --name=Chocolate --title=Mr. \
  --flagfile=py/tenpoint7/flags/flagfile_example --greeting='Go away!'
  >> Hello Mr. Chocolate
     Go away!
"""
from atc import flags

name = flags.create("name", flags.FlagType.STRING, 'Name of the person',
                    required=True)
title = flags.create("title", flags.FlagType.STRING, 'Title if available',
                     default='')
greeting = flags.create("greeting", flags.FlagType.STRING,
                        'Greeting message if available')

class FlagTest:
  def __init__(self):
    pass

  def say_something(self):
    print('Hello %s %s' % (title.value(), name.value()))
    if greeting.value() is not None:
      print('%s' % greeting.value())

if __name__ == '__main__':
  flags.parse_flags()
  FlagTest().say_something()
