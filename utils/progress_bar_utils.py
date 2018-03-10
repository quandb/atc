# -*- coding: utf-8 -*-
import sys


def print_progress_bar(iteration, total, prefix='Progress', suffix='Completed',
                       decimals=1, bar_length=20):
  """
  Call in a loop to create terminal progress bar
  Args:
    iteration (int): current iteration
    total (int): total iterations
    prefix (str): prefix string
    suffix (str): suffix string
    decimals (int): positive number of decimals in percent complete
    bar_length (int): character length of bar

  Returns:
    None

  """
  str_format = "{0:." + str(decimals) + "f}"
  percents = str_format.format(100 * (iteration / float(total)))
  filled_length = int(round(bar_length * iteration / float(total)))
  bar = '█' * filled_length + '-' * (bar_length - filled_length)

  sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

  if iteration == total:
    sys.stdout.write('\n')
  sys.stdout.flush()

