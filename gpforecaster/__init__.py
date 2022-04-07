__version__ = "0.1.8"

from gpforecaster import model
from gpforecaster import results

# Only print in interactive mode
import __main__ as main
if not hasattr(main, '__file__'):
    print("""Importing the gpforecaster module. L. Roque. 
    Algorithm to forecast Hierarchical Time Series providing point forecast and uncertainty intervals.\n""")
