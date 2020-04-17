from mpasview.utils import *
from mpasview.data import *
import warnings

# Force warnings.warn() to omit the source code line in the message
# https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    formatwarning_orig(message, category, filename, lineno, line='')

