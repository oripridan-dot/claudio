
import os
import sys
import runpy

# Ensure the 'claudio/src' directory is on the Python path
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Run claudio.demo_server as a module to establish package context
runpy.run_module('claudio.demo_server', run_name='__main__')
