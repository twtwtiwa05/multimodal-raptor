#!/usr/bin/env python3
"""
Command line interface for mmraptor
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mmraptor'))

from mmraptor.pipeline.route import cli_main

if __name__ == "__main__":
    cli_main()