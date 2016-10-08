#!/usr/bin/python
from pydft import msg
import numpy as np

def examples():

    """Prints examples of using the script to the console using colored output.
    """
    script = "DFT: Performs a Density Functional Theory ."
    explain = ("This code produces a numerical solution for the energy of a system of "
               "atoms.")
    contents = [(("Examples."), 
                 "",
                 ".")]
    required = ("REQUIRED: .")
    output = ("RETURNS: .")
    details = (".")
    outputfmt = ("")

    msg.example(script, explain, contents, required, output, outputfmt, details)

script_options = {
    "N": dict(default=100, type=int,
              help=(".")),
    "-plot": dict(help=("")),    
     }
"""dict: default command-line arguments and their
    :meth:`argparse.ArgumentParser.add_argument` keyword arguments.
"""

def _parser_options():
    """Parses the options and arguments from the command line."""
    #We have two options: get some of the details from the config file,
    import argparse
    from pydft import base
    pdescr = "Numerical DFT code."
    parser = argparse.ArgumentParser(parents=[base.bparser], description=pdescr)
    for arg, options in script_options.items():
        parser.add_argument(arg, **options)
        
    args = base.exhandler(examples, parser)
    if args is None:
        return

    return args # pragma: no cover

def run(args):
    return 0

if __name__ == '__main__': # pragma: no cover
    run(_parser_options())

