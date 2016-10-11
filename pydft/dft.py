#!/usr/bin/python
from pydft import msg
import numpy as np

def examples():

    """Prints examples of using the script to the console using colored output.
    """
    script = "DFT: Performs a Density Functional Theory ."
    explain = ("This code produces a numerical solution for the energy of a system of "
               "atoms.")
    contents = [(("Find the charge density for a sc cell with lattice parameter of 6 "
                  "with 4 divisions along each lattice vector."), 
                 "dft.py 6 -s [4,4,4]",
                 "This prints the value of phi evaluated at all the sample points "
                 "to the screen")]
    # required = ("REQUIRED: .")
    # output = ("RETURNS: .")
    # details = (".")
    outputfmt = ("")

    msg.example(script, explain, contents, required, output, outputfmt, details)

script_options = {
    "a": dict(default=1., type=float,
              help=("The lattice parameter for the crystal structure.")),
    "-crystal": dict(defalt="sc", type=str,
                     help=("The type of primitive cell to use options are (sc)")),
    "-s": dict(default=[3,3,3], type=list,
              help=("The number of sampling points along each basis vector.")),
    "-poisson": dict(default=None,
                     help=("Prints the solution to the poisson equation to the screen.")),
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

    if args["crystal"] != "sc":
        raise ValueError("Only sc cubic lattice are supported at this time.")

    return args # pragma: no cover

def run(args):

    if args["poisson"]:
        from pydft.poisson import poisson, charge_dist
        R = args["a"]*np.array([[1,0,0],[0,1,0],[0,0,1]])
        n = charge_dist(args["s"],R,[-1,1],[0.5,0.75])
        phi = poisson(s,R,n)
        print(phi)
    else:
        return 0
    
if __name__ == '__main__': # pragma: no cover
    run(_parser_options())

