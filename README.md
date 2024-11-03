Things currently in progress:
<br> Pycache keeps being uploaded 
<br>-Change the step function to be more JAX-compatible 
<br>-requirements.txt and setup.py files and making the project into a package and removing sys.path.append("..") OR make unit_tests a module and then do from .Unit_Tests import Environment
<br>-Some environment functions need some changes to work for multi-agent (ex. get_obs)
<br>-Fix the warnings when running pytest and main_training

Things that will be changed in the future if needed
<br>-Write a jittable version of Delauney triangulation if environment creation becomes a bottleneck

Future:
<br>-Add an extra constructor to CTPGraphRealisation so that it's possible to sample blocking prob given a CTPGraph.

Things that needed to be done to extend to multi-agent:
<br>-A way to automatically choose the multiple origins and goals for the agents (random?)

New:
<br> One goal for now
<br> Using setters and getters - property in Python
<br> Whether to use self's properties in functions or pass them in as as arguments (try to follow functional programming paradigm?)

<br>-Didn't do the adding (adding maybe better for multi-agent)
<br>-Try jitting the environment

<br>-use printer to debug unit tests that fail (use pytest_print and input printer as argument and then do printer("thing to print"))


