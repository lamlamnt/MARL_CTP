Things currently in progress:
<br>-Separate into CTPGraph and Realisation
<br>-Use pytest for unit testing
<br>-Make the step function and run_episode function more JAX-compatible (just use jax.lax.cond)
<br>-requirements.txt and setup.py files and making the project into a package and removing sys.path.append("..")
<br>-Do blocked_status.Blocked and blocked_status.Unblocked instead of 0 and 1 to prevent confusion
<br>-Some environment functions need some changes to work for multi-agent (ex. get_obs)

Things that will be changed in the future if needed
<br>-Write a jittable version of Delauney triangulation if environment creation becomes a bottleneck

Things under consideration:
<br>-Whether to use a custom Graph class instead of using jraph

Things that needed to be done to extend to multi-agent:
<br>-A way to automatically choose the multiple origins and goals for the agents (random?)


