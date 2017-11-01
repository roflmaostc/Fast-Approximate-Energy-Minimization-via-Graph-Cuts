# Fast-Approximate-Energy-Minimization-via-Graph-Cuts

This code implements the swap-algorithm according to the paper [Fast Approximate Energy Minimization via Graph Cuts](https://dl.acm.org/citation.cfm?id=505473) from Boykov, Veksler and Zabih.

For the Graph-Cut we use [PyMaxflow](https://github.com/pmneila/PyMaxflow).

Install requirements:

    sudo pip install -r requirements.txt

Executing is done by:

    python minimization.py path/filename.png cycles

Cycles can be an arbitrary number. Something below 10 ist quite good.
**Hint: Don't use images larger than 100x100 pixel, it will take a lot of time otherwise.**
