# cmsac_football_bart
Material for BART Workshop at 2024 CMSAC Football Workshop

## To Do

  1. Step function approximations: use a smooth function and show the staircase approximation; then motivate sum-of-trees
  2. Then show the unit square representations. Instead of writing the rules, potentially color the interior nodes based on the cutpoints
  3. Describe what BART is: it's a Bayesian approach to learning a tree ensemble. Operationally it is different than boosting and it's different than RF. This is important to emphasis
  4. Prior: it is meant to regularize (not necessarily reflect our true beliefs, which seem hard to elicit). Key is to make sure the trees don't get too large **unless the data says they should**
  5. Computation: 3 slides on MCMC;
  6. Summarization: given these draws, we often want.
  7. Why use it? It enables much more coherent UQ: do we care about f(x) or do we care about a new y (i.e. f + noise). Can embed BART into much larger workflows (in plate discipline paper, it's about uncertainty propagation). And it's a probabilistic model. 
  8. Why not use it? It's a bit harder and doesn't (yet) solve the same problems (namely, multi-class classificaiton). 
