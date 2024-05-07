REPRODUCIBLITY_STATE:int = 0
# == 0: can reproduce paper's results but slow
# >= 1: replace pooling by CUDA code
# >= 2: replace self-attention by CUDA code (pairwise sum for softmax)
# >= 3: replace self-attention by CUDA code (+ divided sum for running sum) 
# >= 4: replace periodic encoding with better numerical accuracy code
