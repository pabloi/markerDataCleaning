# markerDataCleaning

TO DO:
Improve the 'fast' reconstruction for naiveDistances. Currently it works well if we have only one missing marker, but it fails if more than that.
Perhaps reconstruct one missing marker at a time, assuming the other 'bad' ones as fixed? (key feature here is that because weights are binary, we either trust markers or we don't, there is no gray area).

Improve normal reconstruction: similar to before, sometimes a single bad marker will make other tightly coupled markers have low logL, so they are 'loose' in the optimization.
This shouldn't matter much (even if 'loose' if the measured position is consistent with the model, it will be close to the optimal solution), but may be generating numerical issues (the position weight is small compared to the distance ones).
