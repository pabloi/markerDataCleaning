# markerDataCleaning

TO DO:

Reconstruct: it doesn't work very well. The same reconstruction on the same frames returns different results each time. Why? Outliers remain even after reconstruction.
Alos, sometimes a single bad marker will make other tightly coupled markers have low logL, so they are 'loose' in the optimization.
Generate good default weights for positions, based on marker scoring.

Reconstruct fast:
Perhaps reconstruct one missing marker at a time, assuming the other 'bad' ones as fixed? (key feature here is that because weights are binary, we either trust markers or we don't, there is no gray area).
Generate good default trusted/not-trusted classification.

Incorporate Kalman onto reconstruct:
The idea is that for each frame:
1) We do a Kalman-like integration of our knowledge of the previous frame with the measurements in this frame. We obtain a best position estimate and its uncertainty. This requires training a dynamical model.
2) We do reconstruct() but using the results from the kalman integration as priors, instead of the measured position with some given weights by itself
3) After reconstruct() we need to be able to return an uncertainty/variance matrix for each marker to be used in the next frame (currently we just return the estimated position).

