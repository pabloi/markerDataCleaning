function [lp] = determineLikelihoodFromZeroPlusModel(X1,D,sD,I)
%Computes a likelihood for a set of marker positions given a 'zero-model'
%(naive bayes) which assumes independent gaussian distributions for each
%distance between models. 
%------
%INPUTS:
%X is a Nx3xM matrix of N marker positions in 3D, samples M times.
%D is a NxN matrix containing the expected distances between the N markers
%sD is a NxN matrix containing the standard deviation of the distances
%across time
%------
%OUTPUT:
%p is a NxM matrix containing log-likelihoods for each of the N markers in each
%of the M samples

[X] = expandDataForZeroPlus(X1,I);
[N,tres,M]=size(X);
if size(D,1)~=N || ~issquare(D)
    error()
end
if size(sD,1)~=N || ~issquare(sD)
    error()
end
if tres~=3
    error()
end


[lp] = determineLikelihoodFromZeroModel(X,D,sD);
lp=lp(1:size(X1,1),:);

end

function bool=issquare(D)
bool = size(D,1)==size(D,2);
end

