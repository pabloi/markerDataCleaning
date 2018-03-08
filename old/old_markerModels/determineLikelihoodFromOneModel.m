function [lp,p] = determineLikelihoodFromOneModel(X,D,sD)
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


[p] = computeOneModelCollectionProb(X,D,sD);
Q=size(p,2);
lp1=zeros(N,M,Q);
relModels=bsxfun(@gt,p,max(p,[],2)/3); %Soft membership with threshold
%relModels=p==max(p,[],2); %Hard memebership

for i=1:Q
    idx=relModels(:,i); %Frames for which the i-th model is relevant
    if sum(idx)>0
    [lp1(:,idx,i)] = determineLikelihoodFromZeroModel(X(:,:,idx),D(:,:,i),sD(:,:,i));
    end
end
modP=p;
modP(~relModels)=0;
modP=bsxfun(@rdivide,modP,sum(modP,2));
lp=log(nansum(bsxfun(@times,exp(lp1),reshape(modP,1,M,Q)),3));


end

function bool=issquare(D)
bool = size(D,1)==size(D,2);
end

