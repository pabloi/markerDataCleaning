
%% Generate some random frame:
X=500*randn(12,3);
[M,dim]=size(X);

%% Compute distances:
D=sqrt(reshape(sum((X-reshape(X',1,dim,M)).^2,2),M,M));
%D=D+randn(size(D)); %Adding noise for incompatibility of exact distances, as would happen in a real model

%% Remove one marker:
oldX=X;
X(7,:)=0;
X=X+randn(size(X)); %Adding measurement noise

%% Define weights:
distWeights=ones(size(D));
posWeights=ones(size(X,1),1);
posWeights(7)=1e-8; %Unknown

%% Reconstruct:
[pos] = getPositionFromDistances_v3(X,D,posWeights,distWeights,X);
[pos2] = getPositionFromDistances_v2(X([1:6,8:end],:),D([1:6,8:end],7),distWeights([1:6,8:end],7),X(7,:));

%v2 finds the proper solution but v3 doesnt. why? is the gradient working
%properly?

%% Compute new distances & distance to original:
distancesBeforeOpt=sum((X-oldX).^2,2)
distancesToOriginalPos=sum((oldX-pos).^2,2)
newDistancesDiff=reshape(sum((pos-reshape(pos',1,dim,M)).^2,2),M,M)-D
