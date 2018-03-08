
%% Generate some random frame:
X=500*randn(12,3);
X(7,:)=0;
[M,dim]=size(X);

%% Compute distances:
D=pos2Dist(X);
%D=D+randn(size(D)); %Adding noise for incompatibility of exact distances, as would happen in a real model

%% Remove one marker:
oldX=X;
X(7,:)=500*randn(1,3);
%X=X+randn(size(X)); %Adding measurement noise

%% Define weights:
distWeights=ones(size(D));
posWeights=ones(size(X,1),1);
posWeights(7)=0; %Unknown

%% Reconstruct:
[pos] = getPositionFromDistances_v3(X,D,posWeights,distWeights,X+randn(size(X)));
[pos2] = getPositionFromDistances_v2(X([1:6,8:end],:),D([1:6,8:end],7),distWeights([1:6,8:end],7),X(7,:));

%% Compute new distances & distance to original:
distancesBeforeOpt=sum((X-oldX).^2,2)
distancesToOriginalPos=sum((oldX-sol).^2,2)
distanceToStartPoint=sum((sol-X).^2,2)
newDistancesDiff=pos2Dist(sol)-D