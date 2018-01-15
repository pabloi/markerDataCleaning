function [pos] = getPositionFromDistances_v3(knownPositions,knownDistances,posWeights,distWeights,initGuess)
%v2 allows for many pos to be estimated simultaneously
%v3 allows for position AND distance weights. Position weights value how
%much we should respect the given positions, while distance weights how
%much we should respect the distances between markers (sames as before)
%INPUT:
%knownPositions: N x D  matrix, D being dimension of space
%knownDistances: (N+M) x M matrix containing distances from unknown points (M) to known ones (N)
%weights: N x M vector to weigh the distances in the regression (larger
%weights means the distance is better preserved)
%OUTPUT:
%pos: M x D matrix containing D-dimensional positions for M points

[N,dim]=size(knownPositions);
[N1,M]=size(knownDistances);
if nargin<4 || isempty(distWeights)
    weights=ones(size(knownDistances));
elseif size(distWeights,1)~=N
    error('Weight dimensions mismatch')
end

if nargin<5 || isempty(initGuess)
    initGuess=mean(knownPositions);
end

if size(knownDistances,1)~=N
    error('Provided distances dimension mismatch. Check that the number of distances is the same as the numer of known positions')
end

%Option 1:
%Do a least-squares regression:
opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Display','off');
pos=fminunc(@(x) distanceDistanceAll(reshape(x,M,dim),knownPositions,knownDistances,posWeights,distWeights),initGuess(:),opts);

end

%Old: (cuadratic weighing of distances)
function [f,g]=distanceDistanceAll(x,kP,kD,w)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    %xx=bsxfun(@minus,x,reshape(kP',1,size(kP,2),size(kP,1))); %M x dim x N
    %normXX=sqrt(sum(xx.^2,2)); %M x 1 x N
    %f=sum(sum((w'.*(reshape(normXX,M,N)-kD')).^2)); %scalar
    [nD,gx]=pos2Dist(x,kP);
    f=sum(sum((w'.*(nD-kD')).^2)); %scalar
    g=reshape(2*sum(w'.^2.*(nD-kD').*gx,2),N,dim); %Gradient with respect to x
    %gg1=2*w'.^2.*(nD-kD'); %M x N
    %gg2=bsxfun(@rdivide,xx,normXX); %M x dim x N
    %gg=bsxfun(@times,reshape(gg1,size(gg2,1),1,size(gg2,3)),gg2); %M x dim x N
    %g=sum(gg,3); %M x dim
    g=g(:);
end

%New: (linear weighing of distances
function [f,g]=distanceDistanceAllNew(x,kP,kD,w)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    %xx=bsxfun(@minus,x,reshape(kP',1,size(kP,2),size(kP,1))); %M x dim x N
    %normXX=sqrt(sum(xx.^2,2)); %M x 1 x N
    %f=sum(sum((w'.*(reshape(normXX,M,N)-kD')).^2)); %scalar
    [nD,gx]=pos2Dist(x,kP);
    f=sum(sum((w'.*abs(nD-kD')))); %scalar
    g=reshape(sum(w'.*sign(nD-kD').*gx,2),N,dim); %Gradient with respect to x
    %gg1=2*w'.^2.*(nD-kD'); %M x N
    %gg2=bsxfun(@rdivide,xx,normXX); %M x dim x N
    %gg=bsxfun(@times,reshape(gg1,size(gg2,1),1,size(gg2,3)),gg2); %M x dim x N
    %g=sum(gg,3); %M x dim
    g=g(:);
end

%Newest: linear weighing of distances between markers + movement from original
function [f,g]=distanceDistanceAllMixed(x,kP,kD,wP,wD)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    [D1,g1]=pos2Dist(x); 
    [D2,g2]=pos2Dist(x,kP); %We care only about the diagonal of this
    D2=diag(D2);
    for i=1:M
    g2=g2();
    
end

function [D,gx]=pos2Dist(x,y)
    %x is Nxdim
    %y is Mxdim
    %D is NxM matrix containing distances
    if nargin<2 || isempty(y)
        y=x;
    end
    [N,dim]=size(x);
    [M,dim]=size(y);
    xx=bsxfun(@minus,x,reshape(y',1,dim,M)); %N x dim x M
    xx=permute(xx,[1,3,2]); %NxMxdim
    D=sqrt(sum(xx.^2,3)); %NxM
    if nargout>1 %Computing gradients too
        gx=bsxfun(@rdivide,xx,d); %NxMxdim
        %gx=xx./d; %allowed in R2017a and newer
        %gy=-gx; -> Gradients are opposite to one another if we preserve the shape
    end
end

%% A little script to test distanceDistanceAll:
% X1=randn(10,3);
% D=computeDistanceMatrix(X1);
% kP=X1(1:7,:);
% kD=D(1:7,8:10);
% w=ones(size(kD));
% 
% %% Eval:
% xA=randn(3,3);
% [fA,gA]=distanceDistanceAll(xA,kP,kD,w);
% xB=bsxfun(@plus,xA,[0, 0, 1e-5]);
% [fB,gB]=distanceDistanceAll(xB,kP,kD,w);
% xC=bsxfun(@plus,xA,[0, 1e-5, 0]);
% [fC,gC]=distanceDistanceAll(xC,kP,kD,w);
% xD=bsxfun(@plus,xA,[1e-5, 0, 0]);
% [fD,gD]=distanceDistanceAll(xD,kP,kD,w);
% sum(reshape(gA,3,3),1)
% 
% [(fD-fA) (fC-fA) (fB-fA)]/1e-5
% 
% [fA,~]=distanceDistanceAll(X1(8:10,:),kP,kD,w)
