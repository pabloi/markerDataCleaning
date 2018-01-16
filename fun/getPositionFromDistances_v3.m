function [pos] = getPositionFromDistances_v3(knownPositions,knownDistances,posWeights,distWeights,initGuess)
%v3: many changes
%pos is the Nx3 matrix that minimizes:
%sum((posWeights).*(pos-knownPositions)*(pos-knownPositions)')+ .5*sum(sum((distWeights.*abs(d(pos_i,pos_j)-knownDistances)))
%INPUT:
%knownPositions: N x D  matrix, D being dimension of space
%knownDistances: NxN matrix
%posWeights: Nx1 weight vector
%distWeights: NxN weight matrix
%OUTPUT:
%pos: N x D matrix containing D-dimensional positions for N points

%TODO: if posWeights=Inf for some component, it gets excluded from the
%optimization. If posWeights=0, it gets excluded from knownPositions.

[N,dim]=size(knownPositions);
[N1,N2]=size(knownDistances);
if N~=N1 || N1~=N2
    error('Dimension mismatch')
end
if nargin<3 || isempty(posWeights)
   posWeights= ones(size(knownPositions));
elseif size(posWeights,1)~=N
    error('Weight dimensions mismatch')
end
if nargin<4 || isempty(distWeights)
    distWeights=ones(size(knownDistances));
elseif size(distWeights,1)~=N || size(distWeights,2)~=N
    error('Weight dimensions mismatch')
end

if nargin<5 || isempty(initGuess)
    initGuess=mean(knownPositions);
end

if N1~=N
    error('Provided distances dimension mismatch. Check that the number of distances is the same as the numer of known positions')
end
distWeights=triu(distWeights,1); %Because distances are doubled, I am only honoring the upper half of the distribution

%Option 1:
%Do a least-squares regression:
opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Display','off');
pos=fminunc(@(x) distanceDistanceAllMixed(reshape(x,N,dim),knownPositions,knownDistances,posWeights,distWeights),initGuess(:),opts);

pos=reshape(pos,N,dim);
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
    [D2,g2a]=pos2Dist(x,kP); %We care only about the diagonal of this
    D2=diag(D2);
    %g2=nan(M,dim);
    for i=1:dim
        g2(:,i)=diag(g2a(:,:,i));
    end
    f=sum(sum(wD.*abs(D1-kD))) + wP'*D2;
    g=reshape(sum(wD.*sign(D1-kD).*g1,2),N,dim) + wP.*g2;
    g=g(:);
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
        gx=bsxfun(@rdivide,xx,D); %NxMxdim
        for k=1:3
            aux=gx(:,:,k);
            aux(D==0)=1;
            gx(:,:,k)=aux;
        end
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
