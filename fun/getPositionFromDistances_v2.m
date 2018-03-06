function [pos] = getPositionFromDistances_v2(knownPositions,knownDistances,distWeights,initGuess)
%v2 allos for many pos to be estimated simultaneously
%INPUT:
%knownPositions: N x D  matrix, D being dimension of space
%knownDistances: N x M matrix containing distances from unknown points to
%weights: N x M vector to weigh the distances in the regression (larger
%weights means the distance is better preserved)
%OUTPUT:
%pos: M x D matrix containing D-dimensional positions for M points

[N,dim]=size(knownPositions);
[N1,M]=size(knownDistances);
if nargin<3 || isempty(distWeights)
    distWeights=ones(size(knownDistances));
elseif size(distWeights,1)~=N
    error('Weight dimensions mismatch')
end
distWeights=distWeights/sum(distWeights); %Normalizing to 1    

if nargin<4 || isempty(initGuess)
    initGuess=mean(knownPositions);
end

if size(knownDistances,1)~=N
    error('Provided distances dimension mismatch. Check that the number of distances is the same as the numer of known positions')
end

%Option 1:
%Do a least-squares regression:
opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Display','off');
pos=fminunc(@(x) distanceDistanceAllNew(reshape(x,M,dim),knownPositions,knownDistances,distWeights),initGuess(:),opts);

end

%Old: (cuadratic weighing of distances)
%Only considers distances from the unknown markers to the known ones, but
%not between the unknown ones (where arguably we can also have priors)
function [f,g]=distanceDistanceAll(x,kP,kD,w)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    [nD,gx]=pos2Dist(x,kP);
    f=sum(sum((w'.*(nD-kD')).^2)); %scalar
    g=reshape(2*sum(w'.^2.*(nD-kD').*gx,2),M,dim); %Gradient with respect to x
    g=g(:);
end

%New: (linear weighing of distances
function [f,g]=distanceDistanceAllNew(x,kP,kD,w)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    [nD,gx]=pos2Dist(x,kP);
    f=sum(sum((w'.*abs(nD-kD')))); %scalar
    g=reshape(sum(w'.*sign(nD-kD').*gx,2),M,dim); %Gradient with respect to x
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
                D1=D;
        D1(D==0)=1;
        gx=bsxfun(@rdivide,xx,D1); %NxMxdim -> This gradient is wrong if nargin==1
        %gx=xx./d; %allowed in R2017a and newer
        %gy=-gx; -> Gradients are opposite to one another if we preserve the shape
    end
end
