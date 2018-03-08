function [pos] = getPositionFromDistances(knownPositions,knownDistances,weights,initGuess)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%INPUT:
%knownPositions: N x D matrix, D being dimension of space
%knownDistances: N x 1 vector containing distances from unknown point to
%knownPoints
%weights: N x 1 vetor to weigh the distances in the regression (larger
%weights means the distance is better preserved)

[N,M]=size(knownPositions);
if nargin<3 || isempty(weights)
    weights=ones(size(knownDistances));
elseif length(weights)~=N
    error('Weight dimensions mismatch')
end
weights=weights/sum(weights); %Normalizing to 1    

if nargin<4 || isempty(initGuess)
    initGuess=mean(knownPositions);
end

if length(knownDistances)~=N
    error('Provided distances dimension mismatch. Check that the number of distances is the same as the numer of known positions')
end

%Option 1:
%Do a least-squares regression:
opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Display','off');
pos=fminunc(@(x) distanceDistance(x,knownPositions,knownDistances,weights),initGuess,opts);

%Do a modified least-squares regression:
%opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'Display','off');
%pos=fminunc(@(x) distanceDistance2(x,knownPositions,knownDistances,weights),initGuess,opts);


%Option 2: use mdscale (very slow, and apparently not better)
% D=computeDistanceMatrix(knownPositions);
% D(:,end+1)=knownDistances;
% D(end+1,:)=[knownDistances' 0];
% posRel=mdscale(D,M); %Here we get best-fitting positions, relative to one another: need to find rotation to match coordinates in abs space vs. mdscale space
% [R,t,~] = getRotationAndTranslation(posRel(1:end-1,:),knownPositions);
% posRel=bsxfun(@plus,posRel*R,t);
% pos=sum(bsxfun(@times,weights,bsxfun(@minus,posRel(end,:),posRel(1:end-1,:))+knownPositions)); %Here we get the absolute frame
end

function [f,g]=distanceDistance(x,kP,kD,w)
    xx=bsxfun(@minus,x,kP);
    normXX=sqrt(sum(xx.^2,2));
    f=norm(w.*(normXX-kD))^2;
    gg1=2*w.^2.*(normXX-kD);
    gg2=bsxfun(@rdivide,xx,normXX);
    gg=bsxfun(@times,gg1,gg2);
    g=sum(gg,1);
end

function [f,g]=distanceDistance2(x,kP,kD,w)
    xx=bsxfun(@minus,x,kP);
    normXX=sqrt(sum(xx.^2,2));
    f=norm(w.^2 .*(normXX.^2-kD.^2))^2;
    gg1=2*w.^4.*(normXX.^2-kD.^2);
    gg2=2*xx;
    gg=bsxfun(@times,gg1,gg2);
    g=sum(gg,1);
end

%% A little script to test distanceDistance:
% 
% X1=randn(10,3);
% D=computeDistanceMatrix(X1);
% kP=X1(1:9,:);
% kD=D(1:9,10);
% w=ones(size(kD));
% 
% %% Eval:
% xA=randn(1,3);
% [fA,gA]=distanceDistance(xA,kP,kD,w);
% xB=xA+[0, 0, 1e-5];
% [fB,gB]=distanceDistance(xB,kP,kD,w);
% xC=xA+[0, 1e-5, 0];
% [fC,gC]=distanceDistance(xC,kP,kD,w);
% xD=xA+[ 1e-5, 0, 0];
% [fD,gD]=distanceDistance(xD,kP,kD,w);
% gA
% [(fD-fA) (fC-fA) (fB-fA)]/1e-5
% 
% [fA,~]=distanceDistance(X1(10,:),kP,kD,w)
