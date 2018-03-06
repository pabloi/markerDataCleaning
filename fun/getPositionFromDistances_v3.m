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
noPositionWeighing=false;
if nargin<3 || isempty(posWeights)
   posWeights= ones(size(knownPositions));
elseif size(posWeights,1)~=N
    error('Weight dimensions mismatch')
end
if nargin<4 || isempty(distWeights)
    distWeights=ones(size(knownDistances)); %Weigh all distances equally
elseif size(distWeights,1)~=N || size(distWeights,2)~=N
    error('Weight dimensions mismatch')
end

if nargin<5 || isempty(initGuess)
    initGuess=[];
end

if N1~=N
    error('Provided distances dimension mismatch. Check that the number of distances is the same as the numer of known positions')
end
distWeights=triu(distWeights,1); %Because distances are doubled, I am only honoring the upper half of the distribution

%Option 1:
%Use Matlab's optim:
%opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','final','FunctionTol',1e-12,'StepTolerance',1e-12);
%trust-region doesn't work well for this problem. Using quasi-newton w/o gradient:
%opts = optimoptions('fminunc','SpecifyObjectiveGradient',false,'Display','final','FunctionTol',1e-6,'StepTolerance',1e-6,'MaxFunctionEvaluations',1e4);
%pos=fminunc(@(x) cost(reshape(x,N,dim),knownPositions,knownDistances,posWeights,distWeights),initGuess(:),opts);
%pos=reshape(pos,N,dim);
%[f,g,h]=cost(pos,knownPositions,knownDistances,posWeights,distWeights);

%Opt2: do my own:
%Fast and loose search with an easy environment:
[bestX,~,~]=minCost(knownPositions,knownDistances,posWeights,distWeights.^.3,initGuess);
%Optimality search around prev. solution:
[pos,bestF,count]=minCost(knownPositions,knownDistances,posWeights,distWeights,bestX);


end
function [f,g,h]=distanceCost(x,kD,wD,kP)
%If kP is given, computing all pairwise distance between elements of {kP,x}
    if nargin>3
        x=[x;kP];
    end
    [N,dim]=size(x);
    wD=(wD).^2; %NxN
    [D1,g1]=pos2Dist(x);  %Can also use pos2Dist2 for quadratic weighing
    a1=wD.*(D1-kD);
    f1=a1.*(D1-kD); %NxN
    f=sum(f1(:));
    g1=reshape(g1,N^2,N*dim);
    g=reshape(2*sum(a1(:)'*g1,1),N,dim);
    h=[];
    %TODO: compute cost of dist(x,x) and dist(x,kP) and sum them, avoiding
    %the computation of dist(kP,kP) which is fixed and useless
end
function [f,g,h]=positionCost(x,kP,wP)
    [D2,g2]=pos2DistDiag(x,kP); %Only care about diagonal of this
    a2=wP.^2.*D2;
    f2=a2.*D2;
    f=sum(f2(:));
    g=2*(a2.*g2);
    h=[];
end
function [f,g,h,f1,f2]=cost2(x,kP,kD,wP,wD)
%     [N,dim]=size(kP);
%     wD=.5*(wD+wD').^2; %NxN
%     [D1,g1]=pos2Dist(x);  %Can also use pos2Dist2 for quadratic weighing
%     [D2,g2]=pos2DistDiag(x,kP); %We care only about the diagonal of this
%     a1=wD.*(D1-kD);
%     a2=wP.^2.*D2;
%     f1=a1.*(D1-kD); %NxN
%     f2=a2.*D2;
%     f=sum(f1(:))+sum(f2(:));
%     g1=reshape(g1,N^2,N*dim);
%     g=reshape(2*sum(a1(:)'*g1,1),N,dim)+2*(a2.*g2);
%     h=[];
[f1,g1,~]=distanceCost(x,kD,wD);
[f2,g2,~]=positionCost(x,kP,wP);
f=f1+f2;
g=g1+g2;
h=[];
end
function [f,g,h]=cost2Fixed(x,kP,kD,wD)
    %Same as cost2, but no position weighing (known positions are fixed)
    [f,g,~]=distanceCost(x,kD,wD,kP);
    h=[];
end
function [bestX,bestF,count]=minCost(Y,kD,wP,wD,initGuess)
if nargin<5 || isempty(initGuess)
    X=Y;
else
    X=initGuess;
end
verbose=false;
display=false;
noPositionWeighing=false;
if all(wP==0)
    noPositionWeighing=true;
end
if ~noPositionWeighing
    [f,gX,~]=cost2(X,Y,kD,wP,wD);
else
    [f,gX,~]=cost2Fixed(X,Y,kD,wD);
end
lambda=.5*f/norm(gX(:))^2;
oldF=Inf;count=0;bestF=Inf;stuckCounter=0;bestX=X; f=Inf; gradTh=1e-1;
countThreshold=1e5;funThreshold=1e-5;stuckThreshold=100; updateCount=10;
if display
    fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
    plot3(Y(:,1),Y(:,2),Y(:,3),'ko','MarkerSize',10)
    hold on
    plot3(X(:,1),X(:,2),X(:,3),'o')
    axis equal
    view(3)
    Q=quiver3(X(:,1),X(:,2),X(:,3),-gX(:,1),-gX(:,2),-gX(:,3),0);
    title(['cost=' num2str(f) ',\lambda=' num2str(lambda) ',bestCost=' num2str(bestF) ',stuckCount=' num2str(stuckCounter) ',max |g|=' num2str(max(sqrt(sum(gX.^2))))])
end
while f>funThreshold && count<countThreshold && stuckCounter<stuckThreshold && any(sum(gX.^2,2)>gradTh.^2)
    if ~noPositionWeighing
        [f,gX]=cost2(X,Y,kD,wP,wD);
    else
        [f,gX]=cost2Fixed(X,Y,kD,wD);
    end
    count=count+1;
    if f<(bestF-.1) %Found best point so far
        bestF=f;bestX=X;stuckCounter=0;
    else
        stuckCounter=stuckCounter+1;
    end
    if mod(count,updateCount)==0 %Every 10 steps, update lambda
        if f>1.01*oldF %Objective function increased noticeably(!) -> reducing lambda
            lambda=.5*lambda;
        elseif f>.95*oldF %Decreasing, but not decreasing fast enough
            lambda=1.1*lambda; oldF=f;%Increasing lambda, in hopes to speed up
        else %Decreasing at good rate: doing nothing
            oldF=f;
        end
        if display
            plot3(bestX(:,1),bestX(:,2),bestX(:,3),'rx')
            title(['cost=' num2str(f) ',\lambda=' num2str(lambda) ',bestCost=' num2str(bestF) ',stuckCount=' num2str(stuckCounter) ',max |g|=' num2str(max(sqrt(sum(gX.^2))))])
            delete(Q)
            Q=quiver3(X(:,1),X(:,2),X(:,3),gX(:,1),gX(:,2),gX(:,3),0);
            drawnow
        end
    end
    dX=lambda.*gX;
    %dX(fixedMarkers,:)=0; %No change for fixed markers
    X=X-dX;
end
%Determining ending criteria:
if f<=funThreshold
    bestF=f;   bestX=X;
    if verbose
    disp('Objective function is below threshold')
    end
elseif count>=countThreshold
    if verbose
    disp('Too many iterations. Stopping.')
    end
elseif stuckCounter>=stuckThreshold
    if verbose
        disp('We are lost. Stopping.')
    end
elseif all(sum(gX.^2,2)<gradTh.^2)
    if verbose
        disp('Gradient is below tolerance for all markers')
    end
else %Should never happen!
    error('')
    %pause
end
if display
plot3(bestX(:,1),bestX(:,2),bestX(:,3),'kx','MarkerSize',10,'LineWidth',4)
title(['cost=' num2str(bestF) ',\lambda=' num2str(lambda) ',bestCost=' num2str(bestF) ',stuckCount=' num2str(stuckCounter)])
drawnow
end
end
