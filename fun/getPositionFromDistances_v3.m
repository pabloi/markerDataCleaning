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
%Use Matlab's optim:
%opts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','final','FunctionTol',1e-12,'StepTolerance',1e-12);
%trust-region doesn't work well for this problem. Using quasi-newton w/o gradient:
%opts = optimoptions('fminunc','SpecifyObjectiveGradient',false,'Display','final','FunctionTol',1e-6,'StepTolerance',1e-6,'MaxFunctionEvaluations',1e4);
%pos=fminunc(@(x) cost(reshape(x,N,dim),knownPositions,knownDistances,posWeights,distWeights),initGuess(:),opts);
%pos=reshape(pos,N,dim);
%[f,g,h]=cost(pos,knownPositions,knownDistances,posWeights,distWeights);

%Opt2: do my own:
%Fast and loose search with an easy environment:
[bestX,~,~]=minCost(knownPositions,knownDistances,posWeights,distWeights.^.3);
%Optimality search around prev. solution:
[pos,bestF,count]=minCost(knownPositions,knownDistances,posWeights,distWeights,bestX);


end

function [f,g,h,f1,f2]=cost(x,kP,kD,wP,wD)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    [D1,g1,h1]=pos2Dist(x);  %Can also use pos2Dist2 for quadratic weighing   
    [D2,g2,h2]=pos2Dist(x,kP); %We care only about the diagonal of this
    f1=.5*(wD+wD').*abs(D1-kD);
    f2=diag(wP).*D2;
    f=sum(sum(f1+f2));
    g=reshape(sum(sum(wD.*sign(D1-kD).*g1,2),1),N*dim,1) + reshape(sum(sum(diag(wP).*g2,1),2),N*dim,1);
    h=reshape(sum(sum(wD.*sign(D1-kD).*h1,2),1),N*dim,N*dim) + reshape(sum(sum(diag(wP).*h2,1),2),N*dim,N*dim);
end
function [f,g,h,f1,f2]=cost2(x,kP,kD,wP,wD)
    [M,dim]=size(x);
    [N,dim]=size(kP);
    wD=wD.^2;
    wP=wP.^2;
    %WLOG: normalizing weights so sum(wD(:)+wP(:))=1;
    %K=N^2*sum(wD(:))+sum(wP(:));
    %wD=wD/K;
    %wP=wP/K;
    [D1,g1,h1]=pos2Dist(x);  %Can also use pos2Dist2 for quadratic weighing   
    [D2,g2,h2]=pos2Dist(x,kP); %We care only about the diagonal of this
    f1=.5*(wD+wD').*(D1-kD).^2;
    f2=diag(wP).*D2.^2;
    f=sum(sum(f1+f2));
    g=reshape(sum(sum(2*wD.*(D1-kD).*g1,2),1),N*dim,1) + reshape(sum(sum(2*diag(wP).*D2.*g2,1),2),N*dim,1);
    h=[];
end

function [bestX,bestF,count]=minCost(Y,kD,wP,wD,initGuess)
if nargin<5 || isempty(initGuess)
    X=Y;
else
    X=initGuess;
end
verbose=true;
[f,g,~]=cost2(X,Y,kD,wP,wD);
lambda=.5*f/norm(g)^2;
%lambda=1e-3;
%lambda=.1;
oldF=Inf;count=0;bestF=Inf;stuckCounter=0;bestX=X; f=Inf; gradTh=1e-1;
countThreshold=1e5;funThreshold=1e-5;stuckThreshold=100; updateCount=10;
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
plot3(Y(:,1),Y(:,2),Y(:,3),'ko','MarkerSize',10)
hold on
plot3(X(:,1),X(:,2),X(:,3),'o')
axis equal
view(3)
gX=-reshape(g,size(X));
Q=quiver3(X(:,1),X(:,2),X(:,3),gX(:,1),gX(:,2),gX(:,3),0);
title(['cost=' num2str(f) ',\lambda=' num2str(lambda) ',bestCost=' num2str(bestF) ',stuckCount=' num2str(stuckCounter) ',max |g|=' num2str(max(sqrt(sum(gX.^2))))])      
while f>funThreshold && count<countThreshold && stuckCounter<stuckThreshold && any(sum(gX.^2,2)>gradTh.^2)
    [f,g,~,f1,f2]=cost2(X,Y,kD,wP,wD);
    count=count+1;
    if f<(bestF-.1) %Found best point so far
        bestF=f;bestX=X;stuckCounter=0;
    else
        stuckCounter=stuckCounter+1;
    end
    if mod(count,updateCount)==0 %Every 10 steps, update lambda
        if f>1.01*oldF %Objective function increased noticeably(!) -> reducing lambda
            lambda=.5*lambda;
            %oldF=f;
        elseif f>.95*oldF %Decreasing, but not decreasing fast enough
            lambda=1.1*lambda; %Increasing lambda, in hopes to speed up
            oldF=f;
        else %Decreasing at good rate: doing nothing
            oldF=f;
        end
        plot3(bestX(:,1),bestX(:,2),bestX(:,3),'rx')
        gX=-lambda*reshape(g,size(X));
        title(['cost=' num2str(f) ',\lambda=' num2str(lambda) ',bestCost=' num2str(bestF) ',stuckCount=' num2str(stuckCounter) ',max |g|=' num2str(max(sqrt(sum(gX.^2))))])      
        delete(Q)
        Q=quiver3(X(:,1),X(:,2),X(:,3),gX(:,1),gX(:,2),gX(:,3),0);
        drawnow
    end   
    
    gX=reshape(g,size(X));
    dX=lambda.*gX;
    %d=sqrt(sum(dX.^2,2));
    %th=50;
    %dX(d>th,:)=dX(d>th,:)*th./d(d>th);
    %aux=randn(size(dX));
    %dX(isnan(dX))=aux(isnan(dX));
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
    f
    count
    stuckCounter
    dX
end
plot3(bestX(:,1),bestX(:,2),bestX(:,3),'kx','MarkerSize',10,'LineWidth',4)
title(['cost=' num2str(bestF) ',\lambda=' num2str(lambda) ',bestCost=' num2str(bestF) ',stuckCount=' num2str(stuckCounter)])
drawnow
end

%% A little script to test cost:
% %% Data
% N=4;
% dim=3;
% X=randn(N,dim);
% x2=randn(N,dim);
% kD=randn(N,N);
% wP=randn(N,1);
% wD=randn(size(kD));
% 
% %% comparing gradient in cross-distance to empirical results
% 
% [d,g,h]=cost(X,x2,kD,wP,wD);
% epsilon=1e-7;
% empG=nan(N,dim);
% empH=nan(N,dim,N,dim);
% for i=1:N
%     for k=1:dim
%         aux=zeros(size(X));
%         aux(i,k)=epsilon;
%         [d1,g1,h1]=cost(X+aux,x2,kD,wP,wD);
%         empG(i,k)=(d1-d)/epsilon;
%         empH(:,:,i,k)=(reshape(g1,N,dim)-reshape(g,N,dim))/epsilon;
%     end
% end
% disp(['Max gradient element: ' num2str(max(abs(g(:))))])
% disp(['Max gradient err: ' num2str(max(abs(g(:)-empG(:))))])
% disp(['Max gradient err (%): ' num2str(100*max(abs(g(:)-empG(:))./abs(g(:))))])
% 
% disp(['Max hessian element: ' num2str(max(abs(h(:))))])
% disp(['Max hessian err: ' num2str(max(abs(h(:)-empH(:))))])
% disp(['Max hessian err (%): ' num2str(100*max(abs(h(:)-empH(:))./abs(h(:))))])
