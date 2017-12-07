classdef markerModel
    
    properties
        markerLabels %Mx1 cell array
        trainingData %Mx3xN
        statMean
        statStd
        trainingLogLPrctiles=[]; %Stores prctiles for log-l distribution in training data
        statPrctiles
    end
    properties(Dependent)
        Nmarkers
    end
    
    methods
        function mm=markerModel(trainData,labels)
            mm.markerLabels=labels;
            mm.trainingData=trainData;
            trainStats=mm.summaryStats(trainData);
            mm.statMean=nanmean(trainStats,2);
            mm.statStd=nanstd(trainStats,[],2);
            mm.statPrctiles=prctile(trainStats,0:100,2);
            mm.trainingLogLPrctiles=prctile(mm.loglikelihood(trainData),0:100,2);
        end
        
        function M = get.Nmarkers(this)
            M=size(this.trainingData,1);
        end
        
        logL = loglikelihood(this,data) %Returns PxN
        i = indicatrix(this)
        
        function outlierMarkers = outlierDetect(this,data)
            i=indicatrix(this); %MxP
            %outStats=summaryStats(this,data)<this.statPrctiles(:,2) | s>this.statPrctiles(:,100); %Finding stats in the 1% tails (both) 
            outStats=this.loglikelihood(data) < this.trainingLogLPrctiles(:,2); %Finding likelihood in 1st percentile
            outlierMarkers=i'\outStats >.1; %This is the correct form, though i*outstats returns comparable results and is sparse
            
        end
        
        function outlierMarkers = outlierDetectv2(this,data)
            i=indicatrix(this); %MxP
            %outStats=summaryStats(this,data)<this.statPrctiles(:,2) | s>this.statPrctiles(:,100); %Finding stats in the 1% tails (both) 
            outStats=this.loglikelihood(data) < this.trainingLogLPrctiles(:,2); %Finding likelihood in 1st percentile
            %Alt: (optimal in some sense, but slow)
            outlierStatCountPerMarker=i * outStats; %Counting outlier stats per marker
            aux=any(outlierStatCountPerMarker>1);
            outlierMarkers=zeros(size(outlierStatCountPerMarker));
            for j=1:size(outlierStatCountPerMarker,2)
                if aux(j) %Not optimizing if at least one marker doesn't have 2 outlying stats
                    outlierMarkers(:,j)=markerModel.untanglePairedStats(outlierStatCountPerMarker(:,j)); %This is better than thresholding to some arbitrary value
                end
            end

        end
        
        function markerScores = naiveScoreMarkers(this,data)
            nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           i=indicatrix(this);
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           markerScores= (i * L)./sum(i,2);
           markerScores(squeeze(any(nn,2)))=NaN;
        end
        
        function markerScores = indScoreMarkers(this,data)
           nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerScores= i' \ L; %Least-squares sense, doesn't really work
        end
        
        function markerScores = indScoreMarkersv2(this,data)
           nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerMeans= (i * L)./sum(i,2); 
           [M,P]=size(i);
           %Solve: 
           uno=ones(M,1);
           cero=zeros(M,1);
           f=[cero;uno];
           lb=[-Inf*uno; cero];
           ub=[Inf*uno;uno];
           K=max(abs(L(:)));
           markerScores=nan(M,size(L,2));
           opts=optimoptions('intlinprog','Display','off');
           for j=1:size(L,2)
           %Ineq 1: L >= i'*p;
           A1=[i' zeros(P,M)]; b1=L(:,j);
           %Ineq 2: y>(mm-p)/K -> -p/K-y < -mm/K
           A2=[-eye(M)/K -eye(M)]; b2=-markerMeans(:,j)/K;
           %Ineq 3: 1-y >= -(mm-p)/K -> 1+mm/K>= -y+p/K
           A3=[eye(M)/K -eye(M)]; b3=uno+markerMeans(:,j)/K;
           py=intlinprog(f,[M+1:2*M],[A1;A2;A3],[b1;b2;b3],[],[],lb,ub,opts);
           markerScores(:,j)=py(1:M);
           end
        end
        
        function markerScores = medianScoreMarkers(this,data)
           nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerScores=nan(size(i,1),size(L,2));
           for j=1:size(i,1)
               markerScores(j,:)=median(L(i(j,:)==1,:));
           end
           markerScores(squeeze(any(nn,2)))=NaN;
        end
        
        function markerScores = rankedScoreMarkers(this,data,N)
            if nargin<3
                N=3; %Using third-worse score
            end
           nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerScores=nan(size(i,1),size(L,2));
           for j=1:size(i,1)
               aux=sort(L(i(j,:)==1,:),1);
               markerScores(j,:)=aux(N,:);
           end
           markerScores(squeeze(any(nn,2)))=NaN;
        end
        
        function frameScores = naiveScoreFrames(this,data) %1xN
            markerScores = naiveScoreMarkers(this,data);
            frameScores=nanmean(markerScores,1);
        end
    end
    
    methods(Static)
        model = learn(data)
        
        [ss,g] = summaryStats(data) %Returns PxN summary stats, and P x 3M x N gradient 
        %gradient can be P x 3M if it is the same for all frames, as is the case in linear models
                
        mleData=invert(ss) %returns global (but possibly non-unique) MLE estimator of
        
        [dataFrame,params]=invertAndAnchor(ss,anchorFrame,anchorWeights) 
        %This is offered on top of anchor alone because for some models it 
        %may be optimal to do both things together rather than in stages
        
        [dataFrame,params]=anchor(ss,anchorFrame,anchorWeights)
        
        function lL=normalLogL(values,means,stds)
            %values is PxN, means and stds are Px1
            %Returns PxN likelihood of value under each normal
            d=.5*((values-means)./stds).^2;
            lL=-d -log(stds) -.9189;
            %.9189 = log(2*pi)/2
        end
        
        function outlierMarkers=untanglePairedStats(outlierStatCountPerMarker)
            %TODO: same thing for arbitrary indicatrix matrices (not just
            %paired ones)
            
            %Idea: given some candidate outliers scored with integers based on how many
            %outlier stats each marker participates, find a classification (binary) for
            %the minimum number of outlier markers that explains the outlier stats.
            %Only works if stats are always computed from 2 and only 2 markers.
            %Let p be the outlier scores of each markers' stats, then the problem is a
            %mixed-integer programming of the form:
            %min 1'*x [minimum number of outlier markers possible]
            %subject to 
            %y+z = p - 1'*x; y>=0; z<=0; [y quantifies how many more outlier stats
            %than there are outlier markers each marker has; z how many less, one of
            %the two is always 0]
            %y>=x-1 & y-max(p)*x<=1 [these two enforce x=(y>=0)]%We allow for
            %up to 1 outlier stat w/o it being an outlier marker
            %(y-1)-max(p)*x <=0 [this enforces x=1 if y>1, but allows any x for y<=1]
            %z+1<=x & z>=max(p)*(x-1) [these two enforce x=(z==0)]

            beq=outlierStatCountPerMarker;
            M=numel(beq);
            uno=ones(M,1);
            cero=zeros(M,1);
            %Strict equality: z+y = b-1'*x -> b = 1'*x+y+z;
            Aeq=[ones(M) eye(M) eye(M)]; %M x 3M
            %Inequality: y>=x -> x-y <= 0
            %A=[eye(M), -eye(M), zeros(M)];
            %b=cero;
            %Inequality: y-max(b)*x <=1
            %A=[A; -max(beq)*eye(M), eye(M), zeros(M)];
            %b=[b;uno];
            %Ineq: y-1-max(p)*x<=0 -> y-max(p)*x <=1
            A=[-max(beq)*eye(M), eye(M), zeros(M)];
            b=uno;
            
            %Inequality: z+1<=x -> z-x<=-1
            A=[A;-eye(M), zeros(M), eye(M)];
            b=[b;-uno];
            %Ineq: z>=max(p)*(x-1) -> max(p)*x-z <= max(p)
            A=[A;max(beq)*eye(M), zeros(M), -eye(M)];
            b=[b;max(beq)*uno];
            %Inequality: 
            %Constraints:
            lb=[cero;cero;-Inf*uno];
            ub=[uno; Inf*uno;cero];
            %Solve
            f=[uno;cero;cero];
            opts=optimoptions('intlinprog','Display','off');
            xy=intlinprog(f,1:3*M,A,b,Aeq,beq,lb,ub,opts); %Solving for integer (binary) x
            outlierMarkers=xy(1:M);
            y=xy(M+1:2*M);
            z=xy(2*M+1:end);
        end
        
    end
end

