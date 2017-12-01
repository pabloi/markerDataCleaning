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
            mm.trainingLogLPrctiles=prctile(trainData,0:100,2);
        end
        
        function M = get.Nmarkers(this)
            M=size(this.trainingData,1);
        end
        
        logL = loglikelihood(this,data) %Returns PxN
        i = indicatrix(this)
        
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
           markerScores= i' \ L; %Least-squares sense
           for j=1:size(L,2)
            markerScores(:,j)=fminunc(@(x) sum(abs(L(:,j)-i'*x)),markerScores(:,j)); %L1 sense
           end
           markerScores(squeeze(any(nn,2)))=NaN;
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
            frameScores=sum(markerScores,1);
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
        
    end
end

