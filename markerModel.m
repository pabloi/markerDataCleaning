classdef markerModel
    
    properties
        markerLabels %Mx1 cell array
        trainingData %Mx3xN
    end
    
    methods
        function mm=markerModel()
            mm.markerLabels={};
            mm.trainingData=[];
        end
        
        logL = loglikelihood(this,data) %Returns PxN
        
        function markerScores = naiveScoreMarkers(this,data)
           L=loglikelihood(this,data); %PxN
           i=indicatrix();
           markerScores= i * L;
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
        
        i = indicatrix() %For each model, returns WHICH markers are involved with each stat: MxP binary, sparse
        
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

