classdef markerModel
    
    properties
        markerLabels %Mx1 cell array
        trainingData %Mx3xN
    end
    properties(Dependent)
        statsMean %Px1
        statsCov %PxP
        statsStd %Px1
    end
    
    methods
        function mm=markerModel()
            mm.markerLabels={};
            mm.trainingData=[];
        end
        function s = get.statsStd(this)
            c = this.statsCov;
            s=diag(c);
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
        mleData=invert(ss) %retunns global (but possibly non-unique) MLE estimator of
        function [newDataFrame,R,t]=anchor(dataFrame,anchorFrame)
           %Does a 3D rotation/translation of dataFrame to best match the anchorFrame
           %For a single frame:
           [R,t,newDataFrame]=getTranslationAndRotation(dataFrame,anchorFrame); %TODO
        end
    end
end

