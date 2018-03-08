classdef naiveComponents < markerModel
    %summary stats: pair-wise component differences
    %model: independent normals
    
    properties(Hidden)
        statMean
        statStd
    end
    
    methods
        function this = naiveComponents(statMean,statStd)
            this.statMean=statMean;
            this.statStd=statStd;
        end
        function logL = loglikelihood(this,data)
            ss=this.summaryStats(data);
            logL=markerModel.normalLogL(ss,this.statMean,this.statStd);
        end
    end
    methods(Static)
        function [ss,g] = summaryStats(data)
           ss=computeDifferenceMatrix(data); %Mx3xMxN
           [M,~,N]=size(data);
           %TODO
        end
        function i = indicatrix(M) %MxP
            i=[]; %TODO
        end
        function this = learn(data)
            ss=summaryStats(data);
            m=mean(ss,2);
            s=std(ss,[],2);
            this=naiveComponents(m,s);
        end
        function mleData=invert(ss)
            mleData=[];%TODO
        end
        function [dataFrame,params]=invertAndAnchor(ss,anchorFrame,anchorWeights) 
           dataFrame=[]; %TODO
           params=[];
        end
        function [newDataFrame,params]=anchor(dataFrame,anchorFrame,anchorWeights) %This needs to be model-specific, not all models require 6DoF transformation
           %For a single frame:
           params.t=mean(anchorFrame)-mean(dataFrame);
           newDataFrame=dataFrame+params.t;
        end
    end
end

