classdef naiveDistances < markerModel
    %summary stats: pair-wise component differences
    %model: independent normals
    
    properties(Hidden)
        statMean
        statStd
    end
    
    methods
        function this = naiveDistances(statMean,statStd)
            this.statMean=statMean;
            this.statStd=statStd;
        end
        function logL = loglikelihood(this,data)
            ss=this.summaryStats(data);
            logL=markerModel.normalLogL(ss,this.statMean,this.statStd);
        end
    end
    methods(Static)
        function ss = summaryStats(data)
           ss=computeDistanceMatrix(data);
           [M,~,N]=size(data);
           ss=reshape(ss,M^2,N);
           ind=triu(ones(M),1);
           ss=ss(ind,:); %Keeping only upper half: PxN, with P=M*(M-1)/2
        end
        function i = indicatrix(M) %MxP
            ind=triu(ones(M),1);
            i=nan(M,M*(M-1)/2);
            for j=1:M
                aux=zeros(M);
                aux(:,j)=1;
                aux(j,:)=1;
                i(j,:)=aux(ind);
            end
        end
        function this = learn(data)
            ss=summaryStats(data);
            m=mean(ss,2);
            s=std(ss,[],2);
            this=naiveDistances(m,s);
        end
        function mleData=invert(ss)
            mleData=[];%TODO
        end
        function [newDataFrame,params]=anchor(dataFrame,anchorFrame,anchorWeights) %This needs to be model-specific, not all models require 6DoF transformation
           %Does a 3D rotation/translation of dataFrame to best match the anchorFrame
           %For a single frame:
           [R,t,newDataFrame]=getTranslationAndRotation(dataFrame,anchorFrame);
           params.R=R;
           params.t=t;
        end
        function dataFrame=invertAndAnchor(ss,anchorFrame,anchorWeights)
            %TODO
            %Use distTo3D or getPositionFromDistances (v1 or v2)
        end
    end
end

