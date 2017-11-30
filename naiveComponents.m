classdef naiveComponents < markerModel
    %summary stats: pair-wise component differences
    %model: independent normals
    
    properties
        Property1
    end
    
    methods
        function obj = naiveComponents(statMean,statStd)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end
        
        function logL = loglikelihood(this,data)
            
        end
    end
    methods(Static)
        function ss = summaryStats(data)
           ss=computeDistanceMatrix(data);
           [M,~,N]=size(data);
           ss=reshape(ss,M^2,N);
           ind=triu(ones(M),1);
           ss=ss(ind,:); %Keeping only upper half
        end
        function i = indicatrix()
            i=[];%TODO
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
    end
end

