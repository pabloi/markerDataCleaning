classdef naivePositions < markerModel
    %summary stats: pair-wise component differences
    %model: independent normals
    properties
        validComponents=1:3;
    end
    methods
        function this = naivePositions(trainData,labs,components)    
            this=this@markerModel(trainData,labs);
            if nargin>2 && ~isempty(components)
                this.validComponents=components;
            end
        end
        function logL = loglikelihood(this,data)
            ss=this.summaryStats(data);
            %sigma=this.statStd;
            sigma=this.getRobustStd(.94);
            %mu=this.statMean;
            mu=this.statMedian;
            logL=markerModel.normalLogL(ss,mu,sigma);
            %Inefficient way of doing it: for stats corresponding to
            %invalid components, set logL=NaN;
            for j=1:3
               if ~any((j-this.validComponents)==0) 
                   logL((j-1)*this.Nmarkers+[1:this.Nmarkers],:)=NaN;
               end
            end
        end
        function i = indicatrix(this) %MxP
            M=this.Nmarkers;
            %N=length(this.validComponents);
            N=3;
            i=zeros(M,N*M);
            for j=1:M
                i(j,j+[0:M:N*M-1])=1;
            end
        end
        
        function fh=seeModel(this)
           fh=this.seeModel@markerModel; 
           subplot(3,2,[2,4,6])
           hold on
           m=nanmedian(this.trainingData,3);
           i=this.indicatrix;
           sigma=this.getRobustStd(.94);
           mu=this.statMedian;
            for j=1:size(i,1) %For each marker
                n=mu(i(j,:)==1); %the components present
                q=sigma(i(j,:)==1);
                text(m(j,1),m(j,2),m(j,3)-30,['mu=[' ... 
                    num2str(n(1),3) ',' num2str(n(2),3) ',' num2str(n(3),3) ...
                    '], sigma=[' ...
                    num2str(q(1),3) ',' num2str(q(2),3) ',' num2str(q(3),3) ...
                    ']']);
            end
        end
        
    end
    methods(Static)
        function [ss,g] = summaryStats(data)
           ss=reshape(data,size(data,1)*size(data,2),size(data,3)); %Plain position
           if nargout>2
              g=eye(size(ss,1)); %Gradient of stats w.r.t 
              %each component of markers
              %For each frame? or just mean data?
           end
        end
        function this = learn(data,labels,components,noDisp)
            if nargin<2
                labels={};
            end
            if nargin<3 || isempty(components)
                components=1:3;
            end
            this=naivePositions(data,labels,components);
            if nargin<4 || isempty(noDisp) || ~noDisp
                this.seeModel()
            end
            
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
            [dataFrame] = [];% TODO
        end

    end
    methods(Static,Hidden)
        function [m,s,l]=getRefData()
            load ./data/refDataPosition.mat
        end
        function [D,xl,yl]=stat2Matrix(ss)
            D=reshape(ss,numel(ss)/3,3);
            yl='markerLabels';
            xl={'x','y','z'};
        end
    end
end

