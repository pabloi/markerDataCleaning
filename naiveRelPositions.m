classdef naiveRelPositions < markerModel
    %summary stats: pair-wise component differences
    %model: independent normals
    properties
        validComponents=1:3;
    end
    methods
        function this = naiveRelPositions(trainData,labs,components)    
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
                   logL(j:3:end,:)=NaN;
               end
            end
        end
        function i = indicatrix(this) %MxP
            i=[];
        end
        
        function fh=seeModel(this)
           fh=this.seeModel@markerModel; 
        end
        
    end
    methods(Static)
        function [ss,g] = summaryStats(data)
           %ss=reshape(data,size(data,1)*size(data,2),size(data,3)); %Rel position
           [D] = computeDiffMatrix(data); %Mx3xMxN
           aux=cell(3,1);
           for k=1:3
               aux{k}=distMatrix2stat(squeeze(D(:,k,:,:)));
           end
           ss=cell2mat(aux);
           if nargout>2
              g=[]; %Gradient of stats w.r.t 
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
        function [D,xl,yl]=stat2Matrix(ss)
            D=reshape(ss,numel(ss)/3,3);
            yl='markerLabels';
            xl={'x','y','z'};
        end
    end
    methods(Static,Hidden)
        function [m,s,l]=getRefData()
            load ./data/refDataPosition.mat
        end
        function ss=distMatrix2stat(D)
           %D can be MxM or MxMxN
           M=size(D,1);
           N=size(D,3);
           ss=reshape(D,M^2,N);
           ind=triu(true(M),1);
           ss=ss(ind(:),:); %Keeping only upper half: PxN, with P=M*(M-1)/2
        end
    end
end

