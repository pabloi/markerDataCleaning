classdef naiveDistances < markerModel
    %summary stats: pair-wise component differences
    %model: independent normals
    
    methods
        function this = naiveDistances(trainData,labs)    
            this=this@markerModel(trainData,labs);
        end
        function logL = loglikelihood(this,data)
            ss=this.summaryStats(data);
            %sigma=this.statStd;
            sigma=this.getRobustStd(.94);
            %mu=this.statMean;
            mu=this.statMedian;
            logL=markerModel.normalLogL(ss,mu,sigma);
        end
        function i = indicatrix(this) %MxP
            M=this.Nmarkers;
            ind=triu(true(M),1);
            i=nan(M,M*(M-1)/2);
            for j=1:M
                aux=zeros(M);
                aux(:,j)=1;
                aux(j,:)=1;
                i(j,:)=aux(ind(:));
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
                for j=1:size(i,2)
                    if sigma(j)<10.5
                    aux=find(i(:,j));
                    n=mean(m(aux,:));
                    for k=1:length(aux)
                       plot3([n(1) m(aux(k),1)],[n(2) m(aux(k),2)],[n(3) m(aux(k),3)],'k')
                    end
                    text(n(1),n(2),n(3),['mu=' num2str(mu(j),3) ', sigma=' num2str(sigma(j),2)])
                    end
                end
        end
        
    end
    methods(Static)
        function [ss,g] = summaryStats(data)
           D=computeDistanceMatrix(data);
           ss=naiveDistances.distMatrix2stat(D);
           if nargout>2
              g=[]; %TODO 
           end
        end
        function this = learn(data,labels,noDisp)
            if nargin<2
                labels={};
            end
            this=naiveDistances(data,labels);
            if nargin<3 || isempty(noDisp) || ~noDisp
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
            knownDistances=stat2DistMatrix(ss);
            [dataFrame] = getPositionFromDistances_v2(anchorFrame,knownDistances,anchorWeights);
        end
        function D=stat2DistMatrix(ss)
            %ss is M(M-1)/2 x N
            M=ceil(sqrt(2*size(ss,1)));
            N=size(ss,2);
            ind=triu(true(M),1);
            D=zeros(M^2,N);
            D(ind,:)=ss;
            D=reshape(D,M,M,N);
            D=D+permute(D,[2,1,3]);
        end
        function ss=distMatrix2stat(D)
           %D can be MxM or MxMxN
           M=size(D,1);
           N=size(D,3);
           ss=reshape(D,M^2,N);
           ind=triu(true(M),1);
           ss=ss(ind(:),:); %Keeping only upper half: PxN, with P=M*(M-1)/2
        end
        function [D,xl,yl]=stat2Matrix(ss)
            D=triu(naiveDistances.stat2DistMatrix(ss));
            xl='markerLabels';
           yl='markerLabels';
        end
    end
    methods(Static,Hidden)
        function [m,s,l]=getRefData()
            load ./data/refData.mat
            m=naiveDistances.stat2DistMatrix(m);
            s=naiveDistances.stat2DistMatrix(s);
        end
    end
end

