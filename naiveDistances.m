classdef naiveDistances < markerModel
    %summary stats: pair-wise component differences
    %model: independent normals
    
    methods
        function this = naiveDistances(trainData,labs)    
            this=this@markerModel(trainData,labs);
        end
        function logL = loglikelihood(this,data)
            ss=this.summaryStats(data);
            logL=markerModel.normalLogL(ss,this.statMean,this.statStd);
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
        function [fh]=seeModel(this)
            labels=this.markerLabels;
            data=this.trainingData;
                fh=figure; 
                c=colormap;
                c(1,:)=[1 1 1];
                colormap(c)
                subplot(3,2,1)
                plot(this.statPrctiles',0:100)
                title('Training stat cdfs')
                subplot(3,2,2)
                title('Stat sensitivity to marker components')
                subplot(3,2,4)
                s=triu(naiveDistances.stat2DistMatrix(this.statStd));
                s(s==0)=-5;
                imagesc(s)
                colorbar
                caxis([-5 200])
                title('\sigma training (mm)')
                axis equal
                set(gca,'XTickLabels',labels,'XTickLabelRotation',90,'XTick',1:size(data,1),'YTickLabels',labels,'YTick',1:size(data,1))
                subplot(3,2,3)
                m=triu(naiveDistances.stat2DistMatrix(this.statMean));
                m(m==0)=-25;
                imagesc(m)
                colorbar
                caxis([-25 1000])
                title('\mu training (mm)')
                axis equal
                set(gca,'XTickLabels',labels,'XTickLabelRotation',90,'XTick',1:size(data,1),'YTickLabels',labels,'YTick',1:size(data,1))
                
                %Reference data
                [m,s,l]=naiveDistances.getRefData();
                subplot(3,2,6)
                s=triu(s);
                s(s==0)=-5;
                m=triu(m);
                m(m==0)=-25;
                imagesc(s)
                colorbar
                %colormap(c)
                caxis([-5 200])
                title('\sigma reference (mm)')
                axis equal
                set(gca,'XTickLabels',l,'XTickLabelRotation',90,'XTick',1:size(s,1),'YTickLabels',l,'YTick',1:size(s,1))
                subplot(3,2,5)
                imagesc(m)
                caxis([-25 1000])
                colorbar
                title('\mu reference (mm)')
                axis equal
                set(gca,'XTickLabels',l,'XTickLabelRotation',90,'XTick',1:size(s,1),'YTickLabels',l,'YTick',1:size(s,1))
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
    end
    methods(Static,Hidden)
        function [m,s,l]=getRefData()
            load ./data/refData.mat
            m=naiveDistances.stat2DistMatrix(m);
            s=naiveDistances.stat2DistMatrix(s);
        end
    end
end

