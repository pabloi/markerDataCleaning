classdef markerModel
    
    properties
        markerLabels %Mx1 cell array
        trainingData %Mx3xN
        statMean
        statStd
        trainingLogLPrctiles=[]; %Stores prctiles for log-l distribution in training data
        statPrctiles
        activeStats
    end
    properties(Dependent)
        Nmarkers
        statMedian
    end
    
    methods
        function mm=markerModel(trainData,labels)
            mm.markerLabels=labels;
            mm.trainingData=trainData;
            trainStats=mm.summaryStats(trainData);
            mm.statMean=nanmean(trainStats,2);
            mm.statStd=nanstd(trainStats,[],2);
            mm.statPrctiles=prctile(trainStats,0:100,2);
            mm.trainingLogLPrctiles=prctile(mm.loglikelihood(trainData),0:100,2);
            mm.activeStats=true(size(mm.statMean));
        end
        
        function M = get.Nmarkers(this)
            M=size(this.trainingData,1);
        end
        function mu = get.statMedian(this)
            mu=this.statPrctiles(:,51);
        end
        
        logL = loglikelihood(this,data) %Returns PxN
        i = indicatrix(this)
        
        function s = getRobustStd(this,CI)
           %Uses the central confidence interval CI to estimate the std of the distribution
           %assuming the central part is normally distributed 
           %CI has to be a number in [0.02,.98]
           if nargin<2
               CI=.95;
           elseif CI>.98 || CI<0.02
               error('CI has to be a number in [0.02,.98]')
           end
           lb=50-100*CI/2; %Here I am assuming central CI
           ub=100*CI/2 +50;
           %Get the lb percentile:
           w=1-(lb-floor(lb));
           pl=this.statPrctiles(:,floor(lb)+[0,1]+1)*[w;1-w];
           %Get the ub percentile:
           w=1-(ub-floor(ub));
           pu=this.statPrctiles(:,floor(ub)+[0,1]+1)*[w;1-w];
           
           %Get how many sigmas correspond to each of the chosen
           %percentiles in a normal distribution:
           Nsigma=2*erfinv(CI)*sqrt(2); %For the classical 0.95 value, this is 2*1.96 sigma, asumming central CI again
           s=(pu-pl)/Nsigma;
        end
        
        function [fh]=seeModel(this)
            data=this.trainingData;
                fh=figure; 
                c=colormap;
                c(1,:)=[1 1 1];
                colormap(c)
                subplot(3,2,1)
                plot(this.statPrctiles',0:100)
                title('Training stat cdfs')
                subplot(3,2,[2,4,6])
                idx=all(all(~isnan(this.trainingData)));
                d=this.trainingData(:,:,idx);
                m=median(d,3); %This needs to be rotated accordingly on a frame-by-frame basis for the plot to work well
                plot3(m(:,1),m(:,2),m(:,3),'o','LineWidth',2,'MarkerSize',4)
                view(3)
                axis equal
                hold on
                text(m(:,1),m(:,2),m(:,3),this.markerLabels)

                
                subplot(3,4,5)
                s=this.stat2Matrix(this.statStd);
                [s,xl,yl]=this.stat2Matrix(this.getRobustStd(.95));
                if strcmp(xl,'markerLabels')
                    xl=this.markerLabels;
                end
                if strcmp(yl,'markerLabels')
                    yl=this.markerLabels;
                end
                s(s==0)=-5;
                imagesc(s)
                colorbar
                caxis([-5 100])
                title('\sigma training (mm)')
                axis equal
                axis tight
                set(gca,'XTickLabels',xl,'XTickLabelRotation',90,'XTick',1:size(data,1),'YTickLabels',yl,'YTick',1:size(data,1))
                subplot(3,4,6)
                m=this.stat2Matrix(this.statMedian);
                m(m==0)=NaN;
                imagesc(m)
                colorbar
                caxis([nanmin(m(:))-100 nanmax(m(:))])
                title('\mu training (mm)')
                axis equal
                axis tight
                set(gca,'XTickLabels',xl,'XTickLabelRotation',90,'XTick',1:size(data,1),'YTickLabels',yl,'YTick',1:size(data,1))
                
%                 %Reference data
%                 [m,s,l]=naiveDistances.getRefData();
%                 subplot(3,4,9)
%                 s=triu(s);
%                 s(s==0)=-5;
%                 m=triu(m);
%                 m(m==0)=-25;
%                 imagesc(s)
%                 colorbar
%                 %colormap(c)
%                 caxis([-5 100])
%                 title('\sigma reference (mm)')
%                 axis equal
%                 axis tight
%                 set(gca,'XTickLabels',l,'XTickLabelRotation',90,'XTick',1:size(s,1),'YTickLabels',l,'YTick',1:size(s,1))
%                 subplot(3,4,10)
%                 imagesc(m)
%                 caxis([-25 1000])
%                 colorbar
%                 title('\mu reference (mm)')
%                 axis equal
%                 axis tight
%                 set(gca,'XTickLabels',l,'XTickLabelRotation',90,'XTick',1:size(s,1),'YTickLabels',l,'YTick',1:size(s,1))
        end
        
        function [outlierMarkers,ll] = outlierDetectFast(this,data)
            i=indicatrix(this); %MxP
            %outStats=summaryStats(this,data)<this.statPrctiles(:,2) | s>this.statPrctiles(:,100); %Finding stats in the 1% tails (both) 
            ll=this.loglikelihood(data);
            outStats=ll < -(5^2)/2; %Finding likelihood in 1st percentile
            outlierMarkers=i'\outStats >.1; %This is the correct form, though i*outstats returns comparable results and is sparse
            
        end
        
        function [outlierMarkers,logL] = outlierDetect(this,data,threshold)
            if nargin<3
                threshold=-5;
            end
            %One option:
            %logL=this.scoreMarkers(data);
            %outlierMarkers=logL<-(threshold)^2/2;
            %A more efficient one: [for some reason, this is less efficient]
            logL=this.loglikelihood(data);
            outStats=logL<(-threshold^2/2);
            [outlierMarkers]=markerModel.untangleOutliers(outStats,this.indicatrix);         
        end
        
        function markerScores = scoreMarkersNaive(this,data)
           nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           i=indicatrix(this);
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           markerScores= (i * L)./sum(i,2);
           markerScores(squeeze(any(nn,2)))=NaN;
        end
        
        function markerScores = scoreMarkersFast(this,data)
           markerScores = scoreMarkersRanked(this,data,2);
        end
        
        function markerScores = scoreMarkers(this,data)
            %This works so-so. It does resolve markers that appear bad only
            %because another was bad, but is too optimistic about the
            %likelihood of those non-bad markers.
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           in=indicatrix(this); %MxP
           refScore=this.scoreMarkersRanked(data,2);
           th=-4;
           markerScores=markerModel.untangleLikelihoods(L,in,refScore,th);
        end
        
        function markerScores = scoreMarkersMedian(this,data)
           i=indicatrix(this); %MxP
           L=this.loglikelihood(data);
           markerScores=nan(size(i,1),size(L,2));
           for j=1:size(i,1)
               markerScores(j,:)=median(L(i(j,:)==1,:));
           end
        end
        
        function markerScores = scoreMarkersRanked(this,data,N)
            if nargin<3
                N=2; %Using third-worse score
            end
           L=loglikelihood(this,data); %PxN
           [P,Nn]=size(L);
           i=indicatrix(this); %MxP
           if N>1
               markerScores=nan(size(i,1),size(L,2));
               for j=1:size(i,1)
                   aux=sort(L(i(j,:)==1,:),1);
                   markerScores(j,:)=aux(N,:);
               end
            else
                markerScores=squeeze(min(i.*reshape(L,1,P,Nn),[],2));
           end
        end
        
        mapData = reconstruct(this,data,priors)
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
        
        [M,xl,yl]=stat2Matrix(stats)
        
        function lL=normalLogL(values,means,stds)
            %values is PxN, means and stds are Px1
            %Returns PxN likelihood of value under each normal
            d=.5*((values-means)./stds).^2;
            lL=-d;% -log(stds) -.9189;
            %lL=-sqrt(2*d); %Mahalanobis distance
            %.9189 = log(2*pi)/2
        end
        
        function frameScores=marker2frameScoresNaive(markerScores)
            frameScores=nanmean(markerScores,1);
        end
        
        function [markerScores]=untangleLikelihoods(L,in,refScore,th)
           %Solve: min 1'*y s.t. y=((p-m)<=0) and L=>i*p
           [M,P]=size(in);
           uno=ones(M,1);
           cero=zeros(M,1);
           lb=[-Inf*uno; cero];
           ub=[Inf*uno;uno];
           K=max(abs(L(:)));
           f=[-uno/(M*K);uno];
           opts=optimoptions('intlinprog','Display','off');
           th=-(th)^2/2;
           aux=sum(refScore<th);
           disp(['Running for ' num2str(sum(aux>0)) '/' num2str(length(aux)) ' frames.'])
           warning('off')
           markerScores=refScore;
           for j=1:size(L,2)
               m=th*uno;
               if aux(j)>1 %To reduce complexity, only run if ranked scoring shows 2 or more bad markers
                   %shows at least two outliers
                   %idx=refScore(:,j)<th;
                   ll=L(:,j);
                   p0=in*ll./sum(in,2);
                   y0=(p0-m)<=0;
                   %Ineq 1: L >= i'*p;
                   A1=[in' zeros(P,M)]; b1=ll;
                   %Ineq 2: y>(mm-p)/K -> -p/K-y < -mm/K
                   A2=[-eye(M)/K -eye(M)]; b2=-m/K;
                   %Ineq 3: 1-y >= -(mm-p)/K -> 1+mm/K>= -y+p/K
                   A3=[eye(M)/K -eye(M)]; b3=uno+m/K;
                   %Ineq 4: i'*y >= (ll<th) [each outlying stat has at least one outlying marker involved in it]
                   A4=[zeros(P,M) -in'];     b4=-(ll<th);
                   %Eq:
                   Aeq=[];  beq=[];
                   py=intlinprog(f,[M+1:2*M],[A1;A2;A3;A4],[b1;b2;b3;b4],Aeq,beq,lb,ub,[p0;y0],opts);
                   if ~isempty(py)
                    markerScores(:,j)=py(1:M);
                   end
               end
           end
           warning('on')
        end
        
        function [outMarkers]=untangleOutliers(outStats,in) %Single frame: L is vector
            %Solve: min 1'*y s.t. L<=i'*y and y>=(i*L -1'*y), where L = outlier stats for marker
            %This means we find the minimum outlier set, such that each
            %outlier stat has at least one causing outlier marker AND any
            %marker that has strictly more outlier stats than there are outlier
            %markers is an outlier itself [I think this is implied by the other but am not sure].
            [M,P]=size(in);
            uno=ones(M,1);
            cero=zeros(M,1);
            f=[uno]; %Vector to solve y 
            lb=[cero];
            ub=[uno];
            opts=optimoptions('intlinprog','Display','off');
            outMarkers=(in*outStats)>1; %This marks markers only if they have more than one outlier stat
            for j=1:size(outStats,2)
                b1=outStats(:,j);
                b2=in*outStats(:,j);
                c=outMarkers(:,j);
                if any(b1>in'*c) || any(c<(b2-sum(c))) || sum(c)>1 %Only run if the inequalities are not already satisified or if two markers were outliers in the same frame
                    %Ineq 1: L<=i'*y -> -L>= -i'*y
                    A1=-in';
                    b1=-b1;
                    %Ineq 2: y>=(i*L -1'*y)/K -> -y-(1'*y)/K<=-i*L/K
                    K=max(sum(in,2));
                    A2=[-eye(M)-ones(M)/K];
                    b2=-b2/K;
                    outMarkers(:,j)=intlinprog(f,[1:M],[A1;A2],[b1;b2],[],[],lb,ub,opts);
                end
            end
        end
        
    end
end

