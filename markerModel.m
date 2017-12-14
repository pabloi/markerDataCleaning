classdef markerModel
    
    properties
        markerLabels %Mx1 cell array
        trainingData %Mx3xN
        statMean
        statStd
        trainingLogLPrctiles=[]; %Stores prctiles for log-l distribution in training data
        statPrctiles
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
                m=nanmedian(this.trainingData,3);
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
        
        function outlierMarkers = outlierDetectFast(this,data)
            i=indicatrix(this); %MxP
            %outStats=summaryStats(this,data)<this.statPrctiles(:,2) | s>this.statPrctiles(:,100); %Finding stats in the 1% tails (both) 
            outStats=this.loglikelihood(data) < this.trainingLogLPrctiles(:,2); %Finding likelihood in 1st percentile
            outlierMarkers=i'\outStats >.1; %This is the correct form, though i*outstats returns comparable results and is sparse
            
        end
        
        function outlierMarkers = outlierDetect(this,data)
            i=indicatrix(this); %MxP
            %outStats=summaryStats(this,data)<this.statPrctiles(:,2) | s>this.statPrctiles(:,100); %Finding stats in the 1% tails (both) 
            outStats=this.loglikelihood(data) < this.trainingLogLPrctiles(:,2); %Finding likelihood in 1st percentile
            %Alt: (optimal in some sense, but slow)
            outlierStatCountPerMarker=i * outStats; %Counting outlier stats per marker
            aux=any(outlierStatCountPerMarker>1);
            outlierMarkers=zeros(size(outlierStatCountPerMarker));
            for j=1:size(outlierStatCountPerMarker,2)
                if aux(j) %Not optimizing if at least one marker doesn't have 2 outlying stats
                    outlierMarkers(:,j)=markerModel.untanglePairedStats(outlierStatCountPerMarker(:,j)); %This is better than thresholding to some arbitrary value
                end
            end

        end
        
        function outlierMarkers = outlierDetectThreshold(this,data)
            ll5=mm.scoreMarkers(dd);
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
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerScores= i' \ L; %Least-squares sense, doesn't really work
        end
        
        function markerScores = scoreMarkers(this,data)
            %This works so-so. It does resolve markers that appear bad only
            %because another was bad, but is too optimistic about the
            %likelihood of those non-bad markers.
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerMeans= scoreMarkersNaive(this,data);
           markerMeans(isnan(markerMeans))=0;
           [M,P]=size(i);
           L2=this.scoreMarkersRanked(data,1);
           %Solve: min 1'*y s.t. y=((p-m)<=0) and L=>i*p
           uno=ones(M,1);
           cero=zeros(M,1);
           lb=[-Inf*uno; cero];
           ub=[Inf*uno;uno];
           K=max(abs(L(:)));
           f=[-uno/(M*K);uno];
           markerScores=nan(M,size(L,2));
           opts=optimoptions('intlinprog','Display','off');
           %aux=prctile(this.scoreMarkersNaive(this.trainingData),[5,50],2); %Using 2% worst training data to run
           %th=aux(:,2)-2*(aux(:,2)-aux(:,1));
           th=-(3.4)^2/2;
           aux=sum(L2<th);
           disp(['Running for ' num2str(sum(aux>0)) '/' num2str(length(aux)) ' frames.'])
           for j=1:size(L,2)
               m=th*uno;
               if aux(j)>1 %To reduce complexity, only run if naive scoring 
                   %shows at least two outliers
                   ll=L(:,j);
                   p0=i*ll./sum(i,2);
                   y0=(p0-m)<=0;
                   %Ineq 1: L >= i'*p;
                   A1=[i' zeros(P,M)]; b1=ll;
                   %Ineq 2: y>(mm-p)/K -> -p/K-y < -mm/K
                   A2=[-eye(M)/K -eye(M)]; b2=-m/K;
                   %Ineq 3: 1-y >= -(mm-p)/K -> 1+mm/K>= -y+p/K
                   A3=[eye(M)/K -eye(M)]; b3=uno+m/K;
                   py=intlinprog(f,[M+1:2*M],[A1;A2;A3],[b1;b2;b3],[],[],lb,ub,[p0;y0],opts);
                   markerScores(:,j)=py(1:M);
               else
                   markerScores(:,j)=L2(:,j);
               end
           end
        end
        
        function markerScores = scoreMarkersMedian(this,data)
           nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerScores=nan(size(i,1),size(L,2));
           for j=1:size(i,1)
               markerScores(j,:)=median(L(i(j,:)==1,:));
           end
           markerScores(squeeze(any(nn,2)))=NaN;
        end
        
        function markerScores = scoreMarkersRanked(this,data,N)
            if nargin<3
                N=3; %Using third-worse score
            end
           nn=isnan(data);
           L=loglikelihood(this,data); %PxN
           L(isnan(L))=0; %Is this the value to use? Should we use the mean from training instead?
           i=indicatrix(this); %MxP
           markerScores=nan(size(i,1),size(L,2));
           for j=1:size(i,1)
               aux=sort(L(i(j,:)==1,:),1);
               markerScores(j,:)=aux(N,:);
           end
           markerScores(squeeze(any(nn,2)))=NaN;
        end
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
        
        function outlierMarkers=untanglePairedStats(outlierStatCountPerMarker)
            %TODO: same thing for arbitrary indicatrix matrices (not just
            %paired ones)
            
            %Idea: given some candidate outliers scored with integers based on how many
            %outlier stats each marker participates, find a classification (binary) for
            %the minimum number of outlier markers that explains the outlier stats.
            %Only works if stats are always computed from 2 and only 2 markers.
            %Let p be the outlier scores of each markers' stats, then the problem is a
            %mixed-integer programming of the form:
            %min 1'*x [minimum number of outlier markers possible]
            %subject to 
            %y+z = p - 1'*x; y>=0; z<=0; [y quantifies how many more outlier stats
            %than there are outlier markers each marker has; z how many less, one of
            %the two is always 0]
            %y>=x-1 & y-max(p)*x<=1 [these two enforce x=(y>=0)]%We allow for
            %up to 1 outlier stat w/o it being an outlier marker
            %(y-1)-max(p)*x <=0 [this enforces x=1 if y>1, but allows any x for y<=1]
            %z+1<=x & z>=max(p)*(x-1) [these two enforce x=(z==0)]

            beq=outlierStatCountPerMarker;
            M=numel(beq);
            uno=ones(M,1);
            cero=zeros(M,1);
            %Strict equality: z+y = b-1'*x -> b = 1'*x+y+z;
            Aeq=[ones(M) eye(M) eye(M)]; %M x 3M
            %Inequality: y>=x -> x-y <= 0
            %A=[eye(M), -eye(M), zeros(M)];
            %b=cero;
            %Inequality: y-max(b)*x <=1
            %A=[A; -max(beq)*eye(M), eye(M), zeros(M)];
            %b=[b;uno];
            %Ineq: y-1-max(p)*x<=0 -> y-max(p)*x <=1
            A=[-max(beq)*eye(M), eye(M), zeros(M)];
            b=uno;
            
            %Inequality: z+1<=x -> z-x<=-1
            A=[A;-eye(M), zeros(M), eye(M)];
            b=[b;-uno];
            %Ineq: z>=max(p)*(x-1) -> max(p)*x-z <= max(p)
            A=[A;max(beq)*eye(M), zeros(M), -eye(M)];
            b=[b;max(beq)*uno];
            %Inequality: 
            %Constraints:
            lb=[cero;cero;-Inf*uno];
            ub=[uno; Inf*uno;cero];
            %Solve
            f=[uno;cero;cero]; %vector to solve is [x;y;z]
            opts=optimoptions('intlinprog','Display','off');
            xyz=intlinprog(f,1:M,A,b,Aeq,beq,lb,ub,opts); %Solving for integer (binary) x
            outlierMarkers=xyz(1:M);
            y=xyz(M+1:2*M);
            z=xyz(2*M+1:end);
        end
        
        function frameScores=marker2frameScoresNaive(markerScores)
            frameScores=nanmean(markerScores,1);
        end
        
        function [markerScores]=untangleLikelihoods(L,i) %Single frame: L is vector
           %Solve: min 1'*y s.t. y=((p-m)<=0) and L=>i*p
           %where m= markerMeans for that frame
           markerMeans= (i * L)./sum(i,2); 
           [M,P]=size(i);
           uno=ones(M,1);
           cero=zeros(M,1);
           f=[cero;uno]; %Vector to solve is [p;y] 
           %where p is the likelihood of each marker, 
           %and y is an aux variable that indicates if each exceeds their
           %mean stat likelihood ["outlier" indicator]
           lb=[-Inf*uno; cero];
           ub=[Inf*uno;uno];
           K=max(abs(L(:)));
           markerScores=nan(M,size(L,2));
           opts=optimoptions('intlinprog','Display','off');
           for j=1:size(L,2)
               %Ineq 1: L >= i'*p;
               A1=[i' zeros(P,M)]; b1=L(:,j);
               %Ineq 2: y>(mm-p)/K -> -p/K-y < -mm/K
               A2=[-eye(M)/K -eye(M)]; b2=-markerMeans(:,j)/K;
               %Ineq 3: 1-y >= -(mm-p)/K -> 1+mm/K>= -y+p/K
               A3=[eye(M)/K -eye(M)]; b3=uno+markerMeans(:,j)/K;
               py=intlinprog(f,[M+1:2*M],[A1;A2;A3],[b1;b2;b3],[],[],lb,ub,opts);
               markerScores(:,j)=py(1:M);
           end 
        end
        function [markerScores]=untangleOutliers(L,i) %Single frame: L is vector
           %Solve: min 1'*y s.t. y=((1'*y-m) <=0) and L<=i'*y
           %where m = outlier stats for that marker:
           m= (i * L); 
           [M,P]=size(i);
           uno=ones(M,1);
           cero=zeros(M,1);
           f=[uno]; %Vector to solve y 
           lb=[cero];
           ub=[uno];
           K=P*max(abs(L(:))); %P could be subbed by (max(sum(i,2)))
           markerScores=nan(M,size(L,2));
           opts=optimoptions('intlinprog','Display','off');
           for j=1:size(L,2)
               %Ineq 1: L >= i'*p; L<=i'*y -> -L>=-i'*y
               A1=[-i']; b1=-L(:,j);
               %Ineq 2: y=(p-1'*y >=0) ->y=(1'*y<=p)
               A2=[-ones(M)/K]; b2=-m(:,j)/K;
               %Ineq 3:
               A3=[ones(M)/K]; b3=uno+m(:,j)/K;
               py=intlinprog(f,[1:M],[A1;A2;A3],[b1;b2;b3],[],[],lb,ub,opts);
               markerScores(:,j)=py(1:M);
           end 
        end
        
    end
end

