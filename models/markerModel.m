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
        [logL,g] = loglikelihood(this,data) %Returns PxN likelihood and gradient
        i = indicatrix(this)
        mapData = reconstruct(this,dataPrior,priorConfidence)
        [badFlag,outlierClass1,outlierClass2] = validateMarkerModel(this,verbose)
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
        function [outlierMarkers,markerScores] = outlierDetectFast(this,data,threshold)
            if nargin<3
                threshold=-5;
            end
            %Works well if there is a single outlier per frame, or if two
            %or more outliers are present but they don't have tight
            %distances to any other third marker. Otherwise, problematic.
            markerScores = scoreMarkersFast(this,data);
            outlierMarkers=markerScores < -(threshold^2)/2;     %Roughly asking if marker was more than X std away from expectation.     
        end
        function [outlierMarkers,markerScores,logL] = outlierDetect(this,data,threshold,fastFlag)
            if nargin<3 || isempty(threshold)
                threshold=-5;
            end
            if nargin<4 || isempty(fastFlag)
                fastFlag=false;
            end
            markerScores=[];
            logL=[];
            %One option: find outlier stats and disentangle those
            %logL=this.loglikelihood(data);
            %outStats=logL<(-threshold^2/2);
            %[outlierMarkers]=markerModel.untangleOutliers(outStats,this.indicatrix);     
            %Another: disentangle likelihoods themselves & then detect
            markerScores = scoreMarkers(this,data,fastFlag);
            outlierMarkers=markerScores<(-threshold^2/2);
        end
        function markerScores = scoreMarkers(this,data,fastFlag)
           if nargin<3 || isempty(fastFlag)
                fastFlag=false;
           end
           markerScores=this.scoreMarkersRanked(data,2); %Fast scoring
           %This fast scoring works well if only one outlier marker is
           %present, or if two or more non-adjacent (ie. no strong distance
           %constraints) markers are present. Otherwise problematic.
            if ~fastFlag %Improve scoring for complex situations
           i=indicatrix(this); %MxP
           markerScores=this.scoreMarkersFast(data);
           outMarkers=this.outlierDetectFast(data);
           badFrames=sum(outMarkers)>1; %More than one bad marker per frame
           L=this.loglikelihood(data(:,:,badFrames));
           badMarkers=any(outMarkers(:,badFrames),2); %This may only be useful on not too long trials (otherwise every marker will possibly be a part of at least one bad frame, and all markers will get selected)
           badMarkers=true(size(badMarkers));
           in=i(badMarkers,:); %Indicatrix of bad markers only
           activeConstraints=any(in,1);
           [markerScores(badMarkers,badFrames)]=markerModel.untangleLikelihoods(L(activeConstraints,:),in(:,activeConstraints));
            else

            end
        end
        function labels = labelData(this,dataFrames)
            error('Unimplemented')
            %Heuristic: assign random labels, then try pair-wise
            %permutations and see if they improve likelihood of data.
            %Keep permutations that do, and reset pairwise list to try
            %End when all pairwise permutations have been tried and none
            %improves reconstruction
            %Perhaps repeat a couple times and select best of final results
        end
        function sortedData=sortMarkerOrder(model,data,labels)
            % Convenience function that re-sorts the rows of data, so that
            % they match the order of the labels in this model.
            error('Unimplemented')
            %To do: compare lists of labels and model.markerLabels, find
            %sorting, and sort the data. Use compareLists()
        end
    end
    methods(Hidden)
        function markerScores = scoreMarkersRanked(this,data,N)
            if nargin<3
                N=2; %Using second-worse score
            end
           L=loglikelihood(this,data); %PxN
           [P,Nn]=size(L);
           i=indicatrix(this); %MxP
           if N>1
               markerScores=nan(size(i,1),size(L,2));
               for j=1:size(i,1) %Each marker
                   aux=sort(L(i(j,:)==1,:),1);
                   NN=min(N,size(aux,1));
                   markerScores(j,:)=aux(NN,:);
               end
            else
                markerScores=squeeze(min(i.*reshape(L,1,P,Nn),[],2));
           end
        end
        function [dataFrame,permutationList]=tryPermutations(model,dataFrame,listToPermute)
            %Tries to permute labels on single dataFrame to maximize
            %likelihood of data.
            permutationList=zeros(0,2);
            %Benchmark to compare:
            L0=model.loglikelihood(dataFrame);
            nBad=sum(L0<-5^2/2);
            %TODO: start with markers associated to bad distances, and then move to a larger set
            if nargin<3 || isempty(listToPermute)
                %Run for bad markers.
                if any(nBad)
                    listToPermute=find(nBad);
                    [dataFrame,permutationList]=tryPermutations(model,dataFrame,listToPermute);
                end
                %Run for all markers.
                [dataFrame,permutationList2]=tryPermutations(model,dataFrame,1:size(dataFrame,1));
                permutationList=[permutationList;permutationList2];
            else
                while nBad>0 %If no log-L is too bad, do nothing, save time.
                    %Permutations to consider:
                    listOfPermutations=nchoosek(listToPermute,2); %Pairwise permutations
                    N=size(listOfPermutations,1);
                    count=0;
                    while count<N
                       count=count+1;
                       newDataFrame=dataFrame;
                       newDataFrame(fliplr(listOfPermutations(count,:)),:)=dataFrame(listOfPermutations(count,:),:);
                       L=model.loglikelihood(newDataFrame);
                       if sum(L)>sum(L0) %Found permutation that improves things
                           break %Exiting inner while-loop
                       end
                    end
                    if sum(L)<=sum(L0) %Checking if while exited without making any improvements
                        return %Inner loop finished without progress
                    else %Improvement was made!
                        %Update benchmark:
                        dataFrame=newDataFrame;
                        L0=L;
                        nBad=sum(L0<-5^2/2);
                        permutationList(end+1,:)=listOfPermutations(count,:); %Adding permutation to list
                    end
                end
            end
        end
    end
    methods(Static)
        model = learn(data)
        [ss,g] = summaryStats(data) %Returns PxN summary stats, and P x 3M x N gradient 
        %gradient can be P x 3M if it is the same for all frames, as is the case in linear models       
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
        function [markerScores]=untangleLikelihoods(L,indicatrix)
            A=-indicatrix';
            options = optimoptions('fmincon','Display','off','SpecifyObjectiveGradient',false,'OptimalityTolerance',1e-1,'StepTolerance',1e-1,'ConstraintTolerance',1e-1); %Relax tolerances for fast compute
            options2 = optimoptions('linprog','Display','off');
            N=size(indicatrix,1);
            LB=zeros(N,1);
            f=ones(N,1);
            M=size(L,2); %number of frames
            markerScores=nan(N,M);
            display(['Running for ' num2str(M) ' frames. Expect ' num2str(M/75,2) ' sec. processing time (75 fps).'])
            tic
            for j=1:M
               b=-sqrt(-2*L(:,j)); %transform likelihood to std away from expectation
               %Get linear programming solution:
               [x0,~,~,~,lam]=linprog(f,A,b,[],[],LB,[],options2); %Being used to identify active inequalities only
               %f'*x0
               %Sparsify solution: (for each active constraint, selecting the most
               %unequal solution possible, notice this maintains the cost f'*x constant)
               activeConstraints=(lam.ineqlin==1);
               A1=-A(activeConstraints,:); %Active inequalities
               idx=any(A1~=0); %For each marker there is at least 1 active inequality, or one of the bounds was reached
               A2=A1(:,idx);
               aux=fmincon(@(x) [-sum(x.^2)],x0(idx),A(~activeConstraints,idx),b(~activeConstraints),A2,A2*x0(idx),LB(idx),[],[],options);  %Notice this adds as constraint that active inequalities from linprog are maintained
               x0(idx)=aux;
               markerScores(:,j)=-.5*x0.^2;
               %f'*x0 %Check that this was maintained
            end
            toc
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

