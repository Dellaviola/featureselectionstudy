%EVALUATION SCRIPT
%LOAD DATA////////////////////////////////////////////////////////////////
clear
DATASET = 'newvotes.mat';
Data = load(DATASET);

Data.X = Data.newvotes(:,2:17);
Data.Y = Data.newvotes(:,1);

%PREPROCESS////////////////////////////////////////////////////////////////
rand('state',69);
instances = size(Data.X,1);
features = size(Data.X,2);
r = randperm(instances);
indTr = r(round(1:instances*0.75));
indTe = r(round(instances*0.75)+1:end);
X1 = Data.X;% Features Set
Y1 = grp2idx(Data.Y);% Class Information
Xtrain = X1(indTr,:);
Xtest = X1(indTe,:);
Ytrain = Y1(indTr);
Ytest = Y1(indTe);

%RUN BASELINES/////////////////////////////////////////////////////////////
baseKNN = doKNN(Xtrain,Ytrain,Xtest,Ytest);
baseSVM = doSVM(Xtrain,Ytrain,Xtest,Ytest);
baseIDT = doIDT(Xtrain,Ytrain,Xtest,Ytest);

%FEATURE SELECTION USING FORWARD SEQUENTIAL FEATURES///////////////////////
% opts = statset('display','iter');
% 
% ffsinKNN = sequentialfs(@doKNN,Xtrain,Ytrain,'options',opts);
% ffsKNNacc = doKNN(Xtrain(:,ffsinKNN),Ytrain,Xtest(:,ffsinKNN),Ytest);
% ffsKNNfs = sum(ffsinKNN);
% 
% ffsinSVM = sequentialfs(@doSVM,Xtrain,Ytrain,'options',opts);
% ffsSVMacc = doSVM(Xtrain(:,ffsinSVM),Ytrain,Xtest(:,ffsinSVM),Ytest);
% ffsSVMfs = sum(ffsinSVM);
% 
% ffsinIDT = sequentialfs(@doIDT,Xtrain,Ytrain,'options',opts);
% ffsIDTacc = doIDT(Xtrain(:,ffsinIDT),Ytrain,Xtest(:,ffsinIDT),Ytest);
% ffsIDTfs = sum(ffsinIDT);
% 
% %FEATURE SELECTION USING BACKWARDS SEQUENTIAL FEATURES///////////////////////
% opts = statset('display','iter');
% 
% bfsinKNN = sequentialfs(@doKNN,Xtrain,Ytrain,'options',opts,'direction','backward');
% bfsKNNacc = doKNN(Xtrain(:,bfsinKNN),Ytrain,Xtest(:,bfsinKNN),Ytest);
% bfsKNNfs = sum(bfsinKNN);
% 
% bfsinSVM = sequentialfs(@doSVM,Xtrain,Ytrain,'options',opts,'direction','backward');
% bfsSVMacc = doSVM(Xtrain(:,bfsinSVM),Ytrain,Xtest(:,bfsinSVM),Ytest);
% bfsSVMfs = sum(bfsinSVM);
% 
% bfsinIDT = sequentialfs(@doIDT,Xtrain,Ytrain,'options',opts,'direction','backward');
% bfsIDTacc = doIDT(Xtrain(:,bfsinIDT),Ytrain,Xtest(:,bfsinIDT),Ytest);
% bfsIDTfs = sum(bfsinIDT);
% 
%FEATURE REDUCTION USING GENETIC ALGORITHM/////////////////////////////////
tic;
[bestKNN, bestScoreKNN, allKNN, allScoreKNN] = Binary_Genetic_Algorithm_Hezy_2013('KNN', DATASET);
indKNN = find(allScoreKNN(1) == allScoreKNN(:));
featureSetsKNN = sum(allKNN(indKNN,:),2);
bestindKNN =find(min(featureSetsKNN) == featureSetsKNN(:));
bestFeaturesKNN = unique(allKNN(bestindKNN,:),'rows');
knnTime = toc;
fh = findobj( 'Type', 'Figure', 'Name', 'Genetic Algorithm' );
saveas(fh, 'KNN.png')

if(numel(Data.X) <= 20000)
    tic;
    [bestSVM, bestScoreSVM, allSVM, allScoreSVM] = Binary_Genetic_Algorithm_Hezy_2013('SVM', DATASET);
    indSVM = find(allScoreSVM(1) == allScoreSVM(:));
    featureSetsSVM = sum(allSVM(indSVM,:),2);
    bestindSVM =find(min(featureSetsSVM) == featureSetsSVM(:));
    bestFeaturesSVM= unique(allSVM(bestindSVM,:),'rows');
    svmTime = toc;
    fh = findobj( 'Type', 'Figure', 'Name', 'Genetic Algorithm' );
    saveas(fh, 'SVM.png')

end

tic;
[bestIDT, bestScoreIDT, allIDT, allScoreIDT] = Binary_Genetic_Algorithm_Hezy_2013('IDT', DATASET);
indIDT = find(allScoreIDT(1) == allScoreIDT(:));
featureSetsIDT = sum(allIDT(indIDT,:),2);
bestindIDT =find(min(featureSetsIDT) == featureSetsIDT(:));
bestFeaturesIDT = unique(allIDT(bestindIDT,:),'rows');
idtTime = toc;
fh = findobj( 'Type', 'Figure', 'Name', 'Genetic Algorithm' );
saveas(fh, 'IDT.png')

fprintf('\n\nDataset: %s\n\n',DATASET)
init = zeros(features,1);
    fprintf('Genetic Algorithm Using KNN Classifier:\n')
    for i = 1:size(bestFeaturesKNN,1)
        g=sprintf('%d', bestFeaturesKNN(i,:));
        fprintf('    Best Chromosome %d: %s\n',i ,g)
    end
    fprintf('    %d of %d features obtains %0.2f%% classification accuracy using 3-KNN with 25%% holdout\n', sum(bestFeaturesKNN(1,:)),features,(1-bestScoreKNN)*100)
    fprintf('    Computation time: %0.3f seconds\n\n\n',knnTime)
    fprintf('Genetic Algorithm Using SVM Classifier:\n')
    if (numel(Data.X) > 20000)
       fprintf('    Dataset too large for time consideration\n\n\n')
    else
        for i = 1:size(bestFeaturesSVM,1)
                g=sprintf('%d', bestFeaturesSVM(i,:));
                fprintf('    Best Chromosome %d: %s\n',i, g)
        end
        fprintf('    %d of %d features obtains %0.2f%% classification accuracy using SVM with 25%% holdout\n', sum(bestFeaturesSVM(1,:)),features,(1-bestScoreSVM)*100)
        fprintf('    Computation time: %0.3f seconds\n\n\n',svmTime)
    end
     fprintf('Genetic Algorithm Using Decision Tree Classifier:\n')
    for i = 1:size(bestFeaturesIDT,1)
            g=sprintf('%d', bestFeaturesIDT(i,:));
            fprintf('    Best Chromosome %d: %s\n',i, g)
    end
        fprintf('    %d of %d features obtains %0.2f%% classification accuracy using Decision Tree with 25%% holdout\n', sum(bestFeaturesIDT(1,:)),features,(1-bestScoreIDT)*100)
        fprintf('    Computation time: %0.3f seconds\n\n\n',idtTime)
        
 figure(2);
 imshow('KNN.png','InitialMagnification', 50);
 title('\fontsize{16}GA with KNN Classification');
 
 if(numel(Data.X) <= 20000)
    figure(3);
    imshow('SVM.png','InitialMagnification', 50);
    title('\fontsize{16}GA with SVM Classification');
 end
 
 figure(4);
 imshow('IDT.png','InitialMagnification', 50);
 title('\fontsize{16}GA with Decision Tree Classification');
 
%BASELINES/////////////////////////////////////////////////////////////////
function baselineKNN = doKNN(Xtrain,Ytrain,Xtest,Ytest)
    ComputeKNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',3);
    inFitValKNN = resubLoss(ComputeKNN);
    [cvKNN] = predict(ComputeKNN,Xtest);
    matchKNN = find(cvKNN ~= Ytest);
    baselineKNN = (1 - size(matchKNN,1)/numel(Ytest))*100;
end

function baselineSVM = doSVM(Xtrain,Ytrain,Xtest,Ytest)
    ComputeSVM = fitcecoc(Xtrain',Ytrain','ObservationsIn','columns','Coding','onevsall');
    inFitValSVM = resubLoss(ComputeSVM);
    [cvSVM] = predict(ComputeSVM,Xtest);
    matchSVM = find(cvSVM ~= Ytest);
    baselineSVM = (1 - size(matchSVM,1)/numel(Ytest))*100;
end

function baselineIDT = doIDT(Xtrain,Ytrain,Xtest,Ytest)
    ComputeIDT = fitctree(Xtrain,Ytrain);
    inFitValIDT = resubLoss(ComputeIDT);
    [cvIDT] = predict(ComputeIDT,Xtest);
    matchIDT = find(cvIDT ~= Ytest);
    baselineIDT = (1 - size(matchIDT,1)/numel(Ytest))*100;
end 