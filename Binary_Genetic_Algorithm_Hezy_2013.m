function [Feat_Index, BestAccuracy, AllChromosomes, AllScores] =  Binary_Genetic_Algorithm_Hezy_2013(input1, datafileName)
                        
%   ECE 470 Project Code
%   Mario Dellaviola, Trevor Hassel, Karl Hallquist
%   For use in TestScript.m
%   Originally prepared by below:


        % NOP For Loop to collapse the license comments
        for i = 1:2
% Written by BABATUNDE Oluleye H, PhD Student

% Address: eAgriculture Research Group, School of Computer and Security
% Science, Edith Cowan University, Mt Lawley, 6050, WA, Australia
% Date:  2013
% Please cite any of the article below (if you use the code), thank you

%  "BABATUNDE Oluleye, ARMSTRONG Leisa J, LENG Jinsong and DIEPEVEEN Dean (2014). 
%  Zernike Moments and Genetic Algorithm: Tutorial and APPLICATION. 
%  British Journal of Mathematics & Computer Science. 
%  4(15):2217-2236."

%%% OR
%BABATUNDE, Oluleye and ARMSTRONG, Leisa and LENG, Jinsong and DIEPEVEEN (2014).
% A Genetic Algorithm-Based Feature Selection. International Journal 
%of Electronics Communication and Computer Engineering: 5(4);889--905.

% DataSet here
%Ionosphere dataset from the UCI machine learning repository:                   
%http://archive.ics.uci.edu/ml/datasets/Ionosphere                              
%X is a 351x34 real-valued matrix of predictors. Y is a categorical response:   
%"b" for bad radar returns and "g" for good radar returns.                      

% NOTE: You can run this code directory on your PC as the dataset is
% available in MATLAB software
        end  % For Loop to collapse the license comments
%   Available: https://www.mathworks.com/matlabcentral/fileexchange/46961-binary-genetic-algorithm-feature-selection-zip


    global Data
    Data  = load(datafileName); % This is available in Mathworks
    
        Data.X = Data.newvotes(:,2:17);
        Data.Y = Data.newvotes(:,1);

    GenomeLength = size(Data.X,2); % This is the number of features in the dataset
%    tournamentSize = 2;
    options = gaoptimset('CreationFcn', {@PopFunction},...
                         'PopulationSize',50,...
                         'Generations',100,...
                         'PopulationType', 'bitstring',... 
                         'SelectionFcn',{@selectionstochunif},...
                         'MutationFcn',{@mutationuniform, 0.1},...
                         'CrossoverFcn', {@crossoverscattered},...
                         'EliteCount',2,...
                         'StallGenLimit',40,...
                         'PlotFcns',{@gaplotbestf, @gaplotscores},...
                         'Display', 'iter'); 
    nVars = GenomeLength;  

    FitnessFcnKNN = @FitFunc_KNN; 
    FitnessFcnSVM = @FitFunc_SVM;
    FitnessFcnIDT = @FitFunc_IDT;

    switch(input1)
        case 'KNN'
            [chromosomeKNN,AccuracyKNN,~,~,morechromosomesKNN,scoresKNN] = ga(FitnessFcnKNN,nVars,options);
            Best_chromosomeKNN = chromosomeKNN; % Best Chromosome
            Feat_Index = find(Best_chromosomeKNN==1); % Index of Chromosome
            AllChromosomes = morechromosomesKNN;
            AllScores = scoresKNN;
            BestAccuracy = AccuracyKNN;


        case 'SVM'
            if (numel(Data.X) > 20000)
            else
            [chromosomeSVM,AccuracySVM,~,~,morechromosomesSVM,scoresSVM] = ga(FitnessFcnSVM,nVars,options);
            Best_chromosomeSVM = chromosomeSVM; % Best Chromosome
            Feat_Index = find(Best_chromosomeSVM==1); % Index of Chromosome
            AllChromosomes = morechromosomesSVM;
            AllScores = scoresSVM;
            BestAccuracy = AccuracySVM;
            end


        case 'IDT'
            [chromosomeIDT,AccuracyIDT,~,~,morechromosomesIDT,scoresIDT] = ga(FitnessFcnIDT,nVars,options);
            Best_chromosomeIDT = chromosomeIDT; % Best Chromosome
            Feat_Index = find(Best_chromosomeIDT==1); % Index of Chromosome
            AllChromosomes = morechromosomesIDT;
            AllScores = scoresIDT;
            BestAccuracy = AccuracyIDT;


    end

end

%%% POPULATION FUNCTION
function [pop] = PopFunction(GenomeLength,~,options)
    RD = rand; 
    pop = [ones(options.PopulationSize,1),(rand(options.PopulationSize, GenomeLength-1)> RD)]; % Initial Population
end

%%% FITNESS FUNCTION   KNN Method
function [FitValKNN] = FitFunc_KNN(pop)
    global Data
    rand('state',69);
    X1 = Data.X;% Features Set
    Y1 = grp2idx(Data.Y);% Class Information
    instances = size(X1,1);
    r = randperm(instances);
    indTr = r(round(1:instances*0.75));
    indTe = r(round(instances*0.75)+1:end);
    FeatIndexKNN = find(pop==1); %Feature Index
    if ~any(pop(:))
        FeatIndexKNN(1) = 1;        %Check that at least one feature is selected
    end 
    Xtrain = X1(indTr,[FeatIndexKNN]);
    Xtest = X1(indTe,[FeatIndexKNN]);
    Ytrain = Y1(indTr);
    Ytest = Y1(indTe);
    ComputeKNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',3); 
    inFitValKNN = resubLoss(ComputeKNN);
    [cvKNN] = predict(ComputeKNN,Xtest);
    matchKNN = find(cvKNN ~= Ytest);
    FitValKNN = size(matchKNN,1)/numel(indTe);  
end

%%% FITNESS FUNCTION SVM Method
function [FitValSVM] = FitFunc_SVM(pop)
    global Data
    rand('state',69);
    instances = size(Data.X,1);
    r = randperm(instances);
    indTr = r(round(1:instances*0.75));
    indTe = r(round(instances*0.75)+1:end);
    FeatIndexSVM = find(pop==1); %Feature Index
    if ~any(pop(:))
        FeatIndexSVM(1) = 1;        %Check that at least one feature is selected
    end 
    X1 = Data.X;% Features Set
    Y1 = grp2idx(Data.Y);% Class Information
    Xtrain = X1(indTr,[FeatIndexSVM]);
    Xtest = X1(indTe,[FeatIndexSVM]);
    Ytrain = Y1(indTr);
    Ytest = Y1(indTe);
    ComputeSVM = fitcecoc(Xtrain',Ytrain','ObservationsIn','columns','Coding','onevsall'); 
    inFitValSVM = resubLoss(ComputeSVM);
    [cvSVM] = predict(ComputeSVM,Xtest);
    matchSVM = find(cvSVM ~= Ytest);
    FitValSVM = size(matchSVM,1)/numel(indTe);  
end

%%% FITNESS FUNCTION IDT Method
function [FitValIDT] = FitFunc_IDT(pop)
    global Data
    rand('state',69);
    instances = size(Data.X,1);
    r = randperm(instances);
    indTr = r(round(1:instances*0.75));
    indTe = r(round(instances*0.75)+1:end);
    FeatIndexIDT = find(pop==1); %Feature Index
    if ~any(pop(:))
        FeatIndexIDT(1) = 1;        %Check that at least one feature is selected
    end 
    X1 = Data.X;% Features Set
    Y1 = grp2idx(Data.Y);% Class Information
    Xtrain = X1(indTr,[FeatIndexIDT]);
    Xtest = X1(indTe,[FeatIndexIDT]);
    Ytrain = Y1(indTr);
    Ytest = Y1(indTe);
    ComputeIDT = fitctree(Xtrain,Ytrain);
    inFitValIDT = resubLoss(ComputeIDT);
    [cvIDT] = predict(ComputeIDT,Xtest);
    matchIDT = find(cvIDT ~= Ytest);
    FitValIDT = size(matchIDT,1)/numel(indTe);  
end


