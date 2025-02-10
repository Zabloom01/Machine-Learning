% Code by Zach Bloom for detecting and quantifying skinning marks on
% sweetpotatoes.

%% Main - Step 1
% If you have the .mat file for the training data then you can skip this section.  
% To skip while still running code, run this section until the epoch/iteration/time elapsed/etc. 
% table pops up in the command window, then stop the code.  That will give you all the variables you need.

clear; clc
% Loading this image will automatically determine the size
exampleLabel = imread('C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data\image\train\roi_non_skinned_potato9_1.png');

% Create image data store and pixel label data store for training,
% validation, and test sets.
trainingImages = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data\image\train';
trainingLabels = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data\label\train';
validationImages = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data\image\validate';
validationLabels = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data\label\validate';
trainingImages = imageDatastore(trainingImages);
validationImages = imageDatastore(validationImages);

% Class and IDs and PixelLabelDatastore for storing the pixelDS.
classNames = ["skinning" "potato" "background"];
labelIDs   = [255 122 0];
trainingLabels = pixelLabelDatastore(trainingLabels, classNames, labelIDs);
validationLabels = pixelLabelDatastore(validationLabels, classNames, labelIDs);

% Create a data structure that has both an input (RGB image) and an output (semantic
% mask)
trainingData = combine(trainingImages,trainingLabels);
validationData = combine(validationImages,validationLabels);

% Use DeepLabV3+ to convert ResNet50 into a network capable of
% semantic segmentation.
image_Size = size(exampleLabel);
zachNeuralNetwork = deeplabv3plusLayers(image_Size,numel(labelIDs),"resnet50");

% Semantic segmentation networks tend to be unbalanced so this is to prevent this.
tbl = countEachLabel(trainingLabels);
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

% Rather than have a third directory you can split the validation data in the middle to make two new data sets.
testData = partition(validationData,2,2);    % This is our test set
validationData = partition(validationData,2,1);     % This is our validation set

% Set training parameters
options = trainingOptions('adam', ... % Adaptive learning rate and momentum control for optimizing gradients
    'MaxEpochs', 10, ... % Number of full passes through the entire training dataset
    'LearnRateSchedule', 'piecewise', ... % Adjusts the learning rate at specified intervals during training
    'LearnRateDropFactor', 0.9, ... % Reduces the learning rate by 10% (multiplies by 0.9)
    'LearnRateDropPeriod', 5, ... % After every 5 epochs, the learning rate is reduced by the drop factor
    'ValidationData', validationData, ... % Uses validation data to evaluate model performance during training
    'ValidationFrequency', 1, ... % Performs validation after each epoch
    'ValidationPatience', inf, ... % Disables early stopping to allow full convergence without interruption
    'MiniBatchSize', 32, ...  % Uses 32 randomly selected samples per batch during each training step
    'OutputNetwork','best-validation-loss',... % Saves the model with the lowest validation loss
    'Verbose', true); % Provides detailed output during the training process



% The trainNetwork command will train the network and the 
% output is used to evaluate the test set and any future data.
trainedNetwork = trainNetwork(trainingData, zachNeuralNetwork, options);


% Evaluate the network as trained on the newly trained
% network. This is done by performing semanticseg() on the test data set
% with our trained network. 
predictions = semanticseg(testData, trainedNetwork, 'WriteLocation', 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\results');

% To evaluate against the human-made labels in the test set using the evaluateSemanticSegmentation() command.
neuralNetworkSuccessMetrics = evaluateSemanticSegmentation(predictions,testData);

% Save the trained model
save('C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\trained_Model\zachNeuralNetwork.mat', 'trainedNetwork');
disp('Trained model saved.');


%% Evaluate the Model with Test Images - Step 2

% Define file paths - trained model and test images file path.
trainedModelPath = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\trained_Model\zachNeuralNetwork_publishable_mar18.mat';
testImagesPath = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data_With_HoldOut\test\Quality_check\train';

% Load the trained model
data = load(trainedModelPath, 'trainedNetwork'); % Loading trainedNetwork variable from Main code.
trainedNetwork = data.trainedNetwork;

% Get a list of test image files with '.jpg' and '.png' extensions
testImageFilesJPG = dir(fullfile(testImagesPath, '*.jpg'));
testImageFilesPNG = dir(fullfile(testImagesPath, '*.png'));

% Concatenate the two lists
testImageFiles = [testImageFilesJPG; testImageFilesPNG];

% Check if there are any test image files
if isempty(testImageFiles)
    error('No test image files found in the specified directory.');
end

% Loop through each test image and display it
for i = 1:numel(testImageFiles)
    % Read the test image
    imagePath = fullfile(testImagesPath, testImageFiles(i).name);
    testImage = imread(imagePath);

    % Perform semantic segmentation on the test image
    C = semanticseg(testImage, trainedNetwork);

    % Overlay the segmentation result on the original image
    segmentedImage = labeloverlay(testImage, C);

    % Display the original image and the segmented image side by side
    figure;

    % Original image
    subplot(1, 2, 1);
    imshow(testImage);
    title(['Test Image ', num2str(i)]);

    % Segmented image
    subplot(1, 2, 2);
    imshow(segmentedImage);
    title(['Segmentation Result for Test Image ', num2str(i)]);
end


%% Calculate Skinning Percentage - Step 3

% Loop through each test image and display it
for i = 1:numel(testImageFiles)
    % Read the test image
    imagePath = fullfile(testImagesPath, testImageFiles(i).name);
    testImage = imread(imagePath);

    % Perform semantic segmentation on the test image
    C = semanticseg(testImage, trainedNetwork);

    % Convert segmentation mask to binary for skinning and potato
    skinningMask = C == 'skinning'; % Adjust based on actual label
    potatoMask = C == 'potato'; % Adjust based on actual label

    % Use bwlabel to identify connected components
    labeledSkinning = bwlabel(skinningMask);
    labeledPotato = bwlabel(potatoMask);

    % Use regionprops to get area of each connected component
    statsSkinning = regionprops(labeledSkinning, 'Area');
    statsPotato = regionprops(labeledPotato, 'Area');

    % Assuming you have only one region for skinning and potato
    skinningArea = statsSkinning(1).Area;
    potatoArea = statsPotato(1).Area;

    % Calculate the ratio of skinning area to potato area as a percentage
    skinningToPotatoRatio = (skinningArea / (skinningArea + potatoArea)) * 100;

    % Overlay the segmentation result on the original image
    segmentedImage = labeloverlay(testImage, C);

    % Display the original image, segmented image, and calculated ratio
    figure;

    % Original image
    subplot(1, 3, 1);
    imshow(testImage);
    title(['Test Image ', num2str(i)], 'FontSize', 20); % Larger title font size

    % Segmented image
    subplot(1, 3, 2);
    imshow(segmentedImage);
    title(['Segmentation Result for Test Image ', num2str(i)], 'FontSize', 20); % Larger title font size

    % Display the calculated ratio as a percentage
    subplot(1, 3, 3);
    text(0.5, 0.5, ['Skinning Percentage: ', num2str(skinningToPotatoRatio), '%'], 'FontSize', 30, 'HorizontalAlignment', 'center');
    axis off;
end

%% Pull masks to verify how they look - Step 4

% Loop through each test image and display it
for i = 1:numel(testImageFiles)
    % Read the test image
    imagePath = fullfile(testImagesPath, testImageFiles(i).name);
    testImage = imread(imagePath);

    % Perform semantic segmentation on the test image
    C = semanticseg(testImage, trainedNetwork);

    % Convert segmentation mask to binary for skinning and potato
    skinningMask = C == 'skinning'; % Adjust based on actual label
    potatoMask = C == 'potato'; % Adjust based on actual label

    % Visualize the skinning and potato masks
    figure;
    subplot(1, 3, 1);
    imshow(testImage);
    title(['Test Image ', num2str(i)], 'FontSize', 24); % Larger title font size

    subplot(1, 3, 2);
    imshow(skinningMask);
    title('Skinning Mask', 'FontSize', 24); % Larger title font size

    subplot(1, 3, 3);
    imshow(potatoMask);
    title('Potato Mask', 'FontSize', 24); % Larger title font size

    % Use bwlabel to identify connected components
    labeledSkinning = bwlabel(skinningMask);
    labeledPotato = bwlabel(potatoMask);

    % Use regionprops to get area of each connected component
    statsSkinning = regionprops(labeledSkinning, 'Area');
    statsPotato = regionprops(labeledPotato, 'Area');

    % Assuming you have only one region for skinning and potato
    skinningArea = statsSkinning(1).Area;
    potatoArea = statsPotato(1).Area;

    % Calculate the ratio of skinning area to potato area as a percentage
    skinningToPotatoRatio = (skinningArea / (skinningArea + potatoArea)) * 100;

    % Overlay the segmentation result on the original image
    segmentedImage = labeloverlay(testImage, C);

    % Display the segmented image and calculated ratio
    figure;
    subplot(1, 2, 1);
    imshow(segmentedImage);
    title(['Segmentation Result for Test Image ', num2str(i)], 'FontSize', 16); % Larger title font size

    subplot(1, 2, 2);
    text(0.5, 0.5, ['Skinning Percentage: ', num2str(skinningToPotatoRatio), '%'], 'FontSize', 30, 'HorizontalAlignment', 'center');
    axis off;
end



%% Plot the % skinning vs % actual skinning - R^2 and RMSE Plots - Step 5

trainedModelPath = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\trained_Model\zachNeuralNetwork_publishable_mar18.mat';

% Specify the directories for the original images and labeled images
imageDirectory = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data_With_HoldOut\test\Quality_check\train';
labelDirectory = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data_With_HoldOut\test\Quality_check\validate';

% List all image files with '.png' and '.jpg' extensions in the image directory
imageFilesPNG = dir(fullfile(imageDirectory, '*.png'));
imageFilesJPG = dir(fullfile(imageDirectory, '*.jpg'));

% Concatenate the two lists
imageFiles = [imageFilesPNG; imageFilesJPG];

% Check if there are any image files
if isempty(imageFiles)
    error('No image files found in the specified directory.');
end

% List all labeled image files with '.png' and '.jpg' extension in the label directory
labelFilesPNG = dir(fullfile(labelDirectory, '*.png'));
labelFilesJPG = dir(fullfile(labelDirectory, '*.jpg'));

% Concatenate the two lists
labelFiles = [labelFilesPNG; labelFilesJPG];

% Check if there are any labeled image files
if isempty(labelFiles)
    error('No labeled image files found in the specified directory.');
end

% Ensure the number of images and labeled images match
if numel(imageFiles) ~= numel(labelFiles)
    error('Mismatch in the number of images and labeled images.');
end

% Initialize arrays to store predicted and actual skinning percentages
predictedSkinningPercentages = zeros(1, numel(imageFiles));
actualSkinningPercentages = zeros(1, numel(imageFiles));

% Load the trained model
load(trainedModelPath, 'trainedNetwork');

% Coefficient and intercept from the best-fit model
coefficient = 1; % If set to 1 then no effect
intercept = 0; % If set to 0 then no effect

% Loop through all images in both directories
for i = 1:numel(imageFiles)
    % Read the image from the first directory
    imagePath = fullfile(imageDirectory, imageFiles(i).name);
    image = imread(imagePath);

    % Read the corresponding labeled image from the second directory
    maskedImagePath = fullfile(labelDirectory, labelFiles(i).name);
    labelImage = imread(maskedImagePath);

    % Perform semantic segmentation on the image
    randomPrediction = semanticseg(image, trainedNetwork);

    % Convert segmentation mask to binary for predicted and actual skinning
    predictedSkinningMask = randomPrediction == 'skinning'; % Set label per class ID
    actualSkinningMask = labelImage == 255; % Set label per class ID
    potatoMask = labelImage == 122; % Set label per class ID

    % Use bwlabel to identify connected components
    predictedLabeledSkinning = bwlabel(predictedSkinningMask);
    actualLabeledSkinning = bwlabel(actualSkinningMask);

    % Use regionprops to get area of each connected component for both skinning and potato
    predictedStatsSkinning = regionprops(predictedLabeledSkinning, 'Area');
    predictedStatsPotato = regionprops(potatoMask, 'Area');

    actualStatsSkinning = regionprops(actualLabeledSkinning, 'Area');
    actualStatsPotato = regionprops(potatoMask, 'Area');

    % Assuming you have only one region for skinning and potato
    predictedSkinningArea = sum([predictedStatsSkinning.Area]);
    predictedPotatoArea = sum([predictedStatsPotato.Area]);

    actualSkinningArea = sum([actualStatsSkinning.Area]);
    actualPotatoArea = sum([actualStatsPotato.Area]);

    % Check for division by zero and handle appropriately
    if predictedSkinningArea + predictedPotatoArea == 0
        predictedSkinningPercentage = 0;
    else
        % Calculate the ratio of skinning area to total labeled area as a percentage
        predictedSkinningPercentage = (predictedSkinningArea / (predictedSkinningArea + predictedPotatoArea)) * 100;
    end

    if actualSkinningArea + actualPotatoArea == 0
        actualSkinningPercentage = 0;
    else
        % Calculate the ratio of skinning area to total labeled area as a percentage
        actualSkinningPercentage = (actualSkinningArea / (actualSkinningArea + actualPotatoArea)) * 100;
    end

    % Adjust predicted skinning percentage using the best-fit coefficient and intercept
    adjustedPredictedSkinningPercentage = coefficient * predictedSkinningPercentage + intercept;

    % Display image name if predicted skinning percentage is more than 10% off from actual
    if abs(adjustedPredictedSkinningPercentage - actualSkinningPercentage) > 10
        fprintf('Image: %s, Predicted Skinning Percentage: %.2f%%, Actual Skinning Percentage: %.2f%%, Adjusted Predicted: %.2f%%\n', ...
            imageFiles(i).name, predictedSkinningPercentage, actualSkinningPercentage, adjustedPredictedSkinningPercentage);
    end

    % Store adjusted predicted and actual skinning percentages
    predictedSkinningPercentages(i) = adjustedPredictedSkinningPercentage;
    actualSkinningPercentages(i) = actualSkinningPercentage;
end

% Fit a line of best fit for all data points with the adjusted predictions
fitLine = polyfit(actualSkinningPercentages, predictedSkinningPercentages, 1);
yFit = polyval(fitLine, actualSkinningPercentages);

% Calculate R-squared for all data points
actualMean = mean(actualSkinningPercentages);
ssTotal = sum((actualSkinningPercentages - actualMean).^2);
ssResidual = sum((predictedSkinningPercentages - yFit).^2);
rSquared = 1 - (ssResidual / ssTotal);

% Calculate RMSE (Root Mean Square Error) for all data points
rmsError = sqrt(sum((predictedSkinningPercentages - actualSkinningPercentages).^2) / numel(predictedSkinningPercentages));

% Plot the line plot with the line of best fit for all data points
figure;
plot(actualSkinningPercentages, predictedSkinningPercentages, 'o', 'Color', 'blue', 'MarkerFaceColor', 'blue', 'MarkerSize', 8, 'DisplayName', 'All Data Points');
hold on;
plot(actualSkinningPercentages, yFit, '-', 'Color', [1, 0.5, 0], 'LineWidth', 2, 'DisplayName', 'Line of Best Fit');
xlabel('Actual Skinning Percentage', 'FontSize', 18); % Set font size to 18 for xlabel
ylabel('Predicted Skinning Percentage', 'FontSize', 18); % Set font size to 18 for ylabel
title(['Actual vs. Predicted Skinning Percentage (R^2 = ', num2str(rSquared), ', RMS = ', num2str(rmsError), ')'], 'FontSize', 20); % Set font size to 20 for title
legend('All Data Points', 'Line of Best Fit', 'FontSize', 16); % Set font size to 16 for legend
grid on;
set(gca, 'FontSize', 24); % Set font size to 24 for axis tick labels
hold off;


%% Confusion matrix and other metrics for test dataset (Units: Pixels) - Step 6

% Clear workspace and command window
clear; clc;

% Load the trained model
load('C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\trained_Model\zachNeuralNetwork_publishable_mar18.mat', 'trainedNetwork');

% Define directories
testImageDirectory = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data_With_HoldOut\test\Quality_check\train';
testLabelDirectory = 'C:\Users\Zbloo\OneDrive\Desktop\MATLAB\Semantic_Segmentation_Model\data_With_HoldOut\test\Quality_check\validate';

% Define class names and label IDs
classNames = ["Skinning" "Sweetpotato" "Background"];
labelIDs = [255 122 0];

% Create image and pixel label datastores
testImages = imageDatastore(testImageDirectory);
testLabels = pixelLabelDatastore(testLabelDirectory, classNames, labelIDs);

% Initialize arrays to store the predicted and ground truth labels (in pixels)
predictedLabels = []; % Units: Pixels
groundTruthLabels = []; % Units: Pixels

% Check GPU availability
if gpuDeviceCount > 0
    gpuEnv = 'gpu'; % Set to GPU if available
else
    gpuEnv = 'cpu'; % Fallback to CPU if no GPU is available
end

% Initialize the wait bar
numImages = numel(testImages.Files); % Number of images
hWaitBar = waitbar(0, 'Processing Images...');

% Process each image individually
for i = 1:numImages
    % Read the image and the ground truth label
    img = readimage(testImages, i);
    groundTruthImage = readimage(testLabels, i);
    
    % Predict the segmentation
    predictedImage = semanticseg(img, trainedNetwork, 'ExecutionEnvironment', gpuEnv);
    
    % Convert categorical to numeric and collect pixel values
    predictedLabels = [predictedLabels; double(predictedImage(:))]; % Units: Pixels
    groundTruthLabels = [groundTruthLabels; double(groundTruthImage(:))]; % Units: Pixels
    
    % Update the wait bar
    waitbar(i / numImages, hWaitBar, sprintf('Processing Image %d of %d', i, numImages));
end

% Close the wait bar
close(hWaitBar);

% Calculate confusion matrix (in pixels)
confMatrix = confusionmat(groundTruthLabels, predictedLabels); % Units: Pixels

% Display the confusion matrix
disp('Confusion Matrix (Units: Pixels):');
disp(confMatrix);

% Calculate metrics
numClasses = length(classNames);
accuracy = zeros(numClasses, 1);
sensitivity = zeros(numClasses, 1);
specificity = zeros(numClasses, 1);
precision = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);

for i = 1:numClasses
    TP = confMatrix(i, i); % True Positives for class i (in pixels)
    FP = sum(confMatrix(:, i)) - TP; % False Positives for class i (in pixels)
    FN = sum(confMatrix(i, :)) - TP; % False Negatives for class i (in pixels)
    TN = sum(confMatrix(:)) - (TP + FP + FN); % True Negatives for class i (in pixels)
    
    % Accuracy for each class
    accuracy(i) = (TP + TN) / (TP + FP + FN + TN);
    
    % Sensitivity (Recall) for each class
    sensitivity(i) = TP / (TP + FN);
    
    % Specificity for each class
    specificity(i) = TN / (TN + FP);
    
    % Precision for each class
    precision(i) = TP / (TP + FP);
    
    % F1 Score for each class
    if (precision(i) + sensitivity(i)) > 0
        f1Score(i) = 2 * (precision(i) * sensitivity(i)) / (precision(i) + sensitivity(i));
    else
        f1Score(i) = 0;
    end
end

% Display metrics
disp('Metrics for Each Class:');
for i = 1:numClasses
    fprintf('Class: %s\n', classNames(i));
    fprintf('  Accuracy: %.4f\n', accuracy(i));
    fprintf('  Sensitivity (Recall): %.4f\n', sensitivity(i));
    fprintf('  Specificity: %.4f\n', specificity(i));
    fprintf('  Precision: %.4f\n', precision(i));
    fprintf('  F1 Score: %.4f\n', f1Score(i));
end

% Normalize the confusion matrix by the number of true instances per class
confMatrixNormalized = confMatrix ./ sum(confMatrix, 2);

% Display the normalized confusion matrix
disp('Normalized Confusion Matrix (Units: Proportion of Pixels):');
disp(confMatrixNormalized);

% Plot the confusion matrix
figure;
confusionchart(confMatrix, classNames);
title('Confusion Matrix: Test Dataset (by Pixels)');



