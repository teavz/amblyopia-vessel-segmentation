%% Vessel Map Analyzer

% MIT License
% Copyright 2025 © Rijul S. Soans, © Susana T. L. Chung

% This script lets you choose a segmented vessel map from the "segmented-images"
% folder, computes vascular features, and appends them to a MAT‐file table.

% Execute the script by pressing the green Run Button in the Editor Tab.

clc; clear; close all;

% Determine script folder and project root
scriptDir  = fileparts(mfilename('fullpath'));  % directory containing this .m file
projectDir = fileparts(scriptDir);               % one level up

% Define project paths
imagesFolder  = fullfile(projectDir, "segmented-images");
tableFileName = fullfile(scriptDir, "vascularDensityFeatures.mat");

% Verify "segmented-images" exists
if ~isfolder(imagesFolder)
    error("Folder '%s' not found. Create a folder named 'segmented-images' in the project directory.", imagesFolder);
end

% Load or initialize the table
if isfile(tableFileName)
    load(tableFileName, 'vascularDensityFeatures');
else
    vascularDensityFeatures = table( ...
        'Size', [0 7], ...
        'VariableTypes',  {'string', 'string', 'string', 'double', 'double', 'double', 'double'}, ...
        'VariableNames',  {'Category', 'Filename', 'Eye', 'Vascular_Area', ...
        'Fractal_Dimension', 'Vascular_Skeleton_Length', ...
        'Vascular_Bifurcation_Points'} );
end

% Prompt user to select one segmented vessel map from "segmented-images"
[chosenFile, chosenPath] = uigetfile( ...
    fullfile(imagesFolder, "*.png"), ...
    "Select a segmented vessel map" );

if isequal(chosenFile, 0)
    disp("User canceled file selection. Exiting.");
    return;
end

fullFilePath = fullfile(chosenPath, chosenFile);

% Skip processing if this filename is already in the table
if any(vascularDensityFeatures.Filename == chosenFile)
    disp("Entry already exists in the table. No changes made.");
else
    % Read the segmented vessel map
    vessel_orig_img = imread(fullFilePath);
    if ndims(vessel_orig_img) == 3
        vesselMask = vessel_orig_img(:, :, 1) > 0;
    else
        vesselMask = vessel_orig_img > 0;
    end

    % Remove small spurious regions (<11 pixels)
    cleanedVesselPixels = bwareaopen(vesselMask, 11);

    % Show cleaned vessel mask
    figure;
    imshow(cleanedVesselPixels);
    title(sprintf("Cleaned Vessel Mask: %s", chosenFile), 'Interpreter','none');

    % Vascular Features
    % 1. --- Vascular Area ---
    vascularArea = nnz(cleanedVesselPixels);
    fprintf("Vascular Area: %d pixels\n", vascularArea);

    % 2. --- Fractal Dimension ---
    blockPlotOn      = true;
    blockSizeDisplay = 12;  % overlay a 12×12 block grid
    graphPlotOn = true;
    [fractalDimension, log_x, log_y] = calculateVascularFractalDimension( ...
        cleanedVesselPixels, blockSizeDisplay, blockPlotOn, graphPlotOn);
    fprintf("Fractal Dimension: %.4f\n", fractalDimension);

    % 3. --- Vascular Skeleton Length ---
    skeletonVesselImage = bwmorph(cleanedVesselPixels, "skeleton", Inf);
    spurSkelRemovImage  = bwmorph(skeletonVesselImage, "spur", 5); % Empirically chosen as 5
    vascularSkeletonLength = nnz(spurSkelRemovImage);
    fprintf("Skeleton Length: %d pixels\n", vascularSkeletonLength);

    % 4. --- Vascular Bifurcation Points ---
    [branchPoints, vbpImage] = detectVBP(spurSkelRemovImage);
    numBifurcationPoints = nnz(branchPoints);
    fprintf("No. of Bifurcation Points: %d\n", numBifurcationPoints);

    % Overlay branch points in green
    RGBSkelImage = repmat(uint8(vbpImage)*255, 1, 1, 3);
    [rowBifur, colBifur] = find(branchPoints);

    for idx = 1:numel(rowBifur)
        RGBSkelImage = insertShape(RGBSkelImage, 'FilledCircle', ...
            [colBifur(idx), rowBifur(idx), 5], 'Color', 'green', 'Opacity', 0.5);
    end

    figure;
    imshow(RGBSkelImage);
    title(sprintf("Skeleton + Bifurcations: %s", chosenFile), 'Interpreter','none');

    % Participant category selection via Dialog
    choice = questdlg( ...
        "Is the subject category 'Normal'?", ...
        "Subject Category", ...
        "Yes", "No", "Yes" );

    switch choice
        case "Yes"
            category = "Normal";
        case "No"
            prompt    = "Enter subject category:";
            dlgtitle  = "Specify Category";
            dims      = [1 40];
            definput  = "";
            answer    = inputdlg(prompt, dlgtitle, dims, definput);
            if isempty(answer) || all(strtrim(answer{1}) == "")
                category = "Unknown";
            else
                category = string(answer{1});
            end
        otherwise
            % If user closes the dialog, default to "Unknown"
            category = "Unknown";
    end

    % Eye Selection via Dialog
    choiceEye = questdlg( ...
        "Is this image of the left or right eye?", ...
        "Eye Selection", ...
        "Left", "Right", "Left" );

    switch choiceEye
        case "Left"
            eye = "OS";
        case "Right"
            eye = "OD";
        otherwise
            eye = "Unknown";
    end

    % Append new row to the table
    newEntry = { category, chosenFile, eye, vascularArea, ...
        fractalDimension, vascularSkeletonLength, ...
        numBifurcationPoints };
    vascularDensityFeatures = [vascularDensityFeatures; newEntry];
    save(tableFileName, "vascularDensityFeatures");

    disp("New entry added to vascularDensityFeatures.mat");
end

% Display updated table
disp(vascularDensityFeatures);
