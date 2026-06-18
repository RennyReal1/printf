%% OCT SVD Static Tissue Removal — Self-Contained Test
% Run this directly in MATLAB. No external files needed.
% It generates synthetic OCT data, applies SVD filtering, and plots results.
%
% To use your REAL data: set USE_REAL_DATA = true and set IMG_FOLDER below.

clear; clc; close all;

%% ===== SETTINGS =====
USE_REAL_DATA = false;              % set true to load your .tif files
IMG_FOLDER    = 'C:\path\to\tifs'; % only used if USE_REAL_DATA = true
K_REMOVE      = 2;                 % number of SVD components to remove
% =====================

%% STEP 1: Get data into matrix M [pixels x frames]

if USE_REAL_DATA
    % --- Load real .tif images ---
    files = dir(fullfile(IMG_FOLDER, '*.tif'));
    N     = numel(files);
    fprintf('Loading %d images...\n', N);

    first      = double(imread(fullfile(IMG_FOLDER, files(1).name)));
    [rows,cols] = size(first);
    M          = zeros(rows*cols, N);

    for i = 1:N
        img      = double(imread(fullfile(IMG_FOLDER, files(i).name)));
        M(:, i)  = img(:);
    end

else
    % --- Generate synthetic OCT data ---
    fprintf('Generating synthetic OCT data...\n');

    rows = 256;   % image height (depth)
    cols = 256;   % image width  (A-scan positions)
    N    = 500;   % number of frames

    % Static tissue background: a few bright horizontal bands
    tissue = zeros(rows, cols);
    tissue(50:60,   :) = 200;
    tissue(100:115, :) = 150;
    tissue(160:170, :) = 180;
    tissue(200:205, :) = 100;
    tissue = imgaussfilt(tissue, 2);   % smooth edges

    % Blood vessel locations: small bright spots that flicker across frames
    vessel_mask = false(rows, cols);
    vessel_mask(80,  80)  = true;
    vessel_mask(80,  150) = true;
    vessel_mask(130, 120) = true;
    vessel_mask(170, 200) = true;
    vessel_mask = imdilate(vessel_mask, strel('disk', 4));

    % Build the N frames
    M = zeros(rows*cols, N);
    for i = 1:N
        noise  = 5  * randn(rows, cols);          % random speckle
        flow   = vessel_mask .* (30 * randn(rows, cols)); % flickering vessels
        frame  = tissue + noise + flow;
        M(:,i) = frame(:);
    end

    fprintf('Synthetic data: %dx%d image, %d frames.\n', rows, cols, N);
end

%% STEP 2: Truncated SVD
fprintf('Running SVD (truncated, k=%d)...\n', K_REMOVE + 8);
k_inspect = K_REMOVE + 8;
[U, S, V] = svds(M, k_inspect);
sv = diag(S);

%% STEP 3: Plot singular value spectrum
figure('Name','Singular Value Spectrum','NumberTitle','off');
plot(sv, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor','b');
hold on;
xline(K_REMOVE, '--r', 'LineWidth', 1.5);
text(K_REMOVE + 0.2, sv(K_REMOVE)*0.95, sprintf('  k=%d removed', K_REMOVE), ...
     'Color','r','FontSize',10);
xlabel('Component index');
ylabel('Singular value');
title('Singular value spectrum — look for the ''elbow'' to choose k\_remove');
grid on;

fprintf('\nSingular values:\n');
for i = 1:k_inspect
    pct = 100 * sv(i) / sum(sv);
    fprintf('  s(%2d) = %8.2f  (%5.1f%%)\n', i, sv(i), pct);
end

%% STEP 4: Remove static components and get dynamic signal
M_static  = U(:,1:K_REMOVE) * S(1:K_REMOVE,1:K_REMOVE) * V(:,1:K_REMOVE)';
M_dynamic = M - M_static;

%% STEP 5: OCTA contrast = variance across frames
octa_map    = reshape(var(M_dynamic, 0, 2), rows, cols);
structural  = reshape(mean(M, 2),           rows, cols);

%% STEP 6: Display results
figure('Name','OCT SVD Result','NumberTitle','off','Position',[100 100 1200 380]);

subplot(1,3,1);
imagesc(structural); colormap(gca,'gray'); axis image off; colorbar;
title('Structural image (mean of frames)');

subplot(1,3,2);
imagesc(octa_map); colormap(gca,'hot'); axis image off; colorbar;
title(sprintf('OCTA map  (k\\_remove = %d)', K_REMOVE));

subplot(1,3,3);
imagesc(log1p(octa_map)); colormap(gca,'hot'); axis image off; colorbar;
title('OCTA map (log scale — shows faint vessels better)');

sgtitle('SVD-based Static Tissue Removal');

fprintf('\nDone. Adjust K_REMOVE at the top and re-run.\n');
