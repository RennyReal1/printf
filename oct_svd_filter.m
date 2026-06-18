%% OCT SVD Static Tissue Removal
% Removes static tissue background from a series of OCT B-scans using SVD.
% The first few singular components capture the dominant static tissue signal.
% What remains after removal is the dynamic signal (blood flow / OCTA contrast).
%
% Input:  Folder of 500 .tif OCT images (same cross-section, repeated frames)
% Output: OCTA contrast map (variance of dynamic signal across frames)

%% --- PARAMETERS (edit these) ---
img_folder  = 'path/to/your/tif/folder';   % folder containing your .tif files
k_remove    = 3;     % number of SVD components to remove (static tissue)
                     % start with 1-3; increase if tissue signal remains
output_file = 'octa_result.tif';            % where to save the OCTA map

% -----------------------------------------------------------------------

%% 1. Load images into data matrix M [pixels x frames]
files = dir(fullfile(img_folder, '*.tif'));
N = numel(files);
fprintf('Found %d images.\n', N);

% Read first image to get dimensions
first = double(imread(fullfile(img_folder, files(1).name)));
[rows, cols] = size(first);
P = rows * cols;

fprintf('Image size: %d x %d  |  Building %d x %d data matrix...\n', rows, cols, P, N);
M = zeros(P, N, 'single');   % single precision saves memory

for i = 1:N
    img = single(imread(fullfile(img_folder, files(i).name)));
    M(:, i) = img(:);
end
fprintf('Data matrix loaded.\n');

%% 2. Inspect singular value spectrum to choose k_remove
% Truncated SVD — much faster than full SVD for large matrices
k_inspect = max(k_remove + 5, 10);
[U, S, V] = svds(double(M), k_inspect);
sv = diag(S);

figure('Name', 'Singular Value Spectrum');
plot(sv, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'b');
xlabel('Component index');
ylabel('Singular value');
title('Singular values — first components represent static tissue');
xline(k_remove, '--r', sprintf('k = %d (removing)', k_remove), 'LabelVerticalAlignment', 'bottom');
grid on;

fprintf('\nSingular values (first %d):\n', k_inspect);
for i = 1:k_inspect
    fprintf('  s(%2d) = %.4f  (%.1f%% of total)\n', i, sv(i), 100*sv(i)/sum(sv));
end
fprintf('\nAdjust k_remove at the top of the script if needed.\n');

%% 3. Remove static components and compute dynamic signal
% Static background = low-rank reconstruction from first k components
M_static  = U(:, 1:k_remove) * S(1:k_remove, 1:k_remove) * V(:, 1:k_remove)';
M_dynamic = double(M) - M_static;

%% 4. Compute OCTA contrast map
% Variance across frames: high variance = moving scatterers (blood flow)
octa_map = reshape(var(M_dynamic, 0, 2), rows, cols);

% Mean structural image for comparison
structural = reshape(mean(double(M), 2), rows, cols);

%% 5. Visualize results
figure('Name', 'OCT SVD Result', 'Position', [100 100 1100 420]);

subplot(1, 3, 1);
imagesc(structural); colormap(gca, gray); axis image off;
title('Structural (mean)'); colorbar;

subplot(1, 3, 2);
imagesc(octa_map); colormap(gca, hot); axis image off;
title(sprintf('OCTA map (k\\_remove = %d)', k_remove)); colorbar;

subplot(1, 3, 3);
% Log scale often reveals finer vessels
imagesc(log1p(octa_map)); colormap(gca, hot); axis image off;
title('OCTA map (log scale)'); colorbar;

sgtitle('SVD-based Static Tissue Removal');

%% 6. Save OCTA map as .tif
octa_norm = uint16(octa_map / max(octa_map(:)) * 65535);
imwrite(octa_norm, output_file);
fprintf('\nOCTA map saved to: %s\n', output_file);
