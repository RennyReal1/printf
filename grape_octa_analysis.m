%% Grape OCT Feature Extraction + Correlation with Quality Metrics
% Extracts: attenuation coefficient, signal intensity profile,
%           speckle contrast, texture features (GLCM), layer thickness,
%           penetration depth, structural comparison
% Then correlates each feature with Brix, firmness, ripeness, spoilage.

clear; clc; close all;

%% ===== 1. FILE PATHS — edit these to match your actual filenames =====
grape(1).name  = 'Green Top';
grape(1).files = {'green_grape_top_1.tif', 'green_grape_top_2.tif'};

grape(2).name  = 'Green Side';
grape(2).files = {'green_grape_side_1.tif', 'green_grape_side_2.tif'};

grape(3).name  = 'Dark Top';
grape(3).files = {'dark_grape_top_1.tif', 'dark_grape_top_2.tif'};

grape(4).name  = 'Dark Side';
grape(4).files = {'dark_grape_side_1.tif', 'dark_grape_side_2.tif'};
% =====================================================================

%% ===== 2. GROUND TRUTH — fill in your measured values =====
% One value per grape configuration (same order as above)
% Brix       = sweetness measured by refractometer (e.g. 14.2)
% Firmness   = texture analyser reading (Newtons or arbitrary score)
% Ripeness   = your 1-5 scale (1=unripe, 5=overripe)
% Spoilage   = your 0-5 scale (0=none, 5=heavily spoiled)

ground_truth.Brix      = [16.0,  16.0,  18.5,  18.5];  % REPLACE with real values
ground_truth.Firmness  = [4.2,   4.2,   2.8,   2.8 ];  % REPLACE with real values
ground_truth.Ripeness  = [2,     2,     4,     4   ];   % REPLACE with real values
ground_truth.Spoilage  = [0,     0,     1,     1   ];   % REPLACE with real values
% NOTE: Top and Side of the same grape share the same ground truth.
%       If you measured them separately, enter different values.
% =====================================================================

%% ===== 3. SETTINGS =====
SPECKLE_WIN = 7;      % local window size for speckle contrast (pixels)
ATT_Z_START = 5;      % pixels below surface peak to start attenuation fit
ATT_Z_LEN   = 80;     % number of pixels to fit over
SKIN_THRESH  = 0.5;   % fraction of peak to define skin/flesh boundary

nG = numel(grape);
R  = struct();        % results struct

%% ===== 4. FEATURE EXTRACTION LOOP =====
fprintf('=== Extracting features from %d grape configurations ===\n\n', nG);

for g = 1:nG
    fprintf('--- %s ---\n', grape(g).name);

    % ---- 4a. Load and average the two .tif files ----
    nF  = numel(grape(g).files);
    img = 0;
    for f = 1:nF
        info   = imfinfo(grape(g).files{f});
        nFrames = numel(info);
        tmp    = zeros(info(1).Height, info(1).Width);
        for fr = 1:nFrames
            tmp = tmp + double(imread(grape(g).files{f}, fr));
        end
        img = img + tmp / nFrames;
    end
    img = img / nF;
    [rows, cols] = size(img);

    % ---- 4b. Signal Intensity Profile: mean intensity at each depth ----
    I_profile = mean(img, 2);          % [rows x 1]
    R(g).I_profile = I_profile;
    [peak_val, z_peak] = max(I_profile);

    % ---- 4c. Attenuation Coefficient (Beer-Lambert fit) ----
    z_start = z_peak + ATT_Z_START;
    z_end   = min(z_start + ATT_Z_LEN, rows);
    z_range = (z_start:z_end)';
    I_range = I_profile(z_range);
    valid   = I_range > 0;
    if sum(valid) > 5
        z_fit = double(z_range(valid) - z_start);
        I_fit = log(I_range(valid));
        p     = polyfit(z_fit, I_fit, 1);
        mu_t  = max(-p(1) / 2, 0);    % 1/pixel, clamp to >= 0
    else
        mu_t  = NaN;
    end
    R(g).mu_t = mu_t;

    % ---- 4d. Layer (Skin) Thickness ----
    below_idx = find(I_profile(z_peak:end) < peak_val * SKIN_THRESH, 1, 'first');
    R(g).skin_thickness = double(below_idx);   % pixels from peak to skin/flesh boundary

    % ---- 4e. Penetration Depth (1/e criterion) ----
    threshold_1e = peak_val / exp(1);
    pen_idx = find(I_profile(z_peak:end) < threshold_1e, 1, 'first');
    if ~isempty(pen_idx)
        R(g).penetration_depth = double(pen_idx) - 1;
    else
        R(g).penetration_depth = rows - z_peak;
    end

    % ---- 4f. Speckle Contrast: SC = std(patch)/mean(patch) ----
    half   = floor(SPECKLE_WIN / 2);
    SC_map = zeros(rows, cols);
    for r = (1+half):(rows-half)
        for c = (1+half):(cols-half)
            patch = img(r-half:r+half, c-half:c+half);
            m     = mean(patch(:));
            if m > 0
                SC_map(r,c) = std(patch(:)) / m;
            end
        end
    end
    R(g).speckle_contrast = mean(SC_map(SC_map > 0));
    R(g).SC_map = SC_map;

    % ---- 4g. Texture Features via GLCM ----
    img_u8 = uint8(img / max(img(:)) * 255);
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm   = graycomatrix(img_u8, 'Offset', offsets, 'NumLevels', 64, 'Symmetric', true);
    stats  = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});

    R(g).glcm_contrast    = mean(stats.Contrast);
    R(g).glcm_correlation = mean(stats.Correlation);
    R(g).glcm_energy      = mean(stats.Energy);
    R(g).glcm_homogeneity = mean(stats.Homogeneity);
    R(g).entropy          = entropy(img_u8);

    % ---- 4h. Mean surface and flesh intensities ----
    if ~isnan(R(g).skin_thickness) && R(g).skin_thickness > 0
        skin_end   = min(z_peak + R(g).skin_thickness, rows);
        flesh_start = skin_end + 1;
        R(g).skin_mean  = mean(I_profile(z_peak:skin_end));
        if flesh_start <= rows
            R(g).flesh_mean = mean(I_profile(flesh_start:min(flesh_start+30, rows)));
        else
            R(g).flesh_mean = NaN;
        end
    else
        R(g).skin_mean  = NaN;
        R(g).flesh_mean = NaN;
    end

    R(g).img  = img;
    R(g).name = grape(g).name;

    fprintf('  mu_t        = %.4f /px\n',  R(g).mu_t);
    fprintf('  Skin thick  = %d px\n',     R(g).skin_thickness);
    fprintf('  Pen depth   = %d px\n',     R(g).penetration_depth);
    fprintf('  Speckle SC  = %.3f\n',      R(g).speckle_contrast);
    fprintf('  GLCM Contr  = %.3f\n',      R(g).glcm_contrast);
    fprintf('  Entropy     = %.3f\n\n',    R(g).entropy);
end

%% ===== 5. FIGURE 1: Signal Intensity Profiles =====
figure('Name','Signal Intensity Profiles','Position',[50 50 650 500]);
cols_plot = {'g-','g--','k-','k--'};
hold on;
for g = 1:nG
    plot(R(g).I_profile, 1:numel(R(g).I_profile), cols_plot{g}, ...
         'LineWidth', 2, 'DisplayName', R(g).name);
end
set(gca,'YDir','reverse');
xlabel('Mean Intensity (a.u.)'); ylabel('Depth (pixels)');
title('Signal Intensity Profile — depth vs intensity');
legend('Location','southeast'); grid on;

%% ===== 6. FIGURE 2: Attenuation Curve Overlay =====
figure('Name','Attenuation Fit','Position',[720 50 650 500]);
hold on;
for g = 1:nG
    I_profile = R(g).I_profile;
    [~, z_peak] = max(I_profile);
    z_start = z_peak + ATT_Z_START;
    z_end   = min(z_start + ATT_Z_LEN, numel(I_profile));
    z_range = (z_start:z_end)';
    I_range = I_profile(z_range);
    valid   = I_range > 0;
    if sum(valid) > 5
        semilogx(I_range(valid), z_range(valid), cols_plot{g}, ...
                 'LineWidth', 2, 'DisplayName', sprintf('%s  \\mu_t=%.3f', R(g).name, R(g).mu_t));
    end
end
set(gca,'YDir','reverse');
xlabel('Intensity (log scale)'); ylabel('Depth (pixels)');
title('Attenuation — log(I) vs depth   (steeper = faster attenuation)');
legend('Location','northeast'); grid on;

%% ===== 7. FIGURE 3: Speckle Contrast Maps =====
figure('Name','Speckle Contrast Maps','Position',[50 550 1200 350]);
for g = 1:nG
    subplot(1,4,g);
    imagesc(R(g).SC_map); colormap(gca,'hot'); axis image off; colorbar;
    caxis([0 1]);
    title(sprintf('%s\nSC = %.3f', R(g).name, R(g).speckle_contrast));
end
sgtitle('Speckle Contrast Maps  (bright = high randomness = rough/heterogeneous tissue)');

%% ===== 8. FIGURE 4: Structural Images =====
figure('Name','Structural Images','Position',[50 950 1200 350]);
for g = 1:nG
    subplot(1,4,g);
    imagesc(R(g).img); colormap(gca,'gray'); axis image off; colorbar;
    title(R(g).name);
end
sgtitle('Structural OCT Images (mean intensity)');

%% ===== 9. FIGURE 5: Feature Comparison Bar Charts =====
names = {R.name};

feature_labels = {'Attenuation \mu_t (1/px)', ...
                  'Skin Thickness (px)', ...
                  'Penetration Depth (px)', ...
                  'Speckle Contrast SC', ...
                  'GLCM Contrast', ...
                  'Entropy (bits)'};

feature_vals = [ [R.mu_t];
                 [R.skin_thickness];
                 [R.penetration_depth];
                 [R.speckle_contrast];
                 [R.glcm_contrast];
                 [R.entropy] ];

bar_colors = [0.2 0.6 0.8; 0.4 0.8 0.4; 0.8 0.4 0.4;
              0.7 0.5 0.9; 0.9 0.7 0.2; 0.5 0.8 0.7];

figure('Name','Feature Comparison','Position',[50 50 1400 750]);
for i = 1:6
    subplot(2,3,i);
    bh = bar(feature_vals(i,:));
    bh.FaceColor = 'flat';
    for g = 1:nG
        bh.CData(g,:) = bar_colors(i,:);
    end
    set(gca,'XTickLabel',names,'XTickLabelRotation',20,'FontSize',9);
    ylabel(feature_labels{i});
    title(feature_labels{i});
    grid on;
end
sgtitle('OCT Feature Comparison — Green vs Dark, Top vs Side', 'FontSize', 13);

%% ===== 10. CORRELATION ANALYSIS =====
% Build feature matrix [nG x nFeatures]
F = feature_vals';    % [4 x 6]
feature_names = {'mu_t','SkinThick','PenDepth','SpeckleC','GLCM_Contr','Entropy'};

gt_fields = fieldnames(ground_truth);
nGT       = numel(gt_fields);

fprintf('\n===== CORRELATION TABLE (Pearson r) =====\n');
fprintf('%-14s', '');
for f = 1:numel(feature_names)
    fprintf('  %-10s', feature_names{f});
end
fprintf('\n%s\n', repmat('-', 1, 14 + numel(feature_names)*12));

corr_matrix = zeros(nGT, numel(feature_names));
for q = 1:nGT
    gt_vec = ground_truth.(gt_fields{q})';
    fprintf('%-14s', gt_fields{q});
    for f = 1:numel(feature_names)
        feat_vec = F(:,f);
        valid    = ~isnan(feat_vec) & ~isnan(gt_vec);
        if sum(valid) > 2
            r = corr(feat_vec(valid), gt_vec(valid));
        else
            r = NaN;
        end
        corr_matrix(q,f) = r;
        fprintf('  %+.3f     ', r);
    end
    fprintf('\n');
end

%% ===== 11. FIGURE 6: Correlation Heatmap =====
figure('Name','Correlation Heatmap','Position',[50 50 800 350]);
imagesc(corr_matrix);
colormap(redblue_map());
clim([-1 1]);
colorbar;
set(gca, 'XTick',1:numel(feature_names), 'XTickLabel',feature_names, ...
         'XTickLabelRotation', 30, ...
         'YTick',1:nGT, 'YTickLabel',gt_fields, 'FontSize', 11);
title('Pearson r: OCT Features vs Quality Metrics');
for q = 1:nGT
    for f = 1:numel(feature_names)
        if abs(corr_matrix(q,f)) > 0.5; txtcol = 'w'; else; txtcol = 'k'; end
        text(f, q, sprintf('%.2f', corr_matrix(q,f)), ...
             'HorizontalAlignment','center', 'FontSize',11, 'FontWeight','bold', ...
             'Color', txtcol);
    end
end

%% ===== 12. FIGURE 7: Scatter Plots — Features vs Brix (sweetness) =====
figure('Name','Feature vs Brix','Position',[50 50 1300 700]);
brix_vec = ground_truth.Brix';
marker_styles = {'go','g^','ko','k^'};

for f = 1:numel(feature_names)
    subplot(2,3,f);
    for g = 1:nG
        plot(F(g,f), brix_vec(g), marker_styles{g}, ...
             'MarkerSize',10, 'MarkerFaceColor','auto', 'LineWidth',1.5, ...
             'DisplayName', R(g).name);
        hold on;
    end
    % Trend line
    feat_vec = F(:,f);
    valid    = ~isnan(feat_vec);
    if sum(valid) > 2
        p = polyfit(feat_vec(valid), brix_vec(valid), 1);
        x_line = linspace(min(feat_vec(valid)), max(feat_vec(valid)), 50);
        plot(x_line, polyval(p, x_line), 'b--', 'LineWidth',1, 'HandleVisibility','off');
        r = corr(feat_vec(valid), brix_vec(valid));
        text(0.05, 0.9, sprintf('r = %.2f', r), 'Units','normalized', ...
             'FontSize',11,'Color','b','FontWeight','bold');
    end
    xlabel(feature_names{f}); ylabel('Brix (sweetness)');
    title(feature_names{f}); grid on;
    if f == 1; legend('Location','best','FontSize',7); end
end
sgtitle('OCT Features vs Brix (sweetness)');

%% ===== 13. PRINT FULL TABLE =====
fprintf('\n===== FULL FEATURE TABLE =====\n');
fprintf('%-16s %8s %10s %10s %9s %10s %8s\n', ...
    'Sample','mu_t','SkinThk','PenDepth','SpeckleC','GLCM_C','Entropy');
fprintf('%s\n', repmat('-',1,80));
for g = 1:nG
    fprintf('%-16s %8.4f %10.1f %10.1f %9.3f %10.3f %8.3f\n', ...
        R(g).name, R(g).mu_t, R(g).skin_thickness, R(g).penetration_depth, ...
        R(g).speckle_contrast, R(g).glcm_contrast, R(g).entropy);
end
fprintf('\nDone. Update ground_truth values at the top and re-run for real correlations.\n');

%% ===== Helper: red-blue colormap =====
function cmap = redblue_map()
    n = 64;
    r = [linspace(0,1,n/2), ones(1,n/2)];
    b = [ones(1,n/2), linspace(1,0,n/2)];
    g_ch = [linspace(0,1,n/2), linspace(1,0,n/2)];
    cmap = [r(:), g_ch(:), b(:)];
end
