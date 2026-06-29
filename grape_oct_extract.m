%% Grape OCT Feature Extraction
% Loads 4 grape OCT configurations and extracts:
%   - Signal intensity profile (depth profile)
%   - Attenuation coefficient
%   - Penetration depth
%   - Skin/surface layer thickness
%   - Speckle contrast
%   - Texture features (GLCM)
% No ground truth needed — just shows what the OCT data contains.

clear; clc; close all;

%% ===== FILE PATHS — edit to match your filenames =====
grape(1).name  = 'Green Top';
grape(1).files = {'green_grape_top_1.tif', 'green_grape_top_2.tif'};

grape(2).name  = 'Green Side';
grape(2).files = {'green_grape_side_1.tif', 'green_grape_side_2.tif'};

grape(3).name  = 'Dark Top';
grape(3).files = {'dark_grape_top_1.tif', 'dark_grape_top_2.tif'};

grape(4).name  = 'Dark Side';
grape(4).files = {'dark_grape_side_1.tif', 'dark_grape_side_2.tif'};
% ======================================================

%% Settings
SPECKLE_WIN = 7;    % window size for speckle contrast
ATT_Z_START = 5;    % pixels below surface to start attenuation fit
ATT_Z_LEN   = 80;   % depth range (pixels) to fit over
SKIN_THRESH  = 0.5; % fraction of peak that marks skin/flesh boundary

nG = numel(grape);
R  = struct();

%% ===== EXTRACTION LOOP =====
for g = 1:nG
    fprintf('Loading: %s\n', grape(g).name);

    % Load and average all frames from both files
    img = 0;
    for f = 1:numel(grape(g).files)
        info    = imfinfo(grape(g).files{f});
        nFrames = numel(info);
        tmp     = zeros(info(1).Height, info(1).Width);
        for fr  = 1:nFrames
            tmp = tmp + double(imread(grape(g).files{f}, fr));
        end
        img = img + tmp / nFrames;
    end
    img = img / numel(grape(g).files);
    [rows, cols] = size(img);

    %% Feature 1: Signal Intensity Profile
    I_profile = mean(img, 2);              % mean intensity at each depth
    [peak_val, z_peak] = max(I_profile);
    R(g).I_profile = I_profile;
    R(g).z_peak    = z_peak;

    %% Feature 2: Attenuation Coefficient  (Beer-Lambert fit)
    z_s   = z_peak + ATT_Z_START;
    z_e   = min(z_s + ATT_Z_LEN, rows);
    zr    = (z_s:z_e)';
    Ir    = I_profile(zr);
    valid = Ir > 0;
    if sum(valid) > 5
        zf   = double(zr(valid) - z_s);
        p    = polyfit(zf, log(Ir(valid)), 1);
        mu_t = max(-p(1)/2, 0);
    else
        mu_t = NaN;
    end
    R(g).mu_t = mu_t;

    %% Feature 3: Penetration Depth  (signal drops to 1/e of peak)
    idx = find(I_profile(z_peak:end) < peak_val/exp(1), 1, 'first');
    R(g).pen_depth = double(idx) - 1;

    %% Feature 4: Skin/Surface Layer Thickness
    idx2 = find(I_profile(z_peak:end) < peak_val * SKIN_THRESH, 1, 'first');
    R(g).skin_thick = double(idx2);

    %% Feature 5: Speckle Contrast  SC = std/mean in local windows
    half   = floor(SPECKLE_WIN/2);
    SC_map = zeros(rows, cols);
    for r  = (1+half):(rows-half)
        for c = (1+half):(cols-half)
            patch = img(r-half:r+half, c-half:c+half);
            m     = mean(patch(:));
            if m > 0; SC_map(r,c) = std(patch(:))/m; end
        end
    end
    R(g).SC      = mean(SC_map(SC_map > 0));
    R(g).SC_map  = SC_map;

    %% Feature 6: Texture (GLCM)
    img_u8 = uint8(img / max(img(:)) * 255);
    glcm   = graycomatrix(img_u8, 'Offset',[0 1;-1 1;-1 0;-1 -1], ...
                          'NumLevels',64,'Symmetric',true);
    st     = graycoprops(glcm, {'Contrast','Correlation','Energy','Homogeneity'});
    R(g).contrast    = mean(st.Contrast);
    R(g).correlation = mean(st.Correlation);
    R(g).energy      = mean(st.Energy);
    R(g).homogeneity = mean(st.Homogeneity);
    R(g).entropy     = entropy(img_u8);

    R(g).img  = img;
    R(g).name = grape(g).name;
end

%% ===== FIGURE 1: Raw Structural Images =====
figure('Name','Structural OCT Images','Position',[50 50 1200 320]);
for g = 1:nG
    subplot(1,4,g);
    imagesc(R(g).img); colormap(gca,'gray'); axis image off; colorbar;
    title(R(g).name,'FontSize',11);
end
sgtitle('Structural OCT Images — averaged over all frames','FontSize',13);

%% ===== FIGURE 2: Signal Intensity Profiles =====
figure('Name','Depth Profiles','Position',[50 420 700 500]);
styles = {'g-','g--','k-','k--'};
hold on;
for g = 1:nG
    plot(R(g).I_profile, 1:numel(R(g).I_profile), styles{g}, ...
         'LineWidth',2.5,'DisplayName',R(g).name);
end
set(gca,'YDir','reverse');
xlabel('Mean Intensity (a.u.)','FontSize',12);
ylabel('Depth (pixels)','FontSize',12);
title('Signal Intensity Profile — how light penetrates into tissue','FontSize',12);
legend('Location','southeast','FontSize',10);
grid on;

%% ===== FIGURE 3: Attenuation Fit on log scale =====
figure('Name','Attenuation Coefficient','Position',[780 420 700 500]);
hold on;
for g = 1:nG
    zp  = R(g).z_peak;
    z_s = zp + ATT_Z_START;
    z_e = min(z_s + ATT_Z_LEN, numel(R(g).I_profile));
    zr  = (z_s:z_e)';
    Ir  = R(g).I_profile(zr);
    v   = Ir > 0;
    semilogx(Ir(v), zr(v), styles{g}, 'LineWidth',2.5, ...
             'DisplayName', sprintf('%s  (\\mu_t=%.3f)', R(g).name, R(g).mu_t));
end
set(gca,'YDir','reverse');
xlabel('Intensity (log scale)','FontSize',12);
ylabel('Depth (pixels)','FontSize',12);
title('Attenuation — steeper slope = faster absorption','FontSize',12);
legend('Location','northeast','FontSize',10);
grid on;

%% ===== FIGURE 4: Speckle Contrast Maps =====
figure('Name','Speckle Contrast','Position',[50 50 1200 320]);
for g = 1:nG
    subplot(1,4,g);
    imagesc(R(g).SC_map,[0 1]); colormap(gca,'hot'); axis image off; colorbar;
    title(sprintf('%s\nSC = %.3f', R(g).name, R(g).SC),'FontSize',10);
end
sgtitle('Speckle Contrast   (bright = rough / heterogeneous tissue)','FontSize',13);

%% ===== FIGURE 5: Feature Comparison Bar Charts =====
names = {R.name};
figure('Name','Feature Summary','Position',[50 50 1300 700]);

feat = { [R.mu_t],         'Attenuation  \mu_t  (1/pixel)',  [0.2 0.6 0.8]  ;
         [R.skin_thick],   'Skin Thickness  (pixels)',        [0.4 0.8 0.4]  ;
         [R.pen_depth],    'Penetration Depth  (pixels)',     [0.8 0.4 0.4]  ;
         [R.SC],           'Speckle Contrast  SC',            [0.7 0.5 0.9]  ;
         [R.contrast],     'GLCM Contrast',                   [0.9 0.7 0.2]  ;
         [R.entropy],      'Entropy  (bits)',                  [0.5 0.8 0.7]  };

for i = 1:6
    subplot(2,3,i);
    vals = feat{i,1};
    bh   = bar(vals);
    bh.FaceColor = 'flat';
    for g = 1:nG; bh.CData(g,:) = feat{i,3}; end
    set(gca,'XTickLabel',names,'XTickLabelRotation',20,'FontSize',9);
    ylabel(feat{i,2},'FontSize',9);
    title(feat{i,2},'FontSize',9);
    % Label values on bars
    for g = 1:nG
        text(g, vals(g)*1.02, sprintf('%.3f', vals(g)), ...
             'HorizontalAlignment','center','FontSize',8);
    end
    grid on; ylim([0, max(vals)*1.15]);
end
sgtitle('Extracted OCT Features — Green vs Dark, Top vs Side','FontSize',13);

%% ===== FIGURE 6: Top vs Side comparison (Green and Dark separately) =====
figure('Name','Top vs Side','Position',[50 50 900 500]);
feat_names = {'mu\_t','SkinThick','PenDepth','SC','GLCM Contr','Entropy'};
green_top  = [R(1).mu_t, R(1).skin_thick, R(1).pen_depth, R(1).SC, R(1).contrast, R(1).entropy];
green_side = [R(2).mu_t, R(2).skin_thick, R(2).pen_depth, R(2).SC, R(2).contrast, R(2).entropy];
dark_top   = [R(3).mu_t, R(3).skin_thick, R(3).pen_depth, R(3).SC, R(3).contrast, R(3).entropy];
dark_side  = [R(4).mu_t, R(4).skin_thick, R(4).pen_depth, R(4).SC, R(4).contrast, R(4).entropy];

% Normalize each feature 0-1 for radar-style comparison
all_rows   = [green_top; green_side; dark_top; dark_side];
mn         = min(all_rows);
mx         = max(all_rows);
rng        = mx - mn;
rng(rng==0) = 1;
norm_rows  = (all_rows - mn) ./ rng;

subplot(1,2,1);
bar([norm_rows(1,:); norm_rows(2,:)]');
set(gca,'XTickLabel',feat_names,'XTickLabelRotation',30,'FontSize',9);
legend({'Top view','Side view'},'Location','northeast');
title('Green Grape: Top vs Side (normalized)','FontSize',11);
ylabel('Normalized feature value'); grid on; ylim([0 1.2]);

subplot(1,2,2);
bar([norm_rows(3,:); norm_rows(4,:)]');
set(gca,'XTickLabel',feat_names,'XTickLabelRotation',30,'FontSize',9);
legend({'Top view','Side view'},'Location','northeast');
title('Dark Grape: Top vs Side (normalized)','FontSize',11);
ylabel('Normalized feature value'); grid on; ylim([0 1.2]);

sgtitle('How imaging angle changes extracted features','FontSize',13);

%% ===== PRINT SUMMARY TABLE =====
fprintf('\n===== EXTRACTED FEATURE SUMMARY =====\n');
fprintf('%-14s %8s %9s %9s %8s %8s %8s\n', ...
    'Sample','mu_t','SkinThk','PenDepth','SC','GLCM_C','Entropy');
fprintf('%s\n', repmat('-',1,72));
for g = 1:nG
    fprintf('%-14s %8.4f %9.1f %9.1f %8.3f %8.3f %8.3f\n', ...
        R(g).name, R(g).mu_t, R(g).skin_thick, R(g).pen_depth, ...
        R(g).SC, R(g).contrast, R(g).entropy);
end

fprintf('\n===== WHAT EACH FEATURE TELLS YOU =====\n');
fprintf('mu_t        : how fast light is absorbed — higher = more pigment/density\n');
fprintf('SkinThick   : depth of surface skin layer in pixels\n');
fprintf('PenDepth    : how deep the OCT signal reaches (1/e criterion)\n');
fprintf('SC          : speckle contrast — higher = rougher/more random tissue\n');
fprintf('GLCM_C      : texture contrast — higher = more structural variation\n');
fprintf('Entropy     : image complexity — higher = more varied internal structure\n');
fprintf('\nDone — 6 figures generated.\n');
