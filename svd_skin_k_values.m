clear
clc
close all

%% ===== CONFIGURATION =====
files = {'NewEliseSkin_1.tif', 'NewEliseSkin2.tif', 'NewEliseSkin3.tif'};
K_VALUES = [1, 2, 4, 6, 10, 20];   % static components to remove — edit freely
N_COMP   = 30;                      % how many singular vectors to compute (>= max(K_VALUES)+some)
% ==========================

N_files  = numel(files);
N_k      = numel(K_VALUES);

%% ---- LOAD ALL 3 FILES ----
sets = struct();
for s = 1:N_files
    fname = files{s};
    fprintf('Loading %s ...\n', fname);

    info       = imfinfo(fname);
    num_frames = numel(info);

    frame = imread(fname, 1);
    if ndims(frame) == 3, frame = frame(:,:,1); end
    [H, W] = size(frame);

    cube = zeros(H, W, num_frames, 'uint8');
    cube(:,:,1) = frame;
    for f = 2:num_frames
        fr = imread(fname, f);
        if ndims(fr) == 3, fr = fr(:,:,1); end
        cube(:,:,f) = fr;
    end

    M = reshape(double(cube), H*W, num_frames);
    fprintf('  -> %d x %d pixels, %d frames\n', H, W, num_frames);

    % Compute SVD once — reuse for all K values
    n_comp = min(N_COMP, min(size(M)));
    [U, S, V] = svds(M, n_comp);
    sv = diag(S);

    sets(s).M          = M;
    sets(s).U          = U;
    sets(s).S          = S;
    sets(s).V          = V;
    sets(s).sv         = sv;
    sets(s).H          = H;
    sets(s).W          = W;
    sets(s).structural = reshape(mean(M, 2), H, W);
    sets(s).name       = fname;
end
fprintf('All files loaded.\n\n');

%% ====================================================
%% FIGURE 1 — Structural images (mean frame, all 3 files)
%% ====================================================
figure('Name','Structural Images','Position',[50 50 1200 380]);
for s = 1:N_files
    subplot(1, N_files, s);
    imagesc(sets(s).structural);
    colormap(gca, gray); axis image off; colorbar;
    title(sets(s).name, 'Interpreter','none', 'FontSize',9, 'FontWeight','bold');
end
sgtitle('Structural Image — Mean of All Frames', 'FontSize',13, 'FontWeight','bold');
saveas(gcf, 'fig1_structural.png');

%% ====================================================
%% FIGURE 2 — Singular value spectra for all 3 files
%% ====================================================
colors = lines(N_files);
figure('Name','Singular Value Spectra','Position',[50 50 900 450]);
hold on;
for s = 1:N_files
    plot(sets(s).sv, 'o-', 'LineWidth', 2, 'Color', colors(s,:), ...
         'MarkerFaceColor', colors(s,:), 'DisplayName', sets(s).name);
end
for ki = 1:N_k
    xline(K_VALUES(ki), '--', sprintf('K=%d', K_VALUES(ki)), ...
          'LineWidth', 1, 'Color', [0.5 0.5 0.5], 'HandleVisibility','off');
end
legend('Interpreter','none','Location','northeast','FontSize',9);
xlabel('Component index'); ylabel('Singular value');
title('Singular Value Spectra — All 3 Files','FontSize',12,'FontWeight','bold');
grid on;
saveas(gcf,'fig2_singular_values.png');

%% ====================================================
%% FIGURE 3 — OCTA maps across K values (one row per file)
%%   columns: K_VALUES(1), K_VALUES(2), ..., no-SVD baseline
%% ====================================================
figure('Name','OCTA Maps vs K','Position',[50 50 300*(N_k+1) 300*N_files]);

col_labels = [arrayfun(@(k) sprintf('K = %d removed', k), K_VALUES, 'UniformOutput', false), ...
              {'No SVD (baseline)'}];

for s = 1:N_files
    M   = sets(s).M;
    U   = sets(s).U;
    S_  = sets(s).S;
    V   = sets(s).V;
    H   = sets(s).H;
    W   = sets(s).W;

    for ki = 1:N_k
        K = K_VALUES(ki);
        M_static  = U(:,1:K) * S_(1:K,1:K) * V(:,1:K)';
        M_dyn     = M - M_static;
        octa      = reshape(var(M_dyn, 0, 2), H, W);

        subplot(N_files, N_k+1, (s-1)*(N_k+1) + ki);
        imagesc(log1p(octa));
        colormap(gca, hot); axis image off;

        if s == 1
            title(col_labels{ki}, 'FontSize', 8, 'FontWeight', 'bold');
        end
        if ki == 1
            ylabel(sets(s).name, 'Interpreter','none', 'FontSize', 8, 'FontWeight','bold');
        end
    end

    % Last column: no SVD
    octa_nosvd = reshape(var(M, 0, 2), H, W);
    subplot(N_files, N_k+1, s*(N_k+1));
    imagesc(log1p(octa_nosvd));
    colormap(gca, hot); axis image off;
    if s == 1
        title('No SVD (baseline)', 'FontSize', 8, 'FontWeight', 'bold');
    end
end

sgtitle('OCTA Maps — Effect of K (static components removed) [log scale]', ...
        'FontSize', 13, 'FontWeight', 'bold');
saveas(gcf, 'fig3_octa_k_comparison.png');

%% ====================================================
%% FIGURE 4 — Quantitative metrics vs K (per file)
%%   mean variance and CNR as a function of K removed
%% ====================================================
figure('Name','Metrics vs K','Position',[50 50 1200 500]);

ax1 = subplot(1,2,1); hold on;
ax2 = subplot(1,2,2); hold on;

for s = 1:N_files
    M   = sets(s).M;
    U   = sets(s).U;
    S_  = sets(s).S;
    V   = sets(s).V;
    H   = sets(s).H;
    W   = sets(s).W;

    mean_var_k = zeros(1, N_k);
    cnr_k      = zeros(1, N_k);

    for ki = 1:N_k
        K = K_VALUES(ki);
        M_static  = U(:,1:K) * S_(1:K,1:K) * V(:,1:K)';
        M_dyn     = M - M_static;
        octa      = reshape(var(M_dyn, 0, 2), H, W);
        p95       = prctile(octa(:), 95);
        mn        = mean(octa(:));
        sd        = std(octa(:));
        mean_var_k(ki) = mn;
        cnr_k(ki)      = (p95 - mn) / sd;
    end

    plot(ax1, K_VALUES, mean_var_k, 'o-', 'LineWidth', 2, 'Color', colors(s,:), ...
         'MarkerFaceColor', colors(s,:), 'DisplayName', sets(s).name);
    plot(ax2, K_VALUES, cnr_k,      'o-', 'LineWidth', 2, 'Color', colors(s,:), ...
         'MarkerFaceColor', colors(s,:), 'DisplayName', sets(s).name);
end

xlabel(ax1, 'K removed'); ylabel(ax1, 'Mean variance');
title(ax1, 'Mean Variance vs K Removed', 'FontSize', 11, 'FontWeight', 'bold');
legend(ax1, 'Interpreter','none','Location','best','FontSize',8);
grid(ax1, 'on');

xlabel(ax2, 'K removed'); ylabel(ax2, 'CNR');
title(ax2, 'CNR vs K Removed', 'FontSize', 11, 'FontWeight', 'bold');
legend(ax2, 'Interpreter','none','Location','best','FontSize',8);
grid(ax2, 'on');

sgtitle('Quantitative Metrics vs Number of Static Components Removed', ...
        'FontSize', 13, 'FontWeight', 'bold');
saveas(gcf, 'fig4_metrics_vs_k.png');

%% ====================================================
%% FIGURE 5 — Side-by-side individual SVD components
%%   For file 1: show spatial maps of U(:,1:6) reshaped
%% ====================================================
N_SHOW = min(6, size(sets(1).U, 2));
figure('Name','Spatial SVD Components (File 1)','Position',[50 50 250*N_SHOW 550]);

for ci = 1:N_SHOW
    comp = reshape(sets(1).U(:, ci), sets(1).H, sets(1).W);

    subplot(2, N_SHOW, ci);
    imagesc(comp); colormap(gca, gray); axis image off;
    title(sprintf('U_%d  (sv=%.0f)', ci, sets(1).sv(ci)), ...
          'FontSize', 8, 'FontWeight', 'bold');
    if ci == 1, ylabel('Spatial map', 'FontSize', 9, 'FontWeight', 'bold'); end

    subplot(2, N_SHOW, N_SHOW + ci);
    plot(sets(1).V(:, ci), 'LineWidth', 1);
    xlabel('Frame'); ylabel('Weight');
    title(sprintf('Temporal V_%d', ci), 'FontSize', 8, 'FontWeight', 'bold');
    grid on;
end

sgtitle(sprintf('Top %d SVD Components — %s', N_SHOW, sets(1).name), ...
        'Interpreter','none','FontSize',12,'FontWeight','bold');
saveas(gcf,'fig5_svd_components.png');

%% ====================================================
%% FIGURE 6 — Best K reconstruction comparison (file 1)
%%   Row 1: structural | K=2 | K=4 | K=10 | no-SVD
%% ====================================================
K_DEMO = [2, 4, 6, 10];
M   = sets(1).M;
U   = sets(1).U;
S_  = sets(1).S;
V   = sets(1).V;
H   = sets(1).H;
W   = sets(1).W;

figure('Name','Reconstruction Comparison (File 1)','Position',[50 50 300*(numel(K_DEMO)+2) 380]);

% structural
subplot(1, numel(K_DEMO)+2, 1);
imagesc(sets(1).structural); colormap(gca,gray); axis image off; colorbar;
title('Structural', 'FontSize', 9, 'FontWeight', 'bold');

for ki = 1:numel(K_DEMO)
    K = K_DEMO(ki);
    M_static = U(:,1:K) * S_(1:K,1:K) * V(:,1:K)';
    M_dyn    = M - M_static;
    octa     = reshape(var(M_dyn, 0, 2), H, W);

    subplot(1, numel(K_DEMO)+2, ki+1);
    imagesc(log1p(octa)); colormap(gca,hot); axis image off; colorbar;
    title(sprintf('OCTA K=%d', K), 'FontSize', 9, 'FontWeight', 'bold');
end

% no SVD
octa_nosvd = reshape(var(M, 0, 2), H, W);
subplot(1, numel(K_DEMO)+2, numel(K_DEMO)+2);
imagesc(log1p(octa_nosvd)); colormap(gca,hot); axis image off; colorbar;
title('No SVD', 'FontSize', 9, 'FontWeight', 'bold');

sgtitle(sprintf('OCTA Reconstruction — %s [log scale]', sets(1).name), ...
        'Interpreter','none','FontSize',12,'FontWeight','bold');
saveas(gcf,'fig6_reconstruction_comparison.png');

%% ====================================================
%% PRINT SUMMARY TABLE
%% ====================================================
fprintf('\n%s\n', repmat('=',1,70));
fprintf('%-30s  %8s  %8s  %8s\n','File','K removed','Mean Var','CNR');
fprintf('%s\n', repmat('-',1,70));
for s = 1:N_files
    M   = sets(s).M;
    U   = sets(s).U;
    S_  = sets(s).S;
    V   = sets(s).V;
    H   = sets(s).H;
    W   = sets(s).W;
    for ki = 1:N_k
        K = K_VALUES(ki);
        M_dyn = M - U(:,1:K)*S_(1:K,1:K)*V(:,1:K)';
        octa  = var(M_dyn, 0, 2);
        p95   = prctile(octa, 95);
        mn    = mean(octa);
        sd    = std(octa);
        fprintf('%-30s  %8d  %8.2f  %8.2f\n', sets(s).name, K, mn, (p95-mn)/sd);
    end
    fprintf('%s\n', repmat('-',1,70));
end
fprintf('%s\n\n', repmat('=',1,70));
