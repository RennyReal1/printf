clear
clc
close all

%% ============================================================
%%  BLOOD VESSEL MAPPING + FLOW VELOCITY ESTIMATION
%% ============================================================
%  KEY IDEA:
%   - mean(frames)      -> static STRUCTURE (tissue background). Flow blurs out.
%   - var(frames)       -> VESSELS light up (blood flicker = high variance).
%   - decorrelation     -> RELATIVE flow SPEED (fast flow decorrelates faster).
%   - frame-to-frame xcorr -> in-plane velocity in mm/s (needs calibration).
%
%  Intensity (grayscale) TIFFs cannot give absolute axial velocity -- that
%  needs Doppler/phase data. With intensity you get a relative speed map,
%  and absolute in-plane mm/s only if features visibly move across frames.
%% ============================================================

%% ===== CONFIG =====
files = {'NewEliseSkin_1.tif', 'NewEliseSkin2.tif', 'NewEliseSkin3.tif'};
FILE_TO_USE = 1;                 % which file to analyze for velocity
K_REMOVE    = 4;                 % static SVD components to remove before flow analysis

% --- Acquisition calibration (FILL THESE IN for absolute units) ---
FRAME_RATE_HZ = NaN;             % e.g. 100  (frames per second). NaN = relative only
PIXEL_SIZE_UM = NaN;             % e.g. 5    (microns per pixel).  NaN = relative only
% ==================

fname = files{FILE_TO_USE};
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
fprintf('  %d x %d, %d frames\n', H, W, num_frames);

cube = double(cube);
M    = reshape(cube, H*W, num_frames);

%% ---- (1) STRUCTURE: average all frames ----
structural = reshape(mean(M, 2), H, W);

%% ---- (2) VESSELS: remove static tissue via SVD, then variance ----
[U, S, V] = svds(M, K_REMOVE + 6);
M_static  = U(:,1:K_REMOVE) * S(1:K_REMOVE,1:K_REMOVE) * V(:,1:K_REMOVE)';
M_dyn     = M - M_static;
vessel_map = reshape(var(M_dyn, 0, 2), H, W);

% Vessel mask: keep the brightest flow pixels (top 15%) as "vessels"
thresh      = prctile(vessel_map(:), 85);
vessel_mask = vessel_map > thresh;

%% ---- (3) RELATIVE SPEED: temporal autocorrelation decay ----
%  For each pixel, how fast does its (dynamic) time-series lose correlation
%  with itself after 1 frame? Fast flow -> low lag-1 correlation -> faster.
Md = reshape(M_dyn, H, W, num_frames);
x0 = Md(:,:,1:end-1);
x1 = Md(:,:,2:end);

mu0 = mean(x0,3);  mu1 = mean(x1,3);
num = mean((x0-mu0).*(x1-mu1), 3);
den = sqrt(var(x0,0,3).*var(x1,0,3)) + eps;
lag1_corr = num ./ den;                 % 1 = static, ->0 = fast decorrelation

% Relative speed: 1 - correlation (higher = faster flow). Only meaningful in vessels.
rel_speed = (1 - lag1_corr);
rel_speed(~vessel_mask) = NaN;          % blank out non-vessel pixels

% Convert decorrelation to a relative "velocity index"; scale to mm/s if calibrated
if ~isnan(FRAME_RATE_HZ) && ~isnan(PIXEL_SIZE_UM)
    % crude: decorrelation rate per second times pixel size -> um/s -> mm/s
    decorr_rate = rel_speed * FRAME_RATE_HZ;          % per second
    velocity_mmps = decorr_rate * PIXEL_SIZE_UM / 1000;
    speed_to_show = velocity_mmps;  speed_units = 'mm/s';
else
    speed_to_show = rel_speed;       speed_units = 'relative (a.u.)';
end

%% ---- (4) IN-PLANE VELOCITY: adjacent-frame cross-correlation ----
%  Estimate how far the flow pattern shifts between consecutive frames by
%  cross-correlating small blocks. Gives mm/s if calibrated.
BLOCK   = 16;                          % block size in pixels
MAXSHIFT= 4;                           % search +/- this many pixels
nby = floor(H/BLOCK); nbx = floor(W/BLOCK);
disp_mag = nan(nby, nbx);

for by = 1:nby
    for bx = 1:nbx
        rs = (by-1)*BLOCK + (1:BLOCK);
        cs = (bx-1)*BLOCK + (1:BLOCK);
        if ~any(vessel_mask(rs,cs),'all'), continue; end

        % average displacement over a few frame pairs
        dmags = [];
        for f = 1:min(20, num_frames-1)
            A = Md(rs,cs,f);  Bk = Md(rs,cs,f+1);
            c = normxcorr2(A - mean(A(:)), Bk - mean(Bk(:)));
            [~,imax] = max(c(:));
            [yp,xp]  = ind2sub(size(c), imax);
            dy = yp - BLOCK;  dx = xp - BLOCK;
            if abs(dy)<=MAXSHIFT && abs(dx)<=MAXSHIFT
                dmags(end+1) = hypot(dy,dx); %#ok<SAGROW>
            end
        end
        if ~isempty(dmags), disp_mag(by,bx) = median(dmags); end
    end
end

if ~isnan(FRAME_RATE_HZ) && ~isnan(PIXEL_SIZE_UM)
    inplane_vel = disp_mag * PIXEL_SIZE_UM * FRAME_RATE_HZ / 1000; % mm/s
    inplane_units = 'mm/s';
else
    inplane_vel = disp_mag;          % pixels/frame
    inplane_units = 'pixels/frame';
end

%% ====================================================
%% FIGURE — structure, vessels, speed
%% ====================================================
figure('Name','Blood Flow Analysis','Position',[50 50 1500 800]);

subplot(2,3,1);
imagesc(structural); colormap(gca,gray); axis image off; colorbar;
title('1. Average of all frames (STRUCTURE)','FontSize',10,'FontWeight','bold');

subplot(2,3,2);
imagesc(log1p(vessel_map)); colormap(gca,hot); axis image off; colorbar;
title('2. Variance map (VESSELS light up)','FontSize',10,'FontWeight','bold');

subplot(2,3,3);
imagesc(vessel_mask); colormap(gca,gray); axis image off;
title(sprintf('3. Vessel mask (top 15%%)'),'FontSize',10,'FontWeight','bold');

subplot(2,3,4);
imagesc(speed_to_show); colormap(gca,jet); axis image off; c=colorbar;
c.Label.String = speed_units;
title('4. Relative flow SPEED (decorrelation)','FontSize',10,'FontWeight','bold');

subplot(2,3,5);
imagesc(inplane_vel); colormap(gca,parula); axis image off; c=colorbar;
c.Label.String = inplane_units;
title('5. In-plane VELOCITY (block xcorr)','FontSize',10,'FontWeight','bold');

% overlay vessels on structure
subplot(2,3,6);
sN = mat2gray(structural);
vN = mat2gray(log1p(vessel_map));
rgb = cat(3, min(sN+vN,1), sN, sN);   % vessels in red over grey tissue
image(rgb); axis image off;
title('6. Vessels overlaid on structure','FontSize',10,'FontWeight','bold');

sgtitle(sprintf('Blood Flow — %s', fname),'Interpreter','none', ...
        'FontSize',13,'FontWeight','bold');
saveas(gcf,'blood_flow_analysis.png');

%% ---- SUMMARY ----
fprintf('\n========== BLOOD FLOW SUMMARY ==========\n');
fprintf('Vessel pixels (top 15%%) : %d (%.1f%% of frame)\n', ...
        nnz(vessel_mask), 100*nnz(vessel_mask)/(H*W));
fprintf('Mean flow speed in vessels: %.4f %s\n', ...
        mean(speed_to_show(vessel_mask),'omitnan'), speed_units);
fprintf('Max  flow speed in vessels: %.4f %s\n', ...
        max(speed_to_show(:),[],'omitnan'), speed_units);
fprintf('Mean in-plane velocity   : %.4f %s\n', ...
        mean(inplane_vel(:),'omitnan'), inplane_units);
if isnan(FRAME_RATE_HZ) || isnan(PIXEL_SIZE_UM)
    fprintf('\n[!] Set FRAME_RATE_HZ and PIXEL_SIZE_UM at the top for absolute mm/s.\n');
end
fprintf('========================================\n');
