%EXAMPLE_GET_VECTOR Demonstrate compact ASTE vector conversion in MATLAB.
%
% This script reads one ASTE transport diagnostic record, reshapes U/V
% components into the stitched ASTE domain, plots the result, and optionally
% writes the converted arrays to binary files.
%
% Required helper functions on the MATLAB path:
%   readbin, rdmds, writebin, get_aste_vector, sym_g_mod

clearvars;
close all;

%% User configuration

nx = 270;
ncut1 = 450;
ncut2 = 180;
nz = 50;

dirroot = '/scratch/atnguyen/aste_270x450x180/';
dirrun = fullfile(dirroot, 'OFFICIAL_ASTE_R1_Sep2019');
dirgrid = fullfile(dirroot, 'GRID_real8');
dirout = '/scratch/mmurakami/ASTE_270/';

transport_file = fullfile(dirrun, 'diags', 'TRSP', ...
    'trsp_2d_set1.0000438048.data');

% readbin uses zero-based record offsets in the existing notebooks/helpers.
theta_u_record = 2;  % ADVx_TH, third field in the metadata.
theta_v_record = 3;  % ADVy_TH, fourth field in the metadata.
write_output = true;

%% ASTE grid dimensions

ny = 2 * ncut1 + nx + ncut2;
nfx = [nx, 0, nx, ncut2, ncut1];
nfy = [ncut1, 0, nx, nx, nx];

fprintf('Input compact size: [%d %d]\n', nx, ny);
fprintf('Output ASTE vector size should be [%d %d]\n', ...
    nfy(5) + nfx(1) + 1, nfy(1) + nfx(3) + nfx(4) + 1);

%% Validate local files

required_files = {
    transport_file
    fullfile(dirgrid, 'hFacW.data')
    fullfile(dirgrid, 'hFacS.data')
};

for ifile = 1:numel(required_files)
    if ~exist(required_files{ifile}, 'file')
        error('Missing required file: %s', required_files{ifile});
    end
end

if ~exist(dirout, 'dir')
    mkdir(dirout);
end

%% Read and convert transport vectors

u_compact = readbin(transport_file, [nx, ny], 1, 'real*4', theta_u_record);
v_compact = readbin(transport_file, [nx, ny], 1, 'real*4', theta_v_record);

% sign_switch = 1 applies vector sign/orientation changes.
[u_aste, v_aste] = get_aste_vector(u_compact, v_compact, nfx, nfy, 1);

%% Read and convert wet-cell masks

hfacw = rdmds(fullfile(dirgrid, 'hFacW'));
hfacw = reshape(hfacw, nx, ny, nz);

hfacs = rdmds(fullfile(dirgrid, 'hFacS'));
hfacs = reshape(hfacs, nx, ny, nz);

% sign_switch = 0 keeps grid metrics/masks positive after rotation.
[hfacw_aste, hfacs_aste] = get_aste_vector(hfacw, hfacs, nfx, nfy, 0);

%% Plot a compact diagnostic figure

figure(1);
clf;
colormap(seismic(21));

subplot(2, 3, 1);
pcolor(1:size(u_aste, 1), 1:size(u_aste, 2), u_aste');
shading flat;
colorbar;
grid on;
title('ADVx\_TH on ASTE domain');

subplot(2, 3, 4);
pcolor(1:size(v_aste, 1), 1:size(v_aste, 2), v_aste');
shading flat;
colorbar;
grid on;
title('ADVy\_TH on ASTE domain');

subplot(2, 3, 2);
pcolor(1:size(hfacw_aste, 1), 1:size(hfacw_aste, 2), hfacw_aste(:, :, 1)');
shading flat;
colorbar;
grid on;
title('hFacW level 1');

subplot(2, 3, 5);
pcolor(1:size(hfacs_aste, 1), 1:size(hfacs_aste, 2), hfacs_aste(:, :, 1)');
shading flat;
colorbar;
grid on;
title('hFacS level 1');

subplot(2, 3, 3);
pcolor(1:size(hfacw_aste, 1), 1:size(hfacw_aste, 2), hfacw_aste(:, :, 30)');
shading flat;
colorbar;
grid on;
title('hFacW level 30');

subplot(2, 3, 6);
pcolor(1:size(hfacs_aste, 1), 1:size(hfacs_aste, 2), hfacs_aste(:, :, 30)');
shading flat;
colorbar;
grid on;
title('hFacS level 30');

set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, 14, 10]);
figure_file = fullfile(dirout, 'test_get_aste_vector.png');
print(figure_file, '-dpng');
fprintf('Wrote figure: %s\n', figure_file);

%% Optionally write converted arrays

if write_output
    u_file = fullfile(dirout, 'uaste.bin');
    v_file = fullfile(dirout, 'vaste.bin');

    writebin(u_file, u_aste, 1, 'real*4');
    writebin(v_file, v_aste, 1, 'real*4');

    fprintf('Wrote binary output: %s\n', u_file);
    fprintf('Wrote binary output: %s\n', v_file);
end
