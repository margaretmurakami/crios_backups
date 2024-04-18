%/scratch/atnguyen/aste_270x450x180/OFFICIAL_ASTE_R1_Sep2019/diags/TRSP/trsp_2d_set1.0000438048.meta
%nDims = [   2 ];
% dimList = [
%    90,    1,   90,
%  4050,    1, 4050
% ];
% dataprec = [ 'float32' ];
% nrecords = [     8 ];
% timeStepNumber = [     438048 ];
% timeInterval = [  2.608848000000E+08  2.628288000000E+08 ];
% missingValue = [ -9.99000000000000E+02 ];
% nFlds = [    8 ];
% fldList = {
% 'DFxE_TH ' 'DFyE_TH ' 'ADVx_TH ' 'ADVy_TH ' 'DFxE_SLT' 'DFyE_SLT' 'ADVx_SLT' 'ADVy_SLT'
% };
clear all;
nx=270;ncut1=450;ncut2=180;ny=2*ncut1+nx+ncut2;nfx=[nx 0 nx ncut2 ncut1];nfy=[ncut1 0 nx nx nx];nz=50;


dirroot='/scratch/atnguyen/aste_270x450x180/';
dirrun=[dirroot 'OFFICIAL_ASTE_R1_Sep2019/'];
dirgrid=[dirroot 'GRID_real8/'];

u=readbin([dirrun 'diags/TRSP/trsp_2d_set1.0000438048.data'],[nx ny],1,'real*4',2);%ADVx_TH (advective of theta ib x-dir)
v=readbin([dirrun 'diags/TRSP/trsp_2d_set1.0000438048.data'],[nx ny],1,'real*4',3);%ADVy_TH

[uaste,vaste]=get_aste_vector(u,v,nfx,nfy,1); %"1" at the end: takes care of sign, [541 901 1];

hfw=rdmds([dirgrid 'hFacW']);hfw=reshape(hfw,nx,ny,nz); %[270 1350 50]
hfs=rdmds([dirgrid 'hFacS']);hfs=reshape(hfs,nx,ny,nz);

[hfwaste,hfsaste]=get_aste_vector(hfw,hfs,nfx,nfy,0);   %[541 901 50]

dirout='/scratch/mmurakami/ASTE_270/';if(~exist(dirout));mkdir(dirout);end;

figure(1);clf;colormap(seismic(21));
subplot(2,3,1);hh=pcolor(1:541,1:901,uaste');shading flat;mycaxis(.3);mythincolorbar;grid;title('u');
subplot(2,3,4);hh=pcolor(1:541,1:901,vaste');shading flat;mycaxis(.3);mythincolorbar;grid;title('v');
subplot(2,3,2);hh=pcolor(1:541,1:901,hfwaste(:,:,1)');shading flat;mythincolorbar;grid;title('hfw(1)');
subplot(2,3,5);hh=pcolor(1:541,1:901,hfsaste(:,:,1)');shading flat;mythincolorbar;grid;title('hfs(1)');
subplot(2,3,3);hh=pcolor(1:541,1:901,hfwaste(:,:,30)');shading flat;mythincolorbar;grid;title('hfw(30)');
subplot(2,3,5);hh=pcolor(1:541,1:901,hfsaste(:,:,30)');shading flat;mythincolorbar;grid;title('hfs(30)');

figure(1);set(gcf,'paperunit','inches','paperposition',[0 0 14 10]);
fpr=[dirout 'test_get_aste_vector.png'];print(fpr,'-dpng');fprintf('%s\n',fpr);

foutu=[dirout 'uaste.bin'];writebin(foutu,uaste,1,'real*4');
foutv=[dirout 'vaste.bin'];writebin(foutv,vaste,1,'real*4');
