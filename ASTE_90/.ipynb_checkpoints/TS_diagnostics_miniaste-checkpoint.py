#!/usr/bin/env python
# coding: utf-8

# ### Add the packages and add python to path

# In[2]:


# packages and plot parameters
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
import numpy.ma as ma
import glob
import os
import re
from operator import mul
import sys

plt.rcParams['figure.figsize'] = (10,4)


# In[10]:


# add mds functions to path
# instead of this command:
# %load "/home/mmurakami/MITgcm/MITgcm_c68r/MITgcm-checkpoint68r/utils/python/MITgcmutils/MITgcmutils/mds.py"
# we can do this one:
sys.path.append("/home/mmurakami/MITgcm/MITgcm_c68r/MITgcm-checkpoint68r/utils/python/MITgcmutils/MITgcmutils/") # go to parent dir
from mds import *

# add reading functions to path
sys.path.append("/home/mmurakami/jupyterfiles/") # go to parent dir
from read_binary import *


# ### See if we created files in TS layers

# In[5]:


# our directories
# dirrun = "/home/mmurakami/MITgcm/MITgcm_c68r/MITgcm-checkpoint68r/verification/lab_sea/tut_layers/runtest/"
dirrun = "/scratch2/atnguyen/aste_90x150x60/run_c67w_layers_03Jun2023_noOL_nlayers24nS24nT24_10d_newLayersDiags"
dir_diags = dirrun + "diags/"


# In[6]:


# nx=20
# ny=16
# nz=23


# In[9]:


mds


# In[7]:


# dir_diags = dirrun + "diags/"
#diags_layers = rdmds(dir_diags + "LAYERS_Cwet", 10)
diags_layers = rdmds(dir_diags + "LAYERS", 10)
print(diags_layers.shape)


# In[6]:


# read in the original array we fed
binsTH = rdmds(dirrun + "layers1TH", -1)
binflatth = binsTH[:,:,0].flatten()
binavgth = (binflatth[:-1] + binflatth[1:])/2
# print(binflat)
print(binavgth.shape)
# print(binavg)

# read in the original array we fed
binsRHO = rdmds(dirrun + "layers2RHO", -1)
binflatrho = binsRHO[:,:,0].flatten()
print()
# print(binflatrho)
binavgrho = (binflatrho[:-1] + binflatrho[1:])/2
# print(binavgrho)

# read in the original array we fed
binsSLT = rdmds(dirrun + "layers3SLT", -1)
binflatslt = binsSLT[:,:,0].flatten()
print()
# print(binflatslt)
binavgslt = (binflatslt[:-1] + binflatslt[1:])/2
# print(binavgslt)


# ### Try to calculate volume from the temperature and average depth

# In[7]:


# get the first value (LaUH1TH) of the layers in T
diags_layers0 = diags_layers[0]
print(diags_layers0.shape)

# get the depth array
depth = rdmds(dirrun + "Depth",-1)

# read in dxg and dyg
dxg = rdmds(dirrun + "DXG",-1)
dyg = rdmds(dirrun + "DYG",-1)
drc = rdmds(dirrun + "DRC",-1)


# In[8]:


# copy the following but make it SLT
fig = plt.figure()
labels = np.array(['LaUH3SLT ' ,'LaVH3SLT '])

volume_transp_u_slt = diags_layers[4] * dyg[None,:,:]   # m^2/s * m
u_sum_slt = volume_transp_u_slt.sum(axis=(1,2))
plt.plot(binavgslt,u_sum_slt,label=labels[0])

volume_transp_v_slt = diags_layers[5] * dxg[None,:,:]   # m^2/s * m
v_sum_slt = volume_transp_v_slt.sum(axis=(1,2))
plt.plot(binavgslt,v_sum_slt,label=labels[1])

# plotting these together
uv_sum = u_sum_slt + v_sum_slt
plt.plot(binavgslt,uv_sum,label="sum u+v")
    
plt.xlabel("SLT centers")
plt.ylabel("velocity*depth, m^3 s^-1")

plt.legend(loc="best")


# In[15]:


fig = plt.figure()
labels = np.array(['LaUH1TH ' ,'LaVH1TH '])

print(diags_layers[0].shape)
volume_transp_u_th = diags_layers[0] * dyg[None,:,:]   # m^2/s * m
u_sum_th = volume_transp_u_th.sum(axis=(1,2))
plt.plot(binavgth,u_sum_th,label=labels[0])

volume_transp_v_th = diags_layers[1] * dxg[None,:,:]   # m^2/s * m
v_sum_th = volume_transp_v_th.sum(axis=(1,2))
plt.plot(binavgth,v_sum_th,label=labels[1])

# plotting these together
uv_sum = u_sum_th + v_sum_th
plt.plot(binavgth,uv_sum,label="sum u+v")
plt.grid(alpha=0.5)
    
plt.xlabel("Temperature Bins")
plt.ylabel("velocity*depth, m^3 s^-1")

plt.legend(loc="best")

plt.savefig("temp_sv_transp.png",dpi=300)


# In[12]:


# plot the same with RHO centers
fig = plt.figure()
labels = np.array(['LaUH2RHO ' ,'LaVH2RHO '])

volume_transp_u = diags_layers[2] * dyg[None,:,:]   # m^2/s * m
u_sum = volume_transp_u.sum(axis=(1,2))
plt.plot(binavgrho,u_sum,label=labels[0])

volume_transp_v = diags_layers[3] * dxg[None,:,:]   # m^2/s * m
v_sum = volume_transp_v.sum(axis=(1,2))
plt.plot(binavgrho,v_sum,label=labels[1])

# plotting these together
uv_sum = u_sum + v_sum
plt.plot(binavgrho,uv_sum,label="sum u+v")
    
plt.xlabel("RHO centers")
plt.ylabel("velocity*depth, m^3 s^-1")

plt.legend(loc="best")


# ### TS diagram?

# In[12]:


# THIS IS WRONG
# try to add to mesh_u where from 
avg_salt = np.mean(binflatslt)
avg_temp = np.mean(binflatth)

fig = plt.figure(figsize=(3,2))
# ax1 = plt.subplot(1,2,1)
# plt.scatter(binavgth,np.full(40,35), c=u_sum_th, cmap='viridis', s=10)
# plt.scatter(np.full(40,3), binavgslt, c=u_sum_slt, cmap='viridis', s=10)

ax2 = plt.subplot(1,3,1)
plt.pcolormesh(T)

ax3 = plt.subplot(1,3,2)
plt.pcolormesh(S)

ax4 = plt.subplot(1,3,3)
plt.pcolormesh((T+S)/2)


# In[13]:


T_pick = 6
# make quivers
u = volume_transp_u_th[T_pick, :, :]
v = volume_transp_v_th[T_pick, :, :]
x = np.arange(0, 20)  # Adjust if your grid spacing is different
y = np.arange(0, 16)  # Adjust if your grid spacing is different
X, Y = np.meshgrid(x, y)


# In[14]:


# try to make a quiver plot and an example transect
volume_transp_u_slt.shape
volume_transp_u_th.shape

# cmap
cmap = plt.get_cmap('RdBu_r',11).copy()
# cmap.set_under(color='white')

# try to plot a temperature at center 2.5 (this has a lot of transport)
fig, (ax1, ax2) = plt.subplots(1, 2)
T_pick = 6
norm = TwoSlopeNorm(vmin=volume_transp_u_th[T_pick, :, :].min(), vcenter=0, vmax=volume_transp_u_th[T_pick, :, :].max())   # for the colorbar
im1 = ax1.imshow(volume_transp_u_th[T_pick, :, :], norm=norm, cmap=cmap, origin='lower')
ax1.set_title(f'u-Transport at Bin of T {binavgth[T_pick]}')
plt.colorbar(im1, ax=ax1, label='u-Transport')
# add quivers - these are from u itself
ax1.quiver(X, Y, u, v, pivot='middle', headwidth=5, headlength=5,alpha=0.5)

im2 = ax2.imshow(volume_transp_v_th[T_pick, :, :], norm=norm, cmap=cmap, origin='lower')
ax2.set_title(f'v-Transport at Bin of T {binavgth[T_pick]}')
plt.colorbar(im2, ax=ax2, label='v-Transport')
ax2.quiver(X, Y, u, v, pivot='middle', headwidth=5, headlength=5,alpha=0.5)

plt.savefig("uv_transp_Tbin.png",dpi=300)


# In[16]:


# Vertical Profile at a specific grid point example
grid_x = 11
grid_y = 10

fig, (ax1, ax2) = plt.subplots(1, 2)
# first subplot do TH
u_bins_profile = volume_transp_u_th[:, grid_x, grid_y]
v_bins_profile = volume_transp_v_th[:, grid_x, grid_y]
ax1.plot(binavgth, u_bins_profile, label='u-Transport', marker='o')
ax1.plot(binavgth, v_bins_profile, label='v-Transport', marker='x')
ax1.set_ylabel('Volume Transport')
ax1.set_xlabel('Temperature Bin')
ax1.set_title(f'Vertical Profile at Grid Point ({grid_x}, {grid_y})')
ax1.legend()
ax1.grid()

# second subplot do SLT
u_bins_profile = volume_transp_u_slt[:, grid_x, grid_y]
v_bins_profile = volume_transp_v_slt[:, grid_x, grid_y]
ax2.plot(binavgth, u_bins_profile, label='u-Transport', marker='o')
ax2.plot(binavgth, v_bins_profile, label='v-Transport', marker='x')
ax2.set_ylabel('Volume Transport')
ax2.set_xlabel('Salinity Bin')
ax2.set_title(f'Vertical Profile at Grid Point ({grid_x}, {grid_y})')
ax2.legend()
ax2.grid()


# ### Try to calculate M

# In[38]:


# M is the total volume flux through a surface Bt
# let's pick a surface 9 and 14 horizontally - this will be the first index in our 16x20 grid

surf1 = 12
surf2 = 10

# here I should only be interested in vertical flow v
# redefine for looking at them here
th_v_1 = volume_transp_v_th[:,surf1,:]
slt_v_1 = volume_transp_v_slt[:,surf1,:]
th_v_2 = volume_transp_v_th[:,surf2,:]
slt_v_2 = volume_transp_v_slt[:,surf2,:]


# In[39]:


# can I create transects for these two locations? where is depth
depth1 = depth[surf1,:]
depth2 = depth[surf2,:]

# I think we want to create a similar plot to the ones I made with An
fig = plt.figure()

ax = plt.subplot(2,1,1)
plt.plot(binavgth,np.sum(th_v_1,axis=1),label="v TH transport, surf1")
plt.plot(binavgth,np.sum(th_v_2,axis=1),label="v TH transport, surf2")
uv_sum_th = np.sum(th_v_1,axis=1) + np.sum(th_v_2,axis=1)
plt.plot(binavgth,uv_sum_th,label="sum u+v")
plt.xlabel("TH centers")
plt.ylabel("velocity*depth, m^3 s^-1")
plt.legend(loc="best")

ax2 = plt.subplot(2,1,2)
plt.plot(binavgslt,np.sum(slt_v_1,axis=1),label="v SLT transport, surf1")
plt.plot(binavgslt,np.sum(slt_v_2,axis=1),label="v SLT transport, surf2")
uv_sum_slt = np.sum(slt_v_1,axis=1) + np.sum(slt_v_2,axis=1)
plt.plot(binavgslt,uv_sum_slt,label="sum u+v")
plt.xlabel("SLT centers")
plt.ylabel("velocity*depth, m^3 s^-1")
plt.legend(loc="best")


# In[40]:


print(sum(np.sum(slt_v_1,axis=1)))
sum(np.sum(slt_v_2,axis=1))


# In[41]:


#WRONG!

# see if these are being conserved
total_volume_temp_y9 = np.sum(th_v_1, axis=1)
total_volume_temp_y14 = np.sum(th_v_2, axis=1)
total_volume_sal_y9 = np.sum(slt_v_1, axis=1)
total_volume_sal_y14 = np.sum(slt_v_2, axis=1)

# difference
diff_temp = total_volume_temp_y9 - total_volume_temp_y14
diff_sal = total_volume_sal_y9 - total_volume_sal_y14

# see if there is a threshold for 0
threshold = 0.01   # 1%
conserved_temp = np.all(np.abs(diff_temp) < threshold)
conserved_sal = np.all(np.abs(diff_sal) < threshold)

print(f"Volume conservation for temperature bins: {conserved_temp}")
print(f"Volume conservation for salinity bins: {conserved_sal}")

#visualize
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(binavgth, diff_temp)
plt.title('Differences in Volume Transport for Temperature Bins')
plt.xlabel('Temperature Bins')
plt.ylabel('Difference in Volume/s (m^3/s)')

plt.subplot(1, 2, 2)
plt.plot(binavgslt, diff_sal)
plt.title('Differences in Volume Transport for Salinity Bins')
plt.xlabel('Salinity Bins')
plt.ylabel('Difference in Volume/s (m^3/s)')
plt.tight_layout()
plt.show()


# In[45]:


# maybe I'm doing this wrong!
# From the Hieronymus paper: "M is the volume flow having a salinity less than S and 
# a temperature less than T that exits through the interior control surface B"

def M_through_surface(layer, T_threshold, S_threshold, volume_transp_v_th, volume_transp_v_slt, binavgth,binavgslt):
    '''
    Inputs: 
        val: T or S
        layer: surface B (in y dimension)
    '''
    # temperature and salinity less than some threshold; here let's say 2deg and 35sal
    T_threshold = 7
    S_threshold = 36
    
    # Find the indices where temperature and salinity are less than the given thresholds
    #both_indices = np.where((binavgth < T_threshold) & (binavgslt < S_threshold))[0]
    temp_indices = np.where(binavgth < T_threshold)[0]
    sal_indices = np.where(binavgslt < S_threshold)[0]
    
    # Extract volume flows from the bins for surface B (let's say y=9 for simplicity)
    volume_temp_B = volume_transp_v_th[temp_indices, layer, :]
    volume_sal_B = volume_transp_v_slt[sal_indices, layer, :]
    
    # Sum the volume flows to get M
    M_temp = np.sum(volume_temp_B)
    M_sal = np.sum(volume_sal_B)

    return(M_temp,M_sal)

l1 = 10
M_temp_9,M_sal_9 = M_through_surface(l1,2,35,volume_transp_v_th, volume_transp_v_slt, binavgth,binavgslt)
print("Volume below temp threshold: ",M_temp_9, "m^3 per s through y =",l1)
print("Volume below salt threshold: ",M_sal_9, "m^3 per s through y =",l1)
print()

l2 = 12
M_temp_12,M_sal_12 = M_through_surface(l2,2,35,volume_transp_v_th, volume_transp_v_slt, binavgth,binavgslt)
print("Volume below temp threshold: ",M_temp_12, "m^3 per s through y =",l2)
print("Volume below salt threshold: ",M_sal_12, "m^3 per s through y =",l2)


# In[47]:


# is there a way we can look at total volume through this? Why is this not closed?

print(dxg[10,:])  # m^2/s * m
print(dxg[12,:])  # m^2/s * m


# In[69]:


sum(diags_layers[4][:,10,:] * dyg[None,10,:]).sum()


# In[68]:


sum(diags_layers[4][:,12,:] * dyg[None,12,:]).sum()


# ### Run with LaHC_T_S

# In[ ]:





# ### peek at Helen's 3d results

# In[29]:


dir_helen_out = "/scratch2/pillarh/aste_90x150x60/run_c67w_layers_budget_nlayersorig_nlayers24_10ts/diags/LAYERS/"


# In[33]:


bins_test = rdmds(dir_helen_out + "layers_3d_TSjoint_set3", 2)


# In[34]:


bins_test.shape


# In[ ]:




