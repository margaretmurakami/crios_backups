# import the packages
import numpy as np
import netCDF4 as nc
import time
import xarray as xr
import gsw
import xarray as xr
import shapely.geometry as sg
import matplotlib.pyplot as plt
from shapely import Point

################################################################################################
# define the polygons for categorizing
# add the polygons and the points
smin = 31 - (0.01 * 31)    #salt_ctrl_subregR.min - (0.01 * salt_ctrl_subregR.min)
smax = 38. + (0.01 * 38.)    #salt_ctrl_subregR.max + (0.01 * salt_ctrl_subregR.max)
tmin = -3 + (0.1 * -3)       #temp_ctrl_subregR.min - (0.1 * temp_ctrl_subregR.max)
tmax = 3.3 + (0.1 * 3.3)       #temp_ctrl_subregR.max + (0.1 * temp_ctrl_subregR.max)
print('tmin, tmax, smin, smax sizes=,', tmin, tmax, smin, smax)
# Calculate how many gridcells we need in the x and y dimensions
xdim = 30
ydim = 30
# Create empty grid of zeros
dens2 = np.zeros((ydim,xdim))
# Create temp and salt vectors of appropiate dimensions
ti1 = np.linspace(-3,3.3,ydim)
si1 = np.linspace(31,38,xdim)
Freez_temp = gsw.CT_freezing(si1,0,0)

Si, Ti = np.meshgrid(si1, ti1, sparse=False, indexing='ij')
# Loop to fill in grid with densities
for j in range(0,int(ydim)):
    for i in range(0, int(xdim)):
        #print(si[i],ti[j])
        dens2[j,i]=gsw.rho(si1[i],ti1[j],0)
        # Substract 1000 to convert to sigma-0
dens2 = dens2 - 1000

# convert to practical/potential
long = 73.5089
lat = -66.8245
ti = gsw.pt_from_CT(si1,ti1)
si = gsw.SP_from_SA(si1,0,long,lat)

# create the polygons for the next plot
fig = plt.figure(figsize=(1,1))
ax = fig.add_subplot(1, 1, 1)

# add aabw values
cs = ax.contourf(si, ti, dens2, levels=[27.82,31],colors="black", zorder=1,alpha=0,linestyles='-.')
cl=plt.clabel(cs,fontsize=10,inline=False,fmt="%.2f")
# get the dens2ity vertices
p = cs.collections[0].get_paths()[0]
v = p.vertices
# get the TS vertices
s = np.array([34.5,36,36,34.5,34.5])
t = np.array([-3,-3,0.1,0.1,-3])
b = np.array([[a,b] for a,b in zip(s,t)])
# # find intersection and plot
a = sg.Polygon(v)
b = sg.Polygon(b)
ft = Freez_temp
si2 = si.copy()
si2 = np.append(si2,[max(si2),min(si2),min(si2)])
ft = np.append(ft,[0.1,0.1,max(ft)])
c = np.array([[a,b] for a,b in zip(si2,ft)])
c = sg.Polygon(c)
d = a.intersection(b)
aabw = c.intersection(d)

# find the winter water values
cs = ax.contourf(si, ti, dens2, levels=[27.55,27.73],colors="black", zorder=1,alpha=0,linestyles='-.')
p = cs.collections[0].get_paths()[0]
v = p.vertices
ft = Freez_temp
si2 = si.copy()
si2 = np.append(si2,[max(si2),min(si2),min(si2)])
ft = np.append(ft,[-1.5,-1.5,max(ft)])
b = np.array([[a,b] for a,b in zip(si2,ft)])
a = sg.Polygon(v)
b = sg.Polygon(b)
ww = a.intersection(b)

# find the mcdw values
cs = ax.contourf(si, ti, dens2, levels=[27.73,27.82],colors="black", zorder=1,alpha=0,linestyles='-.')
p = cs.collections[0].get_paths()[0]
v = p.vertices
ft = Freez_temp
si2 = si.copy()
si2 = np.append(si2,[max(si2),min(si2),min(si2)])
ft = np.append(ft,[1.5,1.5,max(ft)])
b = np.array([[a,b] for a,b in zip(si2,ft)])
a = sg.Polygon(v)
b = sg.Polygon(b)
mcdw = a.intersection(b)

# find the dsw values
cs = ax.contourf(si, ti, dens2, levels=[27.82,31],colors="black", zorder=1,alpha=0,linestyles='-.')
p = cs.collections[0].get_paths()[0]
v = p.vertices
a = sg.Polygon(v)
# ftmax = Freez_temp+0.1
# ftmin = Freez_temp-0.5
# ft = np.append(ftmax,np.flip(ftmin))
# si2 = si.copy()
# si2 = np.append(si2,np.flip(si2))
# b = np.array([[a,b] for a,b in zip(si2,ft)])
sx = np.array([35,35,34.5,34.5])
sy = np.array([-3,-1.7,-1.7,-3])
ss = np.array([[a,b] for a,b in zip(sx,sy)])
b = sg.Polygon(ss)             # second shape in salinity
#dsw = b.intersection(c)
dsw = b.intersection(a)

# find the aasw values
cs = ax.contourf(si, ti, dens2, levels=[24,27.73],colors="black", zorder=1,alpha=0,linestyles='-.')
p = cs.collections[0].get_paths()[0]
v = p.vertices
a = sg.Polygon(v)              # first shape in dens2ity
sx = np.array([34.5,34.5,31,31])
sy = np.array([-3,3.5,3.5,-3])
ss = np.array([[a,b] for a,b in zip(sx,sy)])
b = sg.Polygon(ss)             # second shape in salinity
ft = Freez_temp
ft = np.append(ft,[3.5,3.5])
si2 = si.copy()
si2 = np.append(si2,[35,31])
ta = np.array([[a,b] for a,b in zip(si2,ft)])
c = sg.Polygon(ta)              # third shape in temperature
d = b.intersection(c)
aasw = d.intersection(a)

# find the msw values
cs = ax.contourf(si, ti, dens2, levels=[27.82,31],colors="black", zorder=1,alpha=0,linestyles='-.')
p = cs.collections[0].get_paths()[0]
v = p.vertices
a = sg.Polygon(v)
ft = Freez_temp + 0.1
ft = np.append(ft,[-0.4,-0.4])
si2 = si.copy()
si2 = np.append(si2,[35,31])
ta = np.array([[a,b] for a,b in zip(si2,ft)])
c = sg.Polygon(ta)              # third shape in temperature
#dsw = b.intersection(c)
msw = c.intersection(a)
msw = msw.difference(dsw)

# add ISW values
cs = ax.contourf(si, ti, dens2, levels=[25,27.82],colors="black", zorder=1,alpha=0,linestyles='-.')
p = cs.collections[0].get_paths()[0]
v = p.vertices
a = sg.Polygon(v)
ft = Freez_temp# - 0.05
si2 = si.copy()
si2 = np.append(si2,[max(si2),min(si2),min(si2)])
ft = np.append(ft,[-3,-3,max(ft)])
b = np.array([[a,b] for a,b in zip(si2,ft)])
b = sg.Polygon(b)
isw = b.intersection(a)
#isw = isw.difference(dsw)
isw = isw.difference(aabw)
#ucdw = ucdw.difference(mcdw)
aasw = aasw.difference(ww)
#aasw = aasw.difference(ucdw)

# find the aaiw values
cs = ax.contourf(si, ti, dens2, levels=[27.2,27.4],colors="black", zorder=1,alpha=0,linestyles='-.')
p = cs.collections[0].get_paths()[0]
v = p.vertices
a = sg.Polygon(v)              # first shape in dens2ity
sx = np.array([34.6,34.6,32,32])
sy = np.array([2,3.5,3.5,2])
ss = np.array([[a,b] for a,b in zip(sx,sy)])
b = sg.Polygon(ss)             # second shape in salinity
ft = Freez_temp
ft = np.append(ft,[3.5,3.5])
si2 = si.copy()
si2 = np.append(si2,[35,31])
ta = np.array([[a,b] for a,b in zip(si2,ft)])
c = sg.Polygon(ta)              # third shape in temperature
d = b.intersection(c)
aaiw = d.intersection(a)



################################################################################################
# define the function for categorizing
# coding 0 AABW, 1 MCDW, 2 ISW, 3 DSW, 4 AASW, 5 WW, 6 mSW, 7 beached
# use potential T and practical S, 
def wmt_categorize2(temp,salt,depth,aabw,mcdw,isw,dsw,aasw,ww,msw):
#     print(temp.shape,Freez_temp.shape,salt.shape,enddens_allvals.shape,depth.shape,dens.shape)
    mass = np.array([],dtype=int)

    points = np.array([Point(s, t) for s, t in zip(salt, temp)])

    for t,s,d in zip(temp,salt,depth):

        # beached
        if np.isnan(t) or np.isnan(s):
            mass = np.append(mass,7)
        else:
            point = Point(s,t)
            
            # aabw
            if d<-1000 and aabw.contains(point):
                mass = np.append(mass,0)
            elif d>-1000 and msw.contains(point):
                mass = np.append(mass,6)
                
            # mcdw
            elif mcdw.contains(point): # or aabw.contains(point):                             # modified tf
                mass = np.append(mass,1)
    
            # dsw
            elif dsw.contains(point):
                #elif s>=34.5 and t>=(tf-0.5) and t<=(tf+0.1) and rho>=27.68:
                mass = np.append(mass,3)
                
            # isw (also defined in yoon)
            elif isw.contains(point):                                                    # modified tf
                mass = np.append(mass,2)
            # other water masses defined from portela
            # aasw
            elif aasw.contains(point):                                          # modified s
                mass = np.append(mass,4)
            # ww
            elif ww.contains(point):
                mass = np.append(mass,5)
            
            # other shelf waters
            elif aabw.contains(point):
                mass = np.append(mass,6)
            
            else:
                mass = np.append(mass,1)
        
    return(mass)


################################################################################################
# load the file from filename = "/scratch/mmurakami/WAOM/drifter_data_all_withdepth.nc"
# Load your input NetCDF file
input_filename = "/scratch/mmurakami/WAOM/drifter_data_all_withdepth.nc"  # Change to actual file
output_filename = "/scratch/mmurakami/WAOM/categorized.nc"

with nc.Dataset(input_filename, "r") as ds:
    temp = ds.variables["temp"][:]  # Adjust variable names
    salt = ds.variables["salt"][:]
    depth = ds.variables["depth"][:]

categorized_arr = np.empty_like(temp, dtype=int)


# run the function with time
# Track timing
start_time = time.time()
num_rows = temp.shape[0]

for i in range(num_rows):
    if i % 1000 == 0:  # Print status every 10 iterations
        elapsed = time.time() - start_time
        avg_time_per_iter = elapsed / (i + 1) if i > 0 else 0
        est_total_time = avg_time_per_iter * num_rows
        est_remaining = est_total_time - elapsed
        print(f"Processing row {i+1}/{num_rows} | Elapsed: {elapsed:.2f}s | Estimated Remaining: {est_remaining:.2f}s")

    categorized_arr[i] = wmt_categorize2(temp[i], salt[i], depth[i], aabw, mcdw, isw, dsw, aasw, ww, msw)

# Save to NetCDF
with nc.Dataset(output_filename, "w", format="NETCDF4") as ds_out:
    with nc.Dataset(input_filename, "r") as ds_in:
        for dim_name, dim in ds_in.dimensions.items():
            ds_out.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)

    categorized_var = ds_out.createVariable("categorized", "i4", ds_in.variables["temp"].dimensions)
    categorized_var[:] = categorized_arr
    categorized_var.units = "category index"
    categorized_var.description = "Categorized water masses"

print(f"Categorization complete! Saved to {output_filename}")
