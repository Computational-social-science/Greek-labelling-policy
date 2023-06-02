import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from scipy.interpolate import make_interp_spline

shapefile = 'bokeh-app/data/countries_110m/ne_110m_admin_0_countries.shp'

#Read shapefile using Geopandas
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]

#Rename columns.
gdf.columns = ['country', 'country_code', 'geometry']
gdf = gdf.drop(gdf.index[159]) # The width of the map

# #Drop row corresponding to 'Antarctica'
datafile = 'bokeh-app/data/Geographical name_normalization.csv'
data=pd.read_csv(datafile,usecols=[0,1,2,3],names=['country','normalization','number','count'])# import data
print(data['normalization'])

#Read data to json.
data = gpd.GeoDataFrame(data)
merged = gdf.merge(data,on = 'country',how='left')

#low -> high
colorslist = ['#DDEFB1','#FEF4C0','#FEEEBA','#FEE1AA','#FDC68A','#FBBB7F','#FBBA7E','#F5A374','#EF8F6B','#EB7547','#DF5952','#DC402D','#D83428']
mycmaps = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=100)
# Word typical, pass in the personalization parameters related to the layered color.
# In this experiment, the X-axis data range is 0-100, and the interval is 1
# color_bin = [0,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4]
color_bin = np.arange(0,0.5,0.001)
# print(color_bin)
fig,ax = plt.subplots(figsize = (10,6))

merged['coords'] = merged['geometry'].apply(lambda x: x.representative_point().coords[0])

merged.plot(
    column = 'normalization',
    scheme = 'userdefined',
    classification_kwds = {'bins':color_bin},
    cmap = mycmaps,
    edgecolor = 'gray',
    linewidth = 0.5,
    ax = ax,
    missing_kwds={"color":"lightgray","edgecolor": "gray"}
)

plt.xlim(-182,182)
plt.ylim(-58,86)

plt.axis('off')  # Get rid of the axes
# background color
fig.set_facecolor("#EFF5FA")
# Colorbar label
plt.text(-113,-48,'>0.5',family = 'Times New Roman',fontsize = 7)
plt.text(-166,-48,'0',family = 'Times New Roman',fontsize = 7)
plt.text(-140,-48,'0.25',family = 'Times New Roman',fontsize = 7)
plt.text(-160,-52,'Geographical name (s.d.)',family = 'Times New Roman',fontsize = 7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# colorbar
sm = plt.cm.ScalarMappable(cmap=mycmaps)
position=fig.add_axes([0.167, 0.185, 0.12, 0.02])# position[xmin,xmax,ymin,ymax]
cb=plt.colorbar(sm,cax=position,orientation='horizontal',drawedges=False)#The direction of colorbar
cb.outline.set_visible(False)
cb.set_ticks([])  # Get rid of the scale

# Box figure
ax1 = fig.add_axes([0.162,0.165,0.128,0.11])# position[xmin,xmax,ymin,ymax]
# Take the data on the interval (0,65]
# print(data['normalization'][1:152])
f = ax1.boxplot(
    # The drawing data
    x = data['normalization'][5:151],
    vert = False,
    widths=0.18,
    patch_artist=True,
    boxprops = {'color':'gray','facecolor':'white','linewidth':'0.5'},
    showcaps=False,
    flierprops = {'marker':'o','markerfacecolor':'black','color':'black','markersize':'0.5'},
    medianprops= {'linestyle':'-','color':'gray','linewidth':'0.5'},
    whiskerprops={'color':'gray','linewidth':'0.5'}
    )
ax1.axis('off')  # Get rid of the scale

x = data['number'][0:10]
y = data['count'][0:10]
# Use scipy. Interpolate. Spline to fit the curve
x_new = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between x.min and x.max
y_smooth = make_interp_spline(x,y)(x_new)

# Normal distribution diagram
ax2 = fig.add_axes([0.167,0.235,0.12,0.09])# position[xmin,xmax,ymin,ymax]
ax2.set(xlim=(0,max(x_new)), ylim=(-0.3,max(y_smooth)+0.1), autoscale_on=False)
# Color mapping fills the area under the curve
fill_bins = np.arange(0,101,1)
a = np.array([color_bin,color_bin])
ax2.imshow(a, interpolation='bicubic', extent=(0,max(x_new), -0.3,max(y_smooth)),cmap=mycmaps,aspect='auto')
# The background color fills part of the area on the curve
ax2.fill_between(x_new,y_smooth,max(y_smooth)+0.1,color='#EFF5FA')
ax2.plot(x_new,y_smooth,color='gray',linestyle='-',linewidth=0.5)
ax2.axis('off')  # Get rid of the scale

# plt.show()
plt.savefig('picture/result_mycmaps_dpi=150.jpg', bbox_inches='tight', pad_inches = 0,dpi=150)
plt.savefig('picture/result_mycmaps_dpi=150.png', bbox_inches='tight', pad_inches = 0,dpi=150)
plt.savefig('picture/result_mycmaps_dpi=150.tiff', bbox_inches='tight', pad_inches = 0,dpi=150)

fig,ax = plt.subplots(figsize = (10,6))
merged.plot(
    column = 'normalization',
    scheme = 'userdefined',
    classification_kwds ={'bins':color_bin},
    cmap = mycmaps,
    edgecolor = 'gray',
    linewidth = 0.5,
    ax = ax,
    missing_kwds={"color":"lightgray","edgecolor": "gray"}
)
plt.xlim(-182,182)
plt.ylim(-58,86)

plt.axis('off')  # Get rid of the scale

# background color
fig.set_facecolor("#EFF5FA")
# Colorbar label
plt.text(-113,-48,'>0.5',family = 'Times New Roman',fontsize = 7)
plt.text(-166,-48,'0',family = 'Times New Roman',fontsize = 7)
plt.text(-140,-48,'0.25',family = 'Times New Roman',fontsize = 7)
plt.text(-160,-52,'Geographical name (s.d.)',family = 'Times New Roman',fontsize = 7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# colorbar
sm = plt.cm.ScalarMappable(cmap=mycmaps)
position=fig.add_axes([0.167, 0.185, 0.12, 0.02])# position[xmin,xmax,ymin,ymax]
cb=plt.colorbar(sm,cax=position,orientation='horizontal',drawedges=False)#The direction of colorbar
cb.outline.set_visible(False)
cb.set_ticks([])  # Get rid of the scale

# Box figure
ax1 = fig.add_axes([0.162,0.165,0.128,0.11])# position[xmin,xmax,ymin,ymax]
# Take the data on the interval (0,65]
# print(data['normalization'][1:152])
f = ax1.boxplot(
    # The drawing data
    x = data['normalization'][5:151],
    vert = False,
    widths=0.18,
    patch_artist=True,
    boxprops = {'color':'gray','facecolor':'white','linewidth':'0.5'},
    showcaps=False,
    flierprops = {'marker':'o','markerfacecolor':'black','color':'black','markersize':'0.5'},
    medianprops= {'linestyle':'-','color':'gray','linewidth':'0.5'},
    whiskerprops={'color':'gray','linewidth':'0.5'}
    )
ax1.axis('off')  # Get rid of the scale

x = data['number'][0:10]
y = data['count'][0:10]
# Use scipy. Interpolate. Spline to fit the curve
x_new = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between x.min and x.max
y_smooth = make_interp_spline(x,y)(x_new)

# Normal distribution diagram
ax2 = fig.add_axes([0.167,0.235,0.12,0.09])# position[xmin,xmax,ymin,ymax]
ax2.set(xlim=(0,max(x_new)), ylim=(-0.3,max(y_smooth)+0.1), autoscale_on=False)
# Color mapping fills the area under the curve
fill_bins = np.arange(0,101,1)
a = np.array([color_bin,color_bin])
ax2.imshow(a, interpolation='bicubic', extent=(0,max(x_new), -0.3,max(y_smooth)),cmap=mycmaps,aspect='auto')
# The background color fills part of the area on the curve
ax2.fill_between(x_new,y_smooth,max(y_smooth),color='#EFF5FA')
ax2.plot(x_new,y_smooth,color='gray',linestyle='-',linewidth=0.5)
ax2.axis('off')  # Get rid of the scale

plt.savefig('picture/result_mycmaps_dpi=300.jpg', bbox_inches='tight',pad_inches = 0,dpi=300)
plt.savefig('picture/result_mycmaps_dpi=300.png', bbox_inches='tight', pad_inches = 0,dpi=300)
plt.savefig('picture/result_mycmaps_dpi=300.tiff', bbox_inches='tight', pad_inches = 0,dpi=300)

fig,ax = plt.subplots(figsize = (10,6))

merged.plot(
    column = 'normalization',
    scheme = 'userdefined',
    classification_kwds = {'bins':color_bin},
    cmap = mycmaps,
    edgecolor = 'gray',
    linewidth = 0.5,
    ax = ax,
    missing_kwds={"color":"lightgray","edgecolor": "gray"}
)
plt.xlim(-182,182)
plt.ylim(-58,86)

plt.axis('off')  # Get rid of the axes

# background color
fig.set_facecolor("#EFF5FA")
# Colorbar label
plt.text(-113,-48,'>0.5',family = 'Times New Roman',fontsize = 7)
plt.text(-166,-48,'0',family = 'Times New Roman',fontsize = 7)
plt.text(-140,-48,'0.25',family = 'Times New Roman',fontsize = 7)
plt.text(-160,-52,'Geographical name (s.d.)',family = 'Times New Roman',fontsize = 7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# colorbar
sm = plt.cm.ScalarMappable(cmap=mycmaps)
position=fig.add_axes([0.167, 0.185, 0.12, 0.02])# position[xmin,xmax,ymin,ymax]
cb=plt.colorbar(sm,cax=position,orientation='horizontal',drawedges=False)#The direction of colorbar
cb.outline.set_visible(False)
cb.set_ticks([])  # Get rid of the scale

# Box figure
ax1 = fig.add_axes([0.162,0.165,0.128,0.11])# position[xmin,xmax,ymin,ymax]
# Take the data on the interval (0,65]
# print(data['normalization'][1:152])
f = ax1.boxplot(
    # The drawing data
    x = data['normalization'][5:151],
    vert = False,
    widths=0.18,
    patch_artist=True,
    boxprops = {'color':'gray','facecolor':'white','linewidth':'0.5'},
    showcaps=False,
    flierprops = {'marker':'o','markerfacecolor':'black','color':'black','markersize':'0.5'},
    medianprops= {'linestyle':'-','color':'gray','linewidth':'0.5'},
    whiskerprops={'color':'gray','linewidth':'0.5'}
    )
ax1.axis('off')  # Get rid of the scale

x = data['number'][0:10]
y = data['count'][0:10]
# Use scipy. Interpolate. Spline to fit the curve
x_new = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between x.min and x.max
y_smooth = make_interp_spline(x,y)(x_new)

# Normal distribution diagram
ax2 = fig.add_axes([0.167,0.235,0.12,0.09])# position[xmin,xmax,ymin,ymax]
ax2.set(xlim=(0,max(x_new)), ylim=(-0.3,max(y_smooth)+0.1), autoscale_on=False)
# Color mapping fills the area under the curve
fill_bins = np.arange(0,101,1)
a = np.array([color_bin,color_bin])
ax2.imshow(a, interpolation='bicubic', extent=(0,max(x_new), -0.3,max(y_smooth)),cmap=mycmaps,aspect='auto')
# The background color fills part of the area on the curve
ax2.fill_between(x_new,y_smooth,max(y_smooth),color='#EFF5FA')
ax2.plot(x_new,y_smooth,color='gray',linestyle='-',linewidth=0.5)
ax2.axis('off')  # Get rid of the scale

plt.savefig('picture/result_mycmaps_dpi=150.svg', bbox_inches='tight', pad_inches = 0,dpi=150)
plt.savefig('picture/result_mycmaps_dpi=300.svg', bbox_inches='tight', pad_inches = 0,dpi=300)