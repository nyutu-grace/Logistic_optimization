import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
import colorcet as cc

# Load the dataset
df = pd.read_csv('../data/processeddata/mergeddata.csv')

# Ensure 'datetime' is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# Define a function to create Datashader plots
def create_datashader_plot(df, x_col, y_col, agg_col, plot_width=800, plot_height=600, cmap=cc.fire):
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
    agg = cvs.points(df, x_col, y_col, ds.mean(agg_col))
    img = tf.shade(agg, cmap=cmap)
    return img

# Plot driver locations with driver-client distance
img = create_datashader_plot(df, 'lon', 'lat', 'driver_clientdistance')
export_image(img, 'driver_locations')

# Plot accepted and rejected actions
df_accepted = df[df['driver_action'] == 'accepted']
df_rejected = df[df['driver_action'] == 'rejected']

img_accepted = create_datashader_plot(df_accepted, 'lon', 'lat', 'driver_clientdistance', cmap=cc.kbc)
img_rejected = create_datashader_plot(df_rejected, 'lon', 'lat', 'driver_clientdistance', cmap=cc.fire)

export_image(img_accepted, 'driver_locations_accepted')
export_image(img_rejected, 'driver_locations_rejected')

print("Plots have been generated and saved as images.")
