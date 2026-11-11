from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.constants import c
import numpy as np
import pandas as pd
from pathlib import Path
from config_utils import load_config

# Load pipeline configuration
config = load_config()
settings = config["post_imaging"]["barycentric_correct"]

# Target redshift
z = config["target"]["redshift"]

# Observation catalogue
obsids_csv = Path(settings["obsids_csv"])

c_value = c.to_value(u.km/u.s)
HI_freq = 1420.40575177/(1 + z)
    
df = pd.read_csv(obsids_csv)
length = len(df)
velcorr = []
helcorr = []
id = []
freq = []
for r in df.itertuples(index=False):
    obsid = r.obsid
    RA    = r.RA
    DEC   = r.DEC
    id.append(obsid)
    keck = EarthLocation.from_geodetic(lat=-26.70331940*u.deg, lon=116.67081524*u.deg, height=310*u.m)   #310
    sc = SkyCoord(ra=RA, dec=DEC, unit=(u.hourangle, u.deg))       #ra=134.061375, dec=2.399888889
    barycorr = sc.radial_velocity_correction(obstime=Time(obsid, format='gps', scale='utc'), location=keck)
    freqcorr = (barycorr.to_value(u.km/u.s)/c_value) * HI_freq
    velcorr.append(barycorr.to_value(u.km/u.s)) #km/s
    freq.append(freqcorr) #MHz
    #print('barycentric correction: ', barycorr.to(u.km/u.s))
    heliocorr = sc.radial_velocity_correction('heliocentric', obstime=Time(obsid, format='gps', scale='utc'), location=keck) 
    helcorr.append(heliocorr.to_value(u.km/u.s)) #km/s     
    #print('heliocentric correction: ', heliocorr.to(u.km/u.s))
obs_arr = np.array(id)
vel_arr = np.array(velcorr)
hel_arr = np.array(helcorr)
freq_arr = np.array(freq)

df = pd.DataFrame({"obsid": obs_arr, "barycentric correction(km/s)": vel_arr, "frequency correction(MHz)": freq_arr})
output_file = Path(settings["output_file"])
output_file.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False)