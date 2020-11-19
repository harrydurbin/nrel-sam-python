'''
This script is used to create solar power simulations in NREL SAM and generator hourly shape output.
Seattle City Light
Harry Durbin
October 2020

Requirements for running:
+ python must be 64 bit
+ pip install nrel-pysam
+ create a project in SAM, set wind turbine model, and export as json file into folder
+ set the 'SAM_EXPORT_JSON' file path in the script below to the json file you just export

To run:
Enter parameters such as lat lon (or can use address instead), start and end years (or use single year instead):
Possible parameters are: lat,lon,address,year,yearstart,yearend
 - Year must be between 1998 and 2019
 - lat and lon can be either decimal or in degree minute second format (e.g. 127d02m10s)
 - hub height in meters must be either 40,60,80,100
 - if no year or year span provided, model assumes you want from 2007-2014
e.g.
Model_Results = SAMSolarModelSimulator(lat=127.20,lon=-97,year-2012):

'''

import json
import PySAM.PySSC as pssc
import pandas as pd
from sklearn import preprocessing
import numpy as np
import sys, os
import PySAM.Windpower as Windpower
import PySAM.Pvwattsv7 as PVWatts
import PySAM.Grid as Grid
from geopy.geocoders import Nominatim
import utm
import geopy

class SAMSolarModelSimulator(object):

    ## set fixed variables
    LAT = 33.2164 # weather data latitude
    LON =  -97.1292 # weather data longitude
    YEAR = 0 # weather year(s)
    YEARSTART = 1998 # weather year start
    YEAREND = 2019 # weather year end
    API_KEY = 'c0lweNf0IyZOUwmptfTgAMXvE35ym7SHuh0QxG2w' # https://developer.nrel.gov/signup/
    EMAIL = 'harry.durbin@seattle.gov' # email associated with api
    SAM_EXPORT_JSON = r"O:\POOL\PRIVATE\RPFA\2020 IRP\CPM\NREL_SAM_Python\sam_export_solar_example\untitled.json"
    ADDRESS = None

    def __init__(self,lat=LAT,
                 lon=LON,
                 year=YEAR,
                 yearstart=YEARSTART,
                 yearend=YEAREND,
                 api_key=API_KEY,
                 email=EMAIL,
                 sam_export_json=SAM_EXPORT_JSON,
                 df_all=None,
                 address=ADDRESS):
        self.lat = lat
        self.lon = lon

        if year!=0:
            self.yearstart = int(year)
            self.yearend = int(year)
        else:
            self.yearstart = int(yearstart)
            self.yearend = int(yearend)

        if address is not None:
            self.address = self.geocode(address)
            self.lat = self.address.latitude
            self.lon = self.address.longitude

        self.api_key = api_key
        self.email = email
        self.sam_export_json = sam_export_json
        self.df_all = df_all
        self.execute()

    def execute(self):
        self.lat = self.convert_coords(self.lat)
        self.lon = self.convert_coords(self.lon)
        self.get_filepath()
        self.create_years_list()
        for yr in self.years:
            print (f'Using weather year {yr}...')
            self.year = yr
            self.create_url()
            self.query_api()
            self.name_weather_file()
            self.create_weather_file()
            self.get_solar_energy_output()
        self.save_compiled_output()
        self.multiple_to_single_columns()

    @staticmethod
    def geocode(address):
        geolocator = Nominatim(user_agent="scl")
        location = geolocator.geocode(address)
        print(location.address)
        print((location.latitude, location.longitude))
        return location

    @staticmethod
    def convert_coords(dms):
        # convert to decimal value if given in format of __d__m__s.
        try:
            float(dms)
        except:
            deg,m = dms.split('d')
            m,sec = m.split('m')
            deg = int(deg)
            m = int(m)/60
            sec = int(sec.split('s')[0])/3600
            dms = deg+m+sec
        return dms

    def get_filepath(self):
        self.fpath, self.fn = os.path.split(self.sam_export_json)
        print ('==============================================================================')
        print (f'Running SAM Model Simulations for {self.lat}, {self.lon}...')
        return self.fpath , self.fn

    def create_years_list(self):
        self.years = [str(i) for i in range(self.yearstart,self.yearend+1)]

    def create_url(self):
        attr='ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
        self.url = 'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&interval=60&utc=false&full_name=Harry+Durbin&email={email}&affiliation=scl,mailing_list=false&reason=planning&api_key={api}&attributes={attr}'.\
                format(year=self.year, lat=self.lat, lon=self.lon, email=self.email,\
                       api=self.api_key,attr=attr)
#         print (self.url)

    def query_api(self):
        self.df_srw = pd.read_csv(self.url)

    def name_weather_file(self):
        self.fn_srw = 'weather_lat{lat}_lon{lon}_{year}.csv'.\
            format(lat=self.lat,lon=self.lon,year=self.year)

    def create_weather_file(self):
        path = os.path.join(self.fpath,'weather')
        try:
            os.mkdir(path)
        except:
            pass
        self.fp_srw = os.path.join(self.fpath,'weather',self.fn_srw)
        self.df_srw.to_csv(self.fp_srw,index=False)

    def get_solar_energy_output(self):
        ssc = pssc.PySSC()
        f = open(self.sam_export_json)
        self.dic = json.load(f)
        self.dic['solar_resource_file'] = self.fp_srw

        ### uncomment if you want to change any model input parameters
#         self.dic['system_capacity'] = 20000
#         self.dic['module_type'] = 0
#         self.dic['dc_ac_ratio'] = 1.3
#         self.dic['array_type'] =  2
#         self.dic['tilt'] =  35
#         self.dic['azimuth'] = 180
#         self.dic['gcr'] = 0.40
#         self.dic['losses'] = 14
#         self.dic['en_snowloss'] =  0
#         self.dic['inv_eff'] = 95

        pv_dat = pssc.dict_to_ssc_table(self.dic, "pvwattsv7")
        grid_dat = pssc.dict_to_ssc_table(self.dic, "grid")
        f.close()
        pv = PVWatts.wrap(pv_dat)
        grid = Grid.from_existing(pv)
        grid.assign(Grid.wrap(grid_dat).export())
        pv.execute()
        grid.execute()
        self.json_dict = pv.Outputs.export()
#         print(self.json_dict.keys())
        self.df_output = pd.DataFrame(self.json_dict['gen'])
        self.df_output.columns = [self.year]
        for col in self.df_output.columns:
            self.df_output[col] = preprocessing.minmax_scale(self.df_output[col].values.reshape(1, -1), feature_range=(0, 1), axis=1, copy=True).T

        if self.df_all is None:
            self.df_all = self.df_output.copy()
        else:
            self.df_all = pd.concat([self.df_all,self.df_output],axis=1)

    def save_compiled_output(self):
        if self.yearstart != self.yearend:
            fn_output = 'hourlyshape_lat{lat}_lon{lon}_{yearstart}-{yearend}.csv'.\
                format(lat=round(self.lat,2),lon=round(self.lon,2),yearstart=self.yearstart,yearend=self.yearend)
        else:
            fn_output = 'hourlyshape_lat{lat}_lon{lon}_{year}.csv'.\
                format(lat=self.lat,lon=self.lon,year=self.year)
        path = os.path.join(self.fpath,'output')
        try:
            os.mkdir(path)
        except:
            pass
        self.outputpath = os.path.join(self.fpath,'output',fn_output)
        self.df_all.to_csv(self.outputpath)
        print ('Finished getting SAM output.')

    def multiple_to_single_columns(self):
        if self.years:
            self.df_all_melted = self.df_all.melt()
            if self.yearstart != self.yearend:
                fn_output = 'hourlyshape_lat{lat}_lon{lon}_{yearstart}-{yearend}_melted.csv'.\
                    format(lat=round(self.lat,2),lon=round(self.lon,2),yearstart=self.yearstart,yearend=self.yearend)
            else:
                fn_output = 'hourlyshape_lat{lat}_lon{lon}_{year}_melted.csv'.\
                    format(lat=self.lat,lon=self.lon,year=self.year)
            self.df_all_melted.to_csv(os.path.join(self.fpath,'output',fn_output))
            # print ('Finished ')
