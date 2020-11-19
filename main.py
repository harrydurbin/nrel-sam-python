'''
This script is used to create power simulations in NREL SAM and generator hourly shape output.
Seattle City Light
Harry Durbin
October 2020

Requirements for running:
+ python must be 64 bit
+ pip install nrel-pysam
+ create a project in SAM, set wind turbine model, and export as json file into folder
+ set the 'SAM_EXPORT_JSON' file path to the json file you just export

To run:
Enter parameters such as lat lon (or can use address instead), start and end years (or use single year instead):
Possible parameters are: lat,lon,address,year,yearstart,yearend
 - Year must be between 1998 and 2019
 - lat and lon can be either decimal or in degree minute second format (e.g. 127d02m10s)
 - hub height in meters must be either 40,60,80,100
 - if no year or year span provided, model assumes you want from 2007-2014 for wind and 1998-2019 for solar
e.g.
Model_Results = SAMSolarModelSimulator(lat=127.20,lon=-97,year=2012)

Note:
The wind data is available in a grid of 5km x 5km. A change of 5km is roughly equivalent to a change in lat and lon of 0.045.
This value can be added to the site lat and lon if wanting to use other wind data near the site location.

'''

from SAMSolarModelSimulator import SAMSolarModelSimulator
from SAMWindModelSimulator import SAMWindModelSimulator
from SAMOutputAnalysis import SAMOutputAnalysis

SAM_EXPORT_JSON = r"C:\Users\NissleP\Desktop\Solar\HarryPaulSAMWindExample\stateline.json"
#SAM_EXPORT_JSON = r"C:\Users\NissleP\Desktop\Solar\HarryPaulSAMTestExample\untitled.json"
#SAM_EXPORT_JSON = r"O:\POOL\PRIVATE\RPFA\2020 IRP\CPM\NREL_SAM_Python\sam_export_solar_example\untitled.json"
#SAM_EXPORT_JSON = r"O:\POOL\PRIVATE\RPFA\2020 IRP\CPM\NREL_SAM_Python\sam_export_wind_example\wind_example.json"

if __name__ == "__main__":

    ## example - creating a solar model
    # SAM_MODEL = SAMSolarModelSimulator(address='Yakima, WA',sam_export_json=SAM_EXPORT_JSON)
    #SAM_MODEL = SAMSolarModelSimulator(lat=46.067,lon=-118.339,sam_export_json=SAM_EXPORT_JSON)
#    SAM_MODEL1 = SAMSolarModelSimulator(lat=46.067+0.045,lon=-118.339,sam_export_json=SAM_EXPORT_JSON)
#    SAM_MODEL2 = SAMSolarModelSimulator(lat=46.067-0.045,lon=-118.339,sam_export_json=SAM_EXPORT_JSON)
#    SAM_MODEL3 = SAMSolarModelSimulator(lat=46.067,lon=-118.339+0.045,sam_export_json=SAM_EXPORT_JSON)
#    SAM_MODEL4 = SAMSolarModelSimulator(lat=46.067,lon=-118.339-0.045,sam_export_json=SAM_EXPORT_JSON)

    ## example - creating a wind model
    SAM_MODEL = SAMWindModelSimulator(address='Walla Walla, WA',sam_export_json=SAM_EXPORT_JSON, hubheight=60)
    # SAM_MODEL = SAMWindModelSimulator(lat=46.067,lon=-118.339,sam_export_json=SAM_EXPORT_JSON, hubheight=40)
    # SAM_MODEL1 = SAMWindModelSimulator(lat=46.067+0.045,lon=-118.339,sam_export_json=SAM_EXPORT_JSON, hubheight=40)
    # SAM_MODEL2 = SAMWindModelSimulator(lat=46.067-0.045,lon=-118.339,sam_export_json=SAM_EXPORT_JSON, hubheight=40)
    # SAM_MODEL3 = SAMWindModelSimulator(lat=46.067,lon=-118.339+0.045,sam_export_json=SAM_EXPORT_JSON, hubheight=40)
    # SAM_MODEL4 = SAMWindModelSimulator(lat=46.067,lon=-118.339-0.045,sam_export_json=SAM_EXPORT_JSON, hubheight=40)

    ANALYSIS = SAMOutputAnalysis(SAM_MODEL)
#    ANALYSIS1 = SAMOutputAnalysis(SAM_MODEL1)
#    ANALYSIS2 = SAMOutputAnalysis(SAM_MODEL2)
#    ANALYSIS3 = SAMOutputAnalysis(SAM_MODEL3)
#    ANALYSIS4 = SAMOutputAnalysis(SAM_MODEL4)
