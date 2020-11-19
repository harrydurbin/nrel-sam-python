The script 'main.py' is used to create power simulations in NREL SAM and generator hourly shape output and perform some analysis on the results.
Harry Durbin
October 2020

Requirements for running:
+ python must be 64 bit
+ pip install nrel-pysam
+ create a project in SAM and export as json file into folder
+ set the SAM_EXPORT_JSON file path to the json file you just export

Steps:

1) create SAM model (location isn't important--just configure system parameters in SAM)

2) create a new folder where you would like to store the generated data

2) select tab in SAM model, right click, and select export as json, and select the folder you created

3) edit the main.py script
   - set the path for the json file you created
   - assign if it is a solar or wind model
   - edit parameters like location and hub height (if wind) 
Possible parameters are: lat,lon,address,year,yearstart,yearend (and hubheight if a wind model)
 - year must be between 1998 and 2019 for solar, 2007 and 2014 for wind
 - lat and lon can be either decimal or in degree minute second format (e.g. 127d02m10s)
 - lat and lon is not required if an address is provided
 - for wind, hub height in meters allows allows some increments (e.g. 40,60,80,100)
 - if no year or year span provided, model assumes you want data for all available years
e.g.
Solar_Model_Example = SAMSolarModelSimulator(lat=127.20,lon=-97,year-2012):
Wind_Model_Example = SAMWindModelSimulator(address='Walla Walla, WA',hubheight=60)
If wanting to make additional adjustments to SAM variables, edit the dictionary variable in SAMSolarModelSimulator.py or SAMWindModelSimulator.py
to replace certain model parameters (e.g. run the model with a different 'tilt' if a solar model)

4) save and close main.py and run the python script(see below for instructions to run in terminal)
   - open terminal window
   - write 'python '
   - click and drag the main.py python script into the terminal window
   - hit enter to run in terminal
   - new folders will appear in the same subfolder as the json export data to contain weather and output files




