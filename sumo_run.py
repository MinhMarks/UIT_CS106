import traci
import time
import traci.constants as tc
import pytz
import datetime
from random import randrange
import pandas as pd

def getdatetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("Asia/Singapore"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME

def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

sumoCmd = ["sumo", "-c", "test.sumocfg"]
traci.start(sumoCmd)

packVehicleData = []
packTLSData = []
packBigData = []

start_time = time.time() 

while traci.simulation.getMinExpectedNumber() > 0:
    
    current_time = time.time()
    if current_time - start_time > 40:  # nếu đã chạy quá 90 giây
        break
    
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()
    trafficlights = traci.trafficlight.getIDList()

    for i in range(0, len(vehicles)):

    # Function descriptions
    # https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html
    # https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getSpeed

        vehid = vehicles[i]
        x, y = traci.vehicle.getPosition(vehicles[i])
        coord = [x, y]
        lon, lat = traci.simulation.convertGeo(x, y)
        gpscoord = [lon, lat]
        spd = round(traci.vehicle.getSpeed(vehicles[i]) * 3.6, 2)
        edge = traci.vehicle.getRoadID(vehicles[i])
        lane = traci.vehicle.getLaneID(vehicles[i])
        displacement = round(traci.vehicle.getDistance(vehicles[i]), 2)
        turnAngle = round(traci.vehicle.getAngle(vehicles[i]), 2)
        nextTLS = traci.vehicle.getNextTLS(vehicles[i])

        # Packing of all the data for export to CSV/XLSX
        vehList = [getdatetime(), vehid, coord, gpscoord, spd, edge, lane, displacement, turnAngle, nextTLS]

        print("Vehicle: ", vehicles[i], " at datetime: ", getdatetime())
        print(vehicles[i], " >>> Position: ", coord, " | GPS Position: ", gpscoord, \
                " | Speed: ", round(traci.vehicle.getSpeed(vehicles[i]) * 3.6, 2), "km/h", \
                " | EdgeID of veh: ", traci.vehicle.getRoadID(vehicles[i]), " |", \
                " LaneID of veh: ", traci.vehicle.getLaneID(vehicles[i]), " |", \
                " Distance: ", round(traci.vehicle.getDistance(vehicles[i]), 2), "m", \
                " | Vehicle orientation: ", round(traci.vehicle.getAngle(vehicles[i]), 2), "deg", \
                " | Upcoming traffic lights: ", traci.vehicle.getNextTLS(vehicles[i]))
        
        idd = traci.vehicle.getLaneID(vehicles[i])

        tlsList = []

        for k in range(0, len(trafficlights)):
            # Function descriptions
            # https://sumo.dlr.de/docs/TraCI/Traffic_Lights_Value_Retrieval.html#structure_of_compound_object_controlled_links
            # https://sumo.dlr.de/pydoc/traci._trafficlight.html#TrafficLightDomain-setRedYellowGreenState

            if idd in traci.trafficlight.getControlledLanes(trafficlights[k]):
                tflight = trafficlights[k]
                tl_state = traci.trafficlight.getRedYellowGreenState(trafficlights[k])
                tl_phase_duration = traci.trafficlight.getPhaseDuration(trafficlights[k])
                tl_lanes_controlled = traci.trafficlight.getControlledLanes(trafficlights[k])
                tl_program = traci.trafficlight.getCompleteRedYellowGreenDefinition(trafficlights[k])
                tl_next_switch = traci.trafficlight.getNextSwitch(trafficlights[k])

                # Packing of all the data for export to CSV/XLSX
                tlsList = [tflight, tl_state, tl_phase_duration, tl_lanes_controlled, tl_program, tl_next_switch]

                print(trafficlights[k], "--->", \
                    " TL state: ", traci.trafficlight.getRedYellowGreenState(trafficlights[k]), " |", \
                    " TLS phase duration: ", traci.trafficlight.getPhaseDuration(trafficlights[k]), " |", \
                    " Lanes controlled: ", traci.trafficlight.getControlledLanes(trafficlights[k]), " |", \
                    " TLS Program: ", traci.trafficlight.getCompleteRedYellowGreenDefinition(trafficlights[k]), " |" 
                    " Next TLS switch: ", traci.trafficlight.getNextSwitch(trafficlights[k]))


            #pack simulated data 
            
        packBigDataLine = flatten_list([vehList, tlsList])
        packBigData.append(packBigDataLine)
            
            # #--------MACHINE LEARNING CODES/FUNCTIONS HERE--------

            # #--------CONTROL Vehicles and Traffic Lights--------

            # #***SET FUNCTION FOR VEHICLES***
            # #REF: https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html
            # NEWSPEED = 15  # value in m/s (15 m/s = 54 km/hr)
            # if vehicles[i] == 'veh2':
            #     traci.vehicle.setSpeedMode('veh2', 0)
            #     traci.vehicle.setSpeed('veh2', NEWSPEED)


            # #***SET FUNCTION FOR TRAFFIC LIGHTS***
            # #REF: https://sumo.dlr.de/docs/TraCI/Change_Traffic_Lights_State.html
            # trafficlightduration = [5, 37, 5, 35, 6, 3]
            # trafficsignal = [
            #     "rrrrrrGGGGgGGGrr", 
            #     "yyyyyyyyrrrrrrrr", 
            #     "rrrrrGGGGGGrrrrr", 
            #     "rrrrryyyyyyrrrrr", 
            #     "GrrrrrrrrrrGGGGg"
            #     "yrrrrrrrrrryyyyy"
            # ]
            # tfl = "cluster_4260917135_5146794610_5146796930_5704674780_5704674783_5704674784_5704674787_6589790747_8370171128_8370171143_8427766841_8427766842_8427766845"

            # traci.trafficlight.setPhaseDuration(tfl, trafficlightduration[randrange(6)])
            # traci.trafficlight.setRedYellowGreenState(tfl, trafficsignal[randrange(6)])

traci.close()

# Generate Excel file
columnnames = ['dateandtime', 'vehid', 'coord', 'gpscoord', 'spd', 'edge', 'lane', 'displacement', 'turnAngle', 'nextTLS',
               'tflight', 'tl_state', 'tl_phase_duration', 'tl_lanes_controlled', 'tl_program', 'tl_next_switch']
dataset = pd.DataFrame(packBigData, index=None, columns=columnnames)
dataset.to_excel("output.xlsx", index=False)
time.sleep(5)
