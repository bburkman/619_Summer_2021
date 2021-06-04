import csv
import math
import numpy as np

def Feature_Engineering():
    with open('2019_Crash_1_Database.csv', 'r') as csvfile:
        Crash = list(csv.reader(csvfile))
    Header = Crash[0]
    ind = Header.index("severity_cd")
#    d = {"A":0.9, "B":0.7, "C":0.5, "D":0.3, "E":0.1}
#    d = {"A":1.0, "B":0.0, "C":0.0, "D":0.0, "E":0.0}
    d = {"A":1.0, "B":-1.0, "C":-1.0, "D":-1.0, "E":-1.0}
    Severity = [d[Crash[i][ind]] for i in range (1,len(Crash))]
    # Turn Severity into an NP vector.
    Y = np.array(Severity)
    
    # Condense different blank cells.  
    for i in range (len(Crash)):
        for j in range (len(Crash[i])):
            if Crash[i][j] == ' ':
                Crash[i][j] = ''
            if Crash[i][j] == '  ':
                Crash[i][j] = ''

    # Round ages to nearest 10.
    for col in ['dr_age_1','dr_age_2']:
        s = Header.index(col)
        for i in range (1,len(Crash)):
            if Crash[i][s].isnumeric():
                Crash[i][s] = str(math.floor(int(Crash[i][s])/10)*10)

    # Round times down to hour.
    s = Header.index('crash_time')
    for i in range (1, len(Crash)):
        if Crash[i][s] != '':
            a = Crash[i][s].find(':')
            Crash[i][s] = Crash[i][s][:a]

    # Take out the uncommented columns.
    for head in [
        'route', 
        'milepoint', 
#        num_tot_kil, 
#        num_tot_inj, 
        'crash_date', 
#        f_harm_ev_cd1, 
#        m_harm_ev_cd1, 
#        man_coll_cd, 
#        crash_type, 
#        surf_cond_cd, 
        'crash_num', 
        'parish_cd', 
#        crash_hour, 
#        intersection, 
        'invest_agency_cd', 
        'travel_dirs', 
        'prior_movements', 
        'crash_year', 
        'csect', 
        'logmile', 
        'lrs_id', 
        'lrs_logmile', 
        'adt', 
#        alcohol, 
#        veh_type_cd1, 
#        veh_type_cd2, 
        'quadrant', 
        'spotted_by', 
        'intersection_id', 
#        severity_cd, 
        'city_cd', 
#        roadway_departure, 
#        lane_departure, 
#        road_rel_cd, 
#        hwy_class, 
#        contributing_factor, 
        'location_type', 
#        veh_severity_cd, 
        'ORIG_LATITUDE', 
        'ORIG_LONGITUDE', 
        'DOTD_LATITUDE', 
        'DOTD_LONGITUDE', 
        'parish_cd', 
        'hwy_type_cd', 
        'pri_hwy_num', 
        'bypass', 
        'milepost', 
        'pri_road_name', 
        'pri_dist', 
        'pri_measure', 
        'pri_dir', 
        'inter_road', 
#        dr_age_1, 
#        dr_age_2, 
#        dr_sex_1, 
#        dr_sex_2, 
#        pri_contrib_fac_cd, 
#        sec_contrib_fac_cd, 
#        vision_obscure_1, 
#        vision_obscure_2, 
#        movement_reason_1, 
#        movement_reason_2, 
#        ped_actions_1, 
#        ped_actions_2, 
#        veh_lighting_1, 
#        veh_lighting_2, 
#        traff_cntl_cond_1, 
#        traff_cntl_cond_2, 
        'pri_road_dir', 
#        lighting_cd, 
#        num_veh, 
#        crash_time, 
#        dr_cond_cd1, 
#        dr_cond_cd2, 
#        veh_cond_cd1, 
#        veh_cond_cd2,
            ]:
        ind = Header.index(head)
        for row in Crash:
            del(row[ind])
        Header = Crash[0]

    # For each column, if it's 'A','B',..., determine how many, and turn each into a one-hot column, in a new numpy matrix.  Keep track of which column is which, in a labels list.
    Header = Crash[0]
    Column_Contents = [[x,[]] for x in Header]
    for j in range (len(Crash[0])):
        A = [x[j] for i, x in enumerate(Crash) if i != 0]
        A = sorted(list(set(A)))
        Column_Contents[j][1] = A
    nrows = len(Crash) - 1
    ncols = sum([len(x[1]) for x in Column_Contents])
    X = np.zeros((nrows, ncols))

    ncol = -1
    for j in range (len(Column_Contents)):
        for k in range (len(Column_Contents[j][1])):
            ncol += 1
            code = Column_Contents[j][1][k]
            for i in range (1, len(Crash)):
                if Crash[i][j] == code:
                    X[i-1][ncol] = 1

    return X, Y, Crash, Column_Contents
                    
if __name__ == "__main__":
    X, Y, Crash, Column_Contents = Feature_Engineering()
    print (np.shape(X))
    print (np.shape(Y))

    print ()
    for row in Column_Contents:
        print (len(row[1]), row)
    print ()
    
    for i in range (10):
        print (Crash[i][:2])
    print ()
    for i in range (10):
        print (X[i][:10])

    
