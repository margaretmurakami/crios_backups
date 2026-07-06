from datetime import datetime, timedelta
import numpy as np


def is_leap(year):
    # determine if a year is a leap year
    if year%100 == 0:
        if year%400 != 0:
            return False
        else:
            return True
    if year%4 == 0:
        return True

def ts2dte(ts, deltat=1200, startyr=1992, startmo=1, startdy=1):
    '''
        # Example usage
        # ts = 1000  # Example time step number
        # dte = ts2dte(ts)
        # print(dte.strftime("%Y-%m-%d %H:%M:%S"))  # Print the date in a specific format
    '''
    
    # Convert time step to seconds
    ts_seconds = ts * deltat
    # Calculate the start date
    start_date = datetime(startyr, startmo, startdy)
    # Add the calculated seconds to the start date
    dte = start_date + timedelta(seconds=ts_seconds)
    return dte

def get_fnames(dt,startyr,endyr):
    '''
    A function to get the filenames from a given model run using the rules about leap years and leap days
    Starts from Jan 1 of the startyr and ends Dec 1 of the end yr
    inputs:
        dt: dtime from the model
        startyr: starting year
        endyr: ending year from the model run
    outputs:
        fnames: filenames from the model run


    Example:
        dt = 600
        startyr = 2002
        endyr = 2019
        fnames = get_fnames(dt,startyr,endyr)
    '''
    days_reg = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    days_leap = np.array([31,29,31,30,31,30,31,31,30,31,30,31])
    days_reg = days_reg*24*3600/dt
    days_leap = days_leap*24*3600/dt

    years = np.arange(startyr,endyr,1)

    # make an array of all the filenames before cumsum
    fnames = np.array([],dtype=int)
    
    for year in years:
        if is_leap(year):
            fnames = np.append(fnames,days_leap)
        else:
            fnames = np.append(fnames,days_reg)
    return(np.cumsum(fnames))

def get_tsteps(times,fnames,dt,startyr,startmo,startdy):
    '''
    A function to get the timesteps of interest in a given model run provided a dictionary of the months, years we are interested in
    inputs:
        times: dictionary of the months and years we want
        fnames: filenames we got from the previous set
        dt, startyr, startmo, startdy: all the time steps for ts2dte that we need

    outputs:
        tsstr: the tsstr of the filenames we want

    Example:
        dt = 600
        startyr = 2002
        endyr = 2019
        fnames = get_fnames(dt,startyr,endyr)
        
        times = {}
        times["2014"] = np.arange(1,13,1)
        times["2015"] = np.array([1])
        
        tsstr,datetimes = get_tsteps(times,fnames,dt,startyr,1,1)
    '''
    tsstr = np.array([])
    datetimes = np.array([])
    # for each year
    for year in times.keys():
        months = times[year]
        f_toread = np.array([],dtype=int)

        # for all the filenames in the model run
        for inf in fnames:
            thisfile = ts2dte(inf,deltat=dt,startyr=startyr,startmo=startmo,startdy=startdy)

            # add if this fname is in the months and years we want
            if thisfile.year == int(year) and thisfile.month in months:
                #print(thisfile)  # this would be used as the second time step for the month of November, not what we want
                datetimes = np.append(datetimes,thisfile)
                f_toread = np.append(f_toread,int(inf))
        
        t_day = f_toread.astype(str)
        mytsstr = np.array([str(item).zfill(10) for item in t_day])
        tsstr = np.append(tsstr,mytsstr)
    return tsstr,datetimes