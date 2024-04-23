from datetime import datetime, timedelta

def ts2dte(ts, deltat=1200, startyr=1992, startmo=1, startdy=1):
    # Convert time step to seconds
    ts_seconds = ts * deltat
    # Calculate the start date
    start_date = datetime(startyr, startmo, startdy)
    # Add the calculated seconds to the start date
    dte = start_date + timedelta(seconds=ts_seconds)
    return dte

# Example usage
ts = 1000  # Example time step number
dte = ts2dte(ts)
print(dte.strftime("%Y-%m-%d %H:%M:%S"))  # Print the date in a specific format

