import pytz
from redcharge import *
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Create datetime objects with local time
d1 = datetime(year=2021, month=3, day=28, hour=0, minute=0, second=0, microsecond=0)
d2 = datetime(year=2021, month=3, day=28, hour=23, minute=59, second=0, microsecond=0)

# Add timezone information (keeps same time, but makes it timezone-aware)
tz = pytz.timezone('Europe/Madrid')
d1 = tz.localize(d1)
d2 = tz.localize(d2)

print(d1, d2)

# Convert to UTC and remove timezone information (we only work with UTC)
d1 = d1.astimezone(pytz.utc).replace(tzinfo=None)
d2 = d2.astimezone(pytz.utc).replace(tzinfo=None)

print(d1, d2)

pd_data = get_price_data(d1, d2)
print(pd_data)

# Get optimal start time
opt = optimize_charge_pd(pd_data, 10)

if opt.empty:
    exit()

# Localize result data
tz = pytz.timezone('Europe/Madrid')
opt["time"] = opt["time"].dt.tz_localize(pytz.utc).dt.tz_convert(tz)
opt["hour"] = pd.to_datetime(opt["time"]).dt.hour
print(opt)

# Plot
# (1): Price data
fig, ax = plot_price_data(pd_data)

# (2): Optimal charging schedule
start = opt["time"][0]
end = opt["time"][0] + timedelta(hours=10)
center = start + timedelta(hours=(end - start).total_seconds()/3600/2)
start_str = ""
ax.text(x=center, y=200, color="black", s=f"Start: {start.strftime('%b %d %H:%M')}", ha="center")
ax.axvspan(start, end, color='red', alpha=0.1)  # Charging schedule

avg = opt["average"][0]
ax.text(x=center, y=160, color="black", s=f"Avg: {avg:.2f} €/MW", ha="center")
ax.hlines(avg, color='red', linestyle='--', xmin=start, xmax=end)  # Average price

plt.show()


# Other examples
# TODO: Clean example code

#d1 = dateutil.parser.parse("2021-03-28T00:00:00+01:00").astimezone(pytz.utc).replace(tzinfo=None)
#d2 = dateutil.parser.parse("2021-03-28T23:59:00+02:00").astimezone(pytz.utc).replace(tzinfo=None)

#print(d1, d2)

#data = fetch_data(d1, d2)

#d1 = datetime.now(tz=tz).replace(day=15, hour=0, minute=0)
#d2 = datetime.now(tz=tz).replace(day=22, hour=0, minute=0)


#fill_year_cache()
#exit()
#optimize_charge_per_day(10, offset=-6)

# Request example data
#data = fetch_todays_data()

# Load example data from file
#data = json.load(open('data.json'))

#pd_data = extract_price_data(data)

#trend_per_hour(pd_data, hour=12)
#s = optimize_charge(pd_data, 10)
#print(pd_data["time"][s])

#optimize_charge_per_day(10, offset=-4)
#exit()