import requests
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import date, datetime, time, timedelta, tzinfo
import calendar
import dateutil.parser
import pytz

import os


def load_cache() -> pd.DataFrame:
    """
    Get cached data.
    """
    try:
        cached_data = pd.read_csv("cache/price_data.tsv", sep="\t", index_col=0)
        cached_data["time"] = pd.to_datetime(cached_data["time"], utc=True).dt.tz_localize(None)

    except FileNotFoundError:
        cached_data = pd.DataFrame({})

    return cached_data


def save_cache(data:pd.DataFrame, filename:str="cache/price_data.tsv"):
    """
    Save data to a file.
    """
    # Load cache and merge with new data
    cache = load_cache()
    cache = pd.concat([cache, data]).drop_duplicates().sort_values("time").reset_index(drop=True)
    # Create path if it does not exist
    filepath = os.path.split(filename)[0]
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    cache.to_csv(filename, sep="\t")


def get_price_data(start_date:datetime=None, end_date:datetime=None, time_trunc:str="hour", geo_limit:str="peninsular", cache:bool=True) -> pd.DataFrame:
    """
    Get price data from the API or cache.
    :param start_date: datetime object with UTC start date
    :param end_date: datetime object with UTC end date
    :param time_trunc: time truncation (hour, day, month, year)
    :param geo_limit: geographical limit (peninsular, canarias, baleares, ceuta, melilla, ccaa)
    """
    # If no start date is given, get today's date
    if start_date is None:
        start_date, end_date = get_todays_date()  # datetime objects in UTC
    
    if cache:
        cached_data = load_cache()
        if not cached_data.empty:
            # Get data within the given date range
            # All comparisons are done in UTC
            cached_data_range = cached_data[(cached_data["time"].dt.to_pydatetime() >= start_date) & (cached_data["time"].dt.to_pydatetime() <= end_date)]
            
            # Check that cached data length is the same as the requested range (assumes the data is the same)
            print(len(cached_data_range))
            print(1 + int((end_date - start_date).total_seconds() / 3600))
            if not cached_data_range.empty and len(cached_data_range) == 1 + int((end_date - start_date).total_seconds() / 3600):
                print("Using cached data.")
                return cached_data_range
        
        else:
            print("Cache empty.")

    # One of the conditions above failed, so we need to get the data from the API
    data = fetch_data(start_date, end_date, time_trunc, geo_limit)
    price_data = extract_price_data(data)

    print("Saving cache...")
    save_cache(price_data)

    return price_data


def fetch_example_data():
    """
    Call the API to get the price data.

    For now, this function just gets example data.
    """

    # Today's date in ISO 8601 format
    start_date = "2021-12-24T00:00"
    end_date = "2021-12-24T23:59"
    time_trunc = "hour"
    geo_limit = "peninsular"
    base_url = "https://apidatos.ree.es/es/datos/"
    
    # API URL
    url = base_url + "mercados/precios-mercados-tiempo-real" + "?start_date=" + start_date + "&end_date=" + end_date + "&time_trunc=" + time_trunc + "&geo_limit=" + geo_limit
    print(url)
    r = requests.get(url)

    return r.json()


def fetch_todays_data():
    """
    Call the API to get the price data between now and the last available data.

    For now, this function just gets example data.
    """
    release_time = datetime.combine(date.today(), datetime.min.time()) + timedelta(hours=20, minutes=30)
    now = datetime.now()
    if now < release_time:
        print("Tomorrow's data is not available yet.")
        ans = input("Do you still want to continue? (Y/n): ")
        if ans.upper() != "Y":
            exit()

    # URL parameters
    start_date = now.strftime("%Y-%m-%dT%H:%M")  # Now in ISO 8601 format
    end_date = datetime.combine(date.today(), datetime.min.time()) + timedelta(days=1, hours=23, minutes=59)
    end_date = end_date.strftime("%Y-%m-%dT%H:%M")  # Tomorrow at 23:59 in ISO 8601 format
    time_trunc = "hour"
    geo_limit = "peninsular"
    base_url = "https://apidatos.ree.es/es/datos/"
    
    # API URL
    url = base_url + "mercados/precios-mercados-tiempo-real" + "?start_date=" + start_date + "&end_date=" + end_date + "&time_trunc=" + time_trunc + "&geo_limit=" + geo_limit
    print(url)
    r = requests.get(url)

    # TODO: Refactor with generic API fucntion

    return r.json()


def get_todays_date() -> tuple[datetime, datetime]:
    """
    Get now's UTC date and tomorrow's 23:59 UTC in datetime objects.
    TODO: May be good to find a better name
    """
    release_time = datetime.combine(date.today(), datetime.min.time()) + timedelta(hours=20, minutes=30)
    start_date = datetime.now()

    """
    
    if start_date < release_time:
        print("Tomorrow's data is not available yet.")
        ans = input("Do you still want to continue? (Y/n): ")
        if ans.upper() != "Y":
            exit()
    """

    end_date = datetime.combine(date.today(), datetime.min.time()) + timedelta(days=1, hours=23, minutes=59)

    return start_date, end_date


def fetch_data(start_date:datetime, end_date:datetime, time_trunc:str="hour", geo_limit:str="peninsular"):
    """
    Call the API to get the price data.
    :param start_date: datetime object with UTC start date
    :param end_date: datetime object with UTC end date
    :param time_trunc: time truncation (hour, day, month, year)
    :param geo_limit: geographical limit (peninsular, canarias, baleares, ceuta, melilla, ccaa)
    """
    tz = pytz.timezone('Europe/Madrid')
    # URL parameters
    start_date = start_date.replace(tzinfo=tz).strftime("%Y-%m-%dT%H:%M")  # ISO 8601 format with timezone
    end_date = end_date.replace(tzinfo=tz).strftime("%Y-%m-%dT%H:%M")  # ISO 8601 format with timezone
    # TODO: Raise custom errors (e.g. if start_date > end_date)

    base_url = "https://apidatos.ree.es/es/datos/"
    
    # API URL
    url = base_url + "mercados/precios-mercados-tiempo-real" + "?start_date=" + start_date + "&end_date=" + end_date + "&time_trunc=" + time_trunc + "&geo_limit=" + geo_limit
    print(url)
    r = requests.get(url)

    return r.json()


def extract_price_data(data:json) -> pd.DataFrame:
    """
    Extract price data from json response
    :param data: json data
    :return: price data
    """
    # The list "included" contains the different data prices.
    # We are interested in the PVPC price, so we take the first element of the list.
    try:
        values = data["included"][0]["attributes"]["values"]
    except KeyError:
        print("Could not find price data.")
        return pd.DataFrame({})

    prices = []
    times = []
    for v in values:
        prices.append(v["value"])

        t = v["datetime"]
        # Create datetime object and ignore timezone, converting it to UTC.
        #t = dateutil.parser.isoparse(t)
        times.append(t)

    """
    #price_data = pd.DataFrame(columns={'time': np.datetime64, 'price': np.float64})
    price_data['time'] = times
    # Remove timezone data
    print(price_data)
    print(price_data.dtypes)
    price_data['time'] = price_data['time'].dt.tz_localize(None)
    price_data['price'] = prices
    """

    price_data = pd.DataFrame({"time": times, "price": prices})
    # Convet time to UTC datetime and remove timezone suffix
    price_data['time'] = pd.to_datetime(price_data['time'], utc=True).dt.tz_localize(None)
    # Convert to tz-aware datetime with timezone suffix
    #price_data['time'] = pd.to_datetime(price_data['time'])

    # Currently not in use, but could be useful in the future
    # price_data = np.column_stack((times, prices))

    return price_data


def plot_price_data(price_data:pd.DataFrame, labels=["Precio de la luz", "Time", "Price (€/ MWh)"], limits=[0,600]) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot price data
    :param price_data: price data
    """
    # Create figure
    fig, ax = plt.subplots()

    # Plot data
    #ax.step(price_data["time"], price_data["price"], where="post")
    ax.plot(price_data["time"], price_data["price"])

    # Format plot
    # (1) Title and labels
    ax.set_title(labels[0])
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])

    # (2) Set x-axis to be time and format it
    tz = pytz.timezone('Europe/Madrid')
    locator = mdates.AutoDateLocator(minticks=4, maxticks=24, tz=tz)
    formatter = mdates.ConciseDateFormatter(locator, tz=tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis_date(tz=tz)

    ax.set_ylim(limits)

    return fig, ax


def optimize_charge(price_data:pd.DataFrame, window:float) -> int:
    """
    Find optimal charging schedule for a given price data.
    :param price_data: price data
    :param window: hours left to full charge
    :return: optimal charging schedule
    """
    price_data = np.asarray(price_data["price"])

    # Create convolution filters for price window calculation
    filters = []
    if int(window) == window:
        # if window is an integer, use it as the number of hours to charge
        window = int(window)
        window_filter = np.ones(window)
        filters.append(window_filter)
    
    else:
        # If window is a float, modify the first or last element of the window filter with the decimal part
        # (e.g. for 3.5 hours, we could use two convolution filters (1,1,1,0.5) or (0.5,1,1,1))
        # This way we can get the optimal schedule for incomplete hours.

        window_filter = np.ones(int(window))
        window_filter[0] = window - int(window)
        filters.append(window_filter)

        alt_filter = np.ones(int(window))
        alt_filter[-1] = window - int(window)
        filters.append(alt_filter)

    # Calculate optimal charging schedule
    # (1) Storage variables
    abs_min = 0
    all_mins = 0
    # TODO: calculate minima for all different filters and find minumum between them

    charge_price = np.convolve(price_data, np.ones(window), mode='valid')
    m = np.amin(charge_price)
    print(f"Minimum prize: {m}")
    start = np.where(charge_price == m)[0][0]
    print(f"Start: {start}")

    # TODO: Allow for float windows (e.g. for 3.5 hours, we could use two convolution filters (1,1,1,0.5) or (0.5,1,1,1)
    return start
    

def optimize_charge_pd(price_data:pd.DataFrame, window:int) -> pd.DataFrame:
    """
    Find optimal charging schedule for a given price data.
    :param price_data: price data
    :param window: hours left to full charge
    :return opt: optimal charging schedule (pd.DataFrame with start time in UTC and price)
    TODO: Think about this: should we return the full dataframe with a new column tagging the optimal hour?
    TODO: Allow for float windows (e.g. for 3.5 hours, we could use two convolution filters (1,1,1,0.5) or (0.5,1,1,1)
    """
    datapoints = len(price_data)
    if datapoints < window:
        print(f"Not enough data to calculate optimal charging schedule (got {datapoints}, minimum is {window})")
        #print(price_data)
        return pd.DataFrame({})

    # Calculate average price over sliding window
    # df.rolling() aligns the window to the rightmost edge, so we need to shift it to the left
    price_data["average"] = price_data.rolling(window).mean().shift(-window+1)

    # Get the minimum price
    opt = price_data[price_data["average"] == price_data["average"].min()].copy()
    opt["hour"] = pd.to_datetime(opt["time"]).dt.hour  # TODO: What is dt?

    opt.reset_index(inplace=True)
    # Right now, we get the optimal window and reset the index of the dataframe.
    # Later, we extract the first element (supposedly the only element) as df[0]
    # Could this have unintended consequences?

    return opt


def trend_per_hour(price_data:pd.DataFrame, hour:int, offset:int=0):
    """
    Calculate the trend of the price at each hour,
    to potentially be able to model the price in the following days
    :param price_data: price data
    :param hour: hour to calculate the trend
    :param offset: offset to the hour. 0 means the first the first element
    of the price data corresponds to hour.
    """
    # Filter dataframe entries by hour and calculate the average
    # Use a mapping function that gets the hour in the datetime object?
    price_data["hour"] = price_data["time"].map(lambda x: x.hour)

    # Fiter dataframe entries by hour
    hour_data = price_data[price_data["hour"] == hour]

    # Plot
    #plt.plot(hour_data["time"], hour_data["price"])
    #plt.show()
    fig, ax = plot_price_data(hour_data)
    plt.show()


def optimize_charge_per_day(window:int, offset:int=0):
    """
    In a for loop:
    - Get the price data for each day of the month using the get_price_data() function
    - Get the optimal charging schedule for each day using the optimize_charge_pd() function
    - Count the number of days that the optimal charging schedule is the same

    TODO Keep cached data in memory instead of loading 365 times from file
    """
    # Storage variables
    all_opt = pd.DataFrame({})
    tz = pytz.timezone('Europe/Madrid')

    for month in range(12,13):
        # Get the price data for each day of the month
        days_in_month = calendar.monthrange(datetime.now().year, month)[1]
        for day in range(1, days_in_month+1):
            # Create reference datetime object (for loop month and day, and hour = 00:00)
            ref_date = datetime(day=day, month=month, hour=0, minute=0, second=0, microsecond=0)
            ref_date = tz.localize(ref_date)

            # Shift start time by offset (usually to the release time of the electricity price data)
            d1 = ref_date + timedelta(hours=offset)  # Get day to analyze
            d2 = ref_date.replace(hour=23, minute=59) + timedelta(hours=offset)  # Get next day to analyze

            # Convert to UTC and make tz-naive
            d1 = d1.astimezone(pytz.utc).replace(tzinfo=None)
            d2 = d2.astimezone(pytz.utc).replace(tzinfo=None)

            print(d1, d2, int((d2-d1).total_seconds()/3600))

            price_data = get_price_data(d1, d2)

            if price_data.empty:
                continue

            # Get the optimal charging schedule for each day
            opt = optimize_charge_pd(price_data, window)

            if opt.empty:
                continue

            all_opt = all_opt.append(opt)
    
    # Count the number of days that the optimal charging schedule is the same
    print(all_opt.groupby(["hour"]).count())  # results in UTC time


def fill_year_cache():
    """
    Fill the cache with the price data for the current year
    """
    #tz = pytz.timezone('Europe/Madrid')
    year = datetime.now().year
    for month in range(1,13):
        print(month)
        days_in_month = calendar.monthrange(datetime.now().year, month)[1]
        d1 = datetime(month=month, day=1, hour=0, minute=0)
        d2 = d1 + timedelta(days=days_in_month) - timedelta(minutes=1)
        price_data = get_price_data(d1, d2)  # This function fills the cache

