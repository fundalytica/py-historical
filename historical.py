import json

import argparse
from colorama import Fore

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter

from utils import utils, data

class DataContainer:
    def __init__(self, symbol, provider, sandbox=False, verbose=False):
        DATA_PATHS = {
            'yahoo':        '/scripts/yahoo/yahoo-historical',
            'iex':          '/scripts/iex-historical'
        }
        if provider == 'iex':
            DATA_PATHS[provider] += f"/{'sandbox' if sandbox else 'cloud'}"

        DATA_COLUMNS = {
            'yahoo':    { 'date': 'Date', 'close': 'Close' },
            'iex':      { 'date': 'date', 'close': 'close' }
        }

        file = f'{DATA_PATHS[provider]}/{symbol}.csv'
        if verbose:
            utils.cprint(file, Fore.CYAN)

        index = DATA_COLUMNS[provider]['date']
        close = DATA_COLUMNS[provider]['close']

        # read dataframe from file
        df = data.df_read(file, index=index)

        if df is not None:
            # convert close to float
            df[close] = df[close].astype(float)

            # columns to keep (exclude index)
            keep = DATA_COLUMNS[provider].values()
            keep = list(filter(index.__ne__, keep))
            df = df[keep]

            # convert index to timestamp format
            df.index = pd.to_datetime(df.index)

            # remove index name
            df.index.name = None

            # make column names consistent
            dict = {value:key for (key, value) in DATA_COLUMNS[provider].items()}
            df.rename(columns=dict, inplace=True)

        self.df = df

def ath_df(df):
    close = 'close'

    m = (df[close] == df.cummax()[close])
    ath_df = df[m]

    return ath_df

def dip_df(df, threshold):
    close = 'close'

    dip_df = df.copy()                                      # make copy
    dip_df['ath'] = dip_df.cummax()                         # add all time high
    dip_df['dip'] = dip_df[close] / dip_df['ath'] - 1       # add dip percentage

    # https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-certain-values-a6ed8e5d8cc
    dip_df['close-eq-ath'] = dip_df[close] == dip_df['ath'] # boolean column (close price equal to all time high)
    dip_df['segment'] = dip_df['close-eq-ath'].cumsum()     # segment number (cumulative sum of boolean values)
    segments = dip_df.groupby('segment')                    # group to create one segment for each ath

    dip_df = pd.DataFrame()                                 # empty dataframe to add segment dips
    for num,segment in segments:
        m1 = segment[close] < segment['ath']                # is not the all time high
        m2 = segment[close] == segment[close].cummin()      # is the cumulative minimum (lower all than prices before it)
        m3 = segment['dip'] <= threshold                    # meets the threshold
        segment = segment[m1 & m2 & m3]
        dip_df = dip_df.append(segment[[close,'ath','dip']])

    return dip_df

def draw_plot(df, ath_df=None, dip_df=None, filename=None):
    plt.figure()
    plt.box(False)

    plt.plot(df, 'k')
    if ath_df is not None:
        plt.plot(ath_df, 'go')
    if dip_df is not None:
        plt.plot(dip_df[['close']], 'ro')

    plt.title(f'{symbol}', **dict(size=10, color='black'))
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b %d \'%y'))
    plt.xticks(rotation=45)

    def annotate(df, column, color, ytext, formatter=None, arrowprops=None):
        for index in df.index:
            x = index
            y = df.at[index, 'close']
            text = df.at[index, column]
            text = formatter(text) if formatter else text
            plt.annotate(text, xy=(x, y), xycoords='data', xytext=(0, ytext), textcoords='offset points', arrowprops=arrowprops, ha='center', va='center', **dict(size=9, color=color))

    if ath_df is not None:
        annotate(ath_df, 'close', 'green', 15, lambda val: f'{val:,.0f}', dict(arrowstyle="]-", lw=1, color='green'))
    if dip_df is not None:
        annotate(dip_df, 'dip', 'red', -30, lambda val: f'{val * 100:,.0f}%', dict(arrowstyle="<-", lw=1, color='red'))

    plt.margins(.1, .2)
    plt.tight_layout()
    plt.show()

    file_path = utils.file_path(__file__)
    plt.savefig(f'{file_path}/figs/{filename}.png', dpi=150)

def stdout(df_dict):
    stdout = {}
    stdout["dates"] = { "from": utils.pd_ts_to_unix_ts(all.index[0]) * 1000, "to": utils.pd_ts_to_unix_ts(all.index[-1]) * 1000 }
    for key in df_dict:
        stdout[key] = json.loads((df_dict[key]).to_json(orient='columns'))
    stdout["size"] = utils.mbsize(stdout)
    print(json.dumps(stdout))

def dfs_print(dict):
    for key in dict:
        utils.cprint(f'\n{key} df', Fore.YELLOW)
        print(dict[key])

argparser = argparse.ArgumentParser(description='Dip Data')
argparser.add_argument("-s", "--symbol", help="stock symbol", required=True)
argparser.add_argument("-p", "--provider", help="market data provider", choices=['yahoo', 'iex'], required=True)

argparser.add_argument("--ath", action="store_true", help="all time highs only")
argparser.add_argument("--dip", help="dip percentage integer")

argparser.add_argument("--plot", action="store_true", help="plot chart")

args = argparser.parse_args()
symbol = args.symbol
provider = args.provider

verbose = utils.terminal()

data = DataContainer(symbol, provider, verbose=verbose)

all = data.df
ath = None
dip = None

if all is None:
    print(json.dumps({"error": "no data"}))
    exit()

# historical data
if not args.ath and not args.dip:
    df_dict = { "all": all }
    stdout(df_dict)
    if args.plot:
        draw_plot(all, filename=symbol)

# all time high data (+ historical)
if args.ath:
    ath = ath_df(all)
    df_dict = { "all": all, "ath": ath }
    stdout(df_dict)
    if args.plot:
        draw_plot(all, ath, filename=f'{symbol}-ATH')

# dip data (+ historical, + all time high)
if args.dip:
    ath = ath_df(all)
    threshold = -(float(args.dip) / 100)
    dip = dip_df(all, threshold)
    df_dict = { "all": all, "ath": ath, "dip": dip }
    stdout(df_dict)
    if args.plot:
        draw_plot(all, ath, dip, filename=f'{symbol}-DIP-{int(args.dip)}')

if verbose:
    dfs_print(df_dict)