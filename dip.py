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

def draw_plot(df, ath_df, dip_df):
    plt.figure()

    plt.box(False)

    # plt.plot(df, 'k')
    plt.plot(ath_df, 'go')
    plt.plot(dip_df[['close']], 'ro')

    style = dict(size=10, color='red')
    plt.title(f'{symbol} Dips < {-threshold:.0f}%', **style)

    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b %d \'%y'))
    plt.xticks(rotation=45)

    def annotate(df, column, color, ytext, formatter=None, arrowprops=None):
        style = dict(size=9, color=color)

        for index in df.index:
            x = index
            y = df.at[index, 'close']
            text = df.at[index, column]
            text = formatter(text) if formatter else text
            plt.annotate(text, xy=(x, y), xycoords='data', xytext=(0, ytext), textcoords='offset points', arrowprops=arrowprops, ha='center', va='center', **style)

    annotate(ath_df, 'close', 'green', 15, lambda val: f'{val:,.0f}', dict(arrowstyle="]-", lw=1, color='green'))
    annotate(dip_df, 'dip', 'red', -30, lambda val: f'{val * 100:,.0f}%', dict(arrowstyle="<-", lw=1, color='red'))

    plt.margins(.1, .2)
    plt.tight_layout()
    plt.show()

    file_path = utils.file_path(__file__)
    plt.savefig(f'{file_path}/figs/{symbol}-{threshold:.0f}.png', dpi=150)

argparser = argparse.ArgumentParser(description='Dip Data')
argparser.add_argument("-s", "--symbol", help="stock symbol", required=True)
argparser.add_argument("-d", "--dip", help="dip percentage (integer)", required=True)
argparser.add_argument("-p", "--provider", help="data provider", choices=['yahoo', 'iex'], required=True)

args = argparser.parse_args()
symbol = args.symbol
threshold = float(args.dip)
provider = args.provider

verbose = utils.terminal()

data = DataContainer(symbol, provider, verbose=verbose)
df = data.df

ath = ath_df(df)
dip = dip_df(df, -(threshold / 100))

stdout = df.to_json(orient='columns') if (df is not None) else df
print(stdout)

if verbose:
    utils.cprint('\nstdout', Fore.YELLOW)
    utils.json_print(stdout)

    utils.cprint('\ndf', Fore.YELLOW)
    print(df)

    utils.cprint('\nath df', Fore.YELLOW)
    print(ath)

    utils.cprint('\ndip df', Fore.YELLOW)
    print(dip)

    draw_plot(df, ath, dip)
