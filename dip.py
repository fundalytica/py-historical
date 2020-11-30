# from datetime import datetime, timedelta

import argparse
import pandas as pd
from colorama import Fore
# import matplotlib.pyplot as plt

from utils import utils, data

# DATE_FORMAT = '%Y-%m-%d'

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

def draw_plot(data, dip_indices, dip_ath_indices, edge_indices, final_indices):
    fig, ax = plt.subplots()

    plt.figure(figsize=(8,5))

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.box(False)

    # fig = plt.figure(linewidth=10, edgecolor='black')

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.plot(indices_days(final_indices, data), indices_prices(final_indices, data), 'k')
    plt.plot(indices_days(dip_indices, data), indices_prices(dip_indices, data), 'ro')
    plt.plot(indices_days(dip_ath_indices, data), indices_prices(dip_ath_indices, data), 'go')
    plt.plot(indices_days(edge_indices, data), indices_prices(edge_indices, data), 'bo')

    style = dict(size=12, color='red')
    plt.title(f'{symbol} {threshold:.0f}%+ Dips', **style)

    def add_text(indices, style, offset, drop):
        days = indices_days(indices, data)
        prices = indices_prices(indices, data)

        for index in range(len(indices)):
            day = days[index]
            price = prices[index]
            text = f'${price:,.0f}'

            if drop:
                drop_percentage = dip_drop_percentage(data, indices[index], dip_ath_indices)
                text += f'\n{drop_percentage:.0%}'

            plt.annotate(text, (day, price), textcoords="offset points", xytext=offset, ha='center', va='center', **style)

    text_size = 9
    add_text(dip_indices, dict(size=text_size, color='red'), (0, -16), drop=True)
    add_text(dip_ath_indices, dict(size=text_size, color='green'), (-12, 10), drop=False)
    add_text(edge_indices, dict(size=text_size, color='blue'), (-12, 10), drop=False)

    plt.xticks(rotation=45)

    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    from matplotlib.ticker import FuncFormatter
    def format_method(x, pos):
        date = first_date(data) + timedelta(days=int(x))
        return date.strftime('%b \'%y')

    formatter = FuncFormatter(format_method)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.margins(.08, .18)
    plt.tight_layout()
    plt.show()

    first = (first_date(data)).strftime('%Y%m%d')
    last = (last_date(data)).strftime('%Y%m%d')

    file_path = utils.file_path(__file__)
    plt.savefig(f'{file_path}/figs/{symbol}-{threshold:.0f}-dips-{first}-{last}.png', dpi=150)

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


# draw_plot(data, dip_indices, dip_ath_indices, edge_indices, final_indices)
