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

        date = DATA_COLUMNS[provider]['date']
        close = DATA_COLUMNS[provider]['close']
        index = date

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

            # lowercase index name
            # df.index.name = df.index.name.lower()
            # remove index name
            df.index.name = None

        if verbose:
            print(df)

        self.df = df

def ath_indices(df):
    column = 'close'

    # first close is an all time high
    ath_loc = 0
    ath_index = df.index.values[ath_loc]
    ath_close = df.iloc[ath_loc][column]
    ath_df = pd.DataFrame([pd.Series({'loc': ath_loc, 'close': ath_close, 'prev': 0}, name=ath_index)])

    # go through all rows
    for index, row in df.iterrows():
        close = df[column][index]

        # found new all time high
        if close > ath_close:
            ath_loc = df.index.get_loc(index)
            ath_df = ath_df.append(pd.Series({'loc': ath_loc, 'close': close, 'prev': ath_close}, name=index))
            ath_close = df[column][ath_loc]

    ath_df['loc'] = ath_df['loc'].astype(int)

    return ath_df['loc'].tolist()

# keep only last of consecutive all time high indices
def remove_adjacent_indices(indices):
    def method(index_value):
        index = indices.index(index_value)

        if index == 0 or index == len(indices) - 1:
            return True

        return ((indices[index] + 1) != indices[index + 1])

    return list(filter(method, indices))

# indices pointing to qualifying dips
def dip_indices(symbol, threshold, data, ath_indices):
    headers = data['headers']
    rows = data['rows']
    PRICE_INDEX = headers.index(CLOSE_ID)

    indices = []

    for i in range(len(ath_indices) - 1):
        # from one all time high to the next
        from_index = ath_indices[i]
        to_index = ath_indices[i + 1]

        lowest_index = from_index
        for j in range(from_index, to_index + 1):
            price = rows[j][PRICE_INDEX]
            lowest_price = rows[lowest_index][PRICE_INDEX]

            # new lower price
            if price < lowest_price:
                lowest_index = j

        from_price = rows[from_index][PRICE_INDEX]
        lowest_price = rows[lowest_index][PRICE_INDEX]
        difference = from_price - lowest_price
        percentage = difference / from_price * 100
        if difference:
            if percentage >= threshold:
                indices.append(lowest_index)

    return indices

# drop from an all time high in percentage terms
def dip_drop_percentage(data, dip_index, ath_indices):
    headers = data['headers']
    rows = data['rows']
    PRICE_INDEX = headers.index(CLOSE_ID)

    from_ath_index = list(filter(lambda index: index < dip_index, ath_indices))[-1]
    from_ath_price = rows[from_ath_index][PRICE_INDEX]
    dip_price = rows[dip_index][PRICE_INDEX]

    drop = dip_price / from_ath_price - 1

    return drop

# verbose dips description
def dips_description(symbol, data, ath_indices, dip_indices):
    print(f'{Fore.CYAN}{symbol} Dips > {threshold}%{Style.RESET_ALL}')

    DATE_OUT_FORMAT = '%b %d, %Y'

    headers = data['headers']
    rows = data['rows']
    PRICE_INDEX = headers.index(CLOSE_ID)
    DATE_INDEX = headers.index(DATE_ID)

    for i in range(len(dip_indices)):
        dip_index = dip_indices[i]
        from_index = list(filter(lambda index: index < dip_index, ath_indices))[-1]
        to_index = list(filter(lambda index: index > dip_index, ath_indices))[0]

        start_date = datetime.strptime(rows[from_index][DATE_INDEX], DATE_FORMAT)
        start_price = rows[from_index][PRICE_INDEX]
        lowest_date = datetime.strptime(rows[dip_index][DATE_INDEX], DATE_FORMAT)
        lowest_price = rows[dip_index][PRICE_INDEX]
        last_date = datetime.strptime(rows[to_index][DATE_INDEX], DATE_FORMAT)
        last_price = rows[to_index][PRICE_INDEX]

        drop = dip_drop_percentage(data, dip_index, ath_indices)
        rise = last_price / lowest_price - 1

        days = (last_date - start_date).days
        months = days / 365 * 12
        years = days / 365

        dates_string = f'{start_date.strftime(DATE_OUT_FORMAT)} to {last_date.strftime(DATE_OUT_FORMAT)}'
        duration_string = f'{days}d {months:.1f}m {years:.1f}y'
        print(f'ATH Range {Fore.BLUE}{dates_string} {duration_string}{Style.RESET_ALL}')

        price_string = f'{start_price}'
        lowest_price_string = f'{lowest_price}'
        drop_string = f'{drop:,.2%} on {lowest_date.strftime(DATE_OUT_FORMAT)} took {(lowest_date - start_date).days}d'
        print(f'Price {price_string} drop to {lowest_price_string} {Fore.RED}{drop_string}{Style.RESET_ALL}')

        last_price_string = f'{last_price}'
        recovery_string = f'+{rise:,.2%} took {(last_date - lowest_date).days}d'
        print(f'Recovery at {last_price_string} {Fore.GREEN}{recovery_string}{Style.RESET_ALL}')

# get days since first day from indices
def indices_days(indices, data):
    DATE_INDEX = data['headers'].index(DATE_ID)
    first_date = datetime.strptime(data['rows'][0][DATE_INDEX], DATE_FORMAT)

    dates = list(map(lambda index: data['rows'][index][DATE_INDEX], indices))
    # fmt_dates = list(map(lambda date: date.strftime(format), py_dates))
    py_dates = list(map(lambda date: datetime.strptime(date, DATE_FORMAT), dates))
    days = list(map(lambda date: (date - first_date).days, py_dates))

    return days

# get prices from indices
def indices_prices(indices, data):
    PRICE_INDEX = data['headers'].index(CLOSE_ID)
    return list(map(lambda index: data['rows'][index][PRICE_INDEX], indices))

def first_date(data):
    headers = data['headers']
    rows = data['rows']
    DATE_INDEX = headers.index(DATE_ID)
    return datetime.strptime(data['rows'][0][DATE_INDEX], DATE_FORMAT)

def last_date(data):
    headers = data['headers']
    rows = data['rows']
    DATE_INDEX = headers.index(DATE_ID)
    return datetime.strptime(data['rows'][len(data['rows']) - 1][DATE_INDEX], DATE_FORMAT)

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

stdout = data.df.to_json(orient='index') if (data.df is not None) else data.df
print(stdout)

# # all time high indices
# ath_indices = ath_indices(symbol, data)
# # without consecutive rises (keep last only)
# # ath_indices = remove_adjacent_indices(ath_indices)
# # dip indices
# dip_indices = dip_indices(symbol, threshold, data, ath_indices)

# if DEBUG:
#     dips_description(symbol, data, ath_indices, dip_indices)

# # JSON response
# response = {}
# response["ath_indices"] = ath_indices
# response["dip_indices"] = dip_indices

# response["data"] = {}
# response["data"]["headers"] = data["headers"]

# # rows = []
# # for index in range(len(data["rows"])):
# #     if index in ath_indices or index in dip_indices:
# #         rows.append(data["rows"][index])
# rows = data["rows"]
# rows = list(filter(lambda row: rows.index(row) in ath_indices or rows.index(row) in dip_indices, rows))

# response["data"]["rows"] = rows

# print(response)

# # the two ath indices surrounding each dip index
# dip_ath_indices = []
# for dip_index in dip_indices:
#     from_index = list(filter(lambda index: index < dip_index, ath_indices))[-1]
#     to_index = list(filter(lambda index: index > dip_index, ath_indices))[0]
#     dip_ath_indices.extend((from_index, to_index))

# # add first and last if not ath
# edge_indices = []
# if dip_ath_indices and dip_ath_indices[0] != 0:
#     edge_indices.append(0)
# if dip_ath_indices and dip_ath_indices[len(dip_ath_indices) - 1] != len(data['rows']) - 1:
#     edge_indices.append(len(data['rows']) - 1)

# # dip indices and its surrounding all time high indices
# final_indices = sorted(dip_indices + dip_ath_indices + edge_indices)

# draw_plot(data, dip_indices, dip_ath_indices, edge_indices, final_indices)