import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from pandas import DataFrame
from datetime import datetime, timedelta


"""
=============== SUMMARY METRICS ===============
| Metric                | Value               |
|-----------------------+---------------------|
| Backtesting from      | 2021-05-01 00:00:00 |
| Backtesting to        | 2021-06-09 17:00:00 |
| Max open trades       | 10                  |
|                       |                     |
| Total trades          | 447                 |
| Starting balance      | 1000.000 USDT       |
| Final balance         | 1714.575 USDT       |
| Absolute profit       | 714.575 USDT        |
| Total profit %        | 71.46%              |
| Trades per day        | 11.46               |
| Avg. stake amount     | 180.278 USDT        |
| Total trade volume    | 80584.326 USDT      |
|                       |                     |
| Best Pair             | ALICE/USDT 24.7%    |
| Worst Pair            | HARD/USDT -35.15%   |
| Best trade            | PSG/USDT 17.98%     |
| Worst trade           | XVS/USDT -26.03%    |
| Best day              | 351.588 USDT        |
| Worst day             | -256.636 USDT       |
| Days win/draw/lose    | 25 / 8 / 4          |
| Avg. Duration Winners | 1:36:00             |
| Avg. Duration Loser   | 9:33:00             |
|                       |                     |
| Min balance           | 962.929 USDT        |
| Max balance           | 1714.575 USDT       |
| Drawdown              | 240.78%             |
| Drawdown              | 289.267 USDT        |
| Drawdown high         | 252.196 USDT        |
| Drawdown low          | -37.071 USDT        |
| Drawdown Start        | 2021-05-19 03:00:00 |
| Drawdown End          | 2021-05-19 20:00:00 |
| Market change         | -34.99%             |
===============================================

"""

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 16,
    "ewo_high": 7.486,
    "ewo_low": -8.405,
    "low_offset": 0.955,
    "rsi_buy": 58,
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 6,
    "high_offset": 0.998,
}

class NormalizerStrategyHO2(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table:
    minimal_roi = {
        "0": 0.463,
        "289": 0.174,
        "995": 0.087,
        "1638": 0
    }
    # Stoploss:
    stoploss = -0.331

    timeframe = '1h'

    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_buy_signal = True

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.251
    trailing_stop_positive_offset = 0.324
    trailing_only_offset_is_reached = True

    # Custom stoploss
    use_custom_stoploss = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 610

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Manage losing trades and open room for better ones.
        if (current_profit < 0) & (current_time - timedelta(minutes=300) > trade.open_date_utc):
            return 0.01
        return 0.99

    def fischer_norm(self, x, lookback):
        res = np.zeros_like(x)
        for i in range(lookback, len(x)):
            x_min = np.min(x[i-lookback: i +1])
            x_max = np.max(x[i-lookback: i +1])
            #res[i] = (2*(x[i] - x_min) / (x_max - x_min)) - 1
            res[i] = (x[i] - x_min) / (x_max - x_min)
        return res
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lookback = [13, 21, 34, 55, 89, 144, 233, 377, 610]
        for look in lookback:
            dataframe[f"norm_{look}"] = self.fischer_norm(dataframe["close"].values, look)
        collist = [col for col in dataframe.columns if col.startswith("norm")]
        dataframe["pct_sum"] = dataframe[collist].sum(axis=1)



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['pct_sum'] < .2) &
            (dataframe['volume'] > 0) # Make sure Volume is not 0
            ,
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['pct_sum'] > 8) &
            (dataframe['volume'] > 0) # Make sure Volume is not 0
            ,
            'sell'
        ] = 1
        return dataframe
