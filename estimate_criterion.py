import numpy as np

def MaxDrawdown(prices_df):
    return round(abs(prices_df.sub(prices_df.expanding().max(), axis=0).div(prices_df.expanding().max(), axis=0).min())*100,3)

def sharpe_ratio(price_df,rf=0.025):
    daily_return_rate = price_df/price_df.shift(1)-1
    daily_rf_rate = rf/252
    SR = ((daily_return_rate.mean()-daily_rf_rate)/daily_return_rate.std())*np.sqrt(252)
    return SR

def calc_estimate(returns_df):
    price_df = returns_df + 1
    # 最大回撤率
    MDD = MaxDrawdown(price_df)
    # 夏普比率
    SR = sharpe_ratio(price_df)

    return {'max_drawback':MDD,'sharpe_ratio':SR}