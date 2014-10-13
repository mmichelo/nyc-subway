import numpy as np
import scipy
import scipy.stats
import pandas

turnstile_weather = pandas.io.parsers.read_csv('/Users/marcellomicheloni/turnstile_data_master_with_weather.csv', index_col=False, header=0)


def mann_whitney_plus_means(turnstile_weather):
    
    with_rain = turnstile_weather[(turnstile_weather.rain == 1.0)]['ENTRIESn_hourly']
    with_rain_mean = np.mean(with_rain)
    without_rain = turnstile_weather[(turnstile_weather.rain == 0.0)]['ENTRIESn_hourly']
    without_rain_mean = np.mean(without_rain)
    U,p = scipy.stats.mannwhitneyu(with_rain,without_rain)
    return with_rain_mean, without_rain_mean, U, p # leave this line for the grader

print mann_whitney_plus_means(turnstile_weather)
