import numpy as np
import pandas
import csv
import matplotlib.pyplot as plt
from ggplot import *

turnstile_weather = pandas.io.parsers.read_csv('/Users/marcellomicheloni/turnstile_data_master_with_weather.csv', index_col=False, header=0)

def entries_histogram(turnstile_weather):
    plt.figure()
    turnstile_weather[(turnstile_weather.rain == 0.0)]['ENTRIESn_hourly'].hist(color='green', label='no rain',bins=50, range=(0,20000)) 
    turnstile_weather[(turnstile_weather.rain == 1.0)]['ENTRIESn_hourly'].hist(color='blue', label='rain',bins=50, range=(0,20000))
    plt.title('Histogram of ENTRIESn_hourly')
    plt.xlabel('ENTRIESn_hourly')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    return plt

print entries_histogram(turnstile_weather)

#def entries_histogram(turnstile_weather):
#    by_unit = turnstile_weather.groupby('UNIT',as_index=False).sum()
#    plot = ggplot(aes(x='ENTRIESn_hourly'), data = by_unit) + geom_histogram(binwidth=25000)
#    return plot
#print entries_histogram(turnstile_weather)
