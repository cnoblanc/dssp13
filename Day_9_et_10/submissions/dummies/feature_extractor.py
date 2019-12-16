import pandas as pd
import os
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def _encode(X_df):
    X_encoded = X_df.copy()
    # -----------------------------------------
    # Manage the date encoding + Date dimension
    # -----------------------------------------
    X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'], format="%Y-%m-%d")

    X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
    X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
    X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
    X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
    X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
    X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)
    X_encoded['isweekend']=X_encoded['DateOfDeparture'].dt.weekday.isin([5,6]).astype(int)

    # Get US Holidays
    cal = calendar()
    holidays = cal.holidays(start=X_encoded['DateOfDeparture'].min()- datetime.timedelta(days=10) \
                            , end=X_encoded['DateOfDeparture'].max()+ datetime.timedelta(days=10))
    X_encoded['isHoliday']=X_encoded['DateOfDeparture'].isin(holidays)

    # get Next days, last days and within days
    X_encoded['isHoliday_in_next_days']=X_encoded['isHoliday']
    X_encoded['isHoliday_in_last_days']=X_encoded['isHoliday']
    for dt in range(2):
        plusd_day=datetime.timedelta(days= dt+1)
        X_encoded['isHoliday_plusdt_day']=(X_encoded['DateOfDeparture']+plusd_day).isin(holidays)
        X_encoded['isHoliday_in_next_days']=X_encoded['isHoliday_in_next_days'] | X_encoded['isHoliday_plusdt_day']
        minusd_day=datetime.timedelta(days= -dt-1)
        X_encoded['isHoliday_minusdt_day']=(X_encoded['DateOfDeparture']+minusd_day).isin(holidays)
        X_encoded['isHoliday_in_last_days']=X_encoded['isHoliday_in_last_days'] | X_encoded['isHoliday_minusdt_day'] 
        # Within x days
        X_encoded['isHoliday_within_days']=X_encoded['isHoliday_in_next_days'] | X_encoded['isHoliday_in_last_days'] 

    X_encoded['isHoliday_in_next_days']=X_encoded['isHoliday_in_next_days'].astype(int)
    X_encoded['isHoliday_in_last_days']=X_encoded['isHoliday_in_last_days'].astype(int)
    X_encoded['isHoliday_within_days']=X_encoded['isHoliday_within_days'].astype(int)
    #X_encoded = X_encoded.drop(['isHoliday_in_next_days','isHoliday_in_last_days'], axis=1)
    X_encoded = X_encoded.drop(['isHoliday_minusdt_day','isHoliday_plusdt_day'], axis=1)
    # convert 'isHoliday' in int 
    X_encoded['isHoliday']=X_encoded['isHoliday'].astype(int)

    # -----------------------------------------
    # Manage the external file : weather
    # -----------------------------------------
    # Read external data file
    path = os.path.dirname(__file__)
    data_events = pd.read_csv(os.path.join(path, 'external_data.csv'))
    # Manage the date feature
    data_events['Date_dt'] = pd.to_datetime(data_events['Date'], format="%Y-%m-%d")
    data_events = data_events.drop('Date', axis=1)

    # Events
    X_events=data_events.groupby(['Date_dt','Airport']).count()
    X_events=X_events.reset_index()
    X_events = X_events.drop('index', axis=1)
    X_events_Arr = X_events.rename(columns={'Airport': 'Arrival'})

    # -----------------------------------------
    # Merge main data with Events (at Arrival)
    # -----------------------------------------
    plusd_day=datetime.timedelta(days=0)
    X_events_Arr['DateOfDeparture']=X_events_Arr['Date_dt']+plusd_day
    X_events_Arr = X_events_Arr.rename(columns={'Event':'Arr_Evts_count_d0'})
    X_encoded = pd.merge(X_encoded, X_events_Arr, how='left',sort=False,
        left_on=['DateOfDeparture', 'Arrival'],right_on=['DateOfDeparture', 'Arrival'])
    X_encoded['Arr_Evts_count_d0'].fillna(0, inplace=True)

    plusd_day=datetime.timedelta(days=1)
    X_events_Arr['DateOfDeparture']=X_events_Arr['Date_dt']+plusd_day
    X_events_Arr = X_events_Arr.rename(columns={'Arr_Evts_count_d0':'Arr_Evts_count_d1'})
    X_encoded = pd.merge(X_encoded, X_events_Arr, how='left',sort=False,
        left_on=['DateOfDeparture', 'Arrival'],right_on=['DateOfDeparture', 'Arrival'])
    X_encoded['Arr_Evts_count_d1'].fillna(0, inplace=True)

    plusd_day=datetime.timedelta(days=2)
    X_events_Arr['DateOfDeparture']=X_events_Arr['Date_dt']+plusd_day
    X_events_Arr = X_events_Arr.rename(columns={'Arr_Evts_count_d1':'Arr_Evts_count_d2'})
    X_encoded = pd.merge(X_encoded, X_events_Arr, how='left',sort=False,
        left_on=['DateOfDeparture', 'Arrival'],right_on=['DateOfDeparture', 'Arrival'])
    X_encoded['Arr_Evts_count_d2'].fillna(0, inplace=True)

    X_encoded['event_in_next_days']=X_encoded['Arr_Evts_count_d0']+ \
                                    X_encoded['Arr_Evts_count_d1']+X_encoded['Arr_Evts_count_d2']

    X_encoded = X_encoded.drop(['Arr_Evts_count_d0','Arr_Evts_count_d1','Arr_Evts_count_d2'], axis=1)
    X_encoded = X_encoded.drop(['Date_dt_x','Date_dt_y','Date_dt'], axis=1)

    # -----------------------------------------
    # Feature encoding
    # -----------------------------------------
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
    #X_encoded['Departure_enc'] = pd.factorize(X_encoded['Departure'])[0]
    #X_encoded['Arrival_enc'] = pd.factorize(X_encoded['Arrival'])[0]

    X_encoded = X_encoded.drop('Departure', axis=1)
    X_encoded = X_encoded.drop('Arrival', axis=1)

    #X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
    X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
    #X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
    #X_encoded = X_encoded.drop('year', axis=1)
    #X_encoded = X_encoded.drop('month', axis=1)
    #X_encoded = X_encoded.drop('day', axis=1)
    #X_encoded = X_encoded.drop('weekday', axis=1)
    #X_encoded = X_encoded.drop('week', axis=1)

    X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
    
    return X_encoded


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        X_encoded = _encode(X_df)
        self.columns = X_encoded.columns
        return self

    def transform(self, X_df):
        X_encoded = _encode(X_df)
        X_empty = pd.DataFrame(columns=self.columns)
        X_encoded = pd.concat([X_empty, X_encoded], axis=0, sort=False)
        X_encoded = X_encoded.fillna(0)

        # Reorder/Pick columns from train
        X_encoded = X_encoded[list(self.columns)]
        # Check that columns of test set are the same than train set
        assert list(X_encoded.columns) == list(self.columns)
        X_array = X_encoded.values
        return X_array
