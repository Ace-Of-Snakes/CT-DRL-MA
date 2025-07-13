import pandas as pd
from datetime import time, timedelta, datetime
from datetime import date
import numpy as np
from meteostat import Point, Daily, Hourly
import gc
import openmeteo_requests
from pytorch_tabular import TabularModel
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import date, timedelta
import holidays
from pytorch_tabular import TabularModel
import joblib
import os
from datetime import date
import bisect
import warnings
warnings.filterwarnings("ignore")




def update_interval_probabilities(interval_probs_list):
    """
    Aktualisiert die Wahrscheinlichkeiten für jedes Intervall basierend auf der aktuellen Zeit.
    Entfernt vergangene Intervalle und fügt entsprechend neue Intervalle hinzu,
    um die Gesamtdauer der Prognose konstant zu halten.

    :param interval_probs_list: Liste von Dictionaries mit 'interval_probs' für jeden Container.
    :return: Tuple mit aktualisierter interval_probs_list und results_df.
    """

    current_time = datetime.now(pytz.timezone('Europe/Berlin'))
    current_time = current_time.replace(minute=0, second=0, microsecond=0)
    updated_interval_probs_list = []

    # Extrahieren der ursprünglichen Zeitintervalle aus der ersten Eintragung
    if len(interval_probs_list) > 0:
        # Wir nehmen an, dass alle Container die gleichen Zeitintervalle haben
        original_time_intervals = sorted(interval_probs_list[0]['interval_probs'].keys())
        interval_duration = original_time_intervals[1] - original_time_intervals[0]
    else:
        print("Die interval_probs_list ist leer.")
        return [], pd.DataFrame()

    for container in interval_probs_list:
        interval_probs_dict = container['interval_probs']  # {datetime: probability}

        # Sortiere die Zeitintervalle
        sorted_intervals = sorted(interval_probs_dict.keys())

        # Anzahl der vergangenen Intervalle ermitteln
        num_passed_intervals = sum(1 for ti in sorted_intervals if ti < current_time)

        # Entferne vergangene Intervalle
        remaining_intervals = [ti for ti in sorted_intervals if ti >= current_time]
        remaining_probs = [interval_probs_dict[ti] for ti in remaining_intervals]

        # Summe der verbleibenden Wahrscheinlichkeiten berechnen
        total_prob = sum(remaining_probs)
        num_remaining = len(remaining_intervals)

        if total_prob > 0:
            # Normiere die verbleibenden Wahrscheinlichkeiten
            normalized_probs = [prob / total_prob for prob in remaining_probs]

            # Neue Intervalle am Ende anfügen mit Wahrscheinlichkeit 0
            num_new_intervals = num_passed_intervals
            last_interval = remaining_intervals[-1] if remaining_intervals else current_time

            new_intervals = [last_interval + interval_duration * (i + 1) for i in range(num_new_intervals)]
            normalized_probs.extend([0.0] * num_new_intervals)
            remaining_intervals.extend(new_intervals)

        else:
            # Fall: Alle ursprünglichen Intervalle sind vergangen oder Wahrscheinlichkeiten summieren sich zu 0
            # Neue Intervalle für den Prognosehorizont erstellen (z.B. 100 Stunden)
            forecast_hours = 100  # Prognosehorizont in Stunden
            num_new_intervals = int(forecast_hours // (interval_duration.total_seconds() / 3600))

            remaining_intervals = [current_time + interval_duration * i for i in range(num_new_intervals)]
            normalized_probs = [1.0 / num_new_intervals for _ in range(num_new_intervals)]

        # Aktualisiertes Dictionary erstellen
        updated_probs_dict = {ti: prob for ti, prob in zip(remaining_intervals, normalized_probs)}

        # Aktualisiere den Container mit den neuen Wahrscheinlichkeiten
        updated_container = container.copy()
        updated_container['interval_probs'] = updated_probs_dict
        updated_interval_probs_list.append(updated_container)

    # Erstellung des updated_results_df
    # Alle eindeutigen Zeitintervalle sammeln und sortieren
    all_time_intervals = set()
    for container in updated_interval_probs_list:
        all_time_intervals.update(container['interval_probs'].keys())
    all_time_intervals = sorted(all_time_intervals)

    # DataFrame initialisieren
    results_df = pd.DataFrame(index=all_time_intervals)

    # Fügen Sie die aktualisierten Wahrscheinlichkeiten für jeden Container hinzu
    for container in updated_interval_probs_list:
        container_index = container['container_index']
        interval_probs_dict = container['interval_probs']
        # Wahrscheinlichkeiten in der Reihenfolge der all_time_intervals extrahieren
        probs = [interval_probs_dict.get(ti, 0.0) for ti in all_time_intervals]
        # Füge die Wahrscheinlichkeiten als Spalte zum DataFrame hinzu
        results_df[container_index] = probs

    return updated_interval_probs_list, results_df

def create_initial_probabilities(random_forest_model, prediction_df, time_intervals):
    # Ihre Implementierung der Funktion
    # Vorhersagen der einzelnen Bäume erhalten
    all_tree_predictions = np.array([
        tree.predict(prediction_df) for tree in random_forest_model.estimators_
    ])  # Form: (n_trees, n_samples)

    # Initialisieren einer Liste zur Speicherung der einzelnen Intervallwahrscheinlichkeiten
    interval_probs_list = []
    
    # Initialisieren des DataFrames mit den Zeitintervallen als Index
    index_intervals = time_intervals[:-1]  # Die Intervalle ohne das letzte Element (rechtes Ende)
    results_df = pd.DataFrame(index=index_intervals)

    # Iteriere über jeden Datenpunkt im Vorhersagedatenframe (jeden Container)
    for idx in range(len(prediction_df)):
        # Extrahiere die Vorhersagewerte der einzelnen Bäume für diesen Datenpunkt (in Stunden)
        predictions_hours = all_tree_predictions[:, idx]

        # Konvertiere die Vorhersagen in absolute Zeitpunkte
        predictions_times = [time_intervals[0] + timedelta(hours=pred) for pred in predictions_hours]
        
        # Erstellen eines Histogramms der Vorhersagen innerhalb der definierten Zeitintervalle
        bins = time_intervals
        hist_counts = np.zeros(len(bins) - 1)

        for pred_time in predictions_times:
            index = bisect.bisect_left(bins, pred_time) - 1
            if 0 <= index < len(hist_counts):
                hist_counts[index] += 1

        # Normiere die Histogrammhöhen, um eine Wahrscheinlichkeitsverteilung zu erhalten
        total_counts = np.sum(hist_counts)
        pdf_values = hist_counts / total_counts if total_counts > 0 else np.zeros_like(hist_counts)

        # Erstellen eines Dictionaries {time_interval_start: probability}
        interval_probs_dict = {bins[i].isoformat(): pdf_values[i] for i in range(len(pdf_values))}
        
        # Speichern der Wahrscheinlichkeiten zusammen mit dem Container-Index
        interval_probs_list.append({
            'container_index': idx,
            'interval_probs': interval_probs_dict
        })

        # Füge die Wahrscheinlichkeiten als neue Spalte in results_df hinzu
        results_df[idx] = pdf_values

    return results_df, interval_probs_list, all_tree_predictions

def create_time_intervals(start_time, time_window_max=160, time_zone='Europe/Berlin'):
    """
    Erstellt eine Liste von datetime-Objekten in der angegebenen Zeitzone, beginnend ab start_time in stündlichen Schritten.

    :param start_time: Startzeitpunkt als datetime-Objekt.
    :param time_window_max: Maximale Zeitspanne (in Stunden).
    :param time_zone: Zeitzone (z.B. 'Europe/Berlin').
    :return: Liste von datetime-Objekten.
    """
    # Zeitzone zuweisen, falls nicht bereits vorhanden
    if start_time.tzinfo is None:
        tz = pytz.timezone(time_zone)
        start_time = tz.localize(start_time)
    else:
        tz = start_time.tzinfo

    # Auf die vorherige volle Stunde abrunden
    start_time = start_time.replace(minute=0, second=0, microsecond=0)

    time_intervals = [start_time + timedelta(hours=i) for i in range(time_window_max + 1)]
    
    return time_intervals

def create_probability_matrix_using_intervals(interval_probs_list):
    # Ihre Implementierung der Funktion
    n_containers = len(interval_probs_list)
    
    # Annahme: Alle Container haben die gleichen Zeitintervalle
    time_intervals = None
    for item in interval_probs_list:
        interval_probs = item['interval_probs']
        if time_intervals is None:
            time_intervals = sorted(interval_probs.keys())
        break  # Da alle Container die gleichen Zeitintervalle haben, können wir abbrechen
    
    n_intervals = len(time_intervals)
    
    # Aufbau der Intervallwahrscheinlichkeitsmatrix
    # Jede Zeile entspricht einem Container, die Spalten den Zeitintervallen
    interval_probs_array = np.zeros((n_containers, n_intervals))
    for idx, item in enumerate(interval_probs_list):
        interval_probs_dict = item['interval_probs']
        # Sicherstellen, dass die Wahrscheinlichkeiten in der Reihenfolge der Zeitintervalle sind
        interval_probs_array[idx, :] = [interval_probs_dict[ti] for ti in time_intervals]

    # Kumulative Summe von rechts nach links für s > t
    cumulative_pj = np.cumsum(interval_probs_array[:, ::-1], axis=1)[:, ::-1] - interval_probs_array  # Shape: (n_containers, n_intervals)

    # Erster Term: P_ij_term1 = p_i(t) * Sum_{s>t} p_j(s)
    P_ij_term1 = interval_probs_array @ cumulative_pj.T  # Shape: (n_containers, n_containers)
    
    # Zweiter Term: P_ij_term2 = 0.5 * Sum_t p_i(t) * p_j(t)
    P_ij_term2 = 0.5 * (interval_probs_array @ interval_probs_array.T)  # Shape: (n_containers, n_containers)

    # Kombination beider Terme
    P_ij_matrix = P_ij_term1 + P_ij_term2  # Shape: (n_containers, n_containers)

    # Diagonalelemente auf 0 setzen (da ein Container nicht vor sich selbst entnommen wird)
    np.fill_diagonal(P_ij_matrix, 0)

    return P_ij_matrix


def load_rf_model(file_path_model ='/home/oschmitz/workspace/Standzeitprognose/RandomForest/random_forest_model.joblib'):
    # Modell laden
    rf = joblib.load(file_path_model)

    # Überprüfen
    print("Modell erfolgreich geladen:", rf)
    
    return rf

def read_data(file_path_data):
        # Überprüfen des Dateiformats
    if file_path_data.endswith('.csv'):
        df = pd.read_csv(file_path_data)
    elif file_path_data.endswith('.xlsx'):
        df = pd.read_excel(file_path_data )
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    if any(col.lower() == 'seq' for col in df.columns):
        df = df.drop(columns=[col for col in df.columns if col.lower() == 'seq'])
   
    df = df.sort_values(by=['REALIZED_ARRIVAL_TRAIN'])

    df.rename(columns={'BAHNOPERATEUR': 'TRAIN_OPERATOR'}, inplace=True)
         
    # remove column with pickup plannned, as those are not relevant to dwell time prediction
    df = df[df['PICKUP_PLANNED'].isna()]
    
    # sort by date and time of arrival and by container number   
    df = df.sort_values(by=['REALIZED_ARRIVAL_TRAIN'])
    df = df.reset_index(drop=True)
    
    #Remove columns wihtout meaningful data
    df.drop(columns=['REFERENCE_NUMBER'], inplace=True)
    
    # Wandele alle Spaltennamen in lowercase um
    df.columns = df.columns.str.lower()
    
    return(df)


def one_hot_encode_test(test_df):

    # Kontinuierliche Merkmale definieren
    continuous_cols = [
    'stack_utilization_average_arrival', 'stack_utilization_average_weekly',
    'stack_utilization_average_monthly', 'dwell_time_average_arrival',
    'dwell_time_average_weekly', 'dwell_time_average_monthly', 'train_delay',
    'weight', 'wait_time_until_open', 'time_since_close',
    'time_until_next_rush_hour', 'realized_arrival_hour_sin',
    'realized_arrival_hour_cos', 'realized_arrival_weekday_sin',
    'realized_arrival_weekday_cos', 'realized_arrival_month_sin',
    'realized_arrival_month_cos', 'realized_arrival_week_sin',
    'realized_arrival_week_cos'
    ]

    # Kategorische Merkmale definieren
    categorical_cols = [
    'dangerous_goods', 'custom', 'waste', 'full_empty', 'origin',
    'modul', 'day_of_arrival', 'month_of_arrival',
    'equipment_identifier', 'weather_today', 'weather_prev_day',
    'weather_next_day_1', 'weather_next_day_2', 'weather_next_day_3',
    'weather_next_day_4', 'weather_next_day_5', 'realized_arrival_day_section',
    'planned_arrival_day_section', 'is_terminal_open',
    'is_planned_terminal_open', 'is_arrival_day_holiday',
    'is_next_day_holiday', 'is_day_after_next_holiday',
    'is_day_before_holiday', 'is_arrival_day_weekend',
    'is_next_day_weekend', 'is_day_after_next_weekend',
    'is_day_before_weekend', 'not_stackable', 'not_stackable_dbi_hel',
    'adjusted_container_type', 'delay_classification',
    'adjusted_train_operator', 'adjusted_owner_prefix',
    'adjusted_ship_company'
    ]   
    
    # Entfernen der Zielspalte 'dwelltime_hours' aus continuous_cols, falls vorhanden
    continuous_cols = [col for col in continuous_cols if col != 'dwelltime_hours']
    
    # encoded Trainingsdaten laden
    train_df = pd.read_csv('/home/oschmitz/workspace/Standzeitprognose/Daten/Training_dataset_columns_encoded_RF.csv')
    train_df = train_df.drop(columns = ['Unnamed: 0'])   
    train_columns = train_df.columns

    # One-Hot-Encoding auf den Testdaten
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
    
    # Fehlende Spalten im Testdatensatz hinzufügen
    missing_cols = set(train_columns) - set(test_encoded.columns)
    #if missing_cols:
        #print('Es fehlen Spalten im Dataframe')
        #print(missing_cols)
    
    for col in missing_cols:
        test_encoded[col] = 0

    # Zusätzliche Spalten im Testdatensatz entfernen
    extra_cols = set(test_encoded.columns) - set(train_columns)
    if extra_cols:
        test_encoded = test_encoded.drop(columns=extra_cols)

    # Sicherstellen, dass die Spaltenreihenfolge im Testdatensatz mit dem Trainingsdatensatz übereinstimmt
    test_encoded = test_encoded[train_columns]
    
    if False in (train_df.columns == test_encoded.columns):
        raise ValueError('Fehler mit der Spaltenaufbereitung')
    
    return test_encoded


def clean_containernumbers(df):
    # remove containernumber for trailers and wechselbrücken and fill up with 'WB/Trailer'
    # extract information on container owner (owner prefix) and equipment type of container based on containernumber
    mask = df['container_number'].str.len() != 13
    df.loc[mask, 'container_number'] = 'WB/Trailer'
    
    def split_fixed_positions(column):
        return column.apply(lambda x: ['unknown', 'WB/Trailer'] if x == 'WB/Trailer' else [x[:3], x[3:4]])

    df[['container_owner_prefix', 'equipment_identifier']] = split_fixed_positions(df['container_number']).tolist()
    
    common_identifiers = ['U', 'D', 'E', 'B', 'A', '0', '7', '4', '3', '6', '1', '2', '5', '9','8']
    df['equipment_identifier'] = df['equipment_identifier'].apply(lambda x: x if x in common_identifiers else 'Andere')
    
    return(df)

def convert_to_datetime(df):
    # function to convert all coluns with some sort of date or time content to datetime format
    
    # list of columns to convert
    columns_to_convert = ['planned_arrival_train','realized_arrival_train', 'pickup_planned']
    
    # conversion loop
    for column in columns_to_convert:
        df[column] = pd.to_datetime(df[column], format='%d.%m.%Y %H:%M')  
       
    return(df)

def clean_train_origin(df):
    common_origins = ['Hamburg', 'Bremerhaven', 'Lehrte', 'Rotterdam', 'Verona', 'Wuppertal', 'Unbekannt', 'Trieste', 'Wilhelmshaven', 'Nürnberg']
    df['origin'] = df['origin'].apply(lambda x: x if x in common_origins else 'Andere')
    
    return(df)

def datetime_calculations(df):
    # function to do all sort of calculations based on datetime columns
    # create train delay
    # create column for day/month of arrival
    # round train delay to full hour in order to improve classification performance
    # create dwell time hours
    
    # create train delay in hours
    df['train_delay'] = (df['realized_arrival_train'] - df['planned_arrival_train']).dt.total_seconds() / 3600
    
    ######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #############
    ## In API entfernen, es ging nur darum einen Ausreißer zu filtern ##
    df = df[df['train_delay'] < 200]
    ######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #############


    # create column with weekday and month of delivery
    df['day_of_arrival'] = df['realized_arrival_train'].dt.day_name()
    df['month_of_arrival'] = df['realized_arrival_train'].dt.month_name()
    
    
    # Define the columns to round
    columns_to_round = [
        'dwell_time_average_arrival', 
        'dwell_time_average_weekly', 
        'train_delay',
        'dwell_time_average_monthly'
        ]

    # Round the values in these columns using .loc to avoid SettingWithCopyWarning
    df.loc[:, columns_to_round] = df[columns_to_round].round()
    
    # Erstelle die Funktion, die die Uhrzeit in eine Klasse unterteilt
    def classify_time(hour):
        if 0 <= hour < 6:
            return 'night'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:
            return 'evening'

    # Extrahiere die Stunde aus der 'realized_arrival_train'-Spalte
    df['hour_realized_arrival'] = df['realized_arrival_train'].dt.hour
    df['hour_planned_arrival'] = df['planned_arrival_train'].dt.hour

    # Wende die Funktion an, um die Zeitklassen zu erstellen
    df['realized_arrival_section_of_the_day'] = df['hour_realized_arrival'].apply(classify_time)
    df['planned_arrival_section_of_the_day'] = df['hour_planned_arrival'].apply(classify_time)

    # Neue Spalte mit Wochentag erstellen
    df['planned_weekday'] = df['planned_arrival_train'].dt.day_name()
    df['realized_weekday'] = df['realized_arrival_train'].dt.day_name()

    # merge columns together
    df['realized_arrival_day_section'] = df['realized_weekday'] + ' ' + df['realized_arrival_section_of_the_day']
    df['planned_arrival_day_section'] = df['planned_weekday'] + ' ' + df['planned_arrival_section_of_the_day']

    # Drop intermediate columns if needed
    df = df.drop(columns=['planned_weekday','realized_weekday','realized_arrival_section_of_the_day',
                          'planned_arrival_section_of_the_day',
                          'hour_realized_arrival','hour_planned_arrival'
                          ])
    ##### 'weekday','weekday_pickup',
    
    #classify train delay
    def classify_delay(delay):
        if delay < -20:
            return 'Extrem zu früh'
        elif -20 <= delay < -10:
            return 'Stark zu früh'
        elif -10 <= delay < -5:
            return 'Zu früh'
        elif -5 <= delay < -3:
            return 'Leicht zu früh'
        elif -3 <= delay < -1:
            return 'fast pünktlich, zu früh'
        elif -1 <= delay <= 1:
            return 'Pünktlich'
        elif 1 < delay <= 3:
            return 'fast pünktlich, zu spät'
        elif 3 < delay <= 5:
            return 'Leicht verspätet'
        elif 5 < delay <= 10:
            return 'Verspätet'
        elif 10 < delay <= 20:
            return 'Stark verspätet'
        else:  # delay > 20
            return 'Extrem verspätet'


    # Neue Spalte mit klassifizierter Verspätung erstellen
    df['delay_classification'] = df['train_delay'].apply(classify_delay)
    
    # Funktion zum Runden je nach Zeitdifferenz
    def round_time(dt):
        if dt.minute < 60:
            return dt.floor('h')  # Abwärts runden
        else:
            return dt.ceil('h')   # Aufwärts runden

    # Neue Spalte mit gerundeter Zeit erstellen
    df['rounded_realized_arrival_train'] = df['realized_arrival_train'].apply(round_time)


    df['rounded_realized_arrival_train'] = pd.to_datetime(df['rounded_realized_arrival_train'])

    df['realized_arrival_weekday'] = df['rounded_realized_arrival_train'].dt.weekday  # 0=Monday, 6=Sunday
    df['realized_arrival_hour'] = df['rounded_realized_arrival_train'].dt.hour
    df['realized_arrival_week'] = df['rounded_realized_arrival_train'].dt.isocalendar().week
    df['realized_arrival_month'] = df['rounded_realized_arrival_train'].dt.month


    # Zyklische Transformation für die Stunde (24 Stunden)
    df['realized_arrival_hour_sin'] = np.sin(2 * np.pi * df['realized_arrival_hour'] / 24)
    df['realized_arrival_hour_cos'] = np.cos(2 * np.pi * df['realized_arrival_hour'] / 24)

    # Zyklische Transformation für den Wochentag (7 Tage)
    df['realized_arrival_weekday_sin'] = np.sin(2 * np.pi * df['realized_arrival_weekday'] / 7)
    df['realized_arrival_weekday_cos'] = np.cos(2 * np.pi * df['realized_arrival_weekday'] / 7)

    # Zyklische Transformation für Monate
    df['realized_arrival_month_sin'] = np.sin(2 * np.pi * df['realized_arrival_month'] / 12)
    df['realized_arrival_month_cos'] = np.cos(2 * np.pi * df['realized_arrival_month'] / 12)

    # Zyklische Transformation für die Kalenderwoche (52 Wochen)
    df['realized_arrival_week_sin'] = np.sin(2 * np.pi * df['realized_arrival_week'] / 52)
    df['realized_arrival_week_cos'] = np.cos(2 * np.pi * df['realized_arrival_week'] / 52)


    # Berechnung der kontinuierlichen Stunden in der Woche
    df['realized_arrival_hour_of_week'] = df['realized_arrival_weekday'] * 24 + df['realized_arrival_hour']

    # Zyklische Transformation für die Stunde der Woche (168 Stunden)
    df['realized_arrival_hour_of_week_sin'] = np.sin(2 * np.pi * df['realized_arrival_hour_of_week'] / 168)
    df['realized_arrival_hour_of_week_cos'] = np.cos(2 * np.pi * df['realized_arrival_hour_of_week'] / 168)

    df['realized_arrival_hour_sin'] = df['realized_arrival_hour_sin'].round(4)
    df['realized_arrival_hour_cos'] = df['realized_arrival_hour_cos'].round(4)
    df['realized_arrival_weekday_sin'] = df['realized_arrival_weekday_sin'].round(4)
    df['realized_arrival_weekday_cos'] =  df['realized_arrival_weekday_cos'].round(4)
    df['realized_arrival_month_sin'] = df['realized_arrival_month_sin'].round(4)
    df['realized_arrival_month_cos'] = df['realized_arrival_month_cos'].round(4)
    df['realized_arrival_week_sin'] = df['realized_arrival_week_sin'].round(4)
    df['realized_arrival_week_cos'] = df['realized_arrival_week_cos'].round(4)
    
    df = df.drop(columns = ['realized_arrival_weekday', 'realized_arrival_hour', 'realized_arrival_week', 'realized_arrival_month'])
    
    return(df)

def adjust_operator(df):
    common_operators = ['DBI', 'BOX', 'MET', 'HEL', 'EGS', 'IGS', 'KOMBI', 'ALP', 'MTL', 'BEL']
    
    df['train_operator'] = df['train_operator'].apply(
        lambda operator: operator if operator in common_operators else 'Andere'
    )
    df.rename(columns={'train_operator': 'adjusted_train_operator'}, inplace=True)
    
    return(df)

def adjust_container_owner(df):
    common_owners = ['DHL', 'MRK', 'MSK', 'TIP', 'unknown', 'TCL', 'MRS', 'HLB', 'TCN',
       'SEG', 'AXI', 'TRH', 'CMA', 'TGB', 'UAC', 'TEM', 'CAI', 'OOL', 'MSD',
       'MED', 'TLL', 'MSM', 'WEB', 'TCK', 'FCI', 'FAN', 'HLX', 'BMO', 'OOC',
       'CSN', 'FFA', 'SUD', 'GES', 'BEA', 'TGH', 'HAS', 'KOC', 'KEI', 'LOE',
       'EIT', 'CAA', 'PNO', 'DFS', 'FCN', 'EGH', 'FSC', 'NYK', 'HEL', 'PON',
       'COB', 'IDS', 'BSI', 'GCX', 'UET', 'EMC', 'DRY', 'EIS', 'SEK', 'HMM',
       'YMM', 'GLD', 'SCT', 'HAM', 'REI', 'CCL', 'APZ', 'TRL', 'ONE', 'MSC',
       'TXG', 'HDM', 'CBH', 'CSL', 'YML', 'MAG', 'GAO', 'TTN', 'LTR', 'GLS',
       'FDC', 'SCH', 'UES', 'SCZ', 'MAE', 'KKF', 'CXD', 'APH', 'EGS', 'MSN',
       'HTL', 'KKT', 'HKP', 'HNK', 'WCA', 'TGC', 'FBI', 'HMC', 'CAX']
    
    df['container_owner_prefix'] = df['container_owner_prefix'].apply(
        lambda owner: owner if owner in common_owners else 'Andere'
    )
    
    df.rename(columns={'container_owner_prefix': 'adjusted_owner_prefix'}, inplace=True)

    return(df)
    
def adjust_shipper(df):
    common_shippers = ['PIC','MET','HEL','TFG','EGS','MAE-HAM','TX','EU','IGS','ERS-ROT','KOMBI','ALP','MTL','BEL','DBI','BOX']
    
    df['ship_company'] = df['ship_company'].apply(lambda shipper: shipper if shipper in common_shippers else 'Andere')
    df.rename(columns={'ship_company': 'adjusted_ship_company'}, inplace=True)

    return(df)    

def create_dwelltime(df):
    # create dwell time hours
    # list of columns to convert
    columns_to_convert = ['pickup_realized','realized_arrival_train']

    # conversion loop
    for column in columns_to_convert:
        df[column] = pd.to_datetime(df[column], format='%d.%m.%Y %H:%M')

    # Berechnen Sie die Differenz zwischen den Spalten
    df['dwelltime_hours'] = df['pickup_realized'] - df['realized_arrival_train']

    # Erhöhen Sie alle Werte, die kleiner als 1 Stunde sind, auf 1 Stunde
    df['dwelltime_hours'] = df['dwelltime_hours'].apply(lambda x: pd.Timedelta(hours=1) if x < pd.Timedelta(hours=1) else x)

    # Konvertieren Sie die Differenz in Stunden und runden Sie auf die nächste Ganzzahl
    df['dwelltime_hours'] = (df['dwelltime_hours'].dt.total_seconds() // 3600).astype(int)
    
    return(df)

def adjust_container_types(df):
    # Dictionary for the custom mappings of unpopular container types
    container_mapping = {
        '40T': '40',
        '20HC': '20',
        '40PW': '40',
        '24JB': '20',
        '45DC': '45',
        '20WB': '20',
        '20GG': '20',
        '23T': '20',
        '40HH': '40',
        '40WB': '40',
        '20B': '40',
        '30GH': '40',
        '45GW': '45',
        '45GB': '45',
        '40H': '40',
        '26T': '20',
        '20RH': '20',
        '20H': '20',
        '20OH': '20'
    }

    # Filter for container types that occur more than 100 times and store them in a list
    popular_containers = [
                        '40GH', '23WB', '20G', '40HC', '40G', 'MTRA', '20GV', '20DC', '30O', '40DC', '45WB', '45GH',
                        'TRA',  '24T',  '24WB', '40RH', '45HC', '24OT', '40O', '22WB', '22T', '30T', '20O', '30B',
                        '30SI', '30G', '40OH', '40R', '30WB', '20GH', '45G', '20R', '20T']
    
    # Function to adjust container types
    def adjust_container_type(container_type):
        if container_type in popular_containers:
            return container_type
        return container_mapping.get(container_type, container_type)  # Return original if no match in dictionary

    # Create the new column 'adjusted_container_type'
    df['container_type'] = df['container_type'].apply(adjust_container_type)
    df.rename(columns={'container_type': 'adjusted_container_type'}, inplace=True)

    return(df)

def create_binaries(df):
    # conversion of all ja/nein values to binary
    
    # List of columns to transform
    columns_to_transform = ['dangerous_goods', 'custom', 'waste']
    
    # Mapping dictionaries for binary conversion
    # map J to 1, N to 0
    binary_map = {'J': 1, 'N': 0}
    # transform voll & leer into 1&0
    another_map={'full':1, 'empty':0}
    
    
    # Apply the mapping to the specified columns
    df[columns_to_transform] = df[columns_to_transform].apply(lambda x: x.map(binary_map)) 
    df['full_empty']=df['full_empty'].map(another_map)
    
    return(df)


def fetch_weather_data(df):
    # infos on https://open-meteo.com/en/docs
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    
    #today = date.today()
    first_day = df['realized_arrival_train'].min().date()
    last_day = df['realized_arrival_train'].max().date()
        
    # API-Parameter setzen
    #url = "https://api.open-meteo.com/v1/forecast"    ####### alte url, halluzination???
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": 49.398042,
        "longitude": 11.05495,
        "daily": "weather_code",
        "timezone": "Europe/Berlin",
        "start_date": (first_day - timedelta(days=2)).isoformat(),
        "end_date": (last_day + timedelta(days=6)).isoformat(), # !!!!!!!!!!!!!!!!!!!!!!!! Anpassen auf heute + 6
        "daily": "weather_code",
    }
    
    # Wetterdaten abrufen
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

      # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_weather_code = daily.Variables(0).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
    )}
    
    daily_data["weather_today"] = daily_weather_code

    daily_dataframe = pd.DataFrame(data = daily_data)

    # Add columns for the previous day and the next 5 days
    daily_dataframe['weather_prev_day'] = daily_dataframe['weather_today'].shift(1)
    for i in range(1, 6):
        daily_dataframe[f'weather_next_day_{i}'] = daily_dataframe['weather_today'].shift(-i)

    #daily_dataframe  = daily_dataframe.rename(columns= {'weather_code':'weather_today'})

    #map weather into less granular categories

    # infos on categories: https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM

    # Erstellen eines Dictionary für die vereinfachten Wetterkategorien
    simplified_weather_categories = {
        **{code: "Klares Wetter" for code in range(0, 20)},
        **{code: "Regnerisch" for code in range(20, 30)},
        **{code: "Stürmisch" for code in range(30, 40)},
        **{code: "Neblig" for code in range(40, 50)},
        **{code: "Niesel" for code in range(50, 60)},
        **{code: "Regen" for code in range(60, 70)},
        **{code: "Schnee" for code in range(70, 80)},
        **{code: "Unwetter" for code in range(80, 100)},
    }

    daily_dataframe['date'] = daily_dataframe['date'].dt.tz_localize(None)

    # Anwenden des Mappings auf alle relevanten Spalten
    columns_to_map = ['weather_today', 'weather_prev_day'] + [f'weather_next_day_{i}' for i in range(1, 6)]

    for column in columns_to_map:
        daily_dataframe[column] = daily_dataframe[column].map(simplified_weather_categories)

    # Extrahieren des Datums (ohne Uhrzeit) aus beiden Spalten
    daily_dataframe['date_only'] = daily_dataframe['date'].dt.date
    df['arrival_date_only'] = df['realized_arrival_train'].dt.date

    # Zusammenführen der DataFrames basierend auf den extrahierten Datumswerten
    df = df.merge(
        daily_dataframe, 
        left_on='arrival_date_only', 
        right_on='date_only', 
        how='left'
    )

    # Entfernen der Hilfsspalten, falls nicht mehr benötigt
    df = df.drop(columns=['date_only', 'arrival_date_only','date'])
    
    return df

def drop_unused_columns(df):
    # drop all columns not needed anymore
    df = df.drop(columns = ['pickup_realized','container_number','pickup_planned',
                            'rounded_realized_arrival_train','realized_arrival_hour_of_week',
                            'realized_arrival_hour_of_week_sin','realized_arrival_hour_of_week_cos'
                            ])  
    # 'planned_arrival_train','realized_arrival_train',
    ### 'train_id','weekday_arrival'
    
    # drop only because at the moment not availiable
    df = df.drop(columns = ['content','trucking_company'])
    
    return(df)


def drop_pickup_planned_rows(df):
    df = df[df['pickup_planned'].isna()]
    return df

def add_time_features(df):
    # Definiere die Zeitabschnitte und Beschriftungen
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    labels = [
        'night_early', 'night_late', 'morning_early', 'morning_late',
        'afternoon_early', 'afternoon_late', 'evening_early', 'evening_late'
    ]
    
    # Extrahiere die Stunde aus den Ankunftsspalten
    df['hour_realized_arrival'] = df['realized_arrival_train'].dt.hour
    df['hour_planned_arrival'] = df['planned_arrival_train'].dt.hour

    # Klassifiziere die Stunden zu Zeitabschnitten
    df['realized_arrival_section_of_the_day'] = pd.cut(
        df['hour_realized_arrival'], bins=bins, labels=labels, right=False, include_lowest=True
    )
    df['planned_arrival_section_of_the_day'] = pd.cut(
        df['hour_planned_arrival'], bins=bins, labels=labels, right=False, include_lowest=True
    )

    # Erstelle neue Spalten mit dem Wochentag
    df['planned_weekday'] = df['planned_arrival_train'].dt.day_name()
    df['realized_weekday'] = df['realized_arrival_train'].dt.day_name()

    # Kombiniere Wochentag und Tagesabschnitt
    df['realized_arrival_day_section'] = df['realized_weekday'] + ' ' + df['realized_arrival_section_of_the_day'].astype(str)
    df['planned_arrival_day_section'] = df['planned_weekday'] + ' ' + df['planned_arrival_section_of_the_day'].astype(str)

    # Entferne unnötige Spalten
    df = df.drop(columns=[
        'hour_realized_arrival', 'hour_planned_arrival',
        'realized_arrival_section_of_the_day', 'planned_arrival_section_of_the_day',
        'planned_weekday', 'realized_weekday'
    ])

    return df

def add_terminal_open_status(df):
    def is_terminal_open(arrival_time):
        day = arrival_time.day_name()
        hour = arrival_time.hour + arrival_time.minute / 60  # Berücksichtigt die Minuten
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            return 6 <= hour < 21
        elif day == 'Saturday':
            return 6 <= hour < 11
        else:
            return False  # Sonntag ist geschlossen

    df['is_terminal_open'] = df['realized_arrival_train'].apply(is_terminal_open)
    return df

def add_wait_time_until_open(df):
    def is_terminal_open(arrival_time):
        day = arrival_time.day_name()
        hour = arrival_time.hour + arrival_time.minute / 60
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            return 6 <= hour < 21
        elif day == 'Saturday':
            return 6 <= hour < 11
        else:
            return False

    def time_until_open(arrival_time):
        if is_terminal_open(arrival_time):
            return 0
        else:
            day = arrival_time.day_name()
            hour = arrival_time.hour + arrival_time.minute / 60

            if day in ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
                # Nächste Öffnungszeit bestimmen
                next_opening = arrival_time.replace(hour=6, minute=0, second=0, microsecond=0)
                if (day == 'Saturday' and hour >= 11) or (day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] and hour >= 21):
                    next_opening += pd.Timedelta(days=1)
                elif day == 'Saturday' and hour < 6:
                    next_opening = arrival_time.replace(hour=6, minute=0, second=0, microsecond=0)
                elif day == 'Sunday':
                    next_opening += pd.Timedelta(days=1)
                wait_time = (next_opening - arrival_time).total_seconds() / 3600  # in Stunden
                return wait_time
            else:
                return None  # Für den unwahrscheinlichen Fall eines unerwarteten Tageswertes

    df['wait_time_until_open'] = df['realized_arrival_train'].apply(time_until_open).round(2)
    return df

def add_planned_terminal_open_status(df):
    def is_terminal_open(arrival_time):
        day = arrival_time.day_name()
        hour = arrival_time.hour + arrival_time.minute / 60
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            return 6 <= hour < 21
        elif day == 'Saturday':
            return 6 <= hour < 11
        else:
            return False

    df['is_planned_terminal_open'] = df['planned_arrival_train'].apply(is_terminal_open)
    return df

def add_time_since_close(df):
    def is_terminal_open(arrival_time):
        day = arrival_time.day_name()
        hour = arrival_time.hour + arrival_time.minute / 60
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            return 6 <= hour < 21
        elif day == 'Saturday':
            return 6 <= hour < 11
        else:
            return False

    def time_since_close(arrival_time):
        if is_terminal_open(arrival_time):
            return 0
        else:
            day = arrival_time.day_name()
            hour = arrival_time.hour + arrival_time.minute / 60

            if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                if hour < 6:
                    last_closing = arrival_time.replace(hour=21, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
                else:
                    last_closing = arrival_time.replace(hour=21, minute=0, second=0, microsecond=0)
            elif day == 'Saturday':
                if hour < 6:
                    last_closing = arrival_time.replace(hour=21, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
                elif hour >= 11:
                    last_closing = arrival_time.replace(hour=11, minute=0, second=0, microsecond=0)
                else:
                    return 0
            else:  # Sonntag
                last_closing = arrival_time.replace(hour=11, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)

            time_since_closed = (arrival_time - last_closing).total_seconds() / 3600
            return time_since_closed

    df['time_since_close'] = df['realized_arrival_train'].apply(time_since_close).round(2)
    return df


def add_holiday_features(df):
    # Deutsche Feiertage für Bayern laden
    de_holidays = holidays.country_holidays('DE', subdiv='BY')

    df['is_arrival_day_holiday'] = df['realized_arrival_train'].dt.date.apply(lambda x: x in de_holidays)
    df['is_next_day_holiday'] = (df['realized_arrival_train'] + pd.Timedelta(days=1)).dt.date.apply(lambda x: x in de_holidays)
    df['is_day_after_next_holiday'] = (df['realized_arrival_train'] + pd.Timedelta(days=2)).dt.date.apply(lambda x: x in de_holidays)
    df['is_day_before_holiday'] = (df['realized_arrival_train'] + pd.Timedelta(days=-1)).dt.date.apply(lambda x: x in de_holidays)
    return df

def add_time_until_next_rush_hour(df):
    def time_until_rush_hour(arrival_time):
        day = arrival_time.weekday()  # Montag=0, Sonntag=6
        hour = arrival_time.hour + arrival_time.minute / 60

        if day < 4:  # Montag bis Mittwoch
            rush_hour = arrival_time.replace(hour=11, minute=0, second=0, microsecond=0)
            if hour >= 11:
                rush_hour += pd.Timedelta(days=1)
        elif day == 4:  # Donnerstag
            rush_hour = arrival_time.replace(hour=11, minute=0, second=0, microsecond=0)
            if hour >= 11:
                rush_hour += pd.Timedelta(days=4)  # Nächste Rush Hour am Montag
        elif day == 5:  # Freitag
            if hour < 14:
                rush_hour = arrival_time.replace(hour=14, minute=0, second=0, microsecond=0)
            else:
                rush_hour = arrival_time.replace(hour=17, minute=0, second=0, microsecond=0)
                if hour >= 17:
                    rush_hour += pd.Timedelta(days=3)  # Nächste Rush Hour am Montag
                    rush_hour = rush_hour.replace(hour=11)
        else:  # Samstag und Sonntag
            rush_hour = arrival_time.replace(hour=21, minute=0, second=0, microsecond=0)
            if hour >= 21:
                rush_hour += pd.Timedelta(days=(7 - day))  # Nächste Rush Hour am Montag
                rush_hour = rush_hour.replace(hour=11)

        time_until_rush = (rush_hour - arrival_time).total_seconds() / 3600
        return time_until_rush

    df['time_until_next_rush_hour'] = df['realized_arrival_train'].apply(time_until_rush_hour).round(2)
    return df

def add_weekend_features(df):
    def is_weekend(date):
        return date.weekday() >= 5  # Samstag (5) oder Sonntag (6)

    df['is_arrival_day_weekend'] = df['realized_arrival_train'].dt.date.apply(is_weekend)
    df['is_next_day_weekend'] = (df['realized_arrival_train'] + pd.Timedelta(days=1)).dt.date.apply(is_weekend)
    df['is_day_after_next_weekend'] = (df['realized_arrival_train'] + pd.Timedelta(days=2)).dt.date.apply(is_weekend)
    df['is_day_before_weekend'] = (df['realized_arrival_train'] - pd.Timedelta(days=1)).dt.date.apply(is_weekend)
    return df

def add_not_stackable(df):
    # Neue Spalte 'not_stackable' erstellen
    df['not_stackable'] = df['container_type'].apply(lambda x: 'WB' in x or 'TRA' in x or 'MTRA' in x)
  
    # Kombinierte Spalte für nicht stapelbar und 'DBI' oder 'HEL' erstellen
    df['not_stackable_dbi_hel'] = np.where(
        (df['not_stackable']) & (df['adjusted_train_operator'].isin(['HEL', 'DBI'])),
        1,
        0
    )
    
    return df

def remove_outliers(df):
    
    
    limit_dwelltime_hours = 160
    df.loc[df['dwelltime_hours'] > limit_dwelltime_hours, 'dwelltime_hours'] = limit_dwelltime_hours

    #fülle ggf. vorhandene nan werte
    #df['weight'] = df['weight'].fillna(0)
    df = df[df['weight'].notna()]

    limit_weight = 31000.00  
    df.loc[df['weight'] > limit_weight, 'weight'] = limit_weight

    
    return df

def prepare_dataframe(file_path_data):
    df = read_data(file_path_data)
    # Sortieren des DataFrames basierend auf der Spalte 'REALIZED_ARRIVAL_TRAIN'
      
    df = clean_containernumbers(df)
    df = convert_to_datetime(df)
    df = clean_train_origin(df)
    df = datetime_calculations(df)
    df = adjust_operator(df)
    df = adjust_container_owner(df)
    df = adjust_shipper(df)
    df = create_dwelltime(df)
    df = add_not_stackable(df)
    df = adjust_container_types(df)
    df = create_binaries(df)
    df = drop_pickup_planned_rows(df)
    df = fetch_weather_data(df)
    df = add_time_features(df)
    df = add_terminal_open_status(df)
    df = add_wait_time_until_open(df)
    df = add_planned_terminal_open_status(df)
    df = add_time_since_close(df)
    df = add_holiday_features(df)
    df = add_time_until_next_rush_hour(df)
    df = add_weekend_features(df)
    df = drop_unused_columns(df)
    df = remove_outliers(df)
    
    return df