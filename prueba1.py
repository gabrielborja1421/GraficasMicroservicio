from flask import Flask, request, jsonify
import pandas as pd
import requests
from requests.exceptions import RequestException
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el ID de usuario del JSON recibido
    data = request.get_json()
    user_id = data['userid']
    
    try:
        # Hacer una solicitud GET a la API de ejercicios
        exercises_response = requests.get(f'https://entrenat.ddns.net/muscle/arm/list/{user_id}')
        exercises_response.raise_for_status()  # Esto lanzará una excepción si la solicitud no fue exitosa
    except RequestException as e:
        return jsonify({'error': f'Failed to fetch data from exercises API: {str(e)}'}), 500

    exercises_data = exercises_response.json()['data']['user']

    # Lista de ejercicios
    ejercicios = [
        'bicepCurl', 'hammerCurl', 'barbellCurl', 'skullcrusher',
        'dumbbellOverheadTricepsExtension', 'tricepsPushdown', 'pushPress',
        'closeGripBenchPress', 'militaryPress', 'lateralRaise',
        'frontRaise', 'reverseFly', 'shoulderPress'
    ]

    # Convertir datos a DataFrame y reemplazar nulos por 0
    df = pd.DataFrame(exercises_data)
    df[ejercicios] = df[ejercicios].fillna(0)

    # Calcular el promedio de los ejercicios por cada registro
    df['average_weight'] = df[ejercicios].mean(axis=1)

    # Filtrar las columnas necesarias
    df = df[['fecha', 'average_weight']]

    # Convertir las fechas a datetime y establecer como índice
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.set_index('fecha', inplace=True)
    
    # Rellenar fechas faltantes con 0s para asegurar una frecuencia diaria
    df = df.asfreq('D', fill_value=0)

    # Verificar si hay suficientes datos para el modelo
    if len(df) < 2:
        return jsonify({'error': 'Not enough data points to make a prediction'}), 400

    # Crear el modelo de Holt-Winters con tendencia aditiva
    model = ExponentialSmoothing(df['average_weight'], trend='add', seasonal=None, seasonal_periods=None)
    model_fit = model.fit()

    # Hacer predicciones
    forecast_steps = 5  # Número de días a predecir
    forecast = model_fit.forecast(steps=forecast_steps)

    # Obtener las fechas de predicción
    future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]

    # Crear un nuevo DataFrame para las predicciones
    forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicción'])

    # Preparar los datos de respuesta
    result = {
        'fechas': df.index.strftime('%Y-%m-%d').tolist() + forecast_df.index.strftime('%Y-%m-%d').tolist(),
        'pesos': df['average_weight'].tolist() + forecast_df['Predicción'].tolist()
    }

    return jsonify(result)

@app.route('/predict_core', methods=['POST'])
def predict_core():
    # Obtener el ID de usuario del JSON recibido
    data = request.get_json()
    user_id = data['userid']
    
    try:
        # Hacer una solicitud GET a la API de ejercicios del core
        exercises_response = requests.get(f'https://entrenat.ddns.net/muscle/core/list/{user_id}')
        exercises_response.raise_for_status()  # Esto lanzará una excepción si la solicitud no fue exitosa
    except RequestException as e:
        return jsonify({'error': f'Failed to fetch data from core exercises API: {str(e)}'}), 500

    exercises_data = exercises_response.json()['data']['user']

    # Lista de ejercicios del core
    ejercicios = [
        'russian_twist', 'reps_russian_twist', 'plank', 'reps_plank', 'crunch', 'reps_crunch'
    ]

    # Convertir datos a DataFrame y reemplazar nulos por 0
    df = pd.DataFrame(exercises_data)
    df[ejercicios] = df[ejercicios].fillna(0)

    # Calcular el promedio de los ejercicios por cada registro
    df['average_weight'] = df[ejercicios].mean(axis=1)

    # Filtrar las columnas necesarias
    df = df[['fecha', 'average_weight']]

    # Convertir las fechas a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Eliminar duplicados basados en la fecha
    df = df.drop_duplicates(subset=['fecha'])

    # Establecer la fecha como índice
    df.set_index('fecha', inplace=True)

    # Imprimir el DataFrame para depuración
    print("DataFrame antes de establecer frecuencia:")
    print(df)
    
    # Asegurar que las fechas están ordenadas
    df = df.sort_index()

    # Rellenar fechas faltantes con 0s para asegurar una frecuencia diaria
    df = df.asfreq('D', fill_value=0)

    # Imprimir el DataFrame después de rellenar fechas para depuración
    print("DataFrame después de establecer frecuencia:")
    print(df)

    # Verificar si hay suficientes datos para el modelo
    print("Número de filas en el DataFrame después de establecer la frecuencia diaria:", len(df))
    if len(df) < 2:
        return jsonify({'error': 'Not enough data points to make a prediction'}), 400

    # Crear el modelo de Holt-Winters con tendencia aditiva
    model = ExponentialSmoothing(df['average_weight'], trend='add', seasonal=None, seasonal_periods=None)
    model_fit = model.fit()

    # Hacer predicciones
    forecast_steps = 5  # Número de días a predecir
    forecast = model_fit.forecast(steps=forecast_steps)

    # Obtener las fechas de predicción
    future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]

    # Crear un nuevo DataFrame para las predicciones
    forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicción'])

    # Preparar los datos de respuesta
    result = {
        'fechas': df.index.strftime('%Y-%m-%d').tolist() + forecast_df.index.strftime('%Y-%m-%d').tolist(),
        'pesos': df['average_weight'].tolist() + forecast_df['Predicción'].tolist()
    }

    return jsonify(result)

@app.route('/predict_chest', methods=['POST'])
def predict_chest():
    # Obtener el ID de usuario del JSON recibido
    data = request.get_json()
    user_id = data['userid']
    
    try:
        # Hacer una solicitud GET a la API de ejercicios del pecho
        exercises_response = requests.get(f'https://entrenat.ddns.net/muscle/chest/list/{user_id}')
        exercises_response.raise_for_status()  # Esto lanzará una excepción si la solicitud no fue exitosa
    except RequestException as e:
        return jsonify({'error': f'Failed to fetch data from chest exercises API: {str(e)}'}), 500

    exercises_data = exercises_response.json()['data']['user']
    print("Datos recibidos de la API:")
    print(exercises_data)

    # Lista de ejercicios del pecho
    ejercicios = [
        'barbellBenchPress', 'reps_barbellBenchPress', 'dumbellBenchPress', 'reps_dumbellBenchPress',
        'inclineBenchPress', 'reps_inclineBenchPress', 'machineChestPress', 'reps_machineChestPress',
        'declinePress', 'reps_declinePress'
    ]

    # Convertir datos a DataFrame y reemplazar nulos por 0
    df = pd.DataFrame(exercises_data)
    print("DataFrame inicial:")
    print(df)

    df[ejercicios] = df[ejercicios].fillna(0)

    # Calcular el promedio de los ejercicios por cada registro
    df['average_weight'] = df[ejercicios].mean(axis=1)
    print("DataFrame con promedio de ejercicios:")
    print(df)

    # Filtrar las columnas necesarias
    df = df[['fecha', 'average_weight']]
    print("DataFrame después de filtrar columnas necesarias:")
    print(df)

    # Convertir las fechas a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Eliminar duplicados basados en la fecha
    df = df.drop_duplicates(subset=['fecha'])
    print("DataFrame después de eliminar duplicados:")
    print(df)

    # Establecer la fecha como índice
    df.set_index('fecha', inplace=True)

    # Imprimir el DataFrame para depuración
    print("DataFrame antes de establecer frecuencia:")
    print(df)
    
    # Asegurar que las fechas están ordenadas
    df = df.sort_index()

    # Rellenar fechas faltantes con 0s para asegurar una frecuencia diaria
    df = df.asfreq('D', fill_value=0)

    # Imprimir el DataFrame después de rellenar fechas para depuración
    print("DataFrame después de establecer frecuencia:")
    print(df)

    # Verificar si hay suficientes datos para el modelo
    print("Número de filas en el DataFrame después de establecer la frecuencia diaria:", len(df))
    if len(df) < 2:
        return jsonify({'error': 'Not enough data points to make a prediction'}), 400

    # Crear el modelo de Holt-Winters con tendencia aditiva
    model = ExponentialSmoothing(df['average_weight'], trend='add', seasonal=None, seasonal_periods=None)
    model_fit = model.fit()

    # Hacer predicciones
    forecast_steps = 5  # Número de días a predecir
    forecast = model_fit.forecast(steps=forecast_steps)

    # Obtener las fechas de predicción
    future_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]

    # Crear un nuevo DataFrame para las predicciones
    forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicción'])

    # Preparar los datos de respuesta
    result = {
        'fechas': df.index.strftime('%Y-%m-%d').tolist() + forecast_df.index.strftime('%Y-%m-%d').tolist(),
        'pesos': df['average_weight'].tolist() + forecast_df['Predicción'].tolist()
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
