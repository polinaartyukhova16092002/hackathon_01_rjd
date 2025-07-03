import pandas as pd
from datetime import datetime, timedelta

# Загрузка данных
routes_df = pd.read_excel('routes (2).xlsx', sheet_name='Маршруты', engine='openpyxl')
park_df = pd.read_excel('routes (2).xlsx', sheet_name='Парк', engine='openpyxl')

# Переименовываем столбцы как указано
routes_df.columns = ['locomotive_type', 'from', 'to', 'route_section', 'travel_time',
                     'train_number', 'departure_time', 'arrival_time']


# Функция для преобразования времени в минуты
def convert_time_to_minutes(time_val):
    if pd.isna(time_val):
        return 0

    # Если время в формате timedelta (число)
    if isinstance(time_val, (int, float)):
        return round(time_val * 60)

    # Если время в строковом формате
    if isinstance(time_val, str):
        try:
            time_str = time_val.strip()
            parts = time_str.split(':')

            if len(parts) == 3:  # HH:MM:SS
                h, m, s = map(int, parts)
                return h * 60 + m + round(s / 60)
            elif len(parts) == 2:  # HH:MM
                return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0
    if hasattr(time_val, 'hour'):  # Для datetime.time или datetime.datetime
        return round(time_val.hour * 60 + time_val.minute + time_val.second / 60)

    return 0


# Добавляем столбец с временем в минутах
routes_df['travel_time_min'] = routes_df['travel_time'].apply(convert_time_to_minutes)

# Группируем по типу локомотива и считаем агрегированные показатели
summary = routes_df.groupby('locomotive_type').agg(
    total_travel_minutes=('travel_time_min', 'sum'),
    trip_count=('travel_time_min', 'count')
).reset_index()

# Добавляем данные из парка (предварительно переименовываем столбец для соединения)
park_df = park_df.rename(columns={'Название л-ва': 'locomotive_type'})
summary = summary.merge(
    park_df,
    on='locomotive_type',
    how='left'
)

# Рассчитываем общую стоимость простоя
summary['time_bad'] = 24 * 60 * summary['trip_count'] - summary['total_travel_minutes']
summary['total_idle_cost'] = summary['time_bad'] * summary['Стоимость простоя (р/мин)']

# Формируем итоговую таблицу
result = summary[[
    'locomotive_type',
    'total_travel_minutes',
    'trip_count',
    'Кол-во в парке',
    'Стоимость порожнего хода (р/км)',
    'Стоимость простоя (р/мин)',
    'time_bad',
    'total_idle_cost'
]]

# Переименовываем столбцы для читаемости
result.columns = [
    'Локомотива',
    'Время в пути (мин)',
    'Кол-во поездок',
    'Кол-во локомотивов',
    'P порожнего хода (р/км)',
    'P простоя (р/мин)',
    'Общ время простоя',
    'Общая стоимость простоя (р)'
]

# Выводим результаты
print("Сводная статистика по типам локомотивам:")
print(result.to_string(index=False))

summary_bad_time = result['Общая стоимость простоя (р)'].sum()
print("Общая стоимость простоя всех:")
print(summary_bad_time)

