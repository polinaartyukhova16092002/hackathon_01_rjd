import pandas as pd
import heapq
from datetime import timedelta, datetime
import geopandas as gpd
from typing import Dict, List, Optional, Tuple

from numpy import sin, atan2, radians, sqrt, cos


class Station:
    """Класс для представления железнодорожной станции"""

    def __init__(self, name: str, station_id: int):
        self.name = name
        self.id = station_id


class RailwayNetwork:
    """Класс для представления железнодорожной сети"""

    def __init__(self):
        self.stations: Dict[str, Station] = {}
        self.distance_matrix: List[List[Optional[float]]] = []


    def build_network(self, gdf: gpd.GeoDataFrame, routes: gpd.GeoDataFrame):
        self.stations = {name: Station(name, idx) for idx, name in enumerate(gdf['name'].tolist())}

        def haversine(lon1, lat1, lon2, lat2):
            R = 6371  # Радиус Земли в км
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        # Создаем матрицу расстояний
        n = max([val.id for val in self.stations.values()]) + 1
        self.distance_matrix = [[None] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    coords_i = (gdf.iloc[i].geometry.x, gdf.iloc[i].geometry.y)
                    coords_j = (gdf.iloc[j].geometry.x, gdf.iloc[j].geometry.y)
                    self.distance_matrix[i][j] = haversine(*coords_i, *coords_j)
                else:
                    self.distance_matrix[i][j] = 0

    def build_network2(self, gdf: gpd.GeoDataFrame, routes: gpd.GeoDataFrame):
        """Строит сеть станций и матрицу расстояний"""
        # Создаем станции
        self.stations = {name: Station(name, idx) for idx, name in enumerate(gdf['name'].tolist())}

        # Инициализируем матрицу расстояний
        n = max([val.id for val in self.stations.values()]) + 1
        self.distance_matrix = [[None] * n for _ in range(n)]

        # Заполняем матрицу расстояний
        for _, row in routes.iterrows():
            origin = row["origin"]
            dest = row["destinatio"]
            if origin in self.stations and dest in self.stations:
                i, j = self.stations[origin].id, self.stations[dest].id
                dist = float(row["length"])
                if self.distance_matrix[i][j] is None or dist < self.distance_matrix[i][j]:
                    self.distance_matrix[i][j] = dist

    def get_distance(self, station1: str, station2: str) -> Optional[float]:
        """Возвращает расстояние между станциями"""
        if station1 not in self.stations or station2 not in self.stations:
            return None
        return self.distance_matrix[self.stations[station1].id][self.stations[station2].id]


class Locomotive:
    """Класс для представления локомотива"""

    def __init__(self, loco_id: str, loco_type: str):
        self.id = loco_id
        self.type = loco_type
        self.current_station: Optional[str] = None
        self.available_time_min: int = 0  # Время доступности в минутах
        self.total_empty_km: float = 0

    def assign_to_route(self, route: 'TrainRoute', distance: float, cost: float):
        """Назначает локомотив на маршрут"""
        self.current_station = route.to_station
        self.available_time_min = route.departure_min + route.travel_duration_min
        self.total_empty_km += distance

    def __lt__(self, other):
        # Для корректной работы heapq
        return self.available_time_min < other.available_time_min


class TrainRoute:
    """Класс для представления маршрута поезда"""

    def __init__(self,
                 train_number: str,
                 loco_type: str,
                 from_station: str,
                 to_station: str,
                 departure_min: int,
                 travel_duration_min: int):
        self.train_number = train_number
        self.loco_type = loco_type
        self.from_station = from_station
        self.to_station = to_station
        self.departure_min = departure_min
        # self.arrive_min = departure_min + travel_duration_min
        self.travel_duration_min = travel_duration_min

    def pretty_print(self):
        print("┌───────────────────────── Информация о маршруте ─────────────────────────┐")
        print(f"│ {'Номер поезда:':<20} {self.train_number:>20} │")
        print(f"│ {'Тип локомотива:':<20} {self.loco_type:>20} │")
        print(f"│ {'Станция отправления:':<20} {self.from_station:>20} │")
        print(f"│ {'Станция прибытия:':<20} {self.to_station:>20} │")
        print(f"│ {'Время отправления (мин):':<20} {self.departure_min:>20} │")
        print(f"│ {'Длительность поездки (мин):':<20} {self.travel_duration_min:>20} │")
        print("└───────────────────────────────────────────────────────────────────────┘")


class LocomotiveDispatcher:
    """Класс для управления распределением локомотивов"""

    def __init__(self, railway_network: RailwayNetwork):
        self.network = railway_network
        self.free_locomotives: List[Locomotive] = []
        self.busy_locomotives: List[Tuple[int, str, Locomotive]] = []  # Куча (время, id, локомотив)
        self.assignments: List[Dict] = []
        self.cost_per_km: Dict[str, float] = {}
        self.errors_count: int = 0

    def initialize_locomotives(self, park_df: pd.DataFrame):
        """Инициализирует парк локомотивов"""
        self.cost_per_km = park_df.set_index('Название л-ва')['Стоимость порожнего хода (р/км)'].to_dict()

        for _, row in park_df.iterrows():
            loco_type = row['Название л-ва']
            count = row['Кол-во в парке']
            for i in range(1, count + 1):
                self.free_locomotives.append(Locomotive(f"{loco_type}_{i}", loco_type))

    def process_routes(self, routes_df: pd.DataFrame):
        """Обрабатывает все маршруты"""
        # Сортируем маршруты по времени отправления
        sorted_routes = routes_df.sort_values(['departure_min', 'arrival_min'])

        for _, row in sorted_routes.iterrows():
            route = TrainRoute(
                train_number=row['train_number'],
                loco_type=row['locomotive_type'],
                from_station=row['from'],
                to_station=row['to'],
                departure_min=row['departure_min'],
                travel_duration_min=row['travel_time_min']
            )
            # print("TTT: " + str(row['travel_time_min']) + str(row['departure_min']) +" " +  str(row['arrival_min']))
            self._assign_locomotive(route)

    def _assign_locomotive(self, route: TrainRoute):
        """Назначает локомотив на конкретный маршрут"""
        # Освобождаем локомотивы, которые завершили рейсы
        self._release_completed_locomotives(route.departure_min)

        # Ищем подходящий локомотив
        best_loco, best_distance = self._find_best_locomotive(route)
        # print(best_distance) #############

        if best_loco:
            # Записываем назначение
            self._record_assignment(route, best_loco, best_distance)

            # Обновляем состояние локомотива
            best_loco.assign_to_route(route, best_distance,
                                      best_distance * self.cost_per_km[best_loco.type])

            # Перемещаем в список занятых
            self.free_locomotives.remove(best_loco)
            heapq.heappush(self.busy_locomotives,
                           (best_loco.available_time_min, best_loco.id, best_loco))
        else:
            print(f"Не найден локомотив типа {route.loco_type} для поезда {route.train_number}")
            self._record_assignment(route, None, 0)
            self.errors_count += 1

    def _release_completed_locomotives(self, current_time_min: int):
        """Освобождает локомотивы, завершившие рейсы"""
        while self.busy_locomotives and self.busy_locomotives[0][0] <= current_time_min:
            _, loco_id, loco = heapq.heappop(self.busy_locomotives)
            self.free_locomotives.append(loco)

    def _find_best_locomotive(self, route: TrainRoute) -> Tuple[Optional[Locomotive], float]:
        """Находит оптимальный локомотив для маршрута"""
        best_loco = None
        min_cost = float('inf')
        best_distance = 0

        # 1. Ищем локомотив на текущей станции
        for loco in self.free_locomotives:
            if loco.type == route.loco_type and loco.current_station == route.from_station:
                return loco, 0  # Идеальный вариант - локомотив уже на станции

        # 2. Ищем ближайший свободный локомотив
        for loco in self.free_locomotives:
            if loco.type == route.loco_type:
                distance = 0 if loco.current_station is None else \
                    self.network.get_distance(loco.current_station, route.from_station) or float('inf')

                cost = distance
                if cost < min_cost:
                    min_cost = cost * self.cost_per_km[loco.type]
                    best_loco = loco
                    best_distance = distance
        # if (min_cost == float('inf')):
        #     print(route.pretty_print())
        return best_loco, best_distance

    def _record_assignment(self, route: TrainRoute, loco: Locomotive, distance: float):
        """Записывает информацию о назначении"""
        self.assignments.append({
            'train_number': route.train_number,
            'locomotive': loco.id if loco else None,
            'from': route.from_station,
            'to': route.to_station,
            'departure_time': self._minutes_to_time(route.departure_min),
            'arrival_time': self._minutes_to_time(route.departure_min + route.travel_duration_min),
            'travel_time_min': route.travel_duration_min,
            'empty_run_km': distance,
            'empty_run_cost': distance * self.cost_per_km[loco.type] if loco else 0,
            'previous_station': loco.current_station if loco else None
        })

    @staticmethod
    def _minutes_to_time(minutes: int) -> str:
        """Конвертирует минуты в формат HH:MM"""
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def get_results(self) -> pd.DataFrame:
        """Возвращает результаты в виде DataFrame"""
        return pd.DataFrame(self.assignments)

# Функции для работы с временем
def time_to_minutes(time_obj) -> int:
    """Конвертирует время в минуты"""
    if pd.isna(time_obj):
        return 0
    if isinstance(time_obj, str):
        try:
            if ':' in time_obj:
                parts = time_obj.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    h, m, s = map(int, parts)
                    return h * 60 + m + (1 if s >= 30 else 0)
                elif len(parts) == 2:  # HH:MM
                    return int(parts[0]) * 60 + int(parts[1])
            return 0
        except:
            return 0
    if isinstance(time_obj, timedelta):
        return int(time_obj.total_seconds() / 60)
    if isinstance(time_obj, datetime):
        return time_obj.hour * 60 + time_obj.minute + (1 if time_obj.second >= 30 else 0)
        # Если время в формате datetime.time
    elif hasattr(time_obj, 'hour'):  # Для datetime.time или datetime.datetime
        return round(time_obj.hour * 60 + time_obj.minute + time_obj.second / 60)
    return 0


def calculate_objective_function(result: pd.DataFrame, park_df, railway_network: RailwayNetwork) -> float:
    """
    Вычисляет целевую функцию для заданных результатов распределения локомотивов

    """
    ERROR_VALUE = 500_000
    results = result.copy()

    cost_per_km = park_df.set_index('Название л-ва')['Стоимость порожнего хода (р/км)'].to_dict()
    cost_per_min = park_df.set_index('Название л-ва')['Стоимость простоя (р/мин)'].to_dict()

    # количество ошибок
    nan_count = results['locomotive'].isna().sum()

    # Сортируем по локомотивам и времени отправления
    results_sorted = results.sort_values(['locomotive', 'departure_time'])

    results_sorted['next_from'] = results_sorted.groupby('locomotive')['from'].shift(-1)
    results_sorted['empty_run_distance'] = results_sorted.apply(
        lambda row: railway_network.get_distance(row['to'], row['next_from'])
        if pd.notna(row['next_from']) else 0,
        axis=1
    )


    grouped = results_sorted.groupby('locomotive')
    value_ans = 0
    km_check = 0
    # Для каждой группы (каждого локомотива) вычисляем время простоя и расстояние перегонки
    for name, group in grouped:
        if name is not None:
            bad_times = group['travel_time_min'].sum()
            total_empty_distance = group['empty_run_distance'].sum()
            # print(bad_times)

            value_ans += (24 * 60 - bad_times) * cost_per_min[name.split("_")[0]] + total_empty_distance * cost_per_km[name.split("_")[0]]
            km_check += total_empty_distance

    for loc in cost_per_km.keys():
        x = park_df.loc[park_df['Название л-ва'] == loc, 'Кол-во в парке'].values[0]
        for i in range(1, x + 1):
            name = loc + "_" + str(i)
            if name not in results['locomotive'].values:
                value_ans += 24 * 60 * cost_per_min[loc]


    value_ans += nan_count * ERROR_VALUE

    return value_ans



def calculate_objective_function_print(result: pd.DataFrame, park_df, railway_network: RailwayNetwork) -> float:
    """
    Вычисляет целевую функцию для заданных результатов распределения локомотивов

    """
    ERROR_VALUE = 500_000
    results = result.copy()

    cost_per_km = park_df.set_index('Название л-ва')['Стоимость порожнего хода (р/км)'].to_dict()
    cost_per_min = park_df.set_index('Название л-ва')['Стоимость простоя (р/мин)'].to_dict()

    # количество ошибок
    nan_count = results['locomotive'].isna().sum()
    print("count: " + str(nan_count))

    # Сортируем по локомотивам и времени отправления
    results_sorted = results.sort_values(['locomotive', 'departure_time'])

    results_sorted['next_from'] = results_sorted.groupby('locomotive')['from'].shift(-1)
    results_sorted['empty_run_distance'] = results_sorted.apply(
        lambda row: railway_network.get_distance(row['to'], row['next_from'])
        if pd.notna(row['next_from']) else 0,
        axis=1
    )


    grouped = results_sorted.groupby('locomotive')
    value_ans = 0
    km_check = 0
    # Для каждой группы (каждого локомотива) вычисляем время простоя и расстояние перегонки
    for name, group in grouped:
        if name is not None:
            bad_times = group['travel_time_min'].sum()
            total_empty_distance = group['empty_run_distance'].sum()
            # print(bad_times)

            value_ans += (24 * 60 - bad_times) * cost_per_min[name.split("_")[0]] + total_empty_distance * cost_per_km[name.split("_")[0]]
            km_check += total_empty_distance

    for loc in cost_per_km.keys():
        x = park_df.loc[park_df['Название л-ва'] == loc, 'Кол-во в парке'].values[0]
        for i in range(1, x + 1):
            name = loc + "_" + str(i)
            if name not in results['locomotive'].values:
                value_ans += 24 * 60 * cost_per_min[loc]
                print("простой: " + name)


    value_ans += nan_count * ERROR_VALUE
    print(value_ans, "км перегонки: " +  str(km_check))

    return value_ans




# Загрузка данных
routes_df = pd.read_excel('routes (3).xlsx', sheet_name='Маршруты')
park_df = pd.read_excel('routes (3).xlsx', sheet_name='Парк')
routs_all = gpd.read_file("all_routes_v2.shp")
gdf = gpd.read_file("selected1.shp")

# Предобработка данных маршрутов
routes_df.columns = ['locomotive_type', 'from', 'to', 'route_section', 'travel_time',
                     'train_number', 'departure_time', 'arrival_time']
routes_df['travel_time_min'] = routes_df['travel_time'].apply(time_to_minutes)
routes_df['departure_min'] = routes_df['departure_time'].apply(time_to_minutes)
routes_df['arrival_min'] = routes_df['arrival_time'].apply(time_to_minutes)

# Создаем и инициализируем железнодорожную сеть
railway_network = RailwayNetwork()
railway_network.build_network(gdf, routs_all)

# Создаем и инициализируем диспетчера локомотивов
dispatcher = LocomotiveDispatcher(railway_network)
dispatcher.initialize_locomotives(park_df)

# Обрабатываем маршруты
dispatcher.process_routes(routes_df)

# Получаем и выводим результаты
results = dispatcher.get_results()

total_empty_km = results['empty_run_km'].sum()
total_empty_cost = results['empty_run_cost'].sum()

print()
print("ERRORS count = " + str(dispatcher.errors_count))
print(f"Суммарный порожний пробег: {total_empty_km:.2f} км")
print(f"Суммарная стоимость порожнего пробега: {total_empty_cost:.2f} руб.")
# print(results.head(10)[['train_number', 'locomotive', 'from', 'to',
#                         'departure_time', 'arrival_time', 'empty_run_km']])


# # созранени в exel
# output_file = "result_data.xlsx"  # Имя файла для сохранения
#
# result = results.copy()
# # Основное сохранение
# result.to_excel(
#     output_file,
#     index=False,          # Не включать индекс в файл
#     sheet_name='Результаты',  # Название листа
#
# )
#
# print(f"Данные успешно сохранены в файл {output_file}")





# dispatcher = LocomotiveDispatcher(railway_network)
# dispatcher.initialize_locomotives(park_df)
# # Обрабатываем маршруты
# dispatcher.process_routes(routes_df)
# results = dispatcher.get_results()


goal_value = calculate_objective_function(results, park_df, railway_network)
print("ЦЕЛЕВАЯ f: " + str(goal_value))

# это если хотим вывести матрицу расстояний
# for station in railway_network.stations.keys():
#     for station2 in railway_network.stations.keys():
#         print(station + " -> " + station2 +" " + str(railway_network.get_distance(station, station2)))

# 15072565.255440742
# 18399799