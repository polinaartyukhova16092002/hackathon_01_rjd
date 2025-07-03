import random
import numpy as np
from deap import base, creator, tools, algorithms
from copy import deepcopy
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import geopandas as gpd
import heapq

import algo2
from algo2 import time_to_minutes

class GeneticLocomotiveDispatcher:
    """Генетический алгоритм для оптимизации расписания локомотивов"""
    
    def __init__(self, railway_network, park_df, routes_df):
        self.network = railway_network
        self.park_df = park_df
        self.routes_df = routes_df
        self.cost_per_km = park_df.set_index('Название л-ва')['Стоимость порожнего хода (р/км)'].to_dict()
        self.cost_per_min = park_df.set_index('Название л-ва')['Стоимость простоя (р/мин)'].to_dict()
        
        # Подготовка данных
        self.routes = self._prepare_routes()
        self.locomotives = self._prepare_locomotives()
        self.time_windows = self._create_time_windows()
        
        # Параметры генетического алгоритма
        self.POPULATION_SIZE = 100
        self.GENERATIONS = 50
        self.CXPB = 0.7
        self.MUTPB = 0.3
        self.ERROR_PENALTY = 500_000
        
    def _prepare_routes(self) -> List[Dict]:
        """Подготавливает маршруты для обработки"""
        routes = []
        for _, row in self.routes_df.iterrows():
            route = {
                'train_number': row['train_number'],
                'loco_type': row['locomotive_type'],
                'from_station': row['from'],
                'to_station': row['to'],
                'departure_min': row['departure_min'],
                'travel_duration_min': row['travel_time_min'],
                'assigned_loco': None,
                'empty_run_distance': 0
            }
            routes.append(route)
        return routes
    
    def _prepare_locomotives(self) -> Dict[str, List[Dict]]:
        """Создает словарь доступных локомотивов по типам"""
        locomotives = defaultdict(list)
        for _, row in self.park_df.iterrows():
            loco_type = row['Название л-ва']
            count = row['Кол-во в парке']
            for i in range(1, count + 1):
                locomotives[loco_type].append({
                    'id': f"{loco_type}_{i}",
                    'type': loco_type,
                    'current_station': None,
                    'available_time': 0,
                    'total_empty_km': 0,
                    'assigned_routes': []
                })
        return locomotives
    
    def _create_time_windows(self) -> List[Tuple[int, int]]:
        """Разбивает временной горизонт на окна для временной декомпозиции"""
        min_time = min(route['departure_min'] for route in self.routes)
        max_time = max(route['departure_min'] + route['travel_duration_min'] for route in self.routes)
        
        # Разбиваем на 6-часовые окна (можно настроить)
        window_size = 6 * 60
        windows = []
        current_start = min_time
        while current_start < max_time:
            windows.append((current_start, current_start + window_size))
            current_start += window_size
        return windows
    
    def _evaluate_individual(self, individual: List[Dict]) -> Tuple[float]:
        """Оценивает качество особи (чем меньше, тем лучше)"""
        total_cost = 0
        error_count = 0
        
        # Создаем копию локомотивов для оценки
        loco_copy = deepcopy(self.locomotives)
        # loco_availability = {loco['id']: loco for type_list in loco_copy.values() for loco in type_list}
        #
        # # Сортируем назначения по времени отправления
        # assignments = sorted(individual, key=lambda x: x['departure_min'])
        #
        # for assignment in assignments:
        #     train_num = assignment['train_number']
        #     loco_id = assignment['locomotive']
        #     route = next(r for r in self.routes if r['train_number'] == train_num)
        #
        #     if loco_id is None:
        #         error_count += 1
        #         continue
        #
        #     loco = loco_availability[loco_id]
        #
        #     # Проверяем доступность локомотива
        #     if loco['type'] != route['loco_type']:
        #         error_count += 1
        #         continue
        #
        #     # Вычисляем расстояние перегонки
        #     if loco['current_station'] is None:
        #         empty_distance = 0  # Локомотив из депо
        #     else:
        #         empty_distance = self.network.get_distance(loco['current_station'], route['from_station']) or 0
        #
        #     # Проверяем, успевает ли локомотив
        #     travel_time = empty_distance / 50 * 60  # Предполагаем скорость 50 км/ч для порожнего пробега
        #     arrival_time = loco['available_time'] + travel_time
        #
        #     if arrival_time > route['departure_min']:
        #         error_count += 1
        #         continue
        #
        #     # Обновляем состояние локомотива
        #     loco['current_station'] = route['to_station']
        #     loco['available_time'] = route['departure_min'] + route['travel_duration_min']
        #     loco['total_empty_km'] += empty_distance
        #     loco['assigned_routes'].append(route['train_number'])
        #
        #     # Добавляем стоимость
        #     total_cost += empty_distance * self.cost_per_km[loco['type']]
        #
        # # Добавляем штраф за ошибки
        # total_cost += error_count * self.ERROR_PENALTY
        #
        # # Добавляем стоимость простоя неиспользованных локомотивов
        # for loco_type, loco_list in loco_copy.items():
        #     for loco in loco_list:
        #         if not loco['assigned_routes']:
        #             total_cost += 24 * 60 * self.cost_per_min[loco_type]
        #
        results = []
        for assignment in individual:
            train_num = assignment['train_number']
            route = next(r for r in self.routes if r['train_number'] == train_num)
            results.append({
                'train_number': train_num,
                'locomotive': assignment['locomotive'],
                'from': route['from_station'],
                'to': route['to_station'],
                'departure_time': self._minutes_to_time(route['departure_min']),
                'arrival_time': self._minutes_to_time(route['departure_min'] + route['travel_duration_min']),
                'travel_time_min': route['travel_duration_min'],
                'empty_run_km': 0,  # Рассчитается внутри calculate_objective_function
                'empty_run_cost': 0  # Необязательное поле
            })

        results_df = pd.DataFrame(results)

        # Вызываем целевую функцию
        total_cost = algo2.calculate_objective_function(results_df, self.park_df, self.network)
        return (total_cost,)

    
    def _create_individual(self) -> List[Dict]:
        """Создает случайную особь (решение)"""
        individual = []
        for route in self.routes:
            # Выбираем случайный локомотив подходящего типа или None
            candidates = self.locomotives.get(route['loco_type'], [])
            if candidates:
                loco = random.choice(candidates + [None])
                loco_id = loco['id'] if loco else None
            else:
                loco_id = None
                
            individual.append({
                'train_number': route['train_number'],
                'locomotive': loco_id,
                'from_station': route['from_station'],
                'to_station': route['to_station'],
                'departure_min': route['departure_min'],
                'travel_duration_min': route['travel_duration_min']
            })
        return individual

    def _create_individual_smart(self) -> List[Dict]:
        """Создает особь на основе исторических данных из result_data.xlsx"""
        # Загружаем данные из файла
        try:
            historical_data = pd.read_excel('result_data.xlsx')
            historical_assignments = historical_data.set_index('train_number')['locomotive'].to_dict()
        except:
            historical_assignments = {}

        individual = []

        for route in self.routes:
            train_num = route['train_number']

            # Пытаемся взять назначение из исторических данных
            loco_id = historical_assignments.get(train_num, None)
            if pd.isna(loco_id):
                loco_id = None
            # print(loco_id)
            # print(historical_assignments)

            # Если в исторических данных нет или локомотив не подходит
            if loco_id is None or not self._is_locomotive_valid(loco_id, route):
                # Возвращаемся к случайному выбору
                candidates = self.locomotives.get(route['loco_type'], [])
                if candidates:
                    loco = random.choice(candidates + [None])
                    loco_id = loco['id'] if loco else None

            individual.append({
                'train_number': train_num,
                'locomotive': loco_id,
                'from_station': route['from_station'],
                'to_station': route['to_station'],
                'departure_min': route['departure_min'],
                'travel_duration_min': route['travel_duration_min']
            })

        return individual

    def _is_locomotive_valid(self, loco_id: str, route: Dict) -> bool:
        """Проверяет, подходит ли локомотив для маршрута"""
        if loco_id is None:
            return False

        # Проверяем тип локомотива
        loco_type = loco_id.split('_')[0]
        if loco_type != route['loco_type']:
            return False

        # Можно добавить дополнительные проверки (доступность и т.д.)
        return True
    
    def _mutate_individual(self, individual: List[Dict]) -> Tuple[List[Dict]]:
        """Мутация особи - случайное изменение назначений"""
        for i in range(len(individual)):
            if random.random() < 0.1:  # Вероятность мутации для каждого гена
                # route = individual[i]
                # print("tttt")
                # print(route)
                # candidates = self.locomotives.get(route['loco_type'], [])
                # if candidates:
                #     loco = random.choice(candidates + [None])
                #     individual[i]['locomotive'] = loco['id'] if loco else None

                train_num = individual[i]['train_number']

                # Находим соответствующий маршрут в self.routes для получения типа локомотива
                original_route = next(r for r in self.routes if r['train_number'] == train_num)

                # Получаем доступные локомотивы этого типа
                candidates = self.locomotives.get(original_route['loco_type'], [])

                # Случайно выбираем новый локомотив или None
                if candidates:
                    loco = random.choice(candidates + [None])
                    individual[i]['locomotive'] = loco['id'] if loco else None

        ############

        return (individual,)
    
    def _cxTwoPoint(self, ind1: List[Dict], ind2: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Двухточечное скрещивание"""
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
            
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        return ind1, ind2
    
    def optimize(self) -> Dict:
        """Запускает генетический алгоритм для оптимизации"""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual_smart)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual)
        toolbox.register("mate", self._cxTwoPoint)
        toolbox.register("mutate", self._mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=self.POPULATION_SIZE)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        algorithms.eaSimple(pop, toolbox, cxpb=self.CXPB, mutpb=self.MUTPB, 
                           ngen=self.GENERATIONS, stats=stats, halloffame=hof, verbose=True)
        
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        
        # Формируем результаты
        results = []
        for assignment in best_individual:
            train_num = assignment['train_number']
            loco_id = assignment['locomotive']
            route = next(r for r in self.routes if r['train_number'] == train_num)
            
            empty_distance = 0
            if loco_id:
                route['loco_type'] = loco_id.split("_")[0] #################
                loco = next((l for type_list in self.locomotives.values() for l in type_list if l['id'] == loco_id), None)
                if loco and loco['current_station']:
                    empty_distance = self.network.get_distance(loco['current_station'], route['from_station']) or 0
            
            results.append({
                'train_number': train_num,
                'locomotive': loco_id,
                'from': route['from_station'],
                'to': route['to_station'],
                'departure_time': self._minutes_to_time(route['departure_min']),
                'arrival_time': self._minutes_to_time(route['departure_min'] + route['travel_duration_min']),
                'travel_time_min': route['travel_duration_min'],
                'empty_run_km': empty_distance,
                'empty_run_cost': empty_distance * self.cost_per_km.get(route['loco_type'], 0),
                'previous_station': None  # Можно добавить при детализации
            })
        
        return {
            'best_fitness': best_fitness,
            'best_individual': best_individual,
            'results_df': pd.DataFrame(results),
            'statistics': stats
        }
    
    @staticmethod
    def _minutes_to_time(minutes: int) -> str:
        """Конвертирует минуты в формат HH:MM"""
        return f"{minutes // 60:02d}:{minutes % 60:02d}"



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
railway_network = algo2.RailwayNetwork()
railway_network.build_network(gdf, routs_all)

# Использование генетического алгоритма
genetic_dispatcher = GeneticLocomotiveDispatcher(railway_network, park_df, routes_df)
optimization_result = genetic_dispatcher.optimize()

# Анализ результатов
best_results = optimization_result['results_df']
total_empty_km = best_results['empty_run_km'].sum()
total_empty_cost = best_results['empty_run_cost'].sum()
error_count = best_results['locomotive'].isna().sum()

print("\nОптимизированные результаты:")
print(f"Лучшее значение целевой функции: {optimization_result['best_fitness']:.2f}")
print(f"Суммарный порожний пробег: {total_empty_km:.2f} км")
print(f"Суммарная стоимость порожнего пробега: {total_empty_cost:.2f} руб.")
print(f"Количество нераспределенных поездов: {error_count}")
print(best_results.head(10)[['train_number', 'locomotive', 'from', 'to',
                            'departure_time', 'arrival_time', 'empty_run_km']])


goal_value = algo2.calculate_objective_function_print(best_results, park_df, railway_network)
print("ЦЕЛЕВАЯ f: " + str(goal_value))



# созранени в exel
output_file = "result_data_genetic.xlsx"  # Имя файла для сохранения

result = best_results.copy()
# Основное сохранение
result.to_excel(
    output_file,
    index=False,          # Не включать индекс в файл
    sheet_name='Результаты',  # Название листа

)

print(f"Данные успешно сохранены в файл {output_file}")
