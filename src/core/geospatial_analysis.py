# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Motor de Análise Geoespacial para Valion
Responsável por processar dados geográficos e criar features de localização.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import json
import os
from pathlib import Path
from functools import lru_cache
from .cache_system import GeospatialCache, MemoryCache, cache_geospatial_feature


@dataclass
class GeospatialFeatures:
    """Features geoespaciais calculadas."""
    distance_to_center: float
    proximity_score: float
    density_score: float
    transport_score: float
    amenities_score: float
    location_cluster: str
    neighborhood_value_index: float


@dataclass
class LocationAnalysis:
    """Resultado da análise de localização."""
    features: GeospatialFeatures
    coordinates: Tuple[float, float]  # (lat, lon)
    address_components: Dict[str, str]
    nearby_pois: List[Dict[str, Any]]
    quality_score: float


class GeospatialAnalyzer:
    """Analisador geoespacial para dados imobiliários."""
    
    def __init__(self, city_center: Optional[Tuple[float, float]] = None, region: str = "Brazil", cache_enabled: bool = True):
        self.logger = logging.getLogger(__name__)
        self.region = region
        self.geocoder = Nominatim(user_agent="valion_geospatial")
        
        # Configurações regionais
        self.regional_config = self._get_regional_config()
        
        # Coordenadas do centro baseadas na região
        self.city_center = city_center or self.regional_config['default_city_center']
        
        # Sistema de cache inteligente
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = GeospatialCache(MemoryCache(max_size=500), ttl=3600)
        else:
            self.cache = None
        
        # Cache legado para compatibilidade
        self.geocode_cache = {}
        
        # POIs (Points of Interest) baseados na região
        self.pois = self._load_regional_pois()
    
    def _get_regional_config(self) -> Dict[str, Any]:
        """
        Obtém configurações regionais para análise geoespacial.
        
        Returns:
            Configurações regionais
        """
        regional_configs = {
            'Brazil': {
                'default_city_center': (-23.5505, -46.6333),  # São Paulo
                'city_name': 'São Paulo',
                'country_code': 'BR',
                'geocoder_language': 'pt',
                'density_reference': 50,  # imóveis por km²
                'poi_categories': ['shopping', 'health', 'education', 'recreation', 'transport', 'finance'],
                'transport_types': ['metro', 'bus', 'train'],
                'address_components': ['road', 'neighbourhood', 'city', 'state', 'country']
            },
            'United States': {
                'default_city_center': (40.7128, -74.0060),  # New York
                'city_name': 'New York',
                'country_code': 'US',
                'geocoder_language': 'en',
                'density_reference': 75,  # properties per km²
                'poi_categories': ['shopping', 'healthcare', 'education', 'recreation', 'transit', 'finance'],
                'transport_types': ['subway', 'bus', 'train', 'light_rail'],
                'address_components': ['house_number', 'road', 'neighbourhood', 'city', 'state', 'country']
            },
            'Europe': {
                'default_city_center': (52.5200, 13.4050),  # Berlin
                'city_name': 'Berlin',
                'country_code': 'DE',
                'geocoder_language': 'en',
                'density_reference': 60,  # properties per km²
                'poi_categories': ['shopping', 'healthcare', 'education', 'recreation', 'public_transport', 'finance'],
                'transport_types': ['metro', 'bus', 'tram', 'train'],
                'address_components': ['house_number', 'road', 'neighbourhood', 'city', 'state', 'country']
            }
        }
        
        return regional_configs.get(self.region, regional_configs['Brazil'])
    
    def _load_regional_pois(self) -> List[Dict[str, Any]]:
        """Carrega POIs baseados na região."""
        region_pois = {
            'Brazil': [
                {
                    "name": "Shopping Iguatemi",
                    "category": "shopping",
                    "coordinates": (-23.5489, -46.6388),
                    "importance": 0.8
                },
                {
                    "name": "Hospital das Clínicas",
                    "category": "health",
                    "coordinates": (-23.5505, -46.6442),
                    "importance": 0.9
                },
                {
                    "name": "USP - Universidade de São Paulo",
                    "category": "education",
                    "coordinates": (-23.5574, -46.7311),
                    "importance": 0.7
                },
                {
                    "name": "Parque Ibirapuera",
                    "category": "recreation",
                    "coordinates": (-23.5476, -46.6567),
                    "importance": 0.6
                },
                {
                    "name": "Estação Sé - Metrô",
                    "category": "transport",
                    "coordinates": (-23.5505, -46.6333),
                    "importance": 1.0
                }
            ],
            'United States': [
                {
                    "name": "Times Square",
                    "category": "shopping",
                    "coordinates": (40.7580, -73.9855),
                    "importance": 0.8
                },
                {
                    "name": "Mount Sinai Hospital",
                    "category": "healthcare",
                    "coordinates": (40.7903, -73.9523),
                    "importance": 0.9
                },
                {
                    "name": "Columbia University",
                    "category": "education",
                    "coordinates": (40.8075, -73.9626),
                    "importance": 0.7
                },
                {
                    "name": "Central Park",
                    "category": "recreation",
                    "coordinates": (40.7829, -73.9654),
                    "importance": 0.6
                },
                {
                    "name": "Grand Central Terminal",
                    "category": "transit",
                    "coordinates": (40.7527, -73.9772),
                    "importance": 1.0
                }
            ],
            'Europe': [
                {
                    "name": "Potsdamer Platz",
                    "category": "shopping",
                    "coordinates": (52.5094, 13.3760),
                    "importance": 0.8
                },
                {
                    "name": "Charité Hospital",
                    "category": "healthcare",
                    "coordinates": (52.5255, 13.3769),
                    "importance": 0.9
                },
                {
                    "name": "Humboldt University",
                    "category": "education",
                    "coordinates": (52.5178, 13.3935),
                    "importance": 0.7
                },
                {
                    "name": "Tiergarten",
                    "category": "recreation",
                    "coordinates": (52.5144, 13.3501),
                    "importance": 0.6
                },
                {
                    "name": "Hauptbahnhof",
                    "category": "public_transport",
                    "coordinates": (52.5251, 13.3694),
                    "importance": 1.0
                }
            ]
        }
        
        return region_pois.get(self.region, region_pois['Brazil'])
        
    def _load_default_pois(self) -> List[Dict[str, Any]]:
        """Carrega POIs padrão para análise."""
        return [
            {
                "name": "Shopping Center",
                "category": "shopping",
                "coordinates": (-23.5489, -46.6388),
                "importance": 0.8
            },
            {
                "name": "Hospital",
                "category": "health",
                "coordinates": (-23.5505, -46.6442),
                "importance": 0.9
            },
            {
                "name": "Universidade",
                "category": "education",
                "coordinates": (-23.5574, -46.7311),
                "importance": 0.7
            },
            {
                "name": "Parque",
                "category": "recreation",
                "coordinates": (-23.5476, -46.6567),
                "importance": 0.6
            },
            {
                "name": "Estação de Metrô",
                "category": "transport",
                "coordinates": (-23.5505, -46.6333),
                "importance": 1.0
            }
        ]
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Geocodifica um endereço para coordenadas com parâmetros regionais e cache inteligente.
        
        Args:
            address: Endereço para geocodificar
            
        Returns:
            Tupla (latitude, longitude) ou None se falhou
        """
        # Verificar cache inteligente primeiro
        if self.cache_enabled and self.cache:
            cached_coords = self.cache.get_coordinates(address)
            if cached_coords:
                self.logger.debug(f"Coordenadas obtidas do cache: {address} -> {cached_coords}")
                return cached_coords
        
        # Fallback para cache legado
        if address in self.geocode_cache:
            return self.geocode_cache[address]
        
        try:
            # Adicionar parâmetros regionais na geocodificação
            geocode_params = {
                'timeout': 10,
                'language': self.regional_config['geocoder_language'],
                'country_codes': self.regional_config['country_code']
            }
            
            # Melhorar o endereço com contexto regional se necessário
            enhanced_address = self._enhance_address_with_context(address)
            
            location = self.geocoder.geocode(enhanced_address, **geocode_params)
            if location:
                coords = (location.latitude, location.longitude)
                
                # Salvar em ambos os caches
                self.geocode_cache[address] = coords
                if self.cache_enabled and self.cache:
                    self.cache.set_coordinates(address, coords)
                
                self.logger.info(f"Geocodificado ({self.region}): {address} -> {coords}")
                return coords
            else:
                self.logger.warning(f"Geocodificação falhou para: {address}")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro na geocodificação de {address}: {e}")
            return None
    
    def _enhance_address_with_context(self, address: str) -> str:
        """
        Melhora o endereço com contexto regional.
        
        Args:
            address: Endereço original
            
        Returns:
            Endereço melhorado com contexto
        """
        # Se o endereço não inclui cidade, adicionar cidade padrão
        city_name = self.regional_config['city_name']
        if city_name.lower() not in address.lower():
            enhanced_address = f"{address}, {city_name}"
        else:
            enhanced_address = address
            
        return enhanced_address
    
    @lru_cache(maxsize=1000)
    def calculate_distance_to_center(self, coordinates: Tuple[float, float]) -> float:
        """
        Calcula distância até o centro da cidade.
        
        Args:
            coordinates: Coordenadas (lat, lon)
            
        Returns:
            Distância em quilômetros
        """
        return geodesic(coordinates, self.city_center).kilometers
    
    @cache_geospatial_feature(ttl=3600)
    def calculate_proximity_score(self, coordinates: Tuple[float, float]) -> float:
        """
        Calcula score de proximidade baseado em POIs.
        
        Args:
            coordinates: Coordenadas (lat, lon)
            
        Returns:
            Score de proximidade (0-10)
        """
        # Verificar cache inteligente primeiro
        if self.cache_enabled and self.cache:
            cached_analysis = self.cache.get_poi_analysis(coordinates, radius=5.0)
            if cached_analysis and 'proximity_score' in cached_analysis:
                return cached_analysis['proximity_score']
        
        total_score = 0.0
        weight_sum = 0.0
        
        for poi in self.pois:
            poi_coords = poi['coordinates']
            distance = geodesic(coordinates, poi_coords).kilometers
            
            # Score decai exponencialmente com a distância
            proximity = np.exp(-distance / 2.0)  # 2km como referência
            weighted_proximity = proximity * poi['importance']
            
            total_score += weighted_proximity
            weight_sum += poi['importance']
        
        # Normalizar para escala 0-10
        if weight_sum > 0:
            normalized_score = (total_score / weight_sum) * 10
            proximity_score = min(10.0, normalized_score)
        else:
            proximity_score = 0.0
        
        # Salvar no cache
        if self.cache_enabled and self.cache:
            analysis_data = {'proximity_score': proximity_score}
            self.cache.set_poi_analysis(coordinates, radius=5.0, analysis=analysis_data)
        
        return proximity_score
    
    def calculate_density_score(self, coordinates: Tuple[float, float], 
                              properties_data: pd.DataFrame) -> float:
        """
        Calcula score de densidade de imóveis na área.
        
        Args:
            coordinates: Coordenadas (lat, lon)
            properties_data: DataFrame com dados de imóveis
            
        Returns:
            Score de densidade (0-10)
        """
        if properties_data.empty or 'latitude' not in properties_data.columns:
            return 5.0  # Score médio se não há dados
        
        # Raio de 1km para análise
        radius_km = 1.0
        nearby_count = 0
        
        for _, prop in properties_data.iterrows():
            if pd.notna(prop['latitude']) and pd.notna(prop['longitude']):
                prop_coords = (prop['latitude'], prop['longitude'])
                distance = geodesic(coordinates, prop_coords).kilometers
                
                if distance <= radius_km:
                    nearby_count += 1
        
        # Normalizar baseado em densidade esperada regional
        density_reference = self.regional_config['density_reference']
        density_score = min(10.0, (nearby_count / density_reference) * 10)
        return density_score
    
    def calculate_transport_score(self, coordinates: Tuple[float, float]) -> float:
        """
        Calcula score de transporte público.
        
        Args:
            coordinates: Coordenadas (lat, lon)
            
        Returns:
            Score de transporte (0-10)
        """
        transport_pois = [poi for poi in self.pois if poi['category'] == 'transport']
        
        if not transport_pois:
            return 5.0  # Score médio se não há dados
        
        min_distance = float('inf')
        for poi in transport_pois:
            distance = geodesic(coordinates, poi['coordinates']).kilometers
            min_distance = min(min_distance, distance)
        
        # Score baseado na distância ao transporte mais próximo
        if min_distance <= 0.5:  # Até 500m
            return 10.0
        elif min_distance <= 1.0:  # Até 1km
            return 8.0
        elif min_distance <= 2.0:  # Até 2km
            return 6.0
        elif min_distance <= 5.0:  # Até 5km
            return 4.0
        else:
            return 2.0
    
    def calculate_amenities_score(self, coordinates: Tuple[float, float]) -> float:
        """
        Calcula score de amenidades (saúde, educação, lazer).
        
        Args:
            coordinates: Coordenadas (lat, lon)
            
        Returns:
            Score de amenidades (0-10)
        """
        amenity_categories = ['health', 'education', 'recreation', 'shopping']
        category_scores = {}
        
        for category in amenity_categories:
            category_pois = [poi for poi in self.pois if poi['category'] == category]
            
            if category_pois:
                distances = []
                for poi in category_pois:
                    distance = geodesic(coordinates, poi['coordinates']).kilometers
                    distances.append(distance)
                
                # Score baseado na menor distância da categoria
                min_distance = min(distances)
                if min_distance <= 1.0:
                    category_scores[category] = 10.0
                elif min_distance <= 3.0:
                    category_scores[category] = 7.0
                elif min_distance <= 5.0:
                    category_scores[category] = 5.0
                else:
                    category_scores[category] = 2.0
            else:
                category_scores[category] = 5.0
        
        # Média ponderada das categorias
        weights = {'health': 0.3, 'education': 0.2, 'recreation': 0.2, 'shopping': 0.3}
        weighted_score = sum(category_scores[cat] * weights[cat] for cat in amenity_categories)
        
        return weighted_score
    
    def classify_location(self, coordinates: Tuple[float, float],
                         proximity_score: float) -> str:
        """
        Classifica a localização em clusters.
        
        Args:
            coordinates: Coordenadas (lat, lon)
            proximity_score: Score de proximidade
            
        Returns:
            Cluster de localização
        """
        distance_to_center = self.calculate_distance_to_center(coordinates)
        
        if proximity_score >= 8.0 and distance_to_center <= 5.0:
            return "Premium Central"
        elif proximity_score >= 6.0 and distance_to_center <= 10.0:
            return "Urbano Consolidado"
        elif proximity_score >= 4.0 and distance_to_center <= 20.0:
            return "Urbano em Desenvolvimento"
        elif distance_to_center <= 30.0:
            return "Suburbano"
        else:
            return "Periférico"
    
    def calculate_neighborhood_value_index(self, coordinates: Tuple[float, float],
                                         properties_data: pd.DataFrame) -> float:
        """
        Calcula índice de valor do bairro baseado em propriedades próximas.
        
        Args:
            coordinates: Coordenadas (lat, lon)
            properties_data: DataFrame com dados de imóveis
            
        Returns:
            Índice de valor (0-10)
        """
        if properties_data.empty or 'valor' not in properties_data.columns:
            return 5.0
        
        # Propriedades num raio de 2km
        radius_km = 2.0
        nearby_values = []
        
        for _, prop in properties_data.iterrows():
            if (pd.notna(prop.get('latitude')) and 
                pd.notna(prop.get('longitude')) and 
                pd.notna(prop['valor'])):
                
                prop_coords = (prop['latitude'], prop['longitude'])
                distance = geodesic(coordinates, prop_coords).kilometers
                
                if distance <= radius_km:
                    nearby_values.append(prop['valor'])
        
        if not nearby_values:
            return 5.0
        
        # Calcular índice baseado na mediana local vs mediana geral
        local_median = np.median(nearby_values)
        global_median = properties_data['valor'].median()
        
        if global_median > 0:
            ratio = local_median / global_median
            # Normalizar para escala 0-10
            index = min(10.0, max(0.0, ratio * 5.0))
            return index
        
        return 5.0
    
    def get_nearby_pois(self, coordinates: Tuple[float, float],
                       radius_km: float = 3.0) -> List[Dict[str, Any]]:
        """
        Obtém POIs próximos às coordenadas.
        
        Args:
            coordinates: Coordenadas (lat, lon)
            radius_km: Raio de busca em km
            
        Returns:
            Lista de POIs próximos
        """
        nearby_pois = []
        
        for poi in self.pois:
            poi_coords = poi['coordinates']
            distance = geodesic(coordinates, poi_coords).kilometers
            
            if distance <= radius_km:
                poi_info = poi.copy()
                poi_info['distance_km'] = round(distance, 2)
                nearby_pois.append(poi_info)
        
        # Ordenar por distância
        nearby_pois.sort(key=lambda x: x['distance_km'])
        return nearby_pois
    
    def analyze_location(self, address: str = None, 
                        coordinates: Tuple[float, float] = None,
                        properties_data: pd.DataFrame = None) -> Optional[LocationAnalysis]:
        """
        Análise completa de localização.
        
        Args:
            address: Endereço para geocodificar
            coordinates: Coordenadas diretas (lat, lon)
            properties_data: DataFrame com dados de imóveis para contexto
            
        Returns:
            Análise completa da localização
        """
        # Obter coordenadas
        if coordinates:
            coords = coordinates
        elif address:
            coords = self.geocode_address(address)
            if not coords:
                self.logger.error(f"Falha na geocodificação de {address}")
                return None
        else:
            self.logger.error("Endereço ou coordenadas devem ser fornecidos")
            return None
        
        try:
            # Calcular features geoespaciais
            distance_to_center = self.calculate_distance_to_center(coords)
            proximity_score = self.calculate_proximity_score(coords)
            
            if properties_data is not None:
                density_score = self.calculate_density_score(coords, properties_data)
                neighborhood_value_index = self.calculate_neighborhood_value_index(coords, properties_data)
            else:
                density_score = 5.0
                neighborhood_value_index = 5.0
            
            transport_score = self.calculate_transport_score(coords)
            amenities_score = self.calculate_amenities_score(coords)
            location_cluster = self.classify_location(coords, proximity_score)
            
            # Criar features
            features = GeospatialFeatures(
                distance_to_center=distance_to_center,
                proximity_score=proximity_score,
                density_score=density_score,
                transport_score=transport_score,
                amenities_score=amenities_score,
                location_cluster=location_cluster,
                neighborhood_value_index=neighborhood_value_index
            )
            
            # POIs próximos
            nearby_pois = self.get_nearby_pois(coords)
            
            # Componentes do endereço
            address_components = {}
            if address:
                try:
                    location = self.geocoder.reverse(coords, timeout=10)
                    if location:
                        address_components = location.raw.get('address', {})
                except Exception as e:
                    self.logger.warning(f"Erro ao obter componentes do endereço: {e}")
            
            # Score de qualidade geral
            quality_score = np.mean([
                proximity_score,
                density_score,
                transport_score,
                amenities_score,
                neighborhood_value_index
            ])
            
            analysis = LocationAnalysis(
                features=features,
                coordinates=coords,
                address_components=address_components,
                nearby_pois=nearby_pois,
                quality_score=quality_score
            )
            
            self.logger.info(f"Análise geoespacial concluída para {coords}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise geoespacial: {e}")
            return None
    
    def enrich_dataset_with_geospatial(self, df: pd.DataFrame, 
                                     address_column: str = 'endereco') -> pd.DataFrame:
        """
        Enriquece dataset com features geoespaciais.
        
        Args:
            df: DataFrame original
            address_column: Nome da coluna com endereços
            
        Returns:
            DataFrame enriquecido com features geoespaciais
        """
        enriched_df = df.copy()
        
        # Inicializar colunas geoespaciais
        geo_columns = [
            'latitude', 'longitude', 'distance_to_center', 'proximity_score',
            'density_score', 'transport_score', 'amenities_score',
            'location_cluster', 'neighborhood_value_index', 'geo_quality_score'
        ]
        
        for col in geo_columns:
            if col not in enriched_df.columns:
                enriched_df[col] = np.nan
        
        processed_count = 0
        total_count = len(enriched_df)
        
        for idx, row in enriched_df.iterrows():
            # Verificar se já tem coordenadas
            if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                coords = (row['latitude'], row['longitude'])
                analysis = self.analyze_location(coordinates=coords, properties_data=enriched_df)
            elif address_column in row and pd.notna(row[address_column]):
                analysis = self.analyze_location(address=row[address_column], properties_data=enriched_df)
            else:
                continue
            
            if analysis:
                # Preencher features geoespaciais
                enriched_df.at[idx, 'latitude'] = analysis.coordinates[0]
                enriched_df.at[idx, 'longitude'] = analysis.coordinates[1]
                enriched_df.at[idx, 'distance_to_center'] = analysis.features.distance_to_center
                enriched_df.at[idx, 'proximity_score'] = analysis.features.proximity_score
                enriched_df.at[idx, 'density_score'] = analysis.features.density_score
                enriched_df.at[idx, 'transport_score'] = analysis.features.transport_score
                enriched_df.at[idx, 'amenities_score'] = analysis.features.amenities_score
                enriched_df.at[idx, 'location_cluster'] = analysis.features.location_cluster
                enriched_df.at[idx, 'neighborhood_value_index'] = analysis.features.neighborhood_value_index
                enriched_df.at[idx, 'geo_quality_score'] = analysis.quality_score
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    self.logger.info(f"Processadas {processed_count}/{total_count} localizações")
        
        self.logger.info(f"Enriquecimento geoespacial concluído: {processed_count}/{total_count} registros processados")
        return enriched_df
    
    def generate_location_heatmap_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera dados para mapa de calor de localizações.
        
        Args:
            df: DataFrame com dados geoespaciais
            
        Returns:
            Dados para visualização em mapa
        """
        if df.empty or 'latitude' not in df.columns:
            return {}
        
        # Filtrar dados válidos
        valid_data = df.dropna(subset=['latitude', 'longitude'])
        
        if valid_data.empty:
            return {}
        
        heatmap_data = {
            'locations': [],
            'center': self.city_center,
            'zoom': 10,
            'statistics': {
                'total_properties': len(valid_data),
                'avg_proximity_score': valid_data['proximity_score'].mean() if 'proximity_score' in valid_data.columns else 0,
                'location_clusters': valid_data['location_cluster'].value_counts().to_dict() if 'location_cluster' in valid_data.columns else {}
            }
        }
        
        for _, row in valid_data.iterrows():
            location_data = {
                'lat': row['latitude'],
                'lon': row['longitude'],
                'value': row.get('valor', 0),
                'proximity_score': row.get('proximity_score', 0),
                'location_cluster': row.get('location_cluster', 'Unknown')
            }
            heatmap_data['locations'].append(location_data)
        
        return heatmap_data


def create_geospatial_analyzer(city_center: Optional[Tuple[float, float]] = None, 
                             valuation_standard: str = "NBR 14653") -> GeospatialAnalyzer:
    """
    Cria um analisador geoespacial baseado na norma de avaliação.
    
    Args:
        city_center: Coordenadas do centro da cidade
        valuation_standard: Norma de avaliação
        
    Returns:
        Analisador geoespacial configurado
    """
    # Mapear norma de avaliação para região
    standard_to_region = {
        'NBR 14653': 'Brazil',
        'USPAP': 'United States',
        'EVS': 'Europe'
    }
    
    region = standard_to_region.get(valuation_standard, 'Brazil')
    
    return GeospatialAnalyzer(city_center=city_center, region=region)


