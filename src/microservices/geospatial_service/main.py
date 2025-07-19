"""
Geospatial Microservice for Valion
Dedicated service for geospatial analysis and location intelligence.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import pandas as pd
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.core.geospatial_analysis import GeospatialAnalyzer, LocationAnalysis, create_geospatial_analyzer
from src.config.settings import Settings

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application initialization
app = FastAPI(
    title="Valion Geospatial Service",
    description="Microservice for geospatial analysis and location intelligence",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings
settings = Settings()

# Pydantic models
class GeospatialAnalysisRequest(BaseModel):
    """Geospatial analysis request."""
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city_center_lat: Optional[float] = None
    city_center_lon: Optional[float] = None
    valuation_standard: str = "NBR 14653"

class DataEnrichmentRequest(BaseModel):
    """Dataset geospatial enrichment request."""
    data: List[Dict[str, Any]]  # Dataset as list of records
    address_column: str = "address"
    city_center_lat: Optional[float] = None
    city_center_lon: Optional[float] = None
    valuation_standard: str = "NBR 14653"

class ProximityAnalysisRequest(BaseModel):
    """Proximity analysis request."""
    coordinates: Tuple[float, float]  # (lat, lon)
    radius_km: float = 5.0
    valuation_standard: str = "NBR 14653"

class HeatmapRequest(BaseModel):
    """Heatmap generation request."""
    data: List[Dict[str, Any]]  # Dataset with coordinates
    valuation_standard: str = "NBR 14653"

# API endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Valion Geospatial Service",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "geospatial",
        "timestamp": datetime.now()
    }

@app.post("/analyze/location")
async def analyze_location(request: GeospatialAnalysisRequest):
    """
    Perform comprehensive geospatial analysis of a location.
    
    Args:
        request: Location analysis request
        
    Returns:
        Complete geospatial analysis with features and scores
    """
    try:
        # Configure city center
        city_center = None
        if request.city_center_lat and request.city_center_lon:
            city_center = (request.city_center_lat, request.city_center_lon)
        
        # Create geospatial analyzer
        analyzer = create_geospatial_analyzer(
            city_center=city_center, 
            valuation_standard=request.valuation_standard
        )
        
        # Perform analysis
        if request.address:
            analysis = analyzer.analyze_location(address=request.address)
        elif request.latitude and request.longitude:
            coordinates = (request.latitude, request.longitude)
            analysis = analyzer.analyze_location(coordinates=coordinates)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either address or coordinates (lat/lon) must be provided"
            )
        
        if not analysis:
            raise HTTPException(
                status_code=400, 
                detail="Geospatial analysis failed"
            )
        
        # Convert to JSON serializable format
        result = {
            "coordinates": {
                "latitude": analysis.coordinates[0],
                "longitude": analysis.coordinates[1]
            },
            "features": {
                "distance_to_center": analysis.features.distance_to_center,
                "proximity_score": analysis.features.proximity_score,
                "density_score": analysis.features.density_score,
                "transport_score": analysis.features.transport_score,
                "amenities_score": analysis.features.amenities_score,
                "location_cluster": analysis.features.location_cluster,
                "neighborhood_value_index": analysis.features.neighborhood_value_index
            },
            "quality_score": analysis.quality_score,
            "address_components": analysis.address_components,
            "nearby_pois": analysis.nearby_pois,
            "analysis_summary": {
                "location_rating": _get_location_rating(analysis.quality_score),
                "key_strengths": _get_location_strengths(analysis.features),
                "key_weaknesses": _get_location_weaknesses(analysis.features),
                "investment_potential": _calculate_investment_potential(analysis.features)
            },
            "metadata": {
                "valuation_standard": request.valuation_standard,
                "analyzer_region": analyzer.region,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in geospatial analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Geospatial analysis error: {str(e)}"
        )

@app.post("/analyze/proximity")
async def analyze_proximity(request: ProximityAnalysisRequest):
    """
    Analyze proximity to points of interest.
    
    Args:
        request: Proximity analysis request
        
    Returns:
        Proximity analysis results
    """
    try:
        analyzer = create_geospatial_analyzer(valuation_standard=request.valuation_standard)
        
        # Calculate proximity metrics
        coordinates = request.coordinates
        proximity_score = analyzer.calculate_proximity_score(coordinates)
        transport_score = analyzer.calculate_transport_score(coordinates)
        amenities_score = analyzer.calculate_amenities_score(coordinates)
        nearby_pois = analyzer.get_nearby_pois(coordinates, request.radius_km)
        
        result = {
            "coordinates": {
                "latitude": coordinates[0],
                "longitude": coordinates[1]
            },
            "proximity_metrics": {
                "proximity_score": proximity_score,
                "transport_score": transport_score,
                "amenities_score": amenities_score,
                "overall_score": (proximity_score + transport_score + amenities_score) / 3
            },
            "nearby_pois": nearby_pois,
            "search_radius_km": request.radius_km,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in proximity analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Proximity analysis error: {str(e)}"
        )

@app.post("/enrich/dataset")
async def enrich_dataset(request: DataEnrichmentRequest):
    """
    Enrich dataset with geospatial features.
    
    Args:
        request: Dataset enrichment request
        
    Returns:
        Enriched dataset with geospatial features
    """
    try:
        # Configure city center
        city_center = None
        if request.city_center_lat and request.city_center_lon:
            city_center = (request.city_center_lat, request.city_center_lon)
        
        # Create analyzer
        analyzer = create_geospatial_analyzer(
            city_center=city_center,
            valuation_standard=request.valuation_standard
        )
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Enrich with geospatial features
        enriched_df = analyzer.enrich_dataset_with_geospatial(
            df, 
            address_column=request.address_column
        )
        
        # Generate statistics
        geo_columns = ['proximity_score', 'transport_score', 'amenities_score', 'geo_quality_score']
        statistics = {
            "total_records": len(enriched_df),
            "enriched_records": len(enriched_df.dropna(subset=['latitude', 'longitude'])),
            "geocoding_success_rate": len(enriched_df.dropna(subset=['latitude', 'longitude'])) / len(enriched_df) * 100,
            "geospatial_features_added": len([col for col in geo_columns if col in enriched_df.columns]),
            "quality_distribution": _calculate_quality_distribution(enriched_df),
            "location_clusters": enriched_df['location_cluster'].value_counts().to_dict() if 'location_cluster' in enriched_df.columns else {},
            "average_scores": {
                score: enriched_df[score].mean() if score in enriched_df.columns else 0
                for score in geo_columns
            }
        }
        
        result = {
            "enriched_data": enriched_df.to_dict('records'),
            "statistics": statistics,
            "feature_descriptions": {
                "distance_to_center": "Distance to city center (km)",
                "proximity_score": "Proximity to important POIs score (0-10)",
                "density_score": "Property density in the area score (0-10)",
                "transport_score": "Public transport access score (0-10)",
                "amenities_score": "Amenities (health, education, recreation) score (0-10)",
                "location_cluster": "Location classification (Premium Central, Urban, etc.)",
                "neighborhood_value_index": "Neighborhood value index (0-10)",
                "geo_quality_score": "Overall location quality score (0-10)"
            },
            "recommendations": _generate_enrichment_recommendations(statistics),
            "metadata": {
                "valuation_standard": request.valuation_standard,
                "enrichment_timestamp": datetime.now().isoformat()
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in dataset enrichment: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Dataset enrichment error: {str(e)}"
        )

@app.post("/generate/heatmap")
async def generate_heatmap(request: HeatmapRequest):
    """
    Generate heatmap data for location visualization.
    
    Args:
        request: Heatmap generation request
        
    Returns:
        Heatmap data for visualization
    """
    try:
        analyzer = create_geospatial_analyzer(valuation_standard=request.valuation_standard)
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Generate heatmap data
        heatmap_data = analyzer.generate_location_heatmap_data(df)
        
        if not heatmap_data:
            raise HTTPException(
                status_code=400, 
                detail="Could not generate heatmap data"
            )
        
        # Add visualization configuration
        heatmap_data.update({
            "visualization_config": {
                "map_style": "OpenStreetMap",
                "heatmap_radius": 20,
                "heatmap_blur": 15,
                "marker_size": 8,
                "color_scale": {
                    "low": "#0066CC",
                    "medium": "#FFCC00", 
                    "high": "#FF6600",
                    "premium": "#CC0000"
                }
            },
            "legend": {
                "value_ranges": {
                    "low": "< 25th percentile",
                    "medium": "25th - 50th percentile",
                    "high": "50th - 75th percentile",
                    "premium": "> 75th percentile"
                },
                "cluster_colors": {
                    "Premium Central": "#CC0000",
                    "Urbano Consolidado": "#FF6600", 
                    "Urbano em Desenvolvimento": "#FFCC00",
                    "Suburbano": "#0066CC",
                    "Periférico": "#6699FF"
                }
            },
            "metadata": {
                "valuation_standard": request.valuation_standard,
                "generation_timestamp": datetime.now().isoformat()
            }
        })
        
        return heatmap_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Heatmap generation error: {str(e)}"
        )

# Helper functions

def _get_location_rating(quality_score: float) -> str:
    """Get location rating based on quality score."""
    if quality_score >= 8:
        return "Excellent"
    elif quality_score >= 6:
        return "Good"
    elif quality_score >= 4:
        return "Regular"
    else:
        return "Limited"

def _get_location_strengths(features) -> List[str]:
    """Identify location strengths."""
    strengths = []
    
    if features.proximity_score >= 8:
        strengths.append("Excellent proximity to points of interest")
    if features.transport_score >= 8:
        strengths.append("Great public transport access")
    if features.amenities_score >= 8:
        strengths.append("Rich in amenities (health, education, recreation)")
    if features.distance_to_center <= 5:
        strengths.append("Privileged central location")
    if features.neighborhood_value_index >= 7:
        strengths.append("High neighborhood value index")
    
    return strengths if strengths else ["Location with development potential"]

def _get_location_weaknesses(features) -> List[str]:
    """Identify location weaknesses."""
    weaknesses = []
    
    if features.transport_score <= 4:
        weaknesses.append("Limited public transport access")
    if features.amenities_score <= 4:
        weaknesses.append("Few amenities in the area")
    if features.distance_to_center >= 20:
        weaknesses.append("Distant from urban center")
    if features.density_score <= 3:
        weaknesses.append("Low development density")
    
    return weaknesses if weaknesses else ["No significant limitations identified"]

def _calculate_investment_potential(features) -> str:
    """Calculate investment potential."""
    score = (
        features.proximity_score * 0.2 +
        features.transport_score * 0.2 +
        features.amenities_score * 0.15 +
        features.neighborhood_value_index * 0.25 +
        (10 - min(features.distance_to_center, 10)) * 0.2
    )
    
    if score >= 8:
        return "High - Excellent appreciation potential"
    elif score >= 6:
        return "Medium-High - Good potential with sustainable growth"
    elif score >= 4:
        return "Medium - Moderate potential with controlled risks"
    else:
        return "Low - Requires careful feasibility analysis"

def _calculate_quality_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Calculate quality score distribution."""
    if 'geo_quality_score' not in df.columns:
        return {"excellent": 0, "good": 0, "regular": 0, "limited": 0}
    
    return {
        "excellent": len(df[df['geo_quality_score'] >= 8]),
        "good": len(df[(df['geo_quality_score'] >= 6) & (df['geo_quality_score'] < 8)]),
        "regular": len(df[(df['geo_quality_score'] >= 4) & (df['geo_quality_score'] < 6)]),
        "limited": len(df[df['geo_quality_score'] < 4])
    }

def _generate_enrichment_recommendations(statistics: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on enrichment statistics."""
    recommendations = []
    
    success_rate = statistics.get("geocoding_success_rate", 0)
    if success_rate < 80:
        recommendations.append("Consider standardizing addresses to improve geocoding success rate")
    
    quality_dist = statistics.get("quality_distribution", {})
    total_records = statistics.get("total_records", 1)
    excellent_pct = quality_dist.get("excellent", 0) / total_records * 100
    
    if excellent_pct >= 50:
        recommendations.append("Portfolio with excellent location quality - focus on premium marketing")
    elif excellent_pct >= 25:
        recommendations.append("Balanced quality mix - diversify strategies by cluster")
    else:
        recommendations.append("Opportunity to improve premium location selection")
    
    avg_transport = statistics.get("average_scores", {}).get("transport_score", 0)
    if avg_transport < 5:
        recommendations.append("Prioritize properties with better public transport access")
    
    clusters = statistics.get("location_clusters", {})
    if "Premium Central" in clusters and clusters["Premium Central"] > 0:
        recommendations.append("Leverage Premium Central location properties to maximize returns")
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)