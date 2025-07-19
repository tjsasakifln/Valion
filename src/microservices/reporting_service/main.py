"""
Reporting Microservice for Valion
Dedicated service for generating evaluation reports in various formats.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import os
import sys
from pathlib import Path
import pandas as pd
import uuid

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.config.settings import Settings

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application initialization
app = FastAPI(
    title="Valion Reporting Service",
    description="Microservice for generating evaluation reports and exports",
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

# Report storage directory
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Pydantic models
class ReportGenerationRequest(BaseModel):
    """Report generation request."""
    evaluation_id: str
    report_type: str = "comprehensive"  # "comprehensive", "summary", "technical"
    format: str = "pdf"  # "pdf", "csv", "excel", "json"
    include_sections: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None

class DataExportRequest(BaseModel):
    """Data export request."""
    evaluation_id: str
    data_type: str = "results"  # "results", "audit_trail", "shap_analysis", "raw_data"
    format: str = "csv"  # "csv", "excel", "json"
    include_metadata: bool = True
    anonymize: bool = False

class ReportStatus(BaseModel):
    """Report generation status."""
    report_id: str
    status: str  # "pending", "generating", "completed", "failed"
    progress: float
    message: str
    download_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# In-memory storage for demo purposes
report_status_store = {}

# API endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Valion Reporting Service",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "reporting",
        "timestamp": datetime.now(),
        "reports_generated": len(report_status_store),
        "storage_available": True
    }

@app.post("/reports/generate")
async def generate_report(request: ReportGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        request: Report generation request
        background_tasks: FastAPI background tasks
        
    Returns:
        Report generation status
    """
    try:
        report_id = str(uuid.uuid4())
        
        # Initialize report status
        status = ReportStatus(
            report_id=report_id,
            status="pending",
            progress=0.0,
            message="Report generation queued",
            created_at=datetime.now()
        )
        
        report_status_store[report_id] = status.model_dump()
        
        # Add background task to generate report
        background_tasks.add_task(
            _generate_report_task,
            report_id,
            request
        )
        
        return {
            "report_id": report_id,
            "status": "pending",
            "message": "Report generation started",
            "estimated_time_minutes": _estimate_generation_time(request),
            "check_status_url": f"/reports/{report_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Error starting report generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start report generation: {str(e)}"
        )

@app.get("/reports/{report_id}/status")
async def get_report_status(report_id: str):
    """
    Get report generation status.
    
    Args:
        report_id: Report ID
        
    Returns:
        Report status
    """
    if report_id not in report_status_store:
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )
    
    return report_status_store[report_id]

@app.get("/reports/{report_id}/download")
async def download_report(report_id: str):
    """
    Download generated report.
    
    Args:
        report_id: Report ID
        
    Returns:
        File download response
    """
    if report_id not in report_status_store:
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )
    
    status = report_status_store[report_id]
    
    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Current status: {status['status']}"
        )
    
    if not status.get("download_url"):
        raise HTTPException(
            status_code=404,
            detail="Report file not found"
        )
    
    file_path = Path(status["download_url"])
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Report file not found on disk"
        )
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type='application/octet-stream'
    )

@app.post("/reports/export")
async def export_data(request: DataExportRequest, background_tasks: BackgroundTasks):
    """
    Export evaluation data in specified format.
    
    Args:
        request: Data export request
        background_tasks: FastAPI background tasks
        
    Returns:
        Export status
    """
    try:
        export_id = str(uuid.uuid4())
        
        # Initialize export status
        status = ReportStatus(
            report_id=export_id,
            status="pending",
            progress=0.0,
            message="Data export queued",
            created_at=datetime.now()
        )
        
        report_status_store[export_id] = status.model_dump()
        
        # Add background task to export data
        background_tasks.add_task(
            _export_data_task,
            export_id,
            request
        )
        
        return {
            "export_id": export_id,
            "status": "pending",
            "message": "Data export started",
            "check_status_url": f"/reports/{export_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Error starting data export: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start data export: {str(e)}"
        )

@app.get("/reports/templates")
async def get_report_templates():
    """
    Get available report templates.
    
    Returns:
        List of available report templates
    """
    templates = {
        "comprehensive": {
            "name": "Comprehensive Evaluation Report",
            "description": "Complete evaluation report with all analysis sections",
            "sections": [
                "executive_summary",
                "data_analysis",
                "model_performance",
                "validation_results",
                "compliance_summary",
                "recommendations",
                "technical_appendix"
            ],
            "formats": ["pdf", "excel"],
            "estimated_pages": 25
        },
        "summary": {
            "name": "Executive Summary Report",
            "description": "High-level summary for executives and stakeholders",
            "sections": [
                "executive_summary",
                "key_findings",
                "recommendations"
            ],
            "formats": ["pdf", "excel"],
            "estimated_pages": 5
        },
        "technical": {
            "name": "Technical Analysis Report",
            "description": "Detailed technical analysis for data scientists and analysts",
            "sections": [
                "data_analysis",
                "model_performance",
                "validation_results",
                "statistical_tests",
                "shap_analysis",
                "technical_appendix"
            ],
            "formats": ["pdf", "excel", "csv"],
            "estimated_pages": 15
        },
        "compliance": {
            "name": "Compliance Report",
            "description": "Regulatory compliance and audit trail documentation",
            "sections": [
                "compliance_summary",
                "audit_trail",
                "validation_results",
                "regulatory_tests",
                "documentation"
            ],
            "formats": ["pdf"],
            "estimated_pages": 10
        }
    }
    
    return {
        "templates": templates,
        "supported_formats": ["pdf", "excel", "csv", "json"],
        "custom_templates": "Available on request"
    }

@app.get("/reports/history")
async def get_reports_history(limit: int = 50, evaluation_id: Optional[str] = None):
    """
    Get reports generation history.
    
    Args:
        limit: Maximum number of reports to return
        evaluation_id: Filter by evaluation ID
        
    Returns:
        List of report generation history
    """
    reports = list(report_status_store.values())
    
    # Filter by evaluation_id if provided
    if evaluation_id:
        reports = [r for r in reports if r.get("evaluation_id") == evaluation_id]
    
    # Sort by creation time (newest first)
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Limit results
    reports = reports[:limit]
    
    return {
        "reports": reports,
        "total": len(reports),
        "timestamp": datetime.now()
    }

@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """
    Delete a generated report.
    
    Args:
        report_id: Report ID
        
    Returns:
        Deletion confirmation
    """
    if report_id not in report_status_store:
        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )
    
    try:
        status = report_status_store[report_id]
        
        # Delete file if exists
        if status.get("download_url"):
            file_path = Path(status["download_url"])
            if file_path.exists():
                file_path.unlink()
        
        # Remove from status store
        del report_status_store[report_id]
        
        return {
            "message": "Report deleted successfully",
            "report_id": report_id,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting report {report_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete report: {str(e)}"
        )

# Background tasks

async def _generate_report_task(report_id: str, request: ReportGenerationRequest):
    """Background task to generate report."""
    try:
        # Update status to generating
        _update_report_status(report_id, "generating", 10.0, "Starting report generation")
        
        # Simulate report generation process
        await _simulate_report_generation(report_id, request)
        
        # Generate actual report content
        report_data = await _create_report_content(request)
        
        # Save report to file
        file_path = await _save_report_to_file(report_id, report_data, request.format)
        
        # Update status to completed
        _update_report_status(
            report_id, 
            "completed", 
            100.0, 
            "Report generated successfully",
            download_url=str(file_path)
        )
        
    except Exception as e:
        logger.error(f"Error generating report {report_id}: {e}")
        _update_report_status(
            report_id,
            "failed",
            0.0,
            f"Report generation failed: {str(e)}"
        )

async def _export_data_task(export_id: str, request: DataExportRequest):
    """Background task to export data."""
    try:
        # Update status to generating
        _update_report_status(export_id, "generating", 10.0, "Starting data export")
        
        # Generate export data
        export_data = await _create_export_data(request)
        
        # Save export to file
        file_path = await _save_export_to_file(export_id, export_data, request.format)
        
        # Update status to completed
        _update_report_status(
            export_id,
            "completed",
            100.0,
            "Data export completed successfully",
            download_url=str(file_path)
        )
        
    except Exception as e:
        logger.error(f"Error exporting data {export_id}: {e}")
        _update_report_status(
            export_id,
            "failed",
            0.0,
            f"Data export failed: {str(e)}"
        )

# Helper functions

def _update_report_status(report_id: str, status: str, progress: float, message: str, download_url: str = None):
    """Update report status."""
    if report_id in report_status_store:
        report_status_store[report_id].update({
            "status": status,
            "progress": progress,
            "message": message
        })
        
        if download_url:
            report_status_store[report_id]["download_url"] = download_url
            
        if status == "completed":
            report_status_store[report_id]["completed_at"] = datetime.now().isoformat()

async def _simulate_report_generation(report_id: str, request: ReportGenerationRequest):
    """Simulate report generation progress."""
    import asyncio
    
    # Simulate different generation phases
    phases = [
        (20.0, "Loading evaluation data"),
        (40.0, "Analyzing model performance"),
        (60.0, "Generating visualizations"),
        (80.0, "Compiling report sections"),
        (95.0, "Finalizing report format")
    ]
    
    for progress, message in phases:
        _update_report_status(report_id, "generating", progress, message)
        await asyncio.sleep(1)  # Simulate processing time

async def _create_report_content(request: ReportGenerationRequest) -> Dict[str, Any]:
    """Create report content based on request."""
    # Mock comprehensive report data
    report_content = {
        "metadata": {
            "evaluation_id": request.evaluation_id,
            "report_type": request.report_type,
            "generated_at": datetime.now().isoformat(),
            "valuation_standard": "NBR 14653",
            "compliance_level": "Normal"
        },
        "executive_summary": {
            "property_value": 750000.0,
            "confidence_interval": [700000.0, 800000.0],
            "model_performance": {
                "r2_score": 0.854,
                "rmse": 42150.0,
                "mae": 31200.0
            },
            "key_findings": [
                "Property valuation meets NBR 14653 compliance standards",
                "Model demonstrates strong predictive performance (R² = 0.854)",
                "Location features significantly impact property value",
                "Quality score indicates excellent investment potential"
            ]
        },
        "data_analysis": {
            "dataset_stats": {
                "total_properties": 1250,
                "features_used": 18,
                "geographic_coverage": "São Paulo Metropolitan Area"
            },
            "feature_importance": [
                {"feature": "area_privativa", "importance": 0.342},
                {"feature": "localizacao_score", "importance": 0.287},
                {"feature": "idade_imovel", "importance": -0.156},
                {"feature": "vagas_garagem", "importance": 0.123}
            ]
        },
        "model_performance": {
            "algorithm": "Elastic Net Regression",
            "validation_method": "5-fold Cross Validation",
            "metrics": {
                "r2_score": 0.854,
                "rmse": 42150.0,
                "mae": 31200.0,
                "mape": 12.5
            },
            "statistical_tests": {
                "f_test": {"value": 123.45, "p_value": 0.001, "result": "PASS"},
                "durbin_watson": {"value": 1.89, "result": "PASS"},
                "shapiro_wilk": {"value": 0.045, "result": "FAIL"}
            }
        },
        "compliance_summary": {
            "standard": "NBR 14653-2",
            "grade": "Normal",
            "compliance_score": 0.8,
            "tests_passed": 4,
            "tests_total": 5,
            "documentation_complete": True
        },
        "recommendations": [
            "Property valuation is reliable for mortgage and insurance purposes",
            "Monitor market conditions for quarterly revaluations",
            "Consider geospatial factors in future similar evaluations",
            "Maintain model performance with updated market data"
        ]
    }
    
    return report_content

async def _create_export_data(request: DataExportRequest) -> Dict[str, Any]:
    """Create export data based on request."""
    # Mock export data
    if request.data_type == "results":
        return {
            "evaluation_results": [
                {
                    "property_id": f"PROP_{i:04d}",
                    "predicted_value": 500000 + (i * 10000),
                    "confidence_lower": 450000 + (i * 9000),
                    "confidence_upper": 550000 + (i * 11000),
                    "r2_score": 0.85 + (i * 0.001),
                    "location_score": 7.5 + (i * 0.1)
                }
                for i in range(1, 11)
            ]
        }
    elif request.data_type == "audit_trail":
        return {
            "audit_events": [
                {
                    "timestamp": (datetime.now()).isoformat(),
                    "event_type": "model_training",
                    "user_id": "system",
                    "details": "Elastic Net model trained successfully"
                },
                {
                    "timestamp": (datetime.now()).isoformat(),
                    "event_type": "validation",
                    "user_id": "system", 
                    "details": "NBR 14653 compliance validation passed"
                }
            ]
        }
    else:
        return {"message": f"Export type {request.data_type} not implemented yet"}

async def _save_report_to_file(report_id: str, content: Dict[str, Any], format: str) -> Path:
    """Save report content to file."""
    filename = f"report_{report_id}.{format}"
    file_path = REPORTS_DIR / filename
    
    if format == "json":
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False, default=str)
    elif format == "csv":
        # Convert to CSV format (simplified)
        df = pd.DataFrame([content["executive_summary"]])
        df.to_csv(file_path, index=False)
    elif format == "excel":
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Executive Summary sheet
            summary_df = pd.DataFrame([content["executive_summary"]])
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Model Performance sheet
            performance_df = pd.DataFrame([content["model_performance"]["metrics"]])
            performance_df.to_excel(writer, sheet_name='Model Performance', index=False)
    elif format == "pdf":
        # For PDF, save as JSON for now (would need reportlab for actual PDF)
        with open(file_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False, default=str)
        file_path = file_path.with_suffix('.json')
    
    return file_path

async def _save_export_to_file(export_id: str, content: Dict[str, Any], format: str) -> Path:
    """Save export content to file."""
    filename = f"export_{export_id}.{format}"
    file_path = REPORTS_DIR / filename
    
    if format == "json":
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False, default=str)
    elif format == "csv":
        # Extract the main data array
        data_key = list(content.keys())[0]
        df = pd.DataFrame(content[data_key])
        df.to_csv(file_path, index=False)
    elif format == "excel":
        data_key = list(content.keys())[0]
        df = pd.DataFrame(content[data_key])
        df.to_excel(file_path, index=False)
    
    return file_path

def _estimate_generation_time(request: ReportGenerationRequest) -> int:
    """Estimate report generation time in minutes."""
    base_time = 2  # Base time in minutes
    
    if request.report_type == "comprehensive":
        base_time += 3
    elif request.report_type == "technical":
        base_time += 2
    
    if request.format == "pdf":
        base_time += 1
    
    return base_time

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)