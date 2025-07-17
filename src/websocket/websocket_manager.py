"""
Gerenciador de WebSocket para feedback em tempo real do Celery
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import uuid
from dataclasses import dataclass, asdict


@dataclass
class ProgressUpdate:
    """Estrutura para atualizações de progresso."""
    evaluation_id: str
    status: str
    current_phase: str
    progress: float
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class WebSocketManager:
    """Gerenciador de conexões WebSocket para updates em tempo real."""
    
    def __init__(self):
        # Conexões ativas: evaluation_id -> Set[connection_id]
        self.active_connections: Dict[str, Set[str]] = {}
        # Mapa de connection_id -> WebSocket
        self.connections: Dict[str, WebSocket] = {}
        # Mapa de connection_id -> evaluation_id
        self.connection_evaluations: Dict[str, str] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket, evaluation_id: str) -> str:
        """
        Conecta um cliente WebSocket para receber updates de uma avaliação.
        
        Args:
            websocket: Conexão WebSocket
            evaluation_id: ID da avaliação para monitorar
            
        Returns:
            connection_id: ID único da conexão
        """
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        self.connection_evaluations[connection_id] = evaluation_id
        
        # Adicionar à lista de conexões da avaliação
        if evaluation_id not in self.active_connections:
            self.active_connections[evaluation_id] = set()
        self.active_connections[evaluation_id].add(connection_id)
        
        self.logger.info(f"WebSocket connected: {connection_id} for evaluation {evaluation_id}")
        
        # Enviar mensagem de boas-vindas
        await self._send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "evaluation_id": evaluation_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """
        Desconecta um cliente WebSocket.
        
        Args:
            connection_id: ID da conexão
        """
        if connection_id not in self.connections:
            return
        
        evaluation_id = self.connection_evaluations.get(connection_id)
        
        # Remover da lista de conexões
        if evaluation_id and evaluation_id in self.active_connections:
            self.active_connections[evaluation_id].discard(connection_id)
            if not self.active_connections[evaluation_id]:
                del self.active_connections[evaluation_id]
        
        # Limpar mapeamentos
        del self.connections[connection_id]
        if connection_id in self.connection_evaluations:
            del self.connection_evaluations[connection_id]
        
        self.logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_progress_update(self, evaluation_id: str, update: ProgressUpdate):
        """
        Envia atualização de progresso para todos os clientes conectados à avaliação.
        
        Args:
            evaluation_id: ID da avaliação
            update: Dados da atualização
        """
        if evaluation_id not in self.active_connections:
            self.logger.debug(f"No active connections for evaluation {evaluation_id}")
            return
        
        message = {
            "type": "progress_update",
            **asdict(update)
        }
        
        # Enviar para todas as conexões da avaliação
        connections_to_remove = []
        for connection_id in self.active_connections[evaluation_id].copy():
            try:
                await self._send_to_connection(connection_id, message)
            except Exception as e:
                self.logger.warning(f"Failed to send to connection {connection_id}: {e}")
                connections_to_remove.append(connection_id)
        
        # Remover conexões com falha
        for connection_id in connections_to_remove:
            self.disconnect(connection_id)
    
    async def send_phase_update(self, evaluation_id: str, phase: str, message: str, 
                               progress: Optional[float] = None):
        """
        Envia atualização de fase.
        
        Args:
            evaluation_id: ID da avaliação
            phase: Nome da fase atual
            message: Mensagem descritiva
            progress: Progresso percentual (0-100)
        """
        update = ProgressUpdate(
            evaluation_id=evaluation_id,
            status="processing",
            current_phase=phase,
            progress=progress or 0.0,
            message=message,
            timestamp=datetime.utcnow().isoformat()
        )
        
        await self.send_progress_update(evaluation_id, update)
    
    async def send_completion_update(self, evaluation_id: str, success: bool, 
                                   final_message: str, result_data: Optional[Dict] = None):
        """
        Envia atualização de conclusão.
        
        Args:
            evaluation_id: ID da avaliação
            success: Se a avaliação foi bem-sucedida
            final_message: Mensagem final
            result_data: Dados do resultado (opcional)
        """
        update = ProgressUpdate(
            evaluation_id=evaluation_id,
            status="completed" if success else "failed",
            current_phase="Concluído",
            progress=100.0,
            message=final_message,
            timestamp=datetime.utcnow().isoformat(),
            metadata=result_data
        )
        
        await self.send_progress_update(evaluation_id, update)
    
    async def send_error_update(self, evaluation_id: str, error_message: str, 
                               error_details: Optional[Dict] = None):
        """
        Envia atualização de erro.
        
        Args:
            evaluation_id: ID da avaliação
            error_message: Mensagem de erro
            error_details: Detalhes do erro (opcional)
        """
        update = ProgressUpdate(
            evaluation_id=evaluation_id,
            status="failed",
            current_phase="Erro",
            progress=0.0,
            message=error_message,
            timestamp=datetime.utcnow().isoformat(),
            metadata=error_details
        )
        
        await self.send_progress_update(evaluation_id, update)
    
    async def _send_to_connection(self, connection_id: str, message: Dict):
        """
        Envia mensagem para uma conexão específica.
        
        Args:
            connection_id: ID da conexão
            message: Mensagem a ser enviada
        """
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        websocket = self.connections[connection_id]
        await websocket.send_text(json.dumps(message))
    
    async def broadcast_to_evaluation(self, evaluation_id: str, message: Dict):
        """
        Envia mensagem para todas as conexões de uma avaliação.
        
        Args:
            evaluation_id: ID da avaliação
            message: Mensagem a ser enviada
        """
        if evaluation_id not in self.active_connections:
            return
        
        connections_to_remove = []
        for connection_id in self.active_connections[evaluation_id].copy():
            try:
                await self._send_to_connection(connection_id, message)
            except Exception as e:
                self.logger.warning(f"Failed to broadcast to connection {connection_id}: {e}")
                connections_to_remove.append(connection_id)
        
        # Remover conexões com falha
        for connection_id in connections_to_remove:
            self.disconnect(connection_id)
    
    def get_active_connections_count(self, evaluation_id: str) -> int:
        """
        Retorna número de conexões ativas para uma avaliação.
        
        Args:
            evaluation_id: ID da avaliação
            
        Returns:
            Número de conexões ativas
        """
        return len(self.active_connections.get(evaluation_id, set()))
    
    def get_all_active_evaluations(self) -> Set[str]:
        """
        Retorna conjunto de todas as avaliações com conexões ativas.
        
        Returns:
            Set de IDs de avaliações
        """
        return set(self.active_connections.keys())


# Instância global do gerenciador
websocket_manager = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """Retorna instância global do gerenciador de WebSocket."""
    return websocket_manager