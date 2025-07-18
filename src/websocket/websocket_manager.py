# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
WebSocket manager for real-time Celery feedback
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any, List
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
import time
import redis
from contextlib import suppress


@dataclass
class ProgressUpdate:
    """Structure for progress updates."""
    evaluation_id: str
    status: str
    current_phase: str
    progress: float
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class WebSocketManager:
    """Gerenciador de conexões WebSocket para updates em tempo real com heartbeat e catch-up."""
    
    def __init__(self):
        # Conexões ativas: evaluation_id -> Set[connection_id]
        self.active_connections: Dict[str, Set[str]] = {}
        # Mapa de connection_id -> WebSocket
        self.connections: Dict[str, WebSocket] = {}
        # Mapa de connection_id -> evaluation_id
        self.connection_evaluations: Dict[str, str] = {}
        # Última atividade de cada conexão para heartbeat
        self.last_activity: Dict[str, datetime] = {}
        # Buffer de mensagens para cada avaliação
        self.message_buffer: Dict[str, List[Dict]] = {}
        # Configurações
        self.heartbeat_interval = 30  # segundos
        self.connection_timeout = 90  # segundos
        self.buffer_size = 50  # últimas N mensagens
        
        self.logger = logging.getLogger(__name__)
        
        # Iniciar tarefas de background
        self._heartbeat_task = None
        self._cleanup_task = None
        
        # Conectar ao Redis para persistence do buffer
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
            self.redis_available = True
        except Exception as e:
            self.logger.warning(f"Redis não disponível para buffer WebSocket: {e}")
            self.redis_available = False
    
    async def connect(self, websocket: WebSocket, evaluation_id: str) -> str:
        """
        Conecta um cliente WebSocket para receber updates de uma avaliação com catch-up.
        
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
        self.last_activity[connection_id] = datetime.utcnow()
        
        # Adicionar à lista de conexões da avaliação
        if evaluation_id not in self.active_connections:
            self.active_connections[evaluation_id] = set()
        self.active_connections[evaluation_id].add(connection_id)
        
        # Inicializar buffer se necessário
        if evaluation_id not in self.message_buffer:
            self.message_buffer[evaluation_id] = []
        
        self.logger.info(f"WebSocket connected: {connection_id} for evaluation {evaluation_id}")
        
        # Enviar mensagem de boas-vindas
        welcome_message = {
            "type": "connection_established",
            "connection_id": connection_id,
            "evaluation_id": evaluation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "heartbeat_interval": self.heartbeat_interval
        }
        await self._send_to_connection(connection_id, welcome_message)
        
        # Enviar catch-up das mensagens em buffer
        await self._send_catchup_messages(connection_id, evaluation_id)
        
        # Iniciar heartbeat se esta é a primeira conexão
        if len(self.connections) == 1:
            await self._start_background_tasks()
        
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
                # Limpar buffer da avaliação se não há mais conexões
                if evaluation_id in self.message_buffer:
                    del self.message_buffer[evaluation_id]
        
        # Limpar mapeamentos
        del self.connections[connection_id]
        if connection_id in self.connection_evaluations:
            del self.connection_evaluations[connection_id]
        if connection_id in self.last_activity:
            del self.last_activity[connection_id]
        
        self.logger.info(f"WebSocket disconnected: {connection_id}")
        
        # Parar tarefas de background se não há mais conexões
        if not self.connections:
            self._stop_background_tasks()
    
    async def send_progress_update(self, evaluation_id: str, update: ProgressUpdate):
        """
        Envia atualização de progresso para todos os clientes conectados à avaliação e armazena no buffer.
        
        Args:
            evaluation_id: ID da avaliação
            update: Dados da atualização
        """
        message = {
            "type": "progress_update",
            **asdict(update)
        }
        
        # Adicionar ao buffer
        self._add_to_buffer(evaluation_id, message)
        
        # Se não há conexões ativas, apenas armazenar no buffer
        if evaluation_id not in self.active_connections:
            self.logger.debug(f"No active connections for evaluation {evaluation_id}, message buffered")
            return
        
        # Enviar para todas as conexões da avaliação
        connections_to_remove = []
        for connection_id in self.active_connections[evaluation_id].copy():
            try:
                await self._send_to_connection(connection_id, message)
                # Atualizar última atividade
                self.last_activity[connection_id] = datetime.utcnow()
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
        Envia mensagem para uma conexão específica com tratamento de erro.
        
        Args:
            connection_id: ID da conexão
            message: Mensagem a ser enviada
        """
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")
        
        websocket = self.connections[connection_id]
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            self.logger.error(f"Erro ao enviar mensagem para conexão {connection_id}: {e}")
            raise
    
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


    def _add_to_buffer(self, evaluation_id: str, message: Dict):
        """Adiciona mensagem ao buffer da avaliação."""
        if evaluation_id not in self.message_buffer:
            self.message_buffer[evaluation_id] = []
        
        # Adicionar timestamp se não existir
        if 'timestamp' not in message:
            message['timestamp'] = datetime.utcnow().isoformat()
        
        self.message_buffer[evaluation_id].append(message)
        
        # Manter apenas as últimas N mensagens
        if len(self.message_buffer[evaluation_id]) > self.buffer_size:
            self.message_buffer[evaluation_id] = self.message_buffer[evaluation_id][-self.buffer_size:]
        
        # Persistir no Redis se disponível
        if self.redis_available:
            try:
                buffer_key = f"ws_buffer:{evaluation_id}"
                self.redis_client.lpush(buffer_key, json.dumps(message, default=str))
                self.redis_client.ltrim(buffer_key, 0, self.buffer_size - 1)
                self.redis_client.expire(buffer_key, 3600)  # 1 hora
            except Exception as e:
                self.logger.warning(f"Erro ao persistir buffer no Redis: {e}")
    
    async def _send_catchup_messages(self, connection_id: str, evaluation_id: str):
        """Envia mensagens em buffer para conexão recém-estabelecida."""
        try:
            # Tentar carregar do Redis primeiro
            if self.redis_available:
                try:
                    buffer_key = f"ws_buffer:{evaluation_id}"
                    redis_messages = self.redis_client.lrange(buffer_key, 0, -1)
                    if redis_messages:
                        # Redis retorna em ordem reversa
                        for msg_json in reversed(redis_messages):
                            try:
                                message = json.loads(msg_json)
                                message['type'] = 'catchup_message'
                                await self._send_to_connection(connection_id, message)
                            except json.JSONDecodeError:
                                continue
                        return
                except Exception as e:
                    self.logger.warning(f"Erro ao carregar catch-up do Redis: {e}")
            
            # Fallback para buffer em memória
            if evaluation_id in self.message_buffer:
                for message in self.message_buffer[evaluation_id]:
                    # Marcar como mensagem de catch-up
                    catchup_message = message.copy()
                    catchup_message['type'] = 'catchup_message'
                    await self._send_to_connection(connection_id, catchup_message)
                
                if self.message_buffer[evaluation_id]:
                    self.logger.info(f"Enviadas {len(self.message_buffer[evaluation_id])} mensagens de catch-up para {connection_id}")
        
        except Exception as e:
            self.logger.error(f"Erro no catch-up para conexão {connection_id}: {e}")
    
    async def _start_background_tasks(self):
        """Inicia tarefas de background para heartbeat e limpeza."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    def _stop_background_tasks(self):
        """Para tarefas de background."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def _heartbeat_loop(self):
        """Loop de heartbeat para manter conexões vivas."""
        while self.connections:
            try:
                current_time = datetime.utcnow()
                
                for connection_id in list(self.connections.keys()):
                    try:
                        # Enviar ping
                        ping_message = {
                            "type": "ping",
                            "timestamp": current_time.isoformat()
                        }
                        await self._send_to_connection(connection_id, ping_message)
                        
                    except Exception as e:
                        self.logger.warning(f"Falha no heartbeat para conexão {connection_id}: {e}")
                        self.disconnect(connection_id)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erro no loop de heartbeat: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Loop de limpeza para remover conexões inativas."""
        while self.connections:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=self.connection_timeout)
                
                connections_to_remove = []
                
                for connection_id, last_activity in self.last_activity.items():
                    if last_activity < timeout_threshold:
                        connections_to_remove.append(connection_id)
                
                for connection_id in connections_to_remove:
                    self.logger.info(f"Removendo conexão inativa: {connection_id}")
                    self.disconnect(connection_id)
                
                await asyncio.sleep(30)  # Verificar a cada 30 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erro no loop de limpeza: {e}")
                await asyncio.sleep(5)
    
    async def handle_pong(self, connection_id: str):
        """Trata resposta pong do cliente."""
        if connection_id in self.last_activity:
            self.last_activity[connection_id] = datetime.utcnow()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das conexões."""
        return {
            "total_connections": len(self.connections),
            "evaluations_with_connections": len(self.active_connections),
            "buffered_evaluations": len(self.message_buffer),
            "redis_available": self.redis_available,
            "average_buffer_size": sum(len(buf) for buf in self.message_buffer.values()) / len(self.message_buffer) if self.message_buffer else 0
        }


# Instância global do gerenciador
websocket_manager = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """Retorna instância global do gerenciador de WebSocket."""
    return websocket_manager