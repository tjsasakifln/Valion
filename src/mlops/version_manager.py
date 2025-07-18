# For commercial or production use, you must obtain express written consent from Tiago Sasaki via tiago@confenge.com.br.
"""
Version Manager para MLOps
Sistema de gerenciamento de versões semânticas para modelos de ML.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path

from ..monitoring.logging_config import get_logger
import structlog


class VersionType(Enum):
    """Tipos de versão."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRERELEASE = "prerelease"
    BUILD = "build"


class VersioningStrategy(Enum):
    """Estratégias de versionamento."""
    SEMANTIC = "semantic"  # Major.Minor.Patch
    TIMESTAMP = "timestamp"  # YYYYMMDD.HHMMSS
    SEQUENTIAL = "sequential"  # 1, 2, 3, ...
    HYBRID = "hybrid"  # 1.0.0-20240101.1


@dataclass
class Version:
    """Representação de uma versão."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        """Representação string da versão."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __eq__(self, other) -> bool:
        """Comparação de igualdade."""
        if not isinstance(other, Version):
            return False
        
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease and
            self.build == other.build
        )
    
    def __lt__(self, other) -> bool:
        """Comparação menor que."""
        if not isinstance(other, Version):
            return NotImplemented
        
        # Comparar versão principal
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # Comparar prerelease
        if self.prerelease is None and other.prerelease is not None:
            return False
        elif self.prerelease is not None and other.prerelease is None:
            return True
        elif self.prerelease is not None and other.prerelease is not None:
            return self.prerelease < other.prerelease
        
        # Comparar build
        if self.build is None and other.build is not None:
            return False
        elif self.build is not None and other.build is None:
            return True
        elif self.build is not None and other.build is not None:
            return self.build < other.build
        
        return False
    
    def __le__(self, other) -> bool:
        """Comparação menor ou igual."""
        return self < other or self == other
    
    def __gt__(self, other) -> bool:
        """Comparação maior que."""
        return not self <= other
    
    def __ge__(self, other) -> bool:
        """Comparação maior ou igual."""
        return not self < other
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "build": self.build,
            "version_string": str(self)
        }
    
    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Cria versão a partir de string."""
        return VersionManager.parse_version(version_str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Version':
        """Cria versão a partir de dicionário."""
        return cls(
            major=data["major"],
            minor=data["minor"],
            patch=data["patch"],
            prerelease=data.get("prerelease"),
            build=data.get("build")
        )


@dataclass
class VersionRule:
    """Regra de versionamento."""
    name: str
    description: str
    condition: str  # Python expression
    version_type: VersionType
    auto_increment: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Avalia se a regra se aplica."""
        try:
            # Avaliar condição de forma segura
            allowed_names = {
                "performance_improvement": context.get("performance_improvement", 0),
                "breaking_change": context.get("breaking_change", False),
                "new_feature": context.get("new_feature", False),
                "bug_fix": context.get("bug_fix", False),
                "data_drift": context.get("data_drift", False),
                "architecture_change": context.get("architecture_change", False),
                "experimental": context.get("experimental", False)
            }
            
            return eval(self.condition, {"__builtins__": {}}, allowed_names)
        except Exception:
            return False


class VersionManager:
    """Gerenciador de versões semânticas para modelos."""
    
    def __init__(self, strategy: VersioningStrategy = VersioningStrategy.SEMANTIC):
        self.strategy = strategy
        self.logger = get_logger("version_manager")
        self.struct_logger = structlog.get_logger("version_manager")
        
        # Padrão regex para versões semânticas
        self.semver_pattern = re.compile(
            r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
            r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
            r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))"
            r"?(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        )
        
        # Regras de versionamento padrão
        self.default_rules = [
            VersionRule(
                name="breaking_change",
                description="Incrementa versão major para mudanças quebradas",
                condition="breaking_change or architecture_change",
                version_type=VersionType.MAJOR
            ),
            VersionRule(
                name="new_feature",
                description="Incrementa versão minor para novas funcionalidades",
                condition="new_feature or performance_improvement > 0.1",
                version_type=VersionType.MINOR
            ),
            VersionRule(
                name="bug_fix",
                description="Incrementa versão patch para correções",
                condition="bug_fix or data_drift",
                version_type=VersionType.PATCH
            ),
            VersionRule(
                name="experimental",
                description="Cria versão prerelease para versões experimentais",
                condition="experimental",
                version_type=VersionType.PRERELEASE
            )
        ]
        
        self.rules = self.default_rules.copy()
    
    def parse_version(self, version_str: str) -> Version:
        """Analisa string de versão e retorna objeto Version."""
        if self.strategy == VersioningStrategy.TIMESTAMP:
            return self._parse_timestamp_version(version_str)
        elif self.strategy == VersioningStrategy.SEQUENTIAL:
            return self._parse_sequential_version(version_str)
        elif self.strategy == VersioningStrategy.HYBRID:
            return self._parse_hybrid_version(version_str)
        else:  # SEMANTIC
            return self._parse_semantic_version(version_str)
    
    def _parse_semantic_version(self, version_str: str) -> Version:
        """Analisa versão semântica."""
        match = self.semver_pattern.match(version_str)
        
        if not match:
            raise ValueError(f"Invalid semantic version: {version_str}")
        
        return Version(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            build=match.group("build")
        )
    
    def _parse_timestamp_version(self, version_str: str) -> Version:
        """Analisa versão baseada em timestamp."""
        # Formato: YYYYMMDD.HHMMSS
        parts = version_str.split('.')
        if len(parts) != 2:
            raise ValueError(f"Invalid timestamp version: {version_str}")
        
        try:
            date_part = int(parts[0])
            time_part = int(parts[1])
            
            # Converter para versão semântica
            year = date_part // 10000
            month = (date_part // 100) % 100
            day = date_part % 100
            
            hour = time_part // 10000
            minute = (time_part // 100) % 100
            second = time_part % 100
            
            return Version(
                major=year,
                minor=month * 100 + day,
                patch=hour * 10000 + minute * 100 + second
            )
        except ValueError:
            raise ValueError(f"Invalid timestamp version: {version_str}")
    
    def _parse_sequential_version(self, version_str: str) -> Version:
        """Analisa versão sequencial."""
        try:
            seq_num = int(version_str)
            return Version(major=seq_num, minor=0, patch=0)
        except ValueError:
            raise ValueError(f"Invalid sequential version: {version_str}")
    
    def _parse_hybrid_version(self, version_str: str) -> Version:
        """Analisa versão híbrida."""
        # Formato: 1.0.0-20240101.1
        if '-' in version_str:
            semver_part, timestamp_part = version_str.split('-', 1)
        else:
            semver_part = version_str
            timestamp_part = None
        
        version = self._parse_semantic_version(semver_part)
        
        if timestamp_part:
            version.prerelease = timestamp_part
        
        return version
    
    def increment_version(self, current_version: str, version_type: VersionType,
                         prerelease_identifier: str = None) -> str:
        """Incrementa versão baseada no tipo."""
        version = self.parse_version(current_version)
        
        if version_type == VersionType.MAJOR:
            version.major += 1
            version.minor = 0
            version.patch = 0
            version.prerelease = None
            version.build = None
        elif version_type == VersionType.MINOR:
            version.minor += 1
            version.patch = 0
            version.prerelease = None
            version.build = None
        elif version_type == VersionType.PATCH:
            version.patch += 1
            version.prerelease = None
            version.build = None
        elif version_type == VersionType.PRERELEASE:
            if version.prerelease is None:
                version.prerelease = prerelease_identifier or "alpha.1"
            else:
                # Incrementar prerelease
                version.prerelease = self._increment_prerelease(version.prerelease)
        elif version_type == VersionType.BUILD:
            if version.build is None:
                version.build = "1"
            else:
                # Incrementar build
                version.build = self._increment_build(version.build)
        
        return str(version)
    
    def _increment_prerelease(self, prerelease: str) -> str:
        """Incrementa identificador de prerelease."""
        # Procurar por número no final
        match = re.search(r'(\d+)$', prerelease)
        
        if match:
            num = int(match.group(1))
            return prerelease[:match.start()] + str(num + 1)
        else:
            return prerelease + ".1"
    
    def _increment_build(self, build: str) -> str:
        """Incrementa identificador de build."""
        # Procurar por número no final
        match = re.search(r'(\d+)$', build)
        
        if match:
            num = int(match.group(1))
            return build[:match.start()] + str(num + 1)
        else:
            return build + ".1"
    
    def suggest_version(self, current_version: str, 
                       context: Dict[str, Any]) -> Tuple[str, str]:
        """Sugere próxima versão baseada no contexto."""
        version_type = None
        reason = "No changes detected"
        
        # Avaliar regras em ordem de prioridade
        for rule in sorted(self.rules, key=lambda r: r.version_type.value):
            if rule.evaluate(context):
                version_type = rule.version_type
                reason = rule.description
                break
        
        if version_type is None:
            # Incremento padrão (patch)
            version_type = VersionType.PATCH
            reason = "Default patch increment"
        
        new_version = self.increment_version(current_version, version_type)
        
        self.struct_logger.info(
            "Version suggested",
            current_version=current_version,
            new_version=new_version,
            version_type=version_type.value,
            reason=reason,
            context=context
        )
        
        return new_version, reason
    
    def add_rule(self, rule: VersionRule):
        """Adiciona regra de versionamento."""
        self.rules.append(rule)
        self.struct_logger.info(
            "Version rule added",
            rule_name=rule.name,
            condition=rule.condition,
            version_type=rule.version_type.value
        )
    
    def remove_rule(self, rule_name: str):
        """Remove regra de versionamento."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        self.struct_logger.info("Version rule removed", rule_name=rule_name)
    
    def list_rules(self) -> List[VersionRule]:
        """Lista todas as regras de versionamento."""
        return self.rules.copy()
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """Compara duas versões. Retorna -1, 0, ou 1."""
        v1 = self.parse_version(version1)
        v2 = self.parse_version(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0
    
    def is_compatible(self, version1: str, version2: str, 
                     compatibility_level: str = "minor") -> bool:
        """Verifica se duas versões são compatíveis."""
        v1 = self.parse_version(version1)
        v2 = self.parse_version(version2)
        
        if compatibility_level == "major":
            return v1.major == v2.major
        elif compatibility_level == "minor":
            return v1.major == v2.major and v1.minor == v2.minor
        elif compatibility_level == "patch":
            return (v1.major == v2.major and 
                   v1.minor == v2.minor and 
                   v1.patch == v2.patch)
        else:
            return False
    
    def get_version_range(self, versions: List[str]) -> Tuple[str, str]:
        """Obtém a faixa de versões (min, max)."""
        if not versions:
            return "", ""
        
        parsed_versions = [self.parse_version(v) for v in versions]
        min_version = min(parsed_versions)
        max_version = max(parsed_versions)
        
        return str(min_version), str(max_version)
    
    def filter_versions(self, versions: List[str], 
                       min_version: str = None,
                       max_version: str = None,
                       include_prerelease: bool = False) -> List[str]:
        """Filtra versões baseado em critérios."""
        filtered = []
        
        for version_str in versions:
            try:
                version = self.parse_version(version_str)
                
                # Filtrar prerelease se necessário
                if not include_prerelease and version.prerelease:
                    continue
                
                # Filtrar por versão mínima
                if min_version and version < self.parse_version(min_version):
                    continue
                
                # Filtrar por versão máxima
                if max_version and version > self.parse_version(max_version):
                    continue
                
                filtered.append(version_str)
                
            except ValueError:
                # Versão inválida, ignorar
                continue
        
        return sorted(filtered, key=lambda v: self.parse_version(v))
    
    def generate_changelog(self, version_history: List[Dict[str, Any]]) -> str:
        """Gera changelog baseado no histórico de versões."""
        changelog = ["# Changelog", ""]
        
        for entry in sorted(version_history, key=lambda e: self.parse_version(e["version"]), reverse=True):
            version = entry["version"]
            date = entry.get("date", "")
            changes = entry.get("changes", [])
            
            changelog.append(f"## {version} - {date}")
            changelog.append("")
            
            if changes:
                for change in changes:
                    changelog.append(f"- {change}")
            else:
                changelog.append("- No changes recorded")
            
            changelog.append("")
        
        return "\n".join(changelog)
    
    def create_timestamp_version(self) -> str:
        """Cria versão baseada em timestamp atual."""
        now = datetime.now()
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        return f"{date_part}.{time_part}"
    
    def create_sequential_version(self, last_version: str = None) -> str:
        """Cria próxima versão sequencial."""
        if last_version:
            try:
                last_num = int(last_version)
                return str(last_num + 1)
            except ValueError:
                return "1"
        else:
            return "1"
    
    def validate_version(self, version: str) -> bool:
        """Valida se a versão está no formato correto."""
        try:
            self.parse_version(version)
            return True
        except ValueError:
            return False
    
    def get_version_metadata(self, version: str) -> Dict[str, Any]:
        """Obtém metadados da versão."""
        try:
            v = self.parse_version(version)
            
            return {
                "version": version,
                "strategy": self.strategy.value,
                "major": v.major,
                "minor": v.minor,
                "patch": v.patch,
                "prerelease": v.prerelease,
                "build": v.build,
                "is_prerelease": v.prerelease is not None,
                "is_stable": v.prerelease is None,
                "version_tuple": (v.major, v.minor, v.patch)
            }
        except ValueError:
            return {"version": version, "valid": False}


def create_version_manager(strategy: VersioningStrategy = VersioningStrategy.SEMANTIC) -> VersionManager:
    """Cria instância do version manager."""
    return VersionManager(strategy)