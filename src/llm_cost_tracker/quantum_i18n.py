"""
Internationalization (i18n) support for Quantum Task Planner.
Provides multi-language support and localization features.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for the quantum task planner."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"
    KOREAN = "ko"


@dataclass
class LocalizationConfig:
    """Configuration for localization."""

    default_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    auto_detect_language: bool = True
    cache_translations: bool = True
    translation_file_path: str = "locales"


class QuantumI18n:
    """
    Internationalization system for quantum task planner.
    Provides translation services and locale-aware formatting.
    """

    def __init__(self, config: LocalizationConfig = None):
        self.config = config or LocalizationConfig()
        self.current_language = self.config.default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.translation_cache: Dict[str, str] = {}

        # Load default translations
        self._load_builtin_translations()

    def _load_builtin_translations(self) -> None:
        """Load built-in translations for core quantum planner messages."""

        # Core system messages
        core_translations = {
            SupportedLanguage.ENGLISH.value: {
                # Task states
                "task.state.superposition": "Superposition",
                "task.state.collapsed": "Collapsed",
                "task.state.entangled": "Entangled",
                "task.state.executing": "Executing",
                "task.state.completed": "Completed",
                "task.state.failed": "Failed",
                # Task operations
                "task.created": "Task created successfully",
                "task.validation_failed": "Task validation failed",
                "task.execution_started": "Task execution started",
                "task.execution_completed": "Task execution completed",
                "task.execution_failed": "Task execution failed",
                "task.dependency_unsatisfied": "Task dependencies not satisfied",
                "task.resource_insufficient": "Insufficient resources for task",
                # Quantum operations
                "quantum.superposition_collapsed": "Quantum superposition collapsed",
                "quantum.entanglement_created": "Quantum entanglement created",
                "quantum.interference_applied": "Quantum interference effects applied",
                "quantum.annealing_started": "Quantum annealing optimization started",
                "quantum.annealing_completed": "Quantum annealing optimization completed",
                # System health
                "health.system_healthy": "System is healthy",
                "health.system_unhealthy": "System health issues detected",
                "health.circuit_breaker_open": "Circuit breaker is open",
                "health.circuit_breaker_closed": "Circuit breaker is closed",
                # Monitoring
                "monitor.high_cpu_usage": "High CPU usage detected",
                "monitor.high_memory_usage": "High memory usage detected",
                "monitor.high_error_rate": "High error rate detected",
                "monitor.performance_degraded": "Performance degradation detected",
                # Validation errors
                "validation.task_id_empty": "Task ID cannot be empty",
                "validation.task_name_empty": "Task name cannot be empty",
                "validation.priority_invalid": "Task priority must be between 1 and 10",
                "validation.dependency_circular": "Circular dependency detected",
                "validation.resource_negative": "Resource amounts cannot be negative",
                # General messages
                "success": "Success",
                "failure": "Failure",
                "warning": "Warning",
                "error": "Error",
                "loading": "Loading...",
                "processing": "Processing...",
                "completed": "Completed",
                "cancelled": "Cancelled",
            },
            SupportedLanguage.SPANISH.value: {
                # Task states
                "task.state.superposition": "Superposición",
                "task.state.collapsed": "Colapsado",
                "task.state.entangled": "Entrelazado",
                "task.state.executing": "Ejecutando",
                "task.state.completed": "Completado",
                "task.state.failed": "Fallido",
                # Task operations
                "task.created": "Tarea creada exitosamente",
                "task.validation_failed": "Validación de tarea falló",
                "task.execution_started": "Ejecución de tarea iniciada",
                "task.execution_completed": "Ejecución de tarea completada",
                "task.execution_failed": "Ejecución de tarea falló",
                "task.dependency_unsatisfied": "Dependencias de tarea no satisfechas",
                "task.resource_insufficient": "Recursos insuficientes para la tarea",
                # Quantum operations
                "quantum.superposition_collapsed": "Superposición cuántica colapsada",
                "quantum.entanglement_created": "Entrelazamiento cuántico creado",
                "quantum.interference_applied": "Efectos de interferencia cuántica aplicados",
                "quantum.annealing_started": "Optimización de recocido cuántico iniciada",
                "quantum.annealing_completed": "Optimización de recocido cuántico completada",
                # System health
                "health.system_healthy": "El sistema está saludable",
                "health.system_unhealthy": "Problemas de salud del sistema detectados",
                "health.circuit_breaker_open": "El disyuntor está abierto",
                "health.circuit_breaker_closed": "El disyuntor está cerrado",
                # General messages
                "success": "Éxito",
                "failure": "Fallo",
                "warning": "Advertencia",
                "error": "Error",
                "loading": "Cargando...",
                "processing": "Procesando...",
                "completed": "Completado",
                "cancelled": "Cancelado",
            },
            SupportedLanguage.FRENCH.value: {
                # Task states
                "task.state.superposition": "Superposition",
                "task.state.collapsed": "Effondré",
                "task.state.entangled": "Intriqué",
                "task.state.executing": "En cours d'exécution",
                "task.state.completed": "Terminé",
                "task.state.failed": "Échoué",
                # Task operations
                "task.created": "Tâche créée avec succès",
                "task.validation_failed": "Validation de la tâche échouée",
                "task.execution_started": "Exécution de la tâche commencée",
                "task.execution_completed": "Exécution de la tâche terminée",
                "task.execution_failed": "Exécution de la tâche échouée",
                "task.dependency_unsatisfied": "Dépendances de la tâche non satisfaites",
                "task.resource_insufficient": "Ressources insuffisantes pour la tâche",
                # Quantum operations
                "quantum.superposition_collapsed": "Superposition quantique effondrée",
                "quantum.entanglement_created": "Intrication quantique créée",
                "quantum.interference_applied": "Effets d'interférence quantique appliqués",
                "quantum.annealing_started": "Optimisation de recuit quantique démarrée",
                "quantum.annealing_completed": "Optimisation de recuit quantique terminée",
                # General messages
                "success": "Succès",
                "failure": "Échec",
                "warning": "Avertissement",
                "error": "Erreur",
                "loading": "Chargement...",
                "processing": "Traitement...",
                "completed": "Terminé",
                "cancelled": "Annulé",
            },
            SupportedLanguage.GERMAN.value: {
                # Task states
                "task.state.superposition": "Superposition",
                "task.state.collapsed": "Kollabiert",
                "task.state.entangled": "Verschränkt",
                "task.state.executing": "Ausführend",
                "task.state.completed": "Abgeschlossen",
                "task.state.failed": "Fehlgeschlagen",
                # Task operations
                "task.created": "Aufgabe erfolgreich erstellt",
                "task.validation_failed": "Aufgabenvalidierung fehlgeschlagen",
                "task.execution_started": "Aufgabenausführung gestartet",
                "task.execution_completed": "Aufgabenausführung abgeschlossen",
                "task.execution_failed": "Aufgabenausführung fehlgeschlagen",
                # General messages
                "success": "Erfolg",
                "failure": "Fehler",
                "warning": "Warnung",
                "error": "Fehler",
                "loading": "Laden...",
                "processing": "Verarbeitung...",
                "completed": "Abgeschlossen",
                "cancelled": "Abgebrochen",
            },
            SupportedLanguage.JAPANESE.value: {
                # Task states
                "task.state.superposition": "重ね合わせ",
                "task.state.collapsed": "収束済み",
                "task.state.entangled": "もつれ状態",
                "task.state.executing": "実行中",
                "task.state.completed": "完了",
                "task.state.failed": "失敗",
                # Task operations
                "task.created": "タスクが正常に作成されました",
                "task.validation_failed": "タスクの検証に失敗しました",
                "task.execution_started": "タスクの実行を開始しました",
                "task.execution_completed": "タスクの実行が完了しました",
                "task.execution_failed": "タスクの実行に失敗しました",
                # General messages
                "success": "成功",
                "failure": "失敗",
                "warning": "警告",
                "error": "エラー",
                "loading": "読み込み中...",
                "processing": "処理中...",
                "completed": "完了",
                "cancelled": "キャンセル",
            },
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                # Task states
                "task.state.superposition": "叠加态",
                "task.state.collapsed": "坍缩",
                "task.state.entangled": "纠缠",
                "task.state.executing": "执行中",
                "task.state.completed": "已完成",
                "task.state.failed": "已失败",
                # Task operations
                "task.created": "任务创建成功",
                "task.validation_failed": "任务验证失败",
                "task.execution_started": "任务执行开始",
                "task.execution_completed": "任务执行完成",
                "task.execution_failed": "任务执行失败",
                # General messages
                "success": "成功",
                "failure": "失败",
                "warning": "警告",
                "error": "错误",
                "loading": "加载中...",
                "processing": "处理中...",
                "completed": "已完成",
                "cancelled": "已取消",
            },
        }

        # Store translations
        for lang, translations in core_translations.items():
            self.translations[lang] = translations

        logger.info(f"Loaded translations for {len(core_translations)} languages")

    def set_language(self, language: SupportedLanguage) -> bool:
        """Set the current language for translations."""
        try:
            if language.value in self.translations:
                self.current_language = language
                # Clear cache when language changes
                if self.config.cache_translations:
                    self.translation_cache.clear()
                logger.info(f"Language set to {language.value}")
                return True
            else:
                logger.warning(f"Language {language.value} not available")
                return False
        except Exception as e:
            logger.error(f"Failed to set language to {language.value}: {e}")
            return False

    def translate(self, key: str, **kwargs) -> str:
        """
        Translate a key to the current language with optional parameter substitution.

        Args:
            key: Translation key (e.g., 'task.created')
            **kwargs: Parameters for string formatting

        Returns:
            Translated string or the key if translation not found
        """
        try:
            # Check cache first
            cache_key = f"{self.current_language.value}:{key}"
            if self.config.cache_translations and cache_key in self.translation_cache:
                translation = self.translation_cache[cache_key]
            else:
                # Get translation
                lang_translations = self.translations.get(
                    self.current_language.value, {}
                )
                translation = lang_translations.get(key)

                # Fallback to default language
                if (
                    translation is None
                    and self.current_language != self.config.fallback_language
                ):
                    fallback_translations = self.translations.get(
                        self.config.fallback_language.value, {}
                    )
                    translation = fallback_translations.get(key)

                # Final fallback to key itself
                if translation is None:
                    translation = key
                    logger.warning(f"Translation not found for key: {key}")

                # Cache the result
                if self.config.cache_translations:
                    self.translation_cache[cache_key] = translation

            # Apply parameter substitution
            if kwargs:
                try:
                    translation = translation.format(**kwargs)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Parameter substitution failed for key {key}: {e}")

            return translation

        except Exception as e:
            logger.error(f"Translation failed for key {key}: {e}")
            return key

    def t(self, key: str, **kwargs) -> str:
        """Shorthand alias for translate method."""
        return self.translate(key, **kwargs)

    def get_available_languages(self) -> List[SupportedLanguage]:
        """Get list of available languages."""
        available = []
        for lang in SupportedLanguage:
            if lang.value in self.translations and self.translations[lang.value]:
                available.append(lang)
        return available

    def get_current_language(self) -> SupportedLanguage:
        """Get the current language setting."""
        return self.current_language

    def add_translations(
        self, language: SupportedLanguage, translations: Dict[str, str]
    ) -> None:
        """Add or update translations for a language."""
        try:
            lang_key = language.value
            if lang_key not in self.translations:
                self.translations[lang_key] = {}

            self.translations[lang_key].update(translations)

            # Clear cache for this language
            if self.config.cache_translations:
                keys_to_remove = [
                    k
                    for k in self.translation_cache.keys()
                    if k.startswith(f"{lang_key}:")
                ]
                for key in keys_to_remove:
                    del self.translation_cache[key]

            logger.info(f"Added {len(translations)} translations for {language.value}")

        except Exception as e:
            logger.error(f"Failed to add translations for {language.value}: {e}")

    def load_translations_from_file(self, file_path: str) -> bool:
        """Load translations from a JSON file."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Translation file not found: {file_path}")
                return False

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for lang_key, translations in data.items():
                if lang_key in [lang.value for lang in SupportedLanguage]:
                    self.translations[lang_key] = translations
                else:
                    logger.warning(f"Unsupported language in file: {lang_key}")

            logger.info(f"Loaded translations from file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")
            return False

    def save_translations_to_file(self, file_path: str) -> bool:
        """Save current translations to a JSON file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.translations, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved translations to file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save translations to {file_path}: {e}")
            return False

    def detect_language_from_locale(
        self, locale_string: str
    ) -> Optional[SupportedLanguage]:
        """Detect language from locale string (e.g., 'en_US', 'es_ES')."""
        try:
            # Extract language code
            lang_code = locale_string.split("_")[0].lower()

            # Map to supported language
            for lang in SupportedLanguage:
                if lang.value == lang_code:
                    return lang

            return None

        except Exception as e:
            logger.error(f"Failed to detect language from locale {locale_string}: {e}")
            return None

    def format_number(self, number: float, decimal_places: int = 2) -> str:
        """Format number according to current locale."""
        try:
            if self.current_language == SupportedLanguage.ENGLISH:
                return f"{number:,.{decimal_places}f}"
            elif self.current_language in [
                SupportedLanguage.SPANISH,
                SupportedLanguage.FRENCH,
                SupportedLanguage.ITALIAN,
            ]:
                # European format (comma as decimal separator, space as thousands separator)
                formatted = f"{number:,.{decimal_places}f}"
                return (
                    formatted.replace(",", "TEMP")
                    .replace(".", ",")
                    .replace("TEMP", " ")
                )
            elif self.current_language == SupportedLanguage.GERMAN:
                # German format (comma as decimal separator, period as thousands separator)
                formatted = f"{number:,.{decimal_places}f}"
                return (
                    formatted.replace(",", "TEMP")
                    .replace(".", ",")
                    .replace("TEMP", ".")
                )
            else:
                # Default to English format
                return f"{number:,.{decimal_places}f}"

        except Exception as e:
            logger.error(f"Number formatting failed: {e}")
            return str(number)

    def format_percentage(self, value: float, decimal_places: int = 1) -> str:
        """Format percentage according to current locale."""
        try:
            percentage = value * 100
            formatted_number = self.format_number(percentage, decimal_places)
            return f"{formatted_number}%"

        except Exception as e:
            logger.error(f"Percentage formatting failed: {e}")
            return f"{value * 100:.{decimal_places}f}%"

    def get_translation_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded translations."""
        stats = {
            "current_language": self.current_language.value,
            "available_languages": [
                lang.value for lang in self.get_available_languages()
            ],
            "total_languages": len(self.translations),
            "cache_enabled": self.config.cache_translations,
            "cache_size": len(self.translation_cache),
            "translation_counts": {},
        }

        for lang, translations in self.translations.items():
            stats["translation_counts"][lang] = len(translations)

        return stats


# Global i18n instance
quantum_i18n = QuantumI18n()


# Convenience functions
def t(key: str, **kwargs) -> str:
    """Global translate function."""
    return quantum_i18n.translate(key, **kwargs)


def set_language(language: SupportedLanguage) -> bool:
    """Global function to set language."""
    return quantum_i18n.set_language(language)


def get_current_language() -> SupportedLanguage:
    """Global function to get current language."""
    return quantum_i18n.get_current_language()


def add_translations(language: SupportedLanguage, translations: Dict[str, str]) -> None:
    """Global function to add translations."""
    quantum_i18n.add_translations(language, translations)
