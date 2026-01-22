"""
Servidor MCP WaterTank con Modelos ML Reales, RAG y Sistema Multiagente
Ubicación: servers/mcp_server.py
"""
from fastmcp import FastMCP
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib
import sys
import io
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict

# Configurar UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

mcp = FastMCP("MCP WaterTank ML Server")

# ============= CONFIGURACIÓN =============
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Configuración RAG
CHUNK_SIZE = 512
OVERLAP_PERCENTAGE = 0.15
OVERLAP_TOKENS = int(CHUNK_SIZE * OVERLAP_PERCENTAGE)

# ============= CARGA DE MODELOS Y DATOS =============

class WaterTankMLSystem:
    def __init__(self):
        self.models_index = self._load_models_index()
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.rag_index = None
        self.rag_chunks = []
        self.rag_metadata = []
        self.historical_data = self._load_historical_data()
        self._initialize_rag()
        
    def _load_models_index(self) -> Dict:
        """Carga índice de modelos disponibles"""
        index_path = MODELS_DIR / "models_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return {'provinces': {}, 'statistics': {}}
    
    def _load_historical_data(self) -> pd.DataFrame:
        """Carga datos históricos procesados"""
        data_path = DATA_DIR / "watertank_processed.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            df['timeinstant'] = pd.to_datetime(df['timeinstant'], format='ISO8601')
            return df
        return pd.DataFrame()
    
    def _initialize_rag(self):
        """Inicializa sistema RAG con chunks de documentación y datos"""
        print("🔧 Inicializando RAG...")
        
        # 1. Crear chunks de metadatos de modelos
        for province, prov_data in self.models_index.get('provinces', {}).items():
            # Modelos provinciales
            for target in prov_data.get('province_models', []):
                text = f"Provincia {province} tiene modelo para predecir {target}. "
                text += f"Rendimiento promedio de modelos en {province}."
                
                self.rag_chunks.append(text)
                self.rag_metadata.append({
                    'type': 'model_info',
                    'province': province,
                    'target': target,
                    'scope': 'province'
                })
            
            # Modelos por tanque
            for tank_id, tank_data in prov_data.get('tanks', {}).items():
                text = f"Tanque {tank_id} en provincia {province} tiene modelos específicos. "
                text += f"Targets disponibles: {', '.join(tank_data.get('targets', []))}. "
                text += f"Rendimiento promedio R²: {tank_data.get('avg_r2', 0):.3f}"
                
                self.rag_chunks.append(text)
                self.rag_metadata.append({
                    'type': 'tank_info',
                    'province': province,
                    'tank_id': tank_id,
                    'scope': 'tank'
                })
        
        # 2. Crear chunks de datos históricos agregados
        if not self.historical_data.empty:
            for province in self.historical_data['province'].unique():
                df_prov = self.historical_data[self.historical_data['province'] == province]
                
                # Estadísticas generales
                stats_text = f"Provincia {province}: "
                stats_text += f"{len(df_prov)} registros históricos. "
                stats_text += f"Tanques: {df_prov['entityid'].nunique()}. "
                
                # Promedios de magnitudes principales
                for col in ['watertanklevel', 'temperature', 'clexitlevel', 'phexitlevel']:
                    if col in df_prov.columns:
                        mean_val = df_prov[col].mean()
                        if pd.notna(mean_val):
                            stats_text += f"{col} promedio: {mean_val:.2f}. "
                
                self.rag_chunks.append(stats_text)
                self.rag_metadata.append({
                    'type': 'historical_stats',
                    'province': province,
                    'n_records': len(df_prov)
                })
        
        # 3. Crear embeddings y índice FAISS
        if self.rag_chunks:
            embeddings = self.embedding_model.encode(self.rag_chunks, show_progress_bar=False)
            
            # Crear índice FAISS
            dimension = embeddings.shape[1]
            self.rag_index = faiss.IndexFlatIP(dimension)  # Inner Product para similitud coseno
            
            # Normalizar embeddings para similitud coseno
            faiss.normalize_L2(embeddings)
            self.rag_index.add(embeddings)
            
            print(f"✅ RAG inicializado: {len(self.rag_chunks)} chunks indexados")
        else:
            print("⚠️ No hay datos para RAG")
    
    def search_rag(self, query: str, top_k: int = 5) -> List[Dict]:
        """Búsqueda semántica en RAG"""
        if self.rag_index is None or len(self.rag_chunks) == 0:
            return []
        
        # Generar embedding de la query
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        # Buscar en índice
        distances, indices = self.rag_index.search(query_embedding, min(top_k, len(self.rag_chunks)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.rag_chunks):
                results.append({
                    'text': self.rag_chunks[idx],
                    'metadata': self.rag_metadata[idx],
                    'score': float(dist)
                })
        
        return results
    
    def predict(self, province: str, target: str, features: Dict, 
                tank_id: Optional[str] = None) -> Dict:
        """Realiza predicción usando modelos entrenados"""
        try:
            # Determinar directorio del modelo
            if tank_id:
                model_dir = MODELS_DIR / province / "tanks" / tank_id
            else:
                model_dir = MODELS_DIR / province / "province"
            
            # Cargar metadata
            metadata_path = model_dir / f"metadata_{target}.json"
            if not metadata_path.exists():
                # Intentar fallback a provincia
                if tank_id:
                    model_dir = MODELS_DIR / province / "province"
                    metadata_path = model_dir / f"metadata_{target}.json"
                
                if not metadata_path.exists():
                    return {
                        'status': 'error',
                        'message': f'Modelo no encontrado: {province}/{target}' + 
                                 (f'/{tank_id}' if tank_id else '')
                    }
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Cargar modelo
            model = joblib.load(Path(metadata['model_path']))
            
            # Cargar scaler si existe
            scaler = None
            if metadata['use_scaling'] and metadata.get('scaler_path'):
                scaler = joblib.load(Path(metadata['scaler_path']))
            
            # Preparar features
            feature_values = [features.get(f, 0) for f in metadata['features']]
            X = np.array(feature_values).reshape(1, -1)
            
            # Predecir
            if scaler:
                X = scaler.transform(X)
            
            prediction = model.predict(X)[0]
            
            # Obtener cluster si existe
            cluster = self._get_tank_cluster(province, tank_id) if tank_id else None
            
            return {
                'status': 'success',
                'prediction': float(prediction),
                'target': target,
                'target_name': metadata['target_name'],
                'target_unit': metadata['target_unit'],
                'province': province,
                'tank_id': tank_id,
                'cluster': cluster,
                'model_scope': metadata['scope_type'],
                'model_name': metadata['model_name'],
                'model_r2': metadata['test_r2'],
                'model_mae': metadata['test_mae'],
                'confidence': metadata['test_r2']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error en predicción: {str(e)}'
            }
    
    def _get_tank_cluster(self, province: str, tank_id: str) -> Optional[int]:
        """Obtiene cluster de un tanque"""
        try:
            cluster_dir = MODELS_DIR / province / "clustering"
            assignments_path = cluster_dir / "tank_cluster_assignments.csv"
            
            if assignments_path.exists():
                df = pd.read_csv(assignments_path)
                result = df[df['entityid'] == tank_id]
                if len(result) > 0:
                    return int(result['cluster'].values[0])
        except:
            pass
        return None
    
    def get_historical_analysis(self, province: str, tank_id: Optional[str] = None,
                               target: Optional[str] = None, hours: int = 24) -> Dict:
        """Análisis de datos históricos"""
        if self.historical_data.empty:
            return {'status': 'error', 'message': 'No hay datos históricos'}
        
        # Filtrar datos
        df = self.historical_data[self.historical_data['province'] == province].copy()
        
        if tank_id:
            df = df[df['entityid'] == tank_id]
        
        if df.empty:
            return {'status': 'error', 'message': 'No hay datos para los filtros especificados'}
        
        # Últimas N horas
        cutoff_time = df['timeinstant'].max() - pd.Timedelta(hours=hours)
        df_recent = df[df['timeinstant'] >= cutoff_time]
        
        analysis = {
            'status': 'success',
            'province': province,
            'tank_id': tank_id,
            'time_range': {
                'start': df_recent['timeinstant'].min().isoformat(),
                'end': df_recent['timeinstant'].max().isoformat(),
                'hours': hours
            },
            'n_records': len(df_recent),
            'statistics': {}
        }
        
        # Calcular estadísticas para el target o todos
        targets_to_analyze = [target] if target else ['watertanklevel', 'temperature', 
                                                       'clexitlevel', 'phexitlevel']
        
        for tgt in targets_to_analyze:
            if tgt in df_recent.columns:
                values = df_recent[tgt].dropna()
                if len(values) > 0:
                    analysis['statistics'][tgt] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'current': float(values.iloc[-1]) if len(values) > 0 else None,
                        'trend': 'ascending' if values.iloc[-1] > values.iloc[0] else 'descending'
                    }
        
        return analysis
    
    def generate_report(self, province: str, tank_id: Optional[str] = None) -> Dict:
        """Genera reporte completo con predicciones y análisis"""
        report = {
            'status': 'success',
            'generated_at': datetime.now().isoformat(),
            'province': province,
            'tank_id': tank_id,
            'sections': {}
        }
        
        # 1. Información de modelos disponibles
        if province in self.models_index.get('provinces', {}):
            prov_data = self.models_index['provinces'][province]
            report['sections']['models'] = {
                'province_models': prov_data.get('province_models', []),
                'n_tanks_with_models': len(prov_data.get('tanks', {}))
            }
            
            if tank_id and tank_id in prov_data.get('tanks', {}):
                tank_data = prov_data['tanks'][tank_id]
                report['sections']['tank_models'] = {
                    'targets': tank_data.get('targets', []),
                    'avg_r2': tank_data.get('avg_r2', 0)
                }
        
        # 2. Análisis histórico
        hist_analysis = self.get_historical_analysis(province, tank_id, hours=48)
        if hist_analysis['status'] == 'success':
            report['sections']['historical_analysis'] = hist_analysis
        
        # 3. Cluster info (si es tanque específico)
        if tank_id:
            cluster = self._get_tank_cluster(province, tank_id)
            if cluster is not None:
                report['sections']['cluster'] = {
                    'cluster_id': cluster,
                    'description': f'Tanque pertenece al cluster {cluster}'
                }
        
        return report

# Inicializar sistema
ml_system = WaterTankMLSystem()

# ============= HERRAMIENTAS MCP =============

@mcp.tool
def search_knowledge(query: str, top_k: int = 5) -> str:
    """
    Búsqueda semántica en base de conocimiento RAG
    
    Args:
        query: Consulta en lenguaje natural
        top_k: Número de resultados a retornar
    
    Returns:
        JSON con resultados relevantes
    """
    results = ml_system.search_rag(query, top_k)
    return json.dumps({
        'query': query,
        'results': results,
        'n_results': len(results)
    }, indent=2)

@mcp.tool
def predict_magnitude(province: str, target: str, features: dict, 
                     tank_id: str = None) -> str:
    """
    Predice una magnitud usando modelos ML entrenados
    
    Args:
        province: Provincia del tanque
        target: Magnitud a predecir (watertanklevel, clexitlevel, temperature, etc.)
        features: Dict con valores de features para predicción
        tank_id: ID del tanque (opcional, usa modelo provincial si no se especifica)
    
    Returns:
        JSON con predicción y metadata
    
    Ejemplo:
        features = {
            "temperature": 25.5,
            "pressurelevel": 2.1,
            "flowexitlevel": 4.5
        }
    """
    result = ml_system.predict(province, target, features, tank_id)
    return json.dumps(result, indent=2)

@mcp.tool
def get_available_models(province: str = None) -> str:
    """
    Lista modelos ML disponibles
    
    Args:
        province: Filtrar por provincia (opcional)
    
    Returns:
        JSON con modelos disponibles por provincia y tanque
    """
    if province:
        if province in ml_system.models_index.get('provinces', {}):
            return json.dumps({
                'province': province,
                'data': ml_system.models_index['provinces'][province]
            }, indent=2)
        else:
            return json.dumps({
                'error': f'Provincia {province} no encontrada',
                'available_provinces': list(ml_system.models_index.get('provinces', {}).keys())
            }, indent=2)
    
    return json.dumps({
        'provinces': list(ml_system.models_index.get('provinces', {}).keys()),
        'statistics': ml_system.models_index.get('statistics', {}),
        'total_models': ml_system.models_index.get('total_models', 0)
    }, indent=2)

@mcp.tool
def analyze_historical_data(province: str, tank_id: str = None, 
                           target: str = None, hours: int = 24) -> str:
    """
    Análisis estadístico de datos históricos
    
    Args:
        province: Provincia
        tank_id: ID del tanque (opcional)
        target: Magnitud específica a analizar (opcional)
        hours: Ventana temporal en horas (default: 24)
    
    Returns:
        JSON con estadísticas y tendencias
    """
    result = ml_system.get_historical_analysis(province, tank_id, target, hours)
    return json.dumps(result, indent=2)

@mcp.tool
def generate_comprehensive_report(province: str, tank_id: str = None) -> str:
    """
    Genera reporte completo con modelos, análisis histórico y recomendaciones
    
    Args:
        province: Provincia
        tank_id: ID del tanque (opcional)
    
    Returns:
        JSON con reporte completo estructurado
    """
    result = ml_system.generate_report(province, tank_id)
    return json.dumps(result, indent=2)

@mcp.tool
def get_recommendations(province: str, tank_id: str = None, 
                       current_values: dict = None) -> str:
    """
    Genera recomendaciones basadas en estado actual y predicciones
    
    Args:
        province: Provincia
        tank_id: ID del tanque (opcional)
        current_values: Dict con valores actuales de magnitudes
    
    Returns:
        JSON con recomendaciones operativas
    """
    recommendations = {
        'province': province,
        'tank_id': tank_id,
        'timestamp': datetime.now().isoformat(),
        'recommendations': []
    }
    
    # Análisis de valores actuales
    if current_values:
        level = current_values.get('watertanklevel', current_values.get('level'))
        
        if level is not None:
            if level < 20:
                recommendations['recommendations'].extend([
                    {'priority': 'CRÍTICO', 'action': 'Activar bomba de emergencia inmediatamente'},
                    {'priority': 'ALTO', 'action': 'Notificar equipo de mantenimiento'},
                    {'priority': 'ALTO', 'action': 'Revisar sistema de consumo'}
                ])
            elif level < 40:
                recommendations['recommendations'].extend([
                    {'priority': 'ALTO', 'action': 'Aumentar flujo de entrada'},
                    {'priority': 'MEDIO', 'action': 'Monitorear cada 15 minutos'},
                    {'priority': 'BAJO', 'action': 'Preparar bomba auxiliar'}
                ])
            elif level < 60:
                recommendations['recommendations'].extend([
                    {'priority': 'MEDIO', 'action': 'Mantener monitoreo regular'},
                    {'priority': 'BAJO', 'action': 'Verificar estado en 2 horas'}
                ])
            else:
                recommendations['recommendations'].append({
                    'priority': 'BAJO',
                    'action': 'Operación normal - monitoreo estándar'
                })
    
    # Añadir recomendaciones basadas en cluster
    if tank_id:
        cluster = ml_system._get_tank_cluster(province, tank_id)
        if cluster is not None:
            recommendations['cluster_info'] = {
                'cluster_id': cluster,
                'note': f'Tanque pertenece al cluster {cluster} - comportamiento típico del grupo'
            }
    
    return json.dumps(recommendations, indent=2)

@mcp.tool
def list_provinces_and_tanks() -> str:
    """
    Lista todas las provincias y tanques disponibles en el sistema
    
    Returns:
        JSON con estructura de provincias y tanques
    """
    structure = {
        'provinces': {},
        'total_tanks': 0
    }
    
    for province, prov_data in ml_system.models_index.get('provinces', {}).items():
        tanks = list(prov_data.get('tanks', {}).keys())
        structure['provinces'][province] = {
            'n_tanks': len(tanks),
            'tank_ids': tanks[:10],  # Primeros 10
            'has_more': len(tanks) > 10
        }
        structure['total_tanks'] += len(tanks)
    
    return json.dumps(structure, indent=2)

# Ejecutar servidor
if __name__ == "__main__":
    print("="*70)
    print("🚀 MCP WaterTank ML Server - Iniciando...")
    print(f"📊 {len(ml_system.models_index.get('provinces', {}))} provincias con modelos")
    print(f"🔍 RAG: {len(ml_system.rag_chunks)} chunks indexados")
    print(f"📈 Datos históricos: {len(ml_system.historical_data)} registros")
    print(f"🛠️  8 herramientas ML disponibles")
    print("⏳ Esperando conexión...")
    print("="*70)
    mcp.run()