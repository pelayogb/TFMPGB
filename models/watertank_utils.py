"""
Utilidades para sistema de predicción WaterTank
Soporta predicción por provincia y por tanque individual
"""
import joblib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

MODELS_DIR = Path("models")

def get_available_models() -> Dict:
    """Retorna modelos disponibles por provincia, tanque y target"""
    available = {
        'provinces': {},
        'tanks': {}
    }

    for province_dir in MODELS_DIR.iterdir():
        if not province_dir.is_dir():
            continue

        province = province_dir.name

        # Modelos provinciales
        prov_model_dir = province_dir / "province"
        if prov_model_dir.exists():
            available['provinces'][province] = {}
            for metadata_file in prov_model_dir.glob("metadata_*.json"):
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                available['provinces'][province][meta['target']] = {
                    'model': meta['model_name'],
                    'r2': meta['test_r2'],
                    'mae': meta['test_mae']
                }

        # Modelos por tanque
        tanks_dir = province_dir / "tanks"
        if tanks_dir.exists():
            for tank_dir in tanks_dir.iterdir():
                if not tank_dir.is_dir():
                    continue

                tank_id = tank_dir.name
                tank_key = f"{province}/{tank_id}"
                available['tanks'][tank_key] = {}

                for metadata_file in tank_dir.glob("metadata_*.json"):
                    with open(metadata_file, 'r') as f:
                        meta = json.load(f)
                    available['tanks'][tank_key][meta['target']] = {
                        'model': meta['model_name'],
                        'r2': meta['test_r2'],
                        'mae': meta['test_mae']
                    }

    return available

def get_province_tanks(province: str) -> List[str]:
    """Retorna lista de tanques con modelo en una provincia"""
    tanks_dir = MODELS_DIR / province / "tanks"
    if not tanks_dir.exists():
        return []

    return [d.name for d in tanks_dir.iterdir() if d.is_dir()]

def get_tank_cluster(province: str, tank_id: str) -> Optional[int]:
    """Retorna el cluster asignado a un tanque"""
    try:
        cluster_dir = MODELS_DIR / province / "clustering"
        assignments_path = cluster_dir / "tank_cluster_assignments.csv"

        if not assignments_path.exists():
            return None

        import pandas as pd
        assignments = pd.read_csv(assignments_path)
        result = assignments[assignments['entityid'] == tank_id]

        if len(result) > 0:
            return int(result['cluster'].values[0])
        return None
    except:
        return None

def predict_watertank(
    province: str,
    target: str,
    features: Dict[str, float],
    tank_id: Optional[str] = None,
    prefer_tank_model: bool = True
) -> Dict:
    """
    Predicción inteligente con fallback automático

    Args:
        province: Provincia del tanque
        target: Magnitud a predecir
        features: Valores de features
        tank_id: ID del tanque (opcional)
        prefer_tank_model: Preferir modelo de tanque si existe

    Returns:
        Dict con predicción y metadata
        - status: 'success' o 'error'
        - prediction: Valor predicho
        - model_scope: 'tank' o 'province'
        - fallback_used: True si usó modelo provincial por fallback
    """

    def load_and_predict(model_dir, target, features):
        try:
            metadata_path = model_dir / f"metadata_{target}.json"
            if not metadata_path.exists():
                return None

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            model = joblib.load(Path(metadata['model_path']))

            scaler = None
            if metadata['use_scaling'] and metadata['scaler_path']:
                scaler = joblib.load(Path(metadata['scaler_path']))

            feature_values = [features.get(f, 0) for f in metadata['features']]
            X = np.array(feature_values).reshape(1, -1)

            if scaler:
                X = scaler.transform(X)

            prediction = model.predict(X)[0]

            return {
                'prediction': float(prediction),
                'model_scope': metadata['scope_type'],
                'province': metadata['province'],
                'tank_id': metadata.get('tank_id'),
                'target': target,
                'target_name': metadata['target_name'],
                'target_unit': metadata['target_unit'],
                'model_name': metadata['model_name'],
                'model_r2': metadata['test_r2'],
                'model_mae': metadata['test_mae']
            }
        except Exception:
            return None

    result = None

    # Obtener cluster del tanque
    tank_cluster = None
    if tank_id:
        tank_cluster = get_tank_cluster(province, tank_id)

    # 1. Modelo de tanque (si disponible y preferido)
    if tank_id and prefer_tank_model:
        tank_dir = MODELS_DIR / province / "tanks" / tank_id
        if tank_dir.exists():
            result = load_and_predict(tank_dir, target, features)
            if result:
                result['fallback_used'] = False
                return {**result, 'status': 'success'}

    # 2. Modelo provincial (fallback)
    province_dir = MODELS_DIR / province / "province"
    if province_dir.exists():
        result = load_and_predict(province_dir, target, features)
        if result:
            result['fallback_used'] = tank_id is not None
            return {**result, 'status': 'success'}

    # 3. Modelo de tanque (segunda opción)
    if tank_id and not prefer_tank_model:
        tank_dir = MODELS_DIR / province / "tanks" / tank_id
        if tank_dir.exists():
            result = load_and_predict(tank_dir, target, features)
            if result:
                result['fallback_used'] = False
                return {**result, 'status': 'success'}

    return {
        'status': 'error',
        'message': f'Modelo no encontrado: {province}/{target}' + 
                   (f'/{tank_id}' if tank_id else '')
    }

# Ejemplo de uso en MCP:
# 
# from watertank_utils import predict_watertank, get_available_models
# 
# # Listar modelos disponibles
# models = get_available_models()
# print(f"Provincias: {list(models['provinces'].keys())}")
# print(f"Tanques: {list(models['tanks'].keys())}")
# 
# # Predicción específica de tanque
# result = predict_watertank(
#     province='Madrid',
#     target='watertanklevelpercentage',
#     features={'temperature': 15.5, 'pressurelevel': 2.1},
#     tank_id='TANK_001'
# )
# print(f"Nivel predicho: {result['prediction']}%")
# print(f"Modelo usado: {result['model_scope']}")
