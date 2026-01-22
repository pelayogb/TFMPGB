import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import os
import gc
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE THRESHOLDS SEGÚN MODELO WATERTANK
# ============================================================================

THRESHOLDS = {
    # Cloro (ppm - mg/l)
    'clexitlevel': {'min': 0.1, 'max': 0.45, 'alert_low': 0.05, 'alert_high': 0.6},
    'clinjector': {'min': 0.01, 'max': 0.23, 'alert_low': 0.0, 'alert_high': 0.35},
    'cltanklevel': {'min': 0.02, 'max': 0.21, 'alert_low': 0.0, 'alert_high': 0.3},
    
    # pH (escala 0-14)
    'phexitlevel': {'min': 5.4, 'max': 7.2, 'alert_low': 4.5, 'alert_high': 8.5},
    
    # Turbidez (NTU)
    'waterturbidityexitlevel': {'min': 1000, 'max': 1500, 'alert_low': 800, 'alert_high': 2000},
    'turbidityentrylevel': {'min': 500, 'max': 1500, 'alert_low': 300, 'alert_high': 2000},
    
    # ORP - Potencial REDOX (mV)
    'orpexitlevel': {'min': 126, 'max': 196, 'alert_low': 100, 'alert_high': 250},
    
    # Soiling - Ensuciamiento (%)
    'soiling': {'min': 10, 'max': 80, 'alert_low': 5, 'alert_high': 95},
    
    # RSSI - Señal (dBm)
    'rssi': {'min': 60, 'max': 120, 'alert_low': 40, 'alert_high': 140},
    
    # Temperatura (°C)
    'temperature': {'min': 1, 'max': 44, 'alert_low': 0, 'alert_high': 50},
    'temperatureexitlevel': {'min': 12, 'max': 20, 'alert_low': 8, 'alert_high': 25},
    
    # nASensor (nAmperios)
    'nasensor': {'min': 10, 'max': 20, 'alert_low': 5, 'alert_high': 30},
    
    # Calibración (%)
    'calibration': {'min': 5, 'max': 80, 'alert_low': 0, 'alert_high': 100},
    
    # Flujo (m³/h)
    'flowexitlevel': {'min': 10, 'max': 20, 'alert_low': 5, 'alert_high': 30},
    
    # Presión (bar)
    'pressurelevel': {'min': 0, 'max': 5, 'alert_low': 0, 'alert_high': 6},
    'pressureexitlevel': {'min': 0, 'max': 5, 'alert_low': 0, 'alert_high': 6},
    
    # Cloro Slope (%)
    'clslope': {'min': 15, 'max': 35, 'alert_low': 10, 'alert_high': 45},
    
    # Cloro Gross (µA)
    'clgross': {'min': 100, 'max': 160, 'alert_low': 80, 'alert_high': 200},
    
    # Nivel de Tanque de Agua (cm)
    'watertanklevel': {'min': 100, 'max': 500, 'alert_low': 50, 'alert_high': 550},
}


# ============================================================================
# FUNCIÓN: RELLENAR NULOS EN UN CHUNK
# ============================================================================

def rellenar_chunk(chunk: pd.DataFrame,
                   columnas: List[str],
                   porcentaje_normal: float,
                   seed: int) -> pd.DataFrame:
    """
    Rellena valores nulos en un chunk del DataFrame.
    Trabaja directamente sobre el chunk sin hacer copias.
    
    Parámetros:
    -----------
    chunk : pd.DataFrame
        Chunk del DataFrame a procesar
    columnas : List[str]
        Lista de columnas a rellenar
    porcentaje_normal : float
        Porcentaje de valores dentro del rango normal
    seed : int
        Semilla para reproducibilidad
        
    Retorna:
    --------
    pd.DataFrame
        Chunk con valores nulos rellenados
    """
    
    np.random.seed(seed)
    
    for col in columnas:
        if col not in chunk.columns:
            continue
            
        if col not in THRESHOLDS:
            continue
        
        # Obtener máscara de nulos
        mask_nulls = chunk[col].isnull()
        n_nulls = mask_nulls.sum()
        
        if n_nulls == 0:
            continue
        
        # Obtener thresholds
        thresholds = THRESHOLDS[col]
        
        # Calcular cantidad de valores normales vs alertas
        n_normal = int(n_nulls * porcentaje_normal)
        n_alert = n_nulls - n_normal
        
        # Generar valores normales (dentro del rango)
        valores_normales = np.random.uniform(
            low=thresholds['min'],
            high=thresholds['max'],
            size=n_normal
        )
        
        # Generar valores de alerta (fuera del rango)
        n_alert_low = n_alert // 2
        n_alert_high = n_alert - n_alert_low
        
        valores_alert_low = np.random.uniform(
            low=thresholds['alert_low'],
            high=thresholds['min'],
            size=n_alert_low
        )
        
        valores_alert_high = np.random.uniform(
            low=thresholds['max'],
            high=thresholds['alert_high'],
            size=n_alert_high
        )
        
        # Combinar todos los valores y mezclarlos
        valores_generados = np.concatenate([
            valores_normales,
            valores_alert_low,
            valores_alert_high
        ])
        np.random.shuffle(valores_generados)
        
        # Redondear según el tipo de variable
        if col in ['clexitlevel', 'clinjector', 'cltanklevel']:
            valores_generados = np.round(valores_generados, 2)
        elif col in ['phexitlevel']:
            valores_generados = np.round(valores_generados, 2)
        elif col in ['waterturbidityexitlevel', 'turbidityentrylevel']:
            valores_generados = np.round(valores_generados, 1)
        elif col in ['orpexitlevel', 'rssi']:
            valores_generados = np.round(valores_generados, 0)
        elif col in ['temperature', 'temperatureexitlevel']:
            valores_generados = np.round(valores_generados, 1)
        elif col in ['soiling', 'calibration', 'clslope']:
            valores_generados = np.round(valores_generados, 0)
        elif col in ['watertanklevel']:
            valores_generados = np.round(valores_generados, 1)
        else:
            valores_generados = np.round(valores_generados, 2)
        
        # Asignar valores directamente
        chunk.loc[mask_nulls, col] = valores_generados
    
    # Aplicar coherencia entre variables
    chunk = aplicar_coherencia_ligera(chunk, columnas)
    
    return chunk


# ============================================================================
# FUNCIÓN: APLICAR COHERENCIA LIGERA (OPTIMIZADA PARA CHUNKS)
# ============================================================================

def aplicar_coherencia_ligera(chunk: pd.DataFrame, columnas: List[str]) -> pd.DataFrame:
    """
    Aplica reglas básicas de coherencia en un chunk.
    Versión optimizada para procesamiento por chunks.
    """
    
    # Regla 1: Coherencia Cloro (salida vs tanque)
    if all(col in chunk.columns for col in ['clexitlevel', 'cltanklevel']):
        mask_low_exit = chunk['clexitlevel'] < THRESHOLDS['clexitlevel']['min']
        if mask_low_exit.sum() > 0:
            chunk.loc[mask_low_exit, 'cltanklevel'] = np.minimum(
                chunk.loc[mask_low_exit, 'cltanklevel'],
                chunk.loc[mask_low_exit, 'clexitlevel'] * 0.8
            )
    
    # Regla 2: Coherencia Temperatura
    if all(col in chunk.columns for col in ['temperature', 'temperatureexitlevel']):
        mask_temp = chunk['temperature'].notna() & chunk['temperatureexitlevel'].isna()
        if mask_temp.sum() > 0:
            chunk.loc[mask_temp, 'temperatureexitlevel'] = (
                chunk.loc[mask_temp, 'temperature'] + 
                np.random.uniform(-2, 2, mask_temp.sum())
            )
    
    # Regla 3: Turbidez y Ensuciamiento
    if all(col in chunk.columns for col in ['waterturbidityexitlevel', 'soiling']):
        mask_high_turb = chunk['waterturbidityexitlevel'] > THRESHOLDS['waterturbidityexitlevel']['max']
        if mask_high_turb.sum() > 0:
            incremento = np.random.uniform(10, 30, mask_high_turb.sum())
            chunk.loc[mask_high_turb, 'soiling'] = np.clip(
                chunk.loc[mask_high_turb, 'soiling'] + incremento,
                THRESHOLDS['soiling']['alert_low'],
                THRESHOLDS['soiling']['alert_high']
            )
    
    # Regla 4: Coherencia Nivel de Agua y Flujo
    if all(col in chunk.columns for col in ['watertanklevel', 'flowexitlevel']):
        # Si el nivel de agua es bajo, el flujo tiende a ser menor
        mask_low_level = chunk['watertanklevel'] < THRESHOLDS['watertanklevel']['min']
        if mask_low_level.sum() > 0:
            chunk.loc[mask_low_level, 'flowexitlevel'] = np.minimum(
                chunk.loc[mask_low_level, 'flowexitlevel'],
                THRESHOLDS['flowexitlevel']['min'] + 
                np.random.uniform(0, 5, mask_low_level.sum())
            )
    
    return chunk


# ============================================================================
# FUNCIÓN: OBTENER RUTAS ABSOLUTAS
# ============================================================================

def obtener_rutas():
    """Obtiene las rutas absolutas de los archivos de entrada y salida"""
    # Obtener la ruta absoluta del script actual
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_entrada = os.path.join(ruta_base, '..', 'data', 'raw', 'watertankData.csv')
    ruta_salida = os.path.join(ruta_base, '..', 'data', 'raw', 'watertankDataTop.csv')
    
    return ruta_entrada, ruta_salida


# ============================================================================
# PIPELINE PRINCIPAL: PROCESAMIENTO POR CHUNKS
# ============================================================================

def ejecutar_pipeline_optimizado(ruta_entrada: str,
                                 ruta_salida: str,
                                 chunk_size: int = 100000,
                                 porcentaje_normal: float = 0.95,
                                 seed: int = 42):
    """
    Pipeline optimizado que procesa el archivo por chunks.
    No carga todo el dataset en memoria.
    
    Parámetros:
    -----------
    ruta_entrada : str
        Ruta del archivo CSV de entrada
    ruta_salida : str
        Ruta del archivo CSV de salida
    chunk_size : int
        Tamaño de los chunks (filas por chunk)
    porcentaje_normal : float
        Porcentaje de valores dentro del rango normal (0.95 = 95%)
    seed : int
        Semilla para reproducibilidad
    """
    
    print("\n" + "="*80)
    print("🚀 PIPELINE OPTIMIZADO - PROCESAMIENTO POR CHUNKS")
    print("="*80)
    print(f"📂 Archivo entrada: {ruta_entrada}")
    print(f"💾 Archivo salida:  {ruta_salida}")
    print(f"📦 Tamaño de chunk: {chunk_size:,} filas")
    print(f"🎯 Objetivo: {porcentaje_normal*100}% valores normales")
    
    # Columnas a rellenar
    columnas_a_rellenar = [
        'clexitlevel', 'clinjector', 'cltanklevel', 'phexitlevel', 
        'waterturbidityexitlevel', 'orpexitlevel', 'soiling', 'rssi', 
        'temperature', 'nasensor', 'calibration', 
        'flowexitlevel', 'pressurelevel', 'clslope', 
        'clgross', 'pressureexitlevel', 
        'temperatureexitlevel', 'turbidityentrylevel',
        'watertanklevel'  # Nueva columna añadida
    ]
    
    # Variables de control
    chunk_count = 0
    total_rows = 0
    total_nulls_filled = 0
    first_chunk = True
    columnas_existentes = None
    
    print("\n" + "="*80)
    print("🔧 PROCESANDO DATOS")
    print("="*80)
    
    try:
        # Procesar chunk por chunk
        for chunk in pd.read_csv(ruta_entrada, 
                                  sep=',', 
                                  encoding='utf-8', 
                                  chunksize=chunk_size,
                                  low_memory=False):
            
            chunk_count += 1
            rows_in_chunk = len(chunk)
            total_rows += rows_in_chunk
            
            # Identificar columnas existentes (solo en el primer chunk)
            if columnas_existentes is None:
                columnas_existentes = [col for col in columnas_a_rellenar 
                                      if col in chunk.columns]
                columnas_faltantes = [col for col in columnas_a_rellenar 
                                     if col not in chunk.columns]
                
                print(f"\n📋 Columnas a procesar: {len(columnas_existentes)}")
                if columnas_faltantes:
                    print(f"⚠️  Columnas no encontradas: {len(columnas_faltantes)}")
                    print(f"   {', '.join(columnas_faltantes)}")
                print()
            
            # Contar nulos antes del procesamiento
            nulls_antes = chunk[columnas_existentes].isnull().sum().sum()
            
            # PROCESAR CHUNK: Rellenar nulos
            chunk = rellenar_chunk(
                chunk=chunk,
                columnas=columnas_existentes,
                porcentaje_normal=porcentaje_normal,
                seed=seed + chunk_count
            )
            
            # Contar nulos después
            nulls_despues = chunk[columnas_existentes].isnull().sum().sum()
            nulls_rellenados = nulls_antes - nulls_despues
            total_nulls_filled += nulls_rellenados
            
            # Mostrar progreso
            print(f"✓ Chunk {chunk_count:>4}: {rows_in_chunk:>8,} filas | "
                  f"Nulos rellenados: {nulls_rellenados:>8,} | "
                  f"Total procesado: {total_rows:>10,}")
            
            # ESCRIBIR AL ARCHIVO DE SALIDA
            if first_chunk:
                # Primera vez: crear archivo con headers
                chunk.to_csv(ruta_salida, sep=',', encoding='utf-8', 
                           index=False, mode='w')
                first_chunk = False
            else:
                # Chunks siguientes: append sin headers
                chunk.to_csv(ruta_salida, sep=',', encoding='utf-8', 
                           index=False, mode='a', header=False)
            
            # LIBERAR MEMORIA
            del chunk
            gc.collect()
        
        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        print("\n" + "="*80)
        print("🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*80)
        
        print(f"\n📊 RESUMEN:")
        print(f"   ✅ Total de chunks procesados:  {chunk_count:>10,}")
        print(f"   ✅ Total de filas procesadas:   {total_rows:>10,}")
        print(f"   ✅ Total de nulos rellenados:   {total_nulls_filled:>10,}")
        print(f"   ✅ Columnas procesadas:         {len(columnas_existentes):>10}")
        
        # Verificar archivo de salida
        if os.path.exists(ruta_salida):
            file_size = os.path.getsize(ruta_salida) / (1024 * 1024)
            print(f"\n💾 ARCHIVO GENERADO:")
            print(f"   📁 Ruta:    {ruta_salida}")
            print(f"   📊 Tamaño:  {file_size:.2f} MB")
        
        print("\n" + "="*80)
        print("✨ Archivo listo para EDA y ML")
        print("="*80)
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo")
        print(f"   Ruta buscada: {ruta_entrada}")
        print(f"   Directorio actual: {os.getcwd()}")
        raise
        
    except Exception as e:
        print(f"\n❌ ERROR durante el procesamiento:")
        print(f"   {str(e)}")
        raise


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🌊 WATERTANK DATA PROCESSOR - OPTIMIZADO PARA BIG DATA")
    print("="*80)
    
    # Obtener rutas absolutas basadas en la ubicación del script
    ruta_entrada, ruta_salida = obtener_rutas()
    
    # Normalizar rutas para evitar problemas con '..'
    ruta_entrada = os.path.normpath(ruta_entrada)
    ruta_salida = os.path.normpath(ruta_salida)
    
    print(f"\n📂 RUTAS DETECTADAS:")
    print(f"   Script ubicado en: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"   Entrada:  {ruta_entrada}")
    print(f"   Salida:   {ruta_salida}")
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(ruta_entrada):
        print(f"\n❌ ERROR: No se encontró el archivo de entrada")
        print(f"   Ruta buscada: {ruta_entrada}")
        print(f"   Directorio actual: {os.getcwd()}")
        print(f"\n💡 SOLUCIÓN:")
        print(f"   1. Verifica que el archivo existe en: {ruta_entrada}")
        print(f"   2. O modifica las rutas manualmente en el código:")
        print(f"\n   ruta_entrada = r'C:\\ruta\\completa\\watertankData.csv'")
        print(f"   ruta_salida = r'C:\\ruta\\completa\\watertankDataTop.csv'")
        exit(1)
    
    print(f"\n✅ Archivo encontrado correctamente")
    
    # Verificar espacio en disco
    import shutil
    stats = shutil.disk_usage(os.path.dirname(ruta_salida))
    espacio_libre_gb = stats.free / (1024**3)
    print(f"💾 Espacio libre en disco: {espacio_libre_gb:.2f} GB")
    
    # Obtener tamaño del archivo de entrada
    size_input_mb = os.path.getsize(ruta_entrada) / (1024**2)
    print(f"📊 Tamaño archivo entrada: {size_input_mb:.2f} MB")
    
    # Confirmar ejecución
    print("\n" + "="*80)
    print("⚙️  CONFIGURACIÓN DEL PROCESAMIENTO:")
    print("="*80)
    print(f"   • Chunk size:         100,000 filas")
    print(f"   • Valores normales:   95%")
    print(f"   • Valores en alerta:  5%")
    print(f"   • Seed:               42 (reproducible)")
    print(f"   • Memoria estimada:   ~2-3 GB constante")
    
    print("\n" + "="*80)
    input("▶️  Presiona ENTER para iniciar el procesamiento...")
    
    # EJECUTAR PIPELINE OPTIMIZADO
    ejecutar_pipeline_optimizado(
        ruta_entrada=ruta_entrada,
        ruta_salida=ruta_salida,
        chunk_size=100000,
        porcentaje_normal=0.95,
        seed=42
    )
    
    print("\n" + "="*80)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*80)
    print(f"\n📁 Archivo generado: {ruta_salida}")
    print(f"🎯 El archivo está listo para:")
    print(f"   • Análisis exploratorio de datos (EDA)")
    print(f"   • Entrenamiento de modelos de ML")
    print(f"   • Uso en servidor MCP")
    print(f"   • Reporting y visualización")
    print("\n" + "="*80)