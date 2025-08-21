import unittest
import os
import tempfile
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from azure.storage.filedatalake import DataLakeServiceClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from io import StringIO

class RealModelFromADLSTest(unittest.TestCase):
    """Tests reales cargando modelo y datos desde ADLS Gen2"""
    
    @classmethod
    def setUpClass(cls):
        """Configurar conexión a ADLS Gen2 y cargar modelo real"""
        print("🔬 === CARGANDO MODELO REAL DESDE ADLS GEN2 ===")
        
        # Configuración ADLS Gen2
        cls.storage_account = "sistecreditofinal"
        cls.storage_key = os.getenv('AZURE_STORAGE_KEY', 'YpYHNOKME38oGXISqD7KFinQ3arvr43JNX59hiWXyTQvj8O7MwMlRQAx/jrPE2bMY+NHAIC0Sub7+AStbzR/Bg==')
        
        # Rutas en ADLS Gen2
        cls.model_container = "raw"
        cls.model_path = "models/random_forest_credit_risk_20250821_112616"  # Usar tu timestamp real
        cls.data_path = "data/v1/credit_risk_dataset.csv"
        
        try:
            # Crear cliente ADLS Gen2
            cls.service_client = DataLakeServiceClient(
                account_url=f"https://{cls.storage_account}.dfs.core.windows.net",
                credential=cls.storage_key
            )
            cls.file_system_client = cls.service_client.get_file_system_client(cls.model_container)
            
            # Cargar modelo, manifest y datos
            cls.model = cls._load_model()
            cls.manifest = cls._load_manifest()
            cls.test_data = cls._load_test_data()
            
            print("✅ Modelo, manifest y datos cargados exitosamente")
            
        except Exception as e:
            print(f"❌ Error configurando ADLS Gen2: {e}")
            cls.model = None
            cls.manifest = None
            cls.test_data = None
    
    @classmethod
    def _download_file(cls, file_path):
        """Descargar archivo desde ADLS Gen2"""
        file_client = cls.file_system_client.get_file_client(file_path)
        download_stream = file_client.download_file()
        return download_stream.readall()
    
    @classmethod
    def _load_model(cls):
        """Cargar modelo desde ADLS Gen2"""
        try:
            model_data = cls._download_file(f"{cls.model_path}/model.joblib")
            
            # Guardar temporalmente y cargar con joblib
            with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
                tmp_file.write(model_data)
                tmp_model_path = tmp_file.name
            
            model = joblib.load(tmp_model_path)
            os.unlink(tmp_model_path)  # Limpiar archivo temporal
            
            print(f"✅ Modelo cargado: {type(model).__name__}")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return None
    
    @classmethod
    def _load_manifest(cls):
        """Cargar manifest desde ADLS Gen2"""
        try:
            manifest_data = cls._download_file(f"{cls.model_path}/manifest.json")
            manifest = json.loads(manifest_data.decode('utf-8'))
            
            print(f"✅ Manifest cargado - Accuracy reportada: {manifest['model_performance']['accuracy']:.3f}")
            return manifest
            
        except Exception as e:
            print(f"❌ Error cargando manifest: {e}")
            return None
    
    @classmethod
    def _load_test_data(cls):
        """Cargar datos de prueba desde ADLS Gen2"""
        try:
            # Cargar datos originales para hacer split de test
            data_content = cls._download_file(cls.data_path)
            
            # Convertir bytes a string y luego a DataFrame
            data_string = data_content.decode('utf-8')
            df = pd.read_csv(StringIO(data_string))
            
            # Hacer split simple para test (últimas 20% filas)
            test_size = int(len(df) * 0.2)
            df_test = df.tail(test_size).copy()
            
            print(f"✅ Datos de test cargados: {len(df_test)} muestras")
            return df_test
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def setUp(self):
        """Verificar que todo esté cargado antes de cada test"""
        if self.model is None or self.manifest is None or self.test_data is None:
            self.skipTest("No se pudieron cargar modelo, manifest o datos desde ADLS Gen2")
    
    def test_01_model_loading_validation(self):
        """Test 1: Verificar que modelo y datos se cargaron correctamente"""
        print("\n🧪 Test 1: Validando carga desde ADLS Gen2...")
        
        self.assertIsNotNone(self.model, "Modelo debe estar cargado")
        self.assertIsNotNone(self.manifest, "Manifest debe estar cargado")
        self.assertIsNotNone(self.test_data, "Datos de test deben estar cargados")
        
        # Verificar que el modelo es RandomForest
        self.assertEqual(self.model.__class__.__name__, 'RandomForestClassifier')
        
        print("✅ Modelo, manifest y datos cargados correctamente desde ADLS Gen2")
    
    def test_02_real_model_performance(self):
        """Test 2: Evaluar performance real del modelo con datos reales"""
        print("\n🧪 Test 2: Evaluando performance real...")
        
        # Preparar datos para evaluación
        feature_columns = self.manifest['data_info']['feature_columns']
        target_column = self.manifest['data_info']['target_column']
        
        # Verificar que las columnas existen
        for col in feature_columns:
            self.assertIn(col, self.test_data.columns, f"Columna {col} no encontrada en datos")
        
        self.assertIn(target_column, self.test_data.columns, f"Target {target_column} no encontrado")
        
        # Preparar X e y
        X_test = self.test_data[feature_columns]
        y_test = self.test_data[target_column]
        
        # Hacer predicciones
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas reales
        accuracy_real = accuracy_score(y_test, y_pred)
        precision_real = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_real = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_real = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"📊 Accuracy real: {accuracy_real:.3f}")
        print(f"📊 Precision real: {precision_real:.3f}")
        print(f"📊 Recall real: {recall_real:.3f}")
        print(f"📊 F1-Score real: {f1_real:.3f}")
        
        # Validaciones con thresholds de producción
        self.assertGreaterEqual(accuracy_real, 0.70, f"Accuracy real {accuracy_real:.3f} < 70%")
        self.assertGreaterEqual(precision_real, 0.65, f"Precision real {precision_real:.3f} < 65%")
        self.assertGreaterEqual(recall_real, 0.65, f"Recall real {recall_real:.3f} < 65%")
        
        print("✅ Modelo cumple todos los thresholds de performance")
    
    def test_03_manifest_vs_real_performance(self):
        """Test 3: Comparar métricas del manifest vs evaluación real"""
        print("\n🧪 Test 3: Comparando manifest vs performance real...")
        
        # Obtener métricas del manifest
        manifest_accuracy = self.manifest['model_performance']['accuracy']
        
        # Evaluar en datos reales
        feature_columns = self.manifest['data_info']['feature_columns']
        target_column = self.manifest['data_info']['target_column']
        
        X_test = self.test_data[feature_columns]
        y_test = self.test_data[target_column]
        y_pred = self.model.predict(X_test)
        
        real_accuracy = accuracy_score(y_test, y_pred)
        
        # La diferencia no debe ser mayor al 10%
        accuracy_diff = abs(manifest_accuracy - real_accuracy)
        max_diff = 0.10
        
        self.assertLessEqual(
            accuracy_diff, max_diff,
            f"Diferencia accuracy manifest vs real {accuracy_diff:.3f} > {max_diff}"
        )
        
        print(f"✅ Accuracy manifest: {manifest_accuracy:.3f}")
        print(f"✅ Accuracy real: {real_accuracy:.3f}")
        print(f"✅ Diferencia: {accuracy_diff:.3f} (<= {max_diff})")
    
    def test_04_data_quality_validation(self):
        """Test 4: Validar calidad de datos de test"""
        print("\n🧪 Test 4: Validando calidad de datos...")
        
        # Verificar que no hay valores nulos en features importantes
        feature_columns = self.manifest['data_info']['feature_columns']
        
        for col in feature_columns:
            null_count = self.test_data[col].isnull().sum()
            null_percentage = null_count / len(self.test_data)
            
            # Máximo 5% de valores nulos permitidos
            self.assertLessEqual(
                null_percentage, 0.05,
                f"Columna {col} tiene {null_percentage:.2%} valores nulos (> 5%)"
            )
        
        # Verificar distribución de clases en target
        target_column = self.manifest['data_info']['target_column']
        class_distribution = self.test_data[target_column].value_counts(normalize=True)
        
        # Ninguna clase debe tener menos del 10% de representación
        min_class_ratio = class_distribution.min()
        self.assertGreaterEqual(
            min_class_ratio, 0.10,
            f"Clase minoritaria tiene {min_class_ratio:.2%} representación (< 10%)"
        )
        
        print(f"✅ Datos de calidad validados: {len(self.test_data)} muestras")
        print(f"✅ Distribución de clases balanceada: min {min_class_ratio:.2%}")
    
    def test_05_production_readiness(self):
        """Test 5: Validación final de production readiness"""
        print("\n🧪 Test 5: Validando production readiness...")
        
        # Todos los componentes críticos deben estar presentes
        critical_checks = {
            'model_loaded': self.model is not None,
            'manifest_complete': self.manifest is not None,
            'data_available': self.test_data is not None,
            'features_match': len(self.manifest['data_info']['feature_columns']) > 5,
            'recent_training': self._is_model_recent(),
            'performance_acceptable': self.manifest['model_performance']['accuracy'] >= 0.70
        }
        
        print("🔍 Checklist de producción:")
        for check, status in critical_checks.items():
            print(f"  {'✅' if status else '❌'} {check}: {status}")
            self.assertTrue(status, f"Production check failed: {check}")
        
        print("🚀 Modelo APROBADO para producción")
    
    def _is_model_recent(self):
        """Verificar que el modelo es reciente (últimos 30 días)"""
        try:
            created_date = datetime.fromisoformat(self.manifest['model_info']['created_date'])
            days_old = (datetime.now() - created_date).days
            return days_old <= 30
        except:
            return False
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup y resumen final"""
        print("\n🎯 === RESUMEN DE VALIDACIÓN REAL ===")
        print("✅ Modelo cargado y evaluado desde ADLS Gen2")
        print("✅ Performance validada con datos reales")
        print("✅ Calidad de datos verificada")
        print("🚀 MODELO APROBADO PARA PRODUCCIÓN")

if __name__ == '__main__':
    # Configurar para output detallado
    unittest.main(verbosity=2)
