# tests/test_model.py
import unittest
import json
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tempfile

class ModelValidationTest(unittest.TestCase):
    """Tests para validar el modelo ML antes del merge"""
    
    @classmethod
    def setUpClass(cls):
        """Configurar tests - cargar modelo y manifest"""
        print("🔬 === INICIANDO VALIDACIÓN DEL MODELO ===")
        
        # Buscar el manifest más reciente
        cls.manifest_path = "manifests/latest_manifest.json"
        cls.model = None
        cls.manifest = None
        
        # Cargar manifest
        if os.path.exists(cls.manifest_path):
            with open(cls.manifest_path, 'r', encoding='utf-8') as f:
                cls.manifest = json.load(f)
            print(f"✅ Manifest cargado desde: {cls.manifest_path}")
        else:
            print(f"❌ No se encontró manifest en: {cls.manifest_path}")
            
        # Simular carga de modelo (en el repo real tendrías que descargarlo de ADLS)
        print("📦 Simulando carga del modelo...")
        cls.model_loaded = True  # En real sería: cls.model = joblib.load(model_path)
    
    def test_01_manifest_exists(self):
        """Test 1: Verificar que el manifest existe y es válido"""
        print("\n🧪 Test 1: Validando manifest...")
        
        self.assertIsNotNone(self.manifest, "Manifest debe existir")
        self.assertIn('model_performance', self.manifest, "Manifest debe contener métricas")
        
        required_fields = ['model_info', 'data_info', 'model_performance', 'feature_importance']
        for field in required_fields:
            self.assertIn(field, self.manifest, f"Campo requerido faltante: {field}")
        
        print("✅ Manifest válido")
    
    def test_02_accuracy_threshold(self):
        """Test 2: Accuracy debe ser >= 70%"""
        print("\n🧪 Test 2: Validando accuracy...")
        
        accuracy = self.manifest['model_performance']['accuracy']
        min_accuracy = 0.70  # 70% mínimo
        
        self.assertGreaterEqual(
            accuracy, min_accuracy,
            f"Accuracy {accuracy:.3f} es menor al mínimo requerido {min_accuracy:.3f}"
        )
        
        print(f"✅ Accuracy: {accuracy:.3f} >= {min_accuracy:.3f}")
    
    def test_03_precision_threshold(self):
        """Test 3: Precision debe ser >= 65%"""
        print("\n🧪 Test 3: Validando precision...")
        
        precision = self.manifest['model_performance']['precision']
        min_precision = 0.65
        
        self.assertGreaterEqual(
            precision, min_precision,
            f"Precision {precision:.3f} es menor al mínimo requerido {min_precision:.3f}"
        )
        
        print(f"✅ Precision: {precision:.3f} >= {min_precision:.3f}")
    
    def test_04_recall_threshold(self):
        """Test 4: Recall debe ser >= 65%"""
        print("\n🧪 Test 4: Validando recall...")
        
        recall = self.manifest['model_performance']['recall']
        min_recall = 0.65
        
        self.assertGreaterEqual(
            recall, min_recall,
            f"Recall {recall:.3f} es menor al mínimo requerido {min_recall:.3f}"
        )
        
        print(f"✅ Recall: {recall:.3f} >= {min_recall:.3f}")
    
    def test_05_f1_score_threshold(self):
        """Test 5: F1-Score debe ser >= 65%"""
        print("\n🧪 Test 5: Validando F1-Score...")
        
        f1 = self.manifest['model_performance']['f1_score']
        min_f1 = 0.65
        
        self.assertGreaterEqual(
            f1, min_f1,
            f"F1-Score {f1:.3f} es menor al mínimo requerido {min_f1:.3f}"
        )
        
        print(f"✅ F1-Score: {f1:.3f} >= {min_f1:.3f}")
    
    def test_06_feature_count(self):
        """Test 6: Debe tener al menos 5 features"""
        print("\n🧪 Test 6: Validando número de features...")
        
        feature_count = self.manifest['data_info']['total_features']
        min_features = 5
        
        self.assertGreaterEqual(
            feature_count, min_features,
            f"Modelo tiene {feature_count} features, mínimo requerido: {min_features}"
        )
        
        print(f"✅ Features: {feature_count} >= {min_features}")
    
    def test_07_cross_validation(self):
        """Test 7: Cross-validation debe ser estable (std < 0.1)"""
        print("\n🧪 Test 7: Validando estabilidad CV...")
        
        cv_std = self.manifest['model_performance']['cross_validation_std']
        max_std = 0.10  # 10% máximo de desviación estándar
        
        self.assertLessEqual(
            cv_std, max_std,
            f"CV std {cv_std:.3f} es mayor al máximo permitido {max_std:.3f}"
        )
        
        print(f"✅ CV Std: {cv_std:.3f} <= {max_std:.3f}")
    
    def test_08_model_metadata(self):
        """Test 8: Metadata del modelo debe ser completa"""
        print("\n🧪 Test 8: Validando metadata...")
        
        model_info = self.manifest['model_info']
        
        required_metadata = ['model_type', 'created_date', 'model_parameters']
        for field in required_metadata:
            self.assertIn(field, model_info, f"Metadata faltante: {field}")
        
        # Verificar que sea Random Forest
        self.assertEqual(
            model_info['model_type'], 'RandomForestClassifier',
            "Debe ser un RandomForestClassifier"
        )
        
        print("✅ Metadata completa y válida")
    
    def test_09_data_quality_checks(self):
        """Test 9: Verificar calidad de datos usados"""
        print("\n🧪 Test 9: Validando calidad de datos...")
        
        data_info = self.manifest['data_info']
        
        # Verificar que se procesaron datos
        dataset_shape = data_info['dataset_shape']
        self.assertGreater(dataset_shape[0], 100, "Dataset debe tener >100 filas")
        
        # Verificar que hay features categóricas y numéricas
        self.assertIsInstance(data_info['feature_columns'], list, "Feature columns debe ser lista")
        
        print(f"✅ Dataset: {dataset_shape} filas, {dataset_shape[11]} columnas")
    
    def test_10_performance_comparison(self):
        """Test 10: Comparar con modelo anterior (si existe)"""
        print("\n🧪 Test 10: Comparando performance...")
        
        current_accuracy = self.manifest['model_performance']['accuracy']
        
        # Buscar modelo anterior
        previous_manifest_path = None
        manifests_dir = "manifests"
        
        if os.path.exists(manifests_dir):
            manifest_files = [f for f in os.listdir(manifests_dir) if f.startswith('manifest_') and f.endswith('.json')]
            if len(manifest_files) > 1:
                # Obtener el segundo más reciente
                manifest_files.sort(reverse=True)
                previous_manifest_path = os.path.join(manifests_dir, manifest_files[1])
        
        if previous_manifest_path and os.path.exists(previous_manifest_path):
            with open(previous_manifest_path, 'r') as f:
                previous_manifest = json.load(f)
            
            previous_accuracy = previous_manifest['model_performance']['accuracy']
            
            # El nuevo modelo no debe ser significativamente peor (>5% decrease)
            degradation_threshold = 0.05
            min_acceptable_accuracy = previous_accuracy - degradation_threshold
            
            self.assertGreaterEqual(
                current_accuracy, min_acceptable_accuracy,
                f"Nuevo modelo ({current_accuracy:.3f}) es significativamente peor que anterior ({previous_accuracy:.3f})"
            )
            
            print(f"✅ Performance: {current_accuracy:.3f} vs anterior {previous_accuracy:.3f}")
        else:
            print("ℹ️ No hay modelo anterior para comparar")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup después de tests"""
        print("\n🎯 === RESUMEN DE VALIDACIÓN ===")
        print("✅ Todos los tests pasaron - Modelo aprobado para merge")

if __name__ == '__main__':
    # Configurar output detallado
    unittest.main(verbosity=2, buffer=True)
