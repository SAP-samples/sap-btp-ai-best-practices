"""
Conexión a SAP HANA Cloud
"""

import os
import logging
from typing import Optional
from hdbcli import dbapi
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class HANAConnection:
    """Clase para manejar la conexión a SAP HANA Cloud"""
    
    def __init__(self):
        self.address = os.getenv('HANA_ADDRESS', '3c6e8dda-3289-48c7-bcb2-6c68640eae3b.hna1.prod-eu10.hanacloud.ondemand.com')
        self.port = os.getenv('HANA_PORT', '443')
        self.user = os.getenv('HANA_USER', 'DBADMIN')
        self.password = os.getenv('HANA_PASSWORD', 'Cuerovelez2025')
        self.encrypt = os.getenv('HANA_ENCRYPT', 'True').lower() == 'true'
        
        self._connection = None
        self._engine = None
        self._session_factory = None
    
    def get_connection(self):
        """Obtiene una conexión directa usando hdbcli"""
        try:
            if self._connection is None or not self._connection.isconnected():
                logger.info(f"Conectando a HANA en {self.address}:{self.port}")
                self._connection = dbapi.connect(
                    address=self.address,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    encrypt=self.encrypt,
                    sslValidateCertificate=False
                )
                logger.info("Conexión a HANA establecida exitosamente")
            
            return self._connection
        except Exception as e:
            logger.error(f"Error conectando a HANA: {str(e)}")
            raise
    
    def get_engine(self):
        """Obtiene el engine de SQLAlchemy para HANA"""
        try:
            if self._engine is None:
                # Construir la URL de conexión para SQLAlchemy
                connection_string = f"hana://{self.user}:{self.password}@{self.address}:{self.port}"
                
                self._engine = create_engine(
                    connection_string,
                    echo=False,  # Cambiar a True para debug SQL
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
                
                logger.info("Engine de SQLAlchemy creado exitosamente")
            
            return self._engine
        except Exception as e:
            logger.error(f"Error creando engine de SQLAlchemy: {str(e)}")
            raise
    
    def get_session(self):
        """Obtiene una sesión de SQLAlchemy"""
        try:
            if self._session_factory is None:
                engine = self.get_engine()
                self._session_factory = sessionmaker(bind=engine)
            
            return self._session_factory()
        except Exception as e:
            logger.error(f"Error creando sesión: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[dict] = None):
        """Ejecuta una consulta SQL directamente"""
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Si es una consulta SELECT, retornar los resultados
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                # Para INSERT, UPDATE, DELETE, hacer commit
                connection.commit()
                return cursor.rowcount
                
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Error ejecutando consulta: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def test_connection(self) -> bool:
        """Prueba la conexión a HANA"""
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM DUMMY")
            result = cursor.fetchone()
            cursor.close()
            
            logger.info("Test de conexión exitoso")
            return result[0] == 1
        except Exception as e:
            logger.error(f"Test de conexión falló: {str(e)}")
            return False
    
    def close(self):
        """Cierra todas las conexiones"""
        try:
            if self._connection and self._connection.isconnected():
                self._connection.close()
                logger.info("Conexión HANA cerrada")
        except Exception as e:
            logger.error(f"Error cerrando conexión: {str(e)}")

# Instancia global de conexión
hana_connection = HANAConnection()
