"""
Modelos de datos para las tablas de HANA
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from sqlalchemy import Column, String, Integer, DateTime, Numeric, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class SalesOrderHeader(Base):
    """Tabla de cabecera de órdenes de venta"""
    __tablename__ = 'SALES_ORDER_HEADER'
    
    sales_order_number = Column(String(10), primary_key=True)
    sales_organization = Column(String(4))
    distribution_channel = Column(String(2))
    division = Column(String(2))
    sales_office = Column(String(4))
    sales_group = Column(String(3))
    sold_to_party = Column(String(10))
    ship_to_party = Column(String(10))
    customer_reference = Column(String(35))
    net_value = Column(String(20))
    currency = Column(String(3))
    created_by = Column(String(12))
    created_on = Column(DateTime, default=datetime.now)
    created_at = Column(String(20))
    changed_on = Column(DateTime)
    
    # Relación con items
    items = relationship("SalesOrderItem", back_populates="header")

class SalesOrderItem(Base):
    """Tabla de items/posiciones de órdenes de venta"""
    __tablename__ = 'SALES_ORDER_ITEMS'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sales_order_number = Column(String(10), ForeignKey('SALES_ORDER_HEADER.sales_order_number'))
    item_number = Column(String(6))
    material = Column(String(40))
    description = Column(Text)
    quantity = Column(String(20))
    unit = Column(String(3))
    net_value = Column(String(20))
    currency = Column(String(3))
    created_by = Column(String(12))
    created_on = Column(DateTime, default=datetime.now)
    created_at = Column(String(20))
    
    # Relación con header
    header = relationship("SalesOrderHeader", back_populates="items")

# Modelos Pydantic para la API
class SalesOrderItemRequest(BaseModel):
    """Modelo para item de orden de venta en requests"""
    item: str
    material: str
    description: str
    quantity: str
    unit: str
    netValue: str
    createdBy: str
    createdOn: str
    createdAt: str

class SalesOrderHeaderRequest(BaseModel):
    """Modelo para cabecera de orden de venta en requests"""
    salesOrganization: str
    distributionChannel: str
    division: str
    salesOffice: Optional[str] = ""
    salesGroup: Optional[str] = ""
    soldToParty: str
    shipToParty: str
    customerReference: Optional[str] = ""

class SalesOrderRequest(BaseModel):
    """Modelo completo para orden de venta"""
    header: SalesOrderHeaderRequest
    items: List[SalesOrderItemRequest]
    salesOrderNumber: Optional[str] = None

class SalesOrderResponse(BaseModel):
    """Respuesta al crear una orden de venta"""
    success: bool
    message: str
    salesOrderNumber: str
    data: Optional[dict] = None
