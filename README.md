# credit-analysis-dashboard
# Sistema de Análisis Financiero para Decisiones de Crédito

Este repositorio contiene la especificación y componentes base de un sistema automatizado para analizar estados financieros de empresas, con el objetivo de asistir al departamento de créditos de una entidad bancaria en la toma rápida y fundamentada de decisiones sobre la concesión de préstamos a organizaciones.

---

## Objetivo del Proyecto

Desarrollar una solución integral que permita:

- Cargar y procesar reportes financieros históricos de empresas (últimos 7 años)
- Extraer automáticamente indicadores clave con ayuda de un LLM (ej. ChatGPT)
- Construir un perfil financiero estructurado de la empresa
- Visualizar los datos a través de dashboards en Power BI
- Generar reportes listos para analistas y responsables de decisiones crediticias

---

## Tipos de Documentos Admitidos

- Financial 10-K Reports

Formatos soportados: TXT, HTM

Data: https://www.kaggle.com/code/purvasingh/extracting-financial-10-k-reports-via-sec-edgar-db/input

---

## Flujo del Sistema

1. Carga de documentos  
   El usuario sube los archivos o los conecta desde una carpeta compartida

2. Extracción de texto  
   Uso de OCR / parsers para convertir documentos a texto

3. Procesamiento con LLM  
   Envío del texto a un LLM para extraer:
   - Ingresos
   - EBITDA
   - Utilidad neta
   - Activos / Pasivos
   - Ratios financieros

4. Estructuración  
   Almacenamiento de datos limpios en tablas normalizadas

5. Generación de informes  
   Informe financiero con métricas históricas y observaciones clave

6. Visualización  
   Dashboard interactivo en Power BI con:
   - Evolución de ingresos
   - Margen de utilidad
   - Liquidez, solvencia, eficiencia

---

## Tecnologías Utilizadas

| Tecnología | Uso |
|------------|-----|
| Python | Backend y procesamiento de datos |
| LLM API (ChatGPT) | Extracción inteligente de indicadores |
| Power BI | Visualización de dashboards |
| MySQL | Almacenamiento estructurado de datos |

---

## Posibilidades de Expansión

- Integración con bases financieras oficiales
- Modelos de scoring y predicción de riesgo
- Paneles comparativos multiempresa

---

## Estado del Proyecto

Fase de desarrollo inicial.  
En curso: construcción del pipeline de extracción + integración con LLM.  
Próximamente: EDA y visualización Power BI.

---

## Contacto

¿Tienes dudas o quieres colaborar en este proyecto?  
Email: natalia.a.stekolnikova@gmail.com  
Natalia Stekolnikova
