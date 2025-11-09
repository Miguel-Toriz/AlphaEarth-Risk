import ee 
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_caching import Cache  # Importar Cache
from datetime import datetime, timedelta

# --- 1. Configuración Inicial del Servidor ---
app = Flask(__name__)

# Configurar un caché simple (en memoria) para el hackathon
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
CORS(app) 

# --- 2. Inicialización de Google Earth Engine ---
try:
    ee.Initialize() 
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error: Failed to initialize Earth Engine: {e}")

# --- 3. Funciones de "Cerebro" (IA) ---

# --- ANÁLISIS DE INCENDIO (FIRE) ---
def calculate_ndvi(image):
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

def analyze_fire_damage(image_before, image_after, aoi):
    ndvi_before = calculate_ndvi(image_before)
    ndvi_after = calculate_ndvi(image_after)
    dndvi = ndvi_before.subtract(ndvi_after).rename('dNDVI')
    
    burn_threshold = 0.6  # Umbral de quemado
    burned_area = dndvi.gt(burn_threshold).rename('burned_area').selfMask()

    pixel_area = ee.Image.pixelArea()
    burned_meters = burned_area.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).get('burned_area')

    total_meters = aoi.area().getInfo()
    if burned_meters is None:
        return 0.0, dndvi 

    damage_percent = (ee.Number(burned_meters).getInfo() / total_meters) * 100
    return round(damage_percent, 2), dndvi

# --- ANÁLISIS DE INUNDACIÓN (FLOOD) ---
def analyze_flood_damage(image_before, image_after, aoi):
    ndwi_before = image_before.normalizedDifference(['B3', 'B8']).rename('NDWI_Before')
    ndwi_after = image_after.normalizedDifference(['B3', 'B8']).rename('NDWI_After')

    water_threshold = 0.3
    not_water_before = ndwi_before.lt(water_threshold)
    is_water_after = ndwi_after.gt(water_threshold)
    
    flooded_area = not_water_before.And(is_water_after).rename('flooded_area').selfMask()

    pixel_area = ee.Image.pixelArea()
    flooded_meters = flooded_area.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).get('flooded_area')

    total_meters = aoi.area().getInfo()
    if flooded_meters is None:
        return 0.0, ndwi_after

    damage_percent = (ee.Number(flooded_meters).getInfo() / total_meters) * 100
    return round(damage_percent, 2), ndwi_after

# --- [NUEVA FUNCIÓN] ANÁLISIS DE TERREMOTO (Proxy de Deslizamiento) ---
def analyze_landslide_damage(image_before, image_after, aoi):
    # Reutiliza la lógica de NDVI: la vegetación (alto NDVI) es reemplazada por tierra/escombros (bajo NDVI)
    ndvi_before = calculate_ndvi(image_before)
    ndvi_after = calculate_ndvi(image_after)
    dndvi = ndvi_before.subtract(ndvi_after).rename('dNDVI')
    
    # Umbral más bajo que el de incendio, ya que la tierra desnuda tiene más NDVI que la ceniza
    slide_threshold = 0.4  
    slide_area = dndvi.gt(slide_threshold).rename('slide_area').selfMask()

    pixel_area = ee.Image.pixelArea()
    slide_meters = slide_area.multiply(pixel_area).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).get('slide_area')

    total_meters = aoi.area().getInfo()
    if slide_meters is None:
        return 0.0, dndvi 

    damage_percent = (ee.Number(slide_meters).getInfo() / total_meters) * 100
    return round(damage_percent, 2), dndvi
# --- [FIN NUEVA FUNCIÓN] ---


# --- FUNCIÓN PRINCIPAL DE GEE ---
def get_satellite_imagery(lat, lon, disaster_date_str, disaster_type):
    poi = ee.Geometry.Point(lon, lat)
    aoi = poi.buffer(1000) 

    date_disaster_start = ee.Date(disaster_date_str)
    date_before_start = date_disaster_start.advance(-3, 'month')
    date_after_end = date_disaster_start.advance(3, 'month')

    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                    .filterBounds(aoi) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))

    collection_before = collection.filterDate(date_before_start, date_disaster_start)
    collection_after = collection.filterDate(date_disaster_start, date_after_end)

    if collection_before.size().getInfo() == 0 or collection_after.size().getInfo() == 0:
        raise Exception("No cloud-free images (<15%) were found within 3 months before or after the specified date and location. Please try a different date range or area.")

    image_before_obj = collection_before.sort('system:time_start', False).first()
    image_after_obj = collection_after.sort('system:time_start', True).first()
    
    image_before = image_before_obj.clip(aoi)
    image_after = image_after_obj.clip(aoi)

    # --- [LÓGICA DE BIFURCACIÓN ACTUALIZADA] ---
    vis_params_analysis = {}

    if disaster_type == 'fire':
        print("Processing 'fire' event.")
        damage_percent, analysis_image = analyze_fire_damage(image_before, image_after, aoi)
        vis_params_analysis = {'min': -0.5, 'max': 1, 'palette': ['blue', 'white', 'green', 'yellow', 'red']}
    
    elif disaster_type == 'flood':
        print("Processing 'flood' event.")
        damage_percent, analysis_image = analyze_flood_damage(image_before, image_after, aoi)
        vis_params_analysis = {'min': -0.5, 'max': 1, 'palette': ['orange', 'white', 'blue']}
    
    elif disaster_type == 'hurricane':
        # El daño principal visible por Sentinel-2 es la inundación
        print("Processing 'hurricane' event as a flood analysis.")
        damage_percent, analysis_image = analyze_flood_damage(image_before, image_after, aoi)
        vis_params_analysis = {'min': -0.5, 'max': 1, 'palette': ['orange', 'white', 'blue']} # Misma paleta que inundación

    elif disaster_type == 'earthquake':
        # El daño principal visible son los deslizamientos de tierra (proxy)
        print("Processing 'earthquake' event as a landslide (dNDVI) analysis.")
        damage_percent, analysis_image = analyze_landslide_damage(image_before, image_after, aoi)
        vis_params_analysis = {'min': -0.2, 'max': 0.8, 'palette': ['blue', 'white', 'green', 'brown', 'red']} # Paleta de vegetación a tierra
    
    else:
        # Error actualizado para incluir los nuevos tipos
        raise Exception(f"Disaster type '{disaster_type}' is not supported. Use 'fire', 'flood', 'hurricane', or 'earthquake'.")
    # --- [FIN LÓGICA ACTUALIZADA] ---

    # --- Generar URLs ---
    vis_params_rgb = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
    url_before = image_before.getThumbUrl(vis_params_rgb)
    url_after = image_after.getThumbUrl(vis_params_rgb)
    url_damage_map = analysis_image.getThumbUrl(vis_params_analysis)

    return url_before, url_after, url_damage_map, damage_percent
# --- 4. Definición de la API (Endpoint) ---



@app.route('/analyze', methods=['POST'])
def analyze_damage_endpoint():
    data = request.json
    lat = float(data.get('lat'))
    lon = float(data.get('lon'))
    disaster_date = data.get('disaster_date')
    disaster_type = data.get('disaster_type') 

    if not all([lat, lon, disaster_date, disaster_type]):
        return jsonify({"error": "Faltan 'lat', 'lon', 'disaster_date' o 'disaster_type'"}), 400

    cache_key = f"analysis_{lat}_{lon}_{disaster_date}_{disaster_type}"
    cached_response = cache.get(cache_key)
    
    if cached_response:
        print("Response returned from CACHE!")
        return jsonify(cached_response)

    print("CALCULATED response (not from cache)!")
    
    try:
        url_before, url_after, url_damage_map, damage_percent = get_satellite_imagery(lat, lon, disaster_date, disaster_type)
        
        status = "Light Damage Detected"
        if damage_percent > 70:
            status = "AUTOMATICALLY APPROVED (Severe Damage)"
        elif damage_percent > 30:
            status = "FLAGGED FOR REVIEW (Considerable Damage)"
        elif damage_percent < 5:
            status = "No Significant Damage Detected"

        response_data = {
            "status": status,
            "damage_percent": damage_percent,
            "image_url_before": url_before,
            "image_url_after": url_after,
            "image_url_damage_map": url_damage_map,
            "location_processed": f"Lat: {lat}, Lon: {lon}",
            "date_processed": disaster_date,
            "type_processed": disaster_type
        }

        # --- [LÓGICA DE RIESGO ACTUALIZADA] ---
        # Añadimos contexto para los nuevos tipos de desastre
        
        if disaster_type == 'flood' or disaster_type == 'hurricane':
            # Combinamos inundación y huracán para el riesgo de inundación
            if (lat > 29 and lat < 31) and (lon > -91 and lon < -89): # Zona de Nueva Orleans
                response_data['risk_context'] = "ALERT! This property is in a High-Risk Flood/Storm Surge Zone designated by FEMA (e.g., Zone AE)."
            else:
                response_data['risk_context'] = "Property is outside of known high-risk flood zones."
        
        elif disaster_type == 'earthquake':
            # Añadimos un contexto de riesgo sísmico (ejemplo de California)
            if (lat > 33 and lat < 35) and (lon > -119 and lon < -117): # Zona de Los Ángeles
                response_data['risk_context'] = "ALERT! This property is in a High-Risk Seismic Zone (e.g., near San Andreas Fault)."
            else:
                response_data['risk_context'] = "Property is outside of known high-risk seismic zones."
        
        elif disaster_type == 'fire':
             response_data['risk_context'] = "Wildfire risk context for this area is under analysis."

        else:
            response_data['risk_context'] = "Risk context analysis not applicable for this type of disaster."
        # --- [FIN LÓGICA DE RIESGO] ---

        cache.set(cache_key, response_data, timeout=3600) 

        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in the GEE analysis: {e}")
        return jsonify({"error": f"Error in the analysis: {str(e)}"}), 500
        
# --- 5. Corre el Servidor ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
