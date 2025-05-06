from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.utils.predict import predict_image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trafik İşareti Sınıflandırıcı API",
    description="Trafik işareti resimlerini sınıflandıran bir API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Traffic Sign Classifier API", "status": "active"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        logger.warning(f"Desteklenmeyen dosya türü: {file.content_type}")
        raise HTTPException(status_code=400, detail="Sadece resim dosyaları yükleyiniz")
    
    try:

        contents = await file.read()
        logger.info(f"Resim alındı: {file.filename}, boyut: {len(contents)} bytes")
        
        result = predict_image(contents)
        

        if "error" in result:
            logger.error(f"Tahmin hatası: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        logger.error(f"İşlem sırasında hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
