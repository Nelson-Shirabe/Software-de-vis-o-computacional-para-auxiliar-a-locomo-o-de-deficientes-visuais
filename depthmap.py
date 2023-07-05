import time
import cv2 as cv
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

class depthEstimator():
    
    def __init__(self):      
        # Inicialização do Modelo
        modelPath = "models/midasModel.tflite"
        self.interpreter = Interpreter(model_path = modelPath)
        self.interpreter.allocate_tensors()
        
        # Pegar informações do modelo
        self.modelInputDetails()
        self.modelOutputDetails()
        
    def modelInputDetails(self):
        # Pegar informaçoes das dimenções de entrada da imagem
        self.inputDetails = self.interpreter.get_input_details()
        inputShape = self.inputDetails[0]["shape"]
        self.inputHeight = inputShape[1]
        self.inputWidth = inputShape[2]
        self.channels = inputShape[3]
        
    def modelOutputDetails(self):
        # Pegar informações das dimenções de saída 
        self.outputDetails = self.interpreter.get_output_details()
        outputShape = self.outputDetails[0]["shape"]
        self.outputHeight = outputShape[1]
        self.outputWidth = outputShape[2]
        
        
        
        
    def estimatorDepthMap(self,img):
        # Preparar a imagem de entrada para passar pelo modelo
        inputTensor = self.prepareInput(img)
        
        # Realizar a inferencia da imagem
        outputModel = self.inference(inputTensor)
        
        # Processar a saída do modelo para se obter o Disparity
        disparity = self.processDisparity(outputModel)
          
        return disparity    
    
    def prepareInput(self, img):
        # Converter de BGR para RGB (No input do opencv a imagem vem em BGR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        # Pegar as dimensões da imagem de entrada:
        self.img_shape = img.shape
        
        # Mudar as dimenções da imagem para o tamanho de entrada do modelo
        img = cv.resize(img, (self.inputWidth, self.inputHeight), interpolation = cv.INTER_CUBIC).astype(np.float32)
        
        # O valor de cada pixel deve estar entre -1 e 1, normalizar através do z-score
        mean = np.zeros(3)
        std = np.zeros(3)
        
        for i in range(3):
            mean[i] = np.mean(img[i]/255.0)
            std[i] = np.std((img[i].flatten())/255.0)       
       
        img = ((img/255.0 - mean)/ std).astype(np.float32)
        img = img[np.newaxis,:,:,:]
        
        return img
    
    def inference(self, img):
        # Realiza a inferência, ou seja, determina a saída do modelo
        
        # Entrada
        self.interpreter.set_tensor(self.inputDetails[0]["index"], img)
        # Inferencia
        self.interpreter.invoke()
        # Saída
        output = self.interpreter.get_tensor(self.outputDetails[0]["index"])
        output = output.reshape(self.outputHeight, self.outputWidth)
        
        return output
    
    def processDisparity(self, outputModel):
        # Normalizar a saida para valores entre 0 e 255
        depth_min = outputModel.min()
        depth_max = outputModel.max()
        
        disparity = (255 * (outputModel - depth_min)/(depth_max - depth_min)).astype("uint8")
        
        # Voltar a dimensão da imagem para as dimenções da imagem de entrada
        disparity = cv.resize(disparity, (self.img_shape[1], self.img_shape[0]), interpolation = cv.INTER_CUBIC)
        
        return disparity
        
        
if __name__ == "__main__":
    
    # Inicializar o modelo
    depth_estimator = depthEstimator()

    # Inicializando WebCam
    camera = cv.VideoCapture(0, cv.CAP_DSHOW)
    cv.namedWindow("Depth Map", cv.WINDOW_NORMAL)
    
    while True:
        # Ler um frame da camera
        ret, img = camera.read()
        
        # Estimar Profundidade
        disparity = depth_estimator.estimatorDepthMap(img)
        
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        combinedImg = np.hstack((img, disparity))
        
        # Mostrar Imagem e FPS
        #cv.putText(combinedImg, (f"FPS: {depth_estimator.fps}"), (660, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (163,138,255), 2)
        cv.imshow("Depth Map", combinedImg)
        
        # Pressione a tecla "q" para parar
        if cv.waitKey(1) == ord("q"):
            break
    
    camera.release()
    cv.destroyAllWindows()