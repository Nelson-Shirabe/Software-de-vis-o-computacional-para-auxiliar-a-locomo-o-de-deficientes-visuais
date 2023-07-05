import cv2 as cv
import time
import depthmap
import numpy as np
import nearbyObjects
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # Inicializar o modelo
    depth_estimator = depthmap.depthEstimator()

    # Inicializando WebCam
    # Pegar video do celular usar como argumento: "http://192.168.0.15:4747/video"
    # Video da WebCam do Not usar como argumento: 0, cv.CAP_DSHOW
    camera = cv.VideoCapture(0, cv.CAP_DSHOW)
    cv.namedWindow("Depth Map", cv.WINDOW_NORMAL)
    
    while True:
        # Ler um frame da camera
        ret, img = camera.read()
        
        if ret:
            # Tempo inicial para o FPS
            start_time = time.time()
            
            # Estimar Profundidade
            img = cv.flip(img, 1)
            disparity = depth_estimator.estimatorDepthMap(img)
           
            # Objetos Próximos
            objects = nearbyObjects.nearby_objects()
            objects, I = objects.warning(disparity) 
            print("\nI =",I)
            
            # Combinar imagem de entrada e depth map
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            combinedImg = np.hstack((img, disparity, objects))
            
            # Calculo do FPS
            end_time = time.time()
            fps = round(1/(end_time - start_time), 2)
            print("FPS:",fps)
            # Mostrar Imagem e FPS
            cv.putText(combinedImg, (f"FPS: {fps}"), (660, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (163,138,255), 2)
            cv.imshow("Depth Map", combinedImg)
            
            
        else:
            print("Failed to read the Frame!")
        
        
        # Pressione a tecla "q" para parar
        if cv.waitKey(1) == ord("q"):
            break
            
        # Salvar uma imagem do Depth Map segurar "s" até salvar a imagem
        if cv.waitKey(1) == ord("s"):
            cv.imwrite("Amostra.png", disparity)
        
        
    camera.release()
    cv.destroyAllWindows()