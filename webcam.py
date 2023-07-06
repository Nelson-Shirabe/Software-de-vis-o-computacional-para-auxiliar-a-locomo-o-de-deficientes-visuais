import cv2 as cv
import time
import depthmap
import numpy as np
import nearbyObjects

if __name__ == "__main__":
    
    # Inicializar o modelo
    depth_estimator = depthmap.depthEstimator()

    # Inicializando WebCam
    # Pegar video do celular usar como argumento: "http://192.168.0.11:4747/video"
    # Video da WebCam do Not usar como argumento: 0, cv.CAP_DSHOW
    camera = cv.VideoCapture(0, cv.CAP_DSHOW)
    cv.namedWindow("Depth Map", cv.WINDOW_NORMAL)
    cv.namedWindow("Imagem Original", cv.WINDOW_NORMAL)
    
    while True:
        # Ler um frame da camera
        ret, img = camera.read()
        
        if ret:
            # Tempo inicial para o FPS
            start_time = time.time()
            
            # Estimar Profundidade
            img = cv.flip(img, 1)
            disparity = depth_estimator.estimatorDepthMap(img)
           
            # Combinar imagem de entrada e depth map
            #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #combinedImg = np.hstack((img, disparity))
            
            # Objetos Próximos por Quagrante
            objects = nearbyObjects.nearby_objects()
            disparity_shape = disparity.shape
            
            # 1º Quadrante
            caution1 = disparity[:(disparity_shape[0]//2), (disparity_shape[1]//2): ] 
            cx = 0
            cy = 0
            caution1, I1 = objects.warning(caution1, cx, cy)
            
            # 2º Quadrante
            caution2 = disparity[:(disparity_shape[0]//2), :(disparity_shape[1]//2) ] 
            cx = disparity_shape[1]//2
            cy = 0
            caution2, I2 = objects.warning(caution2, cx, cy)
            
            # 3º Quadrante
            caution3 = disparity[(disparity_shape[0]//2):, :(disparity_shape[1]//2) ] 
            cx = disparity_shape[1]//2
            cy = disparity_shape[0]//2
            caution3, I3 = objects.warning(caution3, cx, cy)
            
            # 4º Quadrante
            caution4 = disparity[(disparity_shape[0]//2): , (disparity_shape[1]//2): ] 
            cx = 0
            cy = disparity_shape[0]//2
            caution4, I4 = objects.warning(caution4, cx, cy)
            
            
            
            
            # Calculo do FPS
            end_time = time.time()
            fps = round(1/(end_time - start_time), 2)
            
            # Mostrar Imagem e FPS
            # cv.putText(combinedImg, (f"FPS: {fps}"), (660, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (163,138,255), 2)
            # cv.imshow("Depth Map", combinedImg)
            
            cv.putText(img, (f"FPS: {fps}"), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (163,138,255), 2)
            cv.imshow("Imagem Original", img)
            
            
            cv.putText(disparity, (f"FPS: {fps}"), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (163,138,255), 2)
            cv.imshow("Depth Map", disparity)
            
            cv.imshow("I1", caution1)
            cv.setWindowTitle("I1", f"I1={I1:.2f}")
            
            cv.imshow("I2", caution2)
            cv.setWindowTitle("I2", f"I2={I2:.2f}")
            
            cv.imshow("I3", caution3)
            cv.setWindowTitle("I3", f"I3={I3:.2f}")
            
            cv.imshow("I4", caution4)
            cv.setWindowTitle("I4", f"I4={I4:.2f}")
            
            
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