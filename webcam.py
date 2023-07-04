import cv2 as cv
import time
import depthmap
import numpy as np

if __name__ == "__main__":
    
    # Inicializar o modelo
    depth_estimator = depthmap.depthEstimator()

    # Inicializando WebCam
    camera = cv.VideoCapture("http://192.168.0.11:4747/video")
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