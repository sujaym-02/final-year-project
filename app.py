import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image

import shutil






app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html', methods=['GET'])
def demo():
      return render_template('index.html')

@app.route('/types.html', methods=['GET'])
def indexBt():
      return render_template('types.html')

@app.route('/disease.html', methods=['GET'])
def disease():
      return render_template('disease.html')






@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
              'http://127.0.0.1:5000/static/loss_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
             'Loss Graph',
             'Confusion Matrix']

            
    
        
    return render_template('graph.html',images=images,content=content)
    


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("disease/"+fileName, dst)
        image = cv2.imread("disease/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
         # # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                    [-1,-1,-1]])

        # # apply the sharpening kernel to the image
        sharpened =cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)

       
        
        
        
        model=load_model('coffeedisease_classifier.h5')
        path='static/images/'+fileName


        # Load the class names
        with open('disease_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        rem=""
        rem1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        str_label=""
        pre=""
        pre1=""
        print(predicted_class, confidence)
        if predicted_class == 'Broken':
            str_label = "Broken"
            pre="Precautions for Broken"
            pre1=["Identification: Circular black spots with fringed edges appear on the upper leaf surface, often surrounded by yellowing tissue.",
            "Pesticide: Use chlorothalonil or mancozeb based fungicides; apply weekly during wet conditions."
            "Prevention: Remove and dispose of infected leaves and ensure proper air circulation."]
            
            
             
           
        elif predicted_class == 'Cut':
            str_label = "Cut"
            pre="Precautions for downy mildw disease"
            pre1=["Identification: Purple to dark red irregular blotches on upper leaf surfaces with gray mold underneath.",
            "Pesticide: Apply metalaxyl or dimethomorph fungicides early in the infection cycle.",
            "Prevention: Avoid overhead watering; water early in the day to reduce leaf wetness duration."]
            

        elif predicted_class == 'Dry Cherry':
            str_label = "Dry Cherry"
            pre="Precautions for fresh leaf"
            pre1=["Identification: Healthy green leaves with no visible signs of disease or pests.",
            "Pesticide: No chemical treatment required; maintain with mild neem oil as a preventive measure.",
            "Prevention: Continue routine monitoring and proper rose care practices to sustain plant health."]
                    
            

        elif predicted_class == 'Full Black':
            str_label = "Full Black"
            pre="Precautions for powdery mildew disease"
            pre1=["Identification: White, powdery fungal growth appears on the surface of leaves, stems, and buds.",
            "Pesticide: Use sulfur-based fungicides or myclobutanil for effective treatment.",
            "Prevention: Prune dense growth to improve airflow; avoid high nitrogen fertilization."]


        elif predicted_class == 'Full Sour':
            str_label = "Full Sour"
            pre="Precautions for rose mosaic disease"
            pre1=["Identification: Yellow wavy lines or mottling patterns on leaves; caused by a virus, not fungi.",
            "Pesticide: No curative pesticide exists; remove and destroy infected plants to stop spread.",
            "Prevention: Use virus-free certified rose stock and control vector pests like aphids."]
            

        elif predicted_class == 'miner':
            str_label = "miner"
            pre="Precautions for rose rust disease"
            pre1=["Identification: Orange or rust-colored pustules appear on the underside of leaves.",
            "Pesticide: Apply tebuconazole or propiconazole fungicides at early stages.",
            "Prevention: Remove infected leaves and disinfect pruning tools after use."]
            
            

        elif predicted_class == 'nodisease':
            str_label = "nodisease"
            pre="Precautions for rose slug disease"
            pre1=["Identification: Skeletonized leaves caused by larvae of sawflies; leaves appear lacy or windowed.",
            "Pesticide: Use spinosad or insecticidal soap for effective slug control.",
            "Prevention: Handpick larvae and regularly inspect the undersides of leaves for eggs."]


        elif predicted_class == 'Partial Black':
            str_label = "Partial Black"
            pre="Precautions for powdery mildew disease"
            pre1=["Identification: White, powdery fungal growth appears on the surface of leaves, stems, and buds.",
            "Pesticide: Use sulfur-based fungicides or myclobutanil for effective treatment.",
            "Prevention: Prune dense growth to improve airflow; avoid high nitrogen fertilization."]


        elif predicted_class == 'Partial Sour':
            str_label = "Partial Sour"
            pre="Precautions for rose mosaic disease"
            pre1=["Identification: Yellow wavy lines or mottling patterns on leaves; caused by a virus, not fungi.",
            "Pesticide: No curative pesticide exists; remove and destroy infected plants to stop spread.",
            "Prevention: Use virus-free certified rose stock and control vector pests like aphids."]
            

        elif predicted_class == 'phoma':
            str_label = "phoma"
            pre="Precautions for rose rust disease"
            pre1=["Identification: Orange or rust-colored pustules appear on the underside of leaves.",
            "Pesticide: Apply tebuconazole or propiconazole fungicides at early stages.",
            "Prevention: Remove infected leaves and disinfect pruning tools after use."]
            
            

        elif predicted_class == 'rust':
            str_label = "rust"
            pre="Precautions for rose slug disease"
            pre1=["Identification: Skeletonized leaves caused by larvae of sawflies; leaves appear lacy or windowed.",
            "Pesticide: Use spinosad or insecticidal soap for effective slug control.",
            "Prevention: Handpick larvae and regularly inspect the undersides of leaves for eggs."]
                    
           
       
                    
           
       
           
            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
       
            

       


        return render_template('disease.html', status=str_label,accuracy=accuracy,Precautions=pre,Precautions1=pre1, 
                               ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,
                               ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",
                               ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",
                               ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",
                               ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
        
    return render_template('index.html')



############################Live########################################
@app.route('/live')
def live():
    dirPath = "static/images"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)

    vs = cv2.VideoCapture(0)
    while True:
        ret, image = vs.read()
        if not ret:
            break
        cv2.imshow('Leaf Disease', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('result.png', image)
            break
    vs.release()
    cv2.destroyAllWindows()
    
    dst = "static/images"

    shutil.copy('result.png', dst)
    image = cv2.imread("result.png")
        
    #color conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('static/gray.jpg', gray_image)
    #apply the Canny edge detection
    edges = cv2.Canny(image, 250, 254)
    cv2.imwrite('static/edges.jpg', edges)
    #apply thresholding to segment the image
    retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
    cv2.imwrite('static/threshold.jpg', threshold2)
        # # create the sharpening kernel
    kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                [-1,-1,-1]])

    # # apply the sharpening kernel to the image
    sharpened =cv2.filter2D(image, -1, kernel_sharpening)

    # save the sharpened image
    cv2.imwrite('static/sharpened.jpg', sharpened)

    
    
    
    
    model=load_model('coffeedisease_classifier.h5')
    path='result.png'

    # Load the class names
    with open('disease_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    
    # Function to preprocess the input image
    def preprocess_input_image(path):
        img = load_img(path, target_size=(150,150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        return img_array

    # Function to make predictions on a single image
    def predict_single_image(path):
        input_image = preprocess_input_image(path)
        prediction = model.predict(input_image)
        print(prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
            
        return predicted_class, confidence 

    predicted_class, confidence = predict_single_image(path)
    #predicted_class, confidence = predict_single_image(path, model, class_names)
    str_label=""
    pre=""
    pre1=""
    print(predicted_class, confidence)
    if predicted_class == 'Broken':
        str_label = "Broken"
        pre="Precautions for Broken"
        pre1=["Identification: Circular black spots with fringed edges appear on the upper leaf surface, often surrounded by yellowing tissue.",
        "Pesticide: Use chlorothalonil or mancozeb based fungicides; apply weekly during wet conditions."
        "Prevention: Remove and dispose of infected leaves and ensure proper air circulation."]
        
        
            
        
    elif predicted_class == 'Cut':
        str_label = "Cut"
        pre="Precautions for downy mildw disease"
        pre1=["Identification: Purple to dark red irregular blotches on upper leaf surfaces with gray mold underneath.",
        "Pesticide: Apply metalaxyl or dimethomorph fungicides early in the infection cycle.",
        "Prevention: Avoid overhead watering; water early in the day to reduce leaf wetness duration."]
        

    elif predicted_class == 'Dry Cherry':
        str_label = "Dry Cherry"
        pre="Precautions for fresh leaf"
        pre1=["Identification: Healthy green leaves with no visible signs of disease or pests.",
        "Pesticide: No chemical treatment required; maintain with mild neem oil as a preventive measure.",
        "Prevention: Continue routine monitoring and proper rose care practices to sustain plant health."]
                
        

    elif predicted_class == 'Full Black':
        str_label = "Full Black"
        pre="Precautions for powdery mildew disease"
        pre1=["Identification: White, powdery fungal growth appears on the surface of leaves, stems, and buds.",
        "Pesticide: Use sulfur-based fungicides or myclobutanil for effective treatment.",
        "Prevention: Prune dense growth to improve airflow; avoid high nitrogen fertilization."]


    elif predicted_class == 'Full Sour':
        str_label = "Full Sour"
        pre="Precautions for rose mosaic disease"
        pre1=["Identification: Yellow wavy lines or mottling patterns on leaves; caused by a virus, not fungi.",
        "Pesticide: No curative pesticide exists; remove and destroy infected plants to stop spread.",
        "Prevention: Use virus-free certified rose stock and control vector pests like aphids."]
        

    elif predicted_class == 'miner':
        str_label = "miner"
        pre="Precautions for rose rust disease"
        pre1=["Identification: Orange or rust-colored pustules appear on the underside of leaves.",
        "Pesticide: Apply tebuconazole or propiconazole fungicides at early stages.",
        "Prevention: Remove infected leaves and disinfect pruning tools after use."]
        
        

    elif predicted_class == 'nodisease':
        str_label = "nodisease"
        pre="Precautions for rose slug disease"
        pre1=["Identification: Skeletonized leaves caused by larvae of sawflies; leaves appear lacy or windowed.",
        "Pesticide: Use spinosad or insecticidal soap for effective slug control.",
        "Prevention: Handpick larvae and regularly inspect the undersides of leaves for eggs."]


    elif predicted_class == 'Partial Black':
        str_label = "Partial Black"
        pre="Precautions for powdery mildew disease"
        pre1=["Identification: White, powdery fungal growth appears on the surface of leaves, stems, and buds.",
        "Pesticide: Use sulfur-based fungicides or myclobutanil for effective treatment.",
        "Prevention: Prune dense growth to improve airflow; avoid high nitrogen fertilization."]


    elif predicted_class == 'Partial Sour':
        str_label = "Partial Sour"
        pre="Precautions for rose mosaic disease"
        pre1=["Identification: Yellow wavy lines or mottling patterns on leaves; caused by a virus, not fungi.",
        "Pesticide: No curative pesticide exists; remove and destroy infected plants to stop spread.",
        "Prevention: Use virus-free certified rose stock and control vector pests like aphids."]
        

    elif predicted_class == 'phoma':
        str_label = "phoma"
        pre="Precautions for rose rust disease"
        pre1=["Identification: Orange or rust-colored pustules appear on the underside of leaves.",
        "Pesticide: Apply tebuconazole or propiconazole fungicides at early stages.",
        "Prevention: Remove infected leaves and disinfect pruning tools after use."]
        
        

    elif predicted_class == 'rust':
        str_label = "rust"
        pre="Precautions for rose slug disease"
        pre1=["Identification: Skeletonized leaves caused by larvae of sawflies; leaves appear lacy or windowed.",
        "Pesticide: Use spinosad or insecticidal soap for effective slug control.",
        "Prevention: Handpick larvae and regularly inspect the undersides of leaves for eggs."]
                
        
    
                
        
    
        
        
    accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
    
        

    


    return render_template('disease.html', status1=str_label,accuracy1=accuracy,Precautions1=pre,Precautions11=pre1, 
                            ImageDisplay5="http://127.0.0.1:5000/static/images/result.png",
                            ImageDisplay6="http://127.0.0.1:5000/static/gray.jpg",
                            ImageDisplay7="http://127.0.0.1:5000/static/edges.jpg",
                            ImageDisplay8="http://127.0.0.1:5000/static/threshold.jpg",
                            ImageDisplay9="http://127.0.0.1:5000/static/sharpened.jpg")
    
   

#####################################################################################



@app.route('/type', methods=['GET', 'POST'])
def type():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("types/"+fileName, dst)
        image = cv2.imread("types/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
         # # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                    [-1,-1,-1]])

        # # apply the sharpening kernel to the image
        sharpened =cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)

       
        
        
        
        model=load_model('coffeetypes_classifier.h5')
        path='static/images/'+fileName


        # Load the class names
        with open('type_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
       
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        str_label=""
        price=""
        print(predicted_class, confidence)
        if predicted_class == 'ARABICA':
            str_label = "ARABICA"
            price="The Price for Arabica:13000Rs to 15000Rs(per 50 kg)"
            
           
             
           
        elif predicted_class == 'LIBERICA':
            str_label = "LIBERICA"
            price="The Price for Liberica:14000Rs to 15000Rs(per 50 kg)"
           

        elif predicted_class == 'ROBUSTA':
            str_label = "ROBUSTA"
            price="The Price for Robusta:12000Rs to 13000Rs(per 50 kg)"
           
           
       
                    
           
       
           
            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
       
            

       


        return render_template('types.html', status=predicted_class,accuracy=accuracy, price=price,
                               ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,
                               ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",
                               ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",
                               ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",
                               ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
        
    return render_template('index.html')

#######################################Live ##############################
@app.route('/live1')
def live1():
    dirPath = "static/images"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)

    vs = cv2.VideoCapture(0)
    while True:
        ret, image = vs.read()
        if not ret:
            break
        cv2.imshow('Leaf Disease', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('image.png', image)
            break
    vs.release()
    cv2.destroyAllWindows()
    
    dst = "static/images"

    shutil.copy('image.png', dst)
    image = cv2.imread("image.png")
    #color conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('static/gray.jpg', gray_image)
    #apply the Canny edge detection
    edges = cv2.Canny(image, 250, 254)
    cv2.imwrite('static/edges.jpg', edges)
    #apply thresholding to segment the image
    retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
    cv2.imwrite('static/threshold.jpg', threshold2)
        # # create the sharpening kernel
    kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                [-1,-1,-1]])

    # # apply the sharpening kernel to the image
    sharpened =cv2.filter2D(image, -1, kernel_sharpening)

    # save the sharpened image
    cv2.imwrite('static/sharpened.jpg', sharpened)

    
    
    
    
    model=load_model('coffeetypes_classifier.h5')
    path='image.png'


    # Load the class names
    with open('type_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    
    # Function to preprocess the input image
    def preprocess_input_image(path):
        img = load_img(path, target_size=(150,150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        return img_array

    # Function to make predictions on a single image
    def predict_single_image(path):
        input_image = preprocess_input_image(path)
        prediction = model.predict(input_image)
        print(prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
            
        return predicted_class, confidence 

    predicted_class, confidence = predict_single_image(path)
    #predicted_class, confidence = predict_single_image(path, model, class_names)
    str_label=""
    
    print(predicted_class, confidence)
    if predicted_class == 'ARABICA':
        str_label = "ARABICA"
        price="The Price for Arabica:13000Rs to 15000Rs(per 50 kg)"
        
        
            
        
    elif predicted_class == 'LIBERICA':
        str_label = "LIBERICA"
        price="The Price for Liberica:14000Rs to 15000Rs(per 50 kg)"
        

    elif predicted_class == 'ROBUSTA':
        str_label = "ROBUSTA"
        price="The Price for Robusta:12000Rs to 13000Rs(per 50 kg)"
    
    
    
                
        
    
        
        
    accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
    
        

    


    return render_template('types.html', status1=predicted_class,accuracy1=accuracy,price=price, 
                            ImageDisplay5="http://127.0.0.1:5000/static/images/image.png",
                            ImageDisplay6="http://127.0.0.1:5000/static/gray.jpg",
                            ImageDisplay7="http://127.0.0.1:5000/static/edges.jpg",
                            ImageDisplay8="http://127.0.0.1:5000/static/threshold.jpg",
                            ImageDisplay9="http://127.0.0.1:5000/static/sharpened.jpg")
    
    




@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
