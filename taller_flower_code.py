import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris
from PIL import Image

# Cargar el modelo entrenado
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# Cargar el conjunto de datos Iris para obtener los nombres de las especies
iris = load_iris()

imagen_setosa = "setosa.jpg"
imagen_virginica = "virginica.jpg"
imagen_versicolor = "versicolor.jpg"

def inicio():
    st.title("Bienvenido a la Aplicación de Predicción de Iris")
    st.write("Esta aplicación te permite predecir la especie de una flor Iris basada en sus medidas.")
    
    st.write("A continuación, puedes ver imágenes de las tres especies de Iris que nuestro modelo puede predecir:")

    # Crear tres columnas para las imágenes
    col1, col2, col3 = st.columns(3)

    # Lista de nombres de archivos de imagen y sus correspondientes especies
    images = [
        ("setosa.jpg", "Iris setosa"),
        ("versicolor.jpg", "Iris versicolor"),
        ("virginica.jpg", "Iris virginica")
    ]

    # Cargar y mostrar las imágenes en las columnas
    for (filename, caption), col in zip(images, [col1, col2, col3]):
        try:
            image = Image.open(filename)
            #col.image(image, caption=caption, use_column_width=True) #deprecated 
            col.image(image, caption=caption, use_container_width=True)
        except FileNotFoundError:
            col.error(f"No se pudo encontrar la imagen {filename}")

    st.write("Estas son las tres especies de Iris que nuestro modelo ha sido entrenado para identificar.")
    st.write("Utiliza la opción 'Predicción' en el menú para probar el modelo con tus propias medidas.")

def acerca_de():
    st.title("Acerca de")
    st.write("Esta aplicación utiliza un modelo de Random Forest entrenado con el conjunto de datos Iris.")
    st.write("El conjunto de datos Iris contiene medidas de tres especies diferentes de Iris:")

    # Crear una tabla con las medidas típicas
    data = {
        'Especie': ['Iris setosa', 'Iris versicolor', 'Iris virginica'],
        'Longitud del sépalo (cm)': ['4.3 - 5.8', '4.9 - 7.0', '4.9 - 7.9'],
        'Ancho del sépalo (cm)': ['2.3 - 4.4', '2.0 - 3.4', '2.2 - 3.8'],
        'Longitud del pétalo (cm)': ['1.0 - 1.9', '3.0 - 5.1', '4.5 - 6.9'],
        'Ancho del pétalo (cm)': ['0.1 - 0.6', '1.0 - 1.8', '1.4 - 2.5']
    }

    st.table(data)

    st.write("El modelo ha sido entrenado para clasificar flores de Iris en una de estas tres especies basándose en estas cuatro medidas.")
    st.write("""
             Cuando uses la función de predicción, intenta usar valores dentro de estos rangos para obtener resultados más precisos.
             
             Autor:  Nalama1 
             
    **Repositorio:** [GitHub - grupo2](https://github.com/nalama1/taller_flores_streamlit)
    """)
 

def prediccion():
    st.title("Predicción de Especie de Iris")
    
    # Crear inputs para las características
    sepal_length = st.number_input("Longitud del sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Ancho del sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Longitud del pétalo (cm)", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Ancho del pétalo (cm)", min_value=0.0, max_value=10.0, value=0.2)
    
    # Botón de predicción
    if st.button("Predecir"):
        # Crear un array con las características
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Hacer la predicción
        prediction = modelo.predict(features)
        
        # Obtener el nombre de la especie predicha
        species = iris.target_names[prediction[0]]
        
        # Mostrar el resultado        
        if prediction[0] == 0:
            st.success(f"La especie predicha es: {species}")
            imagen = Image.open(imagen_setosa)
            st.image(imagen, caption='Iris setosa', use_container_width=True)
        elif prediction[0] == 1:
            st.success(f"La especie predicha es: {species}")
            imagen = Image.open(imagen_versicolor)
            st.image(imagen, caption='Iris versicolor', use_container_width=True)
        elif prediction[0] == 2:
            st.success(f"La especie predicha es: {species}")
            imagen = Image.open(imagen_virginica)
            st.image(imagen, caption='Iris virginica', use_container_width=True)
            
        
        
        

def main():
    st.sidebar.title("Menú")
    
    # Crear el menú desplegable
    menu = st.sidebar.selectbox(
        "Seleccione una opción",
        ("Inicio", "Acerca de", "Predicción", "Salir")
    )
    
    # Mostrar la página correspondiente según la selección del menú
    if menu == "Inicio":
        inicio()
    elif menu == "Acerca de":
        acerca_de()
    elif menu == "Predicción":
        prediccion()
    elif menu == "Salir":
        st.stop()

if __name__ == "__main__":
    main()
