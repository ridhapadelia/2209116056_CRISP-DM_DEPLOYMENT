import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
import seaborn as sns 
import datetime
import pickle 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


st.title("Students Adaptability Level Online Education Analysis And Prediction")

df = pd.read_csv('students_adaptablity_level_online_education2.csv')

st.write(df)

## DISTRIBUSI UMUR RESPONDEN
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribusi Umur Responden')
plt.xlabel('Umur')
plt.ylabel('Jumlah Responden')
plt.grid(True)
# Menampilkan plot di Streamlit
st.pyplot(plt)
st.write("Dapat terlihat pada visualisasi diatas bahwa rata-rata umur responden yang ada pada dataset berkisar antar 20-23 tahun yang artinya kebanyakan responden merupakan seorang mahasiswa")



## Perbandingan Tingkat Adaptabilitas berdasarkan Tingkat Pendidikan
# Menghitung jumlah adaptabilitas berdasarkan tingkat pendidikan
adaptability_by_education = df.groupby('Education Level')['Adaptivity Level'].value_counts().unstack()
# Membuat stacked bar plot
adaptability_by_education.plot(kind='bar', stacked=True, figsize=(8, 6))
# Menambahkan label dan judul
plt.xlabel('Tingkat Pendidikan')
plt.ylabel('Jumlah Responden')
plt.title('Perbandingan Tingkat Adaptabilitas berdasarkan Tingkat Pendidikan')
# Menampilkan plot di Streamlit
st.pyplot(plt)
st.write("Dari visualisasi diatas menunjukkan bahwa rata-rata tingkat education level tertinggi berasal dari tingkat pendidikan universitas, kemudian menempati posisi kedua ada pada tingkat pendidikan sekolah, dan yang terakhir tingkat pendidikan school memiliki tingkat adaptibilitas tinggi yang sangat sedikit") 





## Membuat histogram menggunakan kolom "Location"
plt.figure(figsize=(10, 6))
sns.histplot(df['Location'].dropna(), bins=5, kde=True)
plt.title('Distribusi Lokasi siswa dan mahasiswa')
plt.xlabel('Location')
plt.ylabel('Frekuensi')
# Menampilkan plot di Streamlit
st.pyplot(plt)
st.write("Dari visualisasi diatas menunjukkan bahwa rata-rata siswa dan mahasiswa berlokasi di daerah perkotaan")




## DIAGRAM PIE GENDER
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#1f77b4', '#ff7f0e'])
plt.title('Gender Responden')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(plt)
st.write("Dari diagram pie berikut dapat terlihat bahwa mayoritas responden merupakan seorang laki-laki dengan presentase sebesar 57,8%, sedangkan responden perempuan persentasenya sebesar 42,2%") 




#GNB
# Split data into features and target
X = df[['Adaptivity Level', 'Age', 'Education Level', 'Location']]
y = df['Adaptivity Level']

# Encode categorical feature 'Customer type'
le = LabelEncoder()
X['Adaptivity Level'] = le.fit_transform(X['Adaptivity Level'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, 'gnb.pkl')

def predict(adaptivity_level, age, education_level, location):
    input_data = pd.DataFrame([[adaptivity_level, age, education_level, location]], columns=['Adaptivity Level', 'Age', 'Education Level', 'Location'])
    prediction = clf.predict(input_data)
    return prediction[0]


st.subheader("Making Prediction")

adaptivity_level_option = [3, 2, 1,]  # 1 for 'Member', 0 for 'Normal'
adaptivity_level = st.selectbox('Adaptivity Level', adaptivity_level_option)

age = st.number_input('Age', min_value=0.0)
education_level = st.number_input('Education Level', min_value=1)
location = st.number_input('Location', min_value=0.0, max_value=10.0, step=0.1)
if st.button('Predict'): 
    prediction = predict(adaptivity_level, age, education_level, location)
    if prediction == 1:
        predicted_type = 'Low'
    elif prediction == 2:
        predicted_type = 'Moderate'
    elif prediction == 3:
        predicted_type = 'High'
    else:
        predicted_type = 'Unknown'
    st.write(f'Predicted Adaptivity Level: {predicted_type}')
