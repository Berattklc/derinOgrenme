# derinOgrenme

from sklearn import datasets #....# veriseti yuklemek için ilgili paketi yaz   CVP: datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Soru: Sklearn içindeki Iris veri setini yükleyin ve bağımlı/bağımsız değişkenleri ayırın.
iris = datasets.load_iris()          # datasets.load_iris()
X, y = iris.data, iris.target

# Soru 2: Veri setini tran_test_split metodunu kullanrak eğitim ve test setlerine ayırın (%80 eğitim, %20 test).
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)  


# Soru 3: Eğitim ve test verilerini ölçekleyin. Neden ölçekelemek gerektiğini yazınız?
# modellerinin daha iyi performans göstermesi için önemlidir


sclaer = StandartScaler()
X_train_scaled = scaler.fit_transoform(X_train)
X_test_scaled = scaler.fit(X_test)

# CEVAP:
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Soru 4: Lojistik Regresyon modelini eğitin.
model = # LogisticRegression(max_iter=200) #           CVP: LogisticRegression(max_iter=200) #max itereasyon

# model fit ne işe yarıyor acıklama satırı olarak yaz.
#Modeli eğitim verileri üzerinde eğitir,Veri ile model arasındaki ilişkiyi öğrenir

model.fit(X_train_scaled, """y_train""" )  # noktalı yere hangi parametre gelmelidir neden ? CVP: y_train
# Model, X değerlerinden y değerlerini tahmin etmeyi öğrenmelidir. Bu nedenle hem özellik verileri (X) hem de bu özelliklere karşılık gelen hedef değerler (y) gereklidir


# Soru 5: Modelin doğruluk oranını hesaplayın.

y_pred = model_predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluk oranı: {accuracy:.2f}")

# CVP:

"""
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluk Oranı: {accuracy:.2f}")
"""

