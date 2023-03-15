#lade notwendige Bibliotheken
import pandas as pd

#lade die Daten
df = pd.read_csv("laptops.csv", encoding='latin-1')

#sehe erste Datensätze
print(df.head())

#sehe Informationen für die Daten
df.info()


#das GB von ram weglassen und in einen numerischen Wert ändern
df['Ram'] = df['Ram'].replace('[GB]', '', regex=True)
df['Ram'] = pd.to_numeric(df['Ram'])
df.rename(columns={'Ram':'Ram (GB)'}, inplace=True)

#jetzt für Weight
#das 'KG' entfernen und in eine numerische Zahl ändern
df['Weight'] = df['Weight'].replace('[kg]', '', regex=True)
df['Weight'] = pd.to_numeric(df['Weight'])
df.rename(columns={'Weight':'Weight (Kg)'}, inplace=True)

print(df.head())

df.info()

#erstelle eine neue Variable mit der Name Angebot, die 100 Euro weniger als die aktuelle Preis ist
df['Angebot'] = df['Price_euros'] -100

print(df.head())

df.info()

#unbennene die Variable Angebot zum maxAngebot
df.rename(columns = {'Angebot':'maxAngebot'}, inplace = True)

print(df.head())

#erstelle eine neue Dataframe mit der Name apple, welche alle Daten von df enthält, die als Company Apple haben
is_apple =  df['Company']=="Apple"

apple = df[is_apple]

print(apple.head())

#erstelle eine neue Dataframe mit der Name laptop_unt_1000, welche alle Daten von df enthält, die weniger als 1000 Euro kosten

unt_1000 =  df['Price_euros']<= 1000

laptop_unt_1000 = df[unt_1000]

print(laptop_unt_1000.head())