#setwd('C:/Users/wkivagn/Documents/zfs_uni_freiburg/ki_vorlesung')
#getwd()
#lade notwendige Paketen
library(dplyr)
library(stringr)

#lade die Daten

df <- read.csv2("laptops.csv", header = TRUE, sep= ",", dec=".")

#sehe die Struktur der Daten
str(df)

#sehe Informationen für die Daten
summary(df)

#sehe erste Datensätze
head(df)

#sehe den Anzahl der Spalten
ncol(df)

#sehe den Anzahl der Reihen
nrow(df)

#das GB von ram weglassen und in einen numerischen Wert ändern

df$Ram = str_replace(df$Ram,"GB","")
str(df)

df$Ram = as.numeric(df$Ram)
str(df)

#jetzt für Weight
#das 'KG' entfernen und in eine numerische Zahl ändern

df$Weight = str_replace(df$Weight,"kg","")
str(df)

df$Weight = as.numeric(df$Weight)
str(df)

#erstelle eine neue Variable mit der Name Angebot, die 100 Euro weniger als die aktuelle Preis ist

df$Angebot <- df$Price_euros - 100

str(df)

#unbennene die Variable Angebot zum maxAngebot

df= dplyr::rename(df, maxAngebot = Angebot)


#erstelle eine neue Dataframe mit der Name apple, welche alle Daten von df enthält, die als Company Apple haben

Apple=filter(df, Company == "Apple")

#erstelle eine neue Dataframe mit der Name laptop_unt_1000, welche alle Daten von df enthält, die weniger als 1000 Euro kosten

laptop_unt_1000=filter(df, Price_euros <= 1000)